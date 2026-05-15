import json
from pathlib import Path
from typing import Mapping

from prime_rl.configs.rl import RLConfig

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
ROLE_SCRIPT_NAME = "prime-rl-ray-role.sh"
RAY_DRIVER_NAME = "prime-rl-ray-driver.py"


def write_rayjob_manifest(config: RLConfig, config_dir: Path, manifest_path: Path) -> str:
    """Write a portable KubeRay RayJob manifest stream for a resolved multi-node RL config."""
    subconfigs = {
        TRAINER_TOML: (config_dir / TRAINER_TOML).read_text(),
        ORCHESTRATOR_TOML: (config_dir / ORCHESTRATOR_TOML).read_text(),
    }
    if config.inference is not None:
        subconfigs[INFERENCE_TOML] = (config_dir / INFERENCE_TOML).read_text()

    rendered = render_rayjob_yaml(config, subconfigs)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(rendered)
    return rendered


def render_rayjob_yaml(config: RLConfig, subconfigs: Mapping[str, str]) -> str:
    """Render a deterministic YAML stream.

    The stream uses JSON documents separated by YAML document markers. JSON is valid YAML and avoids adding a
    serializer dependency to the runtime package.
    """
    if config.rayjob is None:
        raise ValueError("RayJob rendering requires config.rayjob.")
    if config.inference is None:
        raise ValueError("RayJob rendering requires config.inference.")
    if INFERENCE_TOML not in subconfigs:
        raise ValueError(f"RayJob rendering requires {INFERENCE_TOML}.")

    docs = [
        _runtime_config_map(config, subconfigs),
        _rayjob(config),
    ]
    return "\n---\n".join(json.dumps(doc, indent=2) for doc in docs) + "\n"


def _runtime_config_map(config: RLConfig, subconfigs: Mapping[str, str]) -> dict:
    rayjob = config.rayjob
    assert rayjob is not None
    data = {
        TRAINER_TOML: subconfigs[TRAINER_TOML],
        ORCHESTRATOR_TOML: subconfigs[ORCHESTRATOR_TOML],
        INFERENCE_TOML: subconfigs[INFERENCE_TOML],
        ROLE_SCRIPT_NAME: _role_script(config),
        RAY_DRIVER_NAME: _ray_driver_script(config),
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": _config_map_name(config),
            "namespace": rayjob.namespace,
            "labels": _labels(config),
        },
        "data": data,
    }


def _rayjob(config: RLConfig) -> dict:
    rayjob = config.rayjob
    assert rayjob is not None
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": rayjob.job_name,
            "namespace": rayjob.namespace,
            "labels": _labels(config),
            "annotations": {
                "prime-rl.ai/runtime-backend": "rayjob",
                "prime-rl.ai/runtime-status": "rayjob-role-entrypoints-rendered",
                "prime-rl.ai/output-dir": str(config.output_dir),
            },
        },
        "spec": _drop_none(
            {
                "shutdownAfterJobFinishes": True,
                "ttlSecondsAfterFinished": rayjob.ttl_seconds_after_finished,
                "submissionMode": "HTTPMode",
                "entrypoint": f"python {rayjob.runtime_mount_path / RAY_DRIVER_NAME}",
                "rayClusterSpec": {
                    "rayVersion": rayjob.ray_version,
                    "headGroupSpec": {
                        "rayStartParams": {
                            "dashboard-host": "0.0.0.0",
                            "num-gpus": "0",
                        },
                        "template": _pod_template(config, "head"),
                    },
                    "workerGroupSpecs": [
                        {
                            "groupName": "prime-trainer",
                            "replicas": config.deployment.num_train_nodes,
                            "minReplicas": config.deployment.num_train_nodes,
                            "maxReplicas": config.deployment.num_train_nodes,
                            "rayStartParams": {
                                "num-gpus": str(config.deployment.gpus_per_node),
                                "resources": _ray_start_resources("prime_trainer"),
                            },
                            "template": _pod_template(config, "trainer"),
                        },
                        {
                            "groupName": "prime-inference",
                            "replicas": config.deployment.total_infer_nodes,
                            "minReplicas": config.deployment.total_infer_nodes,
                            "maxReplicas": config.deployment.total_infer_nodes,
                            "rayStartParams": {
                                "num-gpus": str(config.deployment.gpus_per_node),
                                "resources": _ray_start_resources("prime_inference"),
                            },
                            "template": _pod_template(config, "inference"),
                        },
                    ],
                },
            }
        ),
    }


def _pod_template(config: RLConfig, role: str) -> dict:
    rayjob = config.rayjob
    assert rayjob is not None
    gpu_count = config.deployment.gpus_per_node if role in {"trainer", "inference"} else 0
    container_name = "ray-head" if role == "head" else "ray-worker"
    resources = _resources(config, gpu_count)
    pod_spec = _drop_none(
        {
            "restartPolicy": "Never",
            "serviceAccountName": rayjob.service_account_name,
            "containers": [
                {
                    "name": container_name,
                    "image": rayjob.image,
                    "imagePullPolicy": rayjob.image_pull_policy,
                    "env": _env(config),
                    "resources": resources,
                    "volumeMounts": [
                        {
                            "name": "prime-rl-runtime",
                            "mountPath": str(rayjob.config_mount_path),
                            "readOnly": True,
                        },
                        {
                            "name": "prime-rl-runtime",
                            "mountPath": str(rayjob.runtime_mount_path),
                            "readOnly": True,
                        },
                        {
                            "name": "shared-output",
                            "mountPath": str(rayjob.shared_mount_path),
                        },
                        {
                            "name": "dshm",
                            "mountPath": "/dev/shm",
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "prime-rl-runtime",
                    "configMap": {"name": _config_map_name(config), "defaultMode": 0o555},
                },
                {
                    "name": "shared-output",
                    "persistentVolumeClaim": {"claimName": rayjob.shared_pvc_name},
                },
                {
                    "name": "dshm",
                    "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"},
                },
            ],
        }
    )
    if gpu_count > 0:
        pod_spec["tolerations"] = [{"key": rayjob.gpu_resource_name, "operator": "Exists", "effect": "NoSchedule"}]

    return {
        "metadata": {
            "labels": {
                **_labels(config),
                "prime-rl.ai/role": "orchestrator" if role == "head" else role,
            },
        },
        "spec": pod_spec,
    }


def _resources(config: RLConfig, gpu_count: int) -> dict:
    rayjob = config.rayjob
    assert rayjob is not None
    cpu = rayjob.head_cpu_request if gpu_count == 0 else rayjob.worker_cpu_request
    memory = rayjob.head_memory_request if gpu_count == 0 else rayjob.worker_memory_request
    values = {"cpu": cpu, "memory": memory}
    if gpu_count > 0:
        values[rayjob.gpu_resource_name] = gpu_count
    return {"requests": values, "limits": values}


def _ray_start_resources(resource_name: str) -> str:
    return "'" + json.dumps({resource_name: 1}, separators=(",", ":")) + "'"


def _env(config: RLConfig) -> list[dict[str, str]]:
    rayjob = config.rayjob
    assert rayjob is not None
    inference = config.inference
    assert inference is not None
    inference_deployment = inference.deployment
    env = {
        "PRIME_RL_CONFIG_DIR": str(rayjob.config_mount_path),
        "PRIME_RL_OUTPUT_DIR": str(config.output_dir),
        "PRIME_RL_RAYJOB_NAME": rayjob.job_name,
        "PRIME_RL_GPUS_PER_NODE": str(config.deployment.gpus_per_node),
        "PRIME_RL_NUM_TRAIN_NODES": str(config.deployment.num_train_nodes),
        "PRIME_RL_NUM_INFER_NODES": str(config.deployment.total_infer_nodes),
        "PRIME_RL_INFERENCE_TP": str(inference.parallel.tp),
        "PRIME_RL_INFERENCE_ROUTER_PORT": str(getattr(inference_deployment, "router_port", 8000)),
        "PRIME_RL_INFERENCE_BACKEND_PORT": str(getattr(inference_deployment, "backend_port", 8100)),
        "PRIME_RL_INFERENCE_DP_RPC_PORT": str(inference.data_parallel_rpc_port),
        "PRIME_RL_TRAINER_MASTER_PORT": "29500",
        "PRIME_RL_WEIGHT_BROADCAST_TYPE": config.weight_broadcast.type if config.weight_broadcast else "filesystem",
        "PRIME_RL_INFERENCE_STARTUP_GRACE_SECONDS": str(rayjob.inference_startup_grace_seconds),
        **rayjob.env,
    }
    return [{"name": key, "value": env[key]} for key in sorted(env)]


def _labels(config: RLConfig) -> dict[str, str]:
    rayjob = config.rayjob
    assert rayjob is not None
    return {
        "app.kubernetes.io/name": "prime-rl",
        "app.kubernetes.io/component": "rl",
        "app.kubernetes.io/instance": rayjob.job_name,
        "prime-rl.ai/backend": "rayjob",
    }


def _config_map_name(config: RLConfig) -> str:
    rayjob = config.rayjob
    assert rayjob is not None
    return rayjob.config_map_name or f"{rayjob.job_name}-runtime"


def _drop_none(value: dict) -> dict:
    return {k: v for k, v in value.items() if v is not None}


def _ray_driver_script(config: RLConfig) -> str:
    rayjob = config.rayjob
    assert rayjob is not None
    return f"""#!/usr/bin/env python3
import os
import signal
import socket
import subprocess
import time

import ray


ROLE_SCRIPT = "{rayjob.runtime_mount_path / ROLE_SCRIPT_NAME}"


def log(message):
    print(f"[prime-rl-rayjob] {{message}}", flush=True)


def local_ip():
    return socket.gethostbyname(socket.gethostname())


class RoleProcess:
    def __init__(self, role):
        self.role = role
        self.proc = None
        self.host = socket.gethostname()
        self.ip = local_ip()

    def info(self):
        return {{"role": self.role, "host": self.host, "ip": self.ip}}

    def start(self, extra_env):
        if self.proc is not None:
            raise RuntimeError(f"{{self.role}} already started")
        env = os.environ.copy()
        env.update(extra_env)
        env["PRIME_RL_ROLE"] = self.role
        env["PRIME_RL_POD_IP"] = self.ip
        env["PRIME_RL_POD_NAME"] = self.host
        log(f"starting {{self.role}} on {{self.host}} ({{self.ip}})")
        self.proc = subprocess.Popen(["bash", ROLE_SCRIPT, self.role, extra_env.get("PRIME_RL_JOB_INDEX", "0")], env=env)
        return {{"role": self.role, "host": self.host, "ip": self.ip, "pid": str(self.proc.pid)}}

    def wait(self):
        if self.proc is None:
            raise RuntimeError(f"{{self.role}} was not started")
        return self.proc.wait()

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def actor(role, resource, num_gpus):
    return ray.remote(num_cpus=1, num_gpus=num_gpus, resources={{resource: 0.01}})(RoleProcess).remote(role)


def common_env(trainer_ip, inference_ip):
    router_port = os.environ.get("PRIME_RL_INFERENCE_ROUTER_PORT", "8000")
    backend_port = os.environ.get("PRIME_RL_INFERENCE_BACKEND_PORT", "8100")
    return {{
        "PRIME_RL_TRAINER_MASTER_HOST": trainer_ip,
        "PRIME_RL_INFER_URLS": f"http://{{inference_ip}}:{{router_port}}/v1",
        "PRIME_RL_ADMIN_URLS": f"http://{{inference_ip}}:{{backend_port}}",
    }}


def run_orchestrator(env):
    orchestrator_env = os.environ.copy()
    orchestrator_env.update(env)
    orchestrator_env["PRIME_RL_ROLE"] = "orchestrator"
    orchestrator_env["PRIME_RL_POD_IP"] = local_ip()
    orchestrator_env["PRIME_RL_POD_NAME"] = socket.gethostname()
    log("starting orchestrator on Ray head")
    return subprocess.call(["bash", ROLE_SCRIPT, "orchestrator", "0"], env=orchestrator_env)


def main():
    ray.init(address="auto")
    gpus_per_node = int(os.environ.get("PRIME_RL_GPUS_PER_NODE", "8"))
    trainer = actor("trainer", "prime_trainer", gpus_per_node)
    inference = actor("inference", "prime_inference", gpus_per_node)

    trainer_info, inference_info = ray.get([trainer.info.remote(), inference.info.remote()])
    env = common_env(trainer_info["ip"], inference_info["ip"])
    log(f"trainer={{trainer_info['ip']}} inference={{inference_info['ip']}}")

    inference_env = dict(env, PRIME_RL_JOB_INDEX="0")
    trainer_env = dict(env, PRIME_RL_JOB_INDEX="0")
    ray.get(inference.start.remote(inference_env))
    time.sleep(int(os.environ.get("PRIME_RL_INFERENCE_STARTUP_GRACE_SECONDS", "30")))
    ray.get(trainer.start.remote(trainer_env))

    orchestrator_code = run_orchestrator(env)
    trainer_code = ray.get(trainer.wait.remote())
    ray.get(inference.stop.remote())
    if orchestrator_code != 0:
        log(f"orchestrator exited {{orchestrator_code}}")
        return orchestrator_code
    if trainer_code != 0:
        log(f"trainer exited {{trainer_code}}")
        return trainer_code
    log("Prime-RL RayJob completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
"""


def _role_script(config: RLConfig) -> str:
    inference = config.inference
    assert inference is not None
    ranks_filter = ",".join(map(str, config.trainer.log.ranks_filter))
    return f"""#!/usr/bin/env bash
set -euo pipefail

role="${{1:-${{PRIME_RL_ROLE:-}}}}"
rank="${{2:-${{PRIME_RL_JOB_INDEX:-0}}}}"

log() {{
  printf '[prime-rl-rayjob][%s] %s\\n' "${{role:-unknown}}" "$*" >&2
}}

run_uv() {{
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}}

wait_for_http() {{
  url="$1"
  timeout="$2"
  deadline=$((SECONDS + timeout))
  until python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

urlopen(sys.argv[1], timeout=5).read(1)
PY
  do
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "timed out waiting for $url"
      return 1
    fi
    sleep 5
  done
}}

setup_common_env() {{
  export PRIME_RL_CONFIG_DIR="${{PRIME_RL_CONFIG_DIR:-/prime-rl/configs}}"
  export PRIME_RL_OUTPUT_DIR="${{PRIME_RL_OUTPUT_DIR:-/data/outputs/prime-rl}}"
  export PRIME_RL_LOG_DIR="${{PRIME_RL_LOG_DIR:-$PRIME_RL_OUTPUT_DIR/logs}}"
  export PRIME_RL_GPUS_PER_NODE="${{PRIME_RL_GPUS_PER_NODE:-{config.deployment.gpus_per_node}}}"
  export PRIME_RL_NUM_TRAIN_NODES="${{PRIME_RL_NUM_TRAIN_NODES:-{config.deployment.num_train_nodes}}}"
  export PRIME_RL_NUM_INFER_NODES="${{PRIME_RL_NUM_INFER_NODES:-{config.deployment.total_infer_nodes}}}"
  export PRIME_RL_INFERENCE_TP="${{PRIME_RL_INFERENCE_TP:-{inference.parallel.tp}}}"
  export PRIME_RL_INFERENCE_ROUTER_PORT="${{PRIME_RL_INFERENCE_ROUTER_PORT:-8000}}"
  export PRIME_RL_INFERENCE_BACKEND_PORT="${{PRIME_RL_INFERENCE_BACKEND_PORT:-8100}}"
  export PRIME_RL_INFERENCE_DP_RPC_PORT="${{PRIME_RL_INFERENCE_DP_RPC_PORT:-{inference.data_parallel_rpc_port}}}"
  export PRIME_RL_TRAINER_MASTER_PORT="${{PRIME_RL_TRAINER_MASTER_PORT:-29500}}"
  export PRIME_RL_WEIGHT_BROADCAST_TYPE="${{PRIME_RL_WEIGHT_BROADCAST_TYPE:-{config.weight_broadcast.type if config.weight_broadcast else "filesystem"}}}"

  export CUDA_DEVICE_ORDER="${{CUDA_DEVICE_ORDER:-PCI_BUS_ID}}"
  export PYTHONUNBUFFERED="${{PYTHONUNBUFFERED:-1}}"
  export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
  export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
  export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-/tmp/triton-cache-${{USER:-prime}}-rayjob}}"

  if [ -n "${{PRIME_RL_PROJECT_DIR:-}}" ]; then
    cd "$PRIME_RL_PROJECT_DIR"
    [ -f .env ] && source .env
    [ -f .venv/bin/activate ] && source .venv/bin/activate
  fi

  mkdir -p "$PRIME_RL_LOG_DIR/trainer" "$PRIME_RL_LOG_DIR/inference"
  ln -sfn trainer/node_0.log "$PRIME_RL_LOG_DIR/trainer.log" 2>/dev/null || true
  ln -sfn inference/node_0.log "$PRIME_RL_LOG_DIR/inference.log" 2>/dev/null || true
}}

run_inference() {{
  export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}}"
  export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"
  local_ip="${{PRIME_RL_POD_IP:-$(hostname -I | awk '{{print $1}}')}}"
  log_file="$PRIME_RL_LOG_DIR/inference/node_${{rank}}.log"
  router_log="$PRIME_RL_LOG_DIR/inference/router_${{rank}}.log"
  admin_url="http://${{local_ip}}:${{PRIME_RL_INFERENCE_BACKEND_PORT}}"
  dp_per_node=$((PRIME_RL_GPUS_PER_NODE / PRIME_RL_INFERENCE_TP))

  log "starting vllm-router on $local_ip:$PRIME_RL_INFERENCE_ROUTER_PORT for $admin_url"
  vllm-router \\
    --policy consistent_hash \\
    --worker-urls "$admin_url" \\
    --host 0.0.0.0 \\
    --port "$PRIME_RL_INFERENCE_ROUTER_PORT" \\
    --intra-node-data-parallel-size "$dp_per_node" \\
    --worker-startup-timeout-secs 4200 \\
    >> "$router_log" 2>&1 &

  log "starting PRIME-RL inference backend on port $PRIME_RL_INFERENCE_BACKEND_PORT"
  run_uv inference \\
    @ "$PRIME_RL_CONFIG_DIR/inference.toml" \\
    --server.host 0.0.0.0 \\
    --server.port "$PRIME_RL_INFERENCE_BACKEND_PORT" \\
    --data-parallel-rpc-port "$PRIME_RL_INFERENCE_DP_RPC_PORT" \\
    2>&1 | tee -a "$log_file"
}}

run_orchestrator() {{
  for url in $PRIME_RL_INFER_URLS; do
    log "waiting for inference router $url"
    wait_for_http "$url" 4200
  done

  args=("@"
    "$PRIME_RL_CONFIG_DIR/orchestrator.toml"
    "--client.base-url" "$PRIME_RL_INFER_URLS"
    "--client.admin-base-url" "$PRIME_RL_ADMIN_URLS")
  if [ "$PRIME_RL_WEIGHT_BROADCAST_TYPE" = "nccl" ]; then
    args+=("--weight_broadcast.host" "$PRIME_RL_TRAINER_MASTER_HOST")
  fi

  log "starting PRIME-RL orchestrator"
  run_uv orchestrator "${{args[@]}}" 2>&1 | tee "$PRIME_RL_LOG_DIR/orchestrator.log"
}}

run_trainer() {{
  export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
  log_file="$PRIME_RL_LOG_DIR/trainer/node_${{rank}}.log"

  log "starting PRIME-RL trainer rank $rank/$PRIME_RL_NUM_TRAIN_NODES"
  run_uv torchrun \\
    --role=trainer \\
    --nnodes="$PRIME_RL_NUM_TRAIN_NODES" \\
    --nproc-per-node="$PRIME_RL_GPUS_PER_NODE" \\
    --node-rank="$rank" \\
    --rdzv-endpoint="$PRIME_RL_TRAINER_MASTER_HOST:$PRIME_RL_TRAINER_MASTER_PORT" \\
    --rdzv-id="$PRIME_RL_RAYJOB_NAME" \\
    --log-dir="$PRIME_RL_LOG_DIR/trainer/torchrun" \\
    --tee=3 \\
    --redirects=3 \\
    --local-ranks-filter="{ranks_filter}" \\
    -m prime_rl.trainer.rl.train \\
    @ "$PRIME_RL_CONFIG_DIR/trainer.toml" \\
    2>&1 | sed -u 's/^\\[[a-zA-Z]*[0-9]*\\]://' | tee -a "$log_file"
}}

setup_common_env
case "$role" in
  inference) run_inference ;;
  orchestrator) run_orchestrator ;;
  trainer) run_trainer ;;
  *) log "unsupported role=$role"; exit 64 ;;
esac
"""
