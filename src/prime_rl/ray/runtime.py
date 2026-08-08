import asyncio
import copy
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import tomli_w

from prime_rl.configs.rl import RayExecutionConfig, RLConfig
from prime_rl.ray._utils import require_ray, role_context
from prime_rl.ray.inference import start_inference
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, set_proc_title


def _node_ip() -> str:
    from ray.util import get_node_ip_address

    return get_node_ip_address()


def _run_orchestrator(config, env: dict[str, str], log_path: Path) -> None:
    with role_context(env, log_path):
        set_proc_title("RayOrchestrator")
        from prime_rl.orchestrator.orchestrator import run_orchestrator

        asyncio.run(run_orchestrator(config))


def _wait_for_http(urls: list[str], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = set(urls)
    while pending and time.monotonic() < deadline:
        for url in list(pending):
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status < 500:
                        pending.remove(url)
            except OSError:
                pass
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError(f"Inference endpoints did not become ready: {sorted(pending)}")


def _inference_replica_count(config: RLConfig) -> int:
    assert config.inference is not None
    tp = config.inference.parallel.tp
    total_gpus = config.deployment.total_infer_nodes * config.deployment.gpus_per_node
    if total_gpus % tp != 0:
        raise ValueError(f"Inference GPU count {total_gpus} must be divisible by TP size {tp}.")
    return total_gpus // tp


def inference_bundles(config: RLConfig) -> list[dict[str, float]]:
    execution = config.execution
    assert isinstance(execution, RayExecutionConfig)
    assert config.inference is not None
    tp = config.inference.parallel.tp
    resource_name = f"accelerator_type:{execution.inference.accelerator_type}"
    return [
        {"CPU": execution.inference_num_cpus, "GPU": tp, resource_name: 0.001}
        for _ in range(_inference_replica_count(config))
    ]


def _write_subconfigs(config: RLConfig, config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "trainer.toml", "wb") as file:
        tomli_w.dump(to_toml_dict(config.trainer), file)
    with open(config_dir / "orchestrator.toml", "wb") as file:
        tomli_w.dump(to_toml_dict(config.orchestrator), file)


def run_ray(config: RLConfig) -> None:
    ray = require_ray()
    execution = config.execution
    assert isinstance(execution, RayExecutionConfig)
    assert config.inference is not None

    ray.init(
        address=execution.address,
        namespace=execution.namespace,
        log_to_driver=execution.log_to_driver,
    )

    config_dir = config.output_dir / "configs"
    log_dir = get_log_dir(config.output_dir)
    _write_subconfigs(config, config_dir)
    if config.dry_run:
        return

    attempt_id = uuid.uuid4().hex
    shared_env = {
        **os.environ,
        **DEFAULT_COMMON_ENV_VARS,
        **config.env_vars,
        "WANDB_SHARED_MODE": "1",
        "WANDB_SHARED_RUN_ID": os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex),
        "PRIME_RL_ATTEMPT_ID": attempt_id,
    }

    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    tp = config.inference.parallel.tp
    replica_count = _inference_replica_count(config)
    placement_groups = []
    refs: dict[str, Any] = {}
    try:
        api_urls: list[str] = []
        admin_urls: list[str] = []
        bundles = inference_bundles(config)
        for replica in range(replica_count):
            group = placement_group(
                [bundles[replica]],
                strategy="STRICT_PACK",
                name=f"prime-rl-{attempt_id}-inference-{replica}",
            )
            placement_groups.append(group)
            ray.get(group.ready(), timeout=execution.placement_timeout_seconds)
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            node_ip_ref = (
                ray.remote(_node_ip)
                .options(
                    num_cpus=0,
                    scheduling_strategy=strategy,
                )
                .remote()
            )
            node_ip = ray.get(node_ip_ref)
            replica_config = copy.deepcopy(config.inference)
            replica_config.parallel.dp = 1
            replica_config.data_parallel_size_local = None
            replica_config.api_server_count = 1
            replica_config.server.port += replica
            replica_config.backend_port += replica
            replica_path = config_dir / f"inference_{replica}.toml"

            with open(replica_path, "wb") as file:
                tomli_w.dump(
                    to_toml_dict(
                        replica_config,
                        exclude={"deployment", "slurm", "output_dir", "dry_run"},
                    ),
                    file,
                )
            refs[f"inference-{replica}"] = start_inference(
                ray,
                replica_config,
                config_path=replica_path,
                env={
                    **shared_env,
                    **DEFAULT_INFERENCE_ENV_VARS,
                    **replica_config.env_vars,
                },
                log_path=log_dir / f"inference_{replica}.log",
                accelerator_type=execution.inference.accelerator_type,
                num_cpus=execution.inference_num_cpus,
                num_gpus=tp,
                scheduling_strategy=strategy,
            )
            api_urls.append(f"http://{node_ip}:{replica_config.server.port}/v1")
            admin_urls.append(f"http://{node_ip}:{replica_config.backend_port}/v1")

        _wait_for_http(
            [f"{url.removesuffix('/v1')}/health" for url in api_urls],
            execution.placement_timeout_seconds,
        )
        orchestrator_config = copy.deepcopy(config.orchestrator)
        orchestrator_config.model.client.base_url = api_urls
        orchestrator_config.model.client.admin_base_url = admin_urls
        refs["orchestrator"] = (
            ray.remote(_run_orchestrator)
            .options(
                num_cpus=1,
                max_retries=0,
            )
            .remote(
                orchestrator_config,
                {
                    **shared_env,
                    **orchestrator_config.env_vars,
                    "WANDB_SHARED_LABEL": "orchestrator",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(sys.argv),
                },
                log_dir / "orchestrator.log",
            )
        )

        from prime_rl.ray.train import run_trainer

        trainer_ref = (
            ray.remote(run_trainer)
            .options(num_cpus=0, max_retries=0)
            .remote(
                config,
                log_dir,
                shared_env,
                sys.argv,
            )
        )
        refs["trainer"] = trainer_ref

        while refs:
            ready, _ = ray.wait(list(refs.values()), num_returns=1, timeout=1)
            if not ready:
                continue
            finished = ready[0]
            role = next(name for name, ref in refs.items() if ref == finished)
            ray.get(finished)
            refs.pop(role)
            if role == "trainer":
                break
    finally:
        outstanding = list(refs.values())
        for ref in outstanding:
            ray.cancel(ref, force=False, recursive=True)
        if outstanding:
            _, remaining = ray.wait(outstanding, num_returns=len(outstanding), timeout=30)
            for ref in remaining:
                ray.cancel(ref, force=True, recursive=True)
            if remaining:
                ray.wait(remaining, num_returns=len(remaining), timeout=30)
        for group in reversed(placement_groups):
            remove_placement_group(group)
        ray.shutdown()
