import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.shared import ClientConfig, RayTransportConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.ray._utils import require_ray, role_context
from prime_rl.ray.inference import start_prime_vllm_inference
from prime_rl.transport.ray import create_ray_transport_actor
from prime_rl.utils.logger import get_logger
from prime_rl.utils.process import set_proc_title

LOCAL_INFERENCE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _run_orchestrator_role(config: OrchestratorConfig, env: dict[str, str], log_path: Path) -> None:
    with role_context(env, log_path):
        set_proc_title("RayOrchestrator")
        from prime_rl.orchestrator.orchestrator import orchestrate

        asyncio.run(orchestrate(config))


def _run_trainer_rank(
    config: TrainerConfig,
    env: dict[str, str],
    log_path: Path,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    rank_env = {
        **env,
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "PYTHONUNBUFFERED": "1",
    }
    rank_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with role_context(rank_env, log_path):
        set_proc_title(f"RayTrainerRank{rank}")
        from prime_rl.trainer.rl.train import train
        from prime_rl.trainer.world import reset_world

        reset_world()
        train(config)


def _get_worker_node_ip() -> str:
    from ray.util import get_node_ip_address

    return get_node_ip_address()


def _is_local_inference_url(url: str) -> bool:
    return urlparse(url).hostname in LOCAL_INFERENCE_HOSTS


def _replace_inference_url_host(url: str, host: str, port: int) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    path = parsed.path or "/v1"
    netloc = f"{host}:{port}"
    return urlunparse((scheme, netloc, path, "", parsed.query, parsed.fragment))


def _rewrite_local_client_urls(client: ClientConfig, host: str, port: int) -> bool:
    if client.is_elastic:
        return False

    changed = False
    base_url = []
    for url in client.base_url:
        if _is_local_inference_url(url):
            base_url.append(_replace_inference_url_host(url, host, port))
            changed = True
        else:
            base_url.append(url)
    client.base_url = base_url

    if client.admin_base_url is not None:
        admin_base_url = []
        for url in client.admin_base_url:
            if _is_local_inference_url(url):
                admin_base_url.append(_replace_inference_url_host(url, host, port))
                changed = True
            else:
                admin_base_url.append(url)
        client.admin_base_url = admin_base_url

    return changed


def _orchestrator_with_ray_inference_endpoint(
    config: OrchestratorConfig, inference_config: InferenceConfig, host: str
) -> OrchestratorConfig:
    config = config.model_copy(deep=True)
    port = inference_config.server.port
    if _rewrite_local_client_urls(config.client, host, port):
        get_logger().info(f"Using Ray inference endpoint http://{host}:{port}/v1 for orchestrator rollouts")
    return config


def _orchestrator_with_ray_teacher_endpoint(
    config: OrchestratorConfig, teacher_inference_config: InferenceConfig, host: str
) -> OrchestratorConfig:
    config = config.model_copy(deep=True)
    if config.teacher_model is None:
        raise ValueError(
            "Ray-native teacher_inference requires orchestrator.teacher_model. "
            "Set deployment.num_teacher_gpus to auto-configure it or configure orchestrator.teacher_model manually."
        )
    port = teacher_inference_config.server.port
    if _rewrite_local_client_urls(config.teacher_model.client, host, port):
        get_logger().info(f"Using Ray teacher inference endpoint http://{host}:{port}/v1 for teacher logprobs")
    return config


def _init_ray(config: RLConfig):
    ray = require_ray()
    ray_config = config.experimental.ray
    if not ray.is_initialized():
        kwargs: dict[str, Any] = {"namespace": ray_config.namespace, "log_to_driver": ray_config.log_to_driver}
        if ray_config.address is not None:
            kwargs["address"] = ray_config.address
        runtime_env: dict[str, Any] = {}
        if ray_config.runtime_env.working_dir is not None:
            runtime_env["working_dir"] = ray_config.runtime_env.working_dir
        if ray_config.runtime_env.env_vars:
            runtime_env["env_vars"] = ray_config.runtime_env.env_vars
        if runtime_env:
            kwargs["runtime_env"] = runtime_env
        ray.init(**kwargs)
    return ray


def _get_placement_strategy(ray, config: RLConfig):
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray_config = config.experimental.ray
    bundles: list[dict[str, float]] = []

    if config.inference is not None:
        bundles.append({"CPU": ray_config.role_num_cpus, "GPU": config.deployment.num_infer_gpus})
    if config.teacher_inference is not None:
        bundles.append({"CPU": ray_config.role_num_cpus, "GPU": config.deployment.num_teacher_gpus or 0})

    bundles.append({"CPU": ray_config.role_num_cpus})
    if ray_config.trainer_backend == "tasks":
        for _ in range(config.deployment.num_train_gpus):
            bundles.append({"CPU": ray_config.trainer_worker_num_cpus, "GPU": 1})

    pg = placement_group(bundles, strategy=ray_config.placement_strategy)
    ray.get(pg.ready())

    def strategy(bundle_index: int):
        return PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=bundle_index)

    return pg, strategy


def _cancel_refs(ray, refs: dict[Any, object]) -> None:
    for ref in refs:
        ray.cancel(ref, force=True)


def _monitor_roles(
    ray,
    refs: dict[Any, tuple[str, Path]],
    critical_names: set[str],
    poll_interval_seconds: float,
) -> None:
    remaining = dict(refs)
    unfinished_critical = set(critical_names)

    while unfinished_critical:
        ready, _ = ray.wait(list(remaining), num_returns=1, timeout=poll_interval_seconds)
        if not ready:
            continue

        for ref in ready:
            name, log_path = remaining.pop(ref)
            try:
                ray.get(ref)
            except Exception as exc:
                _cancel_refs(ray, remaining)
                raise RuntimeError(f"Ray-native role {name} failed; see {log_path}") from exc

            if name in unfinished_critical:
                unfinished_critical.remove(name)
            elif unfinished_critical:
                _cancel_refs(ray, remaining)
                raise RuntimeError(
                    f"Long-running Ray-native role {name} exited before training finished; see {log_path}"
                )

    _cancel_refs(ray, remaining)


def run_ray_native(
    config: RLConfig,
    *,
    log_dir: Path,
    shared_env: dict[str, str],
    master_port: int,
    start_command: list[str],
) -> None:
    ray = _init_ray(config)
    pg = None
    transport_actor = None
    refs: dict[Any, tuple[str, Path]] = {}
    bundle_idx = 0

    try:
        pg, strategy = _get_placement_strategy(ray, config)
        ray_config = config.experimental.ray
        if not isinstance(config.trainer.rollout_transport, RayTransportConfig):
            raise TypeError("Ray-native trainer rollout transport must use RayTransportConfig")
        transport_actor = create_ray_transport_actor(config.trainer.rollout_transport)

        if config.inference is not None:
            if ray_config.inference_backend != "prime_vllm":
                raise ValueError(f"Unsupported Ray inference backend: {ray_config.inference_backend}")
            ip_task = ray.remote(_get_worker_node_ip).options(num_cpus=0, scheduling_strategy=strategy(bundle_idx))
            inference_host = ray.get(ip_task.remote())
            orchestrator_config = _orchestrator_with_ray_inference_endpoint(
                config.orchestrator, config.inference, inference_host
            )
            log_path = log_dir / "inference.log"
            ref = start_prime_vllm_inference(
                ray,
                config.inference,
                env=None,
                log_path=log_path,
                role_name="Inference",
                num_cpus=ray_config.role_num_cpus,
                num_gpus=config.deployment.num_infer_gpus,
                scheduling_strategy=strategy(bundle_idx),
            )
            refs[ref] = ("inference", log_path)
            bundle_idx += 1
        else:
            orchestrator_config = config.orchestrator

        if config.teacher_inference is not None:
            if ray_config.inference_backend != "prime_vllm":
                raise ValueError(f"Unsupported Ray inference backend: {ray_config.inference_backend}")
            ip_task = ray.remote(_get_worker_node_ip).options(num_cpus=0, scheduling_strategy=strategy(bundle_idx))
            teacher_inference_host = ray.get(ip_task.remote())
            orchestrator_config = _orchestrator_with_ray_teacher_endpoint(
                orchestrator_config, config.teacher_inference, teacher_inference_host
            )
            log_path = log_dir / "teacher_inference.log"
            ref = start_prime_vllm_inference(
                ray,
                config.teacher_inference,
                env=None,
                log_path=log_path,
                role_name="TeacherInference",
                num_cpus=ray_config.role_num_cpus,
                num_gpus=config.deployment.num_teacher_gpus or 0,
                scheduling_strategy=strategy(bundle_idx),
            )
            refs[ref] = ("teacher_inference", log_path)
            bundle_idx += 1

        orchestrator_env = {
            **shared_env,
            "WANDB_SHARED_LABEL": "orchestrator",
            "LOGURU_FORCE_COLORS": "1",
            "WANDB_PROGRAM": "uv run rl",
            "WANDB_ARGS": json.dumps(start_command),
        }
        orchestrator_task = ray.remote(_run_orchestrator_role).options(
            num_cpus=ray_config.role_num_cpus,
            scheduling_strategy=strategy(bundle_idx),
        )
        log_path = log_dir / "orchestrator.log"
        ref = orchestrator_task.remote(orchestrator_config, orchestrator_env, log_path)
        refs[ref] = ("orchestrator", log_path)
        bundle_idx += 1

        trainer_env = {
            **shared_env,
            "WANDB_SHARED_LABEL": "trainer",
            "LOGURU_FORCE_COLORS": "1",
            "WANDB_PROGRAM": "uv run rl",
            "WANDB_ARGS": json.dumps(start_command),
        }

        if ray_config.trainer_backend == "ray_train":
            # Wrap trainer.fit() in a Ray task so its ObjectRef can be polled alongside
            # inference and orchestrator refs. Without this, trainer.fit() blocks the
            # driver and an inference crash mid-run goes undetected — the trainer hangs
            # forever waiting for rollout batches that will never arrive.
            from prime_rl.ray.train import run_trainer_with_ray_train_remote

            trainer_task = ray.remote(run_trainer_with_ray_train_remote).options(
                num_cpus=ray_config.role_num_cpus,
            )
            log_path = log_dir / "trainer" / "ray_train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            ref = trainer_task.remote(config, log_dir.as_posix(), shared_env, start_command)
            refs[ref] = ("trainer", log_path)
            critical_names = {"orchestrator", "trainer"}
        else:
            trainer_start_bundle_idx = bundle_idx
            ip_task = ray.remote(_get_worker_node_ip).options(
                num_cpus=0, scheduling_strategy=strategy(trainer_start_bundle_idx)
            )
            master_addr = ray.get(ip_task.remote())

            trainer_task = ray.remote(_run_trainer_rank)
            for rank in range(config.deployment.num_train_gpus):
                task = trainer_task.options(
                    num_cpus=ray_config.trainer_worker_num_cpus,
                    num_gpus=1,
                    scheduling_strategy=strategy(bundle_idx),
                )
                log_path = log_dir / "trainer" / f"rank_{rank}.log"
                ref = task.remote(
                    config.trainer,
                    trainer_env,
                    log_path,
                    rank,
                    config.deployment.num_train_gpus,
                    master_addr,
                    master_port,
                )
                refs[ref] = (f"trainer_rank_{rank}", log_path)
                bundle_idx += 1

            critical_names = {"orchestrator"} | {
                f"trainer_rank_{rank}" for rank in range(config.deployment.num_train_gpus)
            }
        _monitor_roles(ray, refs, critical_names, ray_config.poll_interval_seconds)
    except KeyboardInterrupt:
        _cancel_refs(ray, refs)
        raise
    except Exception:
        _cancel_refs(ray, refs)
        raise
    finally:
        if pg is not None:
            try:
                from ray.util import remove_placement_group

                remove_placement_group(pg)
            except Exception as exc:
                get_logger().warning(f"Failed to remove Ray placement group: {exc}")
        if transport_actor is not None:
            try:
                ray.kill(transport_actor, no_restart=True)
            except Exception as exc:
                get_logger().warning(f"Failed to kill Ray transport actor: {exc}")
