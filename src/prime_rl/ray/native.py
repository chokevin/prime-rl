import asyncio
import json
from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.shared import RayTransportConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.ray._utils import require_ray, role_context
from prime_rl.transport.ray import create_ray_transport_actor
from prime_rl.utils.process import set_proc_title


def _run_inference_role(config: InferenceConfig, env: dict[str, str], log_path: Path) -> None:
    with role_context(env, log_path):
        set_proc_title("RayInference")
        from prime_rl.entrypoints.inference import inference_local

        inference_local(config)


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
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    with role_context(rank_env, log_path):
        set_proc_title(f"RayTrainerRank{rank}")
        from prime_rl.trainer.rl.train import train
        from prime_rl.trainer.world import reset_world

        reset_world()
        train(config)


def _get_worker_node_ip() -> str:
    from ray.util import get_node_ip_address

    return get_node_ip_address()


def _init_ray(config: RLConfig):
    ray = require_ray()
    ray_config = config.experimental.ray
    if not ray.is_initialized():
        kwargs: dict[str, Any] = {"namespace": ray_config.namespace, "log_to_driver": ray_config.log_to_driver}
        if ray_config.address is not None:
            kwargs["address"] = ray_config.address
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
            inference_task = ray.remote(_run_inference_role).options(
                num_cpus=ray_config.role_num_cpus,
                num_gpus=config.deployment.num_infer_gpus,
                scheduling_strategy=strategy(bundle_idx),
            )
            log_path = log_dir / "inference.log"
            ref = inference_task.remote(config.inference, {}, log_path)
            refs[ref] = ("inference", log_path)
            bundle_idx += 1

        if config.teacher_inference is not None:
            teacher_task = ray.remote(_run_inference_role).options(
                num_cpus=ray_config.role_num_cpus,
                num_gpus=config.deployment.num_teacher_gpus or 0,
                scheduling_strategy=strategy(bundle_idx),
            )
            log_path = log_dir / "teacher_inference.log"
            ref = teacher_task.remote(config.teacher_inference, {}, log_path)
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
        ref = orchestrator_task.remote(config.orchestrator, orchestrator_env, log_path)
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
            from prime_rl.ray.train import run_trainer_with_ray_train

            run_trainer_with_ray_train(config, log_dir=log_dir, shared_env=shared_env, start_command=start_command)
            critical_names = {"orchestrator"}
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

            critical_names = {"orchestrator"} | {f"trainer_rank_{rank}" for rank in range(config.deployment.num_train_gpus)}
        _monitor_roles(ray, refs, critical_names, ray_config.poll_interval_seconds)
    except KeyboardInterrupt:
        _cancel_refs(ray, refs)
        raise
    except Exception:
        _cancel_refs(ray, refs)
        raise
    finally:
        if pg is not None:
            from ray.util import remove_placement_group

            remove_placement_group(pg)
        if transport_actor is not None:
            ray.kill(transport_actor, no_restart=True)
