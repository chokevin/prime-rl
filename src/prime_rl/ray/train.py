import json
import os
from pathlib import Path
from typing import Any

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.ray._utils import require_ray, role_context
from prime_rl.utils.process import set_proc_title


def _run_ray_train_worker(train_loop_config: dict[str, Any]) -> None:
    from ray import train as ray_train

    config: TrainerConfig = train_loop_config["trainer_config"]
    shared_env: dict[str, str] = train_loop_config["shared_env"]
    log_dir = Path(train_loop_config["log_dir"])

    context = ray_train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    local_rank = context.get_local_rank()
    local_world_size = context.get_local_world_size()

    env = {
        **shared_env,
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": str(local_world_size),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for key in ("MASTER_ADDR", "MASTER_PORT"):
        if key in os.environ:
            env[key] = os.environ[key]

    log_path = log_dir / "trainer" / f"rank_{rank}.log"
    with role_context(env, log_path):
        set_proc_title(f"RayTrainTrainerRank{rank}")
        from prime_rl.trainer.rl.train import train
        from prime_rl.trainer.world import reset_world

        reset_world()
        train(config)


def run_trainer_with_ray_train(
    config: RLConfig,
    *,
    log_dir: Path,
    shared_env: dict[str, str],
    start_command: list[str],
) -> Any:
    require_ray()
    try:
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer
    except ImportError as exc:
        raise ImportError(
            "experimental.ray.trainer_backend = 'ray_train' requires Ray Train. "
            "Install Ray with the train extra, for example: uv pip install 'ray[default,train]>=2.40.0'."
        ) from exc

    ray_config = config.experimental.ray
    run_config_kwargs: dict[str, Any] = {}
    if ray_config.train_run_name is not None:
        run_config_kwargs["name"] = ray_config.train_run_name
    if ray_config.train_storage_path is not None:
        run_config_kwargs["storage_path"] = ray_config.train_storage_path

    trainer = TorchTrainer(
        train_loop_per_worker=_run_ray_train_worker,
        train_loop_config={
            "trainer_config": config.trainer,
            "shared_env": {
                **shared_env,
                "WANDB_SHARED_LABEL": "trainer",
                "LOGURU_FORCE_COLORS": "1",
                "WANDB_PROGRAM": "uv run rl",
                "WANDB_ARGS": json.dumps(start_command),
            },
            "log_dir": log_dir.as_posix(),
        },
        scaling_config=ScalingConfig(
            num_workers=config.deployment.num_train_gpus,
            use_gpu=True,
            resources_per_worker={"CPU": ray_config.trainer_worker_num_cpus},
            placement_strategy=ray_config.placement_strategy,
        ),
        run_config=RunConfig(**run_config_kwargs) if run_config_kwargs else None,
    )
    return trainer.fit()
