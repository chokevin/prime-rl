import json
import os
from pathlib import Path
from typing import Any

from prime_rl.configs.rl import RayExecutionConfig, RLConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.ray._utils import role_context
from prime_rl.utils.process import set_proc_title


def _run_ray_train_worker(train_loop_config: dict[str, Any]) -> None:
    from ray import train as ray_train

    config: TrainerConfig = train_loop_config["trainer_config"]
    shared_env: dict[str, str] = train_loop_config["shared_env"]
    log_dir = Path(train_loop_config["log_dir"])
    context = ray_train.get_context()
    rank = context.get_world_rank()
    env = {
        **shared_env,
        "RANK": str(rank),
        "WORLD_SIZE": str(context.get_world_size()),
        "LOCAL_RANK": str(context.get_local_rank()),
        "LOCAL_WORLD_SIZE": str(context.get_local_world_size()),
        # Ray Train owns the base process group and tears it down after this
        # worker function returns.
        "PRIME_RL_SKIP_DIST_DESTROY": "1",
        "PYTHONUNBUFFERED": "1",
    }
    for key in ("MASTER_ADDR", "MASTER_PORT"):
        if key in os.environ:
            env[key] = os.environ[key]

    with role_context(env, log_dir / "trainer" / f"rank_{rank}.log"):
        set_proc_title(f"RayTrainTrainerRank{rank}")
        from prime_rl.trainer.rl.train import train
        from prime_rl.trainer.world import reset_world

        reset_world()
        train(config)


def run_trainer(config: RLConfig, log_dir: Path, shared_env: dict[str, str], start_command: list[str]) -> Any:
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    execution = config.execution
    assert isinstance(execution, RayExecutionConfig)
    num_workers = config.deployment.num_train_nodes * config.deployment.gpus_per_node
    trainer = TorchTrainer(
        train_loop_per_worker=_run_ray_train_worker,
        train_loop_config={
            "trainer_config": config.trainer,
            "shared_env": {
                **shared_env,
                "WANDB_SHARED_LABEL": "trainer",
                "WANDB_PROGRAM": "uv run rl",
                "WANDB_ARGS": json.dumps(start_command),
            },
            "log_dir": log_dir.as_posix(),
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={
                "CPU": execution.trainer_num_cpus,
                f"accelerator_type:{execution.trainer.accelerator_type}": 0.001,
            },
            placement_strategy="PACK",
        ),
        run_config=RunConfig(name=f"prime-rl-{execution.trainer.accelerator_type.lower()}-trainer"),
    )
    return trainer.fit()
