from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.ray._utils import role_context
from prime_rl.utils.process import set_proc_title


def _run_prime_vllm_inference_role(
    config: InferenceConfig,
    env: dict[str, str],
    log_path: Path,
    role_name: str,
) -> None:
    with role_context(env, log_path):
        set_proc_title(f"RayPrimeVLLM{role_name}")
        from prime_rl.entrypoints.inference import inference_local

        inference_local(config)


def start_prime_vllm_inference(
    ray: Any,
    config: InferenceConfig,
    *,
    env: dict[str, str] | None,
    log_path: Path,
    role_name: str,
    num_cpus: float,
    num_gpus: int,
    scheduling_strategy: Any,
) -> Any:
    task = ray.remote(_run_prime_vllm_inference_role).options(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        scheduling_strategy=scheduling_strategy,
    )
    return task.remote(config, env or {}, log_path, role_name)
