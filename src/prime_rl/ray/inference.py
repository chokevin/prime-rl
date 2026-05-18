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
        # Force vLLM to use the multiprocessing distributed executor instead of its
        # Ray executor. The outer Ray task already holds num_infer_gpus GPUs in a
        # placement-group bundle, so vLLM's RayDistributedExecutor would try to
        # allocate a *second*, conflicting set of GPUs from the cluster and fail
        # with "Current node has no GPU available". The mp backend instead reuses
        # the CUDA_VISIBLE_DEVICES Ray set on this task and spawns local TP workers.
        config.vllm_extra["distributed_executor_backend"] = "mp"
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
