import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.ray._utils import role_context
from prime_rl.utils.process import set_proc_title


def _run_inference_subprocess(config_path: Path, env: dict[str, str], log_path: Path) -> None:
    with role_context(env, log_path):
        set_proc_title("RayPrimeVLLMSupervisor")
        process = subprocess.Popen(
            ["inference", "@", config_path.as_posix()],
            start_new_session=True,
        )

        def terminate_child(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, terminate_child)
        signal.signal(signal.SIGINT, terminate_child)
        try:
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Inference subprocess failed with exit code {return_code}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 30
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def start_inference(
    ray: Any,
    config: InferenceConfig,
    *,
    config_path: Path,
    env: dict[str, str],
    log_path: Path,
    accelerator_type: str,
    num_cpus: float,
    num_gpus: int,
    scheduling_strategy: Any,
) -> Any:
    supervisor = ray.remote(_run_inference_subprocess).options(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        resources={f"accelerator_type:{accelerator_type}": 0.001},
        scheduling_strategy=scheduling_strategy,
        max_retries=0,
    )
    return supervisor.remote(config_path, env, log_path)
