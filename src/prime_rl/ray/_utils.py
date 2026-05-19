import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator


@contextmanager
def role_context(env: dict[str, str], log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            try:
                yield
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


def require_ray():
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        import ray
    except ImportError as exc:
        raise ImportError(
            "Ray-native RL requires the optional 'ray' package. "
            "Install Ray before setting experimental.ray.enabled = true."
        ) from exc
    return ray
