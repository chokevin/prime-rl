from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from typing import Sequence

from tau.eval_tools.fs_safety import open_directory_nofollow

_OUTPUT_PREFIX = Path("pretraining-data/prime-rl-math-7b-h200")
_MODE_OUTPUTS = {
    "smoke": "smoke",
    "freeze-draft": "manifest",
    "freeze-finalize": "manifest",
    "train": "train",
}
_EVAL_OUTPUTS = {
    "baseline": "eval-baseline",
    "post": "eval-post",
}


def expected_output_path(
    mode: str,
    *,
    eval_label: str | None = None,
    data_root: Path = Path("/data"),
) -> Path:
    eval_label = eval_label or None
    if mode == "eval":
        if eval_label not in _EVAL_OUTPUTS:
            raise ValueError("eval mode requires PRIME_RL_EVAL_LABEL=baseline or post")
        leaf = _EVAL_OUTPUTS[eval_label]
    else:
        if eval_label is not None:
            raise ValueError("PRIME_RL_EVAL_LABEL is valid only for eval mode")
        try:
            leaf = _MODE_OUTPUTS[mode]
        except KeyError:
            raise ValueError(f"unsupported PRIME_RL_RUN_MODE: {mode}") from None
    return Path(os.path.abspath(data_root)) / _OUTPUT_PREFIX / leaf


def prepare_output_directory(
    mode: str,
    output_dir: str | Path,
    *,
    eval_label: str | None = None,
    data_root: Path = Path("/data"),
) -> Path:
    expected = expected_output_path(mode, eval_label=eval_label, data_root=data_root)
    raw_output = os.fspath(output_dir)
    supplied = Path(raw_output)
    if not supplied.is_absolute() or raw_output != str(supplied) or ".." in supplied.parts:
        raise ValueError(f"TAU_OUTPUT_DIR must be an absolute canonical path: {raw_output}")
    if supplied != expected:
        raise ValueError(f"TAU_OUTPUT_DIR is {supplied}, expected exactly {expected} for mode {mode}")

    data_root = Path(os.path.abspath(data_root))
    relative = expected.relative_to(data_root)
    root_descriptor = open_directory_nofollow(data_root)
    current_descriptor = root_descriptor
    try:
        for component in relative.parts:
            try:
                os.mkdir(component, 0o755, dir_fd=current_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise NotADirectoryError(f"output path component is not a directory: {component}")
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
    finally:
        if current_descriptor != root_descriptor:
            os.close(current_descriptor)
        os.close(root_descriptor)
    return expected


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the exact mode-bound Tau output directory")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-label", default="")
    args = parser.parse_args(argv)
    prepare_output_directory(
        args.mode,
        args.output_dir,
        eval_label=args.eval_label,
    )


if __name__ == "__main__":
    main()
