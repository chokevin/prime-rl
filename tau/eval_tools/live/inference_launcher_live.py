from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NoReturn, Sequence

from tau.eval_tools.fs_safety import open_directory_nofollow

_LOG_NAME = "inference.log"


def open_inference_log(output_dir: str | Path) -> int:
    raw_output = os.fspath(output_dir)
    directory = Path(raw_output)
    if not directory.is_absolute() or raw_output != str(directory) or ".." in directory.parts:
        raise ValueError(f"output directory must be an absolute canonical path: {raw_output}")

    directory_descriptor = open_directory_nofollow(directory)
    try:
        return os.open(
            _LOG_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory_descriptor,
        )
    finally:
        os.close(directory_descriptor)


def launch_with_inference_log(output_dir: str | Path, command: Sequence[str]) -> NoReturn:
    if not command:
        raise ValueError("inference command must not be empty")
    descriptor = open_inference_log(output_dir)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor > 2:
            os.close(descriptor)
    os.execvp(command[0], command)


def main(argv: Sequence[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(description="Launch inference with an exclusive no-follow output log")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    launch_with_inference_log(args.output_dir, command)


if __name__ == "__main__":
    main()
