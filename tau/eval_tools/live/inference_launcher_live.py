from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import NoReturn, Sequence

from tau.eval_tools.fs_safety import open_directory_nofollow
from tau.eval_tools.json_io import FileEvidenceSnapshot, promote_file_noreplace

_LOG_NAME = "inference.log"
_ATTEMPT_ID_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


def _validate_attempt_id(attempt_id: str) -> str:
    if not _ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("inference attempt ID must be a lowercase DNS label of at most 63 characters")
    return attempt_id


def attempt_log_name(attempt_id: str) -> str:
    return f"inference.attempt-{_validate_attempt_id(attempt_id)}.log"


def _canonical_output_directory(output_dir: str | Path) -> Path:
    raw_output = os.fspath(output_dir)
    directory = Path(raw_output)
    if not directory.is_absolute() or raw_output != str(directory) or ".." in directory.parts:
        raise ValueError(f"output directory must be an absolute canonical path: {raw_output}")
    return directory


def open_inference_log(output_dir: str | Path, attempt_id: str) -> int:
    directory = _canonical_output_directory(output_dir)
    directory_descriptor = open_directory_nofollow(directory)
    try:
        return os.open(
            attempt_log_name(attempt_id),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory_descriptor,
        )
    finally:
        os.close(directory_descriptor)


def _after_inference_log_validation(_path: Path) -> None:
    pass


def _after_inference_log_install(_path: Path) -> None:
    pass


def promote_inference_log(output_dir: str | Path, attempt_id: str) -> FileEvidenceSnapshot:
    directory = _canonical_output_directory(output_dir)
    return promote_file_noreplace(
        directory / attempt_log_name(attempt_id),
        directory / _LOG_NAME,
        after_stage_validation=_after_inference_log_validation,
        after_install=_after_inference_log_install,
    )


def launch_with_inference_log(output_dir: str | Path, attempt_id: str, command: Sequence[str]) -> NoReturn:
    if not command:
        raise ValueError("inference command must not be empty")
    descriptor = open_inference_log(output_dir, attempt_id)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor > 2:
            os.close(descriptor)
    os.execvp(command[0], command)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage exclusive attempt-scoped inference logs")
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--output-dir", required=True)
    launch_parser.add_argument("--attempt-id", required=True)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--output-dir", required=True)
    promote_parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    if args.action == "promote":
        promote_inference_log(args.output_dir, args.attempt_id)
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    launch_with_inference_log(args.output_dir, args.attempt_id, command)


if __name__ == "__main__":
    main()
