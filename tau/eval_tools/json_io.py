from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any

from tau.eval_tools.fs_safety import open_directory_nofollow, rename_entry_noreplace


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_json_bytes(raw: bytes) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )


def load_json_with_sha256(path: Path) -> tuple[Any, str]:
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"refusing to read JSON through a symlink component: {current}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"JSON evidence must be a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"JSON evidence changed while it was being read: {path}")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    payload = parse_json_bytes(raw)
    return payload, hashlib.sha256(raw).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = open_directory_nofollow(Path(path))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written == 0:
            raise OSError("short write while staging durable evidence")
        offset += written


def _before_json_install(_path: Path) -> None:
    pass


def _after_json_install(_path: Path) -> None:
    pass


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor = open_directory_nofollow(path.parent)
    stage_name = None
    stage_descriptor = None
    try:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(path)
        for _ in range(8):
            candidate = f".{path.name}.stage-{secrets.token_hex(16)}"
            try:
                stage_descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            stage_name = candidate
            break
        else:
            raise FileExistsError(f"could not allocate a unique staging path for {path}")
        _write_all(stage_descriptor, payload)
        os.fsync(stage_descriptor)
        staged_metadata = os.fstat(stage_descriptor)
        _before_json_install(path.parent / stage_name)
        current_stage = os.stat(stage_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            stat.S_IFMT(current_stage.st_mode),
            current_stage.st_dev,
            current_stage.st_ino,
        ) != (
            stat.S_IFMT(staged_metadata.st_mode),
            staged_metadata.st_dev,
            staged_metadata.st_ino,
        ):
            raise RuntimeError(f"durable evidence staging changed before installation: {path}")
        rename_entry_noreplace(parent_descriptor, stage_name, path.name)
        os.fsync(parent_descriptor)
        _after_json_install(path)
        final_metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(final_metadata.st_mode):
            raise ValueError(f"durable evidence path must be a regular file: {path}")
        if (final_metadata.st_dev, final_metadata.st_ino, final_metadata.st_size) != (
            staged_metadata.st_dev,
            staged_metadata.st_ino,
            staged_metadata.st_size,
        ):
            raise RuntimeError(f"durable evidence changed during installation: {path}")
        final_descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
        try:
            opened_metadata = os.fstat(final_descriptor)
            if (opened_metadata.st_dev, opened_metadata.st_ino) != (
                staged_metadata.st_dev,
                staged_metadata.st_ino,
            ):
                raise RuntimeError(f"durable evidence changed while being reopened: {path}")
            chunks: list[bytes] = []
            while chunk := os.read(final_descriptor, 1024 * 1024):
                chunks.append(chunk)
            reloaded_metadata = os.fstat(final_descriptor)
            if (
                reloaded_metadata.st_dev,
                reloaded_metadata.st_ino,
                reloaded_metadata.st_size,
                reloaded_metadata.st_mtime_ns,
            ) != (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
                opened_metadata.st_size,
                opened_metadata.st_mtime_ns,
            ):
                raise RuntimeError(f"durable evidence changed while being verified: {path}")
        finally:
            os.close(final_descriptor)
        reloaded = b"".join(chunks)
        if hashlib.sha256(reloaded).digest() != hashlib.sha256(payload).digest() or reloaded != payload:
            raise RuntimeError(f"durable evidence bytes do not match staged bytes: {path}")
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        os.close(parent_descriptor)


def write_json_exclusive(path: Path, payload: Any) -> None:
    canonical = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected = parse_json_bytes(canonical)
    write_bytes_exclusive(path, canonical)
    reloaded, digest = load_json_with_sha256(path)
    if digest != hashlib.sha256(canonical).hexdigest() or reloaded != expected:
        raise RuntimeError(f"durable JSON evidence failed strict verification: {path}")
