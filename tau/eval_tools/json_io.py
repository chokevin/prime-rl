from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau.eval_tools.fs_safety import (
    open_directory_nofollow,
    quarantine_entry,
    rename_entries_noreplace,
    rename_entry_noreplace,
)


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


_NO_EXPECTED_OBJECT = object()


@dataclass(frozen=True)
class JsonEvidenceSnapshot:
    raw: bytes
    digest: str
    payload: Any
    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


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
        os.fsync(descriptor)
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


def _after_json_stage_validation(_path: Path) -> None:
    pass


def _after_json_install(_path: Path) -> None:
    pass


def _read_exact_file(
    descriptor: int,
    *,
    expected_metadata: os.stat_result,
    expected_bytes: bytes,
    expected_digest: bytes,
    expected_object: Any,
    path: Path,
) -> None:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    reloaded_metadata = os.fstat(descriptor)
    if (
        reloaded_metadata.st_dev,
        reloaded_metadata.st_ino,
        reloaded_metadata.st_size,
        reloaded_metadata.st_mtime_ns,
    ) != (
        expected_metadata.st_dev,
        expected_metadata.st_ino,
        expected_metadata.st_size,
        expected_metadata.st_mtime_ns,
    ):
        raise RuntimeError(f"durable evidence changed while being verified: {path}")
    reloaded = b"".join(chunks)
    if hashlib.sha256(reloaded).digest() != expected_digest or reloaded != expected_bytes:
        raise RuntimeError(f"durable evidence bytes do not match staged bytes: {path}")
    if expected_object is not _NO_EXPECTED_OBJECT and parse_json_bytes(reloaded) != expected_object:
        raise RuntimeError(f"durable JSON evidence object does not match staged object: {path}")


def _quarantine_failed_install(
    parent_descriptor: int,
    path: Path,
    error: Exception,
    *,
    quarantine_prefix: str | None = None,
) -> None:
    try:
        os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    try:
        quarantine_entry(
            parent_descriptor,
            path.name,
            quarantine_prefix=quarantine_prefix or f".{path.name}.quarantine",
        )
    except Exception as quarantine_error:
        error.add_note(f"failed to quarantine invalid durable evidence: {quarantine_error}")


def _snapshot_metadata(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(metadata.st_mode),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot_token(snapshot: JsonEvidenceSnapshot) -> tuple[int, int, int, int, int, int]:
    return (
        snapshot.mode,
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )


def _snapshot_identity(snapshot: JsonEvidenceSnapshot) -> tuple[int, int, int]:
    return (snapshot.mode, snapshot.device, snapshot.inode)


def _open_json_snapshot(directory_descriptor: int, name: str, path: Path) -> JsonEvidenceSnapshot:
    pathname_metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(pathname_metadata.st_mode):
        raise ValueError(f"JSON evidence path must be a regular file: {path}")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if _snapshot_metadata(pathname_metadata) != _snapshot_metadata(before):
            raise RuntimeError(f"JSON evidence path changed while being opened: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _snapshot_metadata(before) != _snapshot_metadata(after):
            raise RuntimeError(f"JSON evidence changed while being read: {path}")
        raw = b"".join(chunks)
        snapshot = JsonEvidenceSnapshot(
            raw=raw,
            digest=hashlib.sha256(raw).hexdigest(),
            payload=parse_json_bytes(raw),
            mode=stat.S_IFMT(after.st_mode),
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if _snapshot_metadata(current) != _snapshot_token(snapshot):
            raise RuntimeError(f"JSON evidence path changed after being read: {path}")
        return snapshot
    finally:
        os.close(descriptor)


def promote_json_noreplace(
    staging_path: Path,
    final_path: Path,
    *,
    expected_object: Any = _NO_EXPECTED_OBJECT,
    expected_raw: bytes | None = None,
    expected_digest: str | None = None,
    expected_identity: tuple[int, int] | None = None,
    after_stage_validation: Callable[[Path], None] | None = None,
    after_install: Callable[[Path], None] | None = None,
    on_verified: Callable[[], None] | None = None,
) -> JsonEvidenceSnapshot:
    staging_path = Path(os.path.abspath(staging_path))
    final_path = Path(os.path.abspath(final_path))
    source_descriptor = open_directory_nofollow(staging_path.parent)
    destination_descriptor = None
    installed = False
    try:
        destination_descriptor = open_directory_nofollow(final_path.parent)
        if os.fstat(source_descriptor).st_dev != os.fstat(destination_descriptor).st_dev:
            raise RuntimeError("JSON staging and destination must be on the same filesystem")
        staged = _open_json_snapshot(source_descriptor, staging_path.name, staging_path)
        if expected_object is not _NO_EXPECTED_OBJECT and staged.payload != expected_object:
            raise RuntimeError(f"staged JSON object does not match expected evidence: {staging_path}")
        if expected_raw is not None and staged.raw != expected_raw:
            raise RuntimeError(f"staged JSON bytes do not match expected evidence: {staging_path}")
        if expected_digest is not None and staged.digest != expected_digest:
            raise RuntimeError(f"staged JSON digest does not match expected evidence: {staging_path}")
        if expected_identity is not None and (staged.device, staged.inode) != expected_identity:
            raise RuntimeError(f"staged JSON inode does not match expected evidence: {staging_path}")
        current_stage = os.stat(
            staging_path.name,
            dir_fd=source_descriptor,
            follow_symlinks=False,
        )
        if _snapshot_metadata(current_stage) != _snapshot_token(staged):
            raise RuntimeError(f"JSON staging path changed after validation: {staging_path}")
        if after_stage_validation is not None:
            after_stage_validation(staging_path)
        rename_entries_noreplace(
            source_descriptor,
            staging_path.name,
            destination_descriptor,
            final_path.name,
        )
        installed = True
        os.fsync(source_descriptor)
        if source_descriptor != destination_descriptor:
            os.fsync(destination_descriptor)
        if after_install is not None:
            after_install(final_path)
        final = _open_json_snapshot(destination_descriptor, final_path.name, final_path)
        if (
            _snapshot_identity(final) != _snapshot_identity(staged)
            or final.raw != staged.raw
            or final.digest != staged.digest
            or final.payload != staged.payload
        ):
            raise RuntimeError(f"installed JSON evidence does not match validated staging: {final_path}")
        if expected_object is not _NO_EXPECTED_OBJECT and final.payload != expected_object:
            raise RuntimeError(f"installed JSON object does not match expected evidence: {final_path}")
    except Exception as error:
        if installed and destination_descriptor is not None:
            _quarantine_failed_install(
                destination_descriptor,
                final_path,
                error,
                quarantine_prefix=f"{final_path.name}.quarantine",
            )
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    if on_verified is not None:
        on_verified()
    return final


def write_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    expected_object: Any = _NO_EXPECTED_OBJECT,
) -> None:
    path = Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor = open_directory_nofollow(path.parent)
    stage_name = None
    stage_descriptor = None
    expected_digest = hashlib.sha256(payload).digest()
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
        _after_json_stage_validation(path.parent / stage_name)
        rename_entry_noreplace(parent_descriptor, stage_name, path.name)
        os.fsync(parent_descriptor)
        _after_json_install(path)
        try:
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
                if (
                    stat.S_IFMT(opened_metadata.st_mode),
                    opened_metadata.st_dev,
                    opened_metadata.st_ino,
                ) != (
                    stat.S_IFMT(staged_metadata.st_mode),
                    staged_metadata.st_dev,
                    staged_metadata.st_ino,
                ):
                    raise RuntimeError(f"durable evidence changed while being reopened: {path}")
                _read_exact_file(
                    final_descriptor,
                    expected_metadata=opened_metadata,
                    expected_bytes=payload,
                    expected_digest=expected_digest,
                    expected_object=expected_object,
                    path=path,
                )
                current_final = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                if (
                    stat.S_IFMT(current_final.st_mode),
                    current_final.st_dev,
                    current_final.st_ino,
                ) != (
                    stat.S_IFMT(opened_metadata.st_mode),
                    opened_metadata.st_dev,
                    opened_metadata.st_ino,
                ):
                    raise RuntimeError(f"durable evidence path changed after verification: {path}")
            finally:
                os.close(final_descriptor)
        except Exception as verification_error:
            _quarantine_failed_install(parent_descriptor, path, verification_error)
            raise
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        os.close(parent_descriptor)


def write_json_exclusive(path: Path, payload: Any) -> None:
    canonical = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected = parse_json_bytes(canonical)
    write_bytes_exclusive(path, canonical, expected_object=expected)
