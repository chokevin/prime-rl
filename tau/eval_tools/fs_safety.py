from __future__ import annotations

import ctypes
import errno
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LINUX_RENAME_NOREPLACE = 1
DARWIN_RENAME_EXCL = 4


@dataclass(frozen=True)
class DirectoryOwnershipToken:
    path: Path
    device: int
    inode: int
    owner_uid: int

    @classmethod
    def capture(cls, path: Path) -> DirectoryOwnershipToken:
        path = Path(os.path.abspath(path))
        parent_descriptor = open_directory_nofollow(path.parent)
        descriptor = None
        try:
            pathname_metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(pathname_metadata.st_mode):
                raise ValueError(f"owned directory root must be a non-symlink directory: {path}")
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            opened_metadata = os.fstat(descriptor)
            if not _same_entry(pathname_metadata, opened_metadata):
                raise RuntimeError(f"owned directory root changed while capturing its token: {path}")
            if opened_metadata.st_uid != os.getuid():
                raise PermissionError(f"owned directory root belongs to another user: {path}")
            return cls(
                path=path,
                device=opened_metadata.st_dev,
                inode=opened_metadata.st_ino,
                owner_uid=opened_metadata.st_uid,
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)


def _require_directory_fd_support() -> None:
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.unlink not in os.supports_dir_fd
        or os.rmdir not in os.supports_dir_fd
    ):
        raise OSError(errno.ENOTSUP, "safe directory-relative filesystem operations are unavailable")


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
    )


def open_directory_nofollow(path: Path) -> int:
    _require_directory_fd_support()
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_relative_directory_nofollow(directory_descriptor: int, name: str) -> int:
    """Open a single path component as a directory beneath an already-open directory
    descriptor, following no symlinks. Used to walk a canonical relative path (e.g. a
    signed manifest's ``FileRecord.path``) one component at a time without ever trusting
    a parent directory's enumeration (``os.scandir``) of what children exist."""
    _require_directory_fd_support()
    if "/" in name or name in ("", ".", ".."):
        raise ValueError(f"unsafe directory-relative name: {name!r}")
    metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"path component is not a directory: {name}")
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_descriptor)
    if not _same_entry(metadata, os.fstat(descriptor)):
        os.close(descriptor)
        raise RuntimeError(f"directory component changed while being opened: {name}")
    return descriptor


def rename_entries_noreplace(
    source_directory_descriptor: int,
    source_name: str,
    destination_directory_descriptor: int,
    destination_name: str,
) -> None:
    if "/" in source_name or "/" in destination_name:
        raise ValueError("directory-relative rename names must not contain path separators")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_directory_descriptor,
            os.fsencode(source_name),
            destination_directory_descriptor,
            os.fsencode(destination_name),
            LINUX_RENAME_NOREPLACE,
        )
    else:
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise OSError(errno.ENOTSUP, "safe directory-relative no-replace rename is unavailable")
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_directory_descriptor,
            os.fsencode(source_name),
            destination_directory_descriptor,
            os.fsencode(destination_name),
            DARWIN_RENAME_EXCL,
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(destination_name)
    raise OSError(error, os.strerror(error), source_name)


def rename_entry_noreplace(directory_descriptor: int, source_name: str, destination_name: str) -> None:
    rename_entries_noreplace(
        directory_descriptor,
        source_name,
        directory_descriptor,
        destination_name,
    )


def quarantine_entry(
    directory_descriptor: int,
    source_name: str,
    *,
    quarantine_prefix: str,
    before_rename: Callable[[], None] | None = None,
) -> str:
    if "/" in source_name or "/" in quarantine_prefix:
        raise ValueError("directory-relative quarantine names must not contain path separators")
    source_metadata = os.stat(source_name, dir_fd=directory_descriptor, follow_symlinks=False)
    if before_rename is not None:
        before_rename()
    for _ in range(8):
        quarantine_name = f"{quarantine_prefix}-{secrets.token_hex(16)}"
        try:
            rename_entry_noreplace(directory_descriptor, source_name, quarantine_name)
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(f"could not allocate a unique quarantine for {source_name}")
    os.fsync(directory_descriptor)
    quarantine_metadata = os.stat(
        quarantine_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if not _same_entry(source_metadata, quarantine_metadata):
        raise RuntimeError(f"quarantined entry inode does not match captured source: {quarantine_name}")
    return quarantine_name


def rename_noreplace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            LINUX_RENAME_NOREPLACE,
        )
    else:
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable", destination)
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), DARWIN_RENAME_EXCL)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(destination)
    raise OSError(error, os.strerror(error), destination)


def _remove_directory_contents(directory_descriptor: int) -> None:
    os.fchmod(directory_descriptor, 0o700)
    with os.scandir(directory_descriptor) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                if not _same_entry(metadata, os.fstat(child_descriptor)):
                    raise RuntimeError(f"owned directory entry changed while being opened: {name}")
                _remove_directory_contents(child_descriptor)
            finally:
                os.close(child_descriptor)
            current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if not _same_entry(metadata, current):
                raise RuntimeError(f"owned directory entry changed before removal: {name}")
            os.rmdir(name, dir_fd=directory_descriptor)
            continue
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not _same_entry(metadata, current):
            raise RuntimeError(f"owned directory entry changed before unlink: {name}")
        os.unlink(name, dir_fd=directory_descriptor)


def remove_owned_directory_tree(
    token: DirectoryOwnershipToken,
    *,
    before_root_remove: Callable[[Path], None] | None = None,
) -> None:
    _require_directory_fd_support()
    path = Path(token.path)
    parent_descriptor = open_directory_nofollow(path.parent)
    root_descriptor = None
    try:
        pathname_metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(pathname_metadata.st_mode):
            raise ValueError(f"owned directory root is no longer a directory: {path}")
        if (
            pathname_metadata.st_dev,
            pathname_metadata.st_ino,
            pathname_metadata.st_uid,
        ) != (token.device, token.inode, token.owner_uid):
            raise RuntimeError(f"owned directory root no longer matches its ownership token: {path}")
        root_descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        opened_metadata = os.fstat(root_descriptor)
        if not _same_entry(pathname_metadata, opened_metadata):
            raise RuntimeError(f"owned directory root changed while being opened: {path}")
        _remove_directory_contents(root_descriptor)
        if before_root_remove is not None:
            before_root_remove(path)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not _same_entry(pathname_metadata, current):
            raise RuntimeError(f"owned directory root changed before removal: {path}")
        os.rmdir(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
