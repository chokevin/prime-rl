from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau.eval_tools.fs_safety import rename_entry_noreplace as _rename_entry_noreplace
from tau.eval_tools.fs_safety import rename_noreplace as _rename_noreplace
from tau.eval_tools.json_io import (
    fsync_directory,
    load_json_with_sha256,
    parse_json_bytes,
    write_bytes_exclusive,
    write_json_exclusive,
)
from tau.eval_tools.manifest import (
    FileManifest,
    FrozenEvalManifest,
    RLConfigIdentity,
    build_file_manifest,
    make_tree_immutable,
    validate_file_manifest,
    validate_tree_immutable,
)

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)
ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}$")
SOURCE_CONFIG_REL = "configs/tau/math-7b-h200/train.toml"
_CANCELLATION_CHECK: ContextVar[Callable[[], None] | None] = ContextVar(
    "training_cancellation_check",
    default=None,
)


@dataclass(frozen=True)
class AdapterInstallationToken:
    device: int
    inode: int


@dataclass
class AdapterPublicationState:
    installed_by_invocation: bool = False
    ownership_transferred: bool = False


_ADAPTER_PUBLICATION_STATE: ContextVar[AdapterPublicationState | None] = ContextVar(
    "adapter_publication_state",
    default=None,
)


@contextmanager
def _cancellation_scope(check: Callable[[], None], adapter_state: AdapterPublicationState) -> Iterator[None]:
    cancellation_token = _CANCELLATION_CHECK.set(check)
    adapter_token = _ADAPTER_PUBLICATION_STATE.set(adapter_state)
    try:
        yield
    finally:
        _ADAPTER_PUBLICATION_STATE.reset(adapter_token)
        _CANCELLATION_CHECK.reset(cancellation_token)


def _check_cancelled() -> None:
    check = _CANCELLATION_CHECK.get()
    if check is not None:
        check()


def _capture_adapter_installation_token(path: Path) -> AdapterInstallationToken:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("staged final adapter must be a regular directory")
    return AdapterInstallationToken(device=metadata.st_dev, inode=metadata.st_ino)


def _record_adapter_installation(path: Path, expected: AdapterInstallationToken) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != expected.device or metadata.st_ino != expected.inode:
        raise RuntimeError("installed final adapter inode does not match the validated staging directory")
    state = _ADAPTER_PUBLICATION_STATE.get()
    if state is None:
        return
    if state.installed_by_invocation:
        raise RuntimeError("adapter installation ownership was already recorded")
    state.installed_by_invocation = True


def _transfer_adapter_ownership() -> None:
    state = _ADAPTER_PUBLICATION_STATE.get()
    if state is not None and state.installed_by_invocation:
        state.ownership_transferred = True


@dataclass(frozen=True)
class AttemptPaths:
    directory: Path
    preflight: Path
    resolved_config: Path
    completion: Path
    completion_staging: Path
    publication: Path
    publication_staging: Path
    run_output: Path


def attempt_paths(output_dir: Path, attempt_id: str, *, require_existing: bool) -> AttemptPaths:
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("attempt id must use YYYYMMDDTHHMMSSZ-<16 lowercase hex>")
    output_dir = Path(output_dir).resolve(strict=True)
    if output_dir.is_symlink():
        raise ValueError("training output directory must not be a symlink")
    attempts_dir = output_dir / "attempts"
    attempt_dir = attempts_dir / attempt_id
    if require_existing:
        attempt_dir = attempt_dir.resolve(strict=True)
        if attempt_dir.parent != attempts_dir.resolve(strict=True) or attempt_dir.is_symlink():
            raise ValueError("attempt directory is not the expected canonical child")
    return AttemptPaths(
        directory=attempt_dir,
        preflight=attempt_dir / "preflight.json",
        resolved_config=attempt_dir / "resolved-train.toml",
        completion=attempt_dir / "completion.json",
        completion_staging=attempt_dir / ".completion.json.stage",
        publication=attempt_dir / "publication.json",
        publication_staging=attempt_dir / ".publication.stage",
        run_output=attempt_dir / "run-output",
    )


class SmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["success"] = "success"
    created_at: str
    source_revision: str
    config_path: str
    resolved_configs: list[str]


class TrainingPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    status: Literal["verified"] = "verified"
    attempt_id: str
    created_at: str
    source_revision: str
    manifest_identity_hash: str
    model_name: str
    model_revision: str
    model_files_sha256: str
    training_data_digest: str
    rl_config: RLConfigIdentity
    artifact_output_dir: str
    attempt_output_dir: str
    run_root: str
    model_path: str
    dataset_path: str
    private_config_path: str
    resolved_config_path: str
    resolved_config_sha256: str

    @field_validator(
        "manifest_identity_hash",
        "model_files_sha256",
        "training_data_digest",
        "resolved_config_sha256",
    )
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("preflight digests must be lowercase SHA-256 hex")
        return digest

    @field_validator(
        "artifact_output_dir",
        "attempt_output_dir",
        "run_root",
        "model_path",
        "dataset_path",
        "private_config_path",
        "resolved_config_path",
    )
    @classmethod
    def validate_absolute_path(cls, path: str) -> str:
        if not Path(path).is_absolute():
            raise ValueError("preflight materialization paths must be absolute")
        return path

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, attempt_id: str) -> str:
        if not ATTEMPT_ID_RE.fullmatch(attempt_id):
            raise ValueError("attempt id must use YYYYMMDDTHHMMSSZ-<16 lowercase hex>")
        return attempt_id

    @classmethod
    def load(cls, path: Path) -> "TrainingPreflight":
        payload, _ = load_json_with_sha256(path)
        return cls.model_validate(payload)


class TrainingCompletionAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    status: Literal["success"] = "success"
    attempt_id: str
    manifest_identity_hash: str
    preflight_sha256: str
    resolved_config_sha256: str
    rl_pid: int = Field(gt=1)
    started_at: str
    ended_at: str
    return_code: Literal[0] = 0
    command_argv: list[str]
    executable: str
    rl_config: RLConfigIdentity
    run_root: str
    private_config_path: str
    resolved_config_path: str
    artifact_output_dir: str
    attempt_output_dir: str
    source_step: int = Field(gt=0)
    stable_marker_path: str
    stable_marker_sha256: str
    source_adapter_path: str
    source_adapter_files: FileManifest

    @field_validator(
        "manifest_identity_hash",
        "preflight_sha256",
        "resolved_config_sha256",
        "stable_marker_sha256",
    )
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("completion attestation digests must be lowercase SHA-256 hex")
        return digest

    @field_validator(
        "executable",
        "run_root",
        "private_config_path",
        "resolved_config_path",
        "artifact_output_dir",
        "attempt_output_dir",
        "stable_marker_path",
        "source_adapter_path",
    )
    @classmethod
    def validate_absolute_path(cls, path: str) -> str:
        if not Path(path).is_absolute():
            raise ValueError("completion attestation paths must be absolute")
        return path

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, attempt_id: str) -> str:
        if not ATTEMPT_ID_RE.fullmatch(attempt_id):
            raise ValueError("completion attempt id is invalid")
        return attempt_id

    @field_validator("command_argv")
    @classmethod
    def validate_command_argv(cls, argv: list[str]) -> list[str]:
        if len(argv) != 6 or argv[:5] != ["uv", "run", "--no-sync", "rl", "@"]:
            raise ValueError("completion command must be exact 'uv run --no-sync rl @ <private-config>'")
        if not Path(argv[5]).is_absolute():
            raise ValueError("completion command config path must be absolute")
        return argv

    @model_validator(mode="after")
    def validate_process_interval(self) -> "TrainingCompletionAttestation":
        started_at = datetime.fromisoformat(self.started_at)
        ended_at = datetime.fromisoformat(self.ended_at)
        if started_at.tzinfo is None or ended_at.tzinfo is None:
            raise ValueError("completion timestamps must include a timezone")
        if ended_at < started_at:
            raise ValueError("completion end time precedes start time")
        return self

    @classmethod
    def load(cls, path: Path) -> tuple["TrainingCompletionAttestation", str]:
        payload, digest = load_json_with_sha256(path)
        return cls.model_validate(payload), digest


class TrainingPublicationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["published"] = "published"
    created_at: str
    attempt_id: str
    manifest_identity_hash: str
    preflight_sha256: str
    completion_attestation_sha256: str
    resolved_config_sha256: str
    rl_config: RLConfigIdentity
    source_step: int = Field(gt=0)
    source_adapter_path: str
    source_adapter_files: FileManifest
    final_adapter_path: str
    final_adapter_files: FileManifest

    @field_validator(
        "manifest_identity_hash",
        "preflight_sha256",
        "completion_attestation_sha256",
        "resolved_config_sha256",
    )
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("publication digests must be lowercase SHA-256 hex")
        return digest

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, attempt_id: str) -> str:
        if not ATTEMPT_ID_RE.fullmatch(attempt_id):
            raise ValueError("publication attempt id is invalid")
        return attempt_id

    @field_validator("source_adapter_path", "final_adapter_path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if not Path(path).is_absolute():
            raise ValueError("publication adapter paths must be absolute")
        return path

    @classmethod
    def load(cls, path: Path) -> tuple["TrainingPublicationEvidence", str]:
        payload, digest = load_json_with_sha256(path)
        return cls.model_validate(payload), digest


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[4] = 4
    status: Literal["success"] = "success"
    attempt_id: str
    created_at: str
    source_revision: str
    manifest_identity_hash: str
    model_name: str
    model_revision: str
    model_files_sha256: str
    training_data_digest: str
    rl_config: RLConfigIdentity
    preflight_sha256: str
    completion_attestation_sha256: str
    publication_evidence_sha256: str
    resolved_config_sha256: str
    resolved_config_path: str
    source_step: int = Field(gt=0)
    source_adapter_path: str
    final_adapter_path: str
    adapter_sha256: str
    adapter_files: FileManifest
    lora_rank: int = Field(gt=0)

    @field_validator(
        "adapter_sha256",
        "model_files_sha256",
        "training_data_digest",
        "manifest_identity_hash",
        "preflight_sha256",
        "completion_attestation_sha256",
        "publication_evidence_sha256",
        "resolved_config_sha256",
    )
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("artifact digests must be lowercase SHA-256 hex")
        return digest

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, attempt_id: str) -> str:
        if not ATTEMPT_ID_RE.fullmatch(attempt_id):
            raise ValueError("training result attempt id is invalid")
        return attempt_id

    @field_validator("source_revision", "model_revision")
    @classmethod
    def validate_git_sha(cls, revision: str) -> str:
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("source and model revisions must be lowercase 40-character commit SHAs")
        return revision

    @field_validator("source_adapter_path", "final_adapter_path", "resolved_config_path")
    @classmethod
    def validate_absolute_path(cls, path: str) -> str:
        if not Path(path).is_absolute():
            raise ValueError("training artifact paths must be absolute")
        return path

    @field_validator("adapter_files")
    @classmethod
    def validate_adapter_files(cls, manifest: FileManifest) -> FileManifest:
        names = {record.path for record in manifest.files}
        if ADAPTER_CONFIG not in names or len(names) != 2:
            raise ValueError("training result adapter manifest must contain config plus exactly one weight file")
        if not any(name in names for name in ADAPTER_WEIGHT_FILES):
            raise ValueError("training result adapter manifest is missing its supported weight file")
        return manifest

    @model_validator(mode="after")
    def validate_adapter_digest(self):
        if self.adapter_sha256 != self.adapter_files.aggregate_sha256:
            raise ValueError("adapter aggregate digest does not match adapter file manifest")
        return self

    @classmethod
    def load(cls, path: Path) -> "TrainingResult":
        payload, _ = load_json_with_sha256(path)
        return cls.model_validate(payload)


def sha256_files(path: Path, files: list[Path]) -> str:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {path}")
    if path.is_symlink():
        raise ValueError(f"artifact directory is a symlink: {path}")
    digest = hashlib.sha256()
    if not files:
        raise ValueError("at least one artifact file is required")
    for item in sorted(files):
        if item.is_symlink():
            raise ValueError(f"artifact file is a symlink: {item}")
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_adapter_directory(path: Path, *, expected_rank: int) -> str:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"adapter directory is a symlink: {path}")
    config_path = path / ADAPTER_CONFIG
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(f"adapter config not found: {config_path}")
    config, _ = load_json_with_sha256(config_path)
    if not isinstance(config, dict) or config.get("r") != expected_rank:
        actual_rank = config.get("r") if isinstance(config, dict) else None
        raise ValueError(f"adapter rank is {actual_rank!r}, expected {expected_rank}")
    weight_files = [path / name for name in ADAPTER_WEIGHT_FILES if (path / name).is_file()]
    if len(weight_files) != 1:
        raise ValueError(
            f"adapter must contain exactly one supported weight file, found {[item.name for item in weight_files]}"
        )
    manifest = build_file_manifest(path)
    expected_names = {ADAPTER_CONFIG, weight_files[0].name}
    actual_names = {record.path for record in manifest.files}
    if actual_names != expected_names:
        raise ValueError(f"adapter directory files are {sorted(actual_names)}, expected {sorted(expected_names)}")
    return manifest.aggregate_sha256


def copy_adapter_exclusive(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=False, exist_ok=False)
    required_files = [source / ADAPTER_CONFIG]
    required_files.extend(source / name for name in ADAPTER_WEIGHT_FILES if (source / name).is_file())
    for source_file in required_files:
        destination_file = destination / source_file.name
        with source_file.open("rb") as input_file, destination_file.open("xb") as output_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    fsync_directory(destination)


def materialize_adapter_for_eval(
    *,
    result: TrainingResult,
    durable_adapter_path: Path,
    run_root: Path,
) -> Path:
    run_root = Path(run_root).resolve(strict=True)
    durable_adapter_path = Path(durable_adapter_path)
    destination = run_root / "adapter"
    copy_adapter_exclusive(durable_adapter_path, destination)
    validate_file_manifest(destination, result.adapter_files)
    make_tree_immutable(destination)
    validate_tree_immutable(destination, result.adapter_files)
    marker = run_root / "adapter-materialization-complete"
    write_bytes_exclusive(marker, (result.adapter_sha256 + "\n").encode())
    marker.chmod(0o444)
    return destination


def select_final_adapter(
    weights_dir: Path,
    *,
    expected_step: int,
    expected_rank: int,
) -> tuple[int, Path, str]:
    weights_dir = Path(weights_dir)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    if expected_step < 1:
        raise ValueError(f"expected final step must be positive, got {expected_step}")
    step_dir = weights_dir / f"step_{expected_step}"
    stable = step_dir / "STABLE"
    if stable.is_symlink() or not stable.is_file() or not stat.S_ISREG(stable.stat(follow_symlinks=False).st_mode):
        raise FileNotFoundError(f"expected final checkpoint is not stable: {step_dir / 'STABLE'}")
    adapter_path = step_dir / "lora_adapters"
    digest = validate_adapter_directory(adapter_path, expected_rank=expected_rank)
    return expected_step, adapter_path, digest


def write_smoke_result(
    *,
    output_path: Path,
    source_revision: str,
    config_path: str,
    configs_dir: Path,
) -> SmokeResult:
    required = ["inference.toml", "orchestrator.toml", "trainer.toml"]
    missing = [name for name in required if not (Path(configs_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"rl --dry-run did not write required resolved config(s): {missing}")
    result = SmokeResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=source_revision,
        config_path=config_path,
        resolved_configs=required,
    )
    write_json_exclusive(output_path, result.model_dump())
    return result


def write_training_preflight(
    *,
    output_path: Path,
    attempt_id: str,
    manifest: FrozenEvalManifest,
    artifact_output_dir: Path,
    attempt_output_dir: Path,
    run_root: Path,
    model_path: Path,
    dataset_path: Path,
    private_config_path: Path,
    resolved_config_path: Path,
) -> TrainingPreflight:
    from tau.eval_tools.live.validate_training_data_live import validate_resolved_rl_config

    if manifest.rl_config is None:
        raise ValueError("training preflight requires a finalized RL config identity")
    output_path = Path(output_path)
    paths = attempt_paths(artifact_output_dir, attempt_id, require_existing=True)
    if output_path != paths.preflight or Path(resolved_config_path) != paths.resolved_config:
        raise ValueError("preflight/config paths must be the fixed files for this exact attempt")
    _, resolved_bytes, config_identity = validate_resolved_rl_config(
        resolved_config_path,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=attempt_output_dir,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    if config_identity != manifest.rl_config:
        raise ValueError("recomputed resolved RL config identity does not match frozen manifest")
    result = TrainingPreflight(
        attempt_id=attempt_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=manifest.source_revision,
        manifest_identity_hash=manifest.identity_hash(),
        model_name=manifest.model.name,
        model_revision=manifest.model.revision,
        model_files_sha256=manifest.model.file_manifest.aggregate_sha256,
        training_data_digest=manifest.training_data.record_digest,
        rl_config=config_identity,
        artifact_output_dir=str(artifact_output_dir),
        attempt_output_dir=str(attempt_output_dir),
        run_root=str(run_root),
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        private_config_path=str(private_config_path),
        resolved_config_path=str(resolved_config_path),
        resolved_config_sha256=hashlib.sha256(resolved_bytes).hexdigest(),
    )
    write_json_exclusive(output_path, result.model_dump())
    return result


def _before_staging_quarantine(_path: Path) -> None:
    pass


def _quarantine_owned_staging(path: Path, *, expected_name: str) -> Path | None:
    path = Path(path)
    allowed_names = {
        ".completion.json.stage",
        ".publication.stage",
        "completion.json",
        "publication.json",
    }
    if expected_name not in allowed_names:
        raise ValueError(f"unsupported attempt evidence name for quarantine: {expected_name}")
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise OSError(errno.ENOTSUP, "safe no-follow attempt staging quarantine is unavailable", path)
    if (
        path.name != expected_name
        or not ATTEMPT_ID_RE.fullmatch(path.parent.name)
        or path.parent.parent.name != "attempts"
    ):
        raise ValueError(f"refusing to quarantine unsafe staging path: {path}")
    canonical_parent = path.parent.resolve(strict=True)
    if canonical_parent != Path(os.path.abspath(path.parent)):
        raise ValueError(f"attempt staging parent is not canonical: {path.parent}")
    path = canonical_parent / path.name
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = os.open(path.parent, directory_flags)
    source_descriptor = None
    try:
        if not stat.S_ISDIR(os.fstat(parent_descriptor).st_mode):
            raise ValueError(f"attempt staging parent must be a directory: {path.parent}")
        try:
            pathname_metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        expected_identity = (pathname_metadata.st_dev, pathname_metadata.st_ino)
        if stat.S_ISREG(pathname_metadata.st_mode) or stat.S_ISDIR(pathname_metadata.st_mode):
            file_flags = os.O_RDONLY
            if stat.S_ISDIR(pathname_metadata.st_mode):
                file_flags |= os.O_DIRECTORY
            file_flags |= os.O_NOFOLLOW
            source_descriptor = os.open(path.name, file_flags, dir_fd=parent_descriptor)
            opened_metadata = os.fstat(source_descriptor)
            if (
                stat.S_IFMT(opened_metadata.st_mode) != stat.S_IFMT(pathname_metadata.st_mode)
                or (opened_metadata.st_dev, opened_metadata.st_ino) != expected_identity
            ):
                raise RuntimeError(f"staging path changed while being opened: {path}")
        _before_staging_quarantine(path)
        for _ in range(8):
            quarantine_name = f"{expected_name}.quarantine-{secrets.token_hex(16)}"
            try:
                _rename_entry_noreplace(parent_descriptor, path.name, quarantine_name)
            except FileExistsError:
                continue
            break
        else:
            raise FileExistsError("could not allocate a unique staging quarantine name")
        os.fsync(parent_descriptor)
        quarantine_metadata = os.stat(quarantine_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISREG(quarantine_metadata.st_mode) or stat.S_ISDIR(quarantine_metadata.st_mode):
            file_flags = os.O_RDONLY
            if stat.S_ISDIR(quarantine_metadata.st_mode):
                file_flags |= os.O_DIRECTORY
            file_flags |= os.O_NOFOLLOW
            descriptor = os.open(quarantine_name, file_flags, dir_fd=parent_descriptor)
            try:
                reopened_metadata = os.fstat(descriptor)
                quarantine_identity = (reopened_metadata.st_dev, reopened_metadata.st_ino)
            finally:
                os.close(descriptor)
        else:
            quarantine_identity = (quarantine_metadata.st_dev, quarantine_metadata.st_ino)
        quarantine_path = path.parent / quarantine_name
        if quarantine_identity != expected_identity:
            raise RuntimeError(f"quarantined staging inode does not match the captured entry: {quarantine_path}")
        if expected_name == ".publication.stage" and not stat.S_ISDIR(quarantine_metadata.st_mode):
            raise ValueError(f"quarantined attempt evidence has the wrong file type: {quarantine_path}")
        return quarantine_path
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(parent_descriptor)


def _write_json_staged_noreplace(
    *,
    final_path: Path,
    staging_path: Path,
    payload: dict,
    on_installed: Callable[[], None] | None = None,
) -> None:
    final_path = Path(final_path)
    staging_path = Path(staging_path)
    _check_cancelled()
    if final_path.parent != staging_path.parent:
        if final_path.parent.stat().st_dev != staging_path.parent.stat().st_dev:
            raise RuntimeError("JSON staging and destination must be on the same filesystem")
    if os.path.lexists(final_path):
        raise FileExistsError(final_path)
    if os.path.lexists(staging_path):
        if staging_path.name != ".completion.json.stage":
            raise FileExistsError(f"refusing to replace existing JSON staging evidence: {staging_path}")
        _quarantine_owned_staging(
            staging_path,
            expected_name=".completion.json.stage",
        )
    _check_cancelled()
    write_json_exclusive(staging_path, payload)
    _check_cancelled()
    _rename_noreplace(staging_path, final_path)
    fsync_directory(final_path.parent)
    if on_installed is not None:
        on_installed()
    _check_cancelled()


@dataclass(frozen=True)
class _CompletionEvidenceSnapshot:
    raw: bytes
    digest: str
    attestation: TrainingCompletionAttestation
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@contextmanager
def _open_completion_evidence(path: Path) -> Iterator[_CompletionEvidenceSnapshot]:
    path = Path(path)
    pathname_metadata = os.lstat(path)
    if not stat.S_ISREG(pathname_metadata.st_mode):
        raise ValueError(f"completion evidence path must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"completion evidence must be a regular file: {path}")
        if (before.st_dev, before.st_ino) != (pathname_metadata.st_dev, pathname_metadata.st_ino):
            raise RuntimeError(f"completion evidence path changed while being opened: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        metadata = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != metadata:
            raise RuntimeError(f"completion evidence changed while being read: {path}")
        raw = b"".join(chunks)
        attestation = TrainingCompletionAttestation.model_validate(parse_json_bytes(raw))
        os.fsync(descriptor)
        yield _CompletionEvidenceSnapshot(
            raw=raw,
            digest=hashlib.sha256(raw).hexdigest(),
            attestation=attestation,
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _require_snapshot_path(path: Path, snapshot: _CompletionEvidenceSnapshot) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"completion staging path is no longer a regular file: {path}")
    if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns) != (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    ):
        raise RuntimeError("completion staging path changed after handle-bound validation")


def _before_completion_stage_promotion(_path: Path) -> None:
    pass


def _after_completion_stage_promotion(_path: Path) -> None:
    pass


def _load_attempt_evidence(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    attempt_id: str,
    expected_step: int,
    expected_rank: int,
    completion_path: Path | None = None,
    completion_evidence: tuple[TrainingCompletionAttestation, str] | None = None,
) -> tuple[
    AttemptPaths,
    TrainingPreflight,
    str,
    TrainingCompletionAttestation,
    str,
    RLConfigIdentity,
    str,
]:
    from tau.eval_tools.live.validate_training_data_live import validate_resolved_rl_config

    if manifest.rl_config is None:
        raise ValueError("attempt evidence requires a finalized manifest RL config")
    paths = attempt_paths(output_dir, attempt_id, require_existing=True)
    if paths.run_output.is_symlink() or paths.run_output.resolve(strict=True).parent != paths.directory:
        raise ValueError("attempt run output must be the fixed canonical directory")
    preflight_payload, preflight_sha256 = load_json_with_sha256(paths.preflight)
    preflight = TrainingPreflight.model_validate(preflight_payload)
    if completion_evidence is None:
        attestation, attestation_sha256 = TrainingCompletionAttestation.load(completion_path or paths.completion)
    else:
        attestation, attestation_sha256 = completion_evidence
    if preflight.attempt_id != attempt_id or attestation.attempt_id != attempt_id:
        raise ValueError("attempt evidence IDs do not match the requested attempt")
    output_dir = Path(output_dir).resolve(strict=True)
    if (
        Path(preflight.artifact_output_dir) != output_dir
        or Path(preflight.attempt_output_dir) != paths.run_output
        or Path(preflight.resolved_config_path) != paths.resolved_config
    ):
        raise ValueError("preflight paths do not match the fixed attempt layout")
    _, resolved_bytes, config_identity = validate_resolved_rl_config(
        paths.resolved_config,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=Path(preflight.model_path),
        dataset_path=Path(preflight.dataset_path),
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    resolved_config_sha256 = hashlib.sha256(resolved_bytes).hexdigest()
    if config_identity != manifest.rl_config or config_identity != preflight.rl_config:
        raise ValueError("recomputed RL config identity does not match manifest/preflight")
    if attestation.rl_config != config_identity:
        raise ValueError("completion attestation RL config identity does not match recomputed config")
    if preflight.manifest_identity_hash != manifest.identity_hash():
        raise ValueError("preflight manifest identity does not match frozen manifest")
    if preflight.source_revision != manifest.source_revision:
        raise ValueError("preflight source revision does not match frozen manifest")
    if preflight.model_name != manifest.model.name or preflight.model_revision != manifest.model.revision:
        raise ValueError("preflight model identity does not match frozen manifest")
    if preflight.model_files_sha256 != manifest.model.file_manifest.aggregate_sha256:
        raise ValueError("preflight model content identity does not match frozen manifest")
    if preflight.training_data_digest != manifest.training_data.record_digest:
        raise ValueError("preflight training identity does not match frozen manifest")
    if preflight.resolved_config_sha256 != resolved_config_sha256:
        raise ValueError("resolved config bytes do not match preflight digest")
    if (
        attestation.manifest_identity_hash != preflight.manifest_identity_hash
        or attestation.preflight_sha256 != preflight_sha256
        or attestation.resolved_config_sha256 != resolved_config_sha256
    ):
        raise ValueError("completion identity/config/preflight digests do not match immutable evidence")
    if (
        attestation.run_root != preflight.run_root
        or attestation.private_config_path != preflight.private_config_path
        or attestation.resolved_config_path != preflight.resolved_config_path
        or Path(attestation.artifact_output_dir) != output_dir
        or Path(attestation.attempt_output_dir) != paths.run_output
    ):
        raise ValueError("completion paths do not match preflight and fixed attempt layout")
    expected_argv = ["uv", "run", "--no-sync", "rl", "@", preflight.private_config_path]
    executable = Path(attestation.executable)
    if attestation.command_argv != expected_argv or not executable.is_absolute() or executable.name != "uv":
        raise ValueError("completion command does not attest the exact trusted RL invocation")
    if datetime.fromisoformat(attestation.ended_at) < datetime.fromisoformat(attestation.started_at):
        raise ValueError("completion process end time precedes its start time")
    if attestation.source_step != expected_step:
        raise ValueError("completion source step does not match the fixed final step")
    expected_stable = paths.run_output / "weights" / f"step_{expected_step}" / "STABLE"
    expected_adapter = expected_stable.parent / "lora_adapters"
    if Path(attestation.stable_marker_path) != expected_stable:
        raise ValueError("completion STABLE marker path does not match fixed final step")
    if expected_stable.is_symlink() or not expected_stable.is_file():
        raise FileNotFoundError("attested final checkpoint STABLE marker is missing")
    stable_sha = hashlib.sha256(expected_stable.read_bytes()).hexdigest()
    if stable_sha != attestation.stable_marker_sha256:
        raise ValueError("final checkpoint STABLE marker digest changed")
    if Path(attestation.source_adapter_path) != expected_adapter:
        raise ValueError("completion adapter path does not match fixed final step")
    validate_adapter_directory(expected_adapter, expected_rank=expected_rank)
    validate_file_manifest(expected_adapter, attestation.source_adapter_files)
    return (
        paths,
        preflight,
        preflight_sha256,
        attestation,
        attestation_sha256,
        config_identity,
        resolved_config_sha256,
    )


def _load_or_recover_attempt_evidence(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    attempt_id: str,
    expected_step: int,
    expected_rank: int,
    recovery: bool,
) -> tuple[
    AttemptPaths,
    TrainingPreflight,
    str,
    TrainingCompletionAttestation,
    str,
    RLConfigIdentity,
    str,
]:
    paths = attempt_paths(output_dir, attempt_id, require_existing=True)
    final_exists = os.path.lexists(paths.completion)
    stage_exists = os.path.lexists(paths.completion_staging)

    if final_exists:
        final_evidence = _load_attempt_evidence(
            output_dir=output_dir,
            manifest=manifest,
            attempt_id=attempt_id,
            expected_step=expected_step,
            expected_rank=expected_rank,
        )
        _check_cancelled()
        if stage_exists:
            try:
                with _open_completion_evidence(paths.completion_staging) as staged_snapshot:
                    staged_evidence = _load_attempt_evidence(
                        output_dir=output_dir,
                        manifest=manifest,
                        attempt_id=attempt_id,
                        expected_step=expected_step,
                        expected_rank=expected_rank,
                        completion_evidence=(staged_snapshot.attestation, staged_snapshot.digest),
                    )
            except Exception as error:
                try:
                    _quarantine_owned_staging(
                        paths.completion_staging,
                        expected_name=".completion.json.stage",
                    )
                except Exception as cleanup_error:
                    raise ValueError(
                        "final completion is valid but its unsafe staging entry could not be quarantined"
                    ) from cleanup_error
                raise ValueError(
                    "final completion is valid; malformed staged completion was quarantined and recovery must be retried"
                ) from error
            _check_cancelled()
            if staged_evidence[3] != final_evidence[3] or staged_evidence[4] != final_evidence[4]:
                raise ValueError("staged completion attestation does not match finalized completion evidence")
            _check_cancelled()
            _quarantine_owned_staging(
                paths.completion_staging,
                expected_name=".completion.json.stage",
            )
        return final_evidence

    if not stage_exists or not recovery:
        return _load_attempt_evidence(
            output_dir=output_dir,
            manifest=manifest,
            attempt_id=attempt_id,
            expected_step=expected_step,
            expected_rank=expected_rank,
        )

    _check_cancelled()
    promoted = False
    try:
        with _open_completion_evidence(paths.completion_staging) as staged_snapshot:
            staged_evidence = _load_attempt_evidence(
                output_dir=output_dir,
                manifest=manifest,
                attempt_id=attempt_id,
                expected_step=expected_step,
                expected_rank=expected_rank,
                completion_evidence=(staged_snapshot.attestation, staged_snapshot.digest),
            )
            _check_cancelled()
            _before_completion_stage_promotion(paths.completion_staging)
            _require_snapshot_path(paths.completion_staging, staged_snapshot)
            _check_cancelled()
            _rename_noreplace(paths.completion_staging, paths.completion)
            promoted = True
            fsync_directory(paths.directory)
            _after_completion_stage_promotion(paths.completion)
            with _open_completion_evidence(paths.completion) as final_snapshot:
                if (
                    final_snapshot.raw != staged_snapshot.raw
                    or final_snapshot.digest != staged_snapshot.digest
                    or final_snapshot.attestation != staged_snapshot.attestation
                    or (final_snapshot.device, final_snapshot.inode) != (staged_snapshot.device, staged_snapshot.inode)
                ):
                    raise RuntimeError("promoted completion evidence does not match its validated staging handle")
                final_evidence = _load_attempt_evidence(
                    output_dir=output_dir,
                    manifest=manifest,
                    attempt_id=attempt_id,
                    expected_step=expected_step,
                    expected_rank=expected_rank,
                    completion_evidence=(final_snapshot.attestation, final_snapshot.digest),
                )
            if final_evidence != staged_evidence:
                raise RuntimeError("promoted completion validation changed after atomic installation")
    except Exception as error:
        if not promoted:
            _quarantine_owned_staging(
                paths.completion_staging,
                expected_name=".completion.json.stage",
            )
        _check_cancelled()
        if promoted:
            raise ValueError("promoted completion attestation failed strict handle-bound reload") from error
        raise ValueError("staged completion attestation is incomplete or invalid") from error
    _check_cancelled()
    return final_evidence


def publish_training_result(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    attempt_id: str,
    expected_step: int,
    expected_rank: int,
    recovery: bool = False,
) -> TrainingResult:
    output_dir = Path(output_dir).resolve(strict=True)
    _check_cancelled()
    (
        paths,
        preflight,
        preflight_sha256,
        attestation,
        attestation_sha256,
        config_identity,
        resolved_config_sha256,
    ) = _load_or_recover_attempt_evidence(
        output_dir=output_dir,
        manifest=manifest,
        attempt_id=attempt_id,
        expected_step=expected_step,
        expected_rank=expected_rank,
        recovery=recovery,
    )
    _check_cancelled()
    result_path = output_dir / "training-result.json"
    final_adapter = output_dir / "final-adapter"
    staging = paths.publication_staging
    if os.path.lexists(result_path):
        result = TrainingResult.load(result_path)
        if result.attempt_id != attempt_id:
            raise ValueError(f"training already completed by different attempt {result.attempt_id}")
        validate_adapter_handoff(
            result=result,
            manifest=manifest,
            training_output_dir=output_dir,
            expected_adapter_path=final_adapter,
            expected_step=expected_step,
            expected_rank=expected_rank,
        )
        _check_cancelled()
        _quarantine_owned_staging(staging, expected_name=".publication.stage")
        _check_cancelled()
        return result
    existing_publication = None
    publication_sha256 = None
    if os.path.lexists(paths.publication):
        existing_publication, publication_sha256 = TrainingPublicationEvidence.load(paths.publication)
    _check_cancelled()
    _quarantine_owned_staging(staging, expected_name=".publication.stage")
    _check_cancelled()
    step, source_adapter, source_digest = select_final_adapter(
        paths.run_output / "weights",
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    _check_cancelled()
    if step != attestation.source_step or source_adapter != Path(attestation.source_adapter_path):
        raise ValueError("selected stable adapter does not match successful RL completion attestation")
    source_manifest = build_file_manifest(source_adapter)
    if source_manifest != attestation.source_adapter_files or source_manifest.aggregate_sha256 != source_digest:
        raise RuntimeError("source adapter changed after successful process attestation")
    _check_cancelled()
    if os.path.lexists(final_adapter):
        if not recovery:
            raise FileExistsError(
                "final adapter already exists without a completed result; use explicit verified publication recovery"
            )
        published_digest = validate_adapter_directory(final_adapter, expected_rank=expected_rank)
        if published_digest != source_digest:
            raise ValueError("incomplete final adapter does not match the fixed stable source checkpoint")
        _check_cancelled()
    else:
        if existing_publication is not None:
            raise ValueError("publication evidence exists but its final adapter is missing")
        staging.mkdir()
        if staging.stat().st_dev != output_dir.stat().st_dev:
            raise RuntimeError("publication staging and destination must be on the same filesystem")
        staged_adapter = staging / "final-adapter"
        copy_adapter_exclusive(source_adapter, staged_adapter)
        _check_cancelled()
        staged_digest = validate_adapter_directory(staged_adapter, expected_rank=expected_rank)
        if staged_digest != source_digest:
            raise RuntimeError(f"staged adapter digest {staged_digest} does not match source digest {source_digest}")
        staged_adapter_token = _capture_adapter_installation_token(staged_adapter)
        _check_cancelled()
        _rename_noreplace(staged_adapter, final_adapter)
        _record_adapter_installation(final_adapter, staged_adapter_token)
        fsync_directory(output_dir)
        _check_cancelled()
        published_digest = validate_adapter_directory(final_adapter, expected_rank=expected_rank)
    if published_digest != source_digest:
        raise RuntimeError(f"published adapter digest {published_digest} does not match source digest {source_digest}")
    published_manifest = build_file_manifest(final_adapter)
    _check_cancelled()
    publication = TrainingPublicationEvidence(
        created_at=datetime.now(timezone.utc).isoformat(),
        attempt_id=attempt_id,
        manifest_identity_hash=manifest.identity_hash(),
        preflight_sha256=preflight_sha256,
        completion_attestation_sha256=attestation_sha256,
        resolved_config_sha256=resolved_config_sha256,
        rl_config=config_identity,
        source_step=attestation.source_step,
        source_adapter_path=attestation.source_adapter_path,
        source_adapter_files=source_manifest,
        final_adapter_path=str(final_adapter),
        final_adapter_files=published_manifest,
    )
    if existing_publication is not None:
        if existing_publication != publication.model_copy(update={"created_at": existing_publication.created_at}):
            raise ValueError("existing attempt publication evidence does not match verified artifacts")
        publication = existing_publication
    else:
        if not os.path.lexists(staging):
            staging.mkdir()
            fsync_directory(staging.parent)
        _write_json_staged_noreplace(
            final_path=paths.publication,
            staging_path=staging / paths.publication.name,
            payload=publication.model_dump(),
            on_installed=_transfer_adapter_ownership,
        )
        _check_cancelled()
        _, publication_sha256 = load_json_with_sha256(paths.publication)
        _check_cancelled()
    if publication_sha256 is None:
        raise RuntimeError("publication evidence digest was not established")
    result = TrainingResult(
        attempt_id=attempt_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=preflight.source_revision,
        manifest_identity_hash=preflight.manifest_identity_hash,
        model_name=preflight.model_name,
        model_revision=preflight.model_revision,
        model_files_sha256=preflight.model_files_sha256,
        training_data_digest=preflight.training_data_digest,
        rl_config=config_identity,
        preflight_sha256=preflight_sha256,
        completion_attestation_sha256=attestation_sha256,
        publication_evidence_sha256=publication_sha256,
        resolved_config_sha256=resolved_config_sha256,
        resolved_config_path=str(paths.resolved_config),
        source_step=attestation.source_step,
        source_adapter_path=attestation.source_adapter_path,
        final_adapter_path=str(final_adapter),
        adapter_sha256=published_digest,
        adapter_files=published_manifest,
        lora_rank=expected_rank,
    )
    _check_cancelled()
    if not os.path.lexists(staging):
        staging.mkdir()
    staged_result = staging / result_path.name
    _write_json_staged_noreplace(
        final_path=result_path,
        staging_path=staged_result,
        payload=result.model_dump(),
    )
    _quarantine_owned_staging(staging, expected_name=".publication.stage")
    fsync_directory(output_dir)
    return result


def validate_adapter_handoff(
    *,
    result: TrainingResult,
    manifest: FrozenEvalManifest,
    training_output_dir: Path,
    expected_adapter_path: Path,
    expected_step: int,
    expected_rank: int,
) -> None:
    (
        paths,
        _,
        preflight_sha256,
        attestation,
        attestation_sha256,
        config_identity,
        resolved_digest,
    ) = _load_attempt_evidence(
        output_dir=training_output_dir,
        manifest=manifest,
        attempt_id=result.attempt_id,
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    if result.manifest_identity_hash != manifest.identity_hash():
        raise ValueError("training result manifest identity does not match the frozen eval manifest")
    if result.source_revision != manifest.source_revision:
        raise ValueError("training result source revision does not match the frozen eval manifest")
    if result.model_name != manifest.model.name or result.model_revision != manifest.model.revision:
        raise ValueError("training result base model identity does not match the frozen eval manifest")
    if result.model_files_sha256 != manifest.model.file_manifest.aggregate_sha256:
        raise ValueError("training result model file identity does not match the frozen eval manifest")
    if result.training_data_digest != manifest.training_data.record_digest:
        raise ValueError("training result training-data identity does not match the frozen eval manifest")
    if result.rl_config != manifest.rl_config or result.rl_config != config_identity:
        raise ValueError("training result effective RL config identity does not match the frozen eval manifest")
    if result.preflight_sha256 != preflight_sha256:
        raise ValueError("training result preflight digest does not match immutable preflight evidence")
    if result.completion_attestation_sha256 != attestation_sha256:
        raise ValueError("training result completion digest does not match immutable completion attestation")
    publication, publication_sha256 = TrainingPublicationEvidence.load(paths.publication)
    if result.publication_evidence_sha256 != publication_sha256:
        raise ValueError("training result publication digest does not match attempt evidence")
    if (
        publication.attempt_id != result.attempt_id
        or publication.manifest_identity_hash != manifest.identity_hash()
        or publication.preflight_sha256 != preflight_sha256
        or publication.completion_attestation_sha256 != attestation_sha256
        or publication.rl_config != config_identity
    ):
        raise ValueError("attempt publication evidence identity does not match recomputed contract")
    if result.resolved_config_sha256 != resolved_digest:
        raise ValueError("training result resolved config digest does not match exact published bytes")
    if Path(result.resolved_config_path) != paths.resolved_config:
        raise ValueError("training result resolved config path is not the fixed attempt path")
    if result.source_step != expected_step:
        raise ValueError(f"training result source step is {result.source_step}, expected {expected_step}")
    if result.lora_rank != expected_rank:
        raise ValueError(f"training result LoRA rank is {result.lora_rank}, expected {expected_rank}")
    expected_adapter_path = Path(expected_adapter_path)
    expected_source_path = paths.run_output / "weights" / f"step_{result.source_step}" / "lora_adapters"
    if Path(result.source_adapter_path) != expected_source_path:
        raise ValueError(
            f"training result source adapter path is {result.source_adapter_path}, expected {expected_source_path}"
        )
    if Path(result.final_adapter_path) != expected_adapter_path:
        raise ValueError(
            f"training result final adapter path is {result.final_adapter_path}, expected {expected_adapter_path}"
        )
    digest = validate_adapter_directory(expected_adapter_path, expected_rank=expected_rank)
    if digest != result.adapter_sha256:
        raise ValueError(f"final adapter digest {digest} does not match training result digest {result.adapter_sha256}")
    validate_file_manifest(expected_adapter_path, result.adapter_files)
    if (
        publication.resolved_config_sha256 != resolved_digest
        or publication.source_step != result.source_step
        or Path(publication.source_adapter_path) != expected_source_path
        or publication.source_adapter_files != attestation.source_adapter_files
        or Path(publication.final_adapter_path) != expected_adapter_path
        or publication.final_adapter_files != result.adapter_files
    ):
        raise ValueError("attempt publication evidence does not match resolved config and adapter handoff")
