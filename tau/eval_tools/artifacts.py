from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau.eval_tools.json_io import fsync_directory, load_json_with_sha256, write_json_exclusive
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


@dataclass(frozen=True)
class AttemptPaths:
    directory: Path
    preflight: Path
    resolved_config: Path
    completion: Path
    publication: Path
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
        publication=attempt_dir / "publication.json",
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
    with marker.open("xb") as output:
        output.write((result.adapter_sha256 + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    marker.chmod(0o444)
    fsync_directory(run_root)
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


def _remove_owned_staging(path: Path, attempt_id: str) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.name != f".training-publication-{attempt_id}.stage":
        raise ValueError(f"refusing to remove unowned publication staging path: {path}")
    shutil.rmtree(path)
    fsync_directory(path.parent)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        os.rename(source, destination)
        return
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), destination)


def _load_attempt_evidence(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    attempt_id: str,
    expected_step: int,
    expected_rank: int,
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
    attestation, attestation_sha256 = TrainingCompletionAttestation.load(paths.completion)
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
    if attestation.command_argv != expected_argv or Path(attestation.executable).name != "uv":
        raise ValueError("completion command does not attest the exact trusted RL invocation")
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


def publish_training_result(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    attempt_id: str,
    expected_step: int,
    expected_rank: int,
) -> TrainingResult:
    output_dir = Path(output_dir).resolve(strict=True)
    (
        paths,
        preflight,
        preflight_sha256,
        attestation,
        attestation_sha256,
        config_identity,
        resolved_config_sha256,
    ) = _load_attempt_evidence(
        output_dir=output_dir,
        manifest=manifest,
        attempt_id=attempt_id,
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    result_path = output_dir / "training-result.json"
    final_adapter = output_dir / "final-adapter"
    staging = output_dir / f".training-publication-{attempt_id}.stage"
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
        _remove_owned_staging(staging, attempt_id)
        return result
    _remove_owned_staging(staging, attempt_id)
    step, source_adapter, source_digest = select_final_adapter(
        paths.run_output / "weights",
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    if step != attestation.source_step or source_adapter != Path(attestation.source_adapter_path):
        raise ValueError("selected stable adapter does not match successful RL completion attestation")
    source_manifest = build_file_manifest(source_adapter)
    if source_manifest != attestation.source_adapter_files or source_manifest.aggregate_sha256 != source_digest:
        raise RuntimeError("source adapter changed after successful process attestation")
    if os.path.lexists(final_adapter):
        published_digest = validate_adapter_directory(final_adapter, expected_rank=expected_rank)
        if published_digest != source_digest:
            raise ValueError("incomplete final adapter does not match the fixed stable source checkpoint")
    else:
        staging.mkdir()
        if staging.stat().st_dev != output_dir.stat().st_dev:
            raise RuntimeError("publication staging and destination must be on the same filesystem")
        staged_adapter = staging / "final-adapter"
        copy_adapter_exclusive(source_adapter, staged_adapter)
        staged_digest = validate_adapter_directory(staged_adapter, expected_rank=expected_rank)
        if staged_digest != source_digest:
            raise RuntimeError(f"staged adapter digest {staged_digest} does not match source digest {source_digest}")
        _rename_noreplace(staged_adapter, final_adapter)
        fsync_directory(output_dir)
        published_digest = validate_adapter_directory(final_adapter, expected_rank=expected_rank)
    if published_digest != source_digest:
        raise RuntimeError(f"published adapter digest {published_digest} does not match source digest {source_digest}")
    published_manifest = build_file_manifest(final_adapter)
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
    if os.path.lexists(paths.publication):
        existing_publication, publication_sha256 = TrainingPublicationEvidence.load(paths.publication)
        if existing_publication != publication.model_copy(update={"created_at": existing_publication.created_at}):
            raise ValueError("existing attempt publication evidence does not match verified artifacts")
        publication = existing_publication
    else:
        write_json_exclusive(paths.publication, publication.model_dump())
        _, publication_sha256 = load_json_with_sha256(paths.publication)
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
    if not os.path.lexists(staging):
        staging.mkdir()
    staged_result = staging / result_path.name
    write_json_exclusive(staged_result, result.model_dump())
    _rename_noreplace(staged_result, result_path)
    fsync_directory(output_dir)
    staging.rmdir()
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
