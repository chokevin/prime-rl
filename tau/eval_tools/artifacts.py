from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
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
    validate_file_manifest,
)

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)


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

    schema_version: Literal[1] = 1
    status: Literal["verified"] = "verified"
    created_at: str
    manifest_identity_hash: str
    model_files_sha256: str
    training_data_digest: str
    rl_config: RLConfigIdentity

    @field_validator("manifest_identity_hash", "model_files_sha256", "training_data_digest")
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("preflight digests must be lowercase SHA-256 hex")
        return digest

    @classmethod
    def load(cls, path: Path) -> "TrainingPreflight":
        payload, _ = load_json_with_sha256(path)
        return cls.model_validate(payload)


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    status: Literal["success"] = "success"
    created_at: str
    source_revision: str
    manifest_identity_hash: str
    model_name: str
    model_revision: str
    model_local_path: str
    model_files_sha256: str
    training_data_digest: str
    rl_config: RLConfigIdentity
    source_step: int = Field(gt=0)
    source_adapter_path: str
    final_adapter_path: str
    adapter_sha256: str
    adapter_files: FileManifest
    lora_rank: int = Field(gt=0)

    @field_validator("adapter_sha256", "model_files_sha256", "training_data_digest", "manifest_identity_hash")
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("artifact digests must be lowercase SHA-256 hex")
        return digest

    @field_validator("source_revision", "model_revision")
    @classmethod
    def validate_git_sha(cls, revision: str) -> str:
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("source and model revisions must be lowercase 40-character commit SHAs")
        return revision

    @field_validator("model_local_path", "source_adapter_path", "final_adapter_path")
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
    manifest: FrozenEvalManifest,
    config_identity: RLConfigIdentity,
) -> TrainingPreflight:
    expected = {
        "manifest_identity_hash": manifest.identity_hash(),
        "model_files_sha256": manifest.model.file_manifest.aggregate_sha256,
        "training_data_digest": manifest.training_data.record_digest,
        "rl_config": config_identity,
    }
    output_path = Path(output_path)
    if output_path.exists():
        existing = TrainingPreflight.load(output_path)
        for field, value in expected.items():
            if getattr(existing, field) != value:
                raise ValueError(f"existing training preflight has mismatched {field}")
        return existing
    result = TrainingPreflight(
        created_at=datetime.now(timezone.utc).isoformat(),
        **expected,
    )
    write_json_exclusive(output_path, result.model_dump())
    return result


def _remove_owned_staging(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.name != ".training-publication.stage":
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


def publish_training_result(
    *,
    output_dir: Path,
    manifest: FrozenEvalManifest,
    source_revision: str,
    expected_step: int,
    expected_rank: int,
) -> TrainingResult:
    output_dir = Path(output_dir)
    if manifest.source_revision != source_revision:
        raise ValueError(f"manifest source revision is {manifest.source_revision}, expected {source_revision}")
    result_path = output_dir / "training-result.json"
    final_adapter = output_dir / "final-adapter"
    staging = output_dir / ".training-publication.stage"
    if os.path.lexists(result_path):
        result = TrainingResult.load(result_path)
        validate_adapter_handoff(
            result=result,
            manifest=manifest,
            expected_adapter_path=final_adapter,
            expected_step=expected_step,
            expected_rank=expected_rank,
        )
        _remove_owned_staging(staging)
        return result
    _remove_owned_staging(staging)
    step, source_adapter, source_digest = select_final_adapter(
        output_dir / "weights",
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    source_manifest = build_file_manifest(source_adapter)
    if source_manifest.aggregate_sha256 != source_digest:
        raise RuntimeError("source adapter changed while it was being selected")

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
    result = TrainingResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=source_revision,
        manifest_identity_hash=manifest.identity_hash(),
        model_name=manifest.model.name,
        model_revision=manifest.model.revision,
        model_local_path=manifest.model.local_path,
        model_files_sha256=manifest.model.file_manifest.aggregate_sha256,
        training_data_digest=manifest.training_data.record_digest,
        rl_config=manifest.rl_config,
        source_step=step,
        source_adapter_path=str(source_adapter),
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
    expected_adapter_path: Path,
    expected_step: int,
    expected_rank: int,
) -> None:
    if result.manifest_identity_hash != manifest.identity_hash():
        raise ValueError("training result manifest identity does not match the frozen eval manifest")
    if result.source_revision != manifest.source_revision:
        raise ValueError("training result source revision does not match the frozen eval manifest")
    if result.model_name != manifest.model.name or result.model_revision != manifest.model.revision:
        raise ValueError("training result base model identity does not match the frozen eval manifest")
    if result.model_local_path != manifest.model.local_path:
        raise ValueError("training result model snapshot path does not match the frozen eval manifest")
    if result.model_files_sha256 != manifest.model.file_manifest.aggregate_sha256:
        raise ValueError("training result model file identity does not match the frozen eval manifest")
    if result.training_data_digest != manifest.training_data.record_digest:
        raise ValueError("training result training-data identity does not match the frozen eval manifest")
    if result.rl_config != manifest.rl_config:
        raise ValueError("training result effective RL config identity does not match the frozen eval manifest")
    if result.source_step != expected_step:
        raise ValueError(f"training result source step is {result.source_step}, expected {expected_step}")
    if result.lora_rank != expected_rank:
        raise ValueError(f"training result LoRA rank is {result.lora_rank}, expected {expected_rank}")
    expected_adapter_path = Path(expected_adapter_path)
    expected_source_path = expected_adapter_path.parent / "weights" / f"step_{result.source_step}" / "lora_adapters"
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
