from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau.eval_tools.json_io import load_json, write_json_exclusive
from tau.eval_tools.manifest import FrozenEvalManifest

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)


class SmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    status: Literal["success"] = "success"
    created_at: str
    source_revision: str
    config_path: str
    resolved_configs: list[str]


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    status: Literal["success"] = "success"
    created_at: str
    source_revision: str
    manifest_identity_hash: str
    model_name: str
    model_revision: str
    model_local_path: str
    training_data_digest: str
    source_step: int = Field(gt=0)
    source_adapter_path: str
    final_adapter_path: str
    adapter_sha256: str
    lora_rank: int = Field(gt=0)

    @field_validator("adapter_sha256", "training_data_digest", "manifest_identity_hash")
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

    @classmethod
    def load(cls, path: Path) -> "TrainingResult":
        return cls.model_validate(load_json(path))


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
    config_path = path / ADAPTER_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter config not found: {config_path}")
    config = load_json(config_path)
    if not isinstance(config, dict) or config.get("r") != expected_rank:
        actual_rank = config.get("r") if isinstance(config, dict) else None
        raise ValueError(f"adapter rank is {actual_rank!r}, expected {expected_rank}")
    weight_files = [path / name for name in ADAPTER_WEIGHT_FILES if (path / name).is_file()]
    if len(weight_files) != 1:
        raise ValueError(
            f"adapter must contain exactly one supported weight file, found {[item.name for item in weight_files]}"
        )
    return sha256_files(path, [config_path, weight_files[0]])


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
    if not (step_dir / "STABLE").is_file():
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
    if result_path.exists():
        raise FileExistsError(f"{result_path} already exists; training evidence is immutable")
    if final_adapter.exists():
        raise FileExistsError(f"{final_adapter} already exists; adapter handoff is immutable")
    step, source_adapter, source_digest = select_final_adapter(
        output_dir / "weights",
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    copy_adapter_exclusive(source_adapter, final_adapter)
    published_digest = validate_adapter_directory(final_adapter, expected_rank=expected_rank)
    if published_digest != source_digest:
        raise RuntimeError(f"published adapter digest {published_digest} does not match source digest {source_digest}")
    result = TrainingResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=source_revision,
        manifest_identity_hash=manifest.identity_hash(),
        model_name=manifest.model.name,
        model_revision=manifest.model.revision,
        model_local_path=manifest.model.local_path,
        training_data_digest=manifest.training_data.prompt_hash_digest,
        source_step=step,
        source_adapter_path=str(source_adapter),
        final_adapter_path=str(final_adapter),
        adapter_sha256=published_digest,
        lora_rank=expected_rank,
    )
    write_json_exclusive(result_path, result.model_dump())
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
    if result.training_data_digest != manifest.training_data.prompt_hash_digest:
        raise ValueError("training result training-data identity does not match the frozen eval manifest")
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
