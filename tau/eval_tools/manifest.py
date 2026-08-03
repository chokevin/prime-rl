"""The frozen eval manifest: the single artifact that pins `(model, taskset/version,
example IDs and hashes, seed, decoding parameters, N)` before training starts, and that
every later eval (baseline or post-training) must replay unmodified.

A draft is exclusively written before baseline; finalization exclusively writes a second
manifest that binds the baseline digest and resolved RL configuration. Baseline and post
bind the same stable evaluation identity, while post also binds the complete finalized
identity. The comparison utility re-validates both so an accidental copy/edit is rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tau.eval_tools.json_io import load_json_with_sha256, write_json_exclusive

SCHEMA_VERSION = 5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_EVAL_TASKSET_ID = "math500-v1"
EXPECTED_EVAL_DATASET = "HuggingFaceH4/MATH-500"
EXPECTED_EVAL_DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
EXPECTED_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_TRAIN_TASKSET_ID = "math-env-v1"
EXPECTED_TRAIN_DATASET = "PrimeIntellect/Hendrycks-Math"
EXPECTED_TRAIN_DATASET_REVISION = "3ed63f49541bdca4382fba28146aadf20d95cb38"
EXPECTED_TRAIN_DATASET_SUBSET = "default"
EXPECTED_TRAIN_DATA_FILE = "data/train-00000-of-00001.parquet"
EXPECTED_GRADER = "verifiers.v1.scoring.verify_boxed_math_answer"
EXPECTED_EVAL_N = 500
EXPECTED_DECODING = {
    "temperature": 0.0,
    "top_p": None,
    "max_completion_tokens": None,
    "seed": 0,
}
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin")

#: Below this, a baseline is considered too hard to show a defensible hill-climb and too
#: easy to rule out reward hacking / saturation. Mirrors the goal harness's `[0.10, 0.80]`
#: headroom band (see the parent plan's "Measured hill climb" done-check).
MIN_HEADROOM_MEAN = 0.10
MAX_HEADROOM_MEAN = 0.80


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        parsed = PurePosixPath(path)
        if not path or parsed.is_absolute() or "." in parsed.parts or ".." in parsed.parts:
            raise ValueError(f"file manifest path must be canonical and relative: {path!r}")
        if parsed.as_posix() != path:
            raise ValueError(f"file manifest path is not canonical POSIX form: {path!r}")
        return path

    @field_validator("size")
    @classmethod
    def validate_size(cls, size: int) -> int:
        if size < 0:
            raise ValueError("file size must be non-negative")
        return size

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("file SHA-256 must be a lowercase hex digest")
        return digest


def _file_manifest_digest(files: list[FileRecord]) -> str:
    payload = [record.model_dump() for record in files]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[FileRecord]
    aggregate_sha256: str

    @model_validator(mode="after")
    def validate_manifest(self):
        paths = [record.path for record in self.files]
        if not paths:
            raise ValueError("file manifest must contain at least one file")
        if paths != sorted(paths):
            raise ValueError("file manifest records must be sorted by canonical relative path")
        if len(paths) != len(set(paths)):
            raise ValueError("file manifest contains duplicate paths")
        expected = _file_manifest_digest(self.files)
        if self.aggregate_sha256 != expected:
            raise ValueError(
                f"file manifest aggregate digest {self.aggregate_sha256} does not match computed digest {expected}"
            )
        return self

    @classmethod
    def from_records(cls, records: list[FileRecord]) -> "FileManifest":
        ordered = sorted(records, key=lambda record: record.path)
        return cls(files=ordered, aggregate_sha256=_file_manifest_digest(ordered))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"path contains a symlink component: {current}")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise ValueError(f"path does not resolve canonically to itself: {absolute} -> {resolved}")
    return absolute


def _canonical_child(path: Path, parent: Path) -> Path:
    canonical_parent = _assert_no_symlink_components(parent)
    canonical_path = _assert_no_symlink_components(path)
    if canonical_path == canonical_parent or not canonical_path.is_relative_to(canonical_parent):
        raise ValueError(f"{canonical_path} must be a strict child of {canonical_parent}")
    return canonical_path


def build_file_manifest(root: Path) -> FileManifest:
    root = _assert_no_symlink_components(root)
    if not root.is_dir():
        raise FileNotFoundError(f"file manifest root is not a directory: {root}")

    records: list[FileRecord] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError(f"file manifest tree contains a symlink: {entry_path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(entry_path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"file manifest tree contains a non-regular entry: {entry_path}")
            relative = entry_path.relative_to(root).as_posix()
            records.append(
                FileRecord(
                    path=relative, size=entry.stat(follow_symlinks=False).st_size, sha256=_sha256_file(entry_path)
                )
            )
    return FileManifest.from_records(records)


def validate_file_manifest(root: Path, expected: FileManifest) -> None:
    actual = build_file_manifest(root)
    if actual != expected:
        expected_by_path = {record.path: record for record in expected.files}
        actual_by_path = {record.path: record for record in actual.files}
        missing = sorted(expected_by_path.keys() - actual_by_path.keys())
        extra = sorted(actual_by_path.keys() - expected_by_path.keys())
        changed = sorted(
            path
            for path in expected_by_path.keys() & actual_by_path.keys()
            if expected_by_path[path] != actual_by_path[path]
        )
        raise ValueError(
            "file manifest mismatch "
            f"(missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}, "
            f"actual_aggregate={actual.aggregate_sha256}, expected_aggregate={expected.aggregate_sha256})"
        )


def materialize_regular_snapshot(source: Path, cache_root: Path, destination: Path) -> FileManifest:
    cache_root = _assert_no_symlink_components(cache_root)
    source = _canonical_child(source, cache_root)
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination_parent = _canonical_child(destination_parent, cache_root)
    destination = Path(os.path.abspath(destination))
    if destination.parent != destination_parent:
        raise ValueError("trusted snapshot destination must be a direct child of its verified parent")
    staging = destination.with_name(f".{destination.name}.materializing")

    records: list[FileRecord] = []
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise ValueError(f"snapshot contains a symlinked directory: {directory_path / name}")
        for name in filenames:
            item = directory_path / name
            target = item.resolve(strict=True)
            if not target.is_relative_to(cache_root):
                raise ValueError(f"snapshot file target escapes the verified cache root: {item} -> {target}")
            mode = target.stat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"snapshot file target is not regular: {item} -> {target}")
            relative = item.relative_to(source).as_posix()
            records.append(FileRecord(path=relative, size=target.stat().st_size, sha256=_sha256_file(target)))
    source_manifest = FileManifest.from_records(records)

    if destination.exists():
        validate_file_manifest(destination, source_manifest)
        return source_manifest
    if os.path.lexists(staging):
        if staging.is_symlink() or staging.parent != destination_parent:
            raise ValueError(f"unsafe snapshot staging path exists: {staging}")
        raise FileExistsError(f"snapshot staging path already exists: {staging}")
    staging.mkdir()
    for record in source_manifest.files:
        source_item = (source / record.path).resolve(strict=True)
        destination_item = staging / record.path
        destination_item.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_item, destination_item)
    validate_file_manifest(staging, source_manifest)
    os.rename(staging, destination)
    return source_manifest


def make_tree_immutable(root: Path) -> None:
    root = _assert_no_symlink_components(root)
    if not root.is_dir():
        raise FileNotFoundError(f"immutable tree root is not a directory: {root}")
    directories = [root]
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"immutable tree contains a symlinked directory: {child}")
            directories.append(child)
        for name in filenames:
            child = directory_path / name
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"immutable tree contains a non-regular file: {child}")
            child.chmod(0o444)
    for directory in reversed(directories):
        directory.chmod(0o555)


def validate_tree_immutable(root: Path, expected: FileManifest) -> None:
    root = _assert_no_symlink_components(root)
    validate_file_manifest(root, expected)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.stat().st_mode & 0o222:
            raise ValueError(f"private materialization directory is writable: {directory_path}")
        for name in [*dirnames, *filenames]:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"private materialization contains a symlink: {child}")
            if child.stat().st_mode & 0o222:
                raise ValueError(f"private materialization entry is writable: {child}")


class ModelSnapshot(BaseModel):
    """The exact model artifact the eval must be reproducible against."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Pinned Hugging Face model id; serving uses only a verified private materialization."""

    revision: str
    """Exact resolved HF commit SHA for `name`."""

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, revision: str) -> str:
        if not GIT_SHA_RE.fullmatch(revision):
            raise ValueError("model revision must be an exact 40-character lowercase commit SHA")
        return revision

    file_manifest: FileManifest


class TasksetRef(BaseModel):
    """Identity of one verifiers taskset load, precise enough to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    taskset_revision: str
    dataset_name: str
    dataset_subset: str | None
    dataset_split: str
    dataset_revision: str

    @field_validator("taskset_revision", "dataset_revision")
    @classmethod
    def validate_revision(cls, revision: str) -> str:
        if not GIT_SHA_RE.fullmatch(revision):
            raise ValueError("taskset and dataset revisions must be exact 40-character lowercase commit SHAs")
        return revision


class DecodingConfig(BaseModel):
    """Sampling parameters the eval replay must reuse exactly."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    top_p: float | None = None
    max_completion_tokens: int | None = None
    seed: int


class ExampleRecord(BaseModel):
    """Identity anchor for one eval example — deliberately hash-only, not the raw
    prompt/answer text. `eval_taskset` (id + pinned `dataset_revision`) is enough to
    deterministically reload the exact same ordered examples from the source dataset;
    a live eval replay (`live/run_frozen_eval_live.py`) reloads the taskset and asserts
    the freshly-computed hash still matches this record before evaluating, which also
    catches unexpected upstream dataset drift under a nominally-pinned revision."""

    model_config = ConfigDict(extra="forbid")

    id: int
    prompt_hash: str
    answer_hash: str

    @field_validator("prompt_hash", "answer_hash")
    @classmethod
    def validate_hash(cls, digest: str) -> str:
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("example hashes must be lowercase SHA-256 hex digests")
        return digest


class TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    prompt_hash: str
    answer_hash: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, record_id: int) -> int:
        if record_id < 0:
            raise ValueError("training record id must be non-negative")
        return record_id

    @field_validator("prompt_hash", "answer_hash")
    @classmethod
    def validate_hash(cls, digest: str) -> str:
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("training record hashes must be lowercase SHA-256 hex digests")
        return digest


def _training_record_digest(records: list[TrainingRecord]) -> str:
    canonical = json.dumps(
        [record.model_dump() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrainingDataIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    records: list[TrainingRecord]
    record_digest: str
    file_manifest: FileManifest

    @model_validator(mode="after")
    def validate_identity(self):
        if self.n != len(self.records):
            raise ValueError(f"training-data n={self.n} does not match len(records)={len(self.records)}")
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("training data contains duplicate record ids")
        prompt_hashes = [record.prompt_hash for record in self.records]
        if len(prompt_hashes) != len(set(prompt_hashes)):
            raise ValueError("training data contains duplicate prompt hashes")
        expected_digest = _training_record_digest(self.records)
        if self.record_digest != expected_digest:
            raise ValueError(
                f"training record digest {self.record_digest} does not match computed digest {expected_digest}"
            )
        return self

    @property
    def prompt_hashes(self) -> list[str]:
        return [record.prompt_hash for record in self.records]

    @classmethod
    def from_records(cls, records: list[TrainingRecord], *, file_manifest: FileManifest) -> "TrainingDataIdentity":
        return cls(
            n=len(records),
            records=records,
            record_digest=_training_record_digest(records),
            file_manifest=file_manifest,
        )


class RLConfigIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[1] = 1
    source_config_rel: Literal["configs/tau/math-7b-h200/train.toml"]
    source_toml_sha256: str
    output_dir: str
    max_steps: int
    canonical_resolved_sha256: str

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, output_dir: str) -> str:
        if not Path(output_dir).is_absolute():
            raise ValueError("RL output directory must be absolute")
        return output_dir

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, max_steps: int) -> int:
        if max_steps < 1:
            raise ValueError("RL max_steps must be positive")
        return max_steps

    @field_validator("source_toml_sha256", "canonical_resolved_sha256")
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("RL config digests must be lowercase SHA-256 hex")
        return digest


class FrozenEvalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    state: Literal["draft", "finalized"]
    created_at: str
    """UTC ISO-8601 timestamp, informational only — excluded from `identity_hash()`."""

    source_revision: str
    verifiers_revision: str
    model: ModelSnapshot

    eval_taskset: TasksetRef
    train_taskset: TasksetRef
    training_data: TrainingDataIdentity
    examples: list[ExampleRecord]
    decoding: DecodingConfig
    grader: str
    """Fully-qualified name of the scoring function, e.g.
    ``verifiers.v1.scoring.verify_boxed_math_answer``."""

    n: int
    baseline_mean: float | None
    baseline_mean_headroom_ok: bool
    baseline_rewards_sha256: str | None = None
    rl_config: RLConfigIdentity | None = None
    """Whether the baseline mean reward (measured once, before freezing) fell inside
    `[MIN_HEADROOM_MEAN, MAX_HEADROOM_MEAN]`. `freeze_manifest_live.py` refuses to write
    a manifest with this False; kept as an explicit field so a manifest on disk is
    self-documenting about why it was (or wasn't) accepted."""

    @model_validator(mode="after")
    def validate_manifest(self):
        for name, revision in (
            ("source_revision", self.source_revision),
            ("verifiers_revision", self.verifiers_revision),
        ):
            if not GIT_SHA_RE.fullmatch(revision):
                raise ValueError(f"{name} must be an exact 40-character lowercase commit SHA")
        if self.n != len(self.examples):
            raise ValueError(f"n={self.n} does not match len(examples)={len(self.examples)}")
        if len(self.examples) < 200:
            raise ValueError(
                f"frozen eval manifest must cover >= 200 examples, got {len(self.examples)} "
                "(see the goal harness's 'Measured hill climb' done-check)"
            )
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate example ids in frozen eval manifest: {duplicates[:5]}")
        prompt_hashes = [example.prompt_hash for example in self.examples]
        if len(prompt_hashes) != len(set(prompt_hashes)):
            raise ValueError("duplicate eval prompt hashes in frozen eval manifest")
        if self.state == "draft":
            if (
                self.baseline_mean is not None
                or self.baseline_mean_headroom_ok
                or self.baseline_rewards_sha256 is not None
                or self.rl_config is not None
            ):
                raise ValueError("draft manifest must not claim finalized baseline headroom")
        elif self.baseline_mean is None or not check_headroom(self.baseline_mean):
            raise ValueError("finalized manifest must record a baseline mean inside the accepted headroom band")
        elif not self.baseline_mean_headroom_ok:
            raise ValueError("finalized manifest must record baseline_mean_headroom_ok=true")
        elif self.baseline_rewards_sha256 is None or not SHA256_RE.fullmatch(self.baseline_rewards_sha256):
            raise ValueError("finalized manifest must record the baseline rewards artifact SHA-256")
        elif self.rl_config is None:
            raise ValueError("finalized manifest must record the effective RL config identity")
        return self

    @property
    def eval_prompt_hashes(self) -> set[str]:
        return {e.prompt_hash for e in self.examples}

    @property
    def example_ids(self) -> set[int]:
        return {e.id for e in self.examples}

    def _evaluation_identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "verifiers_revision": self.verifiers_revision,
            "model": self.model.model_dump(),
            "eval_taskset": self.eval_taskset.model_dump(),
            "train_taskset": self.train_taskset.model_dump(),
            "training_data": self.training_data.model_dump(),
            "examples": [e.model_dump() for e in self.examples],
            "decoding": self.decoding.model_dump(),
            "grader": self.grader,
            "n": self.n,
        }

    def evaluation_identity_hash(self) -> str:
        """Identity available before baseline evaluation and stable through finalization."""
        canonical = json.dumps(self._evaluation_identity_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def identity_hash(self) -> str:
        """Complete immutable finalized identity, including baseline evidence and RL config."""
        payload = {
            **self._evaluation_identity_payload(),
            "state": self.state,
            "baseline_mean": self.baseline_mean,
            "baseline_mean_headroom_ok": self.baseline_mean_headroom_ok,
            "baseline_rewards_sha256": self.baseline_rewards_sha256,
            "rl_config": self.rl_config.model_dump() if self.rl_config is not None else None,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"{path} already exists — the frozen eval manifest is immutable once "
                "written. Use a new experiment output path to re-freeze."
            )
        write_json_exclusive(path, self.model_dump())

    @classmethod
    def load(cls, path: Path) -> "FrozenEvalManifest":
        payload, _ = load_json_with_sha256(path)
        return cls.model_validate(payload)


class LeakageError(ValueError):
    """Raised when a training prompt hash also appears in the frozen eval set."""


class HeadroomError(ValueError):
    """Raised when the measured baseline mean reward falls outside the accepted
    headroom band (too saturated or too hard to show a defensible hill-climb)."""


def check_disjoint(eval_hashes: set[str], train_hashes: set[str]) -> None:
    """Prove zero overlap between eval and training prompts. Different dataset names
    alone are not accepted as proof (per the W2 selection memo) — this checks actual
    content hashes."""
    overlap = eval_hashes & train_hashes
    if overlap:
        sample = sorted(overlap)[:3]
        raise LeakageError(
            f"{len(overlap)} eval prompt hash(es) also appear in the training prompt "
            f"set (e.g. {sample}); refusing to freeze a leaking manifest."
        )


def check_headroom(baseline_mean: float) -> bool:
    """Returns whether `baseline_mean` is inside the accepted headroom band. Does not
    raise — callers decide whether to abort or record the (documented) failure, since
    an out-of-band baseline is itself a valid, reportable outcome (choose a different
    tuple *before* training; never after)."""
    return MIN_HEADROOM_MEAN <= baseline_mean <= MAX_HEADROOM_MEAN


def build_manifest(
    *,
    state: Literal["draft", "finalized"],
    source_revision: str,
    verifiers_revision: str,
    model: ModelSnapshot,
    eval_taskset: TasksetRef,
    train_taskset: TasksetRef,
    training_data: TrainingDataIdentity,
    examples: list[ExampleRecord],
    decoding: DecodingConfig,
    grader: str,
    baseline_mean: float | None,
    baseline_rewards_sha256: str | None,
    rl_config: RLConfigIdentity | None,
    created_at: str,
) -> FrozenEvalManifest:
    return FrozenEvalManifest(
        state=state,
        created_at=created_at,
        source_revision=source_revision,
        verifiers_revision=verifiers_revision,
        model=model,
        eval_taskset=eval_taskset,
        train_taskset=train_taskset,
        training_data=training_data,
        examples=examples,
        decoding=decoding,
        grader=grader,
        n=len(examples),
        baseline_mean=baseline_mean,
        baseline_mean_headroom_ok=baseline_mean is not None and check_headroom(baseline_mean),
        baseline_rewards_sha256=baseline_rewards_sha256,
        rl_config=rl_config,
    )


def validate_model_materialization(model: ModelSnapshot, path: Path, run_root: Path) -> Path:
    run_root = _assert_no_symlink_components(run_root)
    path = _canonical_child(path, run_root)
    if path != run_root / "model":
        raise ValueError(f"private model path is {path}, expected {run_root / 'model'}")
    if not path.is_dir():
        raise FileNotFoundError(f"private model snapshot directory not found: {path}")
    manifest_paths = {record.path for record in model.file_manifest.files}
    if "config.json" not in manifest_paths:
        raise ValueError("trusted model file manifest is missing config.json")
    if not any(path.endswith(MODEL_WEIGHT_SUFFIXES) for path in manifest_paths):
        raise ValueError("trusted model file manifest contains no model weight file")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"materialized model snapshot is missing config.json: {path}")
    validate_file_manifest(path, model.file_manifest)
    return path


def validate_training_materialization(manifest: FrozenEvalManifest, path: Path, run_root: Path) -> Path:
    run_root = _assert_no_symlink_components(run_root)
    path = _canonical_child(path, run_root)
    if path != run_root / "training-dataset":
        raise ValueError(f"private training dataset path is {path}, expected {run_root / 'training-dataset'}")
    if not path.is_dir():
        raise FileNotFoundError(f"private training dataset snapshot directory not found: {path}")
    data_file = path / EXPECTED_TRAIN_DATA_FILE
    if not data_file.is_file():
        raise FileNotFoundError(f"materialized training dataset is missing pinned data file: {data_file}")
    _assert_no_symlink_components(data_file)
    validate_file_manifest(path, manifest.training_data.file_manifest)
    return path


def validate_training_prompt_hashes(
    manifest: FrozenEvalManifest,
    actual_records: list[TrainingRecord],
) -> None:
    actual = TrainingDataIdentity.from_records(
        actual_records,
        file_manifest=manifest.training_data.file_manifest,
    )
    if actual != manifest.training_data:
        raise ValueError(
            "actual math-env-v1 ordered training records do not match the frozen training-data identity "
            f"(actual={actual.record_digest}, expected={manifest.training_data.record_digest})"
        )
    check_disjoint(manifest.eval_prompt_hashes, set(actual.prompt_hashes))


def validate_manifest_contract(
    manifest: FrozenEvalManifest,
    *,
    expected_source_revision: str,
    expected_verifiers_revision: str,
    expected_tasksets_revision: str,
    expected_model_name: str,
    expected_model_revision: str,
    require_finalized: bool,
) -> None:
    if require_finalized and manifest.state != "finalized":
        raise ValueError(f"manifest state is {manifest.state!r}, expected 'finalized'")
    if manifest.source_revision != expected_source_revision:
        raise ValueError(f"manifest source revision is {manifest.source_revision}, expected {expected_source_revision}")
    if manifest.verifiers_revision != expected_verifiers_revision:
        raise ValueError(
            f"manifest verifiers revision is {manifest.verifiers_revision}, expected {expected_verifiers_revision}"
        )
    if expected_model_name != EXPECTED_MODEL_NAME or expected_model_revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            "requested base model does not match the frozen experiment contract "
            f"{EXPECTED_MODEL_NAME}@{EXPECTED_MODEL_REVISION}"
        )
    if manifest.model.name != expected_model_name or manifest.model.revision != expected_model_revision:
        raise ValueError(
            "manifest model identity does not match the expected base model "
            f"{expected_model_name}@{expected_model_revision}"
        )
    expected_eval_ref = TasksetRef(
        id=EXPECTED_EVAL_TASKSET_ID,
        taskset_revision=expected_tasksets_revision,
        dataset_name=EXPECTED_EVAL_DATASET,
        dataset_subset=None,
        dataset_split="test",
        dataset_revision=EXPECTED_EVAL_DATASET_REVISION,
    )
    if manifest.eval_taskset != expected_eval_ref:
        raise ValueError("manifest eval taskset identity does not match the pinned math500-v1 contract")
    if (
        manifest.train_taskset.id != EXPECTED_TRAIN_TASKSET_ID
        or manifest.train_taskset.taskset_revision != expected_tasksets_revision
        or manifest.train_taskset.dataset_name != EXPECTED_TRAIN_DATASET
        or manifest.train_taskset.dataset_subset != EXPECTED_TRAIN_DATASET_SUBSET
        or manifest.train_taskset.dataset_split != "train"
        or manifest.train_taskset.dataset_revision != EXPECTED_TRAIN_DATASET_REVISION
    ):
        raise ValueError("manifest training taskset identity does not match the pinned math-env-v1 contract")
    if manifest.n != EXPECTED_EVAL_N or manifest.example_ids != set(range(EXPECTED_EVAL_N)):
        raise ValueError(f"manifest must contain all {EXPECTED_EVAL_N} math500-v1 ids exactly once")
    if manifest.decoding.model_dump() != EXPECTED_DECODING:
        raise ValueError(f"manifest decoding tuple is {manifest.decoding.model_dump()}, expected {EXPECTED_DECODING}")
    if manifest.grader != EXPECTED_GRADER:
        raise ValueError(f"manifest grader is {manifest.grader!r}, expected {EXPECTED_GRADER!r}")
    validate_training_prompt_hashes(manifest, manifest.training_data.records)
