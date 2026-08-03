"""The frozen eval manifest: the single artifact that pins `(model, taskset/version,
example IDs and hashes, seed, decoding parameters, N)` before training starts, and that
every later eval (baseline or post-training) must replay unmodified.

Only one manifest is ever written per experiment (immutable and exclusively created).
Both the baseline-eval and
post-eval Tau jobs read the *same* manifest file; they never build their own, so
"identical example set" is true by construction rather than by re-derivation. The
comparison utility (`compare.py`) still re-validates this via `identity_hash()` so an
accidental copy/edit of the manifest is caught rather than silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tau.eval_tools.hashing import hash_sequence
from tau.eval_tools.json_io import load_json, write_json_exclusive

SCHEMA_VERSION = 3
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

#: Below this, a baseline is considered too hard to show a defensible hill-climb and too
#: easy to rule out reward hacking / saturation. Mirrors the goal harness's `[0.10, 0.80]`
#: headroom band (see the parent plan's "Measured hill climb" done-check).
MIN_HEADROOM_MEAN = 0.10
MAX_HEADROOM_MEAN = 0.80


class ModelSnapshot(BaseModel):
    """The exact model artifact the eval must be reproducible against."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """HF model id or local path passed as the inference server's ``model.name``."""

    revision: str
    """Exact resolved HF commit SHA for `name`."""

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, revision: str) -> str:
        if not GIT_SHA_RE.fullmatch(revision):
            raise ValueError("model revision must be an exact 40-character lowercase commit SHA")
        return revision

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, local_path: str) -> str:
        if not Path(local_path).is_absolute():
            raise ValueError("model local_path must be absolute")
        return local_path

    local_path: str
    """Local filesystem path to the pinned snapshot, when one was pre-downloaded so the
    exact revision above is guaranteed served (prime-rl's model config takes a name or a
    local path, not a revision — see the W2 selection memo's caveat)."""


class TasksetRef(BaseModel):
    """Identity of one verifiers taskset load, precise enough to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    taskset_revision: str
    dataset_name: str
    dataset_subset: str | None
    dataset_split: str
    dataset_revision: str
    dataset_local_path: str | None = None

    @field_validator("taskset_revision", "dataset_revision")
    @classmethod
    def validate_revision(cls, revision: str) -> str:
        if not GIT_SHA_RE.fullmatch(revision):
            raise ValueError("taskset and dataset revisions must be exact 40-character lowercase commit SHAs")
        return revision

    @field_validator("dataset_local_path")
    @classmethod
    def validate_dataset_local_path(cls, local_path: str | None) -> str | None:
        if local_path is not None and not Path(local_path).is_absolute():
            raise ValueError("dataset local path must be absolute")
        return local_path


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


class TrainingDataIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    prompt_hashes: list[str]
    prompt_hash_digest: str

    @field_validator("prompt_hashes")
    @classmethod
    def validate_prompt_hashes(cls, prompt_hashes: list[str]) -> list[str]:
        invalid = [digest for digest in prompt_hashes if not SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError("training prompt hashes must be lowercase SHA-256 hex digests")
        return prompt_hashes

    @model_validator(mode="after")
    def validate_identity(self):
        if self.n != len(self.prompt_hashes):
            raise ValueError(f"training-data n={self.n} does not match len(prompt_hashes)={len(self.prompt_hashes)}")
        expected_digest = hash_sequence(self.prompt_hashes)
        if self.prompt_hash_digest != expected_digest:
            raise ValueError(
                f"training prompt hash digest {self.prompt_hash_digest} does not match computed digest {expected_digest}"
            )
        return self

    @classmethod
    def from_prompt_hashes(cls, prompt_hashes: list[str]) -> "TrainingDataIdentity":
        return cls(
            n=len(prompt_hashes),
            prompt_hashes=prompt_hashes,
            prompt_hash_digest=hash_sequence(prompt_hashes),
        )


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
            if self.baseline_mean is not None or self.baseline_mean_headroom_ok:
                raise ValueError("draft manifest must not claim finalized baseline headroom")
        elif self.baseline_mean is None or not check_headroom(self.baseline_mean):
            raise ValueError("finalized manifest must record a baseline mean inside the accepted headroom band")
        elif not self.baseline_mean_headroom_ok:
            raise ValueError("finalized manifest must record baseline_mean_headroom_ok=true")
        return self

    @property
    def eval_prompt_hashes(self) -> set[str]:
        return {e.prompt_hash for e in self.examples}

    @property
    def example_ids(self) -> set[int]:
        return {e.id for e in self.examples}

    def identity_hash(self) -> str:
        """Deterministic fingerprint of everything a replay eval must match exactly:
        taskset identity, the ordered example set, decoding settings, grader, and N.
        Excludes only finalization metadata (`state`, `created_at`, `baseline_mean`, and
        `baseline_mean_headroom_ok`) so the baseline reward file produced from the draft
        remains valid after finalization. The immutable base model, source revisions, and
        training-data identity are included."""
        payload = {
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
        return cls.model_validate(load_json(path))


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
    )


def validate_model_snapshot(model: ModelSnapshot) -> Path:
    path = Path(model.local_path)
    if not path.is_dir():
        raise FileNotFoundError(f"materialized model snapshot directory not found: {path}")
    if path.name != model.revision:
        raise ValueError(
            f"materialized model snapshot path {path} does not end in the pinned revision {model.revision}"
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"materialized model snapshot is missing config.json: {path}")
    return path


def validate_training_snapshot(taskset: TasksetRef) -> Path:
    if taskset.dataset_local_path is None:
        raise ValueError("training taskset must record its materialized dataset snapshot path")
    path = Path(taskset.dataset_local_path)
    if not path.is_dir():
        raise FileNotFoundError(f"materialized training dataset snapshot directory not found: {path}")
    if path.name != taskset.dataset_revision:
        raise ValueError(
            f"materialized training dataset path {path} does not end in the pinned revision {taskset.dataset_revision}"
        )
    data_file = path / EXPECTED_TRAIN_DATA_FILE
    if not data_file.is_file():
        raise FileNotFoundError(f"materialized training dataset is missing pinned data file: {data_file}")
    return path


def validate_training_prompt_hashes(
    manifest: FrozenEvalManifest,
    actual_prompt_hashes: list[str],
) -> None:
    actual = TrainingDataIdentity.from_prompt_hashes(actual_prompt_hashes)
    if actual != manifest.training_data:
        raise ValueError(
            "actual math-env-v1 training prompts do not match the frozen training-data identity "
            f"(actual={actual.prompt_hash_digest}, expected={manifest.training_data.prompt_hash_digest})"
        )
    check_disjoint(manifest.eval_prompt_hashes, set(actual_prompt_hashes))


def validate_manifest_contract(
    manifest: FrozenEvalManifest,
    *,
    expected_source_revision: str,
    expected_verifiers_revision: str,
    expected_tasksets_revision: str,
    expected_model_name: str,
    expected_model_revision: str,
    require_finalized: bool,
) -> Path:
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
        dataset_local_path=None,
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
    validate_training_prompt_hashes(manifest, manifest.training_data.prompt_hashes)
    validate_training_snapshot(manifest.train_taskset)
    return validate_model_snapshot(manifest.model)
