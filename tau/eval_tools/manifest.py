"""The frozen eval manifest: the single artifact that pins `(model, taskset/version,
example IDs and hashes, seed, decoding parameters, N)` before training starts, and that
every later eval (baseline or post-training) must replay unmodified.

Only one manifest is ever written per experiment (immutable once frozen — see
`FrozenEvalManifest.save`'s ``force`` guard in the CLI). Both the baseline-eval and
post-eval Tau jobs read the *same* manifest file; they never build their own, so
"identical example set" is true by construction rather than by re-derivation. The
comparison utility (`compare.py`) still re-validates this via `identity_hash()` so an
accidental copy/edit of the manifest is caught rather than silently accepted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, model_validator

SCHEMA_VERSION = 1

#: Below this, a baseline is considered too hard to show a defensible hill-climb and too
#: easy to rule out reward hacking / saturation. Mirrors the goal harness's `[0.10, 0.80]`
#: headroom band (see the parent plan's "Measured hill climb" done-check).
MIN_HEADROOM_MEAN = 0.10
MAX_HEADROOM_MEAN = 0.80


class ModelSnapshot(BaseModel):
    """The exact model artifact the eval must be reproducible against."""

    name: str
    """HF model id or local path passed as the inference server's ``model.name``."""

    revision: str
    """Exact resolved HF commit SHA for `name` (via `huggingface_hub.model_info(...).sha`
    at freeze time), or the literal string ``"local"`` when `local_path` is a
    pre-materialized, already-pinned snapshot directory."""

    local_path: str | None = None
    """Local filesystem path to the pinned snapshot, when one was pre-downloaded so the
    exact revision above is guaranteed served (prime-rl's model config takes a name or a
    local path, not a revision — see the W2 selection memo's caveat)."""


class TasksetRef(BaseModel):
    """Identity of one verifiers taskset load, precise enough to reproduce it."""

    id: str
    dataset_name: str
    dataset_split: str
    dataset_revision: str | None = None
    """None only for `math-env-v1` (the training taskset), which does not pin a
    revision upstream; the manifest still records the hash of every prompt it loaded at
    freeze time as the leakage-check evidence of record."""


class DecodingConfig(BaseModel):
    """Sampling parameters the eval replay must reuse exactly."""

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

    id: int
    prompt_hash: str
    answer_hash: str


class FrozenEvalManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    created_at: str
    """UTC ISO-8601 timestamp, informational only — excluded from `identity_hash()`."""

    model: ModelSnapshot
    """The *baseline* model snapshot. Post-training eval serves this same base model
    plus a LoRA adapter, which is intentionally not modeled here — `identity_hash()`
    covers only what must stay identical across baseline and post (taskset, examples,
    decoding, grader), not the model, which is expected to differ."""

    eval_taskset: TasksetRef
    train_taskset: TasksetRef
    examples: list[ExampleRecord]
    decoding: DecodingConfig
    grader: str
    """Fully-qualified name of the scoring function, e.g.
    ``verifiers.v1.scoring.verify_boxed_math_answer``."""

    n: int
    baseline_mean_headroom_ok: bool
    """Whether the baseline mean reward (measured once, before freezing) fell inside
    `[MIN_HEADROOM_MEAN, MAX_HEADROOM_MEAN]`. `freeze_manifest_live.py` refuses to write
    a manifest with this False; kept as an explicit field so a manifest on disk is
    self-documenting about why it was (or wasn't) accepted."""

    @model_validator(mode="after")
    def validate_examples(self):
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
        Excludes `created_at` (timestamps) and `model` (expected to differ between
        baseline and post). Two manifests with the same `identity_hash()` are
        interchangeable for the paired comparison in `compare.py`."""
        payload = {
            "schema_version": self.schema_version,
            "eval_taskset": self.eval_taskset.model_dump(),
            "train_taskset": self.train_taskset.model_dump(),
            "examples": [e.model_dump() for e in self.examples],
            "decoding": self.decoding.model_dump(),
            "grader": self.grader,
            "n": self.n,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path, *, force: bool = False) -> None:
        path = Path(path)
        if path.exists() and not force:
            raise FileExistsError(
                f"{path} already exists — the frozen eval manifest is immutable once "
                "written. Pass force=True only if you are intentionally re-freezing a "
                "new experiment (this invalidates any prior baseline/post eval results)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "FrozenEvalManifest":
        return cls.model_validate_json(Path(path).read_text())


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
    model: ModelSnapshot,
    eval_taskset: TasksetRef,
    train_taskset: TasksetRef,
    examples: list[ExampleRecord],
    decoding: DecodingConfig,
    grader: str,
    baseline_mean: float,
    created_at: str,
) -> FrozenEvalManifest:
    return FrozenEvalManifest(
        created_at=created_at,
        model=model,
        eval_taskset=eval_taskset,
        train_taskset=train_taskset,
        examples=examples,
        decoding=decoding,
        grader=grader,
        n=len(examples),
        baseline_mean_headroom_ok=check_headroom(baseline_mean),
    )
