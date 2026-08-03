"""Paired baseline/post comparison with a deterministic bootstrap confidence interval,
and the pass/fail gate the goal harness requires: mean reward delta >= +0.03 absolute
AND a paired-bootstrap 95% CI lower bound > 0. Aggregate training-log trends never
substitute for this — it is computed only from the two frozen per-example reward files.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from tau.eval_tools.json_io import load_json_with_sha256, write_json_exclusive
from tau.eval_tools.manifest import SHA256_RE, FrozenEvalManifest

#: The goal harness's fixed acceptance thresholds (see the parent plan's "Measured hill
#: climb" done-check and decision ledger's "Metric default"). Not configurable via CLI
#: flags on purpose — relaxing them to fit a budget is explicitly forbidden.
MIN_DELTA = 0.03
CI_ALPHA = 0.05
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


class IdentityMismatchError(ValueError):
    """Raised when the two reward files don't share the same frozen manifest identity
    (different example sets, different decoding/grader settings, or a reward file that
    doesn't match the manifest passed on the CLI)."""


class RewardRecord(BaseModel):
    """One `<label>-rewards.json` file written by `live/run_frozen_eval_live.py`."""

    model_config = ConfigDict(extra="forbid")

    evaluation_identity_hash: str
    frozen_manifest_identity_hash: str | None = None
    model_label: Literal["baseline", "post"]
    created_at: str
    rewards: dict[str, float]
    """Example id (as string, for JSON-object-key portability) -> binary reward."""

    @field_validator("evaluation_identity_hash")
    @classmethod
    def validate_manifest_identity_hash(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("evaluation identity must be a lowercase SHA-256 hex digest")
        return digest

    @field_validator("frozen_manifest_identity_hash")
    @classmethod
    def validate_frozen_manifest_identity_hash(cls, digest: str | None) -> str | None:
        if digest is not None and not SHA256_RE.fullmatch(digest):
            raise ValueError("frozen manifest identity must be a lowercase SHA-256 hex digest")
        return digest

    @field_validator("rewards", mode="before")
    @classmethod
    def validate_rewards(cls, rewards):
        if not isinstance(rewards, dict):
            raise ValueError("rewards must be a JSON object keyed by example id")
        for example_id, reward in rewards.items():
            if not isinstance(example_id, str) or not example_id.isdigit() or str(int(example_id)) != example_id:
                raise ValueError(f"reward example id {example_id!r} is not a canonical non-negative integer")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise ValueError(f"reward for example {example_id!r} must be numeric")
            if not math.isfinite(reward) or float(reward) not in (0.0, 1.0):
                raise ValueError(
                    f"reward for example {example_id!r} is {reward!r}; "
                    "the pinned math grader domain is exactly {0.0, 1.0}"
                )
        return rewards

    @classmethod
    def load(cls, path: Path) -> "RewardRecord":
        payload, _ = load_json_with_sha256(path)
        return cls.model_validate(payload)

    @classmethod
    def load_with_digest(cls, path: Path) -> tuple["RewardRecord", str]:
        payload, digest = load_json_with_sha256(path)
        return cls.model_validate(payload), digest


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    baseline_mean: float
    post_mean: float
    delta: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: Literal[10_000]
    bootstrap_seed: Literal[0]
    ci_alpha: Literal[0.05]
    min_delta: Literal[0.03]
    manifest_identity_hash: str
    baseline_rewards_sha256: str
    post_rewards_sha256: str
    passed: bool

    def report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] n={self.n} baseline_mean={self.baseline_mean:.4f} "
            f"post_mean={self.post_mean:.4f} delta={self.delta:+.4f} "
            f"95% CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] "
            f"(gate: delta>={MIN_DELTA:+.2f} and ci_lower>0)"
        )


def paired_bootstrap_ci(
    deltas: np.ndarray,
) -> tuple[float, float]:
    """Deterministic (seeded) percentile-bootstrap CI for the mean of `deltas` (one
    per-example post-minus-baseline reward difference). Resampling with a fixed seed
    and `numpy`'s legacy `RandomState` keeps this reproducible across machines/numpy
    versions for the same input, which the goal harness's replay requirement needs."""
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("deltas must be a non-empty 1-D array")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = deltas.size
    resample_idx = rng.randint(0, n, size=(N_BOOTSTRAP, n))
    resampled_means = deltas[resample_idx].mean(axis=1)
    lower = float(np.percentile(resampled_means, 100 * (CI_ALPHA / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - CI_ALPHA / 2)))
    return lower, upper


def compare_runs(
    manifest: FrozenEvalManifest,
    baseline: RewardRecord,
    post: RewardRecord,
    *,
    baseline_sha256: str,
    post_sha256: str,
) -> ComparisonResult:
    """Validate manifest/example identity between the two reward files, then compute
    the paired delta, bootstrap CI, and pass/fail gate."""
    if manifest.state != "finalized":
        raise IdentityMismatchError("comparison requires a finalized manifest")
    identity = manifest.identity_hash()
    evaluation_identity = manifest.evaluation_identity_hash()
    for label, record in (("baseline", baseline), ("post", post)):
        if record.model_label != label:
            raise IdentityMismatchError(
                f"{label} input carries model_label={record.model_label!r}; refusing a swapped or mislabeled comparison"
            )
        if record.evaluation_identity_hash != evaluation_identity:
            raise IdentityMismatchError(
                f"{label} reward file's evaluation_identity_hash "
                f"({record.evaluation_identity_hash}) does not match the frozen "
                f"manifest's ({evaluation_identity}) — refusing to compare rewards computed "
                "against a different example set, decoding config, or grader."
            )
    if baseline.frozen_manifest_identity_hash is not None:
        raise IdentityMismatchError(
            "baseline reward evidence must predate finalization and omit frozen manifest identity"
        )
    if post.frozen_manifest_identity_hash != identity:
        raise IdentityMismatchError("post reward evidence does not bind the finalized manifest identity")
    if baseline_sha256 != manifest.baseline_rewards_sha256:
        raise IdentityMismatchError(
            f"baseline rewards SHA-256 {baseline_sha256} does not match finalized manifest "
            f"digest {manifest.baseline_rewards_sha256}"
        )
    for label, digest in (("baseline", baseline_sha256), ("post", post_sha256)):
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{label} rewards digest must be a lowercase SHA-256 hex digest")

    manifest_ids = {str(i) for i in manifest.example_ids}
    for label, record in (("baseline", baseline), ("post", post)):
        record_ids = set(record.rewards)
        if record_ids != manifest_ids:
            missing = manifest_ids - record_ids
            extra = record_ids - manifest_ids
            raise IdentityMismatchError(
                f"{label} reward file's example ids do not match the frozen manifest "
                f"exactly (missing={len(missing)}, extra={len(extra)})."
            )

    ordered_ids = sorted(manifest_ids, key=int)
    baseline_values = np.array([baseline.rewards[i] for i in ordered_ids], dtype=float)
    post_values = np.array([post.rewards[i] for i in ordered_ids], dtype=float)
    deltas = post_values - baseline_values

    delta_mean = float(deltas.mean())
    ci_lower, ci_upper = paired_bootstrap_ci(deltas)
    passed = bool(delta_mean >= MIN_DELTA and ci_lower > 0)

    return ComparisonResult(
        n=len(ordered_ids),
        baseline_mean=float(baseline_values.mean()),
        post_mean=float(post_values.mean()),
        delta=delta_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
        ci_alpha=CI_ALPHA,
        min_delta=MIN_DELTA,
        manifest_identity_hash=identity,
        baseline_rewards_sha256=baseline_sha256,
        post_rewards_sha256=post_sha256,
        passed=passed,
    )


def compare_from_paths(
    manifest_path: Path,
    baseline_path: Path,
    post_path: Path,
) -> ComparisonResult:
    manifest = FrozenEvalManifest.load(manifest_path)
    baseline, baseline_sha256 = RewardRecord.load_with_digest(baseline_path)
    post, post_sha256 = RewardRecord.load_with_digest(post_path)
    return compare_runs(
        manifest,
        baseline,
        post,
        baseline_sha256=baseline_sha256,
        post_sha256=post_sha256,
    )


def write_result(result: ComparisonResult, path: Path) -> None:
    write_json_exclusive(path, result.model_dump())
