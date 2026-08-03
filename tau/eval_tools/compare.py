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

from tau.eval_tools.json_io import load_json, write_json_exclusive
from tau.eval_tools.manifest import FrozenEvalManifest

#: The goal harness's fixed acceptance thresholds (see the parent plan's "Measured hill
#: climb" done-check and decision ledger's "Metric default"). Not configurable via CLI
#: flags on purpose — relaxing them to fit a budget is explicitly forbidden.
MIN_DELTA = 0.03
CI_ALPHA = 0.05


class IdentityMismatchError(ValueError):
    """Raised when the two reward files don't share the same frozen manifest identity
    (different example sets, different decoding/grader settings, or a reward file that
    doesn't match the manifest passed on the CLI)."""


class RewardRecord(BaseModel):
    """One `<label>-rewards.json` file written by `live/run_frozen_eval_live.py`."""

    model_config = ConfigDict(extra="forbid")

    manifest_identity_hash: str
    model_label: Literal["baseline", "post"]
    created_at: str
    rewards: dict[str, float]
    """Example id (as string, for JSON-object-key portability) -> binary reward."""

    @field_validator("manifest_identity_hash")
    @classmethod
    def validate_manifest_identity_hash(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("manifest identity must be a lowercase SHA-256 hex digest")
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
        return cls.model_validate(load_json(path))


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    baseline_mean: float
    post_mean: float
    delta: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    bootstrap_seed: int
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
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = CI_ALPHA,
) -> tuple[float, float]:
    """Deterministic (seeded) percentile-bootstrap CI for the mean of `deltas` (one
    per-example post-minus-baseline reward difference). Resampling with a fixed seed
    and `numpy`'s legacy `RandomState` keeps this reproducible across machines/numpy
    versions for the same input, which the goal harness's replay requirement needs."""
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("deltas must be a non-empty 1-D array")
    rng = np.random.RandomState(seed)
    n = deltas.size
    resample_idx = rng.randint(0, n, size=(n_bootstrap, n))
    resampled_means = deltas[resample_idx].mean(axis=1)
    lower = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lower, upper


def compare_runs(
    manifest: FrozenEvalManifest,
    baseline: RewardRecord,
    post: RewardRecord,
    *,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> ComparisonResult:
    """Validate manifest/example identity between the two reward files, then compute
    the paired delta, bootstrap CI, and pass/fail gate."""
    identity = manifest.identity_hash()
    for label, record in (("baseline", baseline), ("post", post)):
        if record.model_label != label:
            raise IdentityMismatchError(
                f"{label} input carries model_label={record.model_label!r}; refusing a swapped or mislabeled comparison"
            )
        if record.manifest_identity_hash != identity:
            raise IdentityMismatchError(
                f"{label} reward file's manifest_identity_hash "
                f"({record.manifest_identity_hash}) does not match the frozen "
                f"manifest's ({identity}) — refusing to compare rewards computed "
                "against a different example set, decoding config, or grader."
            )

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
    ci_lower, ci_upper = paired_bootstrap_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    passed = bool(delta_mean >= MIN_DELTA and ci_lower > 0)

    return ComparisonResult(
        n=len(ordered_ids),
        baseline_mean=float(baseline_values.mean()),
        post_mean=float(post_values.mean()),
        delta=delta_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        passed=passed,
    )


def compare_from_paths(
    manifest_path: Path,
    baseline_path: Path,
    post_path: Path,
    *,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> ComparisonResult:
    manifest = FrozenEvalManifest.load(manifest_path)
    baseline = RewardRecord.load(baseline_path)
    post = RewardRecord.load(post_path)
    return compare_runs(manifest, baseline, post, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)


def write_result(result: ComparisonResult, path: Path) -> None:
    write_json_exclusive(path, result.model_dump())
