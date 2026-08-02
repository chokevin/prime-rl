from datetime import datetime, timezone

import numpy as np
import pytest

from tau.eval_tools.compare import (
    IdentityMismatchError,
    RewardRecord,
    compare_runs,
    paired_bootstrap_ci,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.manifest import DecodingConfig, ExampleRecord, ModelSnapshot, TasksetRef, build_manifest

N = 200


def _manifest(n: int = N):
    examples = [
        ExampleRecord(id=i, prompt_hash=hash_text(f"problem {i}"), answer_hash=hash_text(f"answer {i}"))
        for i in range(n)
    ]
    return build_manifest(
        model=ModelSnapshot(name="Qwen/Qwen2.5-7B-Instruct", revision="deadbeef"),
        eval_taskset=TasksetRef(id="math500-v1", dataset_name="HuggingFaceH4/MATH-500", dataset_split="test"),
        train_taskset=TasksetRef(id="math-env-v1", dataset_name="PrimeIntellect/Hendrycks-Math", dataset_split="train"),
        examples=examples,
        decoding=DecodingConfig(temperature=0.0, seed=0),
        grader="verifiers.v1.scoring.verify_boxed_math_answer",
        baseline_mean=0.4,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _reward_record(manifest, rewards: dict[int, float], *, model_label: str) -> RewardRecord:
    return RewardRecord(
        manifest_identity_hash=manifest.identity_hash(),
        model_label=model_label,
        created_at=datetime.now(timezone.utc).isoformat(),
        rewards={str(i): r for i, r in rewards.items()},
    )


def test_paired_bootstrap_ci_is_deterministic_for_same_seed():
    deltas = np.array([1.0, 0.0, -1.0, 1.0, 0.0] * 40)
    ci_a = paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=42)
    ci_b = paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=42)
    assert ci_a == ci_b


def test_paired_bootstrap_ci_differs_for_different_seed():
    deltas = np.array([1.0, 0.0, -1.0, 1.0, 0.0] * 40)
    ci_a = paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=1)
    ci_b = paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=2)
    assert ci_a != ci_b


def test_compare_runs_passes_on_clear_one_directional_improvement():
    manifest = _manifest()
    # baseline: first 80 correct (mean 0.4); post: first 100 correct (mean 0.5) - a
    # strictly one-directional +0.10 improvement on 20/200 examples, no regressions.
    baseline_rewards = {i: (1.0 if i < 80 else 0.0) for i in range(N)}
    post_rewards = {i: (1.0 if i < 100 else 0.0) for i in range(N)}
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    post = _reward_record(manifest, post_rewards, model_label="post")

    result = compare_runs(manifest, baseline, post, bootstrap_seed=0)

    assert result.delta == pytest.approx(0.10)
    assert result.passed is True
    assert result.ci_lower > 0


def test_compare_runs_fails_when_delta_below_threshold():
    manifest = _manifest()
    baseline_rewards = {i: (1.0 if i < 80 else 0.0) for i in range(N)}
    # Only 2/200 examples flip: delta = 0.01, below the +0.03 gate.
    post_rewards = {i: (1.0 if i < 82 else 0.0) for i in range(N)}
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    post = _reward_record(manifest, post_rewards, model_label="post")

    result = compare_runs(manifest, baseline, post, bootstrap_seed=0)

    assert result.delta == pytest.approx(0.01)
    assert result.passed is False


def test_compare_runs_fails_when_delta_meets_bar_but_ci_crosses_zero():
    """delta == +0.03 exactly meets the mean-delta gate, but the underlying per-example
    deltas are high-variance (+1/-1, nearly balanced) so the paired-bootstrap 95% CI
    lower bound is not > 0. The gate must fail on the CI condition alone."""
    manifest = _manifest()
    # 103 examples improve (0 -> 1), 97 regress (1 -> 0): mean delta = 6/200 = 0.03.
    baseline_rewards = {i: (0.0 if i < 103 else 1.0) for i in range(N)}
    post_rewards = {i: (1.0 if i < 103 else 0.0) for i in range(N)}
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    post = _reward_record(manifest, post_rewards, model_label="post")

    result = compare_runs(manifest, baseline, post, bootstrap_seed=0)

    assert result.delta == pytest.approx(0.03)
    assert result.ci_lower < 0
    assert result.passed is False


def test_compare_runs_raises_on_identity_mismatch():
    manifest = _manifest()
    other_manifest = _manifest(n=201)
    baseline_rewards = {i: 0.5 for i in range(N)}
    post_rewards = {i: 0.5 for i in range(N)}
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    # post computed against a *different* frozen manifest (wrong identity hash).
    post = _reward_record(other_manifest, post_rewards, model_label="post")

    with pytest.raises(IdentityMismatchError, match="manifest_identity_hash"):
        compare_runs(manifest, baseline, post)


def test_compare_runs_raises_on_example_id_mismatch():
    manifest = _manifest()
    baseline_rewards = {i: 0.5 for i in range(N)}
    post_rewards = {i: 0.5 for i in range(N - 1)}  # missing one id
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    post = _reward_record(manifest, post_rewards, model_label="post")

    with pytest.raises(IdentityMismatchError, match="example ids"):
        compare_runs(manifest, baseline, post)
