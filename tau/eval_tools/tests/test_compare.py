from datetime import datetime, timezone

import numpy as np
import pytest

from tau.eval_tools.compare import (
    IdentityMismatchError,
    RewardRecord,
    compare_runs,
    paired_bootstrap_ci,
    write_result,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    ModelSnapshot,
    TasksetRef,
    TrainingDataIdentity,
    build_manifest,
)

N = 200
SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "d" * 40


def _manifest(n: int = N):
    examples = [
        ExampleRecord(id=i, prompt_hash=hash_text(f"problem {i}"), answer_hash=hash_text(f"answer {i}"))
        for i in range(n)
    ]
    return build_manifest(
        state="finalized",
        source_revision=SOURCE_REVISION,
        verifiers_revision=VERIFIERS_REVISION,
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
            local_path=f"/tmp/models/{MODEL_REVISION}",
        ),
        eval_taskset=TasksetRef(
            id="math500-v1",
            taskset_revision=TASKSETS_REVISION,
            dataset_name="HuggingFaceH4/MATH-500",
            dataset_subset=None,
            dataset_split="test",
            dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        ),
        train_taskset=TasksetRef(
            id="math-env-v1",
            taskset_revision=TASKSETS_REVISION,
            dataset_name="PrimeIntellect/Hendrycks-Math",
            dataset_subset="default",
            dataset_split="train",
            dataset_revision="e" * 40,
        ),
        training_data=TrainingDataIdentity.from_prompt_hashes([hash_text(f"training problem {i}") for i in range(300)]),
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
    baseline_rewards = {i: 0.0 for i in range(N)}
    post_rewards = {i: 0.0 for i in range(N)}
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    # post computed against a *different* frozen manifest (wrong identity hash).
    post = _reward_record(other_manifest, post_rewards, model_label="post")

    with pytest.raises(IdentityMismatchError, match="manifest_identity_hash"):
        compare_runs(manifest, baseline, post)


def test_compare_runs_raises_on_example_id_mismatch():
    manifest = _manifest()
    baseline_rewards = {i: 0.0 for i in range(N)}
    post_rewards = {i: 0.0 for i in range(N - 1)}  # missing one id
    baseline = _reward_record(manifest, baseline_rewards, model_label="baseline")
    post = _reward_record(manifest, post_rewards, model_label="post")

    with pytest.raises(IdentityMismatchError, match="example ids"):
        compare_runs(manifest, baseline, post)


def test_compare_runs_rejects_swapped_labels():
    manifest = _manifest()
    rewards = {i: 0.0 for i in range(N)}
    baseline = _reward_record(manifest, rewards, model_label="post")
    post = _reward_record(manifest, rewards, model_label="baseline")
    with pytest.raises(IdentityMismatchError, match="mislabeled"):
        compare_runs(manifest, baseline, post)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -0.1, 0.5, 1.1])
def test_reward_record_rejects_values_outside_binary_grader_domain(reward):
    manifest = _manifest()
    with pytest.raises(ValueError, match="grader domain"):
        _reward_record(manifest, {0: reward}, model_label="baseline")


def test_reward_record_load_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text(
        '{"manifest_identity_hash":"x","model_label":"baseline","created_at":"now","rewards":{"0":0.0,"0":1.0}}'
    )
    with pytest.raises(DuplicateKeyError, match="duplicate JSON key"):
        RewardRecord.load(path)


def test_reward_record_rejects_noncanonical_example_id():
    manifest = _manifest()
    with pytest.raises(ValueError, match="canonical"):
        RewardRecord(
            manifest_identity_hash=manifest.identity_hash(),
            model_label="baseline",
            created_at="now",
            rewards={"00": 0.0},
        )


def test_write_result_refuses_to_overwrite(tmp_path):
    manifest = _manifest()
    baseline = _reward_record(manifest, {i: 0.0 for i in range(N)}, model_label="baseline")
    post = _reward_record(manifest, {i: 1.0 for i in range(N)}, model_label="post")
    result = compare_runs(manifest, baseline, post, n_bootstrap=100)
    path = tmp_path / "comparison.json"
    write_result(result, path)
    with pytest.raises(FileExistsError):
        write_result(result, path)
