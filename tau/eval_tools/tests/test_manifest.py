from datetime import datetime, timezone

import pytest

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    FrozenEvalManifest,
    LeakageError,
    ModelSnapshot,
    TasksetRef,
    build_manifest,
    check_disjoint,
    check_headroom,
)

N_EXAMPLES = 200


def _examples(n: int = N_EXAMPLES) -> list[ExampleRecord]:
    return [
        ExampleRecord(id=i, prompt_hash=hash_text(f"problem {i}"), answer_hash=hash_text(f"answer {i}"))
        for i in range(n)
    ]


def _manifest(**overrides) -> FrozenEvalManifest:
    kwargs = dict(
        model=ModelSnapshot(name="Qwen/Qwen2.5-7B-Instruct", revision="deadbeef"),
        eval_taskset=TasksetRef(
            id="math500-v1",
            dataset_name="HuggingFaceH4/MATH-500",
            dataset_split="test",
            dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        ),
        train_taskset=TasksetRef(id="math-env-v1", dataset_name="PrimeIntellect/Hendrycks-Math", dataset_split="train"),
        examples=_examples(),
        decoding=DecodingConfig(temperature=0.0, seed=0),
        grader="verifiers.v1.scoring.verify_boxed_math_answer",
        baseline_mean=0.4,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_build_manifest_valid():
    manifest = _manifest()
    assert manifest.n == N_EXAMPLES
    assert manifest.baseline_mean_headroom_ok is True


def test_validate_examples_rejects_fewer_than_200():
    with pytest.raises(ValueError, match=">= 200"):
        _manifest(examples=_examples(199))


def test_validate_examples_rejects_duplicate_ids():
    examples = _examples(200)
    examples[1] = ExampleRecord(
        id=examples[0].id, prompt_hash=examples[1].prompt_hash, answer_hash=examples[1].answer_hash
    )
    with pytest.raises(ValueError, match="duplicate example ids"):
        _manifest(examples=examples)


def test_identity_hash_is_deterministic_and_ignores_model_and_timestamp():
    m1 = _manifest()
    m2 = _manifest(
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct", revision="deadbeef", local_path="/data/models/qwen-lora-post"
        ),
        created_at="2099-01-01T00:00:00+00:00",
    )
    assert m1.identity_hash() == m2.identity_hash()


def test_identity_hash_changes_when_examples_change():
    m1 = _manifest()
    m2 = _manifest(examples=_examples(201))
    assert m1.identity_hash() != m2.identity_hash()


def test_identity_hash_changes_when_decoding_changes():
    m1 = _manifest()
    m2 = _manifest(decoding=DecodingConfig(temperature=0.7, seed=0))
    assert m1.identity_hash() != m2.identity_hash()


def test_check_disjoint_passes_with_no_overlap():
    eval_hashes = {hash_text("eval problem 1"), hash_text("eval problem 2")}
    train_hashes = {hash_text("train problem 1"), hash_text("train problem 2")}
    check_disjoint(eval_hashes, train_hashes)  # does not raise


def test_check_disjoint_raises_leakage_error_on_overlap():
    shared = hash_text("this problem leaked from eval into train")
    eval_hashes = {shared, hash_text("eval only")}
    train_hashes = {shared, hash_text("train only")}
    with pytest.raises(LeakageError, match="1 eval prompt hash"):
        check_disjoint(eval_hashes, train_hashes)


@pytest.mark.parametrize(
    ("mean", "expected"),
    [(0.0, False), (0.05, False), (0.10, True), (0.4, True), (0.80, True), (0.81, False), (1.0, False)],
)
def test_check_headroom_band(mean, expected):
    assert check_headroom(mean) is expected


def test_manifest_out_of_band_headroom_is_recorded_not_rejected():
    """An out-of-band baseline is a valid (negative) outcome, not a validation error —
    the harness must be able to record it and choose a different tuple before training."""
    manifest = _manifest(baseline_mean=0.95)
    assert manifest.baseline_mean_headroom_ok is False


def test_save_refuses_to_overwrite_by_default(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    manifest.save(path)
    with pytest.raises(FileExistsError, match="immutable"):
        manifest.save(path)


def test_save_load_roundtrip_preserves_identity(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    manifest.save(path)
    loaded = FrozenEvalManifest.load(path)
    assert loaded.identity_hash() == manifest.identity_hash()
    assert loaded.example_ids == manifest.example_ids
