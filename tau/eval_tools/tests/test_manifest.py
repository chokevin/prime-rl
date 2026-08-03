from datetime import datetime, timezone

import pytest

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    FrozenEvalManifest,
    LeakageError,
    ModelSnapshot,
    TasksetRef,
    TrainingDataIdentity,
    build_manifest,
    check_disjoint,
    check_headroom,
    validate_manifest_contract,
    validate_training_prompt_hashes,
)

N_EXAMPLES = 200
SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_DATASET_REVISION = "3ed63f49541bdca4382fba28146aadf20d95cb38"


def _examples(n: int = N_EXAMPLES) -> list[ExampleRecord]:
    return [
        ExampleRecord(id=i, prompt_hash=hash_text(f"problem {i}"), answer_hash=hash_text(f"answer {i}"))
        for i in range(n)
    ]


def _manifest(**overrides) -> FrozenEvalManifest:
    training_data = TrainingDataIdentity.from_prompt_hashes([hash_text(f"training problem {i}") for i in range(300)])
    kwargs = dict(
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
            dataset_revision=TRAIN_DATASET_REVISION,
            dataset_local_path=f"/datasets/{TRAIN_DATASET_REVISION}",
        ),
        training_data=training_data,
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


def test_validate_examples_rejects_duplicate_prompt_hashes():
    examples = _examples(200)
    examples[1] = ExampleRecord(
        id=examples[1].id,
        prompt_hash=examples[0].prompt_hash,
        answer_hash=examples[1].answer_hash,
    )
    with pytest.raises(ValueError, match="duplicate eval prompt hashes"):
        _manifest(examples=examples)


def test_identity_hash_is_deterministic_and_ignores_finalization_metadata():
    m1 = _manifest()
    m2 = _manifest(
        created_at="2099-01-01T00:00:00+00:00",
    )
    assert m1.identity_hash() == m2.identity_hash()


def test_identity_hash_changes_when_model_revision_changes():
    m1 = _manifest()
    revision = "f" * 40
    m2 = _manifest(
        model=ModelSnapshot(
            name=m1.model.name,
            revision=revision,
            local_path=f"/tmp/models/{revision}",
        )
    )
    assert m1.identity_hash() != m2.identity_hash()


def test_identity_hash_changes_when_training_data_changes():
    m1 = _manifest()
    m2 = _manifest(
        training_data=TrainingDataIdentity.from_prompt_hashes(
            [hash_text(f"other training problem {i}") for i in range(300)]
        )
    )
    assert m1.identity_hash() != m2.identity_hash()


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


def test_finalized_manifest_rejects_out_of_band_headroom():
    with pytest.raises(ValueError, match="headroom"):
        _manifest(baseline_mean=0.95)


def test_draft_manifest_rejects_finalized_headroom_state():
    with pytest.raises(ValueError, match="draft manifest"):
        _manifest(state="draft")


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


def test_manifest_load_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"state":"draft","state":"finalized"}')
    with pytest.raises(DuplicateKeyError, match="duplicate JSON key"):
        FrozenEvalManifest.load(path)


def test_manifest_rejects_unknown_fields():
    payload = _manifest().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        FrozenEvalManifest.model_validate(payload)


def test_validate_training_prompt_hashes_rejects_actual_drift():
    manifest = _manifest()
    actual = list(manifest.training_data.prompt_hashes)
    actual[0] = hash_text("drifted training prompt")
    with pytest.raises(ValueError, match="training prompts do not match"):
        validate_training_prompt_hashes(manifest, actual)


def test_validate_manifest_contract_accepts_exact_finalized_identity(tmp_path):
    model_dir = tmp_path / MODEL_REVISION
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    dataset_dir = tmp_path / "datasets" / TRAIN_DATASET_REVISION
    (dataset_dir / "data").mkdir(parents=True)
    (dataset_dir / "data" / "train-00000-of-00001.parquet").write_bytes(b"parquet")
    manifest = _manifest(
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
            local_path=str(model_dir),
        ),
        train_taskset=TasksetRef(
            id="math-env-v1",
            taskset_revision=TASKSETS_REVISION,
            dataset_name="PrimeIntellect/Hendrycks-Math",
            dataset_subset="default",
            dataset_split="train",
            dataset_revision=TRAIN_DATASET_REVISION,
            dataset_local_path=str(dataset_dir),
        ),
        examples=_examples(500),
    )
    path = validate_manifest_contract(
        manifest,
        expected_source_revision=SOURCE_REVISION,
        expected_verifiers_revision=VERIFIERS_REVISION,
        expected_tasksets_revision=TASKSETS_REVISION,
        expected_model_name="Qwen/Qwen2.5-7B-Instruct",
        expected_model_revision=MODEL_REVISION,
        require_finalized=True,
    )
    assert path == model_dir


def test_validate_manifest_contract_rejects_draft(tmp_path):
    manifest = _manifest(
        state="draft",
        baseline_mean=None,
        examples=_examples(500),
    )
    with pytest.raises(ValueError, match="expected 'finalized'"):
        validate_manifest_contract(
            manifest,
            expected_source_revision=SOURCE_REVISION,
            expected_verifiers_revision=VERIFIERS_REVISION,
            expected_tasksets_revision=TASKSETS_REVISION,
            expected_model_name="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision=MODEL_REVISION,
            require_finalized=True,
        )
