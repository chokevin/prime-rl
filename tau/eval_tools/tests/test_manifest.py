from datetime import datetime, timezone
from pathlib import Path

import pytest

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    FileManifest,
    FileRecord,
    FrozenEvalManifest,
    LeakageError,
    ModelSnapshot,
    RLConfigIdentity,
    TasksetRef,
    TrainingDataIdentity,
    TrainingRecord,
    build_file_manifest,
    build_manifest,
    check_disjoint,
    check_headroom,
    materialize_regular_snapshot,
    validate_manifest_contract,
    validate_model_snapshot,
    validate_training_prompt_hashes,
)

N_EXAMPLES = 200
SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_DATASET_REVISION = "3ed63f49541bdca4382fba28146aadf20d95cb38"
MODEL_FILES = FileManifest.from_records(
    [
        FileRecord(path="config.json", size=2, sha256="1" * 64),
        FileRecord(path="model.safetensors", size=7, sha256="2" * 64),
    ]
)
RL_CONFIG = RLConfigIdentity(
    source_config_rel="configs/tau/math-7b-h200/train.toml",
    output_dir="/data/pretraining-data/prime-rl-math-7b-h200/train",
    max_steps=50,
    resolved_toml_sha256="3" * 64,
)


def _examples(n: int = N_EXAMPLES) -> list[ExampleRecord]:
    return [
        ExampleRecord(id=i, prompt_hash=hash_text(f"problem {i}"), answer_hash=hash_text(f"answer {i}"))
        for i in range(n)
    ]


def _manifest(**overrides) -> FrozenEvalManifest:
    training_data = TrainingDataIdentity.from_records(
        [
            TrainingRecord(
                id=i,
                prompt_hash=hash_text(f"training problem {i}"),
                answer_hash=hash_text(f"training answer {i}"),
            )
            for i in range(300)
        ]
    )
    kwargs = dict(
        state="finalized",
        source_revision=SOURCE_REVISION,
        verifiers_revision=VERIFIERS_REVISION,
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
            cache_root="/tmp/models",
            local_path=f"/tmp/models/{MODEL_REVISION}",
            file_manifest=MODEL_FILES,
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
        baseline_rewards_sha256="4" * 64,
        rl_config=RL_CONFIG,
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


def test_evaluation_identity_hash_is_deterministic_and_ignores_timestamp():
    m1 = _manifest()
    m2 = _manifest(
        created_at="2099-01-01T00:00:00+00:00",
    )
    assert m1.evaluation_identity_hash() == m2.evaluation_identity_hash()
    assert m1.identity_hash() == m2.identity_hash()


def test_identity_hash_changes_when_model_revision_changes():
    m1 = _manifest()
    revision = "f" * 40
    m2 = _manifest(
        model=ModelSnapshot(
            name=m1.model.name,
            revision=revision,
            cache_root="/tmp/models",
            local_path=f"/tmp/models/{revision}",
            file_manifest=m1.model.file_manifest,
        )
    )
    assert m1.identity_hash() != m2.identity_hash()


def test_identity_hash_changes_when_training_data_changes():
    m1 = _manifest()
    m2 = _manifest(
        training_data=TrainingDataIdentity.from_records(
            [
                TrainingRecord(
                    id=i,
                    prompt_hash=hash_text(f"other training problem {i}"),
                    answer_hash=hash_text(f"training answer {i}"),
                )
                for i in range(300)
            ]
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
    records = list(manifest.training_data.records)
    records[0] = records[0].model_copy(update={"prompt_hash": hash_text("drifted training prompt")})
    with pytest.raises(ValueError, match="ordered training records do not match"):
        validate_training_prompt_hashes(manifest, records)


def test_validate_training_records_rejects_answer_only_mutation_and_reordering():
    manifest = _manifest()
    answer_mutated = list(manifest.training_data.records)
    answer_mutated[0] = answer_mutated[0].model_copy(update={"answer_hash": hash_text("different answer")})
    with pytest.raises(ValueError, match="ordered training records do not match"):
        validate_training_prompt_hashes(manifest, answer_mutated)
    with pytest.raises(ValueError, match="ordered training records do not match"):
        validate_training_prompt_hashes(manifest, list(reversed(manifest.training_data.records)))
    with pytest.raises(ValueError, match="ordered training records do not match"):
        validate_training_prompt_hashes(manifest, manifest.training_data.records[:-1])
    extra = TrainingRecord(id=999, prompt_hash=hash_text("extra prompt"), answer_hash=hash_text("extra answer"))
    with pytest.raises(ValueError, match="ordered training records do not match"):
        validate_training_prompt_hashes(manifest, [*manifest.training_data.records, extra])


def test_training_identity_rejects_duplicate_ids_and_prompts():
    records = list(_manifest().training_data.records)
    duplicate_id = records[1].model_copy(update={"id": records[0].id})
    with pytest.raises(ValueError, match="duplicate record ids"):
        TrainingDataIdentity.from_records([records[0], duplicate_id])
    duplicate_prompt = records[1].model_copy(update={"prompt_hash": records[0].prompt_hash})
    with pytest.raises(ValueError, match="duplicate prompt hashes"):
        TrainingDataIdentity.from_records([records[0], duplicate_prompt])


def _materialized_model(tmp_path):
    model_dir = tmp_path / "prime-rl-trusted-models" / "Qwen--Qwen2.5-7B-Instruct" / MODEL_REVISION
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    return ModelSnapshot(
        name="Qwen/Qwen2.5-7B-Instruct",
        revision=MODEL_REVISION,
        cache_root=str(tmp_path),
        local_path=str(model_dir),
        file_manifest=build_file_manifest(model_dir),
    )


def test_model_manifest_rejects_weight_mutation_missing_extra_and_path_swap(tmp_path):
    model = _materialized_model(tmp_path)
    weight = Path(model.local_path) / "model.safetensors"
    weight.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed=.*model.safetensors"):
        validate_model_snapshot(model)

    weight.write_bytes(b"weights")
    weight.unlink()
    with pytest.raises(ValueError, match="missing=.*model.safetensors"):
        validate_model_snapshot(model)

    weight.write_bytes(b"weights")
    extra = Path(model.local_path) / "unexpected.txt"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="extra=.*unexpected.txt"):
        validate_model_snapshot(model)

    extra.unlink()
    weight.unlink()
    weight.mkdir()
    with pytest.raises(ValueError, match="missing=.*model.safetensors"):
        validate_model_snapshot(model)


def test_file_manifest_rejects_wrong_aggregate_digest(tmp_path):
    model = _materialized_model(tmp_path)
    payload = model.file_manifest.model_dump()
    payload["aggregate_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aggregate digest"):
        FileManifest.model_validate(payload)


def test_model_snapshot_rejects_directory_path_swap(tmp_path):
    model = _materialized_model(tmp_path)
    wrong_path = tmp_path / MODEL_REVISION
    wrong_path.mkdir()
    (wrong_path / "config.json").write_text("{}")
    (wrong_path / "model.safetensors").write_bytes(b"weights")
    swapped = model.model_copy(update={"local_path": str(wrong_path)})
    with pytest.raises(ValueError, match="expected fixed trusted path"):
        validate_model_snapshot(swapped)


def test_model_manifest_rejects_symlink_escape(tmp_path):
    model = _materialized_model(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-weights"
    outside.write_bytes(b"weights")
    weight = Path(model.local_path) / "model.safetensors"
    weight.unlink()
    weight.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="symlink"):
            validate_model_snapshot(model)
    finally:
        outside.unlink()


def test_snapshot_materialization_rejects_symlink_escape(tmp_path):
    cache_root = tmp_path / "cache"
    source = cache_root / "snapshots" / MODEL_REVISION
    source.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").symlink_to(outside)
    destination = cache_root / "prime-rl-trusted-models" / "Qwen--Qwen2.5-7B-Instruct" / MODEL_REVISION
    with pytest.raises(ValueError, match="escapes the verified cache root"):
        materialize_regular_snapshot(source, cache_root, destination)


def test_validate_manifest_contract_accepts_exact_finalized_identity(tmp_path):
    model_dir = tmp_path / "prime-rl-trusted-models" / "Qwen--Qwen2.5-7B-Instruct" / MODEL_REVISION
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    dataset_dir = tmp_path / "prime-rl-trusted-datasets" / "PrimeIntellect--Hendrycks-Math" / TRAIN_DATASET_REVISION
    (dataset_dir / "data").mkdir(parents=True)
    (dataset_dir / "data" / "train-00000-of-00001.parquet").write_bytes(b"parquet")
    manifest = _manifest(
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
            cache_root=str(tmp_path),
            local_path=str(model_dir),
            file_manifest=build_file_manifest(model_dir),
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
        baseline_rewards_sha256=None,
        rl_config=None,
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
