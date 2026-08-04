import os
from datetime import datetime, timezone

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
    make_tree_immutable,
    materialize_regular_snapshot,
    validate_file_manifest_by_exact_paths,
    validate_manifest_contract,
    validate_model_materialization,
    validate_training_materialization,
    validate_training_prompt_hashes,
    validate_tree_immutable,
)
from tau.eval_tools.output_paths import evidence_generation

N_EXAMPLES = 200
SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_DATASET_REVISION = "3ed63f49541bdca4382fba28146aadf20d95cb38"
TEST_GENERATION = evidence_generation(SOURCE_REVISION)
MODEL_FILES = FileManifest.from_records(
    [
        FileRecord(path="config.json", size=2, sha256="1" * 64),
        FileRecord(path="model.safetensors", size=7, sha256="2" * 64),
    ]
)
DATASET_FILES = FileManifest.from_records([FileRecord(path="data/train.parquet", size=7, sha256="5" * 64)])
RL_CONFIG = RLConfigIdentity(
    source_config_rel="configs/tau/math-7b-h200/train.toml",
    source_toml_sha256="4" * 64,
    output_dir=str(TEST_GENERATION.train),
    max_steps=50,
    canonical_resolved_sha256="3" * 64,
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
        ],
        file_manifest=DATASET_FILES,
    )
    kwargs = dict(
        state="finalized",
        source_revision=SOURCE_REVISION,
        verifiers_revision=VERIFIERS_REVISION,
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
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
            ],
            file_manifest=DATASET_FILES,
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
        TrainingDataIdentity.from_records([records[0], duplicate_id], file_manifest=DATASET_FILES)
    duplicate_prompt = records[1].model_copy(update={"prompt_hash": records[0].prompt_hash})
    with pytest.raises(ValueError, match="duplicate prompt hashes"):
        TrainingDataIdentity.from_records([records[0], duplicate_prompt], file_manifest=DATASET_FILES)


def _materialized_model(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    model_dir = run_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    model = ModelSnapshot(
        name="Qwen/Qwen2.5-7B-Instruct",
        revision=MODEL_REVISION,
        file_manifest=build_file_manifest(model_dir),
    )
    return model, model_dir, run_root


def test_model_manifest_rejects_weight_mutation_missing_extra_and_path_swap(tmp_path):
    model, model_dir, run_root = _materialized_model(tmp_path)
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed=.*model.safetensors"):
        validate_model_materialization(model, model_dir, run_root)

    weight.write_bytes(b"weights")
    weight.unlink()
    with pytest.raises(ValueError, match="missing=.*model.safetensors"):
        validate_model_materialization(model, model_dir, run_root)

    weight.write_bytes(b"weights")
    extra = model_dir / "unexpected.txt"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="extra=.*unexpected.txt"):
        validate_model_materialization(model, model_dir, run_root)

    extra.unlink()
    weight.unlink()
    weight.mkdir()
    with pytest.raises(ValueError, match="missing=.*model.safetensors"):
        validate_model_materialization(model, model_dir, run_root)


def test_file_manifest_rejects_wrong_aggregate_digest(tmp_path):
    model, _, _ = _materialized_model(tmp_path)
    payload = model.file_manifest.model_dump()
    payload["aggregate_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aggregate digest"):
        FileManifest.model_validate(payload)


def test_model_snapshot_rejects_directory_path_swap(tmp_path):
    model, _, run_root = _materialized_model(tmp_path)
    wrong_path = run_root / "other"
    wrong_path.mkdir()
    (wrong_path / "config.json").write_text("{}")
    (wrong_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="private model path"):
        validate_model_materialization(model, wrong_path, run_root)


def test_model_manifest_rejects_symlink_escape(tmp_path):
    model, model_dir, run_root = _materialized_model(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-weights"
    outside.write_bytes(b"weights")
    weight = model_dir / "model.safetensors"
    weight.unlink()
    weight.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="symlink"):
            validate_model_materialization(model, model_dir, run_root)
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


def _nested_snapshot(cache_root):
    source = cache_root / "model-cache" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / MODEL_REVISION
    (source / "weights").mkdir(parents=True)
    (source / "config.json").write_bytes(b'{"model_type":"qwen2"}')
    (source / "weights" / "model.safetensors").write_bytes(b"exact-weights")
    return source


@pytest.mark.parametrize("destination_name", ["model", "training-dataset"])
def test_snapshot_materialization_accepts_direct_child_destination(tmp_path, destination_name):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)
    source_manifest = build_file_manifest(source)
    destination = run_root / destination_name

    materialized = materialize_regular_snapshot(source, run_root, destination)

    assert materialized == source_manifest
    assert (destination / "config.json").read_bytes() == b'{"model_type":"qwen2"}'
    assert (destination / "weights" / "model.safetensors").read_bytes() == b"exact-weights"
    assert build_file_manifest(destination) == source_manifest


def test_snapshot_materialization_rejects_destination_root(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)

    with pytest.raises(ValueError, match="strict child"):
        materialize_regular_snapshot(source, run_root, run_root)


def test_snapshot_materialization_rejects_destination_outside_root_before_mkdir(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)
    outside_parent = tmp_path / "outside"

    with pytest.raises(ValueError, match="strict child"):
        materialize_regular_snapshot(source, run_root, outside_parent / "model")

    assert not outside_parent.exists()


def test_snapshot_materialization_rejects_symlinked_destination_parent_before_mkdir(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        materialize_regular_snapshot(source, run_root, run_root / "linked" / "nested" / "model")

    assert not (outside / "nested").exists()


def test_snapshot_materialization_rejects_symlinked_destination(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (run_root / "model").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        materialize_regular_snapshot(source, run_root, run_root / "model")


def test_snapshot_materialization_preserves_stale_direct_child_staging(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    source = _nested_snapshot(run_root)
    staging = run_root / ".model.materializing"
    staging.mkdir()
    marker = staging / "preserve"
    marker.write_text("stale")

    with pytest.raises(FileExistsError, match="staging path"):
        materialize_regular_snapshot(source, run_root, run_root / "model")

    assert marker.read_text() == "stale"


def test_private_training_materialization_is_path_bound_and_immutable(tmp_path):
    run_root = tmp_path / "prime-rl-run-test"
    dataset = run_root / "training-dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "data" / "train-00000-of-00001.parquet").write_bytes(b"parquet")
    actual_files = build_file_manifest(dataset)
    manifest = _manifest(training_data=_manifest().training_data.model_copy(update={"file_manifest": actual_files}))
    assert validate_training_materialization(manifest, dataset, run_root) == dataset.resolve()
    (run_root / "other").mkdir()
    with pytest.raises(ValueError, match="private training dataset path"):
        validate_training_materialization(manifest, run_root / "other", run_root)

    make_tree_immutable(dataset)
    validate_tree_immutable(dataset, actual_files)
    assert dataset.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in dataset.rglob("*"))
    (dataset / "data").chmod(0o755)
    with pytest.raises(ValueError, match="writable"):
        validate_tree_immutable(dataset, actual_files)


def test_validate_manifest_contract_accepts_exact_finalized_identity(tmp_path):
    manifest = _manifest(
        model=ModelSnapshot(
            name="Qwen/Qwen2.5-7B-Instruct",
            revision=MODEL_REVISION,
            file_manifest=MODEL_FILES,
        ),
        train_taskset=TasksetRef(
            id="math-env-v1",
            taskset_revision=TASKSETS_REVISION,
            dataset_name="PrimeIntellect/Hendrycks-Math",
            dataset_subset="default",
            dataset_split="train",
            dataset_revision=TRAIN_DATASET_REVISION,
        ),
        examples=_examples(500),
    )
    result = validate_manifest_contract(
        manifest,
        expected_source_revision=SOURCE_REVISION,
        expected_verifiers_revision=VERIFIERS_REVISION,
        expected_tasksets_revision=TASKSETS_REVISION,
        expected_model_name="Qwen/Qwen2.5-7B-Instruct",
        expected_model_revision=MODEL_REVISION,
        require_finalized=True,
    )
    assert result is None


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


def _empty_scandir_for(monkeypatch, *targets: os.PathLike | str):
    """Simulate a durable mount (e.g. BlobFuse) whose directory enumeration returns
    zero entries even though its named children are directly readable -- the exact
    fault observed in the F12 recovery incident. Delegates to the real `os.scandir`
    for every directory except the given targets."""
    real_scandir = os.scandir
    resolved_targets = {os.path.abspath(target) for target in targets}

    def fake_scandir(path="."):
        if os.path.abspath(path) in resolved_targets:
            return iter([])
        return real_scandir(path)

    monkeypatch.setattr("tau.eval_tools.manifest.os.scandir", fake_scandir)


def test_build_file_manifest_is_fooled_by_empty_directory_enumeration(tmp_path, monkeypatch):
    """Documents the actual F12 regression: enumeration-based `build_file_manifest`
    (and therefore `validate_file_manifest`) is unsafe at a durable mount boundary
    because it trusts `os.scandir`, which can return zero records even though the
    directory's named children are directly readable."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    assert (adapter / "adapter_config.json").is_file()

    _empty_scandir_for(monkeypatch, adapter)
    with pytest.raises(ValueError, match="must contain at least one file"):
        build_file_manifest(adapter)


def test_validate_file_manifest_by_exact_paths_survives_empty_directory_enumeration(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    expected = build_file_manifest(adapter)

    _empty_scandir_for(monkeypatch, adapter)
    with pytest.raises(ValueError, match="must contain at least one file"):
        build_file_manifest(adapter)

    # The exact-path validator never calls os.scandir, so it is unaffected.
    actual = validate_file_manifest_by_exact_paths(adapter, expected)
    assert actual == expected


def test_validate_file_manifest_by_exact_paths_rejects_missing_file(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    expected = build_file_manifest(adapter)

    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises((FileNotFoundError, OSError)):
        validate_file_manifest_by_exact_paths(adapter, expected)


def test_validate_file_manifest_by_exact_paths_rejects_changed_content(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    expected = build_file_manifest(adapter)

    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="exact-path file manifest mismatch"):
        validate_file_manifest_by_exact_paths(adapter, expected)


def test_validate_file_manifest_by_exact_paths_rejects_symlinked_source_file(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    expected = build_file_manifest(adapter)

    outside = tmp_path / "outside-weights"
    outside.write_bytes(b"weights")
    weight = adapter / "adapter_model.safetensors"
    weight.unlink()
    weight.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        validate_file_manifest_by_exact_paths(adapter, expected)


def test_validate_file_manifest_by_exact_paths_ignores_extra_entries(tmp_path):
    """The exact-path validator is intentionally scoped to proving expected records
    are present/unchanged; it cannot see untracked extras since it never enumerates.
    Extra-entry rejection remains the freshly-created private destination's job."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    expected = build_file_manifest(adapter)

    (adapter / "unexpected.txt").write_text("extra")
    validate_file_manifest_by_exact_paths(adapter, expected)
