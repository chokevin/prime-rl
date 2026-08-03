import tomllib
from datetime import datetime, timezone

import pytest

from tau.eval_tools.artifacts import (
    TrainingResult,
    publish_training_result,
    select_final_adapter,
    validate_adapter_handoff,
    write_smoke_result,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError
from tau.eval_tools.live.validate_training_data_live import _write_pinned_training_config
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    ModelSnapshot,
    TasksetRef,
    TrainingDataIdentity,
    build_manifest,
)

SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "d" * 40


def _manifest():
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
        training_data=TrainingDataIdentity.from_prompt_hashes(
            [hash_text(f"training problem {index}") for index in range(300)]
        ),
        examples=[
            ExampleRecord(
                id=index,
                prompt_hash=hash_text(f"problem {index}"),
                answer_hash=hash_text(f"answer {index}"),
            )
            for index in range(200)
        ],
        decoding=DecodingConfig(temperature=0.0, seed=0),
        grader="verifiers.v1.scoring.verify_boxed_math_answer",
        baseline_mean=0.4,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _adapter(output_dir, step: int, *, stable: bool = True, rank: int = 16):
    step_dir = output_dir / "weights" / f"step_{step}"
    adapter_dir = step_dir / "lora_adapters"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(f'{{"r": {rank}}}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
    if stable:
        (step_dir / "STABLE").touch()
    return adapter_dir


def test_select_final_adapter_uses_exact_expected_stable_rank_16_checkpoint(tmp_path):
    _adapter(tmp_path, 10)
    expected = _adapter(tmp_path, 20)
    _adapter(tmp_path, 30, stable=False)
    step, path, digest = select_final_adapter(
        tmp_path / "weights",
        expected_step=20,
        expected_rank=16,
    )
    assert step == 20
    assert path == expected
    assert len(digest) == 64


def test_publish_training_result_copies_fixed_handoff_and_records_metadata(tmp_path):
    source = _adapter(tmp_path, 50)
    manifest = _manifest()
    result = publish_training_result(
        output_dir=tmp_path,
        manifest=manifest,
        source_revision=SOURCE_REVISION,
        expected_step=50,
        expected_rank=16,
    )
    loaded = TrainingResult.load(tmp_path / "training-result.json")
    assert loaded == result
    assert result.source_step == 50
    assert result.source_adapter_path == str(source)
    assert result.final_adapter_path == str(tmp_path / "final-adapter")
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        expected_adapter_path=tmp_path / "final-adapter",
        expected_step=50,
        expected_rank=16,
    )


def test_publish_training_result_refuses_existing_fixed_handoff(tmp_path):
    _adapter(tmp_path, 50)
    manifest = _manifest()
    publish_training_result(
        output_dir=tmp_path,
        manifest=manifest,
        source_revision=SOURCE_REVISION,
        expected_step=50,
        expected_rank=16,
    )
    with pytest.raises(FileExistsError, match="immutable"):
        publish_training_result(
            output_dir=tmp_path,
            manifest=manifest,
            source_revision=SOURCE_REVISION,
            expected_step=50,
            expected_rank=16,
        )


def test_select_final_adapter_rejects_stable_checkpoint_at_wrong_step(tmp_path):
    _adapter(tmp_path, 49)
    with pytest.raises(FileNotFoundError, match="expected final checkpoint"):
        select_final_adapter(
            tmp_path / "weights",
            expected_step=50,
            expected_rank=16,
        )


def test_write_smoke_result_requires_configs_and_is_exclusive(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    for name in ("inference.toml", "orchestrator.toml", "trainer.toml"):
        (configs_dir / name).write_text("")
    output = tmp_path / "smoke-result.json"
    result = write_smoke_result(
        output_path=output,
        source_revision=SOURCE_REVISION,
        config_path="configs/tau/math-7b-h200/train.toml",
        configs_dir=configs_dir,
    )
    assert result.status == "success"
    with pytest.raises(FileExistsError):
        write_smoke_result(
            output_path=output,
            source_revision=SOURCE_REVISION,
            config_path="configs/tau/math-7b-h200/train.toml",
            configs_dir=configs_dir,
        )


def test_training_result_load_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "training-result.json"
    path.write_text('{"status":"success","status":"success"}')
    with pytest.raises(DuplicateKeyError, match="duplicate JSON key"):
        TrainingResult.load(path)


def test_write_pinned_training_config_uses_manifest_snapshot_exclusively(tmp_path):
    manifest = _manifest()
    snapshot_path = tmp_path / ("e" * 40)
    manifest.train_taskset.dataset_local_path = str(snapshot_path)
    source = tmp_path / "train.toml"
    source.write_text(
        """
[orchestrator]
[[orchestrator.train.source]]
name = "math"
[orchestrator.train.source.env.taskset]
id = "math-env-v1"
dataset_name = "PrimeIntellect/Hendrycks-Math"
dataset_subset = "default"
dataset_split = "train"
"""
    )
    output = tmp_path / "pinned.toml"
    _write_pinned_training_config(source, output, manifest)
    parsed = tomllib.loads(output.read_text())
    taskset = parsed["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    assert taskset["dataset_name"] == str(snapshot_path)
    assert taskset["dataset_subset"] == "default"
    assert taskset["dataset_split"] == "train"
    with pytest.raises(FileExistsError):
        _write_pinned_training_config(source, output, manifest)
