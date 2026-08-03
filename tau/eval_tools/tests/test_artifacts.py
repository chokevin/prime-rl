import hashlib
import json
import stat
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tau.eval_tools.artifacts import (
    TrainingResult,
    materialize_adapter_for_eval,
    publish_training_result,
    select_final_adapter,
    validate_adapter_handoff,
    write_smoke_result,
    write_training_completion_attestation,
    write_training_preflight,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError
from tau.eval_tools.live.validate_training_data_live import _bind_run_specific_config, validate_source_toml_contract
from tau.eval_tools.manifest import (
    DecodingConfig,
    ExampleRecord,
    FileManifest,
    FileRecord,
    ModelSnapshot,
    RLConfigIdentity,
    TasksetRef,
    TrainingDataIdentity,
    TrainingRecord,
    build_manifest,
)

SOURCE_REVISION = "a" * 40
VERIFIERS_REVISION = "b" * 40
TASKSETS_REVISION = "c" * 40
MODEL_REVISION = "d" * 40
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
    output_dir="/data/pretraining-data/prime-rl-math-7b-h200/train",
    max_steps=50,
    canonical_resolved_sha256="3" * 64,
)


def _manifest():
    return build_manifest(
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
            dataset_revision="e" * 40,
        ),
        training_data=TrainingDataIdentity.from_records(
            [
                TrainingRecord(
                    id=index,
                    prompt_hash=hash_text(f"training problem {index}"),
                    answer_hash=hash_text(f"training answer {index}"),
                )
                for index in range(300)
            ],
            file_manifest=DATASET_FILES,
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
        baseline_rewards_sha256="4" * 64,
        rl_config=RL_CONFIG,
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


def _evidence(tmp_path: Path, manifest):
    run_root = tmp_path / "prime-rl-run-test"
    run_root.mkdir()
    resolved_config = run_root / "resolved-train.toml"
    resolved_config.write_text("[model]\nname = '/private/model'\n")
    preflight_path = tmp_path / "training-preflight.json"
    preflight = write_training_preflight(
        output_path=preflight_path,
        manifest=manifest,
        config_identity=manifest.rl_config,
        run_root=run_root,
        model_path=run_root / "model",
        dataset_path=run_root / "training-dataset",
        resolved_config_path=resolved_config,
        resolved_config_sha256=hashlib.sha256(resolved_config.read_bytes()).hexdigest(),
    )
    completion_path = tmp_path / "training-completion.json"
    write_training_completion_attestation(
        output_path=completion_path,
        manifest=manifest,
        preflight_path=preflight_path,
        resolved_config_path=resolved_config,
        rl_pid=123,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:01:00+00:00",
        run_root=run_root,
        output_dir=tmp_path,
        source_step=50,
    )
    return preflight, preflight_path, completion_path, tmp_path / "training-resolved.toml"


def _publication_kwargs(tmp_path: Path, manifest):
    _, preflight, completion, resolved_config = _evidence(tmp_path, manifest)
    return {
        "output_dir": tmp_path,
        "manifest": manifest,
        "preflight_path": preflight,
        "completion_attestation_path": completion,
        "resolved_config_path": resolved_config,
        "expected_step": 50,
        "expected_rank": 16,
    }


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
    kwargs = _publication_kwargs(tmp_path, manifest)
    result = publish_training_result(**kwargs)
    loaded = TrainingResult.load(tmp_path / "training-result.json")
    assert loaded == result
    assert result.source_step == 50
    assert result.source_adapter_path == str(source)
    assert result.final_adapter_path == str(tmp_path / "final-adapter")
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        preflight_path=kwargs["preflight_path"],
        completion_attestation_path=kwargs["completion_attestation_path"],
        resolved_config_path=tmp_path / "training-resolved.toml",
        expected_adapter_path=tmp_path / "final-adapter",
        expected_step=50,
        expected_rank=16,
    )
    wrong_config = result.model_copy(
        update={"rl_config": result.rl_config.model_copy(update={"canonical_resolved_sha256": "f" * 64})}
    )
    with pytest.raises(ValueError, match="effective RL config identity"):
        validate_adapter_handoff(
            result=wrong_config,
            manifest=manifest,
            preflight_path=kwargs["preflight_path"],
            completion_attestation_path=kwargs["completion_attestation_path"],
            resolved_config_path=tmp_path / "training-resolved.toml",
            expected_adapter_path=tmp_path / "final-adapter",
            expected_step=50,
            expected_rank=16,
        )


def test_publish_training_result_is_idempotent_after_completion(tmp_path):
    _adapter(tmp_path, 50)
    manifest = _manifest()
    kwargs = _publication_kwargs(tmp_path, manifest)
    publish_training_result(**kwargs)
    staging = tmp_path / ".training-publication.stage"
    staging.mkdir()
    (staging / "stale").write_text("interrupted after completion")
    retried = publish_training_result(**kwargs)
    assert retried == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()


def test_publish_training_result_recovers_interrupted_final_install(tmp_path):
    _adapter(tmp_path, 50)
    manifest = _manifest()
    kwargs = _publication_kwargs(tmp_path, manifest)
    publish_training_result(**kwargs)
    (tmp_path / "training-result.json").unlink()
    staging = tmp_path / ".training-publication.stage"
    staging.mkdir()
    (staging / "partial").write_text("interrupted")

    recovered = publish_training_result(**kwargs)

    assert recovered == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()


def test_private_adapter_materialization_is_readonly_and_content_bound(tmp_path):
    source = _adapter(tmp_path / "durable", 50)
    manifest = _manifest()
    kwargs = _publication_kwargs(tmp_path / "durable", manifest)
    result = publish_training_result(**kwargs)
    run_root = tmp_path / "prime-rl-run-private"
    run_root.mkdir(mode=0o700)
    private = materialize_adapter_for_eval(
        result=result,
        durable_adapter_path=source.parent.parent.parent / "final-adapter",
        run_root=run_root,
    )
    assert private == run_root / "adapter"
    assert stat.S_IMODE(private.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in private.iterdir())
    assert (run_root / "adapter-materialization-complete").is_file()


def test_publication_rejects_config_and_attestation_mismatch(tmp_path):
    _adapter(tmp_path, 50)
    manifest = _manifest()
    kwargs = _publication_kwargs(tmp_path, manifest)
    Path(kwargs["resolved_config_path"]).write_text("changed")
    with pytest.raises(ValueError, match="resolved config bytes"):
        publish_training_result(**kwargs)

    tmp_path_2 = tmp_path / "attestation"
    tmp_path_2.mkdir()
    _adapter(tmp_path_2, 50)
    kwargs = _publication_kwargs(tmp_path_2, manifest)
    completion = Path(kwargs["completion_attestation_path"])
    payload = json.loads(completion.read_text())
    payload["preflight_sha256"] = "f" * 64
    completion.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preflight digest"):
        publish_training_result(**kwargs)


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


def test_source_equivalent_config_contract_accepts_full_current_toml():
    manifest = _manifest()
    config_path = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    source_bytes = config_path.read_bytes()
    validate_source_toml_contract(source_bytes, tomllib.loads(source_bytes.decode()), manifest)


def test_run_specific_config_binds_only_private_materializations():
    source = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    raw = tomllib.loads(source.read_text())
    _bind_run_specific_config(
        raw,
        model_path=Path("/tmp/prime-rl-run-test/model"),
        dataset_path=Path("/tmp/prime-rl-run-test/training-dataset"),
        output_dir=Path("/data/pretraining-data/prime-rl-math-7b-h200/train"),
        max_steps=50,
    )
    assert raw["model"]["name"] == "/tmp/prime-rl-run-test/model"
    assert (
        raw["orchestrator"]["train"]["source"][0]["env"]["taskset"]["dataset_name"]
        == "/tmp/prime-rl-run-test/training-dataset"
    )


def test_source_equivalent_config_contract_rejects_minimal_and_task_config_drift(tmp_path):
    manifest = _manifest()
    minimal = tmp_path / "minimal.toml"
    minimal.write_text("[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n")
    with pytest.raises(ValueError, match="canonical experiment contract"):
        validate_source_toml_contract(minimal.read_bytes(), tomllib.loads(minimal.read_text()), manifest)

    source = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    drifted = tmp_path / "drifted.toml"
    drifted.write_text(source.read_text().replace('judge = "None"', "judge = {}"))
    with pytest.raises(ValueError, match="canonical experiment contract"):
        validate_source_toml_contract(drifted.read_bytes(), tomllib.loads(drifted.read_text()), manifest)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "[orchestrator.eval]",
            "[orchestrator.train.source.sampling]\nmax_completion_tokens = 1\n\n[orchestrator.eval]",
        ),
        ("rank = 16", "rank = 16\ndropout = 1.0"),
        ("rank = 16", "rank = 16\nalpha = 1.0"),
        ("rank = 16", 'rank = 16\ntarget_modules = ["q_proj"]'),
        ("seq_len = 16384", "seq_len = 1"),
        ("[model]", "clean_output_dir = true\n\n[model]"),
        ("[model]", "dry_run = true\n\n[model]"),
        ("[trainer.ckpt.weights]", "[trainer.ckpt]\nresume_step = 1\n\n[trainer.ckpt.weights]"),
    ],
)
def test_canonical_resolved_config_rejects_behavior_drift(tmp_path, needle, replacement):
    source = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    drifted = tmp_path / "drifted.toml"
    drifted.write_text(source.read_text().replace(needle, replacement, 1))
    with pytest.raises(ValueError, match="canonical experiment contract"):
        source_bytes = drifted.read_bytes()
        validate_source_toml_contract(
            source_bytes,
            tomllib.loads(source_bytes.decode()),
            _manifest(),
        )
