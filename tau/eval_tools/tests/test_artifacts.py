import hashlib
import json
import stat
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tau.eval_tools.artifacts import (
    TrainingCompletionAttestation,
    TrainingResult,
    attempt_paths,
    materialize_adapter_for_eval,
    publish_training_result,
    select_final_adapter,
    validate_adapter_handoff,
    write_smoke_result,
    write_training_preflight,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError, load_json_with_sha256, write_json_exclusive
from tau.eval_tools.live.training_supervisor_live import (
    RLProcessResult,
    create_attempt_directory,
    supervise_prepared_attempt,
)
from tau.eval_tools.live.validate_training_data_live import (
    PRIVATE_DATASET_SENTINEL,
    PRIVATE_MODEL_SENTINEL,
    _bind_run_specific_config,
    _read_regular_file,
    _replace_runtime_paths,
    validate_source_toml_contract,
)
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
    build_file_manifest,
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


ATTEMPT_1 = "20260101T000000Z-0123456789abcdef"
ATTEMPT_2 = "20260101T000001Z-fedcba9876543210"


@pytest.fixture
def exact_config_validator(monkeypatch):
    source = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    exact_bytes = source.read_bytes()

    def install(manifest):
        def validate(path, _manifest, **_kwargs):
            payload = Path(path).read_bytes()
            if payload != exact_bytes:
                raise ValueError("resolved RL TOML is not the exact canonical experiment contract")
            validate_source_toml_contract(payload, tomllib.loads(payload.decode()), manifest)
            return object(), payload, manifest.rl_config

        monkeypatch.setattr(
            "tau.eval_tools.live.validate_training_data_live.validate_resolved_rl_config",
            validate,
        )
        monkeypatch.setattr(
            "tau.eval_tools.live.training_supervisor_live.validate_resolved_rl_config",
            validate,
        )
        monkeypatch.setattr(
            "tau.eval_tools.live.training_supervisor_live.validate_model_materialization",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            "tau.eval_tools.live.training_supervisor_live._reload_training_records",
            lambda *_args: manifest.training_data.records,
        )
        return exact_bytes

    return install


def _evidence(
    output_dir: Path,
    manifest,
    exact_config: bytes,
    *,
    attempt_id: str = ATTEMPT_1,
    write_completion: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    create_attempt_directory(output_dir, attempt_id)
    paths = attempt_paths(output_dir, attempt_id, require_existing=True)
    run_root = output_dir / f"private-{attempt_id}"
    run_root.mkdir()
    private_config = run_root / "resolved-train.toml"
    private_config.write_bytes(exact_config)
    paths.resolved_config.write_bytes(exact_config)
    model_path = run_root / "model"
    dataset_path = run_root / "training-dataset"
    model_path.mkdir()
    dataset_path.mkdir()
    source_adapter = _adapter(paths.run_output, 50)
    (paths.run_output / "metrics.jsonl").write_text("{}\n")
    preflight = write_training_preflight(
        output_path=paths.preflight,
        attempt_id=attempt_id,
        manifest=manifest,
        artifact_output_dir=output_dir,
        attempt_output_dir=paths.run_output,
        run_root=run_root,
        model_path=model_path,
        dataset_path=dataset_path,
        private_config_path=private_config,
        resolved_config_path=paths.resolved_config,
    )
    if write_completion:
        _, preflight_sha256 = load_json_with_sha256(paths.preflight)
        stable = source_adapter.parent / "STABLE"
        completion = TrainingCompletionAttestation(
            attempt_id=attempt_id,
            manifest_identity_hash=manifest.identity_hash(),
            preflight_sha256=preflight_sha256,
            resolved_config_sha256=hashlib.sha256(exact_config).hexdigest(),
            rl_pid=123,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:01:00+00:00",
            command_argv=["uv", "run", "--no-sync", "rl", "@", str(private_config)],
            executable="/usr/bin/uv",
            rl_config=manifest.rl_config,
            run_root=str(run_root),
            private_config_path=str(private_config),
            resolved_config_path=str(paths.resolved_config),
            artifact_output_dir=str(output_dir),
            attempt_output_dir=str(paths.run_output),
            source_step=50,
            stable_marker_path=str(stable),
            stable_marker_sha256=hashlib.sha256(stable.read_bytes()).hexdigest(),
            source_adapter_path=str(source_adapter),
            source_adapter_files=build_file_manifest(source_adapter),
        )
        write_json_exclusive(paths.completion, completion.model_dump())
    return preflight, paths


def _publication_kwargs(output_dir: Path, manifest, attempt_id: str = ATTEMPT_1):
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "attempt_id": attempt_id,
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


def test_publish_training_result_copies_fixed_handoff_and_records_metadata(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _, paths = _evidence(tmp_path, manifest, exact_config)
    source = paths.run_output / "weights/step_50/lora_adapters"
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
        training_output_dir=tmp_path,
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
            training_output_dir=tmp_path,
            expected_adapter_path=tmp_path / "final-adapter",
            expected_step=50,
            expected_rank=16,
        )


def test_publish_training_result_is_idempotent_after_completion(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    kwargs = _publication_kwargs(tmp_path, manifest)
    publish_training_result(**kwargs)
    staging = tmp_path / f".training-publication-{ATTEMPT_1}.stage"
    staging.mkdir()
    (staging / "stale").write_text("interrupted after completion")
    retried = publish_training_result(**kwargs)
    assert retried == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()


def test_publish_training_result_recovers_interrupted_final_install(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    kwargs = _publication_kwargs(tmp_path, manifest)
    publish_training_result(**kwargs)
    (tmp_path / "training-result.json").unlink()
    staging = tmp_path / f".training-publication-{ATTEMPT_1}.stage"
    staging.mkdir()
    (staging / "partial").write_text("interrupted")

    recovered = publish_training_result(**kwargs)

    assert recovered == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()


def test_private_adapter_materialization_is_readonly_and_content_bound(tmp_path, exact_config_validator):
    manifest = _manifest()
    durable = tmp_path / "durable"
    _evidence(durable, manifest, exact_config_validator(manifest))
    kwargs = _publication_kwargs(durable, manifest)
    result = publish_training_result(**kwargs)
    run_root = tmp_path / "prime-rl-run-private"
    run_root.mkdir(mode=0o700)
    private = materialize_adapter_for_eval(
        result=result,
        durable_adapter_path=durable / "final-adapter",
        run_root=run_root,
    )
    assert private == run_root / "adapter"
    assert stat.S_IMODE(private.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in private.iterdir())
    assert (run_root / "adapter-materialization-complete").is_file()


def test_publication_rejects_config_and_attestation_mismatch(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _, paths = _evidence(tmp_path, manifest, exact_config)
    kwargs = _publication_kwargs(tmp_path, manifest)
    paths.resolved_config.write_text("[model]\nname = 'false'\n")
    with pytest.raises(ValueError, match="canonical experiment contract"):
        publish_training_result(**kwargs)

    tmp_path_2 = tmp_path / "attestation"
    _, paths = _evidence(tmp_path_2, manifest, exact_config)
    kwargs = _publication_kwargs(tmp_path_2, manifest)
    payload = json.loads(paths.completion.read_text())
    payload["preflight_sha256"] = "f" * 64
    paths.completion.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preflight digest"):
        publish_training_result(**kwargs)


def _process_result(preflight, return_code: int) -> RLProcessResult:
    return RLProcessResult(
        argv=("uv", "run", "--no-sync", "rl", "@", preflight.private_config_path),
        executable="/usr/bin/uv",
        pid=456,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:01:00+00:00",
        return_code=return_code,
    )


def test_supervisor_nonzero_writes_no_attestation_and_fresh_attempt_succeeds(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    failed_preflight, failed_paths = _evidence(
        tmp_path,
        manifest,
        exact_config,
        attempt_id=ATTEMPT_1,
        write_completion=False,
    )
    successful_preflight, successful_paths = _evidence(
        tmp_path,
        manifest,
        exact_config,
        attempt_id=ATTEMPT_2,
        write_completion=False,
    )
    with pytest.raises(RuntimeError, match="no completion attestation"):
        supervise_prepared_attempt(
            manifest=manifest,
            preflight=failed_preflight,
            process_runner=lambda _argv: _process_result(failed_preflight, 7),
        )
    assert not failed_paths.completion.exists()
    assert not (tmp_path / "training-result.json").exists()

    result = supervise_prepared_attempt(
        manifest=manifest,
        preflight=successful_preflight,
        process_runner=lambda _argv: _process_result(successful_preflight, 0),
    )
    assert result.attempt_id == ATTEMPT_2
    assert successful_paths.completion.is_file()
    assert not failed_paths.completion.exists()


def test_supervisor_attestation_supports_explicit_publication_recovery(tmp_path, exact_config_validator):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )

    def interrupt_publication(**_kwargs):
        raise RuntimeError("simulated interruption after attestation")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        supervise_prepared_attempt(
            manifest=manifest,
            preflight=preflight,
            process_runner=lambda _argv: _process_result(preflight, 0),
            publisher=interrupt_publication,
        )
    assert paths.completion.is_file()
    assert not (tmp_path / "training-result.json").exists()

    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest))
    assert recovered.attempt_id == ATTEMPT_1


def test_supervisor_rejects_private_config_drift_and_false_process_attestation(tmp_path, exact_config_validator):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    private_config = Path(preflight.private_config_path)
    private_config.write_text("[model]\nname = 'altered'\n")
    invoked = False

    def runner(_argv):
        nonlocal invoked
        invoked = True
        return _process_result(preflight, 0)

    with pytest.raises(ValueError, match="canonical experiment contract"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight, process_runner=runner)
    assert not invoked
    assert not paths.completion.exists()

    private_config.write_bytes(paths.resolved_config.read_bytes())
    false_result = _process_result(preflight, 0)
    false_result = RLProcessResult(
        **{
            **false_result.__dict__,
            "argv": ("uv", "run", "--no-sync", "rl", "@", "/tmp/not-the-private-config"),
        }
    )
    with pytest.raises(ValueError, match="exact trusted RL command"):
        supervise_prepared_attempt(
            manifest=manifest,
            preflight=preflight,
            process_runner=lambda _argv: false_result,
        )
    assert not paths.completion.exists()
    assert not (tmp_path / "training-result.json").exists()


def test_attempt_id_reuse_and_cross_attempt_evidence_fail_closed(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _, paths = _evidence(tmp_path, manifest, exact_config)
    with pytest.raises(FileExistsError):
        create_attempt_directory(tmp_path, ATTEMPT_1)
    completion = json.loads(paths.completion.read_text())
    completion["attempt_id"] = ATTEMPT_2
    paths.completion.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="attempt evidence IDs"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest))


def test_recomputed_identity_rejects_false_preflight_and_post_config_drift(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _, paths = _evidence(tmp_path, manifest, exact_config)
    preflight = json.loads(paths.preflight.read_text())
    preflight["rl_config"]["canonical_resolved_sha256"] = "f" * 64
    paths.preflight.write_text(json.dumps(preflight))
    completion = json.loads(paths.completion.read_text())
    completion["preflight_sha256"] = hashlib.sha256(paths.preflight.read_bytes()).hexdigest()
    paths.completion.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="manifest/preflight"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest))

    clean = tmp_path / "clean"
    _evidence(clean, manifest, exact_config)
    result = publish_training_result(**_publication_kwargs(clean, manifest))
    clean_paths = attempt_paths(clean, ATTEMPT_1, require_existing=True)
    clean_paths.resolved_config.write_text("[model]\nname = 'altered'\n")
    with pytest.raises(ValueError, match="canonical experiment contract"):
        validate_adapter_handoff(
            result=result,
            manifest=manifest,
            training_output_dir=clean,
            expected_adapter_path=clean / "final-adapter",
            expected_step=50,
            expected_rank=16,
        )


def test_completed_result_prevents_supervisor_rerun_and_handoff_never_lists_attempts(
    tmp_path, exact_config_validator, monkeypatch
):
    manifest = _manifest()
    preflight, _ = _evidence(tmp_path, manifest, exact_config_validator(manifest))
    result = publish_training_result(**_publication_kwargs(tmp_path, manifest))
    invoked = False

    def runner(_argv):
        nonlocal invoked
        invoked = True
        return _process_result(preflight, 0)

    with pytest.raises(FileExistsError, match="refusing to rerun"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight, process_runner=runner)
    assert not invoked

    monkeypatch.setattr(Path, "iterdir", lambda _path: (_ for _ in ()).throw(AssertionError("listed")))
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        training_output_dir=tmp_path,
        expected_adapter_path=tmp_path / "final-adapter",
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


def test_runtime_config_identity_normalizes_all_attempt_private_paths_and_rejects_symlink(tmp_path):
    model = "/tmp/prime-rl-run-test/model"
    dataset = "/tmp/prime-rl-run-test/training-dataset"
    attempt_output = "/data/train/attempts/20260101T000000Z-0123456789abcdef/run-output"
    normalized = _replace_runtime_paths(
        {
            "model": model,
            "source": {"dataset": dataset, "file": f"{dataset}/data/train.parquet"},
            "output_dir": attempt_output,
            "metrics": f"{attempt_output}/metrics.jsonl",
        },
        model_path=model,
        dataset_path=dataset,
        output_dir=attempt_output,
        logical_output_dir="/data/train",
    )
    assert normalized == {
        "model": PRIVATE_MODEL_SENTINEL,
        "source": {
            "dataset": PRIVATE_DATASET_SENTINEL,
            "file": f"{PRIVATE_DATASET_SENTINEL}/data/train.parquet",
        },
        "output_dir": "/data/train",
        "metrics": "/data/train/metrics.jsonl",
    }

    target = tmp_path / "target.toml"
    target.write_text("[model]\nname = 'model'\n")
    link = tmp_path / "resolved.toml"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink component"):
        _read_regular_file(link)


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
