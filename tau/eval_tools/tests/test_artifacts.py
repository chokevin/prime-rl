import hashlib
import inspect
import json
import os
import signal
import stat
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

import tau.eval_tools.artifacts as artifacts_module
import tau.eval_tools.live.training_supervisor_live as supervisor_module
from tau.eval_tools.artifacts import (
    TrainingCompletionAttestation,
    TrainingResult,
    _write_json_staged_noreplace,
    attempt_paths,
    materialize_adapter_for_eval,
    publish_training_result,
    select_final_adapter,
    validate_adapter_handoff,
    write_smoke_result,
    write_training_preflight,
)
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.json_io import DuplicateKeyError, load_json_with_sha256
from tau.eval_tools.live.training_supervisor_live import (
    SupervisorCancelled,
    capture_private_run_root,
    create_attempt_directory,
    create_private_run_root,
    recover_publish,
    remove_private_run_root,
    run_training_attempt,
    supervise_prepared_attempt,
)
from tau.eval_tools.live.training_supervisor_live import main as supervisor_main
from tau.eval_tools.live.validate_training_data_live import (
    PRIVATE_DATASET_SENTINEL,
    PRIVATE_MODEL_SENTINEL,
    _bind_run_specific_config,
    _read_regular_file,
    _replace_runtime_paths,
    resolve_effective_rl_config,
    validate_resolved_rl_config,
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
        _write_json_staged_noreplace(
            final_path=paths.completion,
            staging_path=paths.completion_staging,
            payload=completion.model_dump(),
        )
    return preflight, paths


def _publication_kwargs(output_dir: Path, manifest, attempt_id: str = ATTEMPT_1):
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "attempt_id": attempt_id,
        "expected_step": 50,
        "expected_rank": 16,
    }


def _completion_quarantines(paths):
    return sorted(paths.directory.glob(".completion.json.stage.quarantine-*"))


def _completion_final_quarantines(paths):
    return sorted(paths.directory.glob("completion.json.quarantine-*"))


def _publication_quarantines(paths):
    return sorted(paths.directory.glob(".publication.stage.quarantine-*"))


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
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    existing_quarantines = set(_publication_quarantines(paths))
    staging = paths.publication_staging
    staging.mkdir()
    (staging / "stale").write_text("interrupted after completion")
    retried = publish_training_result(**kwargs)
    assert retried == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()
    new_quarantines = set(_publication_quarantines(paths)) - existing_quarantines
    assert len(new_quarantines) == 1
    assert (new_quarantines.pop() / "stale").read_text() == "interrupted after completion"


def test_publish_training_result_recovers_adapter_install_and_partial_json_stage(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    kwargs = _publication_kwargs(tmp_path, manifest)
    publish_training_result(**kwargs)
    (tmp_path / "training-result.json").unlink()
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    existing_quarantines = set(_publication_quarantines(paths))
    paths.publication.unlink()
    staging = paths.publication_staging
    staging.mkdir()
    (staging / "publication.json").write_text('{"status":"pub')
    paths.completion_staging.write_bytes(paths.completion.read_bytes())

    recovered = publish_training_result(**kwargs, recovery=True)

    assert recovered == TrainingResult.load(tmp_path / "training-result.json")
    assert not staging.exists()
    new_quarantines = set(_publication_quarantines(paths)) - existing_quarantines
    assert len(new_quarantines) == 2
    assert any(
        (quarantine / "publication.json").read_text() == '{"status":"pub'
        for quarantine in new_quarantines
        if (quarantine / "publication.json").is_file()
    )
    assert not paths.completion_staging.exists()
    assert len(_completion_quarantines(paths)) == 1


def test_recovery_promotes_valid_fsynced_completion_stage(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    paths.completion.rename(paths.completion_staging)
    with paths.completion_staging.open("rb") as staged:
        os.fsync(staged.fileno())

    result = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert result.attempt_id == ATTEMPT_1
    assert paths.completion.is_file()
    assert not paths.completion_staging.exists()


@pytest.mark.parametrize(
    ("window", "operation"),
    [
        ("before-promotion", "mutate"),
        ("before-promotion", "swap"),
        ("after-promotion", "mutate"),
        ("after-promotion", "swap"),
    ],
)
def test_recovery_binds_completion_promotion_to_validated_handle(
    tmp_path,
    exact_config_validator,
    monkeypatch,
    window,
    operation,
):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    original = paths.completion.read_bytes()
    paths.completion.rename(paths.completion_staging)

    def alter(path):
        path = Path(path)
        if operation == "mutate":
            path.write_bytes(original + b" ")
        else:
            replacement = path.with_name(f"{path.name}.replacement")
            replacement.write_bytes(original)
            os.replace(replacement, path)

    if window == "before-promotion":
        monkeypatch.setattr(artifacts_module, "_before_completion_stage_promotion", alter)
    else:
        monkeypatch.setattr(artifacts_module, "_after_completion_stage_promotion", alter)

    with pytest.raises(ValueError, match="completion"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()
    assert not (tmp_path / "final-adapter").exists()


def test_recovery_quarantines_malformed_completion_stage_without_other_changes(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    paths.completion.unlink()
    paths.completion_staging.write_text('{"status":"succ')
    preflight_before = paths.preflight.read_bytes()
    config_before = paths.resolved_config.read_bytes()

    with pytest.raises(ValueError, match="staged completion attestation is incomplete or invalid"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert not paths.completion_staging.exists()
    quarantines = _completion_quarantines(paths)
    assert len(quarantines) == 1
    assert quarantines[0].read_text() == '{"status":"succ'
    assert paths.preflight.read_bytes() == preflight_before
    assert paths.resolved_config.read_bytes() == config_before
    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()


@pytest.mark.parametrize("staged_payload", [b'{"status":"succ', b'{"not":"a completion attestation"}'])
def test_recovery_with_valid_final_quarantines_malformed_stage_then_requires_retry(
    tmp_path,
    exact_config_validator,
    staged_payload,
):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    final_before = paths.completion.read_bytes()
    paths.completion_staging.write_bytes(staged_payload)

    with pytest.raises(ValueError, match="malformed staged completion was quarantined"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert paths.completion.read_bytes() == final_before
    assert not paths.completion_staging.exists()
    quarantines = _completion_quarantines(paths)
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == staged_payload
    assert not paths.publication.exists()
    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
    assert recovered.attempt_id == ATTEMPT_1


def test_recovery_with_valid_final_quarantines_symlink_stage_without_following_target(
    tmp_path,
    exact_config_validator,
):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    target = tmp_path / "outside-stage-target.json"
    target.write_bytes(paths.completion.read_bytes())
    paths.completion_staging.symlink_to(target)

    with pytest.raises(ValueError, match="malformed staged completion was quarantined"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert paths.completion.is_file()
    assert not os.path.lexists(paths.completion_staging)
    quarantines = _completion_quarantines(paths)
    assert len(quarantines) == 1
    assert quarantines[0].is_symlink()
    assert target.read_bytes() == paths.completion.read_bytes()
    assert not paths.publication.exists()
    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
    assert recovered.attempt_id == ATTEMPT_1


def test_quarantine_detects_stage_swap_and_preserves_replacement(
    tmp_path,
    exact_config_validator,
    monkeypatch,
):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    original_stage = b'{"status":"succ'
    replacement_stage = b'{"replacement":"must survive"}'
    paths.completion_staging.write_bytes(original_stage)
    displaced = paths.directory / ".completion.json.stage.displaced"

    def swap_stage(_path):
        paths.completion_staging.rename(displaced)
        paths.completion_staging.write_bytes(replacement_stage)

    monkeypatch.setattr(artifacts_module, "_before_staging_quarantine", swap_stage)

    with pytest.raises(ValueError, match="could not be quarantined"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert not paths.completion_staging.exists()
    assert displaced.read_bytes() == original_stage
    quarantines = _completion_quarantines(paths)
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == replacement_stage
    assert not paths.publication.exists()
    monkeypatch.setattr(artifacts_module, "_before_staging_quarantine", lambda _path: None)
    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
    assert recovered.attempt_id == ATTEMPT_1


def test_recovery_quarantines_matching_stage_only_after_validating_final(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    paths.completion_staging.write_bytes(paths.completion.read_bytes())

    result = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert result.attempt_id == ATTEMPT_1
    assert paths.completion.is_file()
    assert not paths.completion_staging.exists()
    assert len(_completion_quarantines(paths)) == 1


def test_recovery_rejects_mismatched_final_and_completion_stage(tmp_path, exact_config_validator):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    staged = json.loads(paths.completion.read_text())
    staged["rl_pid"] += 1
    paths.completion_staging.write_text(json.dumps(staged))

    with pytest.raises(ValueError, match="does not match finalized completion"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)

    assert paths.completion.is_file()
    assert paths.completion_staging.is_file()
    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()


def test_publication_recovery_rejects_malformed_final_and_normal_cross_attempt_reuse(tmp_path, exact_config_validator):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _evidence(tmp_path, manifest, exact_config, attempt_id=ATTEMPT_1)
    publish_training_result(**_publication_kwargs(tmp_path, manifest, ATTEMPT_1))
    (tmp_path / "training-result.json").unlink()
    attempt_1_paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    attempt_1_paths.publication.unlink()
    _evidence(tmp_path, manifest, exact_config, attempt_id=ATTEMPT_2)

    with pytest.raises(FileExistsError, match="explicit verified publication recovery"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest, ATTEMPT_2))
    recovered = publish_training_result(
        **_publication_kwargs(tmp_path, manifest, ATTEMPT_2),
        recovery=True,
    )
    assert recovered.attempt_id == ATTEMPT_2

    (tmp_path / "training-result.json").unlink()
    attempt_2_paths = attempt_paths(tmp_path, ATTEMPT_2, require_existing=True)
    attempt_2_paths.publication.write_text('{"status":"pub')
    with pytest.raises(json.JSONDecodeError):
        publish_training_result(
            **_publication_kwargs(tmp_path, manifest, ATTEMPT_2),
            recovery=True,
        )


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


def _install_fake_popen(monkeypatch, tmp_path, *, return_code: int, cancel_signal: int | None = None):
    executable = tmp_path / "bin" / "uv"
    executable.parent.mkdir(exist_ok=True)
    executable.touch()
    captured = {"signals": []}

    class FakeProcess:
        pid = 456

        def __init__(self, argv, *, start_new_session):
            assert start_new_session is True
            captured["argv"] = tuple(argv)
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self):
            if cancel_signal is not None:
                signal.raise_signal(cancel_signal)
            self.returncode = return_code
            return return_code

    monkeypatch.setattr(
        "tau.eval_tools.live.training_supervisor_live.shutil.which",
        lambda command: str(executable) if command == "uv" else None,
    )
    monkeypatch.setattr("tau.eval_tools.live.training_supervisor_live.subprocess.Popen", FakeProcess)
    monkeypatch.setattr(
        "tau.eval_tools.live.training_supervisor_live._signal_process_group",
        lambda pid, signum: captured["signals"].append((pid, signum)),
    )
    return captured


def test_supervisor_nonzero_writes_no_attestation_and_fresh_attempt_succeeds(
    tmp_path, exact_config_validator, monkeypatch
):
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
    _install_fake_popen(monkeypatch, tmp_path, return_code=7)
    with pytest.raises(RuntimeError, match="no completion attestation"):
        supervise_prepared_attempt(manifest=manifest, preflight=failed_preflight)
    assert not failed_paths.completion.exists()
    assert not (tmp_path / "training-result.json").exists()

    _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    result = supervise_prepared_attempt(manifest=manifest, preflight=successful_preflight)
    assert result.attempt_id == ATTEMPT_2
    assert successful_paths.completion.is_file()
    assert not failed_paths.completion.exists()


def test_supervisor_attestation_supports_explicit_publication_recovery(tmp_path, exact_config_validator, monkeypatch):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )

    def interrupt_publication(**_kwargs):
        raise RuntimeError("simulated interruption after attestation")

    paths.completion_staging.write_text('{"status":"succ')
    _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    monkeypatch.setattr(
        "tau.eval_tools.live.training_supervisor_live.publish_training_result",
        interrupt_publication,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)
    assert paths.completion.is_file()
    assert not paths.completion_staging.exists()
    assert not (tmp_path / "training-result.json").exists()

    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
    assert recovered.attempt_id == ATTEMPT_1


def test_supervisor_rejects_private_config_drift_and_owns_exact_process(tmp_path, exact_config_validator, monkeypatch):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    private_config = Path(preflight.private_config_path)
    private_config.write_text("[model]\nname = 'altered'\n")
    captured = _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    with pytest.raises(ValueError, match="canonical experiment contract"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)
    assert "argv" not in captured
    assert not paths.completion.exists()

    private_config.write_bytes(paths.resolved_config.read_bytes())
    result = supervise_prepared_attempt(manifest=manifest, preflight=preflight)
    assert result.attempt_id == ATTEMPT_1
    assert captured["argv"] == ("uv", "run", "--no-sync", "rl", "@", preflight.private_config_path)
    assert "process_runner" not in inspect.signature(supervise_prepared_attempt).parameters
    assert "process_runner" not in inspect.signature(run_training_attempt).parameters


def test_supervisor_cancellation_forwards_to_rl_process_group_without_attestation(
    tmp_path, exact_config_validator, monkeypatch
):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    captured = _install_fake_popen(
        monkeypatch,
        tmp_path,
        return_code=-signal.SIGTERM,
        cancel_signal=signal.SIGTERM,
    )
    with pytest.raises(InterruptedError, match="cancelled by signal"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)
    assert captured["signals"] == [(456, signal.SIGTERM)]
    assert not paths.completion.exists()
    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()


def _make_hf_snapshot_cache(run_root: Path, external_target: Path) -> None:
    import huggingface_hub
    from huggingface_hub.file_download import _create_symlink

    assert huggingface_hub.__version__ == "1.16.1"
    cache = run_root / "model-cache/models--hf-internal-testing--tiny-random-gpt2"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots/1234567890abcdef"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "config-digest").write_text('{"model_type":"gpt2"}')
    (blobs / "weights-digest").write_bytes(b"locked model weights")
    _create_symlink(str(blobs / "config-digest"), str(snapshot / "config.json"))
    _create_symlink(str(blobs / "weights-digest"), str(snapshot / "model.safetensors"))
    _create_symlink(str(external_target), str(snapshot / "external-tokenizer.json"))
    assert all(path.is_symlink() for path in snapshot.iterdir())


def test_private_cleanup_unlinks_hf_snapshot_symlinks_without_following_targets(tmp_path):
    external_target = tmp_path / "external-tokenizer.json"
    external_target.write_text("must survive")
    run_root = create_private_run_root()
    token = capture_private_run_root(run_root)
    _make_hf_snapshot_cache(run_root, external_target)

    remove_private_run_root(token)

    assert not run_root.exists()
    assert external_target.read_text() == "must survive"


def test_private_cleanup_rejects_symlink_root_and_preserves_external_target(tmp_path):
    external_target = tmp_path / "external"
    external_target.mkdir()
    (external_target / "keep").write_text("must survive")
    run_root = create_private_run_root()
    token = capture_private_run_root(run_root)
    displaced = run_root.with_name(f"{run_root.name}-displaced")
    run_root.rename(displaced)
    run_root.symlink_to(external_target, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="no longer a directory"):
            remove_private_run_root(token)
        assert (external_target / "keep").read_text() == "must survive"
        assert displaced.is_dir()
    finally:
        run_root.unlink(missing_ok=True)
        displaced.rmdir()


def test_private_cleanup_rejects_root_swap_during_removal(tmp_path, monkeypatch):
    external_target = tmp_path / "external"
    external_target.mkdir()
    (external_target / "keep").write_text("must survive")
    run_root = create_private_run_root()
    token = capture_private_run_root(run_root)
    (run_root / "owned").write_text("remove only this")
    displaced = run_root.with_name(f"{run_root.name}-displaced")

    def swap_root(path):
        path.rename(displaced)
        path.symlink_to(external_target, target_is_directory=True)

    monkeypatch.setattr(supervisor_module, "_before_private_root_remove", swap_root)
    try:
        with pytest.raises(RuntimeError, match="changed before removal"):
            remove_private_run_root(token)
        assert (external_target / "keep").read_text() == "must survive"
        assert not (displaced / "owned").exists()
    finally:
        run_root.unlink(missing_ok=True)
        displaced.rmdir()


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (SupervisorCancelled(signal.SIGTERM), "cancelled by signal"),
        (RuntimeError("materialization failed"), "materialization failed"),
    ],
)
def test_training_failure_and_cancellation_remove_hf_private_cache(
    tmp_path,
    monkeypatch,
    error,
    match,
):
    manifest_path = tmp_path / "manifest.json"
    _manifest().save(manifest_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_target = tmp_path / "external-tokenizer.json"
    external_target.write_text("must survive")
    captured = {}
    monkeypatch.setattr(supervisor_module, "generate_attempt_id", lambda: ATTEMPT_1)
    monkeypatch.setattr(supervisor_module, "validate_manifest_contract", lambda *_args, **_kwargs: None)

    def fail_prepare(**kwargs):
        captured["run_root"] = kwargs["run_root"]
        _make_hf_snapshot_cache(kwargs["run_root"], external_target)
        raise error

    monkeypatch.setattr(supervisor_module, "prepare_training_attempt", fail_prepare)
    with pytest.raises(type(error), match=match):
        run_training_attempt(
            manifest_path=manifest_path,
            source_config_path=tmp_path / "unused.toml",
            artifact_output_dir=output_dir,
        )

    assert not captured["run_root"].exists()
    assert external_target.read_text() == "must survive"


def test_cleanup_diagnostic_does_not_invalidate_durable_success(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest_path = tmp_path / "manifest.json"
    _manifest().save(manifest_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_target = tmp_path / "external"
    external_target.mkdir()
    sentinel = object()
    captured = {}
    monkeypatch.setattr(supervisor_module, "generate_attempt_id", lambda: ATTEMPT_1)
    monkeypatch.setattr(supervisor_module, "validate_manifest_contract", lambda *_args, **_kwargs: None)

    def prepare(**kwargs):
        captured["run_root"] = kwargs["run_root"]
        return object()

    def complete(**_kwargs):
        (output_dir / "training-result.json").write_text("{}")
        return sentinel

    def swap_root(path):
        displaced = path.with_name(f"{path.name}-displaced")
        captured["displaced"] = displaced
        path.rename(displaced)
        path.symlink_to(external_target, target_is_directory=True)

    monkeypatch.setattr(supervisor_module, "prepare_training_attempt", prepare)
    monkeypatch.setattr(supervisor_module, "_supervise_prepared_attempt", complete)
    monkeypatch.setattr(supervisor_module, "_before_private_root_remove", swap_root)
    monkeypatch.setattr(TrainingResult, "load", classmethod(lambda _cls, _path: sentinel))
    try:
        result = run_training_attempt(
            manifest_path=manifest_path,
            source_config_path=tmp_path / "unused.toml",
            artifact_output_dir=output_dir,
        )
        assert result is sentinel
        assert "cleanup failed after durable success" in capsys.readouterr().err
    finally:
        captured["run_root"].unlink(missing_ok=True)
        captured["displaced"].rmdir()


@pytest.mark.parametrize(
    ("phase", "signum", "expected_exit"),
    [
        ("checkpoint", signal.SIGINT, 130),
        ("publication", signal.SIGTERM, 143),
    ],
)
def test_supervisor_post_wait_cancellation_aborts_entire_transaction(
    tmp_path,
    exact_config_validator,
    monkeypatch,
    phase,
    signum,
    expected_exit,
):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    captured = _install_fake_popen(monkeypatch, tmp_path, return_code=0)

    if phase == "checkpoint":
        original_select = supervisor_module.select_final_adapter

        def cancel_during_checkpoint(*args, **kwargs):
            signal.raise_signal(signum)
            return original_select(*args, **kwargs)

        monkeypatch.setattr(supervisor_module, "select_final_adapter", cancel_during_checkpoint)
    else:
        original_rename = artifacts_module._rename_noreplace

        def cancel_during_publication(source, destination):
            original_rename(source, destination)
            if Path(destination).name == "final-adapter":
                signal.raise_signal(signum)

        monkeypatch.setattr(artifacts_module, "_rename_noreplace", cancel_during_publication)

    with pytest.raises(SupervisorCancelled) as cancellation:
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)

    assert cancellation.value.exit_code == expected_exit
    assert captured["signals"] == []
    assert not paths.completion_staging.exists()
    assert not paths.publication.exists()
    assert not paths.publication_staging.exists()
    assert not (tmp_path / "training-result.json").exists()
    if phase == "publication":
        assert paths.completion.is_file()
        assert (tmp_path / "final-adapter").is_dir()
        monkeypatch.setattr(artifacts_module, "_rename_noreplace", original_rename)
        recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
        assert recovered.attempt_id == ATTEMPT_1
    else:
        assert not paths.completion.exists()
        assert not (tmp_path / "final-adapter").exists()


def test_later_cancelled_attempt_preserves_identical_adapter_owned_by_earlier_attempt(
    tmp_path,
    exact_config_validator,
    monkeypatch,
):
    manifest = _manifest()
    exact_config = exact_config_validator(manifest)
    _, first_paths = _evidence(tmp_path, manifest, exact_config, attempt_id=ATTEMPT_1)
    publish_training_result(**_publication_kwargs(tmp_path, manifest, ATTEMPT_1))
    (tmp_path / "training-result.json").unlink()
    adapter_inode = (tmp_path / "final-adapter").stat().st_ino
    first_completion = first_paths.completion.read_bytes()
    first_publication = first_paths.publication.read_bytes()
    second_preflight, second_paths = _evidence(
        tmp_path,
        manifest,
        exact_config,
        attempt_id=ATTEMPT_2,
        write_completion=False,
    )
    _install_fake_popen(monkeypatch, tmp_path, return_code=0)

    def cancel_before_reusing_prior_adapter(**_kwargs):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(supervisor_module, "publish_training_result", cancel_before_reusing_prior_adapter)

    with pytest.raises(SupervisorCancelled):
        supervise_prepared_attempt(manifest=manifest, preflight=second_preflight)

    assert (tmp_path / "final-adapter").stat().st_ino == adapter_inode
    assert first_paths.completion.read_bytes() == first_completion
    assert first_paths.publication.read_bytes() == first_publication
    assert not second_paths.completion.exists()
    completion_quarantines = _completion_final_quarantines(second_paths)
    assert len(completion_quarantines) == 1
    assert (
        TrainingCompletionAttestation.model_validate_json(completion_quarantines[0].read_bytes()).attempt_id
        == ATTEMPT_2
    )
    assert not second_paths.publication.exists()


def test_adapter_install_detects_destination_swap_and_preserves_conflict(
    tmp_path,
    exact_config_validator,
    monkeypatch,
):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    original_rename = artifacts_module._rename_noreplace
    displaced = tmp_path / "displaced-installed-adapter"

    def swap_after_adapter_install(source, destination):
        original_rename(source, destination)
        if Path(destination).name == "final-adapter":
            Path(destination).rename(displaced)
            Path(destination).mkdir()
            (Path(destination) / "adapter_config.json").write_text('{"r": 16}')
            (Path(destination) / "adapter_model.safetensors").write_bytes(b"conflicting adapter")

    monkeypatch.setattr(artifacts_module, "_rename_noreplace", swap_after_adapter_install)

    with pytest.raises(RuntimeError, match="inode does not match"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)

    assert displaced.is_dir()
    assert (tmp_path / "final-adapter").is_dir()
    assert paths.completion.is_file()
    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()
    monkeypatch.setattr(artifacts_module, "_rename_noreplace", original_rename)
    with pytest.raises(ValueError, match="does not match"):
        publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)


def test_cancellation_after_publication_transfers_adapter_ownership_to_durable_evidence(
    tmp_path,
    exact_config_validator,
    monkeypatch,
):
    manifest = _manifest()
    preflight, paths = _evidence(
        tmp_path,
        manifest,
        exact_config_validator(manifest),
        write_completion=False,
    )
    _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    original_rename = artifacts_module._rename_noreplace

    def cancel_after_publication_install(source, destination):
        original_rename(source, destination)
        if Path(destination) == paths.publication:
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(artifacts_module, "_rename_noreplace", cancel_after_publication_install)

    with pytest.raises(SupervisorCancelled):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)

    assert paths.completion.is_file()
    assert paths.publication.is_file()
    assert (tmp_path / "final-adapter").is_dir()
    assert not (tmp_path / "training-result.json").exists()
    monkeypatch.setattr(artifacts_module, "_rename_noreplace", original_rename)
    recovered = publish_training_result(**_publication_kwargs(tmp_path, manifest), recovery=True)
    assert recovered.attempt_id == ATTEMPT_1


@pytest.mark.parametrize("replacement_kind", ["directory", "final-adapter-alias"])
def test_publication_staging_swap_is_quarantined_without_deleting_evidence(
    tmp_path,
    exact_config_validator,
    monkeypatch,
    replacement_kind,
):
    manifest = _manifest()
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    kwargs = _publication_kwargs(tmp_path, manifest)
    original_result = publish_training_result(**kwargs)
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    existing_quarantines = set(_publication_quarantines(paths))
    staging = paths.publication_staging
    staging.mkdir()
    (staging / "original").write_text("original staging")
    displaced = paths.directory / "displaced-publication-stage"
    final_adapter = tmp_path / "final-adapter"
    adapter_identity = (final_adapter.stat().st_dev, final_adapter.stat().st_ino)
    replacement_identity = None

    def swap_publication_stage(path):
        nonlocal replacement_identity
        assert path == staging
        staging.rename(displaced)
        if replacement_kind == "directory":
            staging.mkdir()
            (staging / "replacement").write_text("replacement staging")
        else:
            staging.symlink_to(final_adapter, target_is_directory=True)
        metadata = staging.lstat()
        replacement_identity = (metadata.st_dev, metadata.st_ino)

    monkeypatch.setattr(artifacts_module, "_before_staging_quarantine", swap_publication_stage)

    with pytest.raises(RuntimeError, match="inode does not match"):
        publish_training_result(**kwargs)

    assert displaced.is_dir()
    assert (displaced / "original").read_text() == "original staging"
    assert (final_adapter.stat().st_dev, final_adapter.stat().st_ino) == adapter_identity
    new_quarantines = set(_publication_quarantines(paths)) - existing_quarantines
    assert len(new_quarantines) == 1
    quarantine = new_quarantines.pop()
    assert (quarantine.lstat().st_dev, quarantine.lstat().st_ino) == replacement_identity
    if replacement_kind == "directory":
        assert (quarantine / "replacement").read_text() == "replacement staging"
    else:
        assert quarantine.is_symlink()
        assert quarantine.resolve() == final_adapter.resolve()
    assert TrainingResult.load(tmp_path / "training-result.json") == original_result

    monkeypatch.setattr(artifacts_module, "_before_staging_quarantine", lambda _path: None)
    recovered = publish_training_result(**kwargs, recovery=True)
    assert recovered == original_result


@pytest.mark.parametrize(("signum", "expected_exit"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_supervisor_cli_maps_cancellation_to_conventional_exit(monkeypatch, signum, expected_exit):
    def cancel_run(**_kwargs):
        raise SupervisorCancelled(signum)

    monkeypatch.setattr(supervisor_module, "run_training_attempt", cancel_run)
    status = supervisor_main(
        [
            "run",
            "--manifest",
            "manifest.json",
            "--config",
            "train.toml",
            "--output-dir",
            "output",
        ]
    )
    assert status == expected_exit


def test_recovery_cancellation_during_completion_promotion_stops_publication(
    tmp_path,
    exact_config_validator,
    monkeypatch,
):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    _evidence(tmp_path, manifest, exact_config_validator(manifest))
    paths = attempt_paths(tmp_path, ATTEMPT_1, require_existing=True)
    paths.completion.rename(paths.completion_staging)
    original_rename = artifacts_module._rename_noreplace

    def cancel_during_completion_promotion(source, destination):
        if Path(source) == paths.completion_staging:
            signal.raise_signal(signal.SIGTERM)
        original_rename(source, destination)

    monkeypatch.setattr(artifacts_module, "_rename_noreplace", cancel_during_completion_promotion)

    with pytest.raises(SupervisorCancelled) as cancellation:
        recover_publish(
            manifest_path=manifest_path,
            artifact_output_dir=tmp_path,
            attempt_id=ATTEMPT_1,
        )

    assert cancellation.value.exit_code == 143
    assert paths.completion.is_file()
    assert not paths.completion_staging.exists()
    assert not paths.publication.exists()
    assert not (tmp_path / "training-result.json").exists()
    assert not (tmp_path / "final-adapter").exists()


def test_training_wrapper_backgrounds_waits_and_propagates_supervisor_status():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    launch = script.index("tau.eval_tools.live.training_supervisor_live run")
    background = script.index('--output-dir "$TAU_OUTPUT_DIR" &', launch)
    capture_pid = script.index("CHILD_PID=$!", background)
    wait = script.index('if wait "$CHILD_PID"; then', capture_pid)
    clear_pid = script.index('CHILD_PID=""', wait)
    propagate = script.index('exit "$supervisor_status"', clear_pid)
    assert launch < background < capture_pid < wait < clear_pid < propagate
    assert "trap 'forward_signal_and_exit INT 130' INT" in script
    assert "trap 'forward_signal_and_exit TERM 143' TERM" in script


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
    captured = _install_fake_popen(monkeypatch, tmp_path, return_code=0)
    with pytest.raises(FileExistsError, match="refusing to rerun"):
        supervise_prepared_attempt(manifest=manifest, preflight=preflight)
    assert "argv" not in captured

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
        model_path=Path(model),
        dataset_path=Path(dataset),
        output_dir=Path(attempt_output),
        logical_output_dir=Path("/data/train"),
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


def test_real_resolved_config_round_trip_preflight_publication_and_post_handoff(tmp_path):
    source_config = Path(__file__).parents[3] / "configs/tau/math-7b-h200/train.toml"
    output_dir = tmp_path / "train"
    output_dir.mkdir()
    create_attempt_directory(output_dir, ATTEMPT_1)
    paths = attempt_paths(output_dir, ATTEMPT_1, require_existing=True)
    run_root = tmp_path / "private"
    run_root.mkdir()
    model_path = run_root / "model"
    dataset_path = run_root / "training-dataset"
    model_path.mkdir()
    dataset_path.mkdir()
    private_config_path = run_root / "resolved-train.toml"

    _, resolved_bytes, identity = resolve_effective_rl_config(
        source_config,
        _manifest(),
        source_config_rel="configs/tau/math-7b-h200/train.toml",
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=paths.run_output,
        logical_output_dir=Path(RL_CONFIG.output_dir),
        max_steps=50,
    )
    manifest = _manifest().model_copy(update={"rl_config": identity})
    private_config_path.write_bytes(resolved_bytes)
    paths.resolved_config.write_bytes(resolved_bytes)

    _, parsed_bytes, parsed_identity = validate_resolved_rl_config(
        paths.resolved_config,
        manifest,
        source_config_rel="configs/tau/math-7b-h200/train.toml",
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=paths.run_output,
        logical_output_dir=Path(RL_CONFIG.output_dir),
        max_steps=50,
    )
    assert parsed_bytes == resolved_bytes
    assert parsed_identity == identity

    preflight = write_training_preflight(
        output_path=paths.preflight,
        attempt_id=ATTEMPT_1,
        manifest=manifest,
        artifact_output_dir=output_dir,
        attempt_output_dir=paths.run_output,
        run_root=run_root,
        model_path=model_path,
        dataset_path=dataset_path,
        private_config_path=private_config_path,
        resolved_config_path=paths.resolved_config,
    )
    source_adapter = _adapter(paths.run_output, 50)
    _, preflight_sha256 = load_json_with_sha256(paths.preflight)
    stable = source_adapter.parent / "STABLE"
    completion = TrainingCompletionAttestation(
        attempt_id=ATTEMPT_1,
        manifest_identity_hash=manifest.identity_hash(),
        preflight_sha256=preflight_sha256,
        resolved_config_sha256=hashlib.sha256(resolved_bytes).hexdigest(),
        rl_pid=123,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:01:00+00:00",
        command_argv=["uv", "run", "--no-sync", "rl", "@", str(private_config_path)],
        executable="/usr/bin/uv",
        rl_config=identity,
        run_root=str(run_root),
        private_config_path=str(private_config_path),
        resolved_config_path=str(paths.resolved_config),
        artifact_output_dir=str(output_dir),
        attempt_output_dir=str(paths.run_output),
        source_step=50,
        stable_marker_path=str(stable),
        stable_marker_sha256=hashlib.sha256(stable.read_bytes()).hexdigest(),
        source_adapter_path=str(source_adapter),
        source_adapter_files=build_file_manifest(source_adapter),
    )
    _write_json_staged_noreplace(
        final_path=paths.completion,
        staging_path=paths.completion_staging,
        payload=completion.model_dump(),
    )
    result = publish_training_result(**_publication_kwargs(output_dir, manifest))
    assert preflight.rl_config == parsed_identity
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        training_output_dir=output_dir,
        expected_adapter_path=output_dir / "final-adapter",
        expected_step=50,
        expected_rank=16,
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
