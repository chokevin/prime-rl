from pathlib import Path

import pytest
from pydantic import ValidationError

from tau.eval_tools.artifacts import ADAPTER_CONFIG, TrainingResult
from tau.eval_tools.f12_recovery import (
    F12_EXPERIMENT_SOURCE_REVISION,
    F12_LORA_RANK,
    F12_SOURCE_STEP,
    F12RecoveryInputDigests,
    F12RecoveryPreflight,
    run_f12_recovery_preflight,
    validate_f12_recovery_preflight,
    write_f12_recovery_preflight,
)
from tau.eval_tools.json_io import load_json_with_sha256
from tau.eval_tools.manifest import FileManifest, FileRecord, RLConfigIdentity, build_file_manifest

RUNTIME_SOURCE_REVISION = "b" * 40
MANIFEST_SHA256 = "1" * 64
BASELINE_SHA256 = "2" * 64
TRAINING_RESULT_SHA256 = "3" * 64
MANIFEST_IDENTITY = "4" * 64
ADAPTER_AGGREGATE_SEED = "5" * 64


def _private_adapter_files(config_sha256: str = "6" * 64, weight_sha256: str = "7" * 64) -> FileManifest:
    return FileManifest.from_records(
        [
            FileRecord(path=ADAPTER_CONFIG, size=11, sha256=config_sha256),
            FileRecord(path="adapter_model.safetensors", size=22, sha256=weight_sha256),
        ]
    )


def _preflight(**overrides) -> F12RecoveryPreflight:
    private_adapter_files = overrides.pop("private_adapter_files", _private_adapter_files())
    fields = {
        "recovery_runtime_source_revision": RUNTIME_SOURCE_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "baseline_rewards_sha256": BASELINE_SHA256,
        "training_result_sha256": TRAINING_RESULT_SHA256,
        "manifest_identity_hash": MANIFEST_IDENTITY,
        "adapter_aggregate_sha256": private_adapter_files.aggregate_sha256,
        "private_adapter_files": private_adapter_files,
    }
    fields.update(overrides)
    return F12RecoveryPreflight(**fields)


def _training_result(**overrides) -> TrainingResult:
    private_adapter_files = overrides.pop("adapter_files", _private_adapter_files())
    fields = {
        "attempt_id": "20260101T000000Z-0123456789abcdef",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_revision": RUNTIME_SOURCE_REVISION,
        "manifest_identity_hash": MANIFEST_IDENTITY,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "d" * 40,
        "model_files_sha256": "8" * 64,
        "training_data_digest": "9" * 64,
        "rl_config": RLConfigIdentity(
            source_config_rel="configs/tau/math-7b-h200/train-f12.toml",
            source_toml_sha256="a" * 64,
            output_dir="/data/train",
            max_steps=F12_SOURCE_STEP,
            canonical_resolved_sha256="b" * 64,
        ),
        "preflight_sha256": "c" * 64,
        "completion_attestation_sha256": "d" * 64,
        "publication_evidence_sha256": "e" * 64,
        "resolved_config_sha256": "f" * 64,
        "resolved_config_path": "/data/train/resolved-train.toml",
        "source_step": F12_SOURCE_STEP,
        "source_adapter_path": "/data/train/weights/step_200/lora_adapters",
        "final_adapter_path": "/data/train/final-adapter",
        "adapter_sha256": private_adapter_files.aggregate_sha256,
        "adapter_files": private_adapter_files,
        "lora_rank": F12_LORA_RANK,
    }
    fields.update(overrides)
    return TrainingResult(**fields)


# -- F12RecoveryPreflight shape validation -----------------------------------


def test_f12_recovery_preflight_accepts_valid_shape():
    preflight = _preflight()
    assert preflight.schema_version == 1
    assert preflight.status == "verified"
    assert preflight.frozen_experiment_source_revision == F12_EXPERIMENT_SOURCE_REVISION
    assert preflight.source_step == F12_SOURCE_STEP
    assert preflight.lora_rank == F12_LORA_RANK


def test_f12_recovery_preflight_rejects_reusing_experiment_source_as_runtime_source():
    with pytest.raises(ValueError, match="new full lowercase"):
        _preflight(recovery_runtime_source_revision=F12_EXPERIMENT_SOURCE_REVISION)


@pytest.mark.parametrize(
    "field",
    ["manifest_sha256", "baseline_rewards_sha256", "training_result_sha256", "manifest_identity_hash"],
)
def test_f12_recovery_preflight_rejects_non_hex_digest(field):
    with pytest.raises(ValidationError, match="lowercase SHA-256 hex"):
        _preflight(**{field: "not-a-digest"})


def test_f12_recovery_preflight_rejects_adapter_aggregate_mismatch_with_own_manifest():
    with pytest.raises(ValidationError, match="does not match its own private_adapter_files"):
        _preflight(adapter_aggregate_sha256="0" * 64)


def test_f12_recovery_preflight_rejects_private_adapter_missing_config():
    files = FileManifest.from_records(
        [
            FileRecord(path="adapter_model.safetensors", size=22, sha256="7" * 64),
            FileRecord(path="other.safetensors", size=22, sha256="8" * 64),
        ]
    )
    with pytest.raises(ValidationError, match="config plus one weight"):
        _preflight(private_adapter_files=files, adapter_aggregate_sha256=files.aggregate_sha256)


def test_f12_recovery_preflight_rejects_private_adapter_extra_entry():
    files = FileManifest.from_records(
        [
            FileRecord(path=ADAPTER_CONFIG, size=11, sha256="6" * 64),
            FileRecord(path="adapter_model.safetensors", size=22, sha256="7" * 64),
            FileRecord(path="extra.bin", size=1, sha256="9" * 64),
        ]
    )
    with pytest.raises(ValidationError, match="config plus one weight"):
        _preflight(private_adapter_files=files, adapter_aggregate_sha256=files.aggregate_sha256)


def test_f12_recovery_preflight_rejects_private_adapter_unsupported_weight_name():
    files = FileManifest.from_records(
        [
            FileRecord(path=ADAPTER_CONFIG, size=11, sha256="6" * 64),
            FileRecord(path="unsupported_weight.bin", size=22, sha256="7" * 64),
        ]
    )
    with pytest.raises(ValidationError, match="missing its supported weight file"):
        _preflight(private_adapter_files=files, adapter_aggregate_sha256=files.aggregate_sha256)


def test_f12_recovery_preflight_rejects_wrong_source_step():
    with pytest.raises(ValidationError):
        _preflight(source_step=201)


def test_f12_recovery_preflight_rejects_wrong_lora_rank():
    with pytest.raises(ValidationError):
        _preflight(lora_rank=32)


# -- write_f12_recovery_preflight ---------------------------------------------


def test_write_f12_recovery_preflight_publishes_loadable_exclusive_json(tmp_path):
    output_path = tmp_path / "recovery-preflight.json"
    private_adapter_files = _private_adapter_files()
    preflight = write_f12_recovery_preflight(
        output_path=output_path,
        recovery_runtime_source_revision=RUNTIME_SOURCE_REVISION,
        manifest_sha256=MANIFEST_SHA256,
        baseline_rewards_sha256=BASELINE_SHA256,
        training_result_sha256=TRAINING_RESULT_SHA256,
        manifest_identity_hash=MANIFEST_IDENTITY,
        adapter_aggregate_sha256=private_adapter_files.aggregate_sha256,
        source_step=F12_SOURCE_STEP,
        lora_rank=F12_LORA_RANK,
        private_adapter_files=private_adapter_files,
    )
    assert output_path.exists()
    payload, _ = load_json_with_sha256(output_path)
    assert F12RecoveryPreflight.model_validate(payload) == preflight
    with pytest.raises(FileExistsError):
        write_f12_recovery_preflight(
            output_path=output_path,
            recovery_runtime_source_revision=RUNTIME_SOURCE_REVISION,
            manifest_sha256=MANIFEST_SHA256,
            baseline_rewards_sha256=BASELINE_SHA256,
            training_result_sha256=TRAINING_RESULT_SHA256,
            manifest_identity_hash=MANIFEST_IDENTITY,
            adapter_aggregate_sha256=private_adapter_files.aggregate_sha256,
            source_step=F12_SOURCE_STEP,
            lora_rank=F12_LORA_RANK,
            private_adapter_files=private_adapter_files,
        )


# -- validate_f12_recovery_preflight ------------------------------------------


def _written_preflight(tmp_path: Path, **overrides):
    output_path = tmp_path / "recovery-preflight.json"
    preflight = _preflight(**overrides)
    from tau.eval_tools.json_io import write_json_exclusive

    write_json_exclusive(output_path, preflight.model_dump())
    _, sha256 = load_json_with_sha256(output_path)
    return output_path, sha256


def _matching_digests() -> F12RecoveryInputDigests:
    return F12RecoveryInputDigests(
        manifest_sha256=MANIFEST_SHA256,
        baseline_rewards_sha256=BASELINE_SHA256,
        training_result_sha256=TRAINING_RESULT_SHA256,
    )


def test_validate_f12_recovery_preflight_accepts_matching_evidence(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    preflight = validate_f12_recovery_preflight(
        path,
        expected_sha256=sha256,
        runtime_source_revision=RUNTIME_SOURCE_REVISION,
        digests=_matching_digests(),
        manifest_identity_hash=MANIFEST_IDENTITY,
        result=result,
    )
    assert preflight.recovery_runtime_source_revision == RUNTIME_SOURCE_REVISION


def test_validate_f12_recovery_preflight_rejects_digest_mismatch(tmp_path):
    result = _training_result()
    path, _ = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="digest is"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256="0" * 64,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=_matching_digests(),
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_runtime_source_mismatch(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="runtime source does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision="c" * 40,
            digests=_matching_digests(),
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_manifest_digest_mismatch(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    digests = F12RecoveryInputDigests(
        manifest_sha256="0" * 64,
        baseline_rewards_sha256=BASELINE_SHA256,
        training_result_sha256=TRAINING_RESULT_SHA256,
    )
    with pytest.raises(ValueError, match="manifest digest does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=digests,
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_baseline_digest_mismatch(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    digests = F12RecoveryInputDigests(
        manifest_sha256=MANIFEST_SHA256,
        baseline_rewards_sha256="0" * 64,
        training_result_sha256=TRAINING_RESULT_SHA256,
    )
    with pytest.raises(ValueError, match="baseline digest does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=digests,
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_training_result_digest_mismatch(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    digests = F12RecoveryInputDigests(
        manifest_sha256=MANIFEST_SHA256,
        baseline_rewards_sha256=BASELINE_SHA256,
        training_result_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="training result digest does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=digests,
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_manifest_identity_mismatch(tmp_path):
    result = _training_result()
    path, sha256 = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="manifest identity does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=_matching_digests(),
            manifest_identity_hash="0" * 64,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_adapter_aggregate_mismatch_with_result(tmp_path):
    result = _training_result(adapter_files=_private_adapter_files(weight_sha256="f" * 64))
    path, sha256 = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="adapter aggregate does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=_matching_digests(),
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_source_step_mismatch(tmp_path):
    result = _training_result(source_step=100)
    path, sha256 = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="source step does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=_matching_digests(),
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


def test_validate_f12_recovery_preflight_rejects_lora_rank_mismatch(tmp_path):
    result = _training_result(lora_rank=8)
    path, sha256 = _written_preflight(tmp_path)
    with pytest.raises(ValueError, match="LoRA rank does not match"):
        validate_f12_recovery_preflight(
            path,
            expected_sha256=sha256,
            runtime_source_revision=RUNTIME_SOURCE_REVISION,
            digests=_matching_digests(),
            manifest_identity_hash=MANIFEST_IDENTITY,
            result=result,
        )


# Note: preflight.private_adapter_files and preflight.adapter_aggregate_sha256 are
# always kept mutually consistent by F12RecoveryPreflight's own validator, so any
# content difference from result.adapter_files is already caught by the
# adapter-aggregate check above (short of an infeasible SHA-256 collision). The
# direct manifest-equality check in validate_f12_recovery_preflight is intentional
# defense-in-depth, matching TrainingResult's own redundant adapter_sha256/
# adapter_files.aggregate_sha256 pairing.


# -- run_f12_recovery_preflight orchestration --------------------------------


def test_run_f12_recovery_preflight_wires_validate_materialize_and_publish(tmp_path, monkeypatch):
    private_adapter_dir = tmp_path / "private-adapter"
    private_adapter_dir.mkdir()
    (private_adapter_dir / ADAPTER_CONFIG).write_bytes(b'{"r": 16}')
    (private_adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    # The private destination's real on-disk bytes must match what the signed
    # TrainingResult declares, exactly as copy_adapter_exclusive_by_manifest
    # guarantees in production.
    result = _training_result(adapter_files=build_file_manifest(private_adapter_dir))
    manifest_identity_calls = []

    class _FakeManifest:
        def identity_hash(self):
            manifest_identity_calls.append(True)
            return MANIFEST_IDENTITY

    def fake_validate_inputs(**_kwargs):
        return _FakeManifest(), object(), result, _matching_digests()

    materialize_calls = []

    def fake_materialize(*, result, durable_adapter_path, run_root):
        materialize_calls.append((result, durable_adapter_path, run_root))
        return private_adapter_dir

    monkeypatch.setattr("tau.eval_tools.f12_recovery.validate_f12_recovery_inputs", fake_validate_inputs)
    monkeypatch.setattr("tau.eval_tools.f12_recovery.materialize_adapter_for_eval", fake_materialize)

    output_path = tmp_path / "recovery-preflight.json"
    preflight = run_f12_recovery_preflight(
        manifest_path=tmp_path / "manifest.json",
        baseline_path=tmp_path / "baseline.json",
        training_result_path=tmp_path / "training-result.json",
        training_output_dir=tmp_path / "train",
        adapter_path=tmp_path / "durable-adapter",
        runtime_source_revision=RUNTIME_SOURCE_REVISION,
        run_root=tmp_path / "private",
        output_path=output_path,
    )

    assert output_path.exists()
    assert materialize_calls == [(result, tmp_path / "durable-adapter", tmp_path / "private")]
    assert manifest_identity_calls
    assert preflight.recovery_runtime_source_revision == RUNTIME_SOURCE_REVISION
    assert preflight.manifest_identity_hash == MANIFEST_IDENTITY
    assert preflight.adapter_aggregate_sha256 == result.adapter_sha256
    assert preflight.private_adapter_files == result.adapter_files
