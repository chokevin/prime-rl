import json
from pathlib import Path

import pytest

from tau.eval_tools.manifest import GIT_SHA_RE, SHA256_RE
from tau.eval_tools.output_paths import (
    RECOVERY_OUTPUT_LEAF,
    RECOVERY_SOURCE_REVISION,
    evidence_generation,
)
from tau.eval_tools.recovery import (
    F12_ADAPTER_CONFIG_SHA256,
    F12_ADAPTER_MODEL_SHA256,
    F12_ADAPTER_WEIGHT_NAME,
    F12_BASELINE_MEAN,
    F12_BASELINE_REWARDS_FILE_SHA256,
    F12_EXAMPLE_COUNT,
    F12_FINAL_ADAPTER_AGGREGATE_SHA256,
    F12_MANIFEST_FILE_SHA256,
    F12_MANIFEST_IDENTITY_HASH,
    F12_MODEL_FILES_AGGREGATE_SHA256,
    F12_SOURCE_CONFIG_SHA256,
    F12_TRAINING_ATTEMPT_ID,
    F12_TRAINING_DATA_DIGEST,
    F12_TRAINING_RESULT_FILE_SHA256,
    F12_TRAINING_SOURCE_STEP,
    RecoveryContract,
    RecoveryContractError,
    _validate_runtime_source_revision,
    write_recovery_provenance,
)

RUNTIME_SOURCE = "c" * 40
FROZEN_MANIFEST_IDENTITY = "1" * 64
FROZEN_BASELINE_SHA = "2" * 64
POST_REWARDS_SHA = "3" * 64


def test_pinned_f12_digests_are_wellformed():
    digests = [
        F12_MANIFEST_FILE_SHA256,
        F12_MANIFEST_IDENTITY_HASH,
        F12_BASELINE_REWARDS_FILE_SHA256,
        F12_TRAINING_RESULT_FILE_SHA256,
        F12_FINAL_ADAPTER_AGGREGATE_SHA256,
        F12_ADAPTER_CONFIG_SHA256,
        F12_ADAPTER_MODEL_SHA256,
        F12_MODEL_FILES_AGGREGATE_SHA256,
        F12_TRAINING_DATA_DIGEST,
        F12_SOURCE_CONFIG_SHA256,
    ]
    for digest in digests:
        assert SHA256_RE.fullmatch(digest), digest
    assert len(set(digests)) == len(digests)
    assert GIT_SHA_RE.fullmatch(RECOVERY_SOURCE_REVISION)
    assert F12_TRAINING_SOURCE_STEP == 200
    assert F12_EXAMPLE_COUNT == 500
    assert F12_BASELINE_MEAN == 0.738
    assert F12_ADAPTER_WEIGHT_NAME == "adapter_model.safetensors"


def test_runtime_source_must_be_full_sha_distinct_from_frozen():
    assert _validate_runtime_source_revision(RUNTIME_SOURCE) == RUNTIME_SOURCE
    with pytest.raises(RecoveryContractError, match="40-character"):
        _validate_runtime_source_revision("d" * 39)
    with pytest.raises(RecoveryContractError, match="distinct from the frozen F12 source"):
        _validate_runtime_source_revision(RECOVERY_SOURCE_REVISION)


def _contract() -> RecoveryContract:
    return RecoveryContract(
        frozen_source_revision=RECOVERY_SOURCE_REVISION,
        manifest_path=Path("/data/frozen/manifest/frozen-eval-manifest.json"),
        baseline_rewards_path=Path("/data/frozen/eval-baseline/rewards.json"),
        training_result_path=Path("/data/frozen/train/training-result.json"),
        training_output_dir=Path("/data/frozen/train"),
        lora_adapter_path=Path("/data/frozen/train/final-adapter"),
        manifest_identity_hash=FROZEN_MANIFEST_IDENTITY,
        evaluation_identity_hash="4" * 64,
        baseline_rewards_sha256=FROZEN_BASELINE_SHA,
        training_result_sha256=F12_TRAINING_RESULT_FILE_SHA256,
        adapter_aggregate_sha256=F12_FINAL_ADAPTER_AGGREGATE_SHA256,
        training_attempt_id=F12_TRAINING_ATTEMPT_ID,
        source_step=F12_TRAINING_SOURCE_STEP,
    )


def _comparison_payload(**overrides) -> dict:
    payload = {
        "n": F12_EXAMPLE_COUNT,
        "baseline_mean": F12_BASELINE_MEAN,
        "post_mean": 0.80,
        "delta": 0.062,
        "ci_lower": 0.031,
        "ci_upper": 0.093,
        "n_bootstrap": 10_000,
        "bootstrap_seed": 0,
        "ci_alpha": 0.05,
        "min_delta": 0.03,
        "manifest_identity_hash": FROZEN_MANIFEST_IDENTITY,
        "baseline_rewards_sha256": FROZEN_BASELINE_SHA,
        "post_rewards_sha256": POST_REWARDS_SHA,
        "passed": True,
    }
    payload.update(overrides)
    return payload


def _recovery_output(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    runtime = evidence_generation(RUNTIME_SOURCE, data_root=data_root)
    output_dir = runtime.root / RECOVERY_OUTPUT_LEAF
    output_dir.mkdir(parents=True)
    return output_dir


def test_write_recovery_provenance_binds_both_sources(tmp_path):
    data_root = tmp_path / "data"
    output_dir = _recovery_output(tmp_path)
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(_comparison_payload()))

    provenance_path = write_recovery_provenance(
        output_dir=output_dir,
        runtime_source_revision=RUNTIME_SOURCE,
        comparison_path=comparison_path,
        contract=_contract(),
        data_root=data_root,
    )
    assert provenance_path == output_dir / "recovery-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["runtime_source_revision"] == RUNTIME_SOURCE
    assert provenance["frozen_f12_source_revision"] == RECOVERY_SOURCE_REVISION
    assert provenance["recovery_output_leaf"] == RECOVERY_OUTPUT_LEAF
    assert provenance["comparison"]["passed"] is True
    assert provenance["frozen_inputs"]["source_step"] == F12_TRAINING_SOURCE_STEP


def test_write_recovery_provenance_rejects_output_outside_recovery_leaf(tmp_path):
    data_root = tmp_path / "data"
    stray = tmp_path / "stray"
    stray.mkdir()
    comparison_path = stray / "comparison.json"
    comparison_path.write_text(json.dumps(_comparison_payload()))

    with pytest.raises(RecoveryContractError, match="recovery leaf"):
        write_recovery_provenance(
            output_dir=stray,
            runtime_source_revision=RUNTIME_SOURCE,
            comparison_path=comparison_path,
            contract=_contract(),
            data_root=data_root,
        )


def test_write_recovery_provenance_rejects_comparison_identity_mismatch(tmp_path):
    data_root = tmp_path / "data"
    output_dir = _recovery_output(tmp_path)
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(_comparison_payload(manifest_identity_hash="9" * 64)))

    with pytest.raises(RecoveryContractError, match="manifest identity"):
        write_recovery_provenance(
            output_dir=output_dir,
            runtime_source_revision=RUNTIME_SOURCE,
            comparison_path=comparison_path,
            contract=_contract(),
            data_root=data_root,
        )
