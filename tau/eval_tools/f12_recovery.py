"""Narrow immutable contract for the one approved F12 post-eval recovery."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tau.eval_tools.artifacts import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHT_FILES,
    TrainingResult,
    materialize_adapter_for_eval,
    validate_adapter_handoff,
)
from tau.eval_tools.compare import ComparisonResult, RewardRecord, compare_runs
from tau.eval_tools.json_io import load_json_with_sha256, write_json_exclusive
from tau.eval_tools.manifest import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    FileManifest,
    FrozenEvalManifest,
    build_file_manifest,
    validate_manifest_contract,
)

F12_EXPERIMENT_SOURCE_REVISION = "a603e791776a4579440edc5df5b70309c61cf46a"
F12_VERIFIERS_REVISION = "f646beb37eef51869f886f244456e3d07818e4d6"
F12_TASKSETS_REVISION = "6a2dee6fcd805f41128dba0024f7073afd496169"
F12_MANIFEST_SHA256 = "759bb7ba018f75d3c3dee219e0746e00d2ff0c7531ddc986becac4f909291845"
F12_BASELINE_REWARDS_SHA256 = "8fc09bce4b4ecaf5256712d2b0eb1b72979e8f5a250f86e1e8e310ec3b2ec000"
F12_TRAINING_RESULT_SHA256 = "72b197a353978cff2a8c51aad6e8a45157bef189bac449a00384b3755744a0f3"
F12_MANIFEST_IDENTITY = "2a35089008d5f5b2e0606facafe88b5534c155d1fd95787482aaef4f9cdba673"
F12_ADAPTER_AGGREGATE = "48e6e1fe8dd4039e9fb716cff91101bc463f8198ef499c040e81b90b72af6f9e"
F12_ADAPTER_CONFIG_SHA256 = "f1dcd0c6293492cf407eb63d8c8198360f8859433ad08ca1899dbffd7a8f50b5"
F12_ADAPTER_MODEL_SHA256 = "6bea386c91b51916bef28aa9e841e4a48cdc81ac0ce2c39bbb92239789eb0801"
F12_MODEL_AGGREGATE = "d2d9ab0fbeed7ab74ff3dc433209aec9b01952ccc4d88eec16c0d9aaf1fef9c8"
F12_TRAINING_DATA_DIGEST = "d5924980fe96e2a5ae2de7cd9470c061432adc15f5ac488535cd0f6ac0e8a155"
F12_SOURCE_CONFIG_DIGEST = "5450d2ff141172a4ababc5dfe3df572f2764cba5c6e998d982bb21d270e9e2ff"
F12_TRAINING_ATTEMPT_ID = "20260804T013900Z-0de01384c398f8d3"
F12_SOURCE_STEP = 200
F12_EVAL_INTERVAL = 100
F12_LORA_RANK = 16
F12_BASELINE_MEAN = 0.738
F12_EVAL_N = 500


@dataclass(frozen=True)
class F12RecoveryPaths:
    generation_root: Path
    manifest: Path
    baseline_rewards: Path
    training_output_dir: Path
    training_result: Path
    adapter: Path


def f12_recovery_paths(data_root: Path = Path("/data")) -> F12RecoveryPaths:
    generation_root = (
        data_root / "pretraining-data" / "prime-rl-math-7b-h200" / "generations" / F12_EXPERIMENT_SOURCE_REVISION
    )
    training_output_dir = generation_root / "train"
    return F12RecoveryPaths(
        generation_root=generation_root,
        manifest=generation_root / "manifest" / "frozen-eval-manifest.json",
        baseline_rewards=generation_root / "eval-baseline" / "rewards.json",
        training_output_dir=training_output_dir,
        training_result=training_output_dir / "training-result.json",
        adapter=training_output_dir / "final-adapter",
    )


F12_RECOVERY_ENVIRONMENT = {
    "experiment_source_revision": F12_EXPERIMENT_SOURCE_REVISION,
    "manifest_sha256": F12_MANIFEST_SHA256,
    "baseline_rewards_sha256": F12_BASELINE_REWARDS_SHA256,
    "training_result_sha256": F12_TRAINING_RESULT_SHA256,
    "manifest_identity": F12_MANIFEST_IDENTITY,
    "adapter_aggregate": F12_ADAPTER_AGGREGATE,
    "adapter_config_sha256": F12_ADAPTER_CONFIG_SHA256,
    "adapter_model_sha256": F12_ADAPTER_MODEL_SHA256,
    "model_aggregate": F12_MODEL_AGGREGATE,
    "training_data_digest": F12_TRAINING_DATA_DIGEST,
    "source_config_digest": F12_SOURCE_CONFIG_DIGEST,
    "training_attempt_id": F12_TRAINING_ATTEMPT_ID,
    "source_step": str(F12_SOURCE_STEP),
    "max_steps": str(F12_SOURCE_STEP),
    "eval_interval": str(F12_EVAL_INTERVAL),
}


def validate_f12_recovery_environment(**actual: str | None) -> None:
    if set(actual) != set(F12_RECOVERY_ENVIRONMENT):
        raise ValueError("internal error: incomplete F12 recovery environment contract")
    for name, expected in F12_RECOVERY_ENVIRONMENT.items():
        if actual[name] != expected:
            raise ValueError(f"F12 recovery {name} is {actual[name]!r}, expected {expected!r}")


def validate_recovery_runtime_source(runtime_source_revision: str) -> None:
    if (
        len(runtime_source_revision) != 40
        or any(character not in "0123456789abcdef" for character in runtime_source_revision)
        or runtime_source_revision == F12_EXPERIMENT_SOURCE_REVISION
    ):
        raise ValueError("recovery runtime source must be a new full lowercase 40-character commit SHA")


def _assert_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} SHA-256 is {actual}, expected {expected}")


def _assert_adapter_files(result: TrainingResult) -> None:
    expected = {
        "adapter_config.json": F12_ADAPTER_CONFIG_SHA256,
        "adapter_model.safetensors": F12_ADAPTER_MODEL_SHA256,
    }
    actual = {record.path: record.sha256 for record in result.adapter_files.files}
    if actual != expected:
        raise ValueError(f"F12 adapter file digests are {actual}, expected {expected}")


def validate_f12_recovery_objects(
    *,
    manifest: FrozenEvalManifest,
    manifest_sha256: str,
    baseline: RewardRecord,
    baseline_sha256: str,
    result: TrainingResult,
    training_result_sha256: str,
) -> None:
    validate_manifest_contract(
        manifest,
        expected_source_revision=F12_EXPERIMENT_SOURCE_REVISION,
        expected_verifiers_revision=F12_VERIFIERS_REVISION,
        expected_tasksets_revision=F12_TASKSETS_REVISION,
        expected_model_name=EXPECTED_MODEL_NAME,
        expected_model_revision=EXPECTED_MODEL_REVISION,
        require_finalized=True,
    )
    _assert_digest(manifest_sha256, F12_MANIFEST_SHA256, "F12 frozen manifest")
    if manifest.identity_hash() != F12_MANIFEST_IDENTITY:
        raise ValueError("F12 frozen manifest identity does not match the approved recovery contract")
    if manifest.n != F12_EVAL_N:
        raise ValueError(f"F12 frozen manifest contains {manifest.n} examples, expected {F12_EVAL_N}")
    if manifest.rl_config is None:
        raise ValueError("F12 frozen manifest is missing its RL config identity")
    if manifest.rl_config.source_config_rel != "configs/tau/math-7b-h200/train-f12.toml":
        raise ValueError("F12 frozen manifest does not bind train-f12.toml")
    if manifest.rl_config.source_toml_sha256 != F12_SOURCE_CONFIG_DIGEST:
        raise ValueError("F12 frozen manifest source config digest does not match")
    if manifest.rl_config.max_steps != F12_SOURCE_STEP:
        raise ValueError("F12 frozen manifest max_steps does not match step 200")
    if manifest.model.file_manifest.aggregate_sha256 != F12_MODEL_AGGREGATE:
        raise ValueError("F12 frozen manifest model aggregate does not match")
    if manifest.training_data.record_digest != F12_TRAINING_DATA_DIGEST:
        raise ValueError("F12 frozen manifest training data digest does not match")

    _assert_digest(baseline_sha256, F12_BASELINE_REWARDS_SHA256, "F12 baseline rewards")
    if baseline.model_label != "baseline":
        raise ValueError("F12 baseline rewards are mislabeled")
    if baseline.evaluation_identity_hash != manifest.evaluation_identity_hash():
        raise ValueError("F12 baseline rewards do not bind the frozen evaluation identity")
    if baseline.frozen_manifest_identity_hash is not None:
        raise ValueError("F12 baseline rewards must predate finalization")
    if set(baseline.rewards) != {str(index) for index in range(F12_EVAL_N)}:
        raise ValueError("F12 baseline rewards do not contain all 500 canonical example IDs")
    baseline_mean = math.fsum(baseline.rewards.values()) / F12_EVAL_N
    if baseline_mean != F12_BASELINE_MEAN or manifest.baseline_mean != F12_BASELINE_MEAN:
        raise ValueError("F12 baseline mean does not match the approved 0.738 evidence")
    if manifest.baseline_rewards_sha256 != F12_BASELINE_REWARDS_SHA256:
        raise ValueError("F12 frozen manifest does not bind the approved baseline rewards")

    _assert_digest(training_result_sha256, F12_TRAINING_RESULT_SHA256, "F12 training result")
    if result.source_revision != F12_EXPERIMENT_SOURCE_REVISION:
        raise ValueError("F12 training result source revision does not match")
    if result.attempt_id != F12_TRAINING_ATTEMPT_ID:
        raise ValueError("F12 training result attempt does not match")
    if result.source_step != F12_SOURCE_STEP:
        raise ValueError("F12 training result did not publish step 200")
    if result.lora_rank != F12_LORA_RANK:
        raise ValueError("F12 training result LoRA rank does not match")
    if result.manifest_identity_hash != F12_MANIFEST_IDENTITY:
        raise ValueError("F12 training result manifest identity does not match")
    if result.model_name != EXPECTED_MODEL_NAME or result.model_revision != EXPECTED_MODEL_REVISION:
        raise ValueError("F12 training result model identity does not match")
    if result.model_files_sha256 != F12_MODEL_AGGREGATE:
        raise ValueError("F12 training result model aggregate does not match")
    if result.training_data_digest != F12_TRAINING_DATA_DIGEST:
        raise ValueError("F12 training result data digest does not match")
    if result.rl_config != manifest.rl_config:
        raise ValueError("F12 training result RL config identity does not match the frozen manifest")
    if result.adapter_sha256 != F12_ADAPTER_AGGREGATE:
        raise ValueError("F12 training result adapter aggregate does not match")
    _assert_adapter_files(result)


AdapterHandoffValidator = Callable[..., None]


@dataclass(frozen=True)
class F12RecoveryInputDigests:
    manifest_sha256: str
    baseline_rewards_sha256: str
    training_result_sha256: str


def validate_f12_recovery_inputs(
    *,
    manifest_path: Path,
    baseline_path: Path,
    training_result_path: Path,
    training_output_dir: Path,
    adapter_path: Path,
    handoff_validator: AdapterHandoffValidator = validate_adapter_handoff,
    data_root: Path = Path("/data"),
) -> tuple[FrozenEvalManifest, RewardRecord, TrainingResult, F12RecoveryInputDigests]:
    manifest_payload, manifest_sha256 = load_json_with_sha256(manifest_path)
    manifest = FrozenEvalManifest.model_validate(manifest_payload)
    baseline, baseline_sha256 = RewardRecord.load_with_digest(baseline_path)
    result_payload, training_result_sha256 = load_json_with_sha256(training_result_path)
    result = TrainingResult.model_validate(result_payload)
    validate_f12_recovery_objects(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        baseline=baseline,
        baseline_sha256=baseline_sha256,
        result=result,
        training_result_sha256=training_result_sha256,
    )
    expected_paths = f12_recovery_paths(data_root)
    if (
        manifest_path != expected_paths.manifest
        or baseline_path != expected_paths.baseline_rewards
        or training_result_path != expected_paths.training_result
        or training_output_dir != expected_paths.training_output_dir
        or adapter_path != expected_paths.adapter
    ):
        raise ValueError("F12 recovery input paths do not match the immutable source generation")
    handoff_validator(
        result=result,
        manifest=manifest,
        training_output_dir=training_output_dir.resolve(strict=True),
        expected_adapter_path=adapter_path,
        expected_step=F12_SOURCE_STEP,
        expected_rank=F12_LORA_RANK,
    )
    digests = F12RecoveryInputDigests(
        manifest_sha256=manifest_sha256,
        baseline_rewards_sha256=baseline_sha256,
        training_result_sha256=training_result_sha256,
    )
    return manifest, baseline, result, digests


class F12RecoveryComparisonResult(ComparisonResult):
    model_config = ConfigDict(extra="forbid")

    recovery_contract_version: Literal[1] = 1
    recovery_runtime_source_revision: str
    frozen_experiment_source_revision: Literal["a603e791776a4579440edc5df5b70309c61cf46a"] = (
        F12_EXPERIMENT_SOURCE_REVISION
    )

    @field_validator("recovery_runtime_source_revision")
    @classmethod
    def validate_runtime_source(cls, runtime_source_revision: str) -> str:
        validate_recovery_runtime_source(runtime_source_revision)
        return runtime_source_revision


def compare_f12_recovery(
    *,
    runtime_source_revision: str,
    manifest_path: Path,
    baseline_path: Path,
    post_path: Path,
    training_result_path: Path,
    training_output_dir: Path,
    adapter_path: Path,
    data_root: Path = Path("/data"),
) -> F12RecoveryComparisonResult:
    validate_recovery_runtime_source(runtime_source_revision)
    manifest, baseline, _, _ = validate_f12_recovery_inputs(
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        training_result_path=training_result_path,
        training_output_dir=training_output_dir,
        adapter_path=adapter_path,
        data_root=data_root,
    )
    post, post_sha256 = RewardRecord.load_with_digest(post_path)
    result = compare_runs(
        manifest,
        baseline,
        post,
        baseline_sha256=F12_BASELINE_REWARDS_SHA256,
        post_sha256=post_sha256,
    )
    return F12RecoveryComparisonResult(
        **result.model_dump(),
        recovery_runtime_source_revision=runtime_source_revision,
    )


class F12RecoveryPreflight(BaseModel):
    """Deterministic, immutable evidence that the zero-GPU preflight gate verified
    every F12 recovery input and privately materialized the adapter -- with no
    timestamps or other nondeterministic fields, so identical inputs always produce
    byte-identical JSON that the pin can embed a fixed SHA-256 digest for.

    This model only enforces the fixed contract *shape* (recovery source revision
    format/non-reuse, the always-200/16 step and rank, and the adapter manifest's
    name shape). Whether the individual digest fields actually match the one
    approved F12 recovery evidence is checked separately, against the caller's
    already-verified `TrainingResult`/digests, by `validate_f12_recovery_preflight`
    -- exactly the same separation `F12RecoveryComparisonResult` uses for its own
    digest fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["verified"] = "verified"
    recovery_runtime_source_revision: str
    frozen_experiment_source_revision: Literal["a603e791776a4579440edc5df5b70309c61cf46a"] = (
        F12_EXPERIMENT_SOURCE_REVISION
    )
    manifest_sha256: str
    baseline_rewards_sha256: str
    training_result_sha256: str
    manifest_identity_hash: str
    adapter_aggregate_sha256: str
    source_step: Literal[200] = F12_SOURCE_STEP
    lora_rank: Literal[16] = F12_LORA_RANK
    private_adapter_files: FileManifest

    @field_validator("recovery_runtime_source_revision")
    @classmethod
    def validate_runtime_source(cls, runtime_source_revision: str) -> str:
        validate_recovery_runtime_source(runtime_source_revision)
        return runtime_source_revision

    @field_validator(
        "manifest_sha256",
        "baseline_rewards_sha256",
        "training_result_sha256",
        "manifest_identity_hash",
        "adapter_aggregate_sha256",
    )
    @classmethod
    def validate_sha256(cls, digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("F12 recovery preflight digests must be lowercase SHA-256 hex")
        return digest

    @field_validator("private_adapter_files")
    @classmethod
    def validate_private_adapter_files(cls, manifest: FileManifest) -> FileManifest:
        names = {record.path for record in manifest.files}
        if ADAPTER_CONFIG not in names or len(names) != 2:
            raise ValueError("F12 recovery preflight private adapter manifest must contain config plus one weight")
        if not any(name in names for name in ADAPTER_WEIGHT_FILES):
            raise ValueError("F12 recovery preflight private adapter manifest is missing its supported weight file")
        return manifest

    @model_validator(mode="after")
    def _validate_adapter_aggregate_is_self_consistent(self) -> "F12RecoveryPreflight":
        if self.adapter_aggregate_sha256 != self.private_adapter_files.aggregate_sha256:
            raise ValueError(
                "F12 recovery preflight adapter_aggregate_sha256 does not match its own private_adapter_files"
            )
        return self


def write_f12_recovery_preflight(
    *,
    output_path: Path,
    recovery_runtime_source_revision: str,
    manifest_sha256: str,
    baseline_rewards_sha256: str,
    training_result_sha256: str,
    manifest_identity_hash: str,
    adapter_aggregate_sha256: str,
    source_step: int,
    lora_rank: int,
    private_adapter_files: FileManifest,
) -> F12RecoveryPreflight:
    preflight = F12RecoveryPreflight(
        recovery_runtime_source_revision=recovery_runtime_source_revision,
        manifest_sha256=manifest_sha256,
        baseline_rewards_sha256=baseline_rewards_sha256,
        training_result_sha256=training_result_sha256,
        manifest_identity_hash=manifest_identity_hash,
        adapter_aggregate_sha256=adapter_aggregate_sha256,
        source_step=source_step,
        lora_rank=lora_rank,
        private_adapter_files=private_adapter_files,
    )
    write_json_exclusive(output_path, preflight.model_dump())
    return preflight


def validate_f12_recovery_preflight(
    path: Path,
    *,
    expected_sha256: str,
    runtime_source_revision: str,
    digests: F12RecoveryInputDigests,
    manifest_identity_hash: str,
    result: TrainingResult,
) -> F12RecoveryPreflight:
    """Strictly bind the pin-supplied preflight artifact to this exact H200 run's
    freshly loaded/validated F12 recovery evidence, failing closed if the artifact is
    absent, the digest does not match the pin, or any identity field disagrees."""
    payload, actual_sha256 = load_json_with_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"F12 recovery preflight digest is {actual_sha256}, expected {expected_sha256}")
    preflight = F12RecoveryPreflight.model_validate(payload)
    if preflight.recovery_runtime_source_revision != runtime_source_revision:
        raise ValueError("F12 recovery preflight runtime source does not match this run")
    if preflight.manifest_sha256 != digests.manifest_sha256:
        raise ValueError("F12 recovery preflight manifest digest does not match this run's manifest")
    if preflight.baseline_rewards_sha256 != digests.baseline_rewards_sha256:
        raise ValueError("F12 recovery preflight baseline digest does not match this run's baseline")
    if preflight.training_result_sha256 != digests.training_result_sha256:
        raise ValueError("F12 recovery preflight training result digest does not match this run's training result")
    if preflight.manifest_identity_hash != manifest_identity_hash:
        raise ValueError("F12 recovery preflight manifest identity does not match this run's manifest")
    if preflight.adapter_aggregate_sha256 != result.adapter_sha256:
        raise ValueError("F12 recovery preflight adapter aggregate does not match this run's training result")
    if preflight.source_step != result.source_step:
        raise ValueError("F12 recovery preflight source step does not match this run's training result")
    if preflight.lora_rank != result.lora_rank:
        raise ValueError("F12 recovery preflight LoRA rank does not match this run's training result")
    if preflight.private_adapter_files != result.adapter_files:
        raise ValueError("F12 recovery preflight private adapter manifest does not match this run's signed adapter")
    return preflight


def run_f12_recovery_preflight(
    *,
    manifest_path: Path,
    baseline_path: Path,
    training_result_path: Path,
    training_output_dir: Path,
    adapter_path: Path,
    runtime_source_revision: str,
    run_root: Path,
    output_path: Path,
    data_root: Path = Path("/data"),
) -> F12RecoveryPreflight:
    """The zero-GPU preflight gate's single entry point: run the exact same F12
    recovery input validation and private adapter materialization the H200 job would
    run before inference, then publish a deterministic success artifact."""
    validate_recovery_runtime_source(runtime_source_revision)
    manifest, _, result, digests = validate_f12_recovery_inputs(
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        training_result_path=training_result_path,
        training_output_dir=training_output_dir,
        adapter_path=adapter_path,
        data_root=data_root,
    )
    private_adapter_path = materialize_adapter_for_eval(
        result=result,
        durable_adapter_path=adapter_path,
        run_root=run_root,
    )
    private_adapter_files = build_file_manifest(private_adapter_path)
    return write_f12_recovery_preflight(
        output_path=output_path,
        recovery_runtime_source_revision=runtime_source_revision,
        manifest_sha256=digests.manifest_sha256,
        baseline_rewards_sha256=digests.baseline_rewards_sha256,
        training_result_sha256=digests.training_result_sha256,
        manifest_identity_hash=manifest.identity_hash(),
        adapter_aggregate_sha256=result.adapter_sha256,
        source_step=result.source_step,
        lora_rank=result.lora_rank,
        private_adapter_files=private_adapter_files,
    )
