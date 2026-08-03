from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harder_math_v1.catalog import (
    EXPECTED_CATALOG_CONTRACTS,
    CatalogContract,
    CatalogRecord,
    validate_catalog_records_contract,
)
from harder_math_v1.partition import TIERS, Tier

TIER_CURVE_SCHEMA = "tier-curve.v1"
HARDNESS_GAP = 0.15

_RUN_IDENTITY_FIELDS = (
    "baseline_id",
    "package_name",
    "package_version",
    "package_commit",
    "model",
    "model_revision",
    "decoding",
    "renderer",
    "grader",
)
_CATALOG_IDENTITY_FIELDS = (
    "catalog_digest",
    "source_manifest_digest",
    "source_revisions",
)
_RECORD_FIELDS = (
    "record_id",
    "content_sha256",
    "prompt_sha256",
    "partition",
    "tier",
    "reward",
    "failure",
)


@dataclass(frozen=True, slots=True)
class TierCurveRunContract:
    baseline_id: str
    package_name: str
    package_version: str
    package_commit: str
    model: str
    model_revision: str
    decoding: Mapping[str, Any]
    renderer: Mapping[str, Any]
    grader: Mapping[str, Any]
    taskset_config: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in _RUN_IDENTITY_FIELDS}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _validate_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_run_contract(contract: TierCurveRunContract) -> None:
    for field in (
        "baseline_id",
        "package_name",
        "package_version",
        "package_commit",
        "model",
        "model_revision",
    ):
        value = getattr(contract, field)
        if not isinstance(value, str) or not value:
            raise TypeError(f"tier-curve run contract {field} must be a non-empty string")
    if contract.package_name != "harder-math-v1":
        raise ValueError("tier-curve run contract package_name must be 'harder-math-v1'")
    for field in ("decoding", "renderer", "grader", "taskset_config"):
        if not isinstance(getattr(contract, field), Mapping):
            raise TypeError(f"tier-curve run contract {field} must be an object")
    if "tier" in contract.taskset_config:
        raise ValueError("tier-curve run contract taskset_config must omit the per-run tier")
    if contract.taskset_config.get("id") != "harder-math-v1":
        raise ValueError("tier-curve run contract taskset_config.id must be 'harder-math-v1'")
    if contract.taskset_config.get("partition") != "eval":
        raise ValueError("tier-curve run contract taskset_config.partition must be 'eval'")


def _catalog_identity(contract: CatalogContract) -> dict[str, Any]:
    return {
        "catalog_digest": contract.catalog_digest,
        "source_manifest_digest": contract.source_manifest_digest,
        "source_revisions": dict(contract.source_revisions),
    }


def _validate_threshold(minimum_gap: float) -> float:
    if isinstance(minimum_gap, bool) or not isinstance(minimum_gap, int | float):
        raise TypeError("tier-curve minimum gap must be a number")
    threshold = float(minimum_gap)
    if not HARDNESS_GAP <= threshold <= 1.0:
        raise ValueError(f"tier-curve minimum gap must be between {HARDNESS_GAP} and 1.0")
    return threshold


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        raise ValueError("cannot calculate a confidence interval for an empty tier")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_tier_curve(
    observations: Sequence[Mapping[str, Any]],
    eval_catalog: Sequence[CatalogRecord],
    run_contract: TierCurveRunContract,
    *,
    expected_catalog_contract: CatalogContract = EXPECTED_CATALOG_CONTRACTS["eval"],
    minimum_gap: float = HARDNESS_GAP,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("tier-curve aggregation needs at least one observation")
    _validate_run_contract(run_contract)
    threshold = _validate_threshold(minimum_gap)
    trusted_catalog = validate_catalog_records_contract(
        eval_catalog,
        partition="eval",
        expected=expected_catalog_contract,
    )
    expected_records = {record.record_id: record for record in trusted_catalog}
    expected_identity = {
        **run_contract.identity(),
        **_catalog_identity(expected_catalog_contract),
    }

    ordered_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    taskset_configs: dict[Tier, dict[str, Any]] = {}
    for observation in observations:
        missing_identity = [
            field
            for field in (*_RUN_IDENTITY_FIELDS, *_CATALOG_IDENTITY_FIELDS, "taskset_config")
            if field not in observation
        ]
        if missing_identity:
            raise ValueError(f"tier-curve observation is missing identity fields: {missing_identity}")
        actual_identity = {field: observation[field] for field in (*_RUN_IDENTITY_FIELDS, *_CATALOG_IDENTITY_FIELDS)}
        if actual_identity != expected_identity:
            raise ValueError("tier-curve observation does not match the trusted run or catalog identity")

        missing_record = [field for field in _RECORD_FIELDS if field not in observation]
        if missing_record:
            raise ValueError(f"tier-curve observation is missing record fields: {missing_record}")
        tier = observation["tier"]
        if tier not in TIERS:
            raise ValueError(f"unsupported tier {tier!r}")
        if observation["partition"] != "eval":
            raise ValueError("tier-curve observations must use the eval partition")

        taskset_config = observation["taskset_config"]
        if not isinstance(taskset_config, Mapping):
            raise TypeError("tier-curve observation taskset_config must be an object")
        if taskset_config.get("tier") != tier:
            raise ValueError("tier-curve observation taskset_config.tier must match its record tier")
        common_taskset_config = {key: value for key, value in taskset_config.items() if key != "tier"}
        if common_taskset_config != dict(run_contract.taskset_config):
            raise ValueError("tier-curve observation changes taskset settings other than tier")
        tier_config = dict(taskset_config)
        if previous := taskset_configs.get(tier):
            if previous != tier_config:
                raise ValueError(f"tier-curve observations mix taskset configs within tier {tier!r}")
        taskset_configs[tier] = tier_config

        reward = observation["reward"]
        if reward not in (0, 0.0, 1, 1.0):
            raise ValueError(f"tier-curve reward must be binary, got {reward!r}")
        record_id = observation["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise TypeError("tier-curve record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError(f"duplicate tier-curve record id {record_id!r}")
        expected_record = expected_records.get(record_id)
        if expected_record is None:
            raise ValueError(f"tier-curve record id {record_id!r} is not in the trusted eval catalog")
        seen_ids.add(record_id)

        _validate_digest("content_sha256", observation["content_sha256"])
        _validate_digest("prompt_sha256", observation["prompt_sha256"])
        if observation["content_sha256"] != expected_record.content_sha256:
            raise ValueError(f"tier-curve content hash mismatch for {record_id!r}")
        if observation["prompt_sha256"] != expected_record.prompt_sha256:
            raise ValueError(f"tier-curve prompt hash mismatch for {record_id!r}")
        if tier != expected_record.tier:
            raise ValueError(f"tier-curve tier mismatch for {record_id!r}")

        failure = observation["failure"]
        if failure is not None and not isinstance(failure, str):
            raise TypeError("tier-curve failure must be a string or null")
        ordered_records.append(
            {
                "record_id": record_id,
                "content_sha256": observation["content_sha256"],
                "prompt_sha256": observation["prompt_sha256"],
                "partition": "eval",
                "tier": tier,
                "reward": float(reward),
                "failure": failure,
            }
        )

    if missing_ids := sorted(set(expected_records) - seen_ids):
        raise ValueError(
            f"tier-curve observations are missing {len(missing_ids)} trusted eval records; first: {missing_ids[:3]}"
        )
    if set(taskset_configs) != set(TIERS):
        raise ValueError(f"tier-curve taskset configs must cover exactly {list(TIERS)}")
    ordered_records.sort(key=lambda record: record["record_id"])

    summaries: dict[Tier, dict[str, Any]] = {}
    for tier in TIERS:
        tier_records = [record for record in ordered_records if record["tier"] == tier]
        successes = sum(int(record["reward"]) for record in tier_records)
        lower, upper = _wilson_interval(successes, len(tier_records))
        summaries[tier] = {
            "count": len(tier_records),
            "mean": successes / len(tier_records),
            "confidence_interval_95": [lower, upper],
            "failures": sum(record["failure"] is not None for record in tier_records),
        }

    serialized_taskset_configs = {tier: taskset_configs[tier] for tier in TIERS}
    artifact_identity = {
        **expected_identity,
        "taskset_configs": serialized_taskset_configs,
    }
    base_minus_hard = summaries["base"]["mean"] - summaries["hard"]["mean"]
    return {
        "schema_version": TIER_CURVE_SCHEMA,
        **artifact_identity,
        "settings_digest": hashlib.sha256(_canonical_json(artifact_identity)).hexdigest(),
        "ordered_records": ordered_records,
        "tiers": summaries,
        "base_minus_hard": base_minus_hard,
        "hardness_gate": {
            "threshold": threshold,
            "passed": base_minus_hard >= threshold,
        },
    }


def validate_tier_curve_artifact(
    artifact: Mapping[str, Any],
    eval_catalog: Sequence[CatalogRecord],
    run_contract: TierCurveRunContract,
    *,
    expected_catalog_contract: CatalogContract = EXPECTED_CATALOG_CONTRACTS["eval"],
) -> None:
    if artifact.get("schema_version") != TIER_CURVE_SCHEMA:
        raise ValueError(f"tier-curve schema must be {TIER_CURVE_SCHEMA!r}")
    records = artifact.get("ordered_records")
    if not isinstance(records, list):
        raise TypeError("tier-curve ordered_records must be a list")
    taskset_configs = artifact.get("taskset_configs")
    if not isinstance(taskset_configs, Mapping) or set(taskset_configs) != set(TIERS):
        raise ValueError(f"tier-curve taskset_configs must cover exactly {list(TIERS)}")
    hardness_gate = artifact.get("hardness_gate")
    if not isinstance(hardness_gate, Mapping) or "threshold" not in hardness_gate:
        raise ValueError("tier-curve hardness_gate must include its threshold")
    observations = [
        {
            **{field: artifact[field] for field in (*_RUN_IDENTITY_FIELDS, *_CATALOG_IDENTITY_FIELDS)},
            "taskset_config": taskset_configs[record["tier"]],
            **record,
        }
        for record in records
    ]
    expected = aggregate_tier_curve(
        observations,
        eval_catalog,
        run_contract,
        expected_catalog_contract=expected_catalog_contract,
        minimum_gap=hardness_gate["threshold"],
    )
    if artifact != expected:
        raise ValueError("tier-curve artifact summaries or settings digest are inconsistent")


def passes_hardness_gate(
    artifact: Mapping[str, Any],
    eval_catalog: Sequence[CatalogRecord],
    run_contract: TierCurveRunContract,
    *,
    expected_catalog_contract: CatalogContract = EXPECTED_CATALOG_CONTRACTS["eval"],
    minimum_gap: float = HARDNESS_GAP,
) -> bool:
    validate_tier_curve_artifact(
        artifact,
        eval_catalog,
        run_contract,
        expected_catalog_contract=expected_catalog_contract,
    )
    threshold = _validate_threshold(minimum_gap)
    return artifact["base_minus_hard"] >= max(threshold, artifact["hardness_gate"]["threshold"])
