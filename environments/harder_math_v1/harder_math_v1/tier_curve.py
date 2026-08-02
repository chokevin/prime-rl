from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from harder_math_v1.partition import TIERS, Tier

TIER_CURVE_SCHEMA = "tier-curve.v1"
HARDNESS_GAP = 0.15

_IDENTITY_FIELDS = (
    "package_name",
    "package_version",
    "package_commit",
    "catalog_digest",
    "source_manifest_digest",
    "source_revisions",
    "model",
    "model_revision",
    "decoding",
    "renderer",
    "grader",
    "taskset_config",
)
_RECORD_FIELDS = (
    "record_id",
    "content_sha256",
    "partition",
    "tier",
    "reward",
    "failure",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity(observation: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _IDENTITY_FIELDS if field not in observation]
    if missing:
        raise ValueError(f"tier-curve observation is missing identity fields: {missing}")
    return {field: observation[field] for field in _IDENTITY_FIELDS}


def _validate_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        raise ValueError("cannot calculate a confidence interval for an empty tier")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_tier_curve(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("tier-curve aggregation needs at least one observation")

    identity = _identity(observations[0])
    identity_digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    _validate_digest("catalog_digest", identity["catalog_digest"])
    _validate_digest("source_manifest_digest", identity["source_manifest_digest"])

    ordered_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for observation in observations:
        if _identity(observation) != identity:
            raise ValueError("tier-curve observations mix run settings or manifests")
        missing = [field for field in _RECORD_FIELDS if field not in observation]
        if missing:
            raise ValueError(f"tier-curve observation is missing record fields: {missing}")
        if observation["tier"] not in TIERS:
            raise ValueError(f"unsupported tier {observation['tier']!r}")
        if observation["partition"] != "eval":
            raise ValueError("tier-curve observations must use the eval partition")
        reward = observation["reward"]
        if reward not in (0, 0.0, 1, 1.0):
            raise ValueError(f"tier-curve reward must be binary, got {reward!r}")
        record_id = observation["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise TypeError("tier-curve record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError(f"duplicate tier-curve record id {record_id!r}")
        seen_ids.add(record_id)
        _validate_digest("content_sha256", observation["content_sha256"])
        failure = observation["failure"]
        if failure is not None and not isinstance(failure, str):
            raise TypeError("tier-curve failure must be a string or null")
        ordered_records.append(
            {
                "record_id": record_id,
                "content_sha256": observation["content_sha256"],
                "partition": "eval",
                "tier": observation["tier"],
                "reward": float(reward),
                "failure": failure,
            }
        )
    ordered_records.sort(key=lambda record: record["record_id"])

    summaries: dict[Tier, dict[str, Any]] = {}
    for tier in TIERS:
        tier_records = [record for record in ordered_records if record["tier"] == tier]
        if not tier_records:
            raise ValueError(f"tier-curve is missing tier {tier!r}")
        successes = sum(int(record["reward"]) for record in tier_records)
        lower, upper = _wilson_interval(successes, len(tier_records))
        summaries[tier] = {
            "count": len(tier_records),
            "mean": successes / len(tier_records),
            "confidence_interval_95": [lower, upper],
            "failures": sum(record["failure"] is not None for record in tier_records),
        }

    base_minus_hard = summaries["base"]["mean"] - summaries["hard"]["mean"]
    return {
        "schema_version": TIER_CURVE_SCHEMA,
        **identity,
        "settings_digest": identity_digest,
        "ordered_records": ordered_records,
        "tiers": summaries,
        "base_minus_hard": base_minus_hard,
        "hardness_gate": {
            "threshold": HARDNESS_GAP,
            "passed": base_minus_hard >= HARDNESS_GAP,
        },
    }


def validate_tier_curve_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != TIER_CURVE_SCHEMA:
        raise ValueError(f"tier-curve schema must be {TIER_CURVE_SCHEMA!r}")
    records = artifact.get("ordered_records")
    if not isinstance(records, list):
        raise TypeError("tier-curve ordered_records must be a list")
    observations = [
        {
            **{field: artifact[field] for field in _IDENTITY_FIELDS},
            **record,
        }
        for record in records
    ]
    expected = aggregate_tier_curve(observations)
    if artifact != expected:
        raise ValueError("tier-curve artifact summaries or settings digest are inconsistent")


def passes_hardness_gate(
    artifact: Mapping[str, Any],
    minimum_gap: float = HARDNESS_GAP,
) -> bool:
    validate_tier_curve_artifact(artifact)
    return artifact["base_minus_hard"] >= minimum_gap
