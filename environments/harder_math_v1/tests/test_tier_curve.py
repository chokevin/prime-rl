import copy

import pytest
from harder_math_v1.tier_curve import (
    aggregate_tier_curve,
    passes_hardness_gate,
    validate_tier_curve_artifact,
)


def observation(record_id: str, tier: str, reward: float) -> dict:
    return {
        "package_name": "harder-math-v1",
        "package_version": "0.1.0",
        "package_commit": "deadbeef",
        "catalog_digest": "a" * 64,
        "source_manifest_digest": "b" * 64,
        "source_revisions": {"hendrycks_math": "c" * 40},
        "model": "org/model",
        "model_revision": "model@revision",
        "decoding": {"temperature": 0.0, "max_tokens": 4096},
        "renderer": {"id": "default"},
        "grader": {"id": "verify_boxed_math_answer", "timeout_seconds": 5},
        "taskset_config": {"partition": "eval"},
        "record_id": record_id,
        "content_sha256": record_id[0] * 64,
        "partition": "eval",
        "tier": tier,
        "reward": reward,
        "failure": None,
    }


def test_tier_curve_aggregation_validation_and_acceptance_gate() -> None:
    artifact = aggregate_tier_curve(
        [
            observation("d-base-1", "base", 1.0),
            observation("e-base-2", "base", 1.0),
            observation("f-core-1", "core", 1.0),
            observation("1-core-2", "core", 0.0),
            observation("2-hard-1", "hard", 0.0),
            observation("3-hard-2", "hard", 0.0),
        ]
    )

    validate_tier_curve_artifact(artifact)
    assert artifact["tiers"]["base"]["mean"] == 1.0
    assert artifact["tiers"]["core"]["mean"] == 0.5
    assert artifact["tiers"]["hard"]["mean"] == 0.0
    assert artifact["base_minus_hard"] == 1.0
    assert artifact["hardness_gate"] == {"threshold": 0.15, "passed": True}
    assert passes_hardness_gate(artifact)


def test_tier_curve_rejects_mixed_manifest_or_settings() -> None:
    observations = [
        observation("d-base", "base", 1.0),
        observation("e-core", "core", 0.0),
        observation("f-hard", "hard", 0.0),
    ]
    mixed_manifest = copy.deepcopy(observations)
    mixed_manifest[-1]["catalog_digest"] = "9" * 64
    with pytest.raises(ValueError, match="mix run settings or manifests"):
        aggregate_tier_curve(mixed_manifest)

    mixed_settings = copy.deepcopy(observations)
    mixed_settings[-1]["decoding"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="mix run settings or manifests"):
        aggregate_tier_curve(mixed_settings)


def test_tier_curve_rejects_non_binary_and_missing_tier() -> None:
    non_binary = [
        observation("d-base", "base", 1.0),
        observation("e-core", "core", 0.5),
        observation("f-hard", "hard", 0.0),
    ]
    with pytest.raises(ValueError, match="reward must be binary"):
        aggregate_tier_curve(non_binary)

    with pytest.raises(ValueError, match="missing tier"):
        aggregate_tier_curve(
            [
                observation("d-base", "base", 1.0),
                observation("e-hard", "hard", 0.0),
            ]
        )
