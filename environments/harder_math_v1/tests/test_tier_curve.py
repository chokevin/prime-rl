import copy
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from harder_math_v1.catalog import (
    CatalogContract,
    CatalogRecord,
    catalog_contract_from_records,
    normalize_math_row,
)
from harder_math_v1.tier_curve import (
    TierCurveRunContract,
    aggregate_tier_curve,
    passes_hardness_gate,
    validate_tier_curve_artifact,
)


@pytest.fixture
def eval_catalog(source_manifest, rows_by_split) -> tuple[CatalogRecord, ...]:
    source = source_manifest.sources[0]
    return tuple(
        normalize_math_row(
            row,
            source=source,
            config="algebra",
            upstream_split="test",
            partition="eval",
        )
        for row in rows_by_split["test"]
    )


@pytest.fixture
def eval_contract(eval_catalog, source_manifest) -> CatalogContract:
    return catalog_contract_from_records(eval_catalog, source_manifest)


@pytest.fixture
def run_contract() -> TierCurveRunContract:
    return TierCurveRunContract(
        baseline_id="fixture-baseline-v1",
        package_name="harder-math-v1",
        package_version="0.1.0",
        package_commit="deadbeef",
        model="org/model",
        model_revision="model@revision",
        decoding={"temperature": 0.0, "max_tokens": 4096},
        renderer={"id": "default"},
        grader={"id": "verify_boxed_math_answer", "timeout_seconds": 5},
        taskset_config={"id": "harder-math-v1", "partition": "eval"},
    )


def observations(
    eval_catalog: tuple[CatalogRecord, ...],
    eval_contract: CatalogContract,
    run_contract: TierCurveRunContract,
) -> list[dict[str, Any]]:
    rewards = {"base": 1.0, "core": 0.0, "hard": 0.0}
    return [
        {
            **run_contract.identity(),
            "catalog_digest": eval_contract.catalog_digest,
            "source_manifest_digest": eval_contract.source_manifest_digest,
            "source_revisions": dict(eval_contract.source_revisions),
            "taskset_config": {
                **run_contract.taskset_config,
                "tier": record.tier,
            },
            "record_id": record.record_id,
            "content_sha256": record.content_sha256,
            "prompt_sha256": record.prompt_sha256,
            "partition": "eval",
            "tier": record.tier,
            "reward": rewards[record.tier],
            "failure": None,
        }
        for record in eval_catalog
    ]


def aggregate_fixture(
    fixture_observations: list[Mapping[str, Any]],
    eval_catalog: tuple[CatalogRecord, ...],
    eval_contract: CatalogContract,
    run_contract: TierCurveRunContract,
    *,
    minimum_gap: float = 0.15,
) -> dict[str, Any]:
    return aggregate_tier_curve(
        fixture_observations,
        eval_catalog,
        run_contract,
        expected_catalog_contract=eval_contract,
        minimum_gap=minimum_gap,
    )


def test_tier_curve_aggregation_validation_and_acceptance_gate(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    artifact = aggregate_fixture(
        observations(eval_catalog, eval_contract, run_contract),
        eval_catalog,
        eval_contract,
        run_contract,
    )

    validate_tier_curve_artifact(
        artifact,
        eval_catalog,
        run_contract,
        expected_catalog_contract=eval_contract,
    )
    assert artifact["tiers"]["base"]["mean"] == 1.0
    assert artifact["tiers"]["core"]["mean"] == 0.0
    assert artifact["tiers"]["hard"]["mean"] == 0.0
    assert artifact["base_minus_hard"] == 1.0
    assert artifact["hardness_gate"] == {"threshold": 0.15, "passed": True}
    assert artifact["taskset_configs"]["base"]["tier"] == "base"
    assert artifact["taskset_configs"]["core"]["tier"] == "core"
    assert artifact["taskset_configs"]["hard"]["tier"] == "hard"
    assert passes_hardness_gate(
        artifact,
        eval_catalog,
        run_contract,
        expected_catalog_contract=eval_contract,
    )


def test_tier_curve_rejects_partial_fabricated_duplicate_and_extra_catalogs(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    valid = observations(eval_catalog, eval_contract, run_contract)

    with pytest.raises(ValueError, match="missing 1 trusted eval records"):
        aggregate_fixture(valid[:-1], eval_catalog, eval_contract, run_contract)

    fabricated = copy.deepcopy(valid)
    fabricated[0]["record_id"] = "fabricated-record"
    with pytest.raises(ValueError, match="is not in the trusted eval catalog"):
        aggregate_fixture(fabricated, eval_catalog, eval_contract, run_contract)

    duplicate = [*valid, copy.deepcopy(valid[0])]
    with pytest.raises(ValueError, match="duplicate tier-curve record id"):
        aggregate_fixture(duplicate, eval_catalog, eval_contract, run_contract)

    extra = [*valid, {**copy.deepcopy(valid[0]), "record_id": "extra-record"}]
    with pytest.raises(ValueError, match="is not in the trusted eval catalog"):
        aggregate_fixture(extra, eval_catalog, eval_contract, run_contract)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("baseline_id", "other-baseline"),
        ("model", "other/model"),
        ("model_revision", "other-revision"),
        ("decoding", {"temperature": 0.7, "max_tokens": 4096}),
        ("catalog_digest", "9" * 64),
    ],
)
def test_tier_curve_rejects_mismatched_identity(
    field,
    replacement,
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    mixed = observations(eval_catalog, eval_contract, run_contract)
    mixed[-1][field] = replacement

    with pytest.raises(ValueError, match="does not match the trusted run or catalog identity"):
        aggregate_fixture(mixed, eval_catalog, eval_contract, run_contract)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("content_sha256", "9" * 64, "content hash mismatch"),
        ("prompt_sha256", "9" * 64, "prompt hash mismatch"),
        ("tier", "base", "tier mismatch"),
    ],
)
def test_tier_curve_rejects_wrong_record_contract(
    field,
    replacement,
    message,
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    mismatched = observations(eval_catalog, eval_contract, run_contract)
    target = next(item for item in mismatched if item["tier"] == "hard")
    target[field] = replacement
    if field == "tier":
        target["taskset_config"]["tier"] = replacement

    with pytest.raises(ValueError, match=message):
        aggregate_fixture(mismatched, eval_catalog, eval_contract, run_contract)


def test_tier_curve_requires_tier_provenance_and_rejects_other_taskset_drift(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    missing_tier = observations(eval_catalog, eval_contract, run_contract)
    missing_tier[0]["taskset_config"].pop("tier")
    with pytest.raises(ValueError, match="taskset_config.tier must match"):
        aggregate_fixture(missing_tier, eval_catalog, eval_contract, run_contract)

    wrong_tier = observations(eval_catalog, eval_contract, run_contract)
    wrong_tier[0]["taskset_config"]["tier"] = "hard"
    with pytest.raises(ValueError, match="taskset_config.tier must match"):
        aggregate_fixture(wrong_tier, eval_catalog, eval_contract, run_contract)

    drift = observations(eval_catalog, eval_contract, run_contract)
    drift[0]["taskset_config"]["untrusted_override"] = True
    with pytest.raises(ValueError, match="changes taskset settings other than tier"):
        aggregate_fixture(drift, eval_catalog, eval_contract, run_contract)


def test_tier_curve_rejects_non_binary_reward_and_lowered_threshold(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    non_binary = observations(eval_catalog, eval_contract, run_contract)
    non_binary[0]["reward"] = 0.5
    with pytest.raises(ValueError, match="reward must be binary"):
        aggregate_fixture(non_binary, eval_catalog, eval_contract, run_contract)

    with pytest.raises(ValueError, match="minimum gap must be between 0.15"):
        aggregate_fixture(
            observations(eval_catalog, eval_contract, run_contract),
            eval_catalog,
            eval_contract,
            run_contract,
            minimum_gap=0.14,
        )


def test_tier_curve_rejects_tampered_artifact(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    artifact = aggregate_fixture(
        observations(eval_catalog, eval_contract, run_contract),
        eval_catalog,
        eval_contract,
        run_contract,
    )
    artifact["ordered_records"][0]["prompt_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_tier_curve_artifact(
            artifact,
            eval_catalog,
            run_contract,
            expected_catalog_contract=eval_contract,
        )

    artifact = aggregate_fixture(
        observations(eval_catalog, eval_contract, run_contract),
        eval_catalog,
        eval_contract,
        run_contract,
    )
    artifact["hardness_gate"]["threshold"] = 0.0
    with pytest.raises(ValueError, match="minimum gap must be between 0.15"):
        validate_tier_curve_artifact(
            artifact,
            eval_catalog,
            run_contract,
            expected_catalog_contract=eval_contract,
        )


def test_tier_curve_rejects_catalog_and_observation_with_matching_fabricated_prompt_hash(
    eval_catalog,
    eval_contract,
    run_contract,
) -> None:
    fabricated_hash = "9" * 64
    tampered_catalog = (
        replace(eval_catalog[0], prompt_sha256=fabricated_hash),
        *eval_catalog[1:],
    )
    tampered_observations = observations(eval_catalog, eval_contract, run_contract)
    tampered_observations[0]["prompt_sha256"] = fabricated_hash

    with pytest.raises(ValueError, match="prompt hash does not match its presented prompt"):
        aggregate_fixture(
            tampered_observations,
            tampered_catalog,
            eval_contract,
            run_contract,
        )
