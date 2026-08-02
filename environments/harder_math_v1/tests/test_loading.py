import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from harder_math_v1.catalog import (
    build_catalog_manifest,
    canonical_text,
    extract_gold,
    load_source_manifest,
    normalize_aime_row,
    normalize_math_row,
    source_manifest_from_dict,
)
from harder_math_v1.loader import load_verified_catalogs


def test_row_normalization_and_gold_extraction(source_manifest) -> None:
    source = source_manifest.sources[0]
    record = normalize_math_row(
        {
            "problem": "  Compute Ａ.\r\n",
            "level": "Level 2",
            "type": "Algebra",
            "solution": "First \\boxed{wrong}; finally \\boxed{\\frac{1}{2}}.  ",
        },
        source=source,
        config="algebra",
        upstream_split="train",
        partition="train",
    )

    assert record.question == "Compute A."
    assert record.prompt.endswith("Compute A.")
    assert record.gold == "\\frac{1}{2}"
    assert record.tier == "base"
    assert record.record_id.startswith(f"fixture_math@{'a' * 40}:algebra:train:")
    assert len(record.content_sha256) == 64


def test_canonical_text_normalizes_nfkc_lf_and_edge_whitespace() -> None:
    assert canonical_text("  Ａ \r\n B\t\r") == "A\nB"


def test_gold_extraction_rejects_missing_or_unclosed_box() -> None:
    with pytest.raises(ValueError, match="no closed boxed answer"):
        extract_gold("the answer is 7")
    with pytest.raises(ValueError, match="no closed boxed answer"):
        extract_gold("the answer is \\boxed{7")

    assert extract_gold("therefore \\boxed 7$.") == "7"


def test_fixture_loader_validates_counts_and_builds_manifest(
    source_manifest,
    rows_by_split: Mapping[str, list[dict[str, Any]]],
) -> None:
    def load_dataset(
        dataset: str,
        config: str,
        *,
        split: str,
        revision: str,
    ) -> list[dict[str, Any]]:
        assert (dataset, config, revision) == ("fixture/math", "algebra", "a" * 40)
        return rows_by_split[split]

    catalogs = load_verified_catalogs(source_manifest, load_dataset)
    artifact = build_catalog_manifest(
        (*catalogs["train"], *catalogs["eval"]),
        source_manifest,
    )

    assert len(catalogs["train"]) == 3
    assert len(catalogs["eval"]) == 3
    assert artifact["counts"] == {
        "total": 6,
        "partitions": {"train": 3, "eval": 3},
        "tiers": {"base": 2, "core": 2, "hard": 2},
    }
    assert len(artifact["catalog_digest"]) == 64


def test_loader_rejects_schema_and_count_drift(
    source_manifest,
    rows_by_split: Mapping[str, list[dict[str, Any]]],
) -> None:
    def short_loader(*args: Any, split: str, **kwargs: Any) -> list[dict[str, Any]]:
        return rows_by_split[split][:-1]

    with pytest.raises(ValueError, match="count drift"):
        load_verified_catalogs(source_manifest, short_loader)

    source = source_manifest.sources[0]
    with pytest.raises(ValueError, match="schema drift"):
        normalize_math_row(
            {
                "problem": "P",
                "level": "Level 1",
                "solution": "\\boxed{1}",
            },
            source=source,
            config="algebra",
            upstream_split="train",
            partition="train",
        )


def test_packaged_manifest_has_verified_source_pins_licenses_and_exclusions() -> None:
    manifest = load_source_manifest()
    sources = {source.id: source for source in manifest.sources}
    source = sources["hendrycks_math"]

    assert source.dataset == "EleutherAI/hendrycks_math"
    assert source.revision == "21a5633873b6a120296cce3e2df9d5550074f4a3"
    assert sum(config.counts["train"] for config in source.configs) == 7500
    assert sum(config.counts["test"] for config in source.configs) == 5000
    assert len(source.excluded_rows) == 5
    assert "aime_2024" not in sources
    assert sources["aime_2025"].revision == "c94da77eb22bbd6439e62a323bec18493a421302"
    assert sources["aime_2025"].license == "CC-BY-NC-SA-4.0"
    assert [entry["id"] for entry in manifest.blocked_sources] == ["aime_2024"]
    assert Path(__file__).parents[1].joinpath("harder_math_v1/schemas/tier_curve.v1.json").is_file()


def test_active_source_requires_verified_license(source_manifest) -> None:
    raw = copy.deepcopy(source_manifest.raw)
    raw["sources"][0]["license"] = "No license declared"

    with pytest.raises(ValueError, match="has unverified license"):
        source_manifest_from_dict(raw)


def test_aime_fixture_normalization_is_eval_only_hard() -> None:
    source = {source.id: source for source in load_source_manifest().sources}["aime_2025"]
    record = normalize_aime_row(
        {
            "problem_idx": 1,
            "problem": "Compute 1 + 1.",
            "answer": 2,
            "problem_type": ["algebra"],
        },
        source=source,
        config="default",
        upstream_split="train",
        partition="eval",
    )

    assert record.gold == "2"
    assert record.tier == "hard"
    assert record.partition == "eval"


def test_manifest_pinned_exclusion_rejects_source_drift(
    source_manifest,
    rows_by_split: Mapping[str, list[dict[str, Any]]],
) -> None:
    raw = copy.deepcopy(source_manifest.raw)
    source = source_manifest.sources[0]
    excluded = normalize_math_row(
        rows_by_split["train"][0],
        source=source,
        config="algebra",
        upstream_split="train",
        partition="train",
    )
    raw["sources"][0]["excluded_rows"] = [
        {
            "config": "algebra",
            "upstream_split": "train",
            "upstream_index": 0,
            "source_sha256": excluded.source_sha256,
            "reason": "Fixture exclusion.",
        }
    ]
    manifest = source_manifest_from_dict(raw)

    def load_dataset(
        dataset: str,
        config: str,
        *,
        split: str,
        revision: str,
    ) -> list[dict[str, Any]]:
        return rows_by_split[split]

    catalogs = load_verified_catalogs(manifest, load_dataset)
    assert len(catalogs["train"]) == 2

    tampered = copy.deepcopy(rows_by_split)
    tampered["train"][0]["problem"] = "Changed upstream problem"

    def tampered_loader(
        dataset: str,
        config: str,
        *,
        split: str,
        revision: str,
    ) -> list[dict[str, Any]]:
        return tampered[split]

    with pytest.raises(ValueError, match="exclusion .* source drift"):
        load_verified_catalogs(manifest, tampered_loader)
