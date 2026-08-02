from collections.abc import Mapping
from typing import Any

import pytest
from harder_math_v1.catalog import SourceManifest, source_manifest_from_dict


@pytest.fixture
def source_manifest() -> SourceManifest:
    return source_manifest_from_dict(
        {
            "schema_version": "harder-math-source-manifest.v1",
            "sources": [
                {
                    "id": "fixture_math",
                    "kind": "math",
                    "dataset": "fixture/math",
                    "revision": "a" * 40,
                    "license": "MIT",
                    "attribution": "Fixture rows for offline tests.",
                    "fields": ["problem", "level", "type", "solution"],
                    "partitions": {"train": "train", "eval": "test"},
                    "configs": [
                        {
                            "name": "algebra",
                            "counts": {"train": 3, "test": 3},
                        }
                    ],
                    "excluded_rows": [],
                }
            ],
            "blocked_sources": [],
        }
    )


@pytest.fixture
def rows_by_split() -> Mapping[str, list[dict[str, Any]]]:
    return {
        "train": [
            {
                "problem": "Train base",
                "level": "Level 1",
                "type": "Algebra",
                "solution": "Therefore \\boxed{1}.",
            },
            {
                "problem": "Train core",
                "level": "Level 3",
                "type": "Algebra",
                "solution": "Therefore \\boxed{2}.",
            },
            {
                "problem": "Train hard",
                "level": "Level 5",
                "type": "Algebra",
                "solution": "Therefore \\boxed{3}.",
            },
        ],
        "test": [
            {
                "problem": "Eval base",
                "level": "Level 2",
                "type": "Algebra",
                "solution": "Therefore \\boxed{4}.",
            },
            {
                "problem": "Eval core",
                "level": "Level 4",
                "type": "Algebra",
                "solution": "Therefore \\boxed{5}.",
            },
            {
                "problem": "Eval hard",
                "level": "Level 5",
                "type": "Algebra",
                "solution": "Therefore \\boxed{6}.",
            },
        ],
    }
