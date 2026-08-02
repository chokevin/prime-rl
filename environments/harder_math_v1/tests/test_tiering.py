import pytest
from harder_math_v1.partition import tier_for_level


@pytest.mark.parametrize(
    ("level", "tier"),
    [
        ("Level 1", "base"),
        ("Level 2", "base"),
        ("Level 3", "core"),
        ("Level 4", "core"),
        ("Level 5", "hard"),
    ],
)
def test_math_level_to_tier(level: str, tier: str) -> None:
    assert tier_for_level(level) == tier


def test_tier_mapping_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unsupported MATH level"):
        tier_for_level("Level 6")
