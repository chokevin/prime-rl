from typing import Literal

Partition = Literal["train", "eval"]
Tier = Literal["base", "core", "hard"]

PARTITIONS: tuple[Partition, ...] = ("train", "eval")
TIERS: tuple[Tier, ...] = ("base", "core", "hard")

_LEVEL_TO_TIER: dict[str, Tier] = {
    "Level 1": "base",
    "Level 2": "base",
    "Level 3": "core",
    "Level 4": "core",
    "Level 5": "hard",
}


def tier_for_level(level: str) -> Tier:
    try:
        return _LEVEL_TO_TIER[level]
    except KeyError:
        raise ValueError(f"unsupported MATH level {level!r}") from None
