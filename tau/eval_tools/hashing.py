"""Deterministic content hashing for prompts/answers.

Used to (a) fingerprint every frozen eval example so a replay can prove it saw the exact
same prompts, and (b) prove zero overlap between the RL-training prompt set and the
held-out eval prompt set (train/eval leakage check).
"""

from __future__ import annotations

import hashlib
import unicodedata


def normalize_text(text: str) -> str:
    """Canonicalize text before hashing so cosmetic differences (line-ending style,
    Unicode normalization form, incidental leading/trailing whitespace) don't produce
    spurious hash mismatches or false negatives on the disjointness check.

    Deliberately does *not* touch internal whitespace, case, or punctuation — those are
    part of the prompt's actual content and a real difference must still hash
    differently.
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def hash_text(text: str) -> str:
    """SHA-256 hex digest of the normalized text. Stable across processes/platforms
    (unlike Python's salted ``hash()``), which is required for a manifest written by
    one job to be replayed and re-verified by another."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_texts(texts: list[str]) -> list[str]:
    return [hash_text(t) for t in texts]
