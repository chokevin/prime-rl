from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf

from harder_math_v1.partition import PARTITIONS, TIERS, Partition, Tier, tier_for_level

SOURCE_MANIFEST_SCHEMA = "harder-math-source-manifest.v1"
CATALOG_MANIFEST_SCHEMA = "harder-math-catalog-manifest.v1"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
SourceKind = Literal["math", "aime"]
EXPECTED_FIELDS: Mapping[SourceKind, frozenset[str]] = {
    "math": frozenset({"problem", "level", "type", "solution"}),
    "aime": frozenset({"problem", "answer"}),
}
VERIFIED_SOURCE_LICENSES = frozenset({"MIT", "CC-BY-NC-SA-4.0"})


@dataclass(frozen=True, slots=True)
class SourceConfig:
    name: str
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    kind: SourceKind
    dataset: str
    revision: str
    license: str
    attribution: str
    fields: frozenset[str]
    partitions: Mapping[Partition, str]
    configs: tuple[SourceConfig, ...]
    excluded_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SourceManifest:
    schema_version: str
    sources: tuple[Source, ...]
    blocked_sources: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    question: str
    prompt: str
    gold: str
    record_id: str
    content_sha256: str
    source_sha256: str
    source: str
    revision: str
    config: str
    upstream_split: str
    partition: Partition
    tier: Tier
    level: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_source_value(value: Any) -> Any:
    if isinstance(value, str):
        return canonical_text(value)
    if isinstance(value, list):
        return [_canonical_source_value(item) for item in value]
    return value


def _source_row_sha256(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    return _sha256(_canonical_json({field: _canonical_source_value(row[field]) for field in sorted(fields)}))


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in normalized.split("\n")).strip()


def source_manifest_digest(manifest: SourceManifest) -> str:
    return _sha256(_canonical_json(manifest.raw))


def source_manifest_from_dict(raw: Mapping[str, Any]) -> SourceManifest:
    if raw.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(
            f"source manifest schema must be {SOURCE_MANIFEST_SCHEMA!r}, got {raw.get('schema_version')!r}"
        )

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source manifest must define at least one active source")

    sources: list[Source] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise TypeError("each source manifest entry must be an object")
        kind = raw_source.get("kind")
        if kind not in EXPECTED_FIELDS:
            raise ValueError(f"source {raw_source.get('id')!r} has unsupported kind {kind!r}")
        fields = raw_source.get("fields")
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            raise TypeError(f"source {raw_source.get('id')!r} fields must be a string list")
        if not EXPECTED_FIELDS[kind].issubset(fields):
            raise ValueError(
                f"source {raw_source.get('id')!r} schema drift: "
                f"expected at least {sorted(EXPECTED_FIELDS[kind])}, got {sorted(fields)}"
            )

        raw_partitions = raw_source.get("partitions")
        expected_partitions = {"train", "eval"} if kind == "math" else {"eval"}
        if not isinstance(raw_partitions, dict) or set(raw_partitions) != expected_partitions:
            raise ValueError(f"source {raw_source.get('id')!r} must map partitions {sorted(expected_partitions)}")
        if not all(isinstance(split, str) and split for split in raw_partitions.values()):
            raise TypeError("source partition split names must be non-empty strings")

        raw_configs = raw_source.get("configs")
        if not isinstance(raw_configs, list) or not raw_configs:
            raise ValueError(f"source {raw_source.get('id')!r} must define configs")
        configs: list[SourceConfig] = []
        for raw_config in raw_configs:
            if not isinstance(raw_config, dict):
                raise TypeError("source configs must be objects")
            name = raw_config.get("name")
            counts = raw_config.get("counts")
            expected_splits = set(raw_partitions.values())
            if not isinstance(name, str) or not name:
                raise TypeError("source config names must be non-empty strings")
            if not isinstance(counts, dict) or set(counts) != expected_splits:
                raise ValueError(f"source config {name!r} counts must cover splits {sorted(expected_splits)}")
            if not all(isinstance(count, int) and count > 0 for count in counts.values()):
                raise ValueError(f"source config {name!r} counts must be positive integers")
            configs.append(SourceConfig(name=name, counts=dict(counts)))

        required_strings = ("id", "dataset", "revision", "license", "attribution")
        for key in required_strings:
            if not isinstance(raw_source.get(key), str) or not raw_source[key]:
                raise TypeError(f"source field {key!r} must be a non-empty string")
        if raw_source["license"] not in VERIFIED_SOURCE_LICENSES:
            raise ValueError(f"active source {raw_source['id']!r} has unverified license {raw_source['license']!r}")
        revision = raw_source["revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"source revision must be a full lowercase commit SHA, got {revision!r}")
        excluded_rows = raw_source.get("excluded_rows", [])
        if not isinstance(excluded_rows, list) or not all(isinstance(item, dict) for item in excluded_rows):
            raise TypeError("source excluded_rows must be a list of objects")
        config_names = {config.name for config in configs}
        split_names = set(raw_partitions.values())
        for exclusion in excluded_rows:
            required = {
                "config",
                "upstream_split",
                "upstream_index",
                "source_sha256",
                "reason",
            }
            if missing := required - set(exclusion):
                raise ValueError(f"source {raw_source['id']!r} exclusion is missing fields: {sorted(missing)}")
            if exclusion["config"] not in config_names:
                raise ValueError(f"source {raw_source['id']!r} exclusion has unknown config {exclusion['config']!r}")
            if exclusion["upstream_split"] not in split_names:
                raise ValueError(
                    f"source {raw_source['id']!r} exclusion has unknown split {exclusion['upstream_split']!r}"
                )
            if not isinstance(exclusion["upstream_index"], int) or exclusion["upstream_index"] < 0:
                raise ValueError("source exclusion indices must be non-negative integers")
            _validate_sha256("source exclusion source_sha256", exclusion["source_sha256"])
            if "content_sha256" in exclusion:
                _validate_sha256("source exclusion content_sha256", exclusion["content_sha256"])
            if not isinstance(exclusion["reason"], str) or not exclusion["reason"]:
                raise TypeError("source exclusion reasons must be non-empty strings")

        sources.append(
            Source(
                id=raw_source["id"],
                kind=kind,
                dataset=raw_source["dataset"],
                revision=revision,
                license=raw_source["license"],
                attribution=raw_source["attribution"],
                fields=frozenset(fields),
                partitions=dict(raw_partitions),
                configs=tuple(configs),
                excluded_rows=tuple(excluded_rows),
            )
        )

    blocked = raw.get("blocked_sources", [])
    if not isinstance(blocked, list) or not all(isinstance(item, dict) for item in blocked):
        raise TypeError("blocked_sources must be a list of objects")

    source_ids = [source.id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source manifest IDs must be unique")
    return SourceManifest(
        schema_version=SOURCE_MANIFEST_SCHEMA,
        sources=tuple(sources),
        blocked_sources=tuple(blocked),
        raw=dict(raw),
    )


def load_source_manifest(path: Path | None = None) -> SourceManifest:
    manifest_path = path or Path(__file__).with_name("sources.json")
    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("source manifest root must be an object")
    return source_manifest_from_dict(raw)


def extract_gold(solution: str) -> str:
    text = canonical_text(solution)
    if gold := canonical_text(vf.extract_boxed_answer(text, strict=True)):
        return gold

    candidates = [(text.rfind(marker), marker) for marker in (r"\fbox{",) if text.rfind(marker) >= 0]
    if candidates:
        start, marker = max(candidates)
        answer_start = start + len(marker)
        depth = 1
        for index, char in enumerate(text[answer_start:], start=answer_start):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                if gold := canonical_text(text[answer_start:index]):
                    return gold
                break

    # Two rows in the pinned MATH revision use the TeX form `\boxed 2`.
    unbraced = list(re.finditer(r"\\boxed\s+([^$\n.]+)", text))
    if unbraced and (gold := canonical_text(unbraced[-1].group(1))):
        return gold
    raise ValueError("MATH solution has no closed boxed answer")


def normalize_math_row(
    row: Mapping[str, Any],
    *,
    source: Source,
    config: str,
    upstream_split: str,
    partition: Partition,
) -> CatalogRecord:
    if set(row) != source.fields:
        raise ValueError(
            f"{source.id}/{config}/{upstream_split} schema drift: expected {sorted(source.fields)}, got {sorted(row)}"
        )
    if not all(isinstance(row[field], str) for field in source.fields):
        bad = sorted(field for field in source.fields if not isinstance(row[field], str))
        raise TypeError(f"{source.id}/{config}/{upstream_split} fields must be strings: {bad}")

    question = canonical_text(row["problem"])
    if not question:
        raise ValueError(f"{source.id}/{config}/{upstream_split} has an empty problem")
    level = canonical_text(row["level"])
    tier = tier_for_level(level)
    gold = extract_gold(row["solution"])
    prompt = canonical_text(INSTRUCTION + question)
    content_sha256 = _sha256(f"{prompt}\0{gold}")
    record_id = f"{source.id}@{source.revision}:{config}:{upstream_split}:{content_sha256[:16]}"
    source_sha256 = _source_row_sha256(row, source.fields)
    return CatalogRecord(
        question=question,
        prompt=prompt,
        gold=gold,
        record_id=record_id,
        content_sha256=content_sha256,
        source_sha256=source_sha256,
        source=source.id,
        revision=source.revision,
        config=config,
        upstream_split=upstream_split,
        partition=partition,
        tier=tier,
        level=level,
    )


def normalize_aime_row(
    row: Mapping[str, Any],
    *,
    source: Source,
    config: str,
    upstream_split: str,
    partition: Partition,
) -> CatalogRecord:
    if set(row) != source.fields:
        raise ValueError(
            f"{source.id}/{config}/{upstream_split} schema drift: expected {sorted(source.fields)}, got {sorted(row)}"
        )
    question = canonical_text(str(row["problem"]))
    if not question:
        raise ValueError(f"{source.id}/{config}/{upstream_split} has an empty problem")
    try:
        gold = str(int(str(row["answer"])))
    except ValueError as error:
        raise ValueError(f"{source.id}/{config}/{upstream_split} has a non-integer AIME answer") from error
    prompt = canonical_text(INSTRUCTION + question)
    content_sha256 = _sha256(f"{prompt}\0{gold}")
    record_id = f"{source.id}@{source.revision}:{config}:{upstream_split}:{content_sha256[:16]}"
    source_sha256 = _source_row_sha256(row, source.fields)
    return CatalogRecord(
        question=question,
        prompt=prompt,
        gold=gold,
        record_id=record_id,
        content_sha256=content_sha256,
        source_sha256=source_sha256,
        source=source.id,
        revision=source.revision,
        config=config,
        upstream_split=upstream_split,
        partition=partition,
        tier="hard",
        level="AIME",
    )


def normalize_source_row(
    row: Mapping[str, Any],
    *,
    source: Source,
    config: str,
    upstream_split: str,
    partition: Partition,
) -> CatalogRecord:
    if source.kind == "math":
        return normalize_math_row(
            row,
            source=source,
            config=config,
            upstream_split=upstream_split,
            partition=partition,
        )
    return normalize_aime_row(
        row,
        source=source,
        config=config,
        upstream_split=upstream_split,
        partition=partition,
    )


def build_catalog(records: Iterable[CatalogRecord]) -> tuple[CatalogRecord, ...]:
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    ids: dict[str, CatalogRecord] = {}
    hashes: dict[str, CatalogRecord] = {}
    for record in ordered:
        if previous := ids.get(record.record_id):
            raise ValueError(
                f"duplicate record id {record.record_id!r}: "
                f"{previous.source}/{previous.partition} and {record.source}/{record.partition}"
            )
        if previous := hashes.get(record.content_sha256):
            raise ValueError(
                f"duplicate content hash {record.content_sha256}: {previous.record_id} and {record.record_id}"
            )
        ids[record.record_id] = record
        hashes[record.content_sha256] = record
    return ordered


def validate_partition_disjointness(
    train: Sequence[CatalogRecord],
    eval: Sequence[CatalogRecord],
) -> None:
    train_ids = {record.record_id for record in train}
    eval_ids = {record.record_id for record in eval}
    if overlap := train_ids & eval_ids:
        raise ValueError(f"train/eval record id overlap: {sorted(overlap)[:3]}")

    train_hashes = {record.content_sha256 for record in train}
    eval_hashes = {record.content_sha256 for record in eval}
    if overlap := train_hashes & eval_hashes:
        raise ValueError(f"train/eval content hash overlap: {sorted(overlap)[:3]}")


def select_records(
    records: Sequence[CatalogRecord],
    *,
    partition: Partition,
    tier: Tier,
) -> tuple[CatalogRecord, ...]:
    selected = tuple(record for record in records if record.partition == partition and record.tier == tier)
    return tuple(sorted(selected, key=lambda record: record.record_id))


def build_catalog_manifest(
    records: Sequence[CatalogRecord],
    source_manifest: SourceManifest,
) -> dict[str, Any]:
    ordered = build_catalog(records)
    entries = [
        {
            "record_id": record.record_id,
            "content_sha256": record.content_sha256,
            "source_sha256": record.source_sha256,
            "source": record.source,
            "revision": record.revision,
            "config": record.config,
            "upstream_split": record.upstream_split,
            "partition": record.partition,
            "tier": record.tier,
            "level": record.level,
        }
        for record in ordered
    ]
    return {
        "schema_version": CATALOG_MANIFEST_SCHEMA,
        "source_manifest_digest": source_manifest_digest(source_manifest),
        "catalog_digest": _sha256(_canonical_json(entries)),
        "sources": [
            {
                "id": source.id,
                "dataset": source.dataset,
                "revision": source.revision,
                "license": source.license,
            }
            for source in source_manifest.sources
        ],
        "counts": {
            "total": len(entries),
            "partitions": {
                partition: sum(entry["partition"] == partition for entry in entries) for partition in PARTITIONS
            },
            "tiers": {tier: sum(entry["tier"] == tier for entry in entries) for tier in TIERS},
        },
        "ordered_records": entries,
    }


def write_catalog_manifest(
    path: Path,
    records: Sequence[CatalogRecord],
    source_manifest: SourceManifest,
) -> dict[str, Any]:
    artifact = build_catalog_manifest(records, source_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
