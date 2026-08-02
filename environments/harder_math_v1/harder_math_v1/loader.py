from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from harder_math_v1.catalog import (
    CatalogRecord,
    SourceManifest,
    _source_row_sha256,
    build_catalog,
    normalize_source_row,
    validate_partition_disjointness,
)
from harder_math_v1.partition import PARTITIONS, Partition

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


def load_partition_records(
    manifest: SourceManifest,
    partition: Partition,
    load_dataset: DatasetLoader,
) -> tuple[CatalogRecord, ...]:
    records: list[CatalogRecord] = []
    for source in manifest.sources:
        if partition not in source.partitions:
            continue
        upstream_split = source.partitions[partition]
        seen_exclusions: set[tuple[str, str, int]] = set()
        for config in source.configs:
            rows = list(
                load_dataset(
                    source.dataset,
                    config.name,
                    split=upstream_split,
                    revision=source.revision,
                )
            )
            expected_count = config.counts[upstream_split]
            if len(rows) != expected_count:
                raise ValueError(
                    f"{source.id}/{config.name}/{upstream_split} count drift: "
                    f"expected {expected_count}, got {len(rows)}"
                )
            exclusions = {
                (item["config"], item["upstream_split"], item["upstream_index"]): item
                for item in source.excluded_rows
                if item["upstream_split"] == upstream_split
            }
            for upstream_index, row in enumerate(rows):
                exclusion_key = (config.name, upstream_split, upstream_index)
                if exclusion := exclusions.get(exclusion_key):
                    _validate_exclusion(
                        source.id,
                        source.fields,
                        exclusion_key,
                        exclusion,
                        row,
                        source=source,
                        partition=partition,
                    )
                    seen_exclusions.add(exclusion_key)
                    continue
                records.append(
                    normalize_source_row(
                        row,
                        source=source,
                        config=config.name,
                        upstream_split=upstream_split,
                        partition=partition,
                    )
                )
        expected_exclusions = {
            (item["config"], item["upstream_split"], item["upstream_index"])
            for item in source.excluded_rows
            if item["upstream_split"] == upstream_split
        }
        if seen_exclusions != expected_exclusions:
            raise ValueError(
                f"{source.id}/{upstream_split} reviewed exclusions drifted: "
                f"{sorted(expected_exclusions - seen_exclusions)}"
            )
    return build_catalog(records)


def _validate_exclusion(
    source_id: str,
    fields: Iterable[str],
    exclusion_key: tuple[str, str, int],
    exclusion: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    source,
    partition: Partition,
) -> None:
    actual_source_hash = _source_row_sha256(row, fields)
    if actual_source_hash != exclusion["source_sha256"]:
        raise ValueError(
            f"{source_id} exclusion {exclusion_key} source drift: "
            f"expected {exclusion['source_sha256']}, got {actual_source_hash}"
        )
    if expected_level := exclusion.get("level"):
        if row.get("level") != expected_level:
            raise ValueError(
                f"{source_id} exclusion {exclusion_key} level drift: "
                f"expected {expected_level!r}, got {row.get('level')!r}"
            )
    if expected_hash := exclusion.get("content_sha256"):
        config, upstream_split, _ = exclusion_key
        record = normalize_source_row(
            row,
            source=source,
            config=config,
            upstream_split=upstream_split,
            partition=partition,
        )
        if record.content_sha256 != expected_hash:
            raise ValueError(
                f"{source_id} exclusion {exclusion_key} content drift: "
                f"expected {expected_hash}, got {record.content_sha256}"
            )


def load_verified_catalogs(
    manifest: SourceManifest,
    load_dataset: DatasetLoader,
) -> Mapping[Partition, tuple[CatalogRecord, ...]]:
    catalogs = {partition: load_partition_records(manifest, partition, load_dataset) for partition in PARTITIONS}
    validate_partition_disjointness(catalogs["train"], catalogs["eval"])
    return catalogs
