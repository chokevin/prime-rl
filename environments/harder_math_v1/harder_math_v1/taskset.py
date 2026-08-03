from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf
from pydantic import model_validator

from harder_math_v1.catalog import (
    CatalogRecord,
    load_source_manifest,
    select_records,
    write_catalog_manifest,
)
from harder_math_v1.loader import load_verified_catalogs
from harder_math_v1.partition import Partition, Tier


class HarderMathData(vf.TaskData):
    question: str
    answer: str
    record_id: str
    content_sha256: str
    prompt_sha256: str
    source_sha256: str
    source: str
    revision: str
    upstream_config: str
    upstream_split: str
    partition: Partition
    tier: Tier
    level: str


class HarderMathTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5

    @model_validator(mode="after")
    def deterministic_grading_only(self) -> HarderMathTaskConfig:
        if self.judges:
            raise ValueError("harder-math-v1 does not allow judge plugins")
        if self.math_verify_timeout <= 0:
            raise ValueError("math_verify_timeout must be positive")
        return self


class HarderMathTask(vf.Task[HarderMathData, vf.State, HarderMathTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(
            trace.last_reply,
            self.data.answer,
            timeout_seconds=self.config.math_verify_timeout,
        )


class HarderMathConfig(vf.TasksetConfig):
    id: str = "harder-math-v1"
    partition: Partition = "train"
    tier: Tier = "base"
    catalog_manifest_path: Path | None = None
    task: HarderMathTaskConfig = HarderMathTaskConfig()


def _to_task(record: CatalogRecord, idx: int, config: HarderMathTaskConfig) -> HarderMathTask:
    return HarderMathTask(
        HarderMathData(
            idx=idx,
            name=record.record_id,
            prompt=record.prompt,
            question=record.question,
            answer=record.gold,
            record_id=record.record_id,
            content_sha256=record.content_sha256,
            prompt_sha256=record.prompt_sha256,
            source_sha256=record.source_sha256,
            source=record.source,
            revision=record.revision,
            upstream_config=record.config,
            upstream_split=record.upstream_split,
            partition=record.partition,
            tier=record.tier,
            level=record.level,
        ),
        config,
    )


class HarderMathTaskset(vf.Taskset[HarderMathTask, HarderMathConfig]):
    def load(self) -> list[HarderMathTask]:
        from datasets import load_dataset

        source_manifest = load_source_manifest()
        catalogs = load_verified_catalogs(source_manifest, load_dataset)
        records = select_records(
            catalogs[self.config.partition],
            partition=self.config.partition,
            tier=self.config.tier,
        )
        if self.config.catalog_manifest_path is not None:
            write_catalog_manifest(
                self.config.catalog_manifest_path,
                catalogs[self.config.partition],
                source_manifest,
            )
        return [_to_task(record, idx, self.config.task) for idx, record in enumerate(records)]
