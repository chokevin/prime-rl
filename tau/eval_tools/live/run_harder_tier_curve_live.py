"""Evaluate the trusted harder-math-v1 catalog against one fixed model server.

This module runs inside the GPU image after the Tau wrapper has privately
materialized the exact model snapshot and started its OpenAI-compatible server.
It retains one raw result per catalog record, then builds and validates the
package's tier-curve.v1 artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tau.eval_tools.json_io import load_json_with_sha256, write_json_exclusive
from tau.eval_tools.manifest import FrozenEvalManifest, validate_manifest_contract

SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_SCHEMA = "harder-tier-raw.v1"
COMMON_TASKSET_CONFIG = {
    "id": "harder-math-v1",
    "partition": "eval",
    "task": {"math_verify_timeout": 5},
}
RENDERER = {
    "type": "openai_chat_completions",
    "message_shape": "single_user",
    "prompt_field": "catalog.prompt",
}
GRADER = {
    "function": "verifiers.v1.verify_boxed_math_answer",
    "timeout_seconds": 5,
}


async def _evaluate_catalog(
    records,
    *,
    base_url: str,
    served_model_name: str,
    decoding: dict[str, Any],
    max_concurrency: int,
) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI
    from verifiers.v1 import verify_boxed_math_answer

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(max_concurrency)
    completed = 0

    async def _one(record) -> dict[str, Any]:
        nonlocal completed
        completion: str | None = None
        failure: str | None = None
        reward = 0.0
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=served_model_name,
                    messages=[{"role": "user", "content": record.prompt}],
                    temperature=decoding["temperature"],
                    top_p=decoding["top_p"] if decoding["top_p"] is not None else 1.0,
                    max_tokens=decoding["max_completion_tokens"],
                    extra_body={"seed": decoding["seed"]},
                )
            completion = response.choices[0].message.content or ""
            reward = float(
                verify_boxed_math_answer(
                    completion,
                    record.gold,
                    timeout_seconds=GRADER["timeout_seconds"],
                )
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                print(f"evaluated {completed}/{len(records)} harder-math records", file=sys.stderr)
        return {
            "record_id": record.record_id,
            "content_sha256": record.content_sha256,
            "prompt_sha256": record.prompt_sha256,
            "partition": record.partition,
            "tier": record.tier,
            "reward": reward,
            "failure": failure,
            "completion": completion,
        }

    try:
        return await asyncio.gather(*(_one(record) for record in records))
    finally:
        await client.close()


def _catalog_identity(contract) -> dict[str, Any]:
    return {
        "catalog_digest": contract.catalog_digest,
        "source_manifest_digest": contract.source_manifest_digest,
        "source_revisions": dict(contract.source_revisions),
    }


def _observation(raw: dict[str, Any], run_contract, catalog_contract) -> dict[str, Any]:
    return {
        **run_contract.identity(),
        **_catalog_identity(catalog_contract),
        "taskset_config": {
            **run_contract.taskset_config,
            "tier": raw["tier"],
        },
        **{
            field: raw[field]
            for field in (
                "record_id",
                "content_sha256",
                "prompt_sha256",
                "partition",
                "tier",
                "reward",
                "failure",
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    from datasets import load_dataset
    from harder_math_v1.catalog import (
        EXPECTED_CATALOG_CONTRACTS,
        load_source_manifest,
    )
    from harder_math_v1.loader import load_verified_catalogs
    from harder_math_v1.partition import TIERS
    from harder_math_v1.tier_curve import (
        TierCurveRunContract,
        aggregate_tier_curve,
        validate_tier_curve_artifact,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-source-revision", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-concurrency", type=int, default=128)
    args = parser.parse_args(argv)

    if not SOURCE_REVISION_RE.fullmatch(args.source_revision):
        raise ValueError("source revision must be a full lowercase 40-character commit SHA")
    if not SOURCE_REVISION_RE.fullmatch(args.model_source_revision):
        raise ValueError("model source revision must be a full lowercase 40-character commit SHA")
    if not SHA256_RE.fullmatch(args.model_manifest_sha256):
        raise ValueError("model manifest SHA-256 must be a lowercase 64-character digest")
    if args.max_concurrency <= 0:
        raise ValueError("max concurrency must be positive")

    output_dir = Path(args.output_dir)
    output_dir.resolve(strict=True)
    raw_paths = {tier: output_dir / f"raw-{tier}.json" for tier in TIERS}
    curve_path = output_dir / "tier-curve.v1.json"
    for path in (*raw_paths.values(), curve_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; tier-curve evidence is immutable")

    model_manifest_payload, model_manifest_sha256 = load_json_with_sha256(Path(args.model_manifest))
    if model_manifest_sha256 != args.model_manifest_sha256:
        raise ValueError(
            f"model manifest SHA-256 {model_manifest_sha256} does not match "
            f"the pinned digest {args.model_manifest_sha256}"
        )
    model_manifest = FrozenEvalManifest.model_validate(model_manifest_payload)
    if model_manifest.state != "finalized":
        raise ValueError("tier curve requires the finalized F10 model/decoding contract")
    validate_manifest_contract(
        model_manifest,
        expected_source_revision=args.model_source_revision,
        expected_verifiers_revision=model_manifest.verifiers_revision,
        expected_tasksets_revision=model_manifest.eval_taskset.taskset_revision,
        expected_model_name=model_manifest.model.name,
        expected_model_revision=model_manifest.model.revision,
        require_finalized=True,
    )
    decoding = model_manifest.decoding.model_dump()

    source_manifest = load_source_manifest()
    catalogs = load_verified_catalogs(source_manifest, load_dataset)
    eval_catalog = catalogs["eval"]
    catalog_contract = EXPECTED_CATALOG_CONTRACTS["eval"]
    run_contract = TierCurveRunContract(
        baseline_id=model_manifest.identity_hash(),
        package_name="harder-math-v1",
        package_version=version("harder-math-v1"),
        package_commit=args.source_revision,
        model=model_manifest.model.name,
        model_revision=model_manifest.model.revision,
        decoding=decoding,
        renderer=RENDERER,
        grader=GRADER,
        taskset_config=COMMON_TASKSET_CONFIG,
    )

    raw_records = asyncio.run(
        _evaluate_catalog(
            eval_catalog,
            base_url=args.base_url,
            served_model_name=args.served_model_name,
            decoding=decoding,
            max_concurrency=args.max_concurrency,
        )
    )
    created_at = datetime.now(timezone.utc).isoformat()
    for tier in TIERS:
        tier_records = sorted(
            (record for record in raw_records if record["tier"] == tier),
            key=lambda record: record["record_id"],
        )
        write_json_exclusive(
            raw_paths[tier],
            {
                "schema_version": RAW_SCHEMA,
                "created_at": created_at,
                "model_manifest_sha256": model_manifest_sha256,
                "model_contract_source_revision": args.model_source_revision,
                **run_contract.identity(),
                **_catalog_identity(catalog_contract),
                "taskset_config": {
                    **run_contract.taskset_config,
                    "tier": tier,
                },
                "records": tier_records,
            },
        )

    observations = [_observation(record, run_contract, catalog_contract) for record in raw_records]
    artifact = aggregate_tier_curve(
        observations,
        eval_catalog,
        run_contract,
        expected_catalog_contract=catalog_contract,
    )
    validate_tier_curve_artifact(
        artifact,
        eval_catalog,
        run_contract,
        expected_catalog_contract=catalog_contract,
    )
    write_json_exclusive(curve_path, artifact)

    failures = sum(record["failure"] is not None for record in raw_records)
    summaries = artifact["tiers"]
    print(
        "tier curve: "
        f"base={summaries['base']['mean']:.4f} "
        f"core={summaries['core']['mean']:.4f} "
        f"hard={summaries['hard']['mean']:.4f} "
        f"base_minus_hard={artifact['base_minus_hard']:+.4f} "
        f"failures={failures}",
        file=sys.stderr,
    )
    if failures:
        print("tier curve failed because one or more requests or graders failed", file=sys.stderr)
        return 1
    if not artifact["hardness_gate"]["passed"]:
        print("tier curve failed the fixed base-minus-hard >= 0.15 gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
