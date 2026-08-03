# harder-math-v1

`harder-math-v1` is a pinned `verifiers.v1` taskset for deterministic,
single-turn MATH training and MATH/AIME evaluation. It loads public data at
runtime; this package does not redistribute dataset rows.

## Source catalog

The active source is
[`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
at immutable revision
`21a5633873b6a120296cce3e2df9d5550074f4a3`. Its seven subject configs expose
`problem`, `level`, `type`, and `solution`, with 7,500 published training rows
and 5,000 published test rows. The mirror card and the upstream
[`hendrycks/math`](https://github.com/hendrycks/math) repository declare MIT;
cite Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH
Dataset* (NeurIPS 2021). The underlying competition-problem authorship is not
separately clarified by those licenses. Five source-hash-pinned training rows
are excluded: two have no Level 1-5 tier, two have empty boxed gold answers,
and one duplicates a published test problem. Any excluded-row drift fails
loading.

The evaluation-only hard tier also loads
[`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025)
at `c94da77eb22bbd6439e62a323bec18493a421302` (30 rows). Its pinned
card declares CC BY-NC-SA 4.0, attributes the dataset to MathArena, and states
that its AIME 2025 questions were extracted, converted to LaTeX, and verified.
The upstream split happens to be named `train`, but the source is mapped only
to this package's `eval` partition and can never enter RL training.

The pinned `HuggingFaceH4/aime_2024` revision, schema, and count were verified,
but its dataset card declares no license. It is recorded under
`blocked_sources` and is not loaded rather than guessing that a mirror license
applies to the original AIME problem text.

## Taskset contract

The distribution and taskset ID are `harder-math-v1`; the import package is
`harder_math_v1`. One taskset config selects a published partition and tier:

| Config | Upstream rows | Included difficulty |
|---|---|---|
| `partition = "train"` | MATH `train` only (7,495 eligible rows) | RL-eligible |
| `partition = "eval"` | MATH `test` + AIME 2025 (5,030 rows) | Evaluation-only |
| `tier = "base"` | Either partition | Levels 1-2 |
| `tier = "core"` | Either partition | Levels 3-4 |
| `tier = "hard"` | MATH Level 5; eval also includes AIME | Highest tier |

The trusted final catalog contract is:

| Partition | Total | Base | Core | Hard | Catalog digest |
|---|---:|---:|---:|---:|---|
| `train` | 7,495 | 1,912 | 3,282 | 2,301 | `f51df30441c419d3e569c6a9a4588bab9da55c16e13988e627d299fe0314eb8f` |
| `eval` | 5,030 | 1,331 | 2,345 | 1,354 | `ebe68009a7104d960d15e890489bdb6485476d56fe89407babbcc1532608285b` |

Use the null harness with the subprocess runtime:

```toml
[env.taskset]
id = "harder-math-v1"
partition = "train"
tier = "hard"
catalog_manifest_path = "artifacts/harder-math-train-hard.json"

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
```

The loader always reads every pinned source needed to verify train/eval
disjointness, checks every per-config raw count and exact row schema, normalizes
the complete presented prompt and gold answer with Unicode NFKC and LF line
endings, rejects missing boxed golds, sorts source-qualified record IDs, and
rejects duplicate prompts or prompt/gold content across sources and partitions.
It checks the final partition count, tier counts, catalog digest, source-manifest
digest, and source revisions against the package constants before returning a
catalog. If
`catalog_manifest_path` is set, it writes ordered IDs/hashes, counts, the source
pins/licenses, source-manifest digest, and aggregate catalog digest. The output
covers the whole selected partition, so all three tier runs share one catalog
digest. Each catalog/task row carries both `content_sha256` (prompt plus gold)
and `prompt_sha256` (presented prompt only).

`catalog.py` and `partition.py` contain deterministic row, hash, duplicate, and
tier logic. `loader.py` is the only module that invokes a dataset loader, so
fixture tests exercise the catalog contract without network access.

Scoring is binary and deterministic. The only reward directly calls
`verifiers.v1.verify_boxed_math_answer`; there is no LLM or reference-judge
fallback.

## Cache and offline use

The first load needs Hugging Face access to populate the `datasets` cache.
Prewarm both active pinned revisions before a disconnected run. Set
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to force cache-only replay;
missing cache entries or source/schema/count drift fail loudly.

## Tier-curve acceptance

`harder_math_v1.tier_curve` produces and validates `tier-curve.v1` artifacts
with source/catalog digests, baseline/package/model revisions, per-tier taskset
configs, decoding/renderer/grader settings, ordered per-record binary rewards,
and per-tier means, Wilson 95% confidence intervals, failure counts, and
`base_minus_hard`. Aggregation requires the trusted full eval catalog and an
out-of-band `TierCurveRunContract`; observation strings are never accepted as
authority for catalog, baseline, model, or decoding identity. Every one of the
5,030 expected record IDs must appear exactly once with its trusted prompt hash,
content hash, and tier.

For W4c, build one run contract with the common taskset config omitting `tier`,
then provide the full config on each observation:

```python
from harder_math_v1.catalog import EXPECTED_CATALOG_CONTRACTS
from harder_math_v1.tier_curve import TierCurveRunContract, aggregate_tier_curve

run_contract = TierCurveRunContract(
    baseline_id=baseline_id,
    package_name="harder-math-v1",
    package_version=package_version,
    package_commit=package_commit,
    model=model,
    model_revision=model_revision,
    decoding=decoding,
    renderer=renderer,
    grader=grader,
    taskset_config={"id": "harder-math-v1", "partition": "eval", **common_taskset_settings},
)
artifact = aggregate_tier_curve(
    observations,
    eval_catalog,
    run_contract,
    expected_catalog_contract=EXPECTED_CATALOG_CONTRACTS["eval"],
)
```

Each observation must include `baseline_id`, package/model/settings fields,
trusted catalog identity fields, `taskset_config` with `tier` equal to the
record's tier, `record_id`, `content_sha256`, `prompt_sha256`, `partition`,
`tier`, binary `reward`, and nullable `failure`. The only taskset field allowed
to differ across base/core/hard runs is `tier`. The artifact stores these as
`taskset_configs.{base,core,hard}`.

Environment acceptance requires the same frozen model revision, decoding,
renderer, and grader settings across all tiers and:

```text
base_mean - hard_mean >= 0.15
```

The API rejects thresholds below `0.15`; callers may request a stricter
threshold up to `1.0`.

The empirical model run is intentionally outside this package.
