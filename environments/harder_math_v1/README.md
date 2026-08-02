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

The evaluation-only hard tier also loads two verified public mirrors:

- [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  at `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` (30 rows). The
  pinned card declares no license and attributes its AIME 2024 I/II rows to
  `AI-MO/aimo-validation-aime`. Use is subject to source terms; this package
  performs runtime loading and does not redistribute the problem text.
- [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025)
  at `c94da77eb22bbd6439e62a323bec18493a421302` (30 rows). Its pinned
  card declares CC BY-NC-SA 4.0, attributes the dataset to MathArena, and states
  that the questions were extracted, converted to LaTeX, and verified.

Both upstream splits happen to be named `train`, but these sources are mapped
only to this package's `eval` partition and can never enter RL training.

## Taskset contract

The distribution and taskset ID are `harder-math-v1`; the import package is
`harder_math_v1`. One taskset config selects a published partition and tier:

| Config | Upstream rows | Included difficulty |
|---|---|---|
| `partition = "train"` | MATH `train` only (7,495 eligible rows) | RL-eligible |
| `partition = "eval"` | MATH `test` + AIME 2024/2025 (5,060 rows) | Evaluation-only |
| `tier = "base"` | Either partition | Levels 1-2 |
| `tier = "core"` | Either partition | Levels 3-4 |
| `tier = "hard"` | MATH Level 5; eval also includes AIME | Highest tier |

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
rejects duplicate content across sources and partitions. If
`catalog_manifest_path` is set, it writes ordered IDs/hashes, counts, the source
pins/licenses, source-manifest digest, and aggregate catalog digest. The output
covers the whole selected partition, so all three tier runs share one catalog
digest.

`catalog.py` and `partition.py` contain deterministic row, hash, duplicate, and
tier logic. `loader.py` is the only module that invokes a dataset loader, so
fixture tests exercise the catalog contract without network access.

Scoring is binary and deterministic. The only reward directly calls
`verifiers.v1.verify_boxed_math_answer`; there is no LLM or reference-judge
fallback.

## Cache and offline use

The first load needs Hugging Face access to populate the `datasets` cache.
Prewarm all three active pinned revisions before a disconnected run. Set
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to force cache-only replay;
missing cache entries or source/schema/count drift fail loudly.

## Tier-curve acceptance

`harder_math_v1.tier_curve` produces and validates `tier-curve.v1` artifacts
with source/catalog digests, package and model revisions, taskset config,
decoding/renderer/grader settings, ordered per-record binary rewards, and
per-tier means, Wilson 95% confidence intervals, failure counts, and
`base_minus_hard`. Aggregation rejects mixed manifests or run settings.

Environment acceptance requires the same frozen model revision, decoding,
renderer, and grader settings across all tiers and:

```text
base_mean - hard_mean >= 0.15
```

The empirical model run is intentionally outside this package.
