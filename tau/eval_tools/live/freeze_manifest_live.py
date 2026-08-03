"""Builds the frozen eval manifest for the primary experiment tuple
(`Qwen/Qwen2.5-7B-Instruct` + `math-env-v1` train + `math500-v1` held-out eval).

Must run inside the prime-rl GPU container image — it imports `verifiers`,
`math500_v1`/`math_env_v1` (workspace members baked into the image), and
`huggingface_hub`, and needs network access to the HF Hub. Not importable or unit
tested in this repo's macOS dev sandbox; see ../tests/ for what *is* tested here
(hashing, manifest schema/validators, comparison gate — all pure and dependency-light).

Two-phase, because whether a tuple is even eligible to freeze depends on a baseline
mean measured by *running* the model against these exact prompts first:

  1. `draft`    — load math500-v1 (eval) and math-env-v1 (train) tasksets, hash every
                  prompt, prove zero eval/train overlap, and write an unfrozen draft
                  manifest (`baseline_mean_headroom_ok` unset/placeholder). Does not
                  require a running inference server.
  2. `finalize` — given a measured baseline mean (from
                  `run_frozen_eval_live.py --manifest <draft> --label baseline`), checks
                  the `[0.10, 0.80]` headroom band and writes the immutable final
                  manifest. Refuses to finalize (exit 1) if the baseline is out of
                  band — per the goal harness, an out-of-band tuple must be replaced
                  *before* training, never rerolled after.

Usage (inside the container, cwd=/app, baked venv active, invoked by
tau/scripts/run-prime-rl.sh in freeze mode):

    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live draft \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --out /data/pretraining-data/prime-rl-math-7b-h200/manifest/draft-manifest.json \\
        --temperature 0.0 --seed 0

    # ... run the baseline eval against the draft, then:

    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live finalize \\
        --draft /data/pretraining-data/prime-rl-math-7b-h200/manifest/draft-manifest.json \\
        --baseline-mean 0.42 \\
        --out /data/pretraining-data/prime-rl-math-7b-h200/manifest/frozen-eval-manifest.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.manifest import (
    EXPECTED_TRAIN_DATASET,
    EXPECTED_TRAIN_DATASET_REVISION,
    EXPECTED_TRAIN_DATASET_SUBSET,
    GIT_SHA_RE,
    DecodingConfig,
    ExampleRecord,
    FrozenEvalManifest,
    ModelSnapshot,
    TasksetRef,
    TrainingDataIdentity,
    build_manifest,
    check_disjoint,
    check_headroom,
    validate_manifest_contract,
)

GRADER = "verifiers.v1.scoring.verify_boxed_math_answer"


def _resolve_model_snapshot(model_name: str, model_revision: str, cache_dir: str) -> ModelSnapshot:
    from huggingface_hub import model_info, snapshot_download

    if not GIT_SHA_RE.fullmatch(model_revision):
        raise ValueError("model revision must be an exact 40-character lowercase commit SHA")
    info = model_info(model_name, revision=model_revision)
    if info.sha != model_revision:
        raise RuntimeError(f"huggingface_hub resolved {model_name}@{model_revision} to {info.sha!r}")
    local_path = Path(
        snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            cache_dir=cache_dir,
        )
    ).resolve()
    if local_path.name != model_revision or not (local_path / "config.json").is_file():
        raise RuntimeError(f"materialized model path {local_path} is not the exact {model_revision} snapshot")
    return ModelSnapshot(name=model_name, revision=model_revision, local_path=str(local_path))


def _load_eval_examples(n: int, taskset_revision: str) -> tuple[TasksetRef, list[ExampleRecord]]:
    import verifiers.v1 as vf
    from math500_v1.taskset import DATASET_NAME, DATASET_REVISION, DATASET_SPLIT, Math500Taskset

    tasks = Math500Taskset(vf.TasksetConfig()).select(n if n > 0 else None)
    examples = [
        ExampleRecord(id=t.data.idx, prompt_hash=hash_text(t.data.prompt), answer_hash=hash_text(str(t.data.answer)))
        for t in tasks
    ]
    taskset_ref = TasksetRef(
        id="math500-v1",
        taskset_revision=taskset_revision,
        dataset_name=DATASET_NAME,
        dataset_subset=None,
        dataset_split=DATASET_SPLIT,
        dataset_revision=DATASET_REVISION,
        dataset_local_path=None,
    )
    return taskset_ref, examples


def _materialize_training_snapshot(
    dataset_name: str,
    dataset_revision: str,
    cache_dir: str,
) -> Path:
    from huggingface_hub import dataset_info, snapshot_download

    if not GIT_SHA_RE.fullmatch(dataset_revision):
        raise ValueError("training dataset revision must be an exact 40-character lowercase commit SHA")
    info = dataset_info(dataset_name, revision=dataset_revision)
    if info.sha != dataset_revision:
        raise RuntimeError(f"huggingface_hub resolved {dataset_name}@{dataset_revision} to {info.sha!r}")
    local_path = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=dataset_revision,
            cache_dir=cache_dir,
        )
    ).resolve()
    if local_path.name != dataset_revision:
        raise RuntimeError(
            f"materialized training dataset path {local_path} is not the exact {dataset_revision} snapshot"
        )
    return local_path


def _load_train_prompt_hashes(
    dataset_name: str,
    dataset_revision: str,
    dataset_subset: str,
    dataset_split: str,
    taskset_revision: str,
    cache_dir: str,
) -> tuple[TasksetRef, TrainingDataIdentity]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    local_path = _materialize_training_snapshot(dataset_name, dataset_revision, cache_dir)
    config = MathConfig(
        dataset_name=str(local_path),
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
    )
    tasks = MathTaskset(config).select()
    prompt_hashes = [hash_text(t.data.prompt) for t in tasks]
    taskset_ref = TasksetRef(
        id="math-env-v1",
        taskset_revision=taskset_revision,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        dataset_local_path=str(local_path),
    )
    return taskset_ref, TrainingDataIdentity.from_prompt_hashes(prompt_hashes)


def _cmd_draft(args: argparse.Namespace) -> int:
    model = _resolve_model_snapshot(args.model_name, args.model_revision, args.model_cache_dir)
    eval_taskset, examples = _load_eval_examples(args.n, args.tasksets_revision)
    train_taskset, training_data = _load_train_prompt_hashes(
        args.train_dataset_name,
        args.train_dataset_revision,
        args.train_dataset_subset,
        args.train_dataset_split,
        args.tasksets_revision,
        args.model_cache_dir,
    )

    eval_hashes = {e.prompt_hash for e in examples}
    check_disjoint(eval_hashes, set(training_data.prompt_hashes))

    manifest = build_manifest(
        state="draft",
        source_revision=args.source_revision,
        verifiers_revision=args.verifiers_revision,
        model=model,
        eval_taskset=eval_taskset,
        train_taskset=train_taskset,
        training_data=training_data,
        examples=examples,
        decoding=DecodingConfig(temperature=args.temperature, top_p=args.top_p, seed=args.seed),
        grader=GRADER,
        baseline_mean=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    validate_manifest_contract(
        manifest,
        expected_source_revision=args.source_revision,
        expected_verifiers_revision=args.verifiers_revision,
        expected_tasksets_revision=args.tasksets_revision,
        expected_model_name=args.model_name,
        expected_model_revision=args.model_revision,
        require_finalized=False,
    )
    manifest.save(Path(args.out))
    print(f"ok: draft manifest with {manifest.n} disjoint eval examples written to {args.out}")
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    draft = FrozenEvalManifest.load(Path(args.draft))
    if draft.state != "draft":
        raise ValueError(f"cannot finalize manifest in state {draft.state!r}")
    if not check_headroom(args.baseline_mean):
        print(
            f"error: measured baseline mean {args.baseline_mean:.4f} is outside the accepted "
            "[0.10, 0.80] headroom band — refusing to freeze this tuple. Choose a different "
            "model/taskset and start a new draft; do not reroll after training.",
            file=sys.stderr,
        )
        return 1
    finalized = FrozenEvalManifest.model_validate(
        {
            **draft.model_dump(),
            "state": "finalized",
            "baseline_mean": args.baseline_mean,
            "baseline_mean_headroom_ok": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    finalized.save(Path(args.out))
    print(
        f"ok: frozen eval manifest ({finalized.n} examples, baseline_mean={args.baseline_mean:.4f}) written to {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau.eval_tools.live.freeze_manifest_live", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft", help="Build the draft manifest and prove train/eval disjointness.")
    draft.add_argument("--model-name", required=True)
    draft.add_argument("--model-revision", required=True)
    draft.add_argument("--model-cache-dir", required=True)
    draft.add_argument("--source-revision", required=True)
    draft.add_argument("--verifiers-revision", required=True)
    draft.add_argument("--tasksets-revision", required=True)
    draft.add_argument("--out", required=True)
    draft.add_argument("--n", type=int, default=500, choices=[500], help="All 500 held-out eval examples.")
    draft.add_argument("--temperature", type=float, default=0.0)
    draft.add_argument("--top-p", type=float, default=None)
    draft.add_argument("--seed", type=int, default=0)
    draft.add_argument("--train-dataset-name", default=EXPECTED_TRAIN_DATASET, choices=[EXPECTED_TRAIN_DATASET])
    draft.add_argument(
        "--train-dataset-revision",
        default=EXPECTED_TRAIN_DATASET_REVISION,
        choices=[EXPECTED_TRAIN_DATASET_REVISION],
    )
    draft.add_argument(
        "--train-dataset-subset",
        default=EXPECTED_TRAIN_DATASET_SUBSET,
        choices=[EXPECTED_TRAIN_DATASET_SUBSET],
    )
    draft.add_argument("--train-dataset-split", default="train")
    draft.set_defaults(func=_cmd_draft)

    finalize = subparsers.add_parser("finalize", help="Freeze the manifest once the baseline headroom is measured.")
    finalize.add_argument("--draft", required=True)
    finalize.add_argument("--baseline-mean", type=float, required=True)
    finalize.add_argument("--out", required=True)
    finalize.set_defaults(func=_cmd_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
