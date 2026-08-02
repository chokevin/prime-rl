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
    DecodingConfig,
    ExampleRecord,
    FrozenEvalManifest,
    ModelSnapshot,
    TasksetRef,
    build_manifest,
    check_disjoint,
)

GRADER = "verifiers.v1.scoring.verify_boxed_math_answer"


def _resolve_model_snapshot(model_name: str) -> ModelSnapshot:
    """Resolve the exact HF commit SHA served for `model_name`. prime-rl's model config
    takes a name or a local path, not a pinned revision (see the W2 selection memo) —
    recording the resolved SHA here is the manifest's proof of exactly which weights
    were served, independent of whether the caller also pre-downloads a pinned local
    snapshot directory."""
    from huggingface_hub import model_info

    info = model_info(model_name)
    if info.sha is None:
        raise RuntimeError(f"huggingface_hub could not resolve a commit SHA for {model_name!r}")
    return ModelSnapshot(name=model_name, revision=info.sha)


def _load_eval_examples(n: int) -> tuple[TasksetRef, list[ExampleRecord]]:
    import verifiers.v1 as vf
    from math500_v1.taskset import DATASET_NAME, DATASET_REVISION, DATASET_SPLIT, Math500Taskset

    tasks = Math500Taskset(vf.TasksetConfig()).select(n if n > 0 else None)
    examples = [
        ExampleRecord(id=t.data.idx, prompt_hash=hash_text(t.data.prompt), answer_hash=hash_text(str(t.data.answer)))
        for t in tasks
    ]
    taskset_ref = TasksetRef(
        id="math500-v1",
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        dataset_revision=DATASET_REVISION,
    )
    return taskset_ref, examples


def _load_train_prompt_hashes(
    dataset_name: str, dataset_subset: str, dataset_split: str
) -> tuple[TasksetRef, set[str]]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    config = MathConfig(dataset_name=dataset_name, dataset_subset=dataset_subset, dataset_split=dataset_split)
    tasks = MathTaskset(config).select()
    hashes = {hash_text(t.data.prompt) for t in tasks}
    taskset_ref = TasksetRef(
        id="math-env-v1",
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=None,  # math-env-v1 does not pin a dataset revision upstream
    )
    return taskset_ref, hashes


def _cmd_draft(args: argparse.Namespace) -> int:
    model = _resolve_model_snapshot(args.model_name)
    eval_taskset, examples = _load_eval_examples(args.n)
    train_taskset, train_hashes = _load_train_prompt_hashes(
        args.train_dataset_name, args.train_dataset_subset, args.train_dataset_split
    )

    eval_hashes = {e.prompt_hash for e in examples}
    check_disjoint(eval_hashes, train_hashes)  # raises LeakageError and aborts on overlap

    manifest = build_manifest(
        model=model,
        eval_taskset=eval_taskset,
        train_taskset=train_taskset,
        examples=examples,
        decoding=DecodingConfig(temperature=args.temperature, top_p=args.top_p, seed=args.seed),
        grader=GRADER,
        baseline_mean=0.0,  # placeholder — finalize() overwrites this after measuring
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.save(Path(args.out), force=args.force)
    print(f"ok: draft manifest with {manifest.n} disjoint eval examples written to {args.out}")
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    draft = FrozenEvalManifest.load(Path(args.draft))
    finalized = draft.model_copy(
        update={
            "baseline_mean_headroom_ok": args.baseline_mean >= 0.10 and args.baseline_mean <= 0.80,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if not finalized.baseline_mean_headroom_ok:
        print(
            f"error: measured baseline mean {args.baseline_mean:.4f} is outside the accepted "
            "[0.10, 0.80] headroom band — refusing to freeze this tuple. Choose a different "
            "model/taskset and start a new draft; do not reroll after training.",
            file=sys.stderr,
        )
        return 1
    finalized.save(Path(args.out), force=args.force)
    print(
        f"ok: frozen eval manifest ({finalized.n} examples, baseline_mean={args.baseline_mean:.4f}) written to {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau.eval_tools.live.freeze_manifest_live", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft", help="Build the draft manifest and prove train/eval disjointness.")
    draft.add_argument("--model-name", required=True)
    draft.add_argument("--out", required=True)
    draft.add_argument("--n", type=int, default=-1, help="Eval examples to load (-1 = all 500).")
    draft.add_argument("--temperature", type=float, default=0.0)
    draft.add_argument("--top-p", type=float, default=None)
    draft.add_argument("--seed", type=int, default=0)
    draft.add_argument("--train-dataset-name", default="PrimeIntellect/Hendrycks-Math")
    draft.add_argument("--train-dataset-subset", default="default")
    draft.add_argument("--train-dataset-split", default="train")
    draft.add_argument("--force", action="store_true")
    draft.set_defaults(func=_cmd_draft)

    finalize = subparsers.add_parser("finalize", help="Freeze the manifest once the baseline headroom is measured.")
    finalize.add_argument("--draft", required=True)
    finalize.add_argument("--baseline-mean", type=float, required=True)
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--force", action="store_true")
    finalize.set_defaults(func=_cmd_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
