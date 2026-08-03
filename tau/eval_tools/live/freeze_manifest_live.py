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
  2. `finalize` — strict-load the fixed baseline rewards artifact, verify its label,
                  evaluation identity, exact 500 records, binary rewards, and digest;
                  compute the mean internally; validate the effective RL configuration;
                  and write the immutable final manifest. Refuses to finalize if the
                  mean is outside `[0.10, 0.80]`.

Usage (inside the container, cwd=/app, baked venv active, invoked by
tau/scripts/run-prime-rl.sh in freeze mode):

    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live draft \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --out /data/pretraining-data/prime-rl-math-7b-h200/manifest/draft-manifest.json \\
        --temperature 0.0 --seed 0

    # ... run the baseline eval against the draft, then:

    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live finalize \\
        --draft /data/pretraining-data/prime-rl-math-7b-h200/manifest/draft-manifest.json \\
        --baseline-rewards /data/pretraining-data/prime-rl-math-7b-h200/eval-baseline/rewards.json \\
        --out /data/pretraining-data/prime-rl-math-7b-h200/manifest/frozen-eval-manifest.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tau.eval_tools.compare import RewardRecord
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.live.private_materialization_live import (
    materialize_model_for_freeze,
    materialize_training_dataset_for_freeze,
)
from tau.eval_tools.live.validate_training_data_live import resolve_effective_rl_config
from tau.eval_tools.manifest import (
    EXPECTED_TRAIN_DATASET,
    EXPECTED_TRAIN_DATASET_REVISION,
    EXPECTED_TRAIN_DATASET_SUBSET,
    DecodingConfig,
    ExampleRecord,
    FrozenEvalManifest,
    ModelSnapshot,
    TasksetRef,
    TrainingDataIdentity,
    TrainingRecord,
    build_manifest,
    check_disjoint,
    check_headroom,
    validate_manifest_contract,
)

GRADER = "verifiers.v1.scoring.verify_boxed_math_answer"


def _resolve_model_snapshot(model_name: str, model_revision: str, run_root: str) -> ModelSnapshot:
    model, _ = materialize_model_for_freeze(model_name, model_revision, Path(run_root))
    return model


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
    )
    return taskset_ref, examples


def _load_training_records(
    dataset_name: str,
    dataset_revision: str,
    dataset_subset: str,
    dataset_split: str,
    taskset_revision: str,
    run_root: str,
) -> tuple[TasksetRef, TrainingDataIdentity]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    local_path, file_manifest = materialize_training_dataset_for_freeze(
        dataset_name,
        dataset_revision,
        Path(run_root),
    )
    Path(run_root).chmod(0o555)
    config = MathConfig(
        dataset_name=str(local_path),
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
    )
    tasks = MathTaskset(config).select()
    records = [
        TrainingRecord(
            id=task.data.idx,
            prompt_hash=hash_text(task.data.prompt),
            answer_hash=hash_text(str(task.data.answer)),
        )
        for task in tasks
    ]
    taskset_ref = TasksetRef(
        id="math-env-v1",
        taskset_revision=taskset_revision,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
    )
    return taskset_ref, TrainingDataIdentity.from_records(records, file_manifest=file_manifest)


def _cmd_draft(args: argparse.Namespace) -> int:
    model = _resolve_model_snapshot(args.model_name, args.model_revision, args.run_root)
    eval_taskset, examples = _load_eval_examples(args.n, args.tasksets_revision)
    train_taskset, training_data = _load_training_records(
        args.train_dataset_name,
        args.train_dataset_revision,
        args.train_dataset_subset,
        args.train_dataset_split,
        args.tasksets_revision,
        args.run_root,
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
        baseline_rewards_sha256=None,
        rl_config=None,
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


def validate_baseline_evidence(draft: FrozenEvalManifest, path: Path) -> tuple[float, str]:
    baseline, baseline_sha256 = RewardRecord.load_with_digest(path)
    if baseline.model_label != "baseline" or baseline.frozen_manifest_identity_hash is not None:
        raise ValueError("finalization requires immutable pre-finalization baseline reward evidence")
    if baseline.evaluation_identity_hash != draft.evaluation_identity_hash():
        raise ValueError("baseline reward evidence does not match the draft evaluation identity")
    expected_ids = {str(record.id) for record in draft.examples}
    if set(baseline.rewards) != expected_ids or len(baseline.rewards) != draft.n:
        raise ValueError("baseline reward evidence does not contain every frozen example id exactly once")
    baseline_mean = sum(baseline.rewards.values()) / draft.n
    if not check_headroom(baseline_mean):
        raise ValueError(
            f"measured baseline mean {baseline_mean:.4f} is outside the accepted [0.10, 0.80] headroom band"
        )
    return baseline_mean, baseline_sha256


def _cmd_finalize(args: argparse.Namespace) -> int:
    draft = FrozenEvalManifest.load(Path(args.draft))
    if draft.state != "draft":
        raise ValueError(f"cannot finalize manifest in state {draft.state!r}")
    validate_manifest_contract(
        draft,
        expected_source_revision=draft.source_revision,
        expected_verifiers_revision=draft.verifiers_revision,
        expected_tasksets_revision=draft.eval_taskset.taskset_revision,
        expected_model_name=draft.model.name,
        expected_model_revision=draft.model.revision,
        require_finalized=False,
    )
    baseline_mean, baseline_sha256 = validate_baseline_evidence(draft, Path(args.baseline_rewards))
    _, _, config_identity = resolve_effective_rl_config(
        Path(args.config),
        draft,
        source_config_rel=args.config_rel,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    finalized = FrozenEvalManifest.model_validate(
        {
            **draft.model_dump(),
            "state": "finalized",
            "baseline_mean": baseline_mean,
            "baseline_mean_headroom_ok": True,
            "baseline_rewards_sha256": baseline_sha256,
            "rl_config": config_identity.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    finalized.save(Path(args.out))
    print(
        f"ok: frozen eval manifest ({finalized.n} examples, baseline_mean={baseline_mean:.4f}, "
        f"baseline_sha256={baseline_sha256}) written to {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau.eval_tools.live.freeze_manifest_live", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft", help="Build the draft manifest and prove train/eval disjointness.")
    draft.add_argument("--model-name", required=True)
    draft.add_argument("--model-revision", required=True)
    draft.add_argument("--run-root", required=True)
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
    finalize.add_argument("--baseline-rewards", required=True)
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--config-rel", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--max-steps", required=True, type=int)
    finalize.add_argument("--out", required=True)
    finalize.set_defaults(func=_cmd_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
