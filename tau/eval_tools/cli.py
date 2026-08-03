"""CLI entrypoint for the pure eval-tools: identity/leakage checks and the
paired baseline/post comparison gate. Run as a module so the `tau.eval_tools` package
resolves without installation:

    uv run --no-sync python -m tau.eval_tools.cli compare \\
        --manifest /data/.../manifest/frozen-eval-manifest.json \\
        --baseline /data/.../eval-baseline/rewards.json \\
        --post /data/.../eval-post/rewards.json \\
        --output /data/.../comparison.json

Exits 0 only when the comparison passes the goal harness's gate (delta >= +0.03 and
paired-bootstrap 95% CI lower bound > 0); exits 1 on a failed gate, identity mismatch,
or malformed input. See `live/` for the scripts that build the manifest and reward
files against a real model/dataset/inference server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau.eval_tools.artifacts import (
    TrainingResult,
    publish_training_result,
    validate_adapter_handoff,
    write_smoke_result,
)
from tau.eval_tools.compare import compare_from_paths, write_result
from tau.eval_tools.manifest import (
    FrozenEvalManifest,
    LeakageError,
    check_disjoint,
    validate_manifest_contract,
)


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_from_paths(
            Path(args.manifest),
            Path(args.baseline),
            Path(args.post),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: surface any failure as a clean nonzero exit
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.report())
    if args.output:
        write_result(result, Path(args.output))
    return 0 if result.passed else 1


def _cmd_check_disjoint(args: argparse.Namespace) -> int:
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    train_hashes = set(Path(args.train_hashes).read_text().split())
    try:
        check_disjoint(manifest.eval_prompt_hashes, train_hashes)
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"ok: {len(manifest.eval_prompt_hashes)} eval prompt hashes are disjoint from {len(train_hashes)} training prompt hashes"
    )
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    model_path = validate_manifest_contract(
        manifest,
        expected_source_revision=args.source_revision,
        expected_verifiers_revision=args.verifiers_revision,
        expected_tasksets_revision=args.tasksets_revision,
        expected_model_name=args.model_name,
        expected_model_revision=args.model_revision,
        require_finalized=args.require_finalized,
    )
    print(model_path)
    return 0


def _cmd_write_smoke_result(args: argparse.Namespace) -> int:
    result = write_smoke_result(
        output_path=Path(args.output),
        source_revision=args.source_revision,
        config_path=args.config_path,
        configs_dir=Path(args.configs_dir),
    )
    print(f"ok: smoke evidence written for {len(result.resolved_configs)} resolved configs")
    return 0


def _cmd_publish_training_result(args: argparse.Namespace) -> int:
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    if manifest.state != "finalized":
        raise ValueError("training result requires a finalized manifest")
    result = publish_training_result(
        output_dir=Path(args.output_dir),
        manifest=manifest,
        source_revision=args.source_revision,
        expected_step=args.final_step,
        expected_rank=args.lora_rank,
    )
    print(
        f"ok: published stable step {result.source_step} adapter to {result.final_adapter_path} "
        f"(sha256={result.adapter_sha256})"
    )
    return 0


def _cmd_validate_adapter_handoff(args: argparse.Namespace) -> int:
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    result = TrainingResult.load(Path(args.training_result))
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        expected_adapter_path=Path(args.adapter_path),
        expected_step=args.final_step,
        expected_rank=args.lora_rank,
    )
    print(f"ok: verified adapter handoff {args.adapter_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau.eval_tools", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare frozen baseline/post rewards and apply the pass/fail gate."
    )
    compare_parser.add_argument("--manifest", required=True, help="Path to the frozen eval manifest JSON.")
    compare_parser.add_argument("--baseline", required=True, help="Path to the baseline rewards JSON.")
    compare_parser.add_argument("--post", required=True, help="Path to the post-training rewards JSON.")
    compare_parser.add_argument("--output", default=None, help="Optional path to write the comparison result JSON.")
    compare_parser.set_defaults(func=_cmd_compare)

    disjoint_parser = subparsers.add_parser(
        "check-disjoint",
        help="Verify a frozen manifest's eval prompt hashes are absent from a training prompt-hash file.",
    )
    disjoint_parser.add_argument("--manifest", required=True, help="Path to the frozen eval manifest JSON.")
    disjoint_parser.add_argument(
        "--train-hashes",
        required=True,
        help="Path to a whitespace-separated file of training prompt sha256 hex hashes.",
    )
    disjoint_parser.set_defaults(func=_cmd_check_disjoint)

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Strictly validate the experiment manifest and print its materialized model path.",
    )
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--source-revision", required=True)
    validate_parser.add_argument("--verifiers-revision", required=True)
    validate_parser.add_argument("--tasksets-revision", required=True)
    validate_parser.add_argument("--model-name", required=True)
    validate_parser.add_argument("--model-revision", required=True)
    validate_parser.add_argument("--require-finalized", action="store_true")
    validate_parser.set_defaults(func=_cmd_validate_manifest)

    smoke_parser = subparsers.add_parser(
        "write-smoke-result",
        help="Verify rl --dry-run outputs and exclusively write smoke-result.json.",
    )
    smoke_parser.add_argument("--output", required=True)
    smoke_parser.add_argument("--source-revision", required=True)
    smoke_parser.add_argument("--config-path", required=True)
    smoke_parser.add_argument("--configs-dir", required=True)
    smoke_parser.set_defaults(func=_cmd_write_smoke_result)

    training_parser = subparsers.add_parser(
        "publish-training-result",
        help="Select the latest stable adapter, publish it to final-adapter, and write training-result.json.",
    )
    training_parser.add_argument("--manifest", required=True)
    training_parser.add_argument("--output-dir", required=True)
    training_parser.add_argument("--source-revision", required=True)
    training_parser.add_argument("--final-step", type=int, required=True)
    training_parser.add_argument("--lora-rank", type=int, required=True)
    training_parser.set_defaults(func=_cmd_publish_training_result)

    handoff_parser = subparsers.add_parser(
        "validate-adapter-handoff",
        help="Verify training-result.json and the fixed final-adapter directory.",
    )
    handoff_parser.add_argument("--manifest", required=True)
    handoff_parser.add_argument("--training-result", required=True)
    handoff_parser.add_argument("--adapter-path", required=True)
    handoff_parser.add_argument("--final-step", type=int, required=True)
    handoff_parser.add_argument("--lora-rank", type=int, required=True)
    handoff_parser.set_defaults(func=_cmd_validate_adapter_handoff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
