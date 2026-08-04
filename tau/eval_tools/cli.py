"""CLI entrypoint for the pure eval-tools: identity/leakage checks and the
paired baseline/post comparison gate. Run as a module so the `tau.eval_tools` package
resolves without installation:

    uv run --no-sync python -m tau.eval_tools.cli compare \\
        --manifest /data/.../manifest/frozen-eval-manifest.json \\
        --baseline /data/.../eval-baseline/rewards.json \\
        --post /data/.../eval-post/rewards.json \\
        --output /data/.../comparison.json

Exits 0 only when the comparison passes the goal harness's gate (delta >= +0.03 and
paired-bootstrap 95% CI lower bound > 0), 1 when a valid comparison fails the gate,
and 2 on invalid input or publication failure. See `live/` for the scripts that build
the manifest and reward files against a real model/dataset/inference server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau.eval_tools.artifacts import (
    TrainingResult,
    materialize_adapter_for_eval,
    validate_adapter_handoff,
    write_smoke_result,
)
from tau.eval_tools.compare import compare_from_paths, write_result
from tau.eval_tools.live.private_materialization_live import validate_run_root
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
        if args.output:
            write_result(result, Path(args.output))
    except Exception as exc:  # noqa: BLE001 - CLI boundary: surface any failure as a clean nonzero exit
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.report())
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
    validate_manifest_contract(
        manifest,
        expected_source_revision=args.source_revision,
        expected_verifiers_revision=args.verifiers_revision,
        expected_tasksets_revision=args.tasksets_revision,
        expected_model_name=args.model_name,
        expected_model_revision=args.model_revision,
        require_finalized=args.require_finalized,
    )
    print(f"ok: verified manifest identity {manifest.identity_hash()}")
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


def _cmd_validate_adapter_handoff(args: argparse.Namespace) -> int:
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    result = TrainingResult.load(Path(args.training_result))
    training_output_dir = Path(args.training_output_dir).resolve(strict=True)
    validate_adapter_handoff(
        result=result,
        manifest=manifest,
        training_output_dir=training_output_dir,
        expected_adapter_path=Path(args.adapter_path),
        expected_step=manifest.rl_config.max_steps,
        expected_rank=16,
    )
    private_run_root = validate_run_root(Path(args.private_run_root))
    private_adapter = materialize_adapter_for_eval(
        result=result,
        durable_adapter_path=Path(args.adapter_path),
        run_root=private_run_root,
    )
    print(f"ok: verified and privately materialized adapter at {private_adapter}", file=sys.stderr)
    return 0


def _cmd_recover_publish(args: argparse.Namespace) -> int:
    from tau.eval_tools.live.training_supervisor_live import recover_publish

    result = recover_publish(
        manifest_path=Path(args.manifest),
        artifact_output_dir=Path(args.output_dir),
        attempt_id=args.attempt_id,
    )
    print(f"ok: recovered publication for attempt {result.attempt_id}")
    return 0


def _cmd_validate_recovery_contract(args: argparse.Namespace) -> int:
    from tau.eval_tools.recovery import validate_recovery_contract

    contract = validate_recovery_contract(
        manifest_path=Path(args.manifest),
        baseline_rewards_path=Path(args.baseline_rewards),
        training_result_path=Path(args.training_result),
        training_output_dir=Path(args.training_output_dir),
        lora_adapter_path=Path(args.adapter_path),
    )
    print(
        f"ok: verified frozen F12 recovery contract (manifest identity {contract.manifest_identity_hash}, "
        f"training attempt {contract.training_attempt_id})",
        file=sys.stderr,
    )
    return 0


def _cmd_write_recovery_provenance(args: argparse.Namespace) -> int:
    from tau.eval_tools.recovery import validate_recovery_contract, write_recovery_provenance

    contract = validate_recovery_contract(
        manifest_path=Path(args.manifest),
        baseline_rewards_path=Path(args.baseline_rewards),
        training_result_path=Path(args.training_result),
        training_output_dir=Path(args.training_output_dir),
        lora_adapter_path=Path(args.adapter_path),
    )
    provenance = write_recovery_provenance(
        output_dir=Path(args.output_dir),
        runtime_source_revision=args.runtime_source_revision,
        comparison_path=Path(args.comparison_path),
        contract=contract,
    )
    print(f"ok: wrote recovery provenance to {provenance}", file=sys.stderr)
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
        help="Strictly validate the frozen experiment identity.",
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

    recovery_parser = subparsers.add_parser(
        "recover-publish",
        help="Publish one explicitly named successful training attempt without rerunning RL.",
    )
    recovery_parser.add_argument("--manifest", required=True)
    recovery_parser.add_argument("--output-dir", required=True)
    recovery_parser.add_argument("--attempt-id", required=True)
    recovery_parser.set_defaults(func=_cmd_recover_publish)

    handoff_parser = subparsers.add_parser(
        "validate-adapter-handoff",
        help="Verify training-result.json and the fixed final-adapter directory.",
    )
    handoff_parser.add_argument("--manifest", required=True)
    handoff_parser.add_argument("--training-result", required=True)
    handoff_parser.add_argument("--training-output-dir", required=True)
    handoff_parser.add_argument("--private-run-root", required=True)
    handoff_parser.add_argument("--adapter-path", required=True)
    handoff_parser.set_defaults(func=_cmd_validate_adapter_handoff)

    recovery_contract_parser = subparsers.add_parser(
        "validate-recovery-contract",
        help="Validate the frozen F12 post-eval inputs against the pinned cross-generation recovery contract.",
    )
    recovery_contract_parser.add_argument("--manifest", required=True)
    recovery_contract_parser.add_argument("--baseline-rewards", required=True)
    recovery_contract_parser.add_argument("--training-result", required=True)
    recovery_contract_parser.add_argument("--training-output-dir", required=True)
    recovery_contract_parser.add_argument("--adapter-path", required=True)
    recovery_contract_parser.set_defaults(func=_cmd_validate_recovery_contract)

    provenance_parser = subparsers.add_parser(
        "write-recovery-provenance",
        help="Write recovery-provenance.json binding the recovery runtime source and frozen F12 source to the comparison.",
    )
    provenance_parser.add_argument("--manifest", required=True)
    provenance_parser.add_argument("--baseline-rewards", required=True)
    provenance_parser.add_argument("--training-result", required=True)
    provenance_parser.add_argument("--training-output-dir", required=True)
    provenance_parser.add_argument("--adapter-path", required=True)
    provenance_parser.add_argument("--output-dir", required=True)
    provenance_parser.add_argument("--runtime-source-revision", required=True)
    provenance_parser.add_argument("--comparison-path", required=True)
    provenance_parser.set_defaults(func=_cmd_write_recovery_provenance)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
