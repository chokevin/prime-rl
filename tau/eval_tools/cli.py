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

from tau.eval_tools.compare import compare_from_paths, write_result
from tau.eval_tools.manifest import FrozenEvalManifest, LeakageError, check_disjoint


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_from_paths(
            Path(args.manifest),
            Path(args.baseline),
            Path(args.post),
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
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
    compare_parser.add_argument(
        "--n-bootstrap", type=int, default=10_000, help="Bootstrap resample count (default: 10000)."
    )
    compare_parser.add_argument(
        "--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed (default: 0, deterministic)."
    )
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
