from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tau.eval_tools.f12_recovery import (
    F12_EXPERIMENT_SOURCE_REVISION,
    F12_RECOVERY_ENVIRONMENT,
    f12_recovery_paths,
    validate_f12_recovery_environment,
)
from tau.eval_tools.fs_safety import open_directory_nofollow

_OUTPUT_PREFIX = Path("pretraining-data/prime-rl-math-7b-h200/generations")
_SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MODE_OUTPUTS = {
    "smoke": "smoke",
    "freeze-draft": "manifest",
    "freeze-finalize": "manifest",
    "tier-curve": "tier-curve",
    "train": "train",
    "eval-post-recovery": "eval-post-recovery",
    "eval-post-recovery-preflight": "eval-post-recovery-preflight",
}
_EVAL_OUTPUTS = {
    "baseline": "eval-baseline",
    "post": "eval-post",
}


@dataclass(frozen=True)
class EvidenceGeneration:
    root: Path
    smoke: Path
    manifest: Path
    eval_baseline: Path
    train: Path
    eval_post: Path
    eval_post_recovery: Path
    eval_post_recovery_preflight: Path
    tier_curve: Path

    @property
    def draft_manifest(self) -> Path:
        return self.manifest / "draft-manifest.json"

    @property
    def frozen_manifest(self) -> Path:
        return self.manifest / "frozen-eval-manifest.json"

    @property
    def baseline_rewards(self) -> Path:
        return self.eval_baseline / "rewards.json"

    @property
    def training_result(self) -> Path:
        return self.train / "training-result.json"

    @property
    def final_adapter(self) -> Path:
        return self.train / "final-adapter"

    @property
    def recovery_preflight_result(self) -> Path:
        return self.eval_post_recovery_preflight / "recovery-preflight.json"


def evidence_generation(
    source_revision: str,
    *,
    data_root: Path = Path("/data"),
) -> EvidenceGeneration:
    if not _SOURCE_REVISION_RE.fullmatch(source_revision):
        raise ValueError("source revision must be a full lowercase 40-character commit SHA")
    root = Path(os.path.abspath(data_root)) / _OUTPUT_PREFIX / source_revision
    return EvidenceGeneration(
        root=root,
        smoke=root / "smoke",
        manifest=root / "manifest",
        eval_baseline=root / "eval-baseline",
        train=root / "train",
        eval_post=root / "eval-post",
        eval_post_recovery=root / "eval-post-recovery",
        eval_post_recovery_preflight=root / "eval-post-recovery-preflight",
        tier_curve=root / "tier-curve",
    )


def expected_output_path(
    mode: str,
    source_revision: str,
    *,
    eval_label: str | None = None,
    data_root: Path = Path("/data"),
) -> Path:
    generation = evidence_generation(source_revision, data_root=data_root)
    eval_label = eval_label or None
    if mode == "eval":
        if eval_label not in _EVAL_OUTPUTS:
            raise ValueError("eval mode requires PRIME_RL_EVAL_LABEL=baseline or post")
        leaf = _EVAL_OUTPUTS[eval_label]
    elif mode == "eval-post-recovery":
        if eval_label != "post":
            raise ValueError("eval-post-recovery mode requires PRIME_RL_EVAL_LABEL=post")
        if source_revision == F12_EXPERIMENT_SOURCE_REVISION:
            raise ValueError("eval-post-recovery output must use a new runtime source generation")
        leaf = _MODE_OUTPUTS[mode]
    elif mode == "eval-post-recovery-preflight":
        if eval_label is not None:
            raise ValueError("PRIME_RL_EVAL_LABEL is not valid for eval-post-recovery-preflight mode")
        if source_revision == F12_EXPERIMENT_SOURCE_REVISION:
            raise ValueError("eval-post-recovery-preflight output must use a new runtime source generation")
        leaf = _MODE_OUTPUTS[mode]
    else:
        if eval_label is not None:
            raise ValueError("PRIME_RL_EVAL_LABEL is valid only for eval mode")
        try:
            leaf = _MODE_OUTPUTS[mode]
        except KeyError:
            raise ValueError(f"unsupported PRIME_RL_RUN_MODE: {mode}") from None
    return generation.root / leaf


def _validate_exact_path(name: str, supplied: str | Path | None, expected: Path, *, required: bool) -> None:
    if supplied is None or os.fspath(supplied) == "":
        if required:
            raise ValueError(f"{name} must be set to exactly {expected}")
        return
    raw = os.fspath(supplied)
    path = Path(raw)
    if not path.is_absolute() or raw != str(path) or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute canonical path: {raw}")
    if path != expected:
        raise ValueError(f"{name} is {path}, expected exactly {expected}")


def validate_generation_references(
    mode: str,
    source_revision: str,
    *,
    eval_label: str | None = None,
    manifest_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    baseline_rewards_path: str | Path | None = None,
    training_result_path: str | Path | None = None,
    training_output_dir: str | Path | None = None,
    lora_adapter_path: str | Path | None = None,
    comparison_output_path: str | Path | None = None,
    recovery_preflight_path: str | Path | None = None,
    recovery_preflight_output_path: str | Path | None = None,
    tier_curve_model_source_revision: str | None = None,
    tier_curve_model_manifest_path: str | Path | None = None,
    recovery_environment: Mapping[str, str | None] | None = None,
    data_root: Path = Path("/data"),
) -> EvidenceGeneration:
    generation = evidence_generation(source_revision, data_root=data_root)
    expected_output_path(mode, source_revision, eval_label=eval_label, data_root=data_root)
    eval_label = eval_label or None

    _validate_exact_path(
        "PRIME_RL_MANIFEST_DIR",
        manifest_dir,
        generation.manifest,
        required=mode in {"freeze-draft", "freeze-finalize", "train"},
    )
    recovery_mode = mode == "eval-post-recovery"
    preflight_mode = mode == "eval-post-recovery-preflight"
    recovery_input_mode = recovery_mode or preflight_mode
    if recovery_input_mode:
        validate_f12_recovery_environment(
            **(
                dict(recovery_environment)
                if recovery_environment is not None
                else {name: None for name in F12_RECOVERY_ENVIRONMENT}
            )
        )
        recovery_paths = f12_recovery_paths(data_root)
        expected_manifest = recovery_paths.manifest
    else:
        if recovery_environment is not None and any(value is not None for value in recovery_environment.values()):
            raise ValueError("F12 recovery contract values are only valid in eval-post-recovery mode")
        recovery_paths = None
        expected_manifest = generation.draft_manifest if eval_label == "baseline" else generation.frozen_manifest
    _validate_exact_path(
        "PRIME_RL_MANIFEST_PATH",
        manifest_path,
        expected_manifest,
        required=mode in {"eval", "eval-post-recovery", "eval-post-recovery-preflight"},
    )
    expected_baseline = recovery_paths.baseline_rewards if recovery_paths is not None else generation.baseline_rewards
    _validate_exact_path(
        "PRIME_RL_BASELINE_REWARDS_PATH",
        baseline_rewards_path,
        expected_baseline,
        required=mode == "freeze-finalize" or (mode == "eval" and eval_label == "post") or recovery_input_mode,
    )
    expected_training_result = (
        recovery_paths.training_result if recovery_paths is not None else generation.training_result
    )
    expected_training_output = recovery_paths.training_output_dir if recovery_paths is not None else generation.train
    expected_adapter = recovery_paths.adapter if recovery_paths is not None else generation.final_adapter
    _validate_exact_path(
        "PRIME_RL_TRAINING_RESULT_PATH",
        training_result_path,
        expected_training_result,
        required=(mode == "eval" and eval_label == "post") or recovery_input_mode,
    )
    _validate_exact_path(
        "PRIME_RL_TRAINING_OUTPUT_DIR",
        training_output_dir,
        expected_training_output,
        required=(mode == "eval" and eval_label == "post") or recovery_input_mode,
    )
    _validate_exact_path(
        "PRIME_RL_LORA_ADAPTER_PATH",
        lora_adapter_path,
        expected_adapter,
        required=(mode == "eval" and eval_label == "post") or recovery_input_mode,
    )
    if recovery_mode:
        _validate_exact_path(
            "PRIME_RL_COMPARISON_OUTPUT_PATH",
            comparison_output_path,
            generation.eval_post_recovery / "comparison.json",
            required=True,
        )
        _validate_exact_path(
            "PRIME_RL_RECOVERY_PREFLIGHT_PATH",
            recovery_preflight_path,
            generation.recovery_preflight_result,
            required=True,
        )
    elif comparison_output_path is not None and os.fspath(comparison_output_path) != "":
        raw_comparison = os.fspath(comparison_output_path)
        comparison = Path(raw_comparison)
        if (
            not comparison.is_absolute()
            or raw_comparison != str(comparison)
            or ".." in comparison.parts
            or comparison.parent != generation.eval_post
            or comparison.name in {"", ".", ".."}
        ):
            raise ValueError(
                f"PRIME_RL_COMPARISON_OUTPUT_PATH must be a canonical named file directly under {generation.eval_post}"
            )
    if not recovery_mode and recovery_preflight_path is not None and os.fspath(recovery_preflight_path) != "":
        raise ValueError("PRIME_RL_RECOVERY_PREFLIGHT_PATH is only valid in eval-post-recovery mode")
    if preflight_mode:
        _validate_exact_path(
            "PRIME_RL_RECOVERY_PREFLIGHT_OUTPUT_PATH",
            recovery_preflight_output_path,
            generation.recovery_preflight_result,
            required=True,
        )
    elif recovery_preflight_output_path is not None and os.fspath(recovery_preflight_output_path) != "":
        raise ValueError("PRIME_RL_RECOVERY_PREFLIGHT_OUTPUT_PATH is only valid in eval-post-recovery-preflight mode")
    if mode == "tier-curve":
        if not tier_curve_model_source_revision:
            raise ValueError("PRIME_RL_TIER_CURVE_MODEL_SOURCE_REVISION must be set for tier-curve mode")
        model_generation = evidence_generation(tier_curve_model_source_revision, data_root=data_root)
        _validate_exact_path(
            "PRIME_RL_TIER_CURVE_MODEL_MANIFEST_PATH",
            tier_curve_model_manifest_path,
            model_generation.frozen_manifest,
            required=True,
        )
    elif tier_curve_model_source_revision or tier_curve_model_manifest_path:
        raise ValueError("tier-curve model references are valid only for tier-curve mode")
    return generation


def prepare_output_directory(
    mode: str,
    output_dir: str | Path,
    source_revision: str,
    *,
    eval_label: str | None = None,
    manifest_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    baseline_rewards_path: str | Path | None = None,
    training_result_path: str | Path | None = None,
    training_output_dir: str | Path | None = None,
    lora_adapter_path: str | Path | None = None,
    comparison_output_path: str | Path | None = None,
    recovery_preflight_path: str | Path | None = None,
    recovery_preflight_output_path: str | Path | None = None,
    tier_curve_model_source_revision: str | None = None,
    tier_curve_model_manifest_path: str | Path | None = None,
    recovery_environment: Mapping[str, str | None] | None = None,
    data_root: Path = Path("/data"),
) -> Path:
    validate_generation_references(
        mode,
        source_revision,
        eval_label=eval_label,
        manifest_dir=manifest_dir,
        manifest_path=manifest_path,
        baseline_rewards_path=baseline_rewards_path,
        training_result_path=training_result_path,
        training_output_dir=training_output_dir,
        lora_adapter_path=lora_adapter_path,
        comparison_output_path=comparison_output_path,
        recovery_preflight_path=recovery_preflight_path,
        recovery_preflight_output_path=recovery_preflight_output_path,
        tier_curve_model_source_revision=tier_curve_model_source_revision,
        tier_curve_model_manifest_path=tier_curve_model_manifest_path,
        recovery_environment=recovery_environment,
        data_root=data_root,
    )
    expected = expected_output_path(mode, source_revision, eval_label=eval_label, data_root=data_root)
    raw_output = os.fspath(output_dir)
    supplied = Path(raw_output)
    if not supplied.is_absolute() or raw_output != str(supplied) or ".." in supplied.parts:
        raise ValueError(f"TAU_OUTPUT_DIR must be an absolute canonical path: {raw_output}")
    if supplied != expected:
        raise ValueError(f"TAU_OUTPUT_DIR is {supplied}, expected exactly {expected} for mode {mode}")

    data_root = Path(os.path.abspath(data_root))
    relative = expected.relative_to(data_root)
    root_descriptor = open_directory_nofollow(data_root)
    current_descriptor = root_descriptor
    try:
        for component in relative.parts:
            try:
                os.mkdir(component, 0o755, dir_fd=current_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise NotADirectoryError(f"output path component is not a directory: {component}")
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
    finally:
        if current_descriptor != root_descriptor:
            os.close(current_descriptor)
        os.close(root_descriptor)
    return expected


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the exact mode-bound Tau output directory")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--eval-label", default="")
    parser.add_argument("--manifest-dir", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--baseline-rewards-path", default="")
    parser.add_argument("--training-result-path", default="")
    parser.add_argument("--training-output-dir", default="")
    parser.add_argument("--lora-adapter-path", default="")
    parser.add_argument("--comparison-output-path", default="")
    parser.add_argument("--recovery-preflight-path", default="")
    parser.add_argument("--recovery-preflight-output-path", default="")
    parser.add_argument("--tier-curve-model-source-revision", default="")
    parser.add_argument("--tier-curve-model-manifest-path", default="")
    for name in F12_RECOVERY_ENVIRONMENT:
        parser.add_argument(f"--recovery-{name.replace('_', '-')}", default="")
    args = parser.parse_args(argv)
    prepared = prepare_output_directory(
        args.mode,
        args.output_dir,
        args.source_revision,
        eval_label=args.eval_label,
        manifest_dir=args.manifest_dir,
        manifest_path=args.manifest_path,
        baseline_rewards_path=args.baseline_rewards_path,
        training_result_path=args.training_result_path,
        training_output_dir=args.training_output_dir,
        lora_adapter_path=args.lora_adapter_path,
        comparison_output_path=args.comparison_output_path,
        recovery_preflight_path=args.recovery_preflight_path,
        recovery_preflight_output_path=args.recovery_preflight_output_path,
        tier_curve_model_source_revision=args.tier_curve_model_source_revision,
        tier_curve_model_manifest_path=args.tier_curve_model_manifest_path,
        recovery_environment=(
            {name: getattr(args, f"recovery_{name}") for name in F12_RECOVERY_ENVIRONMENT}
            if args.mode in {"eval-post-recovery", "eval-post-recovery-preflight"}
            else None
        ),
    )
    print(prepared.parent)


if __name__ == "__main__":
    main()
