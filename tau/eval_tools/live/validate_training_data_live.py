from __future__ import annotations

import argparse
import hashlib
import os
import tomllib
from pathlib import Path

from tau.eval_tools.artifacts import write_training_preflight
from tau.eval_tools.hashing import hash_text
from tau.eval_tools.live.run_frozen_eval_live import _reload_and_verify_examples
from tau.eval_tools.manifest import (
    EXPECTED_TRAIN_TASKSET_ID,
    FrozenEvalManifest,
    RLConfigIdentity,
    TrainingRecord,
    validate_manifest_contract,
    validate_model_snapshot,
    validate_training_prompt_hashes,
    validate_training_snapshot_location,
)


def validate_source_toml_contract(raw: dict, manifest: FrozenEvalManifest) -> None:
    try:
        deployment = raw["deployment"]
        trainer = raw["trainer"]
        trainer_model = trainer["model"]
        checkpoint_weights = trainer["ckpt"]["weights"]
        inference = raw["inference"]
        orchestrator = raw["orchestrator"]
        train_sources = orchestrator["train"]["source"]
        evaluation = orchestrator["eval"]
        eval_sources = evaluation["source"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"primary RL TOML is missing required contract field {exc}") from exc
    if raw.get("model", {}).get("name") != manifest.model.name:
        raise ValueError("source TOML model name does not match the frozen model name")
    if deployment.get("num_train_gpus") != 1 or deployment.get("num_infer_gpus") != 1:
        raise ValueError("source TOML must request exactly one trainer and one inference GPU")
    if trainer_model.get("lora", {}).get("rank") != 16:
        raise ValueError("source TOML must explicitly configure trainer rank-16 LoRA")
    if (
        checkpoint_weights.get("save_adapter_separately") is not True
        or checkpoint_weights.get("save_format", "safetensors") != "safetensors"
    ):
        raise ValueError("source TOML must save a separate safetensors LoRA adapter")
    for dtype in ("optimization_dtype", "reduce_dtype"):
        if dtype in trainer_model and trainer_model[dtype] != "float32":
            raise ValueError(f"source TOML must not change trainer.model.{dtype} from float32")
    if (
        inference.get("enable_lora") is not True
        or inference.get("seed") != 0
        or inference.get("model", {}).get("max_model_len") != 8192
        or orchestrator.get("renderer", {}).get("name") != "default"
        or orchestrator.get("model", {}).get("lora", {}).get("name") != "r16-math"
    ):
        raise ValueError("source TOML LoRA, renderer, or inference contract is incomplete")
    if len(train_sources) != 1:
        raise ValueError("source TOML must contain exactly one training source")
    train = train_sources[0]
    taskset = train.get("env", {}).get("taskset", {})
    if (
        taskset.get("id") != EXPECTED_TRAIN_TASKSET_ID
        or taskset.get("dataset_name") != manifest.train_taskset.dataset_name
        or taskset.get("dataset_subset") != manifest.train_taskset.dataset_subset
        or taskset.get("dataset_split") != manifest.train_taskset.dataset_split
        or taskset.get("task", {}).get("judge") != "None"
        or train.get("env", {}).get("agent", {}).get("harness", {}).get("id") != "null"
        or train.get("env", {}).get("agent", {}).get("runtime", {}).get("type") != "subprocess"
    ):
        raise ValueError("source TOML math-env-v1 task config does not match the deterministic contract")
    if (
        evaluation.get("interval") != 25
        or evaluation.get("num_examples") != 500
        or evaluation.get("group_size") != 1
        or len(eval_sources) != 1
        or eval_sources[0].get("env", {}).get("taskset", {}).get("id") != "math500-v1"
        or eval_sources[0].get("env", {}).get("agent", {}).get("harness", {}).get("id") != "null"
        or eval_sources[0].get("env", {}).get("agent", {}).get("runtime", {}).get("type") != "subprocess"
    ):
        raise ValueError("source TOML must configure all 500 math500-v1 rows with fixed eval settings")


def _reload_training_records(manifest: FrozenEvalManifest) -> list[TrainingRecord]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    taskset = manifest.train_taskset
    local_path = validate_training_snapshot_location(manifest)
    config = MathConfig(
        dataset_name=str(local_path),
        dataset_subset=taskset.dataset_subset,
        dataset_split=taskset.dataset_split,
    )
    return [
        TrainingRecord(
            id=task.data.idx,
            prompt_hash=hash_text(task.data.prompt),
            answer_hash=hash_text(str(task.data.answer)),
        )
        for task in MathTaskset(config).select()
    ]


def _resolve_rl_config(
    source_path: Path,
    manifest: FrozenEvalManifest,
    *,
    output_dir: Path,
    max_steps: int,
):
    from prime_rl.configs.rl import RLConfig

    validate_model_snapshot(manifest.model)
    raw = tomllib.loads(Path(source_path).read_text())
    validate_source_toml_contract(raw, manifest)
    raw["model"]["name"] = manifest.model.local_path
    raw["max_steps"] = max_steps
    raw["output_dir"] = str(output_dir)
    sources = raw["orchestrator"]["train"]["source"]
    if len(sources) != 1 or sources[0]["env"]["taskset"]["id"] != EXPECTED_TRAIN_TASKSET_ID:
        raise ValueError("source TOML must contain exactly one math-env-v1 training source")
    sources[0]["env"]["taskset"]["dataset_name"] = str(validate_training_snapshot_location(manifest))
    return RLConfig.model_validate(raw)


def _validate_effective_rl_config(config, manifest: FrozenEvalManifest, *, max_steps: int) -> None:
    model_path = manifest.model.local_path
    if config.model is None or config.model.name != model_path:
        raise ValueError("resolved RL config does not use the verified local model snapshot")
    if any(
        component.model.name != model_path
        for component in (config.trainer, config.orchestrator, config.inference)
        if component is not None
    ):
        raise ValueError("resolved component config model names do not match the verified local snapshot")
    if (
        config.max_steps != max_steps
        or config.trainer.max_steps != max_steps
        or config.orchestrator.max_steps != max_steps
    ):
        raise ValueError("resolved max_steps is not propagated identically to trainer and orchestrator")
    if (
        config.deployment.type != "single_node"
        or config.deployment.num_train_gpus != 1
        or config.deployment.num_infer_gpus != 1
    ):
        raise ValueError("primary RL contract requires exactly one trainer GPU and one inference GPU")

    trainer_lora = config.trainer.model.lora
    orchestrator_lora = config.orchestrator.model.lora
    if trainer_lora is None or trainer_lora.rank != 16:
        raise ValueError("primary RL contract requires trainer rank-16 LoRA")
    if (
        orchestrator_lora is None
        or orchestrator_lora.rank != trainer_lora.rank
        or orchestrator_lora.alpha != trainer_lora.alpha
        or orchestrator_lora.name != "r16-math"
    ):
        raise ValueError("orchestrator LoRA must match trainer rank/alpha and use fixed name 'r16-math'")
    if config.inference is None or not config.inference.enable_lora or config.inference.max_lora_rank != 16:
        raise ValueError("inference must enable rank-16 LoRA")
    if config.inference.seed != 0:
        raise ValueError("inference seed must be 0")
    if config.trainer.model.optimization_dtype != "float32" or config.trainer.model.reduce_dtype != "float32":
        raise ValueError("trainer optimization_dtype and reduce_dtype must retain their float32 defaults")
    if (
        config.trainer.ckpt is None
        or config.trainer.ckpt.weights is None
        or not config.trainer.ckpt.weights.save_adapter_separately
        or config.trainer.ckpt.weights.save_format != "safetensors"
    ):
        raise ValueError("trainer must save a separate safetensors LoRA adapter")
    if config.orchestrator.renderer.name != "default":
        raise ValueError("primary RL contract requires the explicit default renderer")

    train_sources = config.orchestrator.train.source
    if len(train_sources) != 1:
        raise ValueError("training config must contain exactly one training source")
    train_source = train_sources[0]
    taskset = train_source.env.taskset
    expected_taskset = manifest.train_taskset
    if (
        taskset.id != EXPECTED_TRAIN_TASKSET_ID
        or taskset.dataset_name != expected_taskset.dataset_local_path
        or taskset.dataset_subset != expected_taskset.dataset_subset
        or taskset.dataset_split != expected_taskset.dataset_split
        or taskset.question_key != "question"
        or taskset.answer_key != "answer"
        or taskset.task.judge is not None
    ):
        raise ValueError("resolved math-env-v1 task config does not match the materialized deterministic contract")
    if train_source.env.agent.harness.id != "null" or train_source.env.agent.runtime.type != "subprocess":
        raise ValueError("training source must use the null harness and subprocess runtime")
    if train_source.sampling.temperature != 1.0 or train_source.group_size != 8:
        raise ValueError("training sampling/group settings drifted from the primary contract")

    evaluation = config.orchestrator.eval
    if (
        evaluation is None
        or evaluation.num_examples != 500
        or evaluation.group_size != 1
        or evaluation.interval != 25
        or evaluation.skip_first_step
        or len(evaluation.source) != 1
    ):
        raise ValueError("primary RL contract requires startup/periodic/final evaluation over all 500 rows")
    eval_source = evaluation.source[0]
    if (
        eval_source.env.taskset.id != "math500-v1"
        or eval_source.num_examples != 500
        or eval_source.group_size != 1
        or eval_source.interval != 25
        or eval_source.env.agent.harness.id != "null"
        or eval_source.env.agent.runtime.type != "subprocess"
        or eval_source.sampling.model_dump()
        != {
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "max_completion_tokens": None,
            "reasoning_effort": None,
            "extra_body": {},
        }
    ):
        raise ValueError("resolved math500-v1 evaluation source does not match the fixed 500-row decoding contract")


def resolve_effective_rl_config(
    source_path: Path,
    manifest: FrozenEvalManifest,
    *,
    source_config_rel: str,
    output_dir: Path,
    max_steps: int,
) -> tuple[object, bytes, RLConfigIdentity]:
    import tomli_w

    from prime_rl.utils.config import to_toml_dict

    config = _resolve_rl_config(source_path, manifest, output_dir=output_dir, max_steps=max_steps)
    _validate_effective_rl_config(config, manifest, max_steps=max_steps)
    resolved_bytes = tomli_w.dumps(to_toml_dict(config)).encode("utf-8")
    identity = RLConfigIdentity(
        source_config_rel=source_config_rel,
        output_dir=str(output_dir),
        max_steps=max_steps,
        resolved_toml_sha256=hashlib.sha256(resolved_bytes).hexdigest(),
    )
    return config, resolved_bytes, identity


def _write_resolved_config(output_path: Path, resolved_bytes: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        output.write(resolved_bytes)
        output.flush()
        os.fsync(output.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-rel", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--preflight", required=True)
    args = parser.parse_args(argv)

    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("HF_DATASETS_OFFLINE=1 is required while validating the pinned training snapshot")
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    if manifest.state != "finalized":
        raise ValueError("training-data validation requires a finalized manifest")
    validate_manifest_contract(
        manifest,
        expected_source_revision=manifest.source_revision,
        expected_verifiers_revision=manifest.verifiers_revision,
        expected_tasksets_revision=manifest.eval_taskset.taskset_revision,
        expected_model_name=manifest.model.name,
        expected_model_revision=manifest.model.revision,
        require_finalized=True,
    )
    eval_examples = _reload_and_verify_examples(manifest)
    records = _reload_training_records(manifest)
    validate_training_prompt_hashes(manifest, records)
    _, resolved_bytes, config_identity = resolve_effective_rl_config(
        Path(args.config),
        manifest,
        source_config_rel=args.config_rel,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    if manifest.rl_config != config_identity:
        raise ValueError(
            "effective RL config identity does not match the finalized manifest "
            f"(actual={config_identity.resolved_toml_sha256}, "
            f"expected={manifest.rl_config.resolved_toml_sha256 if manifest.rl_config else None})"
        )
    _write_resolved_config(Path(args.output_config), resolved_bytes)
    written_digest = hashlib.sha256(Path(args.output_config).read_bytes()).hexdigest()
    if written_digest != config_identity.resolved_toml_sha256:
        raise RuntimeError("written resolved RL TOML does not match its bound config digest")
    validate_training_prompt_hashes(manifest, _reload_training_records(manifest))
    write_training_preflight(
        output_path=Path(args.preflight),
        manifest=manifest,
        config_identity=config_identity,
    )
    print(
        f"ok: revalidated {len(eval_examples)} eval examples and {len(records)} "
        f"ordered training records (sha256={manifest.training_data.record_digest}); "
        f"wrote resolved RL config {config_identity.resolved_toml_sha256} to {args.output_config}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
