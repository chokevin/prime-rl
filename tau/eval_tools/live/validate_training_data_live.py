from __future__ import annotations

import hashlib
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.manifest import (
    EXPECTED_TRAIN_TASKSET_ID,
    FrozenEvalManifest,
    RLConfigIdentity,
    TrainingRecord,
)


@dataclass(frozen=True)
class SourceConfigContract:
    source_config_rel: str
    source_toml_sha256: str
    max_steps: int
    eval_interval: int


SOURCE_CONFIG_CONTRACTS = {
    "configs/tau/math-7b-h200/train.toml": SourceConfigContract(
        source_config_rel="configs/tau/math-7b-h200/train.toml",
        source_toml_sha256="7617f4d2e574ac4eb834b488b8b43d4d9f5034e89e92ea7af761f3c3198a6c2c",
        max_steps=50,
        eval_interval=25,
    ),
    "configs/tau/math-7b-h200/train-f12.toml": SourceConfigContract(
        source_config_rel="configs/tau/math-7b-h200/train-f12.toml",
        source_toml_sha256="5450d2ff141172a4ababc5dfe3df572f2764cba5c6e998d982bb21d270e9e2ff",
        max_steps=200,
        eval_interval=100,
    ),
}
PRIVATE_MODEL_SENTINEL = "/__prime_rl_private__/model"
PRIVATE_DATASET_SENTINEL = "/__prime_rl_private__/training-dataset"
EXPECTED_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
    "fc1_latent_proj",
    "fc2_latent_proj",
]


def source_config_contract(source_config_rel: str) -> SourceConfigContract:
    try:
        return SOURCE_CONFIG_CONTRACTS[source_config_rel]
    except KeyError:
        raise ValueError(f"unsupported immutable RL source config: {source_config_rel}") from None


def _source_config_contract_for_bytes(
    source_bytes: bytes,
    source_config_rel: str | None = None,
) -> SourceConfigContract:
    digest = hashlib.sha256(source_bytes).hexdigest()
    if source_config_rel is not None:
        contract = source_config_contract(source_config_rel)
        if digest != contract.source_toml_sha256:
            raise ValueError(f"RL TOML does not match canonical experiment contract {contract.source_toml_sha256}")
        return contract
    for contract in SOURCE_CONFIG_CONTRACTS.values():
        if digest == contract.source_toml_sha256:
            return contract
    raise ValueError(f"RL TOML does not match any canonical experiment contract: {digest}")


def _read_regular_file(path: Path) -> bytes:
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"resolved RL config contains a symlink component: {current}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("resolved RL config must be a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("resolved RL config changed while it was being read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def validate_source_toml_contract(
    source_bytes: bytes,
    raw: dict,
    manifest: FrozenEvalManifest,
    *,
    source_config_rel: str | None = None,
) -> SourceConfigContract:
    contract = _source_config_contract_for_bytes(source_bytes, source_config_rel)
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
    if raw.get("max_steps") != contract.max_steps:
        raise ValueError(f"source TOML max_steps must be exactly {contract.max_steps}")
    if deployment.get("num_train_gpus") != 1 or deployment.get("num_infer_gpus") != 1:
        raise ValueError("source TOML must request exactly one trainer and one inference GPU")
    if trainer_model.get("impl") != "hf" or trainer_model.get("attn") != "flash_attention_2":
        raise ValueError("source TOML must use the Qwen2.5-compatible HF trainer with flash attention 2")
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
        evaluation.get("interval") != contract.eval_interval
        or evaluation.get("num_examples") != 500
        or evaluation.get("group_size") != 1
        or len(eval_sources) != 1
        or eval_sources[0].get("env", {}).get("taskset", {}).get("id") != "math500-v1"
        or eval_sources[0].get("env", {}).get("agent", {}).get("harness", {}).get("id") != "null"
        or eval_sources[0].get("env", {}).get("agent", {}).get("runtime", {}).get("type") != "subprocess"
    ):
        raise ValueError("source TOML must configure all 500 math500-v1 rows with fixed eval settings")
    return contract


def _reload_training_records(manifest: FrozenEvalManifest, dataset_path: Path) -> list[TrainingRecord]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    taskset = manifest.train_taskset
    config = MathConfig(
        dataset_name=str(dataset_path),
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
    source_config_rel: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int,
):
    from prime_rl.configs.rl import RLConfig

    source_bytes = Path(source_path).read_bytes()
    raw = tomllib.loads(source_bytes.decode("utf-8"))
    contract = validate_source_toml_contract(
        source_bytes,
        raw,
        manifest,
        source_config_rel=source_config_rel,
    )
    if max_steps != contract.max_steps:
        raise ValueError(
            f"requested max_steps {max_steps} does not match {source_config_rel} contract {contract.max_steps}"
        )
    _bind_run_specific_config(
        raw,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    return RLConfig.model_validate(raw)


def _bind_run_specific_config(
    raw: dict,
    *,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int,
) -> None:
    raw["model"]["name"] = str(model_path)
    raw["max_steps"] = max_steps
    raw["output_dir"] = str(output_dir)
    sources = raw["orchestrator"]["train"]["source"]
    if len(sources) != 1 or sources[0]["env"]["taskset"]["id"] != EXPECTED_TRAIN_TASKSET_ID:
        raise ValueError("source TOML must contain exactly one math-env-v1 training source")
    sources[0]["env"]["taskset"]["dataset_name"] = str(dataset_path)


def _validate_effective_rl_config(
    config,
    manifest: FrozenEvalManifest,
    *,
    source_config_rel: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int,
) -> None:
    contract = source_config_contract(source_config_rel)
    if max_steps != contract.max_steps:
        raise ValueError(
            f"resolved max_steps {max_steps} does not match {source_config_rel} contract {contract.max_steps}"
        )
    model_path_str = str(model_path)
    if config.model is None or config.model.name != model_path_str:
        raise ValueError("resolved RL config does not use the verified local model snapshot")
    if any(
        component.model.name != model_path_str
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
    if (
        trainer_lora is None
        or trainer_lora.rank != 16
        or trainer_lora.alpha != 32.0
        or trainer_lora.dropout != 0.0
        or trainer_lora.target_modules != EXPECTED_LORA_TARGET_MODULES
        or trainer_lora.modules_to_save != []
    ):
        raise ValueError("trainer LoRA parameters drifted from the canonical rank-16 contract")
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
    if (
        config.clean_output_dir
        or config.dry_run
        or config.bench
        or config.inference.dry_run
        or config.orchestrator.bench
        or config.trainer.bench is not None
    ):
        raise ValueError("clean-output, dry-run, benchmark, and fake execution bypasses are forbidden")
    if config.trainer.model.optimization_dtype != "float32" or config.trainer.model.reduce_dtype != "float32":
        raise ValueError("trainer optimization_dtype and reduce_dtype must retain their float32 defaults")
    if (
        config.trainer.ckpt is None
        or config.trainer.ckpt.weights is None
        or not config.trainer.ckpt.weights.save_adapter_separately
        or config.trainer.ckpt.weights.save_format != "safetensors"
    ):
        raise ValueError("trainer must save a separate safetensors LoRA adapter")
    trainer_ckpt = config.trainer.ckpt
    if (
        trainer_ckpt.resume_step is not None
        or trainer_ckpt.weights_only
        or trainer_ckpt.skip_gather_master_weights
        or trainer_ckpt.skip_progress
        or trainer_ckpt.skip_dataloader
        or trainer_ckpt.skip_optimizer
        or trainer_ckpt.skip_scheduler
    ):
        raise ValueError("resume/restart and partial checkpoint-state execution are forbidden")
    orchestrator_ckpt = config.orchestrator.ckpt
    if orchestrator_ckpt is None or orchestrator_ckpt.resume_step is not None or orchestrator_ckpt.skip_progress:
        raise ValueError("orchestrator checkpointing must be enabled without resume/skip bypasses")
    if (
        config.output_dir != output_dir
        or config.trainer.output_dir != output_dir
        or config.orchestrator.output_dir != output_dir / "run_default"
        or config.trainer.model.seq_len != 16384
        or config.orchestrator.seq_len != 8192
        or config.inference.model.max_model_len != 8192
    ):
        raise ValueError("resolved output or sequence/context length contract drifted")
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
        or taskset.dataset_name != str(dataset_path)
        or taskset.dataset_subset != expected_taskset.dataset_subset
        or taskset.dataset_split != expected_taskset.dataset_split
        or taskset.question_key != "question"
        or taskset.answer_key != "answer"
        or taskset.task.judge is not None
    ):
        raise ValueError("resolved math-env-v1 task config does not match the materialized deterministic contract")
    if train_source.env.agent.harness.id != "null" or train_source.env.agent.runtime.type != "subprocess":
        raise ValueError("training source must use the null harness and subprocess runtime")
    if (
        train_source.sampling.temperature != 1.0
        or train_source.sampling.max_completion_tokens is not None
        or train_source.group_size != 8
        or config.orchestrator.batch_size != 128
    ):
        raise ValueError("training sampling/group settings drifted from the primary contract")

    evaluation = config.orchestrator.eval
    if (
        evaluation is None
        or evaluation.num_examples != 500
        or evaluation.group_size != 1
        or evaluation.interval != contract.eval_interval
        or evaluation.skip_first_step
        or len(evaluation.source) != 1
    ):
        raise ValueError("primary RL contract requires startup/periodic/final evaluation over all 500 rows")
    eval_source = evaluation.source[0]
    if (
        eval_source.env.taskset.id != "math500-v1"
        or eval_source.num_examples != 500
        or eval_source.group_size != 1
        or eval_source.interval != contract.eval_interval
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


def _replace_runtime_paths(
    value,
    *,
    model_path: str,
    dataset_path: str,
    output_dir: str,
    logical_output_dir: str,
):
    model_path = os.path.normpath(os.fspath(model_path))
    dataset_path = os.path.normpath(os.fspath(dataset_path))
    output_dir = os.path.normpath(os.fspath(output_dir))
    logical_output_dir = os.path.normpath(os.fspath(logical_output_dir))
    if isinstance(value, dict):
        return {
            key: _replace_runtime_paths(
                item,
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                logical_output_dir=logical_output_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_runtime_paths(
                item,
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                logical_output_dir=logical_output_dir,
            )
            for item in value
        ]
    if isinstance(value, str):
        for private_root, logical_root in (
            (model_path, PRIVATE_MODEL_SENTINEL),
            (dataset_path, PRIVATE_DATASET_SENTINEL),
            (output_dir, logical_output_dir),
        ):
            if value == private_root:
                return logical_root
            if value.startswith(private_root + os.sep):
                return logical_root + value[len(private_root) :]
    return value


def _resolved_config_identity(
    config,
    *,
    source_config_rel: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    logical_output_dir: Path,
    max_steps: int,
) -> tuple[bytes, RLConfigIdentity]:
    import tomli_w

    from prime_rl.utils.config import to_toml_dict

    model_path = os.path.normpath(os.fspath(model_path))
    dataset_path = os.path.normpath(os.fspath(dataset_path))
    output_dir = os.path.normpath(os.fspath(output_dir))
    logical_output_dir = os.path.normpath(os.fspath(logical_output_dir))
    resolved_dict = to_toml_dict(config)
    resolved_bytes = tomli_w.dumps(resolved_dict).encode("utf-8")
    canonical_dict = _replace_runtime_paths(
        resolved_dict,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        logical_output_dir=logical_output_dir,
    )
    canonical_bytes = tomli_w.dumps(canonical_dict).encode("utf-8")
    identity = RLConfigIdentity(
        source_config_rel=source_config_rel,
        source_toml_sha256=source_config_contract(source_config_rel).source_toml_sha256,
        output_dir=logical_output_dir,
        max_steps=max_steps,
        canonical_resolved_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )
    return resolved_bytes, identity


def resolve_effective_rl_config(
    source_path: Path,
    manifest: FrozenEvalManifest,
    *,
    source_config_rel: str,
    model_path: Path = Path(PRIVATE_MODEL_SENTINEL),
    dataset_path: Path = Path(PRIVATE_DATASET_SENTINEL),
    output_dir: Path,
    logical_output_dir: Path | None = None,
    max_steps: int,
) -> tuple[object, bytes, RLConfigIdentity]:
    logical_output_dir = logical_output_dir or output_dir
    config = _resolve_rl_config(
        source_path,
        manifest,
        source_config_rel=source_config_rel,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    _validate_effective_rl_config(
        config,
        manifest,
        source_config_rel=source_config_rel,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    resolved_bytes, identity = _resolved_config_identity(
        config,
        source_config_rel=source_config_rel,
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        output_dir=output_dir,
        logical_output_dir=logical_output_dir,
        max_steps=max_steps,
    )
    return config, resolved_bytes, identity


def validate_resolved_rl_config(
    resolved_config_path: Path,
    manifest: FrozenEvalManifest,
    *,
    source_config_rel: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    logical_output_dir: Path,
    max_steps: int,
) -> tuple[object, bytes, RLConfigIdentity]:
    from prime_rl.configs.rl import RLConfig

    resolved_config_path = Path(resolved_config_path)
    input_bytes = _read_regular_file(resolved_config_path)
    raw = tomllib.loads(input_bytes.decode("utf-8"))
    config = RLConfig.model_validate(raw)
    _validate_effective_rl_config(
        config,
        manifest,
        source_config_rel=source_config_rel,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    canonical_resolved_bytes, identity = _resolved_config_identity(
        config,
        source_config_rel=source_config_rel,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        logical_output_dir=logical_output_dir,
        max_steps=max_steps,
    )
    if input_bytes != canonical_resolved_bytes:
        raise ValueError("resolved RL TOML is not the exact canonical serialization consumed by rl")
    return config, input_bytes, identity
