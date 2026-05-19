---
name: config
description: How the prime-rl config system works — TOML files, CLI, config composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Config

prime-rl uses `pydantic_config` (combines `tyro` and `pydantic`) for configuration. 

## Use configs

Every entrypoint accepts TOML files via `@` syntax and CLI overrides to configure it.

```bash
# Configure RL training with a TOML file
uv run rl @ examples/reverse_text/rl.toml

# Override specific fields via CLI
uv run rl @ examples/reverse_text/rl.toml --max-steps 50
```

Config resolve in the following order:

1. CLI arguments
2. Config files (merged left-to-right)
3. Class defaults (lowest)

## Compose configs

Multiple config files are merged left-to-right (later files override earlier ones):

```bash
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml
```

Nested configs can be loaded for specific sections:

```bash
uv run rl --model @ model.toml --data @ data.toml
```

Mixed composition works too:

```bash
uv run rl @ base.toml --trainer @ trainer_override.toml --trainer.lr 1e-3
```

Merging is deep — unset fields in the override are preserved from the base config.

## Inspect & validate configs

Use `--help` to see all available fields and their defaults. When combined with a config file, defaults reflect the TOML values:

```bash
uv run rl --help                                  # shows class defaults
uv run rl @ examples/reverse_text/rl.toml --help  # shows defaults from TOML
```

Use `--dry-run` to validate and dump the fully resolved config:

```bash
uv run rl @ examples/reverse_text/rl.toml --dry-run --output-dir /tmp/test
# Writes resolved TOML to /tmp/test/configs
```

## Naming

CLI uses kebab-case (`--model.max-model-len`), TOML uses snake_case (`max_model_len`). Both refer to the same field.

## General rules

- **Fail early**: incompatible option combinations (e.g. CP requires flash attention, NCCL broadcast requires async level 1) should raise in `model_validator` at config resolution time, not at runtime. When adding new constraints, add a validator to the config class.
- **Deprecation**: when renaming or removing config fields, emit a deprecation warning with a clear migration path (e.g. "field X is deprecated, use Y instead"). Do not silently drop fields — help users update their configs.

## Important patterns

### Boolean fields

```bash
uv run inference --model.enforce-eager          # sets to true
uv run inference --model.no-enforce-eager       # sets to false
```

In TOML, booleans must be explicit:

```toml
[model]
enforce_eager = true
```

### None fields

TOML has no null type. Use the string `"None"`:

```toml
max_model_len = "None"
```

On the CLI, pass `None` as a plain string:

```bash
uv run inference --model.max-model-len None
```

### List fields

In TOML, use `[[double brackets]]` (array of tables) for lists of objects:

```toml
[[orchestrator.env]]
id = "reverse-text"

[[orchestrator.env]]
id = "math-env"
```

On the CLI, list items are indexed: `--env.0.id reverse-text --env.1.id math-env`.

When composing multiple TOML files, list fields are replaced wholesale by the later file. To change one
`orchestrator.filters` entry in an overlay, include the full desired filter list in that overlay.

For quick KL smoke runs with very small rollout batches, enforced `zero_advantage` filtering can remove
every rollout and stop the orchestrator. If the goal is only trainer/inference mismatch KL, keep the
filter present but set `enforce = false` in a temporary overlay and call out that the run is not a reward
learning validation.

### Dict fields

In TOML, use a section:

```toml
[vllm_extra]
key1 = "value1"
key2 = 123
```

On the CLI, pass as a JSON string:

```bash
uv run inference --vllm-extra '{"key1": "value1", "key2": 123}'
```

### Discriminated unions

Some config fields use discriminated unions (e.g. loss type, data type). Set the `type` field to select the variant:

```toml
[trainer.loss]
type = "sft"

[data]
type = "fake"
batch_size = 2
```

On the CLI:

```bash
uv run sft --data.type fake --data.batch-size 4
```

If you wish to configure values of the default variant, you don't need to set the `type` field.

### Ray-native settings

The Ray runtime is explicit and experimental. Enable Ray-native roles under `[experimental.ray]`:

```toml
[experimental.ray]
enabled = true
namespace = "prime-rl"
```

Ray-native mode supports `deployment.type = "single_node"` for local Ray and `deployment.type = "ray_cluster"` for Ray-native multi-node placement. It runs inference and orchestrator workers as in-process Ray tasks. Inference uses `experimental.ray.inference_backend = "prime_vllm"` by default, which runs Prime-RL's existing vLLM server inside a Ray GPU task and preserves custom inference endpoints plus filesystem/NCCL weight updates. If Ray places inference on a different node than the orchestrator, localhost rollout/admin URLs are rewritten to the inference Ray node IP in the orchestrator config copy. Teacher inference is also supported when `deployment.num_teacher_gpus` is set; localhost teacher-model URLs are rewritten to the teacher Ray node IP. By default, trainer rank workers also run as Ray tasks and call `train(config)` directly. Set `experimental.ray.trainer_backend = "ray_train"` to run trainer workers through Ray Train's `TorchTrainer` instead.

Ray-native mode requires Ray rollout transport on both trainer and orchestrator:

```toml
[trainer.rollout_transport]
type = "ray"
address = "auto"
namespace = "prime-rl"
actor_name = "prime-rl-transport"

[orchestrator.rollout_transport]
type = "ray"
address = "auto"
namespace = "prime-rl"
actor_name = "prime-rl-transport"
```

Use `address = "auto"` when trainer and orchestrator Ray tasks should attach to the Ray runtime started by the native launcher.

Optional Ray Train settings live under `[experimental.ray]`:

```toml
[experimental.ray]
trainer_backend = "ray_train"
inference_backend = "prime_vllm"
train_run_name = "my-run"
train_storage_path = "/shared/ray-train"
```

Use shared `train_storage_path` for multi-node RayCluster runs. The current Ray Train backend still keeps Prime-RL's vLLM inference and filesystem/NCCL weight broadcast contracts.

For multi-node RayCluster runs, use the Ray-native deployment variant, run the launcher as a Ray job or from the head pod, and use `address = "auto"`. A normal Kubernetes pod pointed at the head service GCS port has no local raylet and will fail during worker creation. Use Ray `runtime_env` so remote worker pods can import the Prime-RL checkout:

```toml
[deployment]
type = "ray_cluster"
gpus_per_node = 1
num_train_gpus = 1
num_infer_gpus = 1

[experimental.ray]
address = "auto"
placement_strategy = "SPREAD"

[experimental.ray.runtime_env]
working_dir = "/shared/checkouts/prime-rl"

[experimental.ray.runtime_env.env_vars]
PYTHONPATH = "/shared/checkouts/prime-rl/src:/shared/checkouts/prime-rl/packages/prime-rl-configs/src"
```

Do not use `deployment.type = "multi_node"` for Ray. That remains the SLURM template path. `ray_cluster` uses logical Ray role resources: `num_train_gpus` is the Ray Train worker count and can span nodes; `num_infer_gpus`/`num_teacher_gpus` are single Prime-vLLM task GPU reservations and must fit within `gpus_per_node`.

The accepted Ray-native weight update path reuses Prime-RL's HF-compatible filesystem broadcast by default; do not add `weight_broadcast.type = "ray"` unless a separate design proves a better full-model live update path.

### SFT hard distill override

For hosted multi-tenant runs where the trainer image's `trainer.loss.type` is fixed, the orchestrator exposes a per-run override that forces SFT loss on every micro-batch without rebuilding the trainer. Set `orchestrator.use_sft_loss = true` alongside `orchestrator.teacher_rollout_model`; both must be configured together (the orchestrator validator enforces this). The orchestrator stamps each `TrainingSample.sft_loss = True`, which the trainer's `compute_loss` honors by dispatching to `sft_loss_fn` per batch — independent of the trainer's configured default loss.

### RL rollout client defaults

For text-only RL rollouts, the orchestrator defaults to renderer-backed TITO (`use_renderer = true`). VLM configs must explicitly fall back to MITO (`use_renderer = false`) so image preprocessing and chat templating stay server-side. External teacher rollouts must also set `use_renderer = false`.

### Model fields

For `BaseModel | None` fields (like `[ckpt]`, `[wandb]`, `[compile]`), a bare flag enables them with defaults:

```bash
uv run rl @ config.toml --model.compile              # enables compilation with defaults (fullgraph = false)
uv run rl @ config.toml --model.compile.fullgraph    # enables compilation and sets nested field (fullgraph = true)
```

In TOML, an empty section header does the same:

```toml
[ckpt]  # enables checkpointing with defaults
```

## Key files

- `src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli` from pydantic_config
- `src/prime_rl/configs/` — all domain-specific config classes
- `configs/debug/` — minimal debug configs for testing
- `configs/private/` — private configs via git submodule (internal only)
- `examples/` — full example configs for various tasks
