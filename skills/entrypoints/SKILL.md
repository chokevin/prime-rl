---
name: entrypoints
description: All available prime-rl entrypoints — what they do, how to launch them, and which config class they use. Use when running commands, launching training, or starting servers.
---

# Entrypoints

All entrypoints are run via `uv run <command>` and accept TOML configs via `@ path/to/config.toml` with CLI overrides. See the `config` skill for config system details.

## `rl` — RL training

Orchestrates the complete RL loop. By default it launches inference server, orchestrator, and trainer as subprocesses; with `experimental.ray.enabled` it runs `single_node` or `ray_cluster` deployment configs under Ray tasks/Ray Train. Ray inference uses `experimental.ray.inference_backend = "prime_vllm"` to run Prime-RL's existing vLLM server inside a Ray GPU task. If Ray places inference away from the orchestrator, the launcher rewrites default localhost rollout URLs to the inference node IP; when `deployment.num_teacher_gpus` starts teacher inference, localhost teacher-model URLs are rewritten to the teacher node IP. The default Ray trainer backend uses one Ray task per trainer rank; `experimental.ray.trainer_backend = "ray_train"` uses Ray Train's `TorchTrainer`.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml # with SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run # generate scripts without running
uv run rl @ examples/reverse_text/rl.toml --experimental.ray.enabled \
  --trainer.rollout-transport.type ray \
  --orchestrator.rollout-transport.type ray # experimental Ray-native path
uv run rl @ examples/reverse_text/rl.toml --experimental.ray.enabled \
  --experimental.ray.trainer-backend ray_train \
  --trainer.rollout-transport.type ray \
  --orchestrator.rollout-transport.type ray # experimental Ray Train trainer backend
```

- **Config:** `RLConfig` (`src/prime_rl/configs/rl.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/rl.py`
- **SLURM:** yes — single-node and multi-node
- **Ray:** experimental fork path for `deployment.type = "single_node"` and `deployment.type = "ray_cluster"`. The old `multi_node` deployment remains SLURM-shaped. Ray mode runs Prime-RL's vLLM inference server and orchestrator as Ray tasks, rewrites local inference/teacher client URLs when tasks land on different nodes, and runs trainer ranks either as Ray tasks with explicit torch distributed rank env or through Ray Train's `TorchTrainer` when `experimental.ray.trainer_backend = "ray_train"`.

## `sft` — SFT training

Trains a model on labeled data. Uses torchrun for distributed training.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm # with SLURM
uv run sft @ examples/reverse_text/sft.toml --dry-run # generate scripts without running
```

The entrypoint launches torchrun internally — no need to call torchrun directly.

- **Config:** `SFTConfig` (`src/prime_rl/configs/sft.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/sft.py`
- **SLURM:** yes — single-node and multi-node

## `inference` — Standalone inference server

Launches a vLLM-based inference server with OpenAI-compatible API.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Always use the `inference` entrypoint — never `vllm serve` directly.

Custom endpoints beyond standard OpenAI API:
- `/v1/chat/completions/tokens` — accepts token IDs as prompt input
- `/update_weights` — hot-reload model weights from the trainer
- `/load_lora_adapter` — load LoRA adapters at runtime
- `/init_broadcaster` — initialize weight broadcast for distributed training

Check health with:
```bash
curl http://<ip>:<port>/health
```

Check served models with:
```bash
curl http://<ip>:<port>/v1/models
```

Test chat completions with:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- **Config:** `InferenceConfig` (`src/prime_rl/configs/inference.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/inference.py`
- **SLURM:** yes — single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | SLURM | Typical use |
|---------|---------|-------|-------------|
| `rl` | Full RL pipeline | yes | Production RL training |
| `sft` | Supervised fine-tuning | yes | SFT training |
| `inference` | vLLM server | yes | Standalone inference or debugging |

## Key directories

- `src/prime_rl/entrypoints/` — top-level entrypoints (`rl`, `sft`, `inference`)
- `src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal configs for quick testing
- `examples/` — full example configs for various tasks
