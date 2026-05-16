# Ray-native fork architecture

Prime-RL's stable runtime remains process-role based: `rl` writes resolved TOML subconfigs, launches `inference`, `orchestrator`, and a `torchrun` trainer, and coordinates rollout transport through filesystem or ZMQ backends. The RayJob launcher path uses Ray as a cluster launch substrate for those same roles.

This fork adds a Ray-native runtime path. When `experimental.ray.enabled = true`, Ray launches Prime-RL roles as in-process Ray tasks instead of shelling out to the CLI role commands.

```bash
uv pip install "ray[default,train]>=2.40.0"
uv run rl @ examples/reverse_text/rl.toml \
  --experimental.ray.enabled \
  --trainer.rollout-transport.type ray \
  --orchestrator.rollout-transport.type ray
```

## What Ray owns

- **Role lifecycle**: Ray tasks run inference and orchestrator; trainer ranks run either as direct Ray tasks or Ray Train workers.
- **Placement accounting**: the Ray task backend reserves inference GPUs and one GPU per trainer rank in a placement group. The Ray Train backend lets `TorchTrainer` own the trainer worker placement group.
- **Inference backend**: `experimental.ray.inference_backend = "prime_vllm"` runs Prime-RL's existing vLLM server in a Ray GPU task, preserving custom routes and filesystem/NCCL weight updates.
- **Distributed trainer execution**: by default Ray trainer tasks set `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`, then call `train(config)`. With `experimental.ray.trainer_backend = "ray_train"`, `ray.train.torch.TorchTrainer` owns trainer worker orchestration and Prime-RL reuses Ray Train's distributed process group.
- **Failure surfacing**: failed Ray role tasks fail the run and point at the role log file.
- **Rollout transport**: `rollout_transport.type = "ray"` moves `TrainingBatch` and packed micro-batches through a named Ray actor instead of filesystem or ZMQ.

Inference calls Prime-RL's Python `inference_local(config)` function inside a Ray GPU task. The orchestrator calls `asyncio.run(orchestrate(config))`. Trainer ranks call `train(config)` directly after Ray assigns GPUs and rank environment, or through Ray Train workers when `trainer_backend = "ray_train"`.

## Example config

```toml
[experimental.ray]
enabled = true
namespace = "prime-rl"
placement_strategy = "STRICT_PACK"
inference_backend = "prime_vllm"
trainer_backend = "tasks"  # or "ray_train"

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

`experimental.ray.address` configures the launcher-side `ray.init(...)` call. If unset, Ray uses its default behavior and may start a local Ray runtime. `rollout_transport.address = "auto"` lets trainer and orchestrator Ray tasks attach to that runtime.

Prime-RL disables Ray's automatic `uv run` runtime-env propagation for this path so Ray workers use the same active Python environment as the launcher. Install Ray into the active environment before launching `rl`.

For multi-node RayCluster validation, set `experimental.ray.address` to the Ray head service and use `experimental.ray.runtime_env` to make the fork checkout importable on remote worker pods:

```toml
[experimental.ray]
address = "my-raycluster-head-svc.ray.svc.cluster.local:6379"
placement_strategy = "SPREAD"

[experimental.ray.runtime_env]
working_dir = "/shared/checkouts/prime-rl"

[experimental.ray.runtime_env.env_vars]
PYTHONPATH = "/shared/checkouts/prime-rl/src:/shared/checkouts/prime-rl/packages/prime-rl-configs/src"
```

## Ray Train backend

Set `experimental.ray.trainer_backend = "ray_train"` to run trainer ranks with `ray.train.torch.TorchTrainer` instead of one manual Ray task per rank:

```bash
uv run rl @ examples/reverse_text/rl.toml \
  --experimental.ray.enabled \
  --experimental.ray.trainer-backend ray_train \
  --trainer.rollout-transport.type ray \
  --orchestrator.rollout-transport.type ray
```

The Ray Train backend keeps Prime-RL's existing trainer loop and calls `train(config)` inside each Ray Train worker. The trainer setup code reuses an already-initialized Ray Train `torch.distributed` process group instead of calling `dist.init_process_group` a second time. `experimental.ray.train_run_name` and `experimental.ray.train_storage_path` are passed to Ray Train's `RunConfig` when set; use shared storage for future multi-node RayCluster validation.

## Ray inference backend

The supported Ray inference backend is `prime_vllm`. It deliberately reuses the same Prime-RL vLLM server used by the standalone `inference` entrypoint:

1. `InferenceConfig` is translated to vLLM CLI arguments with `InferenceConfig.to_vllm()`.
2. Prime-RL installs its custom vLLM routes and patches, including `/v1/chat/completions/tokens`, `/pause`, `/resume`, `/update_weights`, `/liveness`, and `/init_broadcaster`.
3. The vLLM worker extension remains selected from `inference.weight_broadcast.type`, so existing filesystem and NCCL weight-update paths continue to work.

Ray owns placement, lifecycle, and log/failure surfacing for this server. It does not replace the server implementation with Ray Serve LLM.

## Ray Serve and weight sharing assessment

Ray supports vLLM through Ray Serve LLM and Ray Data LLM, but those APIs do not directly replace Prime-RL's current vLLM server contract. Prime-RL depends on custom endpoints such as `/v1/chat/completions/tokens`, `/pause`, `/resume`, `/update_weights`, `/liveness`, and `/init_broadcaster`. A Ray Serve backend needs a Prime-RL compatibility facade before it can replace the current vLLM server.

Ray Train checkpoint/storage APIs are the right durable checkpoint path, but Ray's object store is not a reliable cluster-wide weight broadcast bus for full model updates. The supported Ray-native path keeps Prime-RL's existing HF-compatible filesystem weight broadcast (or NCCL where configured): trainer writes the update under the broadcast directory, marks it `STABLE`, and the Prime-vLLM inference task reloads through the existing `/update_weights` endpoint.

## Current constraints

- Only `deployment.type = "single_node"` is supported.
- SLURM mode is not supported by `experimental.ray.enabled`.
- Ray is an optional runtime dependency for this fork; non-Ray Prime-RL paths do not import Ray.
- `trainer.rollout_transport.type` and `orchestrator.rollout_transport.type` must both be `ray`.
- The Ray-native `rl` launcher owns the shared transport actor; Ray transport workers fail fast if that actor is missing instead of creating disconnected queues.
- Ray Train support is experimental and currently targets the same local `single_node` deployment as the Ray task backend.
- Ray Serve is not used yet because `prime_vllm` is the supported Ray inference backend and relies on Prime-RL's custom vLLM endpoints.
- Multi-node RayCluster validation requires a shared checkout or Ray `runtime_env` because remote Ray worker pods cannot see a driver pod's local `/tmp` clone.

## Verifier and rollout actor assessment

Verifier and rollout actors are deliberately deferred. The orchestrator owns group scoring, cancellation, `max_inflight_rollouts`, policy-version tracking, and `max_async_level` semantics. Moving those paths to Ray tasks before lifecycle and transport are proven would risk changing training behavior while adding a second scheduler.

The next Ray actor slice should be accepted only if it preserves:

1. Group-scoring behavior for verifier environments that require grouped completions.
2. Cancellation and backpressure when the trainer falls behind.
3. The `max_async_level` bound between rollout policy version and trainer step.
4. Failure visibility equivalent to current verifier server and scheduler errors.
5. The existing HTTP/OpenAI-compatible inference-pool contract unless a separate design replaces it.

Until those checks are satisfied, Ray should own role lifecycle and transport without taking over rollout scheduling.

## Full rewrite gates

A deeper Ray rewrite should wait until these gates are proven:

1. Ray-native role lifecycle can run inference, orchestrator, and trainer ranks without behavior drift.
2. Ray rollout transport is measurably useful versus filesystem/ZMQ.
3. Trainer compatibility preserves rank, process-group, checkpoint, and weight-broadcast semantics.
4. Inference compatibility preserves vLLM engine behavior, custom endpoints, and disaggregated routing.
5. Actor restart semantics are defined for checkpoint resume, rollout buffer recovery, and `max_async_level` policy-version correctness.

Until then, this fork should keep Ray integration behind explicit config flags and keep the upstream process-role behavior as the baseline.
