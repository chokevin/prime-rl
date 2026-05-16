# Ray-native fork architecture

Prime-RL's stable runtime remains process-role based: `rl` writes resolved TOML subconfigs, launches `inference`, `orchestrator`, and a `torchrun` trainer, and coordinates rollout transport through filesystem or ZMQ backends. The RayJob launcher path uses Ray as a cluster launch substrate for those same roles.

This fork adds a Ray-native runtime path. When `experimental.ray.enabled = true`, Ray launches Prime-RL roles as in-process Ray tasks instead of shelling out to the CLI role commands.

```bash
uv pip install "ray[default]>=2.40.0"
uv run rl @ examples/reverse_text/rl.toml \
  --experimental.ray.enabled \
  --trainer.rollout-transport.type ray \
  --orchestrator.rollout-transport.type ray
```

## What Ray owns

- **Role lifecycle**: Ray tasks run inference, orchestrator, and trainer ranks directly.
- **Placement accounting**: a Ray placement group reserves the configured inference GPUs and one GPU per trainer rank.
- **Distributed trainer rank env**: Ray trainer tasks set `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`, then call `train(config)`.
- **Failure surfacing**: failed Ray role tasks fail the run and point at the role log file.
- **Rollout transport**: `rollout_transport.type = "ray"` moves `TrainingBatch` and packed micro-batches through a named Ray actor instead of filesystem or ZMQ.

Inference calls Prime-RL's Python `inference_local(config)` function inside a Ray GPU task. The orchestrator calls `asyncio.run(orchestrate(config))`. Trainer ranks call `train(config)` directly after Ray assigns GPUs and rank environment.

## Example config

```toml
[experimental.ray]
enabled = true
namespace = "prime-rl"
placement_strategy = "STRICT_PACK"

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

## Current constraints

- Only `deployment.type = "single_node"` is supported.
- SLURM mode is not supported by `experimental.ray.enabled`.
- Ray is an optional runtime dependency for this fork; non-Ray Prime-RL paths do not import Ray.
- `trainer.rollout_transport.type` and `orchestrator.rollout_transport.type` must both be `ray`.
- The Ray-native `rl` launcher owns the shared transport actor; Ray transport workers fail fast if that actor is missing instead of creating disconnected queues.
- Ray Train is not used yet; the fork maps Ray trainer tasks onto Prime-RL's existing `torch.distributed` trainer by setting rank environment explicitly.
- Ray Serve is not used yet because inference is vLLM-native and relies on Prime-RL's custom vLLM endpoints.

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
