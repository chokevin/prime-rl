# H200 rollout-tail investigation

This page summarizes the Prime-RL / Prime-vLLM rollout-tail hill climb on
Phi-4 reasoning-plus math with 16 H200 GPUs. It records what was changed in
this fork, what H200 measured, and what should happen next.

## Current state

All of the hill-climb refs below (`e86a4c37` slack4, `69cf6571` wave/minimax,
`38d08f38` proxyguard, `a75e0e11` overhang, `d475c3c3` admin-metrics guard) are
now merged into `main` via PRs #3–#6 — they no longer live only on
`chokevin/prime-vllm-straggler-analysis`. The vLLM latency metrics collector
landed separately (PR #5).

The backend/admin vLLM metrics blocker is **resolved**. The `h200-rl-lab`
preflight harness (`scripts/preflight-vllm-metrics.py`) added the
endpoint/config and warmup-output gates, and H200 proved real backend/admin
metrics reached the scheduler. With metrics proven, `d475c3c3` was scored
against the slack4 baseline and **rejected** — it did not recover the tail
(see the table below).

`d475c3c3` itself remains a safety/config guard, not a policy: throughput-guarded
`wave_minimax` with `dp_rank_count > 1` requires `client.admin_base_url` when
`collect_inference_metrics = true` (see [Current guard](#current-guard)).

There is no launchable next policy row on the Prime-RL side. The next step is a
**tested** metric-backed candidate (direction #1 below) or a stop memo — owned
by the experiment owner, not a config-only rerun.

## Baseline and best candidate

| label | Prime ref | status | W&B runtime | decode TPS | completed RPS | tail max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| slack4 baseline | `e86a4c37` | locked healthy baseline | `1315.17s` | `1133.74` | `0.13665` | `519.65s` |
| proxyguard8 | `38d08f38` | best H200 candidate, throughput caveat | `1284.96s` | `957.14` | `0.10569` | `381.86s` |
| overhang16 | `a75e0e11` | rejected | `1307.27s` | `943.31` | `0.10768` | `511.63s` |
| admin-metrics guard | `d475c3c3` | rejected (metrics proven, no tail recovery) | `1290.10s` | `~1101` | `0.0688` | `511.95s` |

Proxyguard8 is the best measured tradeoff so far: it greatly reduces max
rollout group wall time versus slack4 and wave32, but it is still not a
throughput recovery. It should be treated as a candidate with a throughput
caveat, not a final answer.

## What we learned

1. **llm-d/Envoy was not the fix for this RL loop.** The RL bottleneck is
   bulk-synchronous rollout/update timing and group tail behavior. External
   request routing added overhead and lost Prime lifecycle/topology context.

2. **Final-step NCCL slack was the first material win.** Ref `e86a4c37` with
   `final_step_async_level = 4` moved runtime from `1495.11s` to `1315.17s`
   versus the healthy direct control while keeping updates healthy.

3. **Pure scorer/tail-pressure rows hurt throughput.** `3e973781` variants
   and tail-only scoring regressed decode/completed-RPS without improving the
   root tail mechanism.

4. **Hard admission/cap rows reduced some tail but underfilled decode.**
   Clean cap32 (`2c2fc5a7`) improved final generation/group-wall exposure, but
   W&B was flat/worse and decode/completed-RPS collapsed.

5. **Long-output detection needed activation proof.** `145e90aa` never
   activated its long-output score because completion prediction availability
   stayed zero. `4f1f813d` fixed activation with env/cold-start fallback, but
   H200 rejected it because tail stayed unchanged and throughput regressed.

6. **Wave/minimax found a runtime win but hurt throughput.** `69cf6571`
   wave32 improved runtime by `-2.74%` versus slack4, but decode and completed
   RPS regressed by about `21%`. Wave8 lost runtime, tail, and throughput.

7. **Proxyguard8 capped the catastrophic late tail.** It cut max tail to
   `381.86s`, but request-wall p95 and weighted mean worsened. This suggests
   proxyguard8 compresses the extreme tail while pushing more work into the
   300s band.

8. **Real vLLM metrics were the blocker at overhang16 — since resolved.**
   Overhang16 had `collect_inference_metrics = true`, but the run config only had
   `client.base_url = ["http://localhost:8000/v1"]` and no `client.admin_base_url`.
   The metric collector warned that `/metrics` responded but no tracked vLLM
   metrics were parsed, so scheduler candidates did not receive real backend
   decode/cache/running/waiting signals. This was later fixed: the `h200-rl-lab`
   preflight harness wired `admin_base_url` to the backends and proved the
   metrics reached the scheduler. With metrics proven, `d475c3c3` was scored and
   rejected (see [Current state](#current-state) and the table).

## What landed in this branch

| ref | change |
| --- | --- |
| `c1c0c764` | rollout-tail attribution metrics |
| `e86a4c37` | Rune-compatible final NCCL slack |
| `452b3111` | exact scheduler replay event logs for dispatch/completion joins |
| `69cf6571` | config-gated wave/minimax refill assignment |
| `38d08f38` | wave throughput guardrails using Prime-side per-client proxies |
| `a75e0e11` | backend/admin metric host matching, missing-metrics warnings, adaptive wave overhang limiting |
| `d475c3c3` | config guard requiring `client.admin_base_url` for throughput-guarded DP wave configs |

## Current guard

Throughput-guarded `prime_aware.wave_minimax` with `client.dp_rank_count > 1`
now requires `client.admin_base_url` when `collect_inference_metrics = true`.
This prevents H200 from silently running another proxy-only throughput row
against router metrics.

The guard applies when:

- `wave_minimax_size > 1`
- `client.dp_rank_count > 1`
- `collect_inference_metrics = true`
- any throughput guardrail is enabled:
  - `decode_guardrail_penalty > 0`
  - `completed_rps_deficit_weight > 0`
  - `waiting_backpressure_penalty > 0`

## Next action

Owner: experiment owner / H200 harness session.

The backend/admin metrics gate is **satisfied** — `h200-rl-lab` shipped
`scripts/preflight-vllm-metrics.py` and H200 proved that real backend decode /
completed-RPS / queue signals reach the scheduler. The reference gate that was
required before any throughput-guarded row (and is now passing):

```toml
[client]
base_url = ["http://<router-or-rollout-endpoint>:8000/v1"]
admin_base_url = [
  "http://<backend-0>:8100",
  "http://<backend-1>:8100",
  # include every backend/admin endpoint that exposes vLLM /metrics
]
dp_rank_count = 4
```

```bash
CONFIG="${CONFIG:-configs/prime-rl/phi4-reasoning-plus-math-16h200.toml}"
export PRIME_RL_ADMIN_BASE_URLS="http://<backend-0>:8100 http://<backend-1>:8100"

# Endpoint/config preflight.
./scripts/preflight-vllm-metrics.py --config "$CONFIG"

# Warmup output preflight.
./scripts/preflight-vllm-metrics.py --output-dir "$OUTPUT_DIR" --warmup-step 1
```

The gates that now pass: `scheduler/client_metrics_available > 0`, replay
candidate `metrics_available` nonzero after warmup, candidate
`decode_throughput_tps` / `completed_requests_per_s` non-null after warmup, and
`inference/server/<endpoint>/*` metrics in W&B/logs.

Because the gate is met and `d475c3c3` was already scored and rejected, the next
step is **not** another config-only rerun. It is one of: a tested metric-backed
candidate (direction #1 below), or a stop memo if the tail is judged
unrecoverable with the current dispatch model.

## Candidate direction after metrics are fixed

Do not continue scalar proxyguard sweeps. Proxyguard4 and proxyguard16 failed
non-monotonically.

The next policy direction should be either:

1. **Metric-backed throughput-aware late-wave control**: keep proxyguard8's max
   tail cap while using real backend decode/completed-RPS/queue metrics to avoid
   throughput collapse.
2. **Replay-guided normal-vs-late-overhang lanes**: normal refill early, then
   overhang control only near terminal drain. Do not split solely by predicted
   completion length; replay showed predicted length is not reliable enough.

## Do not rerun

- llm-d/Envoy generation data path rows.
- `3e973781` scorer-only/tail-pressure variants.
- cap32 `2c2fc5a7`.
- long-output rows `145e90aa` or `4f1f813d`.
- wave8 or blind wave-size sweeps.
- proxyguard scalar sweeps without new replay evidence.

## Validation state

Latest branch validation for `d475c3c3`:

- `ruff check` / `ruff format`
- `python -m py_compile` on edited files
- focused scheduler/request-picker/inference-metrics/config tests (`38 passed`)

Local macOS validation used a temporary `uv run --no-project --with ...`
dependency set because the project lock contains Linux/H200-specific packages.
