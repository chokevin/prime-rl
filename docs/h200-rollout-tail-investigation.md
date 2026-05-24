# H200 rollout-tail investigation

This page summarizes the Prime-RL / Prime-vLLM rollout-tail hill climb on
Phi-4 reasoning-plus math with 16 H200 GPUs. It records what was changed in
this fork, what H200 measured, and what should happen next.

## Current state

The current branch is `chokevin/prime-vllm-straggler-analysis`. The latest
Prime-RL ref is `d475c3c38447f5355e4e9f173ad9aa872db7b365`
(`Require admin metrics for guarded waves`).

That ref is a safety/config guard, not a new H200 policy row. The current
blocker is H200-side config: throughput-guarded wave/minimax policies need
backend/admin vLLM metrics. The most recent overhang run polled the router
endpoint and parsed no tracked vLLM metrics, so the policy still relied on
Prime-side proxy signals.

## Baseline and best candidate

| label | Prime ref | status | W&B runtime | decode TPS | completed RPS | tail max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| slack4 baseline | `e86a4c37` | locked healthy baseline | `1315.17s` | `1133.74` | `0.13665` | `519.65s` |
| proxyguard8 | `38d08f38` | best H200 candidate, throughput caveat | `1284.96s` | `957.14` | `0.10569` | `381.86s` |
| overhang16 | `a75e0e11` | rejected | `1307.27s` | `943.31` | `0.10768` | `511.63s` |

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

8. **Real vLLM metrics are still the next blocker.** Overhang16 had
   `collect_inference_metrics = true`, but the run config only had
   `client.base_url = ["http://localhost:8000/v1"]` and no `client.admin_base_url`.
   The metric collector warned that `/metrics` responded but no tracked vLLM
   metrics were parsed. Scheduler candidates therefore did not receive real
   backend decode/cache/running/waiting signals.

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

Owner: H200 harness session / experiment owner.

Before another H200 policy row, update the H200 config so the orchestrator has
backend/admin vLLM metrics:

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

Then run a dry-run/preflight that proves:

- `scheduler/client_metrics_available > 0`
- replay candidate `metrics_available` becomes nonzero after warmup
- replay candidate `decode_throughput_tps` and `completed_requests_per_s` are
  non-null for at least one candidate after warmup
- W&B/logs include `inference/server/<endpoint>/*` metrics

The H200 harness owns this proof. The `h200-rl-lab` preflight branch adds the
concrete gate script:

```bash
CONFIG="${CONFIG:-configs/prime-rl/phi4-reasoning-plus-math-16h200.toml}"
export PRIME_RL_ADMIN_BASE_URLS="http://<backend-0>:8100 http://<backend-1>:8100"

# Endpoint/config preflight.
./scripts/preflight-vllm-metrics.py --config "$CONFIG"

# Warmup output preflight.
./scripts/preflight-vllm-metrics.py --output-dir "$OUTPUT_DIR" --warmup-step 1
```

Only after those gates pass should H200 spend another throughput-guarded
wave/minimax row.

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
