# Tau: prime-rl math-7b-h200 (W4b)

Runs the primary experiment from the `/goal` harness's math hill-climb — LoRA RL
fine-tuning of `Qwen/Qwen2.5-7B-Instruct` on `math-env-v1`, measured against a frozen,
paired `math500-v1` held-out eval — through the `tau` CLI on `aks-ai-runtime-eastus2`.

**GPU execution has not been done in this session.** Everything below through "Static
validation done in this session" is proven; everything under "What is *not* proven yet"
requires a live cluster submit that this W4b implementation session did not perform (its
scope was static config/scripts/tests only — see the parent `/goal` plan's W4b/W5 split).

## Layout

```
tau/
  image.pin.json        # single source of truth for the pinned image digest
  render_image.py        # substitutes the digest into tau/*.yaml -> tau/.rendered/ (gitignored)
  smoke.yaml            # CPU-only config validation (`tau run smoke`)
  freeze-manifest.yaml  # CPU-only: build/finalize the frozen eval manifest (`tau run freeze-manifest`)
  eval-baseline.yaml    # 1 H200: frozen-eval replay against the base model (`tau run eval-baseline`)
  eval-post.yaml        # 1 H200: frozen-eval replay against base+LoRA, then compare (`tau run eval-post`)
  train.yaml            # 2 H200 (1 trainer + 1 inference): bounded RL training (`tau run train`)
  scripts/
    run-prime-rl.sh      # the one self-contained entrypoint all five targets share (mode via $PRIME_RL_RUN_MODE)
  eval_tools/            # pure, macOS-testable: hashing, frozen-manifest schema, paired comparison + gate
    hashing.py
    manifest.py
    compare.py
    cli.py                # `python -m tau.eval_tools.cli {compare,check-disjoint}`
    tests/                 # 32 pytest cases, no GPU/network required
    live/                   # GPU-container-only: real dataset/model/inference-server scripts
      freeze_manifest_live.py
      run_frozen_eval_live.py
configs/tau/math-7b-h200/
  train.toml             # derived from configs/basic/hendrycks-sanity/rl.toml
```

Every `tau/*.yaml` is a **template**: its `runtime.image` is the literal sentinel
`RENDER_REQUIRED__see_tau/render_image.py`, not a real image reference. Run
`python3 tau/render_image.py` first, then point every command below at
`tau/.rendered/<target>.yaml` — see "Image and overlay strategy" below for why.

There is no `tau/workspace.connection.yaml` — the installed `tau v0.1.2-26-g57532229`
rejects the scaffold's `requirements.minTauVersion: 0.3.0`, so every command below passes
`--context aks-ai-runtime-eastus2-admin` and each YAML sets `policy.workspace:
pretraining-data` explicitly instead (see the W1 Tau/cluster memo).

## Cluster contract this assumes (from W1)

`TauWorkspace/pretraining-data` → namespace `pretraining-data`, `LocalQueue/jobqueue`,
`PVC/blob-training` mounted at `/data`, output root `/data/pretraining-data`, H200
node selectors `agentpool=h200pool` + `kueue.azure.com/gpu-series=nd-h200-v5`,
`policy.topology: single-node-nvlink` (required — the H200 `ResourceFlavor` is
TAS-only), 2 GPUs for training / 1 for eval.

## Image and overlay strategy

**Pinned image (`tau/image.pin.json`):** `ghcr.io/primeintellect-ai/prime-rl@sha256:d701d000174052a70ad060197e71915e83e0f2b1fedef956c60b3038cf2b992f`.

Resolved by hand against the public GHCR registry (anonymous pull token +
`v2/.../manifests/commit-bbb90a1b4`), not guessed:

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:primeintellect-ai/prime-rl:pull" | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
curl -sD - -o /dev/null --oauth2-bearer "$TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json" \
  "https://ghcr.io/v2/primeintellect-ai/prime-rl/manifests/commit-bbb90a1b4" \
  | grep -i docker-content-digest
```

The manifest's `org.opencontainers.image.revision` annotation is
`bbb90a1b4132c351cbe8b0ed1fa808dde99f0318` — exactly this branch's merge-base commit (the
sync branch's tree is byte-identical to upstream `bbb90a1b4`; see the `/goal` plan's
Baseline section). So the baked `/app/.venv` already satisfies this branch's locked
dependencies for any commit that only adds TOML/scripts/docs, which is everything on
this branch so far.

**Why a pin file + render step instead of hardcoding the digest 5×:** every checked-in
`tau/*.yaml` carries the sentinel `RENDER_REQUIRED__see_tau/render_image.py` instead of
a real (or worse, fake/placeholder-that-looks-real) digest. `tau/render_image.py`
substitutes the one digest recorded in `tau/image.pin.json` into every template,
writing to gitignored `tau/.rendered/` (and mirrors `tau/scripts/` alongside each
rendered file, since `entrypoint:` paths resolve relative to the config file's own
directory), and refuses to render (nonzero exit, no output) if the pin file is missing,
malformed, or itself contains a placeholder/`latest` value. Re-pinning to a new digest
is a one-line JSON edit, not five find-and-replace edits. Tested empirically: Tau's
client-side `validate`/`--dry-run=client` does **not** reject the un-rendered sentinel
string (it passes straight through into the rendered Job spec) — the real safety net is
one step later, since the sentinel isn't a resolvable image reference and an accidental
un-rendered submit fails loudly at image-pull time (`ErrImagePull`) rather than silently
running. Always render first regardless.

**Why no `uv sync` at job startup:** `tau/scripts/run-prime-rl.sh` fetches this exact
fork/commit into a scratch checkout (`git init` + `git fetch --depth 1 <sha>` +
`git checkout FETCH_HEAD`, then asserts `git rev-parse HEAD` equals the pinned SHA — a
shallow single-commit fetch that lands on anything else is itself proof of tampering or
a moved ref) and runs everything with `uv run --no-sync`, which reuses the baked venv
byte-for-byte. This is deliberately narrower than the base image's own built-in
`docker-entrypoint.sh` override path (`PRIME_RL_REF`/`PRIME_RL_REPO` env vars), which
re-seeds a venv and runs `uv sync --inexact --all-packages ...` — correct for a commit
that *does* change dependencies, but heavier than this branch needs. If a future commit
on this branch adds a Python dependency, switch to that built-in mechanism (or add an
explicit `uv sync --inexact` step to this wrapper) instead of silently going stale.

**Narrow package-install escape hatch:** if a future commit (e.g. the harder-math
environment from W4a/W9) needs one extra installed package, set
`runtime.env.PRIME_RL_EXTRA_ENV_PACKAGE_DIR` to that package's path relative to the repo
root; the wrapper runs `uv pip install --no-deps -e <that one dir>` — never a full/
`--inexact` workspace sync. Not needed by the primary experiment: `math-env-v1` and
`math500-v1` already ship in `deps/research-environments`, baked into the image.

**If no matching image digest existed:** the fallback is a manual
`workflow_dispatch` of `.github/workflows/build_image.yaml` — but that workflow only
allows `ref` values matching `main`, `v*`, or `*.dev*`, and it's scoped to
`primeintellect-ai/prime-rl`'s own GHCR namespace; a fork building its own image would
need to point `REGISTRY`/`IMAGE_NAME` at a repo it can push to. Not needed here since the
upstream digest above matches exactly — but if it hadn't, `tau/image.pin.json` would
carry a placeholder and `render_image.py` would refuse to render (mechanically gating
the GPU targets shut) rather than ever committing a fake digest.

## How Tau runs `run-prime-rl.sh` (why it must be self-contained)

Confirmed empirically with `tau run --dry-run=client` (see below): Tau reads the
`entrypoint:` file's *content* at render time, base64-encodes it into a
`TAU_SCRIPT_B64` env var baked into the rendered `batch/v1 Job`, and the container's
actual command decodes it to `/tmp/run.sh` and `exec`s it — `workingDir` is `/data`, and
the surrounding repo tree is **not** shipped alongside it. That's why the script
`source`s nothing and does its own git fetch (the "immutable overlay" above) rather than
assuming any other file is present next to it.

`entrypoint:` paths resolve relative to the *config file's own directory* — since every
`tau/*.yaml` lives directly in `tau/`, `entrypoint: scripts/run-prime-rl.sh` resolves to
`tau/scripts/run-prime-rl.sh`. (A first attempt at `entrypoint: tau/scripts/run.sh` from
a `tau/probe.yaml` file resolved to the wrong `tau/tau/scripts/run.sh` and failed — noted
here since it's an easy mistake to repeat.)

## Config choices baked into `configs/tau/math-7b-h200/train.toml`

Derived from `configs/basic/hendrycks-sanity/rl.toml` per the W2 selection memo:

| Knob | Value | Why |
|---|---|---|
| `[deployment]` | 1 train + 1 infer GPU | was 4+4 for the 1.5B sanity config |
| `[model].name` | `Qwen/Qwen2.5-7B-Instruct` | primary experiment tuple (W2) |
| `[orchestrator.renderer].name` | `default` | not in `MODEL_RENDERER_MAP` (verified against `deps/renderers/renderers/base.py`); `renderer.name="auto"` would hard-fail its own validator for this model. No `reasoning_parser` — plain instruct model, not a `<think>`-tag reasoning model. |
| `[trainer.model.lora].rank` | 16 | W2's LoRA choice |
| `[trainer.ckpt.weights].save_adapter_separately` | `true` | so `tau/eval-post.yaml` can load just the adapter |
| `[orchestrator.train.source.env.taskset.task].judge` | `"None"` | disables `math-env-v1`'s LLM reference-judge fallback so training reward is purely deterministic (`"None"` → Python `None`, per `deps/pydantic-config/src/pydantic_config/cli.py`'s TOML-null convention) |
| `[orchestrator.eval]` | `math500-v1`, all 500 examples, `group_size=1`, `interval=25` with `max_steps=50` | in-run startup(step 0)/periodic(25)/final(50) eval — a monitoring signal only, **not** the frozen comparison of record |

`max_steps = 50` is a placeholder pending a real throughput measurement (see "What is
not proven yet"). Override per-submit via `runtime.env.PRIME_RL_MAX_STEPS` (the wrapper
passes it as `--max-steps`).

## Frozen eval manifest and the paired comparison gate

- `tau/eval_tools/manifest.py` — `FrozenEvalManifest`: pins model snapshot, taskset refs,
  every example's prompt/answer **hash** (not raw text — replay reloads the pinned
  taskset and re-hashes to detect drift), decoding config, grader, and N (≥ 200
  enforced). `identity_hash()` covers everything a baseline/post pair must share
  (excludes `model` and `created_at` on purpose, since the model is expected to differ).
  `check_disjoint()` proves zero eval/train prompt-hash overlap; `check_headroom()`
  enforces the `[0.10, 0.80]` baseline band.
- `tau/eval_tools/compare.py` — `compare_runs()`: validates both reward files share the
  manifest's `identity_hash()` and exact example-id set (else raises
  `IdentityMismatchError`), computes the paired mean delta, a **deterministic**
  (seeded `numpy.random.RandomState`) percentile bootstrap 95% CI, and the gate: pass
  only if `delta >= +0.03` **and** `ci_lower > 0`.
- `tau/eval_tools/live/` does the same job for real: `freeze_manifest_live.py` (two-phase
  `draft`/`finalize`, needs `verifiers`/`datasets`/`huggingface_hub` + network) and
  `run_frozen_eval_live.py` (replays against a running vLLM server, needs `openai` +
  a live server). Not importable in this repo's macOS sandbox — see "Static validation".

### Proof ladder (in order)

Before any of this: substitute the exact commit SHA into every target (see "Operator
commands" below) and render the image pin — `python3 tau/render_image.py`. Every
command below points at `tau/.rendered/<target>.yaml`, never the bare `tau/<target>.yaml`
template (that still carries the unrendered image sentinel).

```bash
# 0. Render once per commit (after substituting PRIME_RL_REPO_SHA — see below).
python3 tau/render_image.py

# 1. Build the draft manifest (CPU-only) — proves train/eval disjointness up front.
#    (PRIME_RL_RUN_MODE=freeze-draft is the default in the committed YAML.)
tau run --config tau/.rendered/freeze-manifest.yaml --context aks-ai-runtime-eastus2-admin

# 2. Measure the baseline against the draft manifest (1 GPU).
tau run --config tau/.rendered/eval-baseline.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-eval-baseline -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact rewards.json

# 3. Freeze the manifest, recording the measured baseline mean (edit
#    tau/freeze-manifest.yaml: PRIME_RL_RUN_MODE=freeze-finalize, PRIME_RL_BASELINE_MEAN=<step 2's mean>,
#    then re-render).
python3 tau/render_image.py
tau run --config tau/.rendered/freeze-manifest.yaml --context aks-ai-runtime-eastus2-admin

# 4. Train (2 GPU) — the wrapper refuses to start without frozen-eval-manifest.json.
tau run --config tau/.rendered/train.yaml --context aks-ai-runtime-eastus2-admin

# 5. Post-training eval + comparison (1 GPU) — edit tau/eval-post.yaml's
#    PRIME_RL_LORA_ADAPTER_PATH to the trained checkpoint step, re-render, then:
tau run --config tau/.rendered/eval-post.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-eval-post -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact comparison.json
```

`eval-baseline`'s rewards stay valid after freezing without a rerun:
`identity_hash()` is unaffected by `freeze-finalize` (it only changes
`baseline_mean_headroom_ok` and `created_at`, both excluded from the hash).

## Operator commands

**Before every submit:** replace `REPLACE_WITH_EXACT_COMMIT_SHA` in the target's
`runtime.env.PRIME_RL_REPO_SHA` with the exact commit being run, then render the image
pin:

```bash
sed -i '' "s/REPLACE_WITH_EXACT_COMMIT_SHA/$(git rev-parse HEAD)/" tau/<target>.yaml
python3 tau/render_image.py
```

Validate + dry-run client (no cluster mutation) for any target — schema `validate` works
on either the template or the rendered file, but `--dry-run=client` needs the real
digest, so always use the rendered path:

```bash
tau run validate --config tau/.rendered/<target>.yaml
tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin --dry-run=client
```

Submit / monitor / fetch / cancel:

```bash
tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin
tau run status <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin --watch
tau run logs <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin -f
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin --artifact rewards.json
tau run cancel <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin
```

`<job-name>` equals each YAML's `name:` field (`prime-rl-math-7b-h200-{smoke,freeze-manifest,eval-baseline,eval-post,train}`).
Never submit the bare `tau/<target>.yaml` template directly — it still carries the
unrendered image sentinel.

## Secrets

- No tokens, secret names, kubeconfigs, or registry credentials are committed anywhere
  in `tau/` or `configs/tau/`. `Qwen/Qwen2.5-7B-Instruct` is a public model — no
  `runtime.env_secret` (e.g. `HF_TOKEN`) is configured by default.
- `WANDB_MODE: offline` everywhere — no W&B API key needed. To enable online logging,
  an operator would add `runtime.env_secret.WANDB_API_KEY: <existing-secret-name>:<key>`
  themselves (client dry-run redacts the referenced secret name/key while still showing
  the dependency exists — see `tau run explain-config`'s note on `runtime.env_secret`);
  this repo does not reference any such secret name because none has been approved.
- If a private base model or a registry pull secret is ever needed, the same
  `runtime.env_secret` (for HF) or a platform-managed image-pull-secret path (for the
  registry) should be added explicitly and reviewed — do not add a bare token.
- Secrets/diff check for this change: `git diff <merge-base>... | grep -iE
  "token|secret|password|api[_-]?key|BEGIN [A-Z]+ PRIVATE KEY"` — run as part of static
  validation below; only known placeholders (`REPLACE_WITH_EXACT_COMMIT_SHA`,
  `REPLACE_WITH_STEP`) and the harmless string `api_key_var` (a prime-rl config *field
  name*, not a value) should match.

## Static validation done in this session

All of the following were run in this session and passed (see the W4b report for full
output):

- `tau run validate --config tau/.rendered/<target>.yaml` for all 5 targets.
- `tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin --dry-run=client`
  for all 5 targets — rendered `batch/v1 Job`s with correct GPU requests/limits, node
  selectors, topology annotation, storage mounts, and env vars; the embedded
  `TAU_SCRIPT_B64` was verified byte-identical to `tau/scripts/run-prime-rl.sh` by
  decoding it back and diffing.
- `python3 tau/render_image.py` (and its `--check-only`/malformed-pin/missing-pin error
  paths) exercised directly; confirmed the un-rendered `tau/<target>.yaml` templates
  still pass `tau run validate` (schema-only) but the sentinel image string is not
  rejected by `--dry-run=client` either (documented in "Image and overlay strategy" so
  this isn't mistaken for a stronger guarantee than it is).
- `bash -n` and `shellcheck` (zero warnings) on `tau/scripts/run-prime-rl.sh`.
- `python -m tomllib` parse of `configs/tau/math-7b-h200/train.toml`, cross-referenced
  field-by-field against `packages/prime-rl-configs/src/prime_rl/configs/{rl,orchestrator,trainer}.py`.
- 32 `pytest` cases for `tau/eval_tools/{hashing,manifest,compare}.py` (identity
  mismatch, leakage, deterministic bootstrap, and both comparison-gate pass/fail
  branches — including a case where the mean delta meets the `+0.03` bar but the CI
  still crosses zero) — run in an isolated macOS `uv venv` with only `pydantic`/`numpy`/
  `pytest` installed (no project-wide `uv sync`, since torch/vllm are Linux/CUDA-only).
- `ruff check` / `ruff format --check` clean on every new Python file under `tau/`.
- `python -m py_compile` on the `live/` scripts (syntax only — they import
  `verifiers`/`datasets`/`openai`/`huggingface_hub`, unavailable here).

## What is *not* proven yet (explicitly out of this session's scope)

- No Tau job has been submitted (no `--dry-run=server`, no real submit). This session's
  scope (W4b) was static config/scripts/tests; live cluster execution is W5–W8.
- `tau/eval_tools/live/*.py` have never been executed against a real dataset, model, or
  inference server.
- `max_steps = 50` and the resource requests are placeholders, not throughput-measured.
- The frozen eval manifest has not been created; no baseline has been measured; no
  training has run; no comparison has been computed.
- Whether `Qwen/Qwen2.5-7B-Instruct` + `DefaultRenderer` + rank-16 LoRA actually loads
  and trains cleanly on this image is unverified — the W2 memo's compatibility read is
  source-backed, not execution-tested.
