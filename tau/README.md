# Tau: prime-rl math-7b-h200

Runs LoRA RL fine-tuning of `Qwen/Qwen2.5-7B-Instruct` on `math-env-v1`, measured
against a frozen, paired `math500-v1` held-out eval, through the `tau` CLI on
`aks-ai-runtime-eastus2`. The F10 proof ladder completed end to end, including a
50-step two-H200 run, but the frozen comparison was a retained negative result:
`0.7480 -> 0.7500`, delta `+0.0020`, paired-bootstrap 95% CI
`[-0.0160, +0.0200]`.

The additive F11 target measured the same frozen base model over the full 5,030-row
`harder-math-v1` eval catalog. It passed the fixed hardness gate with base `0.9151`,
core `0.7812`, hard `0.5288`, and base-minus-hard `+0.3863`. It is independent of
F10 training evidence and does not rerun or reinterpret the failed F10 tuple.

## Layout

```
tau/
  image.pin.json        # single source of truth for the pinned image digest
  render_image.py        # substitutes the digest into tau/*.yaml -> tau/.rendered/ (gitignored)
  smoke.yaml            # system-CPU config validation, zero GPU (`tau run smoke`)
  freeze-manifest.yaml  # system CPU, zero GPU: build/finalize the frozen eval manifest
  eval-baseline.yaml    # 1 H200: frozen-eval replay against the base model (`tau run eval-baseline`)
  eval-post.yaml        # 1 H200: frozen-eval replay against base+LoRA, then compare (`tau run eval-post`)
  train.yaml            # 2 H200 (1 trainer + 1 inference): bounded RL training (`tau run train`)
  harder-tier-curve.yaml # 1 H200: full harder-math base/core/hard empirical curve
  f12-smoke.yaml          # system-CPU duration-only F12 config validation
  f12-freeze-manifest.yaml # system-CPU F12 manifest draft/finalize
  f12-eval-baseline.yaml  # 1 H200: source-matched F12 baseline
  f12-train.yaml          # 2 H200: 200-step duration-only training
  f12-eval-post.yaml      # 1 H200: F12 frozen post eval + unchanged gate
  f12-eval-post-recovery.yaml # 1 H200: fixed-input recovery of interrupted F12 post eval
  f12-eval-post-recovery-preflight.yaml # system-CPU, zero GPU: exact-path adapter handoff gate the H200 recovery requires
  scripts/
    run-prime-rl.sh      # the one self-contained entrypoint all targets share (mode via $PRIME_RL_RUN_MODE)
  eval_tools/            # pure, macOS-testable: hashing, frozen-manifest schema, paired comparison + gate
    hashing.py
    json_io.py             # duplicate-key rejection + exclusive JSON evidence writes
    manifest.py
    artifacts.py           # attempt-scoped evidence, atomic publication, adapter handoff
    compare.py
    cli.py
    tests/                 # focused pytest cases, no GPU/network required
    live/                   # GPU-container-only: real dataset/model/inference-server scripts
      freeze_manifest_live.py
      run_frozen_eval_live.py
      run_harder_tier_curve_live.py
      validate_training_data_live.py # canonical resolved-config parser/identity
      training_supervisor_live.py    # owns private inputs, RL child, attestation, publication
configs/tau/math-7b-h200/
  train.toml             # derived from configs/basic/hendrycks-sanity/rl.toml
  train-f12.toml         # immutable duration-only F12 contract
```

Every `tau/*.yaml` pins the real public digest recorded in `tau/image.pin.json`. Run
`uv run --no-sync python tau/render_image.py` first: it rejects any target that drifts
from the pin and stages validated copies plus their entrypoint under `tau/.rendered/`.
Point every command below at `tau/.rendered/<target>.yaml`.

There is no `tau/workspace.connection.yaml` — the installed `tau v0.1.2-26-g57532229`
rejects the scaffold's `requirements.minTauVersion: 0.3.0`, so every command below passes
`--context aks-ai-runtime-eastus2-admin` and each YAML sets `policy.workspace:
pretraining-data` explicitly instead (see the W1 Tau/cluster memo).

## Cluster contract this assumes (from W1)

`TauWorkspace/pretraining-data` → namespace `pretraining-data`, `LocalQueue/jobqueue`,
`PVC/blob-training` mounted at `/data`, and output root `/data/pretraining-data`.
The zero-GPU smoke and manifest targets use the W1-proven `policy.topology:
independent` with `kubernetes.azure.com/mode=system`, admitting on system CPU
nodes without H200 selectors or tolerations. GPU targets use H200 node selectors
`agentpool=h200pool` + `kueue.azure.com/gpu-series=nd-h200-v5` and
`policy.topology: single-node-nvlink` (required — the H200 `ResourceFlavor` is
TAS-only), with 2 GPUs for training and 1 GPU for either frozen eval or the
harder-math tier curve.

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
Baseline section). The five F10 targets use immutable runtime overlay
`ce9d0919b6eb266a7c8100b333dc9ea91e95b96f`. The additive tier-curve target uses
F11 runtime `bf0e4c478648ec97eb0eea917cba71348af86d1e`, which adds only the
full-catalog runner, exact F10 model-contract binding, and its isolated output mode.
Both commits contain the workspace-locked `environments/harder_math_v1` package.
Its dependency set matches the pinned image: `harder-math-v1` uses only `datasets`
and `verifiers`, which are already present.

The Qwen2.5 trainer is pinned to prime-rl's Hugging Face implementation with
`flash_attention_2`. Qwen2.5 has no custom PrimeRL trainer implementation, while Hopper's
automatic `flash_attention_3` selection is custom-only; leaving both fields on `auto`
therefore fails before model loading.

Every F10 artifact belongs to the complete source generation rooted at
`/data/pretraining-data/prime-rl-math-7b-h200/generations/ce9d0919b6eb266a7c8100b333dc9ea91e95b96f`.
The wrapper derives this root from the exact fetched commit and rejects output or
cross-mode inputs from any other generation before creating a directory. Any runtime
behavior or config change must therefore land in a new source commit and start a complete
new smoke/manifest/baseline/train/post generation. The later pin commit may only wire
templates and docs to that source SHA. Existing generation directories are immutable and
preserved; they are never migrated, deleted, or reused by a newer source.

F11 tier evidence is isolated under
`/data/pretraining-data/prime-rl-math-7b-h200/generations/bf0e4c478648ec97eb0eea917cba71348af86d1e/tier-curve`.
It deliberately consumes the finalized F10 manifest only as the exact base-model and
decoding contract. Before creating F11 output, the wrapper requires the canonical F10
manifest path, source revision, and verified file SHA-256
`115369d8b3248548b1680de0b8c695e161d1336b64a4b0de10493908c3021abe`.
The runner repeats the digest and semantic-contract checks and records the digest in
each raw tier artifact.

**Why a pin file + render step when the digest is also checked into every target:** the templates
must be directly inspectable and runnable-looking, while `tau/image.pin.json` remains
the source of truth used to detect drift. `tau/render_image.py` validates that every
template carries exactly that real digest, writes validated copies to gitignored
`tau/.rendered/`, and mirrors `tau/scripts/` alongside them because `entrypoint:` paths
resolve relative to the config file's own directory. It refuses to stage anything if
the pin is missing, malformed, set to `latest`, or disagrees with any target. Re-pinning
therefore requires an intentional edit to the pin and all visible targets.

**Why no `uv sync` at job startup:** `tau/scripts/run-prime-rl.sh` creates a private
`/tmp/prime-rl-overlay.XXXXXX` scratch directory, fetches this exact fork/commit into it
(`git init` + `git fetch --depth 1 <sha>` +
`git checkout FETCH_HEAD`, then asserts `git rev-parse HEAD` equals the pinned SHA — a
shallow single-commit fetch that lands on anything else is itself proof of tampering or
a moved ref). It also verifies the commit's `deps/verifiers` and
`deps/research-environments` gitlinks against the exact SHAs in every target. The
scratch path is not configurable and cleanup refuses any path outside its validated
`/tmp/prime-rl-overlay.*` namespace, so mounted data under `/data` or `/app` can never be
recursively removed. Jobs run with `uv run --no-sync`, reusing the baked venv
byte-for-byte. This is deliberately narrower than the base image's own built-in
`docker-entrypoint.sh` override path (`PRIME_RL_REF`/`PRIME_RL_REPO` env vars), which
re-seeds a venv and runs `uv sync --inexact --all-packages ...` — correct for a commit
that *does* change dependencies, but heavier than this branch needs. If a future commit
adds a Python dependency, switch to that built-in mechanism (or add an
explicit `uv sync --inexact` step to this wrapper) instead of silently going stale.

**Narrow package-install escape hatch:** the harder-math tier-curve target sets
`runtime.env.PRIME_RL_EXTRA_ENV_PACKAGE_DIR=environments/harder_math_v1`. No other path
is accepted. The wrapper rejects absolute paths, `..`, symlink components, and canonical
escapes before running `uv pip install --no-deps -e` on that verified checkout path. It
also replaces rather than extends inherited `PYTHONPATH`. The primary experiment does
not need this escape hatch: `math-env-v1` and `math500-v1` already ship in
`deps/research-environments`, baked into the image.

The corresponding prime-rl source-selection overlay is
`configs/tau/math-7b-h200/harder-math-v1-hard.toml`. It is a static fixture for config
selection and a future harder-math training experiment; none of the five primary targets
or the evaluation-only tier target use it. Compose
it after `train.toml` to replace only `orchestrator.train.source`. The primary
`math-env-v1` training and all-500 `math500-v1` frozen-evaluation proof ladder remain the
default.

**Harder-math metadata boundary:** the primary frozen manifest remains a strict,
ordered 500-row MATH-500 contract. The `harder-math-v1` tier curve is a separate
`tier-curve.v1` artifact covering its full 5,030-row eval catalog; it must never be
inserted into `FrozenEvalManifest.examples`. For later disjointness tooling,
`tier-curve.v1.records[].prompt_sha256` is the field corresponding to Tau's
`prompt_hash`. `content_sha256` hashes prompt plus gold and is not Tau's
answer-only `answer_hash`; `record_id` is source-qualified text rather than the
MATH-500 integer ID. Keeping those artifacts separate lets Tau consume the W4a prompt
hash set later without weakening or renaming the primary manifest fields and without
mixing its empirical result into the F10 comparison.

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
| `[inference].seed` | `0` | fixed resolved inference seed; the effective-config preflight rejects drift |
| `[trainer.ckpt.weights].save_adapter_separately` | `true` | so `tau/eval-post.yaml` can load just the adapter |
| `[orchestrator.ckpt]` | enabled, no resume/skip fields | the real current `RLConfig` requires trainer and orchestrator checkpoint state to be enabled together |
| `[file_monitor]` | enabled | writes the documented fixed `metrics.jsonl` artifact |
| `[orchestrator.train.source.env.taskset]` | `math-env-v1`, `PrimeIntellect/Hendrycks-Math`, `default`, `train` | the manifest binds commit `3ed63f49541bdca4382fba28146aadf20d95cb38` and every dataset file; training rewrites only `dataset_name` to its verified job-private immutable copy |
| `[orchestrator.train.source.env.taskset.task].judge` | `"None"` | disables `math-env-v1`'s LLM reference-judge fallback so training reward is purely deterministic (`"None"` → Python `None`, per `deps/pydantic-config/src/pydantic_config/cli.py`'s TOML-null convention) |
| `[orchestrator.eval]` | `math500-v1`, all 500 examples, `group_size=1`, `interval=25` with `max_steps=50` | in-run startup(step 0)/periodic(25)/final(50) eval — a monitoring signal only, **not** the frozen comparison of record |

`max_steps = 50` is the frozen F10 bounded-run budget. It completed successfully but
was not sufficient to meet the statistical improvement gate. Finalization binds it in
the canonical config identity. Training and
post-eval derive the one source-proven final checkpoint path
`weights/step_<max_steps>/lora_adapters` from that frozen identity; there is no runtime
step override. The handoff never enumerates a storage directory.

### F12 duration-only experiment

F12 is an additive five-target ladder pinned to runtime source
`a603e791776a4579440edc5df5b70309c61cf46a` and generation root
`/data/pretraining-data/prime-rl-math-7b-h200/generations/a603e791776a4579440edc5df5b70309c61cf46a/`.
It does not reuse, migrate, or rewrite F10/F11 evidence.

`configs/tau/math-7b-h200/train-f12.toml` differs from the resolved F10 RLConfig only
through `max_steps = 200` and monitoring `orchestrator.eval.interval = 100`. TrainSink
still samples until each update has exactly 128 survivor rollouts after zero-advantage
groups are dropped; F12 does not shrink the batch or alter sampler/filter semantics.
Model and revision, renderer, HF/FA2 model path, rank/alpha-16 LoRA, optimizer, constant
`1e-6` learning rate, scheduler, batch/group sizes, decoding, grader, seeds, 7,474-row
training identity, all-500 held-out eval identity, zero-overlap proof, and trainer dtypes
remain fixed.

In-run MATH-500 evals at startup, step 100, and step 200 are monitoring only and cannot
select a checkpoint. Step 200 is the sole comparison-of-record adapter. The external
paired gate remains delta `>= +0.03` and paired-bootstrap 95% CI lower bound `> 0`.
The complete ladder budget remains below 4 H200-hours. These are frozen pre-run
contracts, not measured F12 results.

Run the original F12 ladder in the same order as F10 using
`f12-smoke.yaml`, `f12-freeze-manifest.yaml`, `f12-eval-baseline.yaml`,
`f12-train.yaml`, and `f12-eval-post.yaml`: smoke, freeze draft, baseline,
freeze finalize, train, and post. Fetch artifacts by exact filenames as described
below.

The original F12 post Job was interrupted before any of its 500 requests completed.
Its source-generation `eval-post/inference.log` remains preserved partial evidence;
there is no F12 post `rewards.json` or `comparison.json`, and that immutable tuple must
not be rerun or cleaned up. The only approved continuation is the additive
`f12-eval-post-recovery.yaml` target. It runs source
`2594a4c0bdd1ba8ccc754e12205f4b0201bfef86`, writes only beneath
`/data/pretraining-data/prime-rl-math-7b-h200/generations/2594a4c0bdd1ba8ccc754e12205f4b0201bfef86/eval-post-recovery/`,
and treats the complete F12 source generation as read-only.

The recovery validates the exact F12 frozen manifest, 500-row baseline, successful
step-200 training result, attempt `20260804T013900Z-0de01384c398f8d3`, model/data/config
identities, and complete rank-16 adapter tree before inference. The approved SHA-256
values are embedded in both runtime code and target environment, so path or digest drift
fails before output creation. Its `comparison.json` records both the recovery runtime
source and frozen F12 experiment source while applying the unchanged deterministic
10,000-sample paired-bootstrap gate. It does not expose an alternate checkpoint,
decoding, grader, seed, baseline, or reroll knob.

**The first recovery attempt at source `0da0f06fe3a214faa7d30fbbe007e2548f8cb232` failed
pre-CUDA at 0/500.** Root cause: the durable BlobFuse adapter handoff validated the
signed `TrainingResult.adapter_files` manifest by first calling `os.scandir()` on the
mounted `final-adapter/` directory to discover file records, and BlobFuse's enumeration
API returned zero entries even though every expected child path opened successfully by
exact name. The generic, enumeration-based `build_file_manifest` /
`validate_file_manifest` used for local/private trees is unchanged and still rejects
missing, changed, symlink, nonregular, and extra entries by directory listing — that
code path is correct for trees it controls. The durable BlobFuse handoff boundary is
different: it is the one place a directory listing crossing an external mount cannot be
trusted as either present or authoritative. `tau/eval_tools/manifest.py` and
`tau/eval_tools/artifacts.py` (`validate_file_manifest_by_exact_paths`,
`validate_adapter_directory_exact`, `copy_adapter_exclusive_by_manifest`) now validate
that boundary by the exact, already-signed child paths only — no-follow descriptor-safe
opens, inode-bound reads, size/SHA-256 streamed and compared per file and in aggregate,
never inferring a trusted file list from what the directory happens to enumerate. The
private destination this materializes is then re-validated by the original
enumeration-based exact-set check, so arbitrary extra files still fail there. Recovery
now runs at fresh runtime source `2594a4c0bdd1ba8ccc754e12205f4b0201bfef86`; source
`0da0f06fe3a214faa7d30fbbe007e2548f8cb232` and its failed
`eval-post-recovery/inference.log` are superseded evidence, never reused or cleaned up.

Because the BlobFuse enumeration failure is provider/mount-state-dependent and cannot be
fully reproduced offline, the H200 recovery target now requires a separate, additive
zero-GPU preflight: `f12-eval-post-recovery-preflight.yaml` runs on system CPU (no GPU,
no H200 selector/toleration, independent topology) against the same read-only F12
inputs, and performs the identical exact-path adapter validation and private
materialization the H200 job performs before inference — cheaply, on general compute,
before any GPU is scheduled. It publishes a deterministic, timestamp-free
`recovery-preflight.json` (`tau/eval_tools/f12_recovery.py`:
`write_f12_recovery_preflight`) recording both source identities, the exact
manifest/baseline/training-result/adapter digests, rank/step, the private materialized
adapter's own manifest, and a success status. The H200 recovery target requires the
exact `PRIME_RL_RECOVERY_PREFLIGHT_PATH` and pins its SHA-256 in
`PRIME_RL_RECOVERY_PREFLIGHT_SHA256`; `validate_f12_recovery_preflight` strictly checks
the artifact's schema, object identity, source identity, input digests, and private
manifest before inference is permitted to start, and fails closed if the artifact is
absent, unreadable, or any field mismatches.

`PRIME_RL_RECOVERY_PREFLIGHT_SHA256` in `f12-eval-post-recovery.yaml` is currently a
placeholder (64 zero-hex digits, commented `PENDING`) because computing the real digest
requires actually running the CPU preflight target once — this repo's runtime source
was repinned in a code-only session with no cluster access, and the real production
adapter weight bytes are not available outside the cluster. The H200 target's fail-closed
digest check means it cannot proceed until `f12-eval-post-recovery-preflight.yaml` is
run for real and this placeholder is replaced with its actual output digest.

Static recovery validation is:

```bash
uv run --no-sync python tau/render_image.py
tau run validate --config tau/.rendered/f12-eval-post-recovery-preflight.yaml
tau run --config tau/.rendered/f12-eval-post-recovery-preflight.yaml \
  --context aks-ai-runtime-eastus2-admin --dry-run=client
tau run validate --config tau/.rendered/f12-eval-post-recovery.yaml
tau run --config tau/.rendered/f12-eval-post-recovery.yaml \
  --context aks-ai-runtime-eastus2-admin --dry-run=client
```

## Frozen eval manifest and the paired comparison gate

- `tau/eval_tools/manifest.py` — schema-v5 `FrozenEvalManifest` pins the overlay,
  verifiers, taskset, model, eval dataset, and training dataset revisions. Freeze builds
  trusted job-private regular-file trees only long enough to record their identities. The model identity records
  every canonical path, byte size, per-file SHA-256, and an aggregate digest; baseline,
  train, and post-eval reject missing, extra, changed, path-escaping, or symlinked files.
  Training identity binds every dataset file plus ordered `{id, prompt_hash, answer_hash}`
  records, count, source revision/config, and digest. Eval identity binds all 500 IDs and prompt/answer hashes,
  decoding, and grader. Finalization additionally binds the versioned canonical full
  `RLConfig` contract/digest and baseline reward artifact SHA-256.
- `tau/eval_tools/compare.py` — strict JSON loading rejects duplicate keys; reward files
  must carry the expected `baseline`/`post` labels, exact example IDs, and finite binary
  rewards (the pinned grader is source-proven to return exactly `0.0` or `1.0`).
  Baseline finalization strict-loads the fixed baseline file, computes its mean internally,
  requires `[0.10, 0.80]`, and binds its digest. `compare_runs()` rechecks that digest and
  computes a paired percentile bootstrap using exactly 10,000 resamples and seed 0.
  Neither is caller-configurable. The gate passes only if `delta >= +0.03` **and**
  `ci_lower > 0`.
- Standalone baseline/post eval disables prime-rl's default router. The one vLLM engine
  serves health, LoRA admin, and OpenAI traffic on port 8000; startup and adapter-load
  failures are fatal before any reward evidence can be written.
- `tau/eval_tools/live/` directly materializes exact model commit
  `a09a35458c702b33eeacc393d103063234e8bc28` and exact training dataset commit
  `3ed63f49541bdca4382fba28146aadf20d95cb38` beneath a fresh owned
  `/tmp/prime-rl-run-<nonce>/`. It verifies hashes after copying regular files, removes
  write permission, and writes completion markers last. RL consumes only the exact
  private dataset/config/model paths. One Python supervisor creates and logs a durable
  attempt ID, launches the exact `uv run --no-sync rl @ <private-config>` child, streams
  its output, captures its real PID/times/return code, and writes
  `attempts/<attempt-id>/completion.json` only after a zero exit and a verified fixed
  `STABLE` checkpoint. Publication re-parses the durable resolved TOML through the full
  canonical config validator, recomputes its identity, verifies all attempt evidence,
  and writes fixed `training-result.json` last. Completion, publication, and result JSON
  are fsynced in attempt-owned staging before one no-follow, directory-relative,
  atomic-no-replace promotion. The promotion binds the validated stage inode, bytes,
  digest, and parsed object to a strict-reopened final before callbacks or return.
  A swapped or unreadable installed entry is atomically inode-checked into preserved
  quarantine, leaving the fixed path retryable; a pre-existing immutable final is never
  quarantined. Normal code never streams bytes into final names. The supervisor's
  INT/TERM handlers remain
  installed across materialization, RL execution, checkpoint verification, attestation,
  publication, and result installation.

### Proof ladder (in order)

Before any of this, confirm the five primary F10 targets pin
`ce9d0919b6eb266a7c8100b333dc9ea91e95b96f`, then render the image pin with
`uv run --no-sync python tau/render_image.py`. The base model and training dataset
revisions are already immutable pins. Every command below points at
`tau/.rendered/<target>.yaml`, never the bare `tau/<target>.yaml` template, so pin
validation and entrypoint mirroring cannot be skipped.

`tau run --config ...` is the mutating submit/apply operation and must be invoked
exactly once for each target while its Job/Workload exists. Monitor that submission
with `tau run get` and read-only `kubectl get`, `kubectl describe`, or `kubectl logs`;
never issue a second `tau run --config ...` as a status check or retry because it
client-side applies the live Job and can suspend/delete/resume its running pod.

```bash
# 0. Render the checked-in immutable source and image pins.
uv run --no-sync python tau/render_image.py

# 1. Smoke the resolved config on a system CPU node (zero GPU) and fetch evidence.
tau run --config tau/.rendered/smoke.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-smoke -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact smoke-result.json

# 2. Build the draft manifest on a system CPU node (zero GPU) — materializes the
#    model and proves train/eval disjointness up front.
#    (PRIME_RL_RUN_MODE=freeze-draft is the default in the committed YAML.)
tau run --config tau/.rendered/freeze-manifest.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-freeze-manifest -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact draft-manifest.json

# 3. Measure the baseline against the draft manifest (1 GPU).
tau run --config tau/.rendered/eval-baseline.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-eval-baseline -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact rewards.json

# 4. Freeze the manifest from immutable baseline evidence (edit only
#    tau/freeze-manifest.yaml: PRIME_RL_RUN_MODE=freeze-finalize, then re-render).
#    Finalization reads the fixed PRIME_RL_BASELINE_REWARDS_PATH; no mean is supplied.
uv run --no-sync python tau/render_image.py
tau run --config tau/.rendered/freeze-manifest.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-freeze-manifest -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact frozen-eval-manifest.json

# 5. Train (2 GPU). Save the exact attempt_id printed by the supervisor in Tau logs.
tau run --config tau/.rendered/train.yaml --context aks-ai-runtime-eastus2-admin
tau run logs prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin | grep '\[training-supervisor\] attempt_id='
ATTEMPT_ID='<copy-the-exact-logged-id>'
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact "attempts/${ATTEMPT_ID}/preflight.json"
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact "attempts/${ATTEMPT_ID}/completion.json"
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact "attempts/${ATTEMPT_ID}/resolved-train.toml"
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact training-result.json
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact "attempts/${ATTEMPT_ID}/run-output/metrics.jsonl"
# Only if supervisor logs report private cleanup failure after durable success:
tau run get prime-rl-math-7b-h200-train -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin \
  --artifact "attempts/${ATTEMPT_ID}/private-cleanup-diagnostic.json"

# 6. Post-training eval + comparison (1 GPU) — consumes only the fixed verified
#    all fixed training evidence, then copies train/final-adapter into job-private
#    immutable storage before loading it.
tau run --config tau/.rendered/eval-post.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-eval-post -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact rewards.json
tau run get prime-rl-math-7b-h200-eval-post -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact comparison.json
```

### Harder-math tier curve

The additive target pins F11 source `bf0e4c478648ec97eb0eea917cba71348af86d1e`
while binding the exact finalized F10 manifest bytes. It evaluates all 5,030 trusted
eval records once with concurrency 128: base 1,331, core 2,345, and hard 1,354.
`verify_boxed_math_answer` remains synchronous so its timeout mechanism is not moved
across worker threads. Any request/grader failure or
`base_mean - hard_mean < 0.15` fails the Job after immutable evidence is written.
Raw artifacts retain completions, IDs, hashes, binary rewards, and failures, but not
dataset prompts or gold answers.

```bash
uv run --no-sync python tau/render_image.py
tau run --config tau/.rendered/harder-tier-curve.yaml \
  --context aks-ai-runtime-eastus2-admin
tau run status prime-rl-harder-math-7b-h200-tier-curve -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --watch
tau run logs prime-rl-harder-math-7b-h200-tier-curve -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin -f
for artifact in raw-base.json raw-core.json raw-hard.json tier-curve.v1.json inference.log; do
  tau run get prime-rl-harder-math-7b-h200-tier-curve -n pretraining-data \
    --context aks-ai-runtime-eastus2-admin --artifact "$artifact"
done
```

The measured run used Job `prime-rl-harder-math-7b-h200-tier-curve`, Workload
`job-prime-rl-harder-math-7b-h200-tier-curve-46b30`, and pod
`prime-rl-harder-math-7b-h200-tier-curve-8wn67` on
`aks-h200pool-35981772-vmss000000`. The server became healthy after 70 seconds,
reached 100% H200 utilization with 129,848 MiB allocated, and completed all 5,030
HTTP-200 requests with zero request/grader failures:

| Tier | Correct / count | Mean | Wilson 95% CI |
|---|---:|---:|---:|
| base | 1,218 / 1,331 | 0.9151 | [0.8989, 0.9289] |
| core | 1,832 / 2,345 | 0.7812 | [0.7641, 0.7975] |
| hard | 716 / 1,354 | 0.5288 | [0.5022, 0.5553] |

`base_mean - hard_mean = 0.3863`, so the fixed `>= 0.15` environment gate passed.
The raw base/core/hard JSON, aggregate, and inference log were fetched before cleanup,
the package's own trusted-catalog API revalidated all records, and every file re-fetched
byte-identically from the PVC after Job/Pod/Workload deletion.

If a job is interrupted **before** `completion.json`, submit the unchanged train target
again. The supervisor creates a new attempt ID and ignores stale attempt evidence; it
never reuses or enumerates attempts. If interruption occurs **after** the logged
attempt's `completion.json` was durably written but before `training-result.json`, do not
rerun RL. Use the exact logged ID to recover publication:

```bash
uv run --no-sync python -m tau.eval_tools.cli recover-publish \
  --manifest /data/pretraining-data/prime-rl-math-7b-h200/generations/ce9d0919b6eb266a7c8100b333dc9ea91e95b96f/manifest/frozen-eval-manifest.json \
  --output-dir /data/pretraining-data/prime-rl-math-7b-h200/generations/ce9d0919b6eb266a7c8100b333dc9ea91e95b96f/train \
  --attempt-id "$ATTEMPT_ID"
```

Run that command only in the same pinned image/overlay with `blob-training` mounted.
It derives exact paths from the supplied ID, strict-verifies the existing preflight,
canonical resolved TOML, real-process attestation, STABLE marker, and adapter hashes,
then quarantines only that attempt's stale staging and completes atomic publication without
launching RL. This includes the adapter-installed/publication-not-yet-installed
interruption point. It also accepts an fsynced `.completion.json.stage` only when
`completion.json` is absent and the staged attestation passes the same full validation,
then atomically promotes it without replacement. Validation, fsync, and promotion remain
bound to one no-follow regular-file handle and captured inode; the installed final is
strict-reopened and must reproduce the exact bytes, digest, object, and inode before
publication. Malformed, stale, or suspicious stages are atomically renamed relative to
the opened attempt directory into unique `.completion.json.stage.quarantine-*` evidence.
The quarantine is no-follow inode-verified and preserved for diagnosis, never unlinked;
recovery ignores quarantine names, so a strict-valid final succeeds on the next call.
The same opened-attempt-directory, no-follow inode protocol moves an interrupted
`.publication.stage` directory to unique `.publication.stage.quarantine-*` evidence,
reopens it as a directory, verifies the captured token, fsyncs the parent, and preserves
it. A swapped directory or adapter alias is quarantined but never traversed or deleted;
an inode mismatch fails the current call while a later strict recovery ignores the
quarantine.
When both names exist, the final wins only if the valid stage matches exactly.
A malformed final publication fails rather than being replaced. A
different attempt can reuse an exactly matching installed adapter only through this
explicit recovery command; normal training fails. A mismatched/reused ID fails, and an
existing fixed `training-result.json` prevents a normal training rerun.

`tau run cancel` sends TERM to the shell entrypoint. During training the shell has the
supervisor in `CHILD_PID`, forwards TERM/INT, waits, and propagates status 143/130. The
supervisor owns a new RL process group, forwards the same signal to that entire group,
waits/reaps it, and keeps the handlers active through all post-wait validation and
publication. A signal at any point before the fixed result is complete aborts as 143/130,
and cannot create later publication or result evidence. Before adapter install, the
supervisor captures the fully validated staged directory inode/device and verifies the
same no-follow token at the final name after atomic rename. Once installed, `final-adapter`
is never recursively deleted: cancellation/error preserves it plus completion evidence
for explicit hash-bound recovery, while a conflicting adapter fails closed. A verified
adapter reused from an earlier attempt is likewise preserved. Cancellation remains failure
even if the child reports zero; the next normal submission receives a fresh attempt ID.

After `training-result.json` and its bound artifacts strict-reload successfully, failure to
remove the job-private run root cannot invalidate that durable success. The supervisor
atomically writes the attempt-scoped
`attempts/<attempt-id>/private-cleanup-diagnostic.json` before logging the cleanup error.
That optional artifact records diagnostics only; it does not alter or replace success
evidence. If diagnostic persistence or stderr itself fails, exit remains zero and the
already-verified success evidence remains immutable.

Eval, recovery eval, and harder-tier-curve inference use the same failure-safe
publication sequence.
The launcher strictly validates the pod `HOSTNAME` as a DNS label and exclusively
opens `inference.attempt-<hostname>.log` with no symlink following. TERM, evaluation
failure, comparison publication failure, or an unexpected server exit preserves that
attempt file and leaves the fixed `inference.log` absent, so a replacement pod with a
different hostname can start without deleting evidence. A same-pod rerun collides with
its own attempt file and fails closed. After rewards are durable, and after post-eval
comparison is durable (including a valid failed gate), the wrapper terminates the
dedicated inference process group and waits until no descendant can retain the writer,
so BlobFuse observes a closed log. It then streams a no-follow snapshot of the attempt
inode, size, and SHA-256 digest without retaining the unbounded log in memory; atomically
renames without replacement; fsyncs; and strict-reopens the fixed final to verify the
same inode, size, and digest.
An existing final, symlink, path swap, or concurrent losing promotion fails without
overwriting either the final or the losing attempt. The fixed log is therefore the
last completion marker: interruption before rewards leaves only the attempt log and
allows a full retry; interruption after rewards but before comparison requires the
documented comparison-only path; interruption after result evidence but before log
promotion preserves complete JSON plus the attempt log but does not claim final log
success; interruption after verified promotion leaves the coherent immutable set.
Comparison exits 0 for a passed gate, 1 for a valid failed gate, and 2 for invalid
input/publication failure, so only the first two states may reach log promotion and a
failed gate still leaves the Job failed.

Every fetch above names an explicit `--artifact <file>` rather than listing the output
directory: W1's storage proof found directory listing on `blob-training` unreliable
(exact-file write/fetch/re-fetch passed; listing did not), so every script in this
implementation writes successful results to a fixed, predictable filename inside its
own `storage.output` (`draft-manifest.json`, `frozen-eval-manifest.json`,
`rewards.json`, `comparison.json`, `inference.log`) specifically so callers never have
to list a directory to find them. Failed inference evidence uses the exact
`inference.attempt-<hostname>.log` name logged before launch. Training attempt artifacts
are found only from the ID printed in logs, never by enumerating `attempts/`. Reward,
comparison, smoke, training, manifest, and final-log evidence is created exclusively;
an existing filename fails instead of being overwritten.

`eval-baseline` binds the stable evaluation identity available in the draft. Finalization
then binds that exact artifact digest and the effective training configuration into the
full immutable identity. Post-eval must match both identities and the same baseline digest.

## Artifact manifest

Every exact filename any target writes, and the `storage.output` directory it lands
under. Nothing here is discovered by listing a directory (see the note above and W1's
storage proof) — fetch each by its exact name with `--artifact <file>`.

| Target | `storage.output` | Exact artifact(s) | Purpose |
|---|---|---|---|
| `smoke` | `<generation-root>/smoke` | `smoke-result.json` | written only after `rl --dry-run` produced all three expected resolved TOMLs |
| `freeze-manifest` | `<generation-root>/manifest` | `draft-manifest.json`, `frozen-eval-manifest.json` | unfrozen draft (pass 1) and the immutable frozen manifest (pass 2) `tau/eval_tools/manifest.py` reads/writes |
| `eval-baseline` | `<generation-root>/eval-baseline` | successful `rewards.json`, `inference.log`; failed `inference.attempt-<hostname>.log` | baseline per-example rewards (`RewardRecord`) `tau/eval_tools/compare.py` consumes; fixed log publishes last |
| `eval-post` | `<generation-root>/eval-post` | successful `rewards.json`, `comparison.json`, `inference.log`; failed `inference.attempt-<hostname>.log` | post-training rewards + the `ComparisonResult` (delta, bootstrap CI, pass/fail); fixed log publishes last |
| `f12-eval-post-recovery` | recovery `<generation-root>/eval-post-recovery` | successful `rewards.json`, `comparison.json`, `inference.log`; failed `inference.attempt-<hostname>.log` | one fixed-input post-only recovery; comparison records recovery runtime source plus frozen F12 source; F12 source generation remains read-only; requires a matching `recovery-preflight.json` sibling before inference |
| `f12-eval-post-recovery-preflight` | recovery `<generation-root>/eval-post-recovery-preflight` | `recovery-preflight.json` | system-CPU, zero-GPU: the same exact-path adapter validation + private materialization the H200 recovery job performs, run first on general compute; deterministic, timestamp-free, digest-pinned by the H200 target |
| `train` | `<generation-root>/train` | fixed `training-result.json` and `final-adapter/`; exact logged `attempts/<attempt-id>/{preflight.json,resolved-train.toml,completion.json,publication.json,run-output/metrics.jsonl}`; optional `attempts/<attempt-id>/private-cleanup-diagnostic.json` | the trusted supervisor owns launch and attestation; attempt evidence binds the exact config/process/STABLE adapter, publication fsyncs and atomically installs without replacement, and writes the fixed result last; a post-success private-cleanup failure writes the optional diagnostic without changing success |
| `harder-tier-curve` | F11 `<generation-root>/tier-curve` | successful `raw-base.json`, `raw-core.json`, `raw-hard.json`, `tier-curve.v1.json`, `inference.log`; failed `inference.attempt-<hostname>.log` | full trusted harder-math catalog, exact F10 base-model/decoding contract, fixed `0.15` hardness gate; fixed log publishes last |

Explicit-file fetch while the run's Workload/Job still exists (the proven, reliable path):

```bash
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin \
  --artifact rewards.json
```

Fallback once the Job/Workload has been deleted (`tau run get <job-name>` then has
nothing to resolve the recorded output path from) — explicit `--path` plus `--pvc`,
proven by W1's storage probe after its Job was cancelled:

```bash
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin \
  --path /data/pretraining-data/prime-rl-math-7b-h200/generations/ce9d0919b6eb266a7c8100b333dc9ea91e95b96f/eval-post/rewards.json \
  --pvc blob-training
```

## Operator commands

**Before every submit:** confirm the five F10 templates pin runtime
`ce9d0919b6eb266a7c8100b333dc9ea91e95b96f` and `harder-tier-curve.yaml` pins
F11 runtime `bf0e4c478648ec97eb0eea917cba71348af86d1e`, then render. Do not pin a
target to its later pin/docs commit: that would be a self-reference. The pinned
`Qwen/Qwen2.5-7B-Instruct` revision is
`a09a35458c702b33eeacc393d103063234e8bc28`. Every model-serving phase downloads that
exact revision directly into a fresh job-private cache, validates the complete frozen
file manifest, and serves only the non-writable private regular-file tree. Post-eval does
the same for the durable final adapter after verifying all training evidence.

```bash
uv run --no-sync python tau/render_image.py
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
# Submit exactly once. Re-running this line updates the existing Job and can replace its pod.
tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin --artifact <name>
kubectl get job,pod,workload -n pretraining-data --context aks-ai-runtime-eastus2-admin
kubectl logs -n pretraining-data --context aks-ai-runtime-eastus2-admin <pod-name>
tau run cancel <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin
```

After submission, only `tau run get` and read-only `kubectl` inspection are monitoring
operations. Never run the submit/apply form a second time while those objects exist.

Always pass `--artifact <name>` or an exact `--path ... --pvc blob-training` — W1's
storage proof on `blob-training` found exact-file
write/fetch/re-fetch reliable but directory listing (the no-`--artifact` form of
`tau run get`) unreliable. Every script here writes to a fixed, known filename per
target (see the proof ladder above), so no command in this doc lists a directory to find
its output.

`<job-name>` equals each YAML's `name:` field
(`prime-rl-math-7b-h200-{smoke,freeze-manifest,eval-baseline,eval-post,train}` or
`prime-rl-harder-math-7b-h200-tier-curve`, or
`prime-rl-math-7b-h200-f12-eval-post-recovery`, or
`prime-rl-math-7b-h200-f12-eval-post-recovery-preflight`).
Never submit the bare `tau/<target>.yaml` template directly; use the validated rendered
copy so image-pin validation cannot be skipped.

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
  validation below; only documented config field names such as `api_key_var` should
  match.

## Static validation contract

Run all of the following before any submit:

- `tau run validate --config tau/.rendered/<target>.yaml` for all 12 checked-in targets.
- `tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin --dry-run=client`
  for all 12 checked-in targets — rendered `batch/v1 Job`s with correct GPU requests/limits, node
  selectors, topology annotation, storage mounts, and env vars; the embedded
  `TAU_SCRIPT_B64` was verified byte-identical to `tau/scripts/run-prime-rl.sh` by
  decoding it back and diffing.
- `uv run --no-sync python tau/render_image.py` (and its
  `--check-only`/malformed-pin/missing-pin/mismatched-template error paths); confirm all
  six checked-in and rendered targets carry the exact public digest.
- `bash -n` and `shellcheck` (zero warnings) on `tau/scripts/run-prime-rl.sh`.
- stdlib JSON/TOML parsing of the image pin and training config, cross-referenced
  field-by-field against `packages/prime-rl-configs/src/prime_rl/configs/{rl,orchestrator,trainer}.py`.
- `PYTHONPATH=.:src uv run --no-project` with editable `prime-rl-configs`,
  `verifiers`, `math-env-v1`, and `math500-v1`, then
  `pytest -q tau/eval_tools/tests` — 193 tests covering content manifests,
  path/symlink rejection, ordered train identity, immutable attempt/process evidence,
  a non-mocked real-`RLConfig` TOML round trip, locked-HF snapshot symlink cleanup,
  fixed bootstrap, cancellation, retry/recovery-safe atomic publication, and
  source-generation isolation across every cross-mode reference, including the
  explicit F10 model contract consumed by F11.
- `ruff check` / `ruff format --check` clean on every new Python file under `tau/`.
- `uv run --no-project python -m py_compile` on the `live/` scripts.
- `uv lock --check`, `git diff --check`, and secret/dependency/submodule/dtype scans.

## Measured result and remaining proof

F10 proved the operational path end to end: CPU smoke, frozen 500-example manifest,
one-H200 baseline, 50 optimizer steps on two H200s, immutable rank-16 adapter
publication, and one-H200 post-eval. The adapter contained 161,533,566 bytes with
aggregate SHA-256
`a2808199450da1506edcb491f20917ff5dbbaa2f54fbeae07f4f3817c4d06d23`.
The authoritative frozen comparison was `0.7480 -> 0.7500`, delta `+0.0020`,
paired-bootstrap 95% CI `[-0.0160, +0.0200]`; it failed the fixed `+0.03` and
positive-lower-bound gate. This negative tuple is retained and must not be rerun,
slice-shopped, or seed-shopped.

Environment reward here is deterministic boxed-answer exact/math verification and is
used as a benchmark-accuracy proxy, not a general reasoning-quality measure. Public
MATH, MATH-500, and AIME material may have appeared in model pretraining, so even a
future positive result would demonstrate only this pinned setup, not uncontaminated
generalization.

F11's 5,030-record harder-math tier curve is proven live with zero failures and
`base_mean - hard_mean = 0.3863`. The repo-local custom environment therefore meets
its empirical acceptance gate. The primary RL hill-climb remains unproven; any future
attempt after F10 must be a new, separately justified and predeclared source generation.
