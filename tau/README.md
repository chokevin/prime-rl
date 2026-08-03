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
    json_io.py             # duplicate-key rejection + exclusive JSON evidence writes
    manifest.py
    artifacts.py           # attempt-scoped evidence, atomic publication, adapter handoff
    compare.py
    cli.py
    tests/                 # focused pytest cases, no GPU/network required
    live/                   # GPU-container-only: real dataset/model/inference-server scripts
      freeze_manifest_live.py
      run_frozen_eval_live.py
      validate_training_data_live.py # canonical resolved-config parser/identity
      training_supervisor_live.py    # owns private inputs, RL child, attestation, publication
configs/tau/math-7b-h200/
  train.toml             # derived from configs/basic/hendrycks-sanity/rl.toml
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
Baseline section). The immutable runtime overlay is
`9444b3dfe143abb1a9d8d392cfe60ad007e0bc1c`. That commit contains the complete Tau
wrapper/eval tooling and the workspace-locked `environments/harder_math_v1` package.
Its dependency set matches the pinned image: `harder-math-v1` uses only `datasets` and
`verifiers`, which are already present. The later integration commit changes only
checked-in Tau pins/docs and the client-side selection fixture; runtime code continues
to come from the immutable overlay.

**Why a pin file + render step when the digest is also checked in 5×:** the templates
must be directly inspectable and runnable-looking, while `tau/image.pin.json` remains
the source of truth used to detect drift. `tau/render_image.py` validates that every
template carries exactly that real digest, writes validated copies to gitignored
`tau/.rendered/`, and mirrors `tau/scripts/` alongside them because `entrypoint:` paths
resolve relative to the config file's own directory. It refuses to stage anything if
the pin is missing, malformed, set to `latest`, or disagrees with any target. Re-pinning
therefore requires an intentional edit to the pin and all five visible targets.

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

**Narrow package-install escape hatch:** a later harder-math tier-curve target may set
`runtime.env.PRIME_RL_EXTRA_ENV_PACKAGE_DIR=environments/harder_math_v1`. No other path
is accepted. The wrapper rejects absolute paths, `..`, symlink components, and canonical
escapes before running `uv pip install --no-deps -e` on that verified checkout path. It
also replaces rather than extends inherited `PYTHONPATH`. The primary experiment does
not need this escape hatch: `math-env-v1` and `math500-v1` already ship in
`deps/research-environments`, baked into the image.

The corresponding prime-rl source-selection overlay is
`configs/tau/math-7b-h200/harder-math-v1-hard.toml`. It is a static fixture for config
selection and a future tier-curve run; none of the five primary targets use it. Compose
it after `train.toml` to replace only `orchestrator.train.source`. The primary
`math-env-v1` training and all-500 `math500-v1` frozen-evaluation proof ladder remain the
default.

**Harder-math metadata boundary:** the primary frozen manifest remains a strict,
ordered 500-row MATH-500 contract. A `harder-math-v1` tier curve is a separate
`tier-curve.v1` artifact covering its full 5,030-row eval catalog; it must never be
inserted into `FrozenEvalManifest.examples`. For later disjointness tooling,
`tier-curve.v1.records[].prompt_sha256` is the field corresponding to Tau's
`prompt_hash`. `content_sha256` hashes prompt plus gold and is not Tau's
answer-only `answer_hash`; `record_id` is source-qualified text rather than the
MATH-500 integer ID. Keeping those artifacts separate lets Tau consume the W4a prompt
hash set later without weakening or renaming the primary manifest fields and without
claiming any empirical tier result.

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

`max_steps = 50` is a placeholder pending a real throughput measurement (see "What is
not proven yet"). Finalization binds it in the canonical config identity. Training and
post-eval derive the one source-proven final checkpoint path
`weights/step_<max_steps>/lora_adapters` from that frozen identity; there is no runtime
step override. The handoff never enumerates a storage directory.

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
  are fsynced in attempt-owned staging before atomic no-replace installation; normal code
  never streams bytes into their final names. The supervisor's INT/TERM handlers remain
  installed across materialization, RL execution, checkpoint verification, attestation,
  publication, and result installation.

### Proof ladder (in order)

Before any of this, confirm every target pins
`9444b3dfe143abb1a9d8d392cfe60ad007e0bc1c`, then render the image pin with
`uv run --no-sync python tau/render_image.py`. The base model and training dataset
revisions are already immutable pins. Every command below points at
`tau/.rendered/<target>.yaml`, never the bare `tau/<target>.yaml` template, so pin
validation and entrypoint mirroring cannot be skipped.

```bash
# 0. Render the checked-in immutable source and image pins.
uv run --no-sync python tau/render_image.py

# 1. Smoke the resolved config and fetch the fixed success evidence.
tau run --config tau/.rendered/smoke.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-smoke -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact smoke-result.json

# 2. Build the draft manifest (CPU-only) — materializes the model and proves
#    train/eval disjointness up front.
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

# 6. Post-training eval + comparison (1 GPU) — consumes only the fixed verified
#    all fixed training evidence, then copies train/final-adapter into job-private
#    immutable storage before loading it.
tau run --config tau/.rendered/eval-post.yaml --context aks-ai-runtime-eastus2-admin
tau run get prime-rl-math-7b-h200-eval-post -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact rewards.json
tau run get prime-rl-math-7b-h200-eval-post -n pretraining-data \
  --context aks-ai-runtime-eastus2-admin --artifact comparison.json
```

If a job is interrupted **before** `completion.json`, submit the unchanged train target
again. The supervisor creates a new attempt ID and ignores stale attempt evidence; it
never reuses or enumerates attempts. If interruption occurs **after** the logged
attempt's `completion.json` was durably written but before `training-result.json`, do not
rerun RL. Use the exact logged ID to recover publication:

```bash
uv run --no-sync python -m tau.eval_tools.cli recover-publish \
  --manifest /data/pretraining-data/prime-rl-math-7b-h200/manifest/frozen-eval-manifest.json \
  --output-dir /data/pretraining-data/prime-rl-math-7b-h200/train \
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

Every fetch above names an explicit `--artifact <file>` rather than listing the output
directory: W1's storage proof found directory listing on `blob-training` unreliable
(exact-file write/fetch/re-fetch passed; listing did not), so every script in this
implementation writes its results to a fixed, predictable filename inside its own
`storage.output` (`draft-manifest.json`, `frozen-eval-manifest.json`, `rewards.json`,
`comparison.json`, `inference.log`) specifically so callers never have to list a
directory to find them. Attempt artifacts are found only from the ID printed in logs,
never by enumerating `attempts/`. Reward, comparison, smoke, training, and manifest
evidence is created exclusively; an existing filename fails instead of being overwritten.

`eval-baseline` binds the stable evaluation identity available in the draft. Finalization
then binds that exact artifact digest and the effective training configuration into the
full immutable identity. Post-eval must match both identities and the same baseline digest.

## Artifact manifest

Every exact filename any target writes, and the `storage.output` directory it lands
under. Nothing here is discovered by listing a directory (see the note above and W1's
storage proof) — fetch each by its exact name with `--artifact <file>`.

| Target | `storage.output` | Exact artifact(s) | Purpose |
|---|---|---|---|
| `smoke` | `.../prime-rl-math-7b-h200/smoke` | `smoke-result.json` | written only after `rl --dry-run` produced all three expected resolved TOMLs |
| `freeze-manifest` | `.../prime-rl-math-7b-h200/manifest` | `draft-manifest.json`, `frozen-eval-manifest.json` | unfrozen draft (pass 1) and the immutable frozen manifest (pass 2) `tau/eval_tools/manifest.py` reads/writes |
| `eval-baseline` | `.../prime-rl-math-7b-h200/eval-baseline` | `rewards.json`, `inference.log` | baseline per-example rewards (`RewardRecord`) `tau/eval_tools/compare.py` consumes |
| `eval-post` | `.../prime-rl-math-7b-h200/eval-post` | `rewards.json`, `comparison.json`, `inference.log` | post-training rewards + the `ComparisonResult` (delta, bootstrap CI, pass/fail) |
| `train` | `.../prime-rl-math-7b-h200/train` | fixed `training-result.json` and `final-adapter/`; exact logged `attempts/<attempt-id>/{preflight.json,resolved-train.toml,completion.json,publication.json,run-output/metrics.jsonl}` | the trusted supervisor owns launch and attestation; attempt evidence binds the exact config/process/STABLE adapter, publication fsyncs and atomically installs without replacement, and writes the fixed result last |

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
  --path /data/pretraining-data/prime-rl-math-7b-h200/eval-post/rewards.json \
  --pvc blob-training
```

## Operator commands

**Before every submit:** confirm all five checked-in templates pin the immutable runtime
overlay `9444b3dfe143abb1a9d8d392cfe60ad007e0bc1c`, then render. Do not replace it with the
later integration/pin commit: that would be a self-reference and that commit changes no
runtime code. The pinned `Qwen/Qwen2.5-7B-Instruct` revision is
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
tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin
tau run status <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin --watch
tau run logs <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin -f
tau run get <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin --artifact <name>
tau run cancel <job-name> -n pretraining-data --context aks-ai-runtime-eastus2-admin
```

Always pass `--artifact <name>` or an exact `--path ... --pvc blob-training` — W1's
storage proof on `blob-training` found exact-file
write/fetch/re-fetch reliable but directory listing (the no-`--artifact` form of
`tau run get`) unreliable. Every script here writes to a fixed, known filename per
target (see the proof ladder above), so no command in this doc lists a directory to find
its output.

`<job-name>` equals each YAML's `name:` field (`prime-rl-math-7b-h200-{smoke,freeze-manifest,eval-baseline,eval-post,train}`).
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

- `tau run validate --config tau/.rendered/<target>.yaml` for all 5 targets.
- `tau run --config tau/.rendered/<target>.yaml --context aks-ai-runtime-eastus2-admin --dry-run=client`
  for all 5 targets — rendered `batch/v1 Job`s with correct GPU requests/limits, node
  selectors, topology annotation, storage mounts, and env vars; the embedded
  `TAU_SCRIPT_B64` was verified byte-identical to `tau/scripts/run-prime-rl.sh` by
  decoding it back and diffing.
- `uv run --no-sync python tau/render_image.py` (and its
  `--check-only`/malformed-pin/missing-pin/mismatched-template error paths); confirm all
  five checked-in and rendered targets carry the exact public digest.
- `bash -n` and `shellcheck` (zero warnings) on `tau/scripts/run-prime-rl.sh`.
- stdlib JSON/TOML parsing of the image pin and training config, cross-referenced
  field-by-field against `packages/prime-rl-configs/src/prime_rl/configs/{rl,orchestrator,trainer}.py`.
- `PYTHONPATH=.:src uv run --no-project` with editable `prime-rl-configs`,
  `verifiers`, `math-env-v1`, and `math500-v1`, then
  `pytest -q tau/eval_tools/tests` — 131 tests covering content manifests,
  path/symlink rejection, ordered train identity, immutable attempt/process evidence,
  a non-mocked real-`RLConfig` TOML round trip, locked-HF snapshot symlink cleanup,
  fixed bootstrap, cancellation, and retry/recovery-safe atomic publication.
- `ruff check` / `ruff format --check` clean on every new Python file under `tau/`.
- `uv run --no-project python -m py_compile` on the `live/` scripts.
- `uv lock --check`, `git diff --check`, and secret/dependency/submodule/dtype scans.

## What is *not* proven yet (explicitly out of this session's scope)

- No Tau job has been submitted (no `--dry-run=server`, no real submit). This session's
  scope (W4b) was static config/scripts/tests; live cluster execution is W5–W8.
- The previously reproduced local runtime blockers are covered at source commit
  `9444b3dfe143abb1a9d8d392cfe60ad007e0bc1c`: cleanup was exercised against a real
  locked-`huggingface-hub==1.16.1` snapshot with symlinks, and fixed JSON evidence has
  interruption/race/retry coverage. This does not prove the end-to-end job.
- The full `tau/eval_tools/live/*.py` phases have never been executed against the real
  experiment dataset, base model, or inference server.
- `max_steps = 50` and the resource requests are placeholders, not throughput-measured.
- The frozen eval manifest has not been created; no baseline has been measured; no
  training has run; no comparison has been computed.
- Whether `Qwen/Qwen2.5-7B-Instruct` + `DefaultRenderer` + rank-16 LoRA actually loads
  and trains cleanly on this image is unverified — the W2 memo's compatibility read is
  source-backed, not execution-tested.
