#!/usr/bin/env bash
#
# Self-contained Tau entrypoint for the prime-rl math-7b-h200 experiment.
#
# Tau embeds this script's *literal file content* as a base64-encoded env var baked
# into the rendered Job spec, and decodes it to /tmp/run.sh inside the pod at start
# time (verified empirically — see tau/README.md's "How Tau runs this script"). It does
# NOT ship the surrounding repo tree alongside it. This script must therefore be fully
# self-contained: no `source`-ing sibling files, and it fetches its own copy of the repo
# (the "immutable overlay") rather than assuming one is already checked out beside it.
# Tau sets the container's workingDir to /data (the durable PVC mount); the baked image
# installs prime-rl under /app with its venv at /app/.venv, so every mode below starts
# with `cd /app`.
#
# Modes (selected via $PRIME_RL_RUN_MODE, set per Tau target in tau/*.yaml's runtime.env):
#   smoke          - CPU-only config validation: `rl --dry-run`, no model/GPU/network.
#   freeze-draft   - load math500-v1 + math-env-v1, prove disjointness, write a draft
#                    frozen-eval manifest (CPU-only, no GPU).
#   freeze-finalize- given a measured baseline mean, freeze the immutable manifest.
#   eval           - 1 GPU: standalone frozen-eval replay (baseline or post-training).
#   train          - 2 GPU: bounded RL training; refuses to start without a frozen
#                    manifest already on durable storage.
set -euo pipefail

log() { printf '[run-prime-rl] %s\n' "$*" >&2; }
die() {
    log "FATAL: $*"
    exit 1
}

: "${PRIME_RL_RUN_MODE:?PRIME_RL_RUN_MODE must be set (smoke|freeze-draft|freeze-finalize|eval|train)}"
: "${TAU_OUTPUT_DIR:?TAU_OUTPUT_DIR not set by Tau}"
: "${PRIME_RL_REPO_URL:?PRIME_RL_REPO_URL must be set (e.g. https://github.com/chokevin/prime-rl.git)}"
: "${PRIME_RL_REPO_SHA:?PRIME_RL_REPO_SHA must be set to the exact commit to overlay}"

OVERLAY_DIR="${PRIME_RL_OVERLAY_DIR:-/tmp/prime-rl-overlay}"

# --- Step 1: immutable source overlay ------------------------------------------------
# Fetch the exact pinned commit into a scratch checkout and verify it landed precisely
# on that SHA (a shallow fetch of one exact commit is itself the verification: git
# refuses to name an unrelated object FETCH_HEAD, and the rev-parse check below catches
# any tag/branch-move surprise). Deliberately never `uv sync` here: the baked image's
# venv (/app/.venv) already satisfies this branch's locked dependencies for any commit
# that only touches TOML/scripts/docs (true for every commit on this branch so far —
# see tau/README.md's "Image and overlay strategy" for the full argument and the
# narrow, opt-in escape hatch below for a future dependency-adding commit).
log "fetching ${PRIME_RL_REPO_URL}@${PRIME_RL_REPO_SHA} into ${OVERLAY_DIR}"
rm -rf "$OVERLAY_DIR"
git init --quiet "$OVERLAY_DIR"
git -C "$OVERLAY_DIR" remote add origin "$PRIME_RL_REPO_URL"
git -C "$OVERLAY_DIR" fetch --quiet --depth 1 origin "$PRIME_RL_REPO_SHA"
git -C "$OVERLAY_DIR" checkout --quiet --force FETCH_HEAD
resolved_sha="$(git -C "$OVERLAY_DIR" rev-parse HEAD)"
[ "$resolved_sha" = "$PRIME_RL_REPO_SHA" ] || die "overlay checkout resolved to ${resolved_sha}, expected ${PRIME_RL_REPO_SHA} — aborting rather than run an unverified tree"
log "overlay verified at ${resolved_sha}"

# Optional narrow install of one repo-local environment package — forward-compatible
# with a future harder-math environment (W4a/W9), not used by the primary
# math-env-v1/math500-v1 experiment, which ships in deps/research-environments already
# baked into the image. --no-deps + a single `-e` target keeps this narrow; never a
# full/`--inexact` `uv sync` of the whole workspace.
if [ -n "${PRIME_RL_EXTRA_ENV_PACKAGE_DIR:-}" ]; then
    pkg_dir="${OVERLAY_DIR}/${PRIME_RL_EXTRA_ENV_PACKAGE_DIR}"
    [ -d "$pkg_dir" ] || die "PRIME_RL_EXTRA_ENV_PACKAGE_DIR=${PRIME_RL_EXTRA_ENV_PACKAGE_DIR} not found in the overlay checkout"
    log "installing extra environment package from ${pkg_dir} (narrow: --no-deps, no full sync)"
    cd /app
    uv pip install --no-deps -e "$pkg_dir"
fi

cd /app
export PYTHONPATH="${OVERLAY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# --- Step 2: child-process lifecycle helpers -----------------------------------------
CHILD_PID=""
cleanup() {
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        log "stopping child process (pid ${CHILD_PID})"
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
}
# Backgrounding the long-running child + trapping here (rather than running it in the
# foreground) is required, not decorative: bash defers a foreground command's signal
# delivery until that command exits, so a SIGTERM sent to this script (Tau/Kubernetes
# cancellation, or a resilience-triggered retry) would otherwise never reach us until
# the child already finished on its own.
trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1" timeout_s="${2:-1800}" waited=0
    until curl -sf -o /dev/null "$url"; do
        if [ "$waited" -ge "$timeout_s" ]; then
            die "inference server did not become healthy within ${timeout_s}s (${url})"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "inference server healthy after ${waited}s (${url})"
}

# --- Step 3: mode dispatch ------------------------------------------------------------
case "$PRIME_RL_RUN_MODE" in
smoke)
    : "${PRIME_RL_CONFIG_REL:?PRIME_RL_CONFIG_REL must be set for smoke mode}"
    log "config smoke: uv run rl --dry-run @ ${OVERLAY_DIR}/${PRIME_RL_CONFIG_REL}"
    uv run --no-sync rl @ "${OVERLAY_DIR}/${PRIME_RL_CONFIG_REL}" \
        --output-dir "$TAU_OUTPUT_DIR" --dry-run
    log "smoke OK: resolved per-process TOMLs written under ${TAU_OUTPUT_DIR}/configs/"
    ;;

freeze-draft)
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live draft \
        --model-name "$PRIME_RL_MODEL_NAME" \
        --out "${PRIME_RL_MANIFEST_DIR}/draft-manifest.json" \
        --temperature "${PRIME_RL_EVAL_TEMPERATURE:-0.0}" \
        --seed "${PRIME_RL_EVAL_SEED:-0}"
    ;;

freeze-finalize)
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    : "${PRIME_RL_BASELINE_MEAN:?PRIME_RL_BASELINE_MEAN must be set to the mean reward measured by the eval-baseline target against the draft manifest}"
    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live finalize \
        --draft "${PRIME_RL_MANIFEST_DIR}/draft-manifest.json" \
        --baseline-mean "$PRIME_RL_BASELINE_MEAN" \
        --out "${PRIME_RL_MANIFEST_DIR}/frozen-eval-manifest.json"
    ;;

eval)
    : "${PRIME_RL_EVAL_LABEL:?PRIME_RL_EVAL_LABEL must be baseline or post}"
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MANIFEST_PATH:?PRIME_RL_MANIFEST_PATH must be set (draft for the very first baseline, frozen thereafter)}"
    if [ "$PRIME_RL_EVAL_LABEL" = "post" ]; then
        : "${PRIME_RL_BASELINE_REWARDS_PATH:?PRIME_RL_BASELINE_REWARDS_PATH must be set for the post-training comparison}"
    fi

    inference_args=(--model.name "$PRIME_RL_MODEL_NAME" --server.port 8000)
    lora_name=""
    if [ -n "${PRIME_RL_LORA_ADAPTER_PATH:-}" ]; then
        inference_args+=(--enable-lora)
        lora_name="post-adapter"
    fi

    log "starting inference server: uv run inference ${inference_args[*]}"
    uv run --no-sync inference "${inference_args[@]}" >"${TAU_OUTPUT_DIR}/inference.log" 2>&1 &
    CHILD_PID=$!

    wait_for_health "http://localhost:8000/health" "${PRIME_RL_HEALTH_TIMEOUT_S:-1800}"

    if [ -n "$lora_name" ]; then
        log "loading LoRA adapter ${PRIME_RL_LORA_ADAPTER_PATH} as ${lora_name}"
        curl -sf -X POST "http://localhost:8000/load_lora_adapter" \
            -H 'Content-Type: application/json' \
            -d "{\"lora_name\": \"${lora_name}\", \"lora_path\": \"${PRIME_RL_LORA_ADAPTER_PATH}\"}" \
            >/dev/null
    fi

    rewards_path="${TAU_OUTPUT_DIR}/rewards.json"
    eval_args=(
        --manifest "$PRIME_RL_MANIFEST_PATH"
        --base-url "http://localhost:8000/v1"
        --served-model-name "$PRIME_RL_MODEL_NAME"
        --label "$PRIME_RL_EVAL_LABEL"
        --output "$rewards_path"
    )
    [ -n "$lora_name" ] && eval_args+=(--lora-name "$lora_name")
    uv run --no-sync python -m tau.eval_tools.live.run_frozen_eval_live "${eval_args[@]}"

    if [ "$PRIME_RL_EVAL_LABEL" = "post" ]; then
        log "comparing baseline vs post rewards against the pass/fail gate"
        uv run --no-sync python -m tau.eval_tools.cli compare \
            --manifest "$PRIME_RL_MANIFEST_PATH" \
            --baseline "$PRIME_RL_BASELINE_REWARDS_PATH" \
            --post "$rewards_path" \
            --output "${TAU_OUTPUT_DIR}/comparison.json"
        # `compare` exits nonzero on a failed gate; with `set -e` that fails this job,
        # which is the intended, honest signal — never overridden or ignored here.
    fi
    ;;

train)
    : "${PRIME_RL_CONFIG_REL:?PRIME_RL_CONFIG_REL must be set for train mode}"
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    frozen_manifest="${PRIME_RL_MANIFEST_DIR}/frozen-eval-manifest.json"
    if [ ! -f "$frozen_manifest" ]; then
        die "frozen eval manifest not found at ${frozen_manifest} — run freeze-draft, measure the baseline eval, then freeze-finalize before training. Refusing to start training without a pre-committed held-out eval (see tau/README.md's proof ladder)."
    fi
    log "frozen eval manifest present at ${frozen_manifest} — proceeding"

    rl_args=(--output-dir "$TAU_OUTPUT_DIR")
    [ -n "${PRIME_RL_MAX_STEPS:-}" ] && rl_args+=(--max-steps "$PRIME_RL_MAX_STEPS")
    [ "${PRIME_RL_CLEAN_OUTPUT_DIR:-0}" = "1" ] && rl_args+=(--clean-output-dir)

    log "starting bounded RL training: uv run rl @ ${OVERLAY_DIR}/${PRIME_RL_CONFIG_REL} ${rl_args[*]}"
    uv run --no-sync rl @ "${OVERLAY_DIR}/${PRIME_RL_CONFIG_REL}" "${rl_args[@]}" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    ;;

*)
    die "unknown PRIME_RL_RUN_MODE=${PRIME_RL_RUN_MODE}"
    ;;
esac
