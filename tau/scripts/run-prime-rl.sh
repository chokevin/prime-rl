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
#   freeze-finalize- validate fixed baseline rewards and freeze the immutable manifest.
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
: "${PRIME_RL_VERIFIERS_SHA:?PRIME_RL_VERIFIERS_SHA must pin deps/verifiers}"
: "${PRIME_RL_TASKSETS_SHA:?PRIME_RL_TASKSETS_SHA must pin deps/research-environments}"

[[ "$PRIME_RL_REPO_SHA" =~ ^[0-9a-f]{40}$ ]] || die "PRIME_RL_REPO_SHA must be a full lowercase 40-character commit SHA"
[[ "$PRIME_RL_VERIFIERS_SHA" =~ ^[0-9a-f]{40}$ ]] || die "PRIME_RL_VERIFIERS_SHA must be a full lowercase 40-character commit SHA"
[[ "$PRIME_RL_TASKSETS_SHA" =~ ^[0-9a-f]{40}$ ]] || die "PRIME_RL_TASKSETS_SHA must be a full lowercase 40-character commit SHA"

CHILD_PID=""
TMP_ROOT="$(realpath -e /tmp)"
[ -d "$TMP_ROOT" ] || die "canonical /tmp root is unavailable"
overlay_raw="$(mktemp -d "${TMP_ROOT}/prime-rl-overlay.XXXXXX")"
OVERLAY_DIR="$(realpath -e "$overlay_raw")"
case "$OVERLAY_DIR" in
"${TMP_ROOT}"/prime-rl-overlay.*) ;;
*) die "mktemp returned unsafe overlay path ${OVERLAY_DIR}" ;;
esac
[ "$OVERLAY_DIR" != "$TMP_ROOT" ] || die "overlay must be a private child, never /tmp itself"
[ "$(stat -c %d "$OVERLAY_DIR")" = "$(stat -c %d "$TMP_ROOT")" ] || die "overlay must remain on the /tmp filesystem"
OVERLAY_DEVICE="$(stat -c %d "$OVERLAY_DIR")"
OVERLAY_INODE="$(stat -c %i "$OVERLAY_DIR")"
run_root_raw="$(mktemp -d "${TMP_ROOT}/prime-rl-run-XXXXXXXX")"
RUN_ROOT="$(realpath -e "$run_root_raw")"
case "$RUN_ROOT" in
"${TMP_ROOT}"/prime-rl-run-*) ;;
*) die "mktemp returned unsafe private run root ${RUN_ROOT}" ;;
esac
[ ! -L "$RUN_ROOT" ] || die "private run root must not be a symlink"
chmod 0700 "$RUN_ROOT"
mkdir "${RUN_ROOT}/hf-home"
chmod 0700 "${RUN_ROOT}/hf-home"
RUN_ROOT_DEVICE="$(stat -c %d "$RUN_ROOT")"
RUN_ROOT_INODE="$(stat -c %i "$RUN_ROOT")"
[ "$RUN_ROOT_DEVICE" = "$(stat -c %d "$TMP_ROOT")" ] || die "private run root must remain on /tmp"
if command -v findmnt >/dev/null 2>&1; then
    [ "$(findmnt -n -o TARGET -T "$OVERLAY_DIR")" = "$(findmnt -n -o TARGET -T "$TMP_ROOT")" ] ||
        die "overlay must not be a nested mount"
    [ "$(findmnt -n -o TARGET -T "$RUN_ROOT")" = "$(findmnt -n -o TARGET -T "$TMP_ROOT")" ] ||
        die "private run root must not be a nested mount"
fi

cleanup() {
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        log "stopping child process (pid ${CHILD_PID})"
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    canonical_overlay="$(realpath -e "$OVERLAY_DIR" 2>/dev/null || true)"
    case "$canonical_overlay" in
    "${TMP_ROOT}"/prime-rl-overlay.*)
        if [ "$canonical_overlay" = "$OVERLAY_DIR" ] &&
            [ ! -L "$OVERLAY_DIR" ] &&
            [ "$(stat -c %d "$canonical_overlay")" = "$OVERLAY_DEVICE" ] &&
            [ "$(stat -c %i "$canonical_overlay")" = "$OVERLAY_INODE" ]; then
            rm -rf -- "$canonical_overlay"
        else
            log "refusing to remove unsafe overlay path ${canonical_overlay}"
        fi
        ;;
    *) log "refusing to remove unsafe overlay path ${OVERLAY_DIR}" ;;
    esac
    canonical_run_root="$(realpath -e "$RUN_ROOT" 2>/dev/null || true)"
    case "$canonical_run_root" in
    "${TMP_ROOT}"/prime-rl-run-*)
        if [ "$canonical_run_root" = "$RUN_ROOT" ] &&
            [ ! -L "$RUN_ROOT" ] &&
            [ "$(stat -c %d "$canonical_run_root")" = "$RUN_ROOT_DEVICE" ] &&
            [ "$(stat -c %i "$canonical_run_root")" = "$RUN_ROOT_INODE" ]; then
            chmod -R u+w -- "$canonical_run_root"
            rm -rf -- "$canonical_run_root"
        else
            log "refusing to remove unsafe private run root ${canonical_run_root}"
        fi
        ;;
    *) log "refusing to remove unsafe private run root ${RUN_ROOT}" ;;
    esac
}

forward_signal_and_exit() {
    local signal_name="$1" exit_status="$2"
    trap - INT TERM
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        log "forwarding ${signal_name} to child process (pid ${CHILD_PID})"
        kill "-${signal_name}" "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    CHILD_PID=""
    exit "$exit_status"
}

trap cleanup EXIT
trap 'forward_signal_and_exit INT 130' INT
trap 'forward_signal_and_exit TERM 143' TERM

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
git init --quiet "$OVERLAY_DIR"
git -C "$OVERLAY_DIR" remote add origin "$PRIME_RL_REPO_URL"
git -C "$OVERLAY_DIR" fetch --quiet --depth 1 origin "$PRIME_RL_REPO_SHA"
git -C "$OVERLAY_DIR" checkout --quiet --force FETCH_HEAD
resolved_sha="$(git -C "$OVERLAY_DIR" rev-parse HEAD)"
[ "$resolved_sha" = "$PRIME_RL_REPO_SHA" ] || die "overlay checkout resolved to ${resolved_sha}, expected ${PRIME_RL_REPO_SHA} — aborting rather than run an unverified tree"
log "overlay verified at ${resolved_sha}"

verify_gitlink() {
    local path="$1" expected="$2" entry mode type actual
    entry="$(git -C "$OVERLAY_DIR" ls-tree HEAD -- "$path")"
    read -r mode type actual _ <<<"$entry"
    [ "$mode" = "160000" ] && [ "$type" = "commit" ] || die "${path} is not a pinned gitlink at ${resolved_sha}"
    [ "$actual" = "$expected" ] || die "${path} is pinned to ${actual}, expected ${expected}"
}
verify_gitlink deps/verifiers "$PRIME_RL_VERIFIERS_SHA"
verify_gitlink deps/research-environments "$PRIME_RL_TASKSETS_SHA"

canonical_overlay_path() {
    local relative="$1" expected_kind="$2" current="$OVERLAY_DIR" component canonical
    case "$relative" in
    /* | *"//"* | "." | ".." | ../* | */../* | */.. | ./* | */./* | */.) die "unsafe overlay-relative path: ${relative}" ;;
    esac
    IFS=/ read -ra components <<<"$relative"
    for component in "${components[@]}"; do
        current="${current}/${component}"
        [ ! -L "$current" ] || die "overlay path contains symlink component: ${current}"
    done
    canonical="$(realpath -e "$current")" || die "overlay path does not exist: ${relative}"
    case "$canonical" in
    "${OVERLAY_DIR}"/*) ;;
    *) die "overlay path escapes verified checkout: ${relative} -> ${canonical}" ;;
    esac
    case "$expected_kind" in
    file) [ -f "$canonical" ] || die "overlay path is not a regular file: ${relative}" ;;
    dir) [ -d "$canonical" ] || die "overlay path is not a directory: ${relative}" ;;
    *) die "internal error: unsupported overlay path kind ${expected_kind}" ;;
    esac
    printf '%s\n' "$canonical"
}

primary_config_path() {
    [ "${PRIME_RL_CONFIG_REL:-}" = "configs/tau/math-7b-h200/train.toml" ] ||
        die "PRIME_RL_CONFIG_REL must be exactly configs/tau/math-7b-h200/train.toml"
    canonical_overlay_path "$PRIME_RL_CONFIG_REL" file
}

# Optional narrow install of one repo-local environment package — forward-compatible
# with a future harder-math environment (W4a/W9), not used by the primary
# math-env-v1/math500-v1 experiment, which ships in deps/research-environments already
# baked into the image. --no-deps + a single `-e` target keeps this narrow; never a
# full/`--inexact` `uv sync` of the whole workspace.
if [ -n "${PRIME_RL_EXTRA_ENV_PACKAGE_DIR:-}" ]; then
    [ "$PRIME_RL_EXTRA_ENV_PACKAGE_DIR" = "environments/harder_math_v1" ] ||
        die "PRIME_RL_EXTRA_ENV_PACKAGE_DIR may only select environments/harder_math_v1"
    pkg_dir="$(canonical_overlay_path "$PRIME_RL_EXTRA_ENV_PACKAGE_DIR" dir)"
    log "installing extra environment package from ${pkg_dir} (narrow: --no-deps, no full sync)"
    cd /app
    uv pip install --no-deps -e "$pkg_dir"
fi

cd /app
export PYTHONPATH="$OVERLAY_DIR"
export HF_HOME="${RUN_ROOT}/hf-home"

uv run --no-sync python -m tau.eval_tools.output_paths \
    --mode "$PRIME_RL_RUN_MODE" \
    --output-dir "$TAU_OUTPUT_DIR" \
    --eval-label "${PRIME_RL_EVAL_LABEL:-}"
log "prepared exact mode-bound output directory ${TAU_OUTPUT_DIR}"

# --- Step 2: child-process lifecycle helpers -----------------------------------------
# Backgrounding the long-running child + trapping here (rather than running it in the
# foreground) is required, not decorative: bash defers a foreground command's signal
# delivery until that command exits, so a SIGTERM sent to this script (Tau/Kubernetes
# cancellation, or a resilience-triggered retry) would otherwise never reach us until
# the child already finished on its own.
wait_for_health() {
    local url="$1" timeout_s="${2:-1800}" waited=0
    until curl -sf -o /dev/null "$url"; do
        if [ -n "$CHILD_PID" ] && ! kill -0 "$CHILD_PID" 2>/dev/null; then
            if wait "$CHILD_PID"; then
                child_status=0
            else
                child_status=$?
            fi
            CHILD_PID=""
            die "inference server exited before readiness (status ${child_status}; ${url})"
        fi
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
    config_path="$(primary_config_path)"
    [ ! -e "${TAU_OUTPUT_DIR}/smoke-result.json" ] || die "${TAU_OUTPUT_DIR}/smoke-result.json already exists; smoke evidence is immutable"
    log "config smoke: uv run rl --dry-run @ ${config_path}"
    uv run --no-sync rl @ "$config_path" \
        --output-dir "$TAU_OUTPUT_DIR" --dry-run
    uv run --no-sync python -m tau.eval_tools.cli write-smoke-result \
        --output "${TAU_OUTPUT_DIR}/smoke-result.json" \
        --source-revision "$resolved_sha" \
        --config-path "$PRIME_RL_CONFIG_REL" \
        --configs-dir "${TAU_OUTPUT_DIR}/configs"
    log "smoke OK: resolved per-process TOMLs written under ${TAU_OUTPUT_DIR}/configs/"
    ;;

freeze-draft)
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MODEL_REVISION:?PRIME_RL_MODEL_REVISION must be an exact HF commit SHA}"
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    : "${PRIME_RL_TRAIN_DATASET_REVISION:?PRIME_RL_TRAIN_DATASET_REVISION must be an exact HF dataset commit SHA}"
    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live draft \
        --model-name "$PRIME_RL_MODEL_NAME" \
        --model-revision "$PRIME_RL_MODEL_REVISION" \
        --run-root "$RUN_ROOT" \
        --train-dataset-revision "$PRIME_RL_TRAIN_DATASET_REVISION" \
        --source-revision "$resolved_sha" \
        --verifiers-revision "$PRIME_RL_VERIFIERS_SHA" \
        --tasksets-revision "$PRIME_RL_TASKSETS_SHA" \
        --out "${PRIME_RL_MANIFEST_DIR}/draft-manifest.json" \
        --temperature "${PRIME_RL_EVAL_TEMPERATURE:-0.0}" \
        --seed "${PRIME_RL_EVAL_SEED:-0}"
    ;;

freeze-finalize)
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MODEL_REVISION:?PRIME_RL_MODEL_REVISION must be an exact HF commit SHA}"
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    : "${PRIME_RL_BASELINE_REWARDS_PATH:?PRIME_RL_BASELINE_REWARDS_PATH must point to immutable baseline rewards.json}"
    : "${PRIME_RL_CONFIG_REL:?PRIME_RL_CONFIG_REL must be set}"
    : "${PRIME_RL_MAX_STEPS:?PRIME_RL_MAX_STEPS must select the exact bounded final training step}"
    [[ "$PRIME_RL_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || die "PRIME_RL_MAX_STEPS must be a positive integer"
    config_path="$(primary_config_path)"
    uv run --no-sync python -m tau.eval_tools.cli validate-manifest \
        --manifest "${PRIME_RL_MANIFEST_DIR}/draft-manifest.json" \
        --source-revision "$resolved_sha" \
        --verifiers-revision "$PRIME_RL_VERIFIERS_SHA" \
        --tasksets-revision "$PRIME_RL_TASKSETS_SHA" \
        --model-name "$PRIME_RL_MODEL_NAME" \
        --model-revision "$PRIME_RL_MODEL_REVISION" \
        >/dev/null
    uv run --no-sync python -m tau.eval_tools.live.freeze_manifest_live finalize \
        --draft "${PRIME_RL_MANIFEST_DIR}/draft-manifest.json" \
        --baseline-rewards "$PRIME_RL_BASELINE_REWARDS_PATH" \
        --config "$config_path" \
        --config-rel "$PRIME_RL_CONFIG_REL" \
        --output-dir "/data/pretraining-data/prime-rl-math-7b-h200/train" \
        --max-steps "$PRIME_RL_MAX_STEPS" \
        --out "${PRIME_RL_MANIFEST_DIR}/frozen-eval-manifest.json"
    ;;

eval)
    : "${PRIME_RL_EVAL_LABEL:?PRIME_RL_EVAL_LABEL must be baseline or post}"
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MODEL_REVISION:?PRIME_RL_MODEL_REVISION must be an exact HF commit SHA}"
    : "${PRIME_RL_MANIFEST_PATH:?PRIME_RL_MANIFEST_PATH must be set (draft for the very first baseline, frozen thereafter)}"
    case "$PRIME_RL_EVAL_LABEL" in
    baseline | post) ;;
    *) die "PRIME_RL_EVAL_LABEL must be baseline or post" ;;
    esac
    manifest_args=(
        --manifest "$PRIME_RL_MANIFEST_PATH"
        --source-revision "$resolved_sha"
        --verifiers-revision "$PRIME_RL_VERIFIERS_SHA"
        --tasksets-revision "$PRIME_RL_TASKSETS_SHA"
        --model-name "$PRIME_RL_MODEL_NAME"
        --model-revision "$PRIME_RL_MODEL_REVISION"
    )
    if [ "$PRIME_RL_EVAL_LABEL" = "post" ]; then
        : "${PRIME_RL_BASELINE_REWARDS_PATH:?PRIME_RL_BASELINE_REWARDS_PATH must be set for the post-training comparison}"
        : "${PRIME_RL_LORA_ADAPTER_PATH:?PRIME_RL_LORA_ADAPTER_PATH must be set for post eval}"
        : "${PRIME_RL_TRAINING_RESULT_PATH:?PRIME_RL_TRAINING_RESULT_PATH must be set for post eval}"
        : "${PRIME_RL_TRAINING_OUTPUT_DIR:?PRIME_RL_TRAINING_OUTPUT_DIR must be set for post eval}"
        manifest_args+=(--require-finalized)
    fi
    uv run --no-sync python -m tau.eval_tools.cli validate-manifest "${manifest_args[@]}" >/dev/null
    uv run --no-sync python -m tau.eval_tools.live.private_materialization_live \
        --manifest "$PRIME_RL_MANIFEST_PATH" \
        --run-root "$RUN_ROOT" \
        >/dev/null
    model_snapshot_path="${RUN_ROOT}/model"

    # A standalone eval has one engine and needs no routing layer. Running the bare
    # engine keeps health, admin, and OpenAI requests on the same source-supported port.
    inference_args=(--model.name "$model_snapshot_path" --server.port 8000 --router None)
    lora_name=""
    if [ "$PRIME_RL_EVAL_LABEL" = "post" ]; then
        uv run --no-sync python -m tau.eval_tools.cli validate-adapter-handoff \
            --manifest "$PRIME_RL_MANIFEST_PATH" \
            --training-result "$PRIME_RL_TRAINING_RESULT_PATH" \
            --training-output-dir "$PRIME_RL_TRAINING_OUTPUT_DIR" \
            --adapter-path "$PRIME_RL_LORA_ADAPTER_PATH" \
            --private-run-root "$RUN_ROOT"
        inference_args+=(--enable-lora --max-lora-rank 16)
        lora_name="post-adapter"
    fi
    chmod 0555 "$RUN_ROOT"

    rewards_path="${TAU_OUTPUT_DIR}/rewards.json"
    if [ "${PRIME_RL_COMPARE_ONLY:-0}" != "1" ]; then
        [ ! -e "$rewards_path" ] || die "${rewards_path} already exists; reward evidence is immutable"
        [ ! -e "${TAU_OUTPUT_DIR}/inference.log" ] || die "${TAU_OUTPUT_DIR}/inference.log already exists"
        log "starting inference server: uv run inference ${inference_args[*]}"
        uv run --no-sync inference "${inference_args[@]}" >"${TAU_OUTPUT_DIR}/inference.log" 2>&1 &
        CHILD_PID=$!

        wait_for_health "http://localhost:8000/health" "${PRIME_RL_HEALTH_TIMEOUT_S:-1800}"

        if [ -n "$lora_name" ]; then
            private_adapter_path="${RUN_ROOT}/adapter"
            log "loading privately materialized LoRA adapter ${private_adapter_path} as ${lora_name}"
            curl -fsS -X POST "http://localhost:8000/load_lora_adapter" \
                -H 'Content-Type: application/json' \
                -d "{\"lora_name\": \"${lora_name}\", \"lora_path\": \"${private_adapter_path}\"}" \
                >/dev/null
            curl -fsS -o /dev/null "http://localhost:8000/health"
        fi

        eval_args=(
            --manifest "$PRIME_RL_MANIFEST_PATH"
            --base-url "http://localhost:8000/v1"
            --served-model-name "$model_snapshot_path"
            --label "$PRIME_RL_EVAL_LABEL"
            --output "$rewards_path"
        )
        [ -n "$lora_name" ] && eval_args+=(--lora-name "$lora_name")
        uv run --no-sync python -m tau.eval_tools.live.run_frozen_eval_live "${eval_args[@]}"
    else
        [ "$PRIME_RL_EVAL_LABEL" = "post" ] || die "PRIME_RL_COMPARE_ONLY=1 is valid only for post eval"
        [ -f "$rewards_path" ] || die "comparison-only retry requires existing immutable post rewards at ${rewards_path}"
    fi

    if [ "$PRIME_RL_EVAL_LABEL" = "post" ]; then
        comparison_path="${PRIME_RL_COMPARISON_OUTPUT_PATH:-${TAU_OUTPUT_DIR}/comparison.json}"
        case "$comparison_path" in
        "${TAU_OUTPUT_DIR}"/*) ;;
        *) die "PRIME_RL_COMPARISON_OUTPUT_PATH must be a named file under ${TAU_OUTPUT_DIR}" ;;
        esac
        log "comparing baseline vs post rewards against the pass/fail gate"
        uv run --no-sync python -m tau.eval_tools.cli compare \
            --manifest "$PRIME_RL_MANIFEST_PATH" \
            --baseline "$PRIME_RL_BASELINE_REWARDS_PATH" \
            --post "$rewards_path" \
            --output "$comparison_path"
        # `compare` exits nonzero on a failed gate; with `set -e` that fails this job,
        # which is the intended, honest signal — never overridden or ignored here.
    fi
    ;;

train)
    : "${PRIME_RL_CONFIG_REL:?PRIME_RL_CONFIG_REL must be set for train mode}"
    : "${PRIME_RL_MANIFEST_DIR:?PRIME_RL_MANIFEST_DIR must be set}"
    : "${PRIME_RL_MODEL_NAME:?PRIME_RL_MODEL_NAME must be set}"
    : "${PRIME_RL_MODEL_REVISION:?PRIME_RL_MODEL_REVISION must be an exact HF commit SHA}"
    config_path="$(primary_config_path)"
    frozen_manifest="${PRIME_RL_MANIFEST_DIR}/frozen-eval-manifest.json"
    if [ ! -f "$frozen_manifest" ]; then
        die "frozen eval manifest not found at ${frozen_manifest} — run freeze-draft, measure the baseline eval, then freeze-finalize before training. Refusing to start training without a pre-committed held-out eval (see tau/README.md's proof ladder)."
    fi
    uv run --no-sync python -m tau.eval_tools.cli validate-manifest \
        --manifest "$frozen_manifest" \
        --source-revision "$resolved_sha" \
        --verifiers-revision "$PRIME_RL_VERIFIERS_SHA" \
        --tasksets-revision "$PRIME_RL_TASKSETS_SHA" \
        --model-name "$PRIME_RL_MODEL_NAME" \
        --model-revision "$PRIME_RL_MODEL_REVISION" \
        --require-finalized \
        >/dev/null
    log "starting trusted bounded-RL supervisor; it logs the durable attempt ID before materialization"
    uv run --no-sync python -m tau.eval_tools.live.training_supervisor_live run \
        --manifest "$frozen_manifest" \
        --config "$config_path" \
        --output-dir "$TAU_OUTPUT_DIR" &
    CHILD_PID=$!
    if wait "$CHILD_PID"; then
        supervisor_status=0
    else
        supervisor_status=$?
    fi
    CHILD_PID=""
    if [ "$supervisor_status" -ne 0 ]; then
        log "training supervisor failed with status ${supervisor_status}"
        exit "$supervisor_status"
    fi
    ;;

*)
    die "unknown PRIME_RL_RUN_MODE=${PRIME_RL_RUN_MODE}"
    ;;
esac
