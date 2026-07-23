#!/usr/bin/env bash
set -euo pipefail

# Interleaved A/B parity benchmark: alternates two app builds round-robin
# (ABBA-style) running --paro-parity-bench, so thermal drift (map #230 trap 2)
# hits both arms equally and within-round comparison stays valid.
#
# Usage:
#   scripts/parity-ab.sh <baseline.app> <experiment.app> <model-id> [rounds] [contexts] [maxNew]
#
# Example:
#   scripts/parity-ab.sh /tmp/tesseract-baseline.app \
#     "$HOME/Library/Developer/Xcode/DerivedData/tesseract-xxx/Build/Products/Release/Tesseract Agent.app" \
#     qwen3.6-35b-a3b-paro 3 128,8192,32768
#
# Output: /tmp/parity-ab/{baseline,experiment}/round<N>/paro-parity-bench/*.json
# Compare with: scripts/parity_compare.py /tmp/parity-ab/baseline /tmp/parity-ab/experiment

BASELINE_APP="${1:?baseline .app path}"
EXPERIMENT_APP="${2:?experiment .app path}"
MODEL_ID="${3:?model id (e.g. qwen3.6-35b-a3b-paro)}"
ROUNDS="${4:-3}"
CONTEXTS="${5:-128,8192,32768}"
MAX_NEW="${6:-256}"
OUT_ROOT="/tmp/parity-ab"

if pgrep -x "Tesseract Agent" >/dev/null; then
    echo "A Tesseract Agent instance is already running — refusing to start." >&2
    exit 1
fi

# Stale reports from a previous experiment would mix into this run's verdict
# (parity_compare.py rglobs everything under OUT_ROOT) — always start clean.
if [ -e "$OUT_ROOT" ]; then
    echo "Removing stale results from a previous run: $OUT_ROOT"
    rm -rf "$OUT_ROOT"
fi

run_arm() {
    local app="$1" label="$2" round="$3"
    local outdir="$OUT_ROOT/$label/round$round"
    mkdir -p "$outdir"
    killall "Tesseract Agent" 2>/dev/null || true
    sleep 1
    echo "── round $round / $label: $(basename "$app")"
    # Optional per-arm env injection (e.g. ARM_ENV_experiment='MLX_X=1'); each
    # entry becomes one `open --env` flag. Harness-only; logged in the ledger.
    local env_var_name="ARM_ENV_${label}"
    local -a env_flags=()
    if [ -n "${!env_var_name:-}" ]; then
        local kv
        for kv in ${!env_var_name}; do
            env_flags+=(--env "$kv")
        done
    fi
    open -W "${env_flags[@]+"${env_flags[@]}"}" "$app" --args \
        --paro-parity-bench \
        --bench-model-id "$MODEL_ID" \
        --bench-output "$outdir" \
        --bench-contexts "$CONTEXTS" \
        --bench-runs "${BENCH_RUNS:-2}" \
        --bench-max-new "$MAX_NEW" &
    local open_pid=$!
    sleep 8  # allow launch before the nice check
    local app_pid
    app_pid=$(pgrep -x "Tesseract Agent" | head -1 || true)
    if [ -n "$app_pid" ]; then
        local nice
        nice=$(ps -o nice= -p "$app_pid" | tr -d ' ')
        if [ "$nice" != "0" ]; then
            echo "FATAL: app running at nice=$nice (expect 0) — map #230 trap 3." >&2
            killall "Tesseract Agent" 2>/dev/null || true
            exit 1
        fi
    fi
    # Watchdog: a bench arm occasionally completes its work but never exits
    # (observed 2026-07-23: app idle in the AppKit run loop, report unwritten)
    # and `open -W` waits forever — kill the app after ARM_TIMEOUT (default
    # 600s; a full 3-context MoE arm is ~110s) so the loop can't park.
    # The subshell is fully redirected: an orphaned watchdog `sleep` holding
    # the script's stdout pipe otherwise delays the caller's `tail` EOF by
    # up to ARM_TIMEOUT seconds per arm (observed, same day).
    (
        sleep "${ARM_TIMEOUT:-600}"
        if pgrep -x "Tesseract Agent" >/dev/null; then
            killall "Tesseract Agent" 2>/dev/null || true
        fi
    ) >/dev/null 2>&1 &
    local watchdog_pid=$!
    wait "$open_pid" 2>/dev/null || true
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    if ! ls "$outdir"/paro-parity-bench/*.json >/dev/null 2>&1; then
        echo "WARNING: no report written for $label round $round (arm hung?) — compare will treat the mismatch as FATAL." >&2
    fi
}

for round in $(seq 1 "$ROUNDS"); do
    # ABBA: alternate which arm runs first each round — the second arm is
    # thermally disadvantaged, which contaminates sub-1% verdicts (E2).
    if (( round % 2 == 1 )); then
        run_arm "$BASELINE_APP" baseline "$round"
        run_arm "$EXPERIMENT_APP" experiment "$round"
    else
        run_arm "$EXPERIMENT_APP" experiment "$round"
        run_arm "$BASELINE_APP" baseline "$round"
    fi
done

killall "Tesseract Agent" 2>/dev/null || true
echo ""
echo "Reports under $OUT_ROOT. Compare with:"
echo "  scripts/parity_compare.py $OUT_ROOT/baseline $OUT_ROOT/experiment"
