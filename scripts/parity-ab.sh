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

run_arm() {
    local app="$1" label="$2" round="$3"
    local outdir="$OUT_ROOT/$label/round$round"
    mkdir -p "$outdir"
    killall "Tesseract Agent" 2>/dev/null || true
    sleep 1
    echo "── round $round / $label: $(basename "$app")"
    open -W "$app" --args \
        --paro-parity-bench \
        --bench-model-id "$MODEL_ID" \
        --bench-output "$outdir" \
        --bench-contexts "$CONTEXTS" \
        --bench-runs 2 \
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
    wait "$open_pid" 2>/dev/null || true
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
