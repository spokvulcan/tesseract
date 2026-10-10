#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/test-stress.sh [--no-build] [--rounds N] [--whole N]
#                               [--focused N] [--slots N] [suite...]
#
# Runs the app's tests the way several agents do at once, to show that a change
# to the tests or the test host holds up under that load. Each round starts
# --whole whole-target runs and --focused runs of the given suites together,
# through scripts/test.sh, waits for all of them, and prints one line per run
# with any failures. Exits 1 if any run failed.
#
#   --rounds N   Rounds to run (default 3).
#   --whole N    Whole-target runs per round (default 4).
#   --focused N  Runs of the given suites per round (default 3 with suites,
#                0 without).
#   --slots N    Whole-target runs allowed at once (TESSERACT_TEST_SLOTS);
#                default --whole, so every whole-target run in a round overlaps.
#   --no-build   Reuse the last build-for-testing. Otherwise builds once first.
#
# Each run's output goes to the log directory printed at the start; its result
# bundle, as for any scripts/test.sh run, under DerivedData/<project>/test-runs/.
#
# Example: scripts/test-stress.sh --rounds 2 LLMGateTests AppBindingsTests
# Hunt a flaky suite: scripts/test-stress.sh --whole 0 --focused 1 --rounds 20 LLMGateTests

TEST="$(cd "$(dirname "$0")" && pwd)/test.sh"

BUILD=1
ROUNDS=3
WHOLE=4
FOCUSED=""
SLOTS=""
SUITES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --rounds | --whole | --focused | --slots)
            case "${2:-}" in ''|*[!0-9]*) echo "$1 takes a count" >&2; exit 2 ;; esac
            case "$1" in
                --rounds) ROUNDS=$2 ;;
                --whole) WHOLE=$2 ;;
                --focused) FOCUSED=$2 ;;
                --slots) SLOTS=$2 ;;
            esac
            shift ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) SUITES+=("$1") ;;
    esac
    shift
done
if [ -z "$FOCUSED" ]; then
    if [ ${#SUITES[@]} -gt 0 ]; then FOCUSED=3; else FOCUSED=0; fi
fi
if [ "$FOCUSED" -gt 0 ] && [ ${#SUITES[@]} -eq 0 ]; then
    echo "--focused runs need suites to run" >&2
    exit 2
fi
if [ "$WHOLE" -eq 0 ] && [ "$FOCUSED" -eq 0 ]; then
    echo "Nothing to run: --whole and --focused are both 0" >&2
    exit 2
fi
SLOTS="${SLOTS:-$WHOLE}"
[ "$SLOTS" -ge 1 ] || SLOTS=1

if [ "$BUILD" = 1 ]; then
    "$TEST" --build-only
fi

TMP="${TMPDIR:-/tmp}"
LOG_DIR="$(mktemp -d "${TMP%/}/tesseract-stress.XXXXXX")"
echo "$ROUNDS rounds of $WHOLE whole-target and $FOCUSED focused runs ($SLOTS slots); logs: $LOG_DIR"

RUNS=0
FAILED=0
for ((round = 1; round <= ROUNDS; round++)); do
    pids=()
    names=()
    for ((i = 1; i <= WHOLE; i++)); do
        name="round$round-whole$i"
        TESSERACT_TEST_SLOTS="$SLOTS" "$TEST" --no-build > "$LOG_DIR/$name.log" 2>&1 &
        pids+=($!)
        names+=("$name")
    done
    for ((i = 1; i <= FOCUSED; i++)); do
        name="round$round-focused$i"
        "$TEST" --no-build "${SUITES[@]}" > "$LOG_DIR/$name.log" 2>&1 &
        pids+=($!)
        names+=("$name")
    done
    for ((i = 0; i < ${#pids[@]}; i++)); do
        status=0
        wait "${pids[$i]}" || status=$?
        RUNS=$((RUNS + 1))
        name="${names[$i]}"
        totals="$(grep -E '^[0-9]+ tests:' "$LOG_DIR/$name.log" | tail -1 || true)"
        if [ "$status" = 0 ]; then
            echo "$name: $totals"
        else
            FAILED=$((FAILED + 1))
            echo "$name: FAILED (exit $status) $totals"
            grep -A4 '^FAILED' "$LOG_DIR/$name.log" | cut -c1-240 || true
        fi
    done
done

echo "$RUNS runs, $FAILED failed"
[ "$FAILED" -eq 0 ]
