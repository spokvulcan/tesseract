#!/usr/bin/env bash
# Usage: scripts/gate.sh [--no-build] <run-folder> <harness-flag> [harness arguments...]
#
# Runs one loaded-model gate (a headless harness: --prefix-cache-e2e,
# --hybrid-cache-correctness, ...) from this checkout's Debug build and keeps
# everything the run wrote in <run-folder>, which must not exist yet:
#
#   bench/<harness>/latest.log   the harness's report, its check lines
#   telemetry/                   the run's own CacheDiagnostics and
#                                PrefixCacheTraces (TESSERACT_TELEMETRY_DIR)
#   run.log                      the binary's output
#   meta.txt                     the commit, the arguments, the exit code
#
# Two runs compare with scripts/gate-compare.py <before> <after>: the check
# lines, the decode lines and the diagnostics masked as ADR-0078 masks them.
# docs/prefix-cache-testing.md → "Prove a refactor" lists the gate set.
#
# It builds Debug first (scripts/dev.sh build) unless --no-build. A running
# Tesseract Agent run (a test host, another gate) is waited for, never killed;
# an open app (the owner's, a scratch launch) fails the gate: quit it first.
#
# Example:
#   scripts/gate.sh "$TMPDIR/gates/after-e2e-27b" --prefix-cache-e2e --bench-model-id qwen3.8-27b
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
. "$PROJECT_DIR/scripts/lib/checkout.sh"
. "$PROJECT_DIR/scripts/lib/app.sh"

BUILD=1
if [ "${1:-}" = "--no-build" ]; then
    BUILD=0
    shift
fi
if [ $# -lt 2 ]; then
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 2
fi
RUN="$1"
FLAG="$2"
shift 2
for arg in "$@"; do
    if [ "$arg" = "--bench-output" ]; then
        echo "The gate owns --bench-output: the report goes to $RUN/bench." >&2
        exit 2
    fi
done
if [ -e "$RUN" ]; then
    echo "$RUN exists: a gate keeps each run's capture, so name a new folder." >&2
    exit 2
fi

if [ "$BUILD" = 1 ]; then
    "$PROJECT_DIR/scripts/dev.sh" build debug
fi
APP="$(built_products_dir "$PROJECT" tesseract Debug)/Tesseract Agent.app"
if [ ! -x "$APP/Contents/MacOS/Tesseract Agent" ]; then
    echo "No Debug build at $APP: run without --no-build." >&2
    exit 1
fi

wait_for_quiet_app 600

mkdir -p "$RUN/bench" "$RUN/telemetry"
{
    echo "commit $(git -C "$PROJECT_DIR" rev-parse HEAD)$( [ -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ] || echo ' (with uncommitted changes)')"
    echo "arguments $FLAG $*"
    echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN/meta.txt"

echo "Running $FLAG $* → $RUN"
STATUS=0
TESSERACT_TELEMETRY_DIR="$RUN/telemetry" \
    "$APP/Contents/MacOS/Tesseract Agent" "$FLAG" --bench-output "$RUN/bench" "$@" \
    > "$RUN/run.log" 2>&1 || STATUS=$?
{
    echo "ended $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit $STATUS"
} >> "$RUN/meta.txt"

REPORT="$(find "$RUN/bench" -name latest.log -print -quit)"
if [ -n "$REPORT" ]; then
    echo "Report: $REPORT"
    grep -E '✅|❌|Overall|^\[harness\]' "$REPORT" | sed -E 's/^\[[^]]*\] *//' | tail -50 || true
else
    echo "No report: the run ended before writing one. See $RUN/run.log" >&2
    tail -20 "$RUN/run.log" >&2 || true
fi
exit "$STATUS"
