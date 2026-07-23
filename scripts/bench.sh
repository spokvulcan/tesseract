#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/bench.sh [quick|full] [--model <model-id>] [--prompt <benchmark|production>] [extra args...]
# Builds Release (for real inference speed), then runs the benchmark headless.
# Example: scripts/bench.sh quick --model qwen3.5-4b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
SCHEME="tesseract"
CONFIGURATION="Release"
DERIVED_DATA_GLOB="$HOME/Library/Developer/Xcode/DerivedData/tesseract-*"
# The app is not sandboxed (ADR-0047), so FileManager.temporaryDirectory is
# the real per-user temp dir, not the container tmp.
BENCH_DIR="$(getconf DARWIN_USER_TEMP_DIR)tesseract-debug/benchmark"
LOG_FILE="$BENCH_DIR/latest.log"
# The parity harness keeps its own log subdirectory; bench.sh must watch that
# one when driven with --paro-parity-bench, otherwise it times out at 30s and
# orphans a still-running benchmark process (map #230 trap 1).
for arg in "$@"; do
    if [ "$arg" = "--paro-parity-bench" ]; then
        LOG_FILE="$BENCH_DIR/paro-parity-bench/latest.log"
        break
    elif [ "$arg" = "--snapshot-bench" ]; then
        LOG_FILE="$BENCH_DIR/snapshot-bench/latest.log"
        break
    elif [ "$arg" = "--prefix-detect-bench" ]; then
        LOG_FILE="$BENCH_DIR/prefix-detect-bench/latest.log"
        break
    elif [ "$arg" = "--prefix-cache-e2e" ]; then
        LOG_FILE="$BENCH_DIR/prefix-cache-e2e/latest.log"
        break
    fi
done
RESULTS_DIR="$BENCH_DIR/results"
REPO_RESULTS="$PROJECT_DIR/benchmarks/results"

# Clean stale module cache on config switch to avoid "Unable to find module dependency" errors
DERIVED_DATA=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
if [ -n "$DERIVED_DATA" ]; then
    LAST_CONFIG=""
    [ -f "$DERIVED_DATA/.last_config" ] && LAST_CONFIG=$(cat "$DERIVED_DATA/.last_config")
    if [ -n "$LAST_CONFIG" ] && [ "$LAST_CONFIG" != "$CONFIGURATION" ]; then
        echo "Configuration changed ($LAST_CONFIG → $CONFIGURATION), cleaning module cache..."
        rm -rf "$DERIVED_DATA/Build/Intermediates.noindex/SwiftExplicitPrecompiledModules" 2>/dev/null
    fi
fi

# Build Release directly (dev.sh build doesn't forward args)
echo "Building tesseract ($CONFIGURATION)..."
BUILD_OUTPUT=$(xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination 'platform=macOS' \
    -skipPackagePluginValidation \
    2>&1) || {
    # Auto-recover from stale module cache
    if echo "$BUILD_OUTPUT" | grep -q "Unable to find module dependency"; then
        echo "Detected stale module cache. Cleaning and retrying..."
        DERIVED_DATA=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
        [ -n "$DERIVED_DATA" ] && rm -rf "$DERIVED_DATA/Build/Intermediates.noindex/SwiftExplicitPrecompiledModules" 2>/dev/null
        xcodebuild -resolvePackageDependencies -project "$PROJECT" -scheme "$SCHEME" 2>&1 | grep -E "^(error:|Resolved)" || true
        BUILD_OUTPUT=$(xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
            -configuration "$CONFIGURATION" \
            -destination 'platform=macOS' \
            -skipPackagePluginValidation \
            2>&1) || {
            echo "$BUILD_OUTPUT" | tail -20
            echo "BUILD FAILED after retry"
            exit 1
        }
    else
        echo "$BUILD_OUTPUT" | tail -20
        echo "BUILD FAILED"
        exit 1
    fi
}
echo "$BUILD_OUTPUT" | grep -E "^(\*\* BUILD)" || true
# Stamp the configuration
DERIVED_DATA=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
[ -n "$DERIVED_DATA" ] && echo "$CONFIGURATION" > "$DERIVED_DATA/.last_config"
echo "Build succeeded."

# Resolve the products dir from xcodebuild, not an mtime-sorted glob —
# stale sibling DerivedData dirs picked hours-old binaries (#165).
PRODUCTS_DIR=$(xcodebuild -project "$PROJECT" -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" -destination 'platform=macOS' \
    -skipPackagePluginValidation -showBuildSettings 2>/dev/null \
    | awk -F' = ' '/[[:space:]]BUILT_PRODUCTS_DIR =/{print $2; exit}')
APP="$PRODUCTS_DIR/Tesseract Agent.app"

if [ -z "$PRODUCTS_DIR" ] || [ ! -d "$APP" ]; then
    echo "Error: 'Tesseract Agent.app' ($CONFIGURATION) not found at BUILT_PRODUCTS_DIR ('${PRODUCTS_DIR:-unresolved}')."
    exit 1
fi

SWEEP="${1:-quick}"
shift 2>/dev/null || true

# Parse --model flag (converts to --bench-model-id for the app)
MODEL_ARGS=()
PROMPT_ARGS=()
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_ARGS=(--bench-model-id "$2")
            shift 2
            ;;
        --prompt)
            PROMPT_ARGS=(--bench-prompt "$2")
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"

echo "Running benchmark (sweep: $SWEEP)..."
echo "App: $APP"

# Kill any existing tesseract instance (process is named "Tesseract Agent" after the rename)
killall "Tesseract Agent" 2>/dev/null || true
killall tesseract 2>/dev/null || true
sleep 0.5

# Clean previous log
mkdir -p "$BENCH_DIR"
rm -f "$LOG_FILE"

# Launch with -W (wait for app to exit), backgrounded so we can tail
open -W "$APP" --args --benchmark --bench-sweep "$SWEEP" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" "${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}" "$@" &
OPEN_PID=$!

# Wait for log file to appear (up to 30s for model loading)
echo "Waiting for benchmark to start..."
for i in $(seq 1 60); do
    if [ -f "$LOG_FILE" ]; then
        break
    fi
    sleep 0.5
done

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not created after 30s. Check Console.app for errors."
    echo "Expected: $LOG_FILE"
    exit 1
fi

# Tail log in background
echo "Tailing benchmark log..."
echo "───────────────────────────────────────"
tail -f "$LOG_FILE" &
TAIL_PID=$!

# Wait for the app to exit
wait $OPEN_PID 2>/dev/null || true

# Stop tailing
kill $TAIL_PID 2>/dev/null || true
echo ""
echo "───────────────────────────────────────"

# Copy results to repo
if [ -d "$RESULTS_DIR" ]; then
    mkdir -p "$REPO_RESULTS"
    NEW_FILES=$(find "$RESULTS_DIR" -name "bench_*.json" -newer "$REPO_RESULTS/.gitkeep" 2>/dev/null || find "$RESULTS_DIR" -name "bench_*.json" 2>/dev/null)
    if [ -n "$NEW_FILES" ]; then
        cp "$RESULTS_DIR"/bench_*.json "$REPO_RESULTS/"
        echo "Results copied to benchmarks/results/"
        ls -la "$REPO_RESULTS"/bench_*.json 2>/dev/null | tail -5
    else
        echo "No new result files to copy."
    fi
else
    echo "No results directory found at $RESULTS_DIR"
fi

# Copy transcripts to repo
TRANSCRIPTS_DIR="$BENCH_DIR/transcripts"
REPO_TRANSCRIPTS="$PROJECT_DIR/benchmarks/transcripts"
if [ -d "$TRANSCRIPTS_DIR" ]; then
    mkdir -p "$REPO_TRANSCRIPTS"
    cp "$TRANSCRIPTS_DIR"/*.transcript.txt "$REPO_TRANSCRIPTS/" 2>/dev/null && \
        echo "Transcripts copied to benchmarks/transcripts/" || true
fi
