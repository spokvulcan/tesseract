#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/bench.sh [quick|full] [--model <model-id>] [extra args...]
# Builds Release (for real inference speed), then runs the benchmark headless.
# Example: scripts/bench.sh quick --model qwen3-4b-instruct-2507

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
SCHEME="tesseract"
CONFIGURATION="Release"
BENCH_DIR="/private/tmp/tesseract-debug/benchmark"
LOG_FILE="$BENCH_DIR/latest.log"
RESULTS_DIR="$BENCH_DIR/results"
REPO_RESULTS="$PROJECT_DIR/benchmarks/results"

# Build Release directly (dev.sh build doesn't forward args)
echo "Building tesseract ($CONFIGURATION)..."
# Use Debug entitlements (has /private/tmp/ exception) with Release optimizations
BUILD_OUTPUT=$(xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    SWIFT_COMPILATION_MODE=incremental \
    DEBUG_INFORMATION_FORMAT=dwarf \
    ONLY_ACTIVE_ARCH=YES \
    CODE_SIGN_ENTITLEMENTS=tesseract/tesseract.entitlements 2>&1) || {
    echo "$BUILD_OUTPUT" | tail -20
    echo "BUILD FAILED"
    exit 1
}
echo "$BUILD_OUTPUT" | grep -E "^(\*\* BUILD)" || true
echo "Build succeeded."

# Find the built app
PRODUCTS_DIR=$(xcodebuild -project "$PROJECT" -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" -showBuildSettings 2>/dev/null \
    | grep '^\s*BUILT_PRODUCTS_DIR' | awk '{print $3}')
APP="$PRODUCTS_DIR/tesseract.app"

if [ ! -d "$APP" ]; then
    echo "Error: tesseract.app not found at $APP"
    exit 1
fi

SWEEP="${1:-quick}"
shift 2>/dev/null || true

# Parse --model flag (converts to --bench-model-id for the app)
MODEL_ARGS=()
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_ARGS=(--bench-model-id "$2")
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

# Kill any existing tesseract instance
killall tesseract 2>/dev/null || true
sleep 0.5

# Clean previous log
mkdir -p "$BENCH_DIR"
rm -f "$LOG_FILE"

# Launch with -W (wait for app to exit), backgrounded so we can tail
open -W "$APP" --args --benchmark --bench-sweep "$SWEEP" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" "$@" &
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
