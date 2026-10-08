#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/vendor-test.sh [--no-build] [--docs] [suite...]
#
# Runs the vendor fork's MLXLMTests (Vendor/mlx-swift-lm) the way that
# finishes: build-for-testing once into the fork's DerivedData, then
# test-without-building serialized with a per-test time allowance. Xcode's
# parallel runner hangs on this GPU-heavy target. Prints the failures and the
# totals; the full logs stay under DerivedData/vendor-test/.
#
#   suite       An XCTest class or Swift Testing suite, or Suite/test. A
#               Swift Testing free function needs its parentheses:
#               'testName()'. No suite runs all of MLXLMTests.
#   --no-build  Reuse the last build-for-testing (iterate on one suite).
#   --docs      DocC with warnings as errors for the engine's targets.
#
# Example: scripts/vendor-test.sh DFlash2Tests Qwen35VisionEngineTests

VENDOR_DIR="$(cd "$(dirname "$0")/../Vendor/mlx-swift-lm" && pwd)"
DERIVED_DATA="$VENDOR_DIR/DerivedData"
LOG_DIR="$DERIVED_DATA/vendor-test"
XCODEBUILD=(xcodebuild -scheme mlx-swift-lm-Package -destination 'platform=macOS'
    -skipPackagePluginValidation -derivedDataPath "$DERIVED_DATA")
# DocC fails before conversion for MLXGuidedGeneration and MLXHuggingFace:
# under Xcode 27, clang -extract-api parses MLXCXGrammar's C++ headers as
# Objective-C. The fork's scripts/verify-docs.sh checks every library target.
DOC_TARGETS=(MLXLMCommon MLXLLM MLXVLM MLXEmbedders MLXRerankers)

SKIP_BUILD=0
DOCS=0
ONLY_TESTING=()
for arg in "$@"; do
    case "$arg" in
        --no-build) SKIP_BUILD=1 ;;
        --docs) DOCS=1 ;;
        -*) echo "Unknown option: $arg" >&2; exit 2 ;;
        MLXLMTests|MLXLMTests/*) ONLY_TESTING+=("-only-testing:$arg") ;;
        *) ONLY_TESTING+=("-only-testing:MLXLMTests/$arg") ;;
    esac
done
[ ${#ONLY_TESTING[@]} -gt 0 ] || ONLY_TESTING=(-only-testing:MLXLMTests)

# Failures first, then the totals the runner prints last.
summarize() {
    grep -E ': error: |^error: |Test Case .* failed|^✘|Restarting after|exceeded' "$1" \
        | cut -c1-300 | awk '!seen[$0]++' | head -40 || true
    grep -E 'Executed [0-9]+ tests|Test run with|\*\* [A-Z -]+ \*\*' "$1" \
        | cut -c1-300 | awk '!seen[$0]++' | tail -4 || true
}

mkdir -p "$LOG_DIR"
cd "$VENDOR_DIR"

if [ "$SKIP_BUILD" = 0 ]; then
    echo "build-for-testing (log: $LOG_DIR/build.log)..."
    if ! "${XCODEBUILD[@]}" build-for-testing > "$LOG_DIR/build.log" 2>&1; then
        summarize "$LOG_DIR/build.log"
        echo "BUILD FAILED"
        exit 1
    fi
fi

STATUS=0
echo "test-without-building ${ONLY_TESTING[*]} (log: $LOG_DIR/test.log)..."
if ! "${XCODEBUILD[@]}" test-without-building \
    -parallel-testing-enabled NO \
    -test-timeouts-enabled YES -default-test-execution-time-allowance 120 \
    "${ONLY_TESTING[@]}" > "$LOG_DIR/test.log" 2>&1; then
    STATUS=1
fi
summarize "$LOG_DIR/test.log"

if [ "$DOCS" = 1 ]; then
    : > "$LOG_DIR/docs.log"
    for target in "${DOC_TARGETS[@]}"; do
        echo "DocC $target (log: $LOG_DIR/docs.log)..."
        if ! MLX_SWIFT_BUILD_DOC=1 swift package generate-documentation \
            --target "$target" --warnings-as-errors >> "$LOG_DIR/docs.log" 2>&1; then
            STATUS=1
            echo "DocC FAILED: $target"
        fi
    done
    grep -E "^error: |doesn't exist at" "$LOG_DIR/docs.log" | cut -c1-300 | head -20 || true
fi

exit $STATUS
