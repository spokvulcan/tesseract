#!/usr/bin/env bash
# Dev workflow automation: build, kill, run tesseract without manual Xcode interaction.
# Usage: scripts/dev.sh <command>
#
# Commands:
#   build       Build the project (Debug; shows errors/warnings only)
#   run         Kill running app + launch the built app (Debug)
#   dev         Build + kill + run using Debug (fast iteration)
#   dev-release Build + kill + run using Release (perf testing)
#   dev-profile Build + kill + run with profiling env vars enabled
#   archive     Create release archive for App Store submission
#   clean       Clean build artifacts and derived data
#   log         Tail system log filtered to tesseract

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
SCHEME="tesseract"
BUNDLE_ID="app.tesseract.agent"
# Uses Xcode's default DerivedData so CLI and GUI share the same build artifacts,
# module cache, and resolved SPM packages.
DERIVED_DATA_GLOB="$HOME/Library/Developer/Xcode/DerivedData/tesseract-*"

# --- Helpers ---------------------------------------------------------------

find_app() {
    local configuration="${1:-Debug}"
    local app_path
    # Search Xcode's default DerivedData (most recently modified first)
    app_path=$(ls -dt $DERIVED_DATA_GLOB/Build/Products/"$configuration"/Tesseract\ Agent.app 2>/dev/null | head -1)
    if [ -z "$app_path" ] || [ ! -d "$app_path" ]; then
        echo "Error: Tesseract Agent.app ($configuration) not found in Xcode DerivedData. Run a matching build first." >&2
        return 1
    fi
    echo "$app_path"
}

kill_app() {
    if pkill -x "Tesseract Agent" 2>/dev/null; then
        echo "Killed running Tesseract Agent process."
        sleep 0.5
    fi
}

# Print data directory paths after launching the app.
# Uses OSC 8 escape sequences for clickable file:// links in supported terminals.
print_data_paths() {
    local container="$HOME/Library/Containers/app.tesseract.agent/Data"
    local agent_data="$container/Library/Application Support/Tesseract Agent/agent"
    local conversations="$agent_data/conversations"
    local debug_root="$container/tmp/tesseract-debug"
    local bench_output="$debug_root/benchmark"
    local agent_debug="$debug_root/agent"
    local tts_debug="$debug_root"

    local ESC
    ESC=$(printf '\033')

    echo ""
    echo "Data:"
    for label_path in \
        "agent:          |$agent_data" \
        "conversations:  |$conversations" \
        "benchmarks:     |$bench_output" \
        "agent debug:    |$agent_debug" \
        "tts debug:      |$tts_debug"; do
        local label="${label_path%%|*}"
        local path="${label_path#*|}"
        if [ -d "$path" ]; then
            local uri
            uri="file://$(echo "$path" | sed 's/ /%20/g')"
            printf '  %s%s]8;;%s%s\\%s%s]8;;%s\\\n' "$label" "$ESC" "$uri" "$ESC" "$path" "$ESC" "$ESC"
        else
            printf '  %s%s (not created yet)\n' "$label" "$path"
        fi
    done
}

# --- Commands --------------------------------------------------------------

cmd_resolve() {
    echo "Resolving package dependencies..."
    xcodebuild -resolvePackageDependencies \
        -project "$PROJECT" -scheme "$SCHEME" 2>&1 \
    | grep -E "^(error:|warning:|Resolved)" || true
}

cmd_build() {
    local configuration="${1:-Debug}"
    shift || true

    local derived_data
    derived_data=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
    if [ -n "$derived_data" ]; then
        # Check SourcePackages integrity — if checkouts are missing/empty, re-resolve
        local checkouts="$derived_data/SourcePackages/checkouts"
        if [ ! -d "$checkouts" ] || [ -z "$(ls -A "$checkouts" 2>/dev/null)" ]; then
            echo "SourcePackages missing or empty, resolving..."
            xcodebuild -resolvePackageDependencies \
                -project "$PROJECT" -scheme "$SCHEME" 2>&1 \
            | grep -E "^(error:|warning:|Resolved)" || true
        fi
    fi

    echo "Building tesseract ($configuration)..."
    local build_log
    build_log=$(mktemp)
    local exit_code=0
    xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
        -configuration "$configuration" \
        -destination 'platform=macOS' \
        -skipPackagePluginValidation \
        "$@" >"$build_log" 2>&1 || exit_code=$?

    # Show errors, warnings, and the final BUILD result
    grep -E "^(error:|warning:|Build |BUILD |\*\*)" "$build_log" || true

    if [ $exit_code -ne 0 ]; then
        # Auto-recover from "Unable to find module dependency" errors caused by
        # Xcode GUI and xcodebuild CLI sharing DerivedData. The GUI's explicit
        # precompiled module cache becomes stale for CLI builds. Cleaning just the
        # module cache is insufficient — XCBuildData still references deleted modules.
        # Instead, nuke all intermediates (build graph + module cache + object files)
        # and rebuild. Build/Products and SourcePackages are preserved.
        if grep -q "Unable to find module dependency\|missing required module" "$build_log"; then
            echo ""
            echo "Detected stale module cache. Cleaning intermediates and retrying..."
            derived_data=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
            [ -n "$derived_data" ] && rm -rf "$derived_data/Build/Intermediates.noindex" 2>/dev/null

            exit_code=0
            xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
                -configuration "$configuration" \
                -destination 'platform=macOS' \
                -skipPackagePluginValidation \
                "$@" >"$build_log" 2>&1 || exit_code=$?

            grep -E "^(error:|warning:|Build |BUILD |\*\*)" "$build_log" || true

            if [ $exit_code -ne 0 ]; then
                echo ""
                echo "BUILD FAILED after retry (exit code $exit_code)"
                tail -20 "$build_log"
                rm -f "$build_log"
                return $exit_code
            fi
        else
            echo ""
            echo "BUILD FAILED (exit code $exit_code)"
            tail -20 "$build_log"
            rm -f "$build_log"
            return $exit_code
        fi
    fi

    rm -f "$build_log"

    # Stamp the configuration for change detection
    derived_data=$(ls -d $DERIVED_DATA_GLOB 2>/dev/null | head -1)
    [ -n "$derived_data" ] && echo "$configuration" > "$derived_data/.last_config"

    echo ""
    echo "Build succeeded."
}

cmd_run() {
    local configuration="${1:-Debug}"
    local app_path
    app_path=$(find_app "$configuration") || return 1

    kill_app
    echo "Launching $app_path ..."
    open "$app_path"
    echo "App launched."
}

cmd_dev() {
    local configuration="Debug"
    cmd_build "$configuration"
    echo ""
    cmd_run "$configuration"
    print_data_paths
}

cmd_dev_release() {
    local configuration="Release"
    cmd_build "$configuration"
    echo ""
    cmd_run "$configuration"
    print_data_paths
}

cmd_dev_profile() {
    local configuration="Release"
    cmd_build "$configuration"
    echo ""

    local app_path
    app_path=$(find_app "$configuration") || return 1
    kill_app
    echo "Launching $app_path with profiling..."
    open "$app_path" --args --flux2-profile --qwen3tts-profile
    echo "App launched with profiling enabled."
    print_data_paths
}

# Build the app, kill any running instance, then exec the binary with the
# given CLI flag. Tails the runner's `latest.log` and propagates the binary's
# exit code. Used by all loaded-model verification subcommands.
_run_loaded_model_check() {
    local flag="$1"
    local report_subdir="$2"
    local configuration="Debug"
    cmd_build "$configuration"
    echo ""

    local app_path
    app_path=$(find_app "$configuration") || return 1
    kill_app

    local binary="$app_path/Contents/MacOS/Tesseract Agent"
    echo "Running $flag against: $binary"
    local report_dir="$HOME/Library/Containers/app.tesseract.agent/Data/tmp/tesseract-debug/benchmark/$report_subdir"

    local exit_code=0
    "$binary" "$flag" || exit_code=$?

    if [ -f "$report_dir/latest.log" ]; then
        echo ""
        echo "── latest.log ──"
        cat "$report_dir/latest.log"
    fi
    echo ""
    echo "Report dir: $report_dir"
    return $exit_code
}

cmd_prefix_cache_e2e() {
    _run_loaded_model_check --prefix-cache-e2e prefix-cache-e2e
}

cmd_hybrid_cache_correctness() {
    _run_loaded_model_check --hybrid-cache-correctness hybrid-cache-correctness
}

cmd_prefill_step_benchmark() {
    _run_loaded_model_check --prefill-step-benchmark prefill-step-benchmark
}

cmd_paroquant_vlm_smoke() {
    _run_loaded_model_check --paroquant-vlm-smoke paroquant-vlm-smoke
}

cmd_archive() {
    local archive_path="$PROJECT_DIR/build/Tesseract.xcarchive"
    echo "Archiving tesseract for App Store (arm64 only)..."
    rm -rf "$archive_path"
    local archive_output
    local exit_code=0
    archive_output=$(xcodebuild archive \
        -project "$PROJECT" \
        -scheme "$SCHEME" \
        -archivePath "$archive_path" \
        ARCHS=arm64 \
        2>&1) || exit_code=$?

    echo "$archive_output" | grep -E "^(error:|warning:|Build |BUILD |\*\* ARCHIVE)" || true

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ARCHIVE FAILED (exit code $exit_code)"
        echo "$archive_output" | tail -20
        return $exit_code
    fi

    echo ""
    echo "Archive created at: $archive_path"
    echo "Next: open Xcode Organizer (Window → Organizer) to upload to App Store Connect."
}

cmd_clean() {
    echo "Cleaning build..."
    xcodebuild clean -project "$PROJECT" -scheme "$SCHEME" 2>&1 | tail -1
    echo "Removing DerivedData..."
    rm -rf $DERIVED_DATA_GLOB
    # Remove legacy CLI-only DerivedData if it still exists
    rm -rf "$PROJECT_DIR/.build/DerivedData"
    echo "Clean complete."
}

cmd_log() {
    echo "Tailing tesseract logs (Ctrl-C to stop)..."
    echo ""

    # Whitelist: our subsystem + print()/NSLog from our process (empty subsystem).
    # To reveal <private> values, mark interpolations as public: \(path, privacy: .public)
    log stream \
        --predicate 'subsystem == "app.tesseract.agent" OR (process == "Tesseract Agent" AND subsystem == "")' \
        --level debug \
        --style compact 2>&1 \
    | while IFS= read -r line; do
        # Skip the header line from log stream
        [[ "$line" == Filtering* ]] && continue

        # Extract timestamp
        if [[ "$line" =~ ([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+) ]]; then
            local time="${BASH_REMATCH[1]}"
            # Truncate to milliseconds
            time="${time%???}"
        else
            echo "$line"
            continue
        fi

        # Extract log level
        local level=""
        if [[ "$line" =~ (Error|Fault) ]]; then
            level="\033[31mERR\033[0m "
        elif [[ "$line" =~ Warning ]]; then
            level="\033[33mWRN\033[0m "
        fi

        # Extract category from [app.tesseract.agent:category]
        local category=""
        if [[ "$line" =~ \[app\.tesseract\.agent:([a-z]+)\] ]]; then
            category="${BASH_REMATCH[1]}"
        fi

        # Extract message: everything after the last ]
        local msg="${line##*\] }"

        if [ -n "$category" ]; then
            printf "\033[2m%s\033[0m %s\033[36m%-14s\033[0m %s\n" "$time" "$level" "[$category]" "$msg"
        else
            # print()/NSLog/vendor logs — no category, show as-is with timestamp
            printf "\033[2m%s\033[0m %s%s\n" "$time" "$level" "$msg"
        fi
    done
}

usage() {
    echo "Usage: scripts/dev.sh <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build the project (Debug; shows errors/warnings only)"
    echo "  run         Kill running app + launch the built app (Debug)"
    echo "  dev         Build + kill + run using Debug (fast iteration)"
    echo "  dev-release Build + kill + run using Release (perf testing)"
    echo "  dev-profile Build + kill + run with profiling (FLUX2_PROFILE=1, QWEN3TTS_PROFILE=1)"
    echo "  prefix-cache-e2e         Build + run Task 1.8 HybridPrefixCacheE2E (loaded-model cache verification)"
    echo "  hybrid-cache-correctness Build + run Task 2.2 logit-equivalence harness (mid-prefill restore bitwise check)"
    echo "  prefill-step-benchmark   Build + run Task 3.2 prefill-step-size benchmark sweep"
    echo "  paroquant-vlm-smoke      Build + run VLM load smoke for PARO models (PR #164 C5 gate)"
    echo "  archive     Create release archive for App Store submission"
    echo "  resolve     Resolve SPM package dependencies"
    echo "  clean       Clean build artifacts and derived data"
    echo "  log         Tail system log filtered to tesseract"
}

# --- Main ------------------------------------------------------------------

case "${1:-}" in
    build)       cmd_build ;;
    run)         cmd_run ;;
    dev)         cmd_dev ;;
    dev-release) cmd_dev_release ;;
    dev-profile) cmd_dev_profile ;;
    prefix-cache-e2e)         cmd_prefix_cache_e2e ;;
    hybrid-cache-correctness) cmd_hybrid_cache_correctness ;;
    prefill-step-benchmark)   cmd_prefill_step_benchmark ;;
    paroquant-vlm-smoke)      cmd_paroquant_vlm_smoke ;;
    archive)     cmd_archive ;;
    resolve)     cmd_resolve ;;
    clean)       cmd_clean ;;
    log)         cmd_log ;;
    *)           usage ;;
esac
