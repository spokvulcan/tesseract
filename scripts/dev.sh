#!/usr/bin/env bash
# Dev workflow automation: build, kill, run tesseract without manual Xcode interaction.
# Usage: scripts/dev.sh <command>
#
# Commands:
#   build       Build the project (Debug; shows errors/warnings only)
#   run         Kill running app + launch the built app (Debug)
#   dev         Build + kill + run using Release (main perf workflow)
#   dev-profile Build + kill + run with profiling env vars enabled
#   clean       Clean build artifacts and derived data
#   log         Tail system log filtered to tesseract

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
SCHEME="tesseract"
BUNDLE_ID="com.tesseract.app"
DERIVED_DATA_GLOB="$HOME/Library/Developer/Xcode/DerivedData/tesseract-*"

# --- Helpers ---------------------------------------------------------------

find_app() {
    local configuration="${1:-Debug}"
    local products_dir
    products_dir=$(xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration "$configuration" -showBuildSettings 2>/dev/null \
        | grep '^\s*BUILT_PRODUCTS_DIR' | awk '{print $3}')
    local app_path="$products_dir/tesseract.app"
    if [ -z "$products_dir" ] || [ ! -d "$app_path" ]; then
        echo "Error: tesseract.app ($configuration) not found at $app_path. Run a matching build first." >&2
        return 1
    fi
    echo "$app_path"
}

kill_app() {
    if pkill -x tesseract 2>/dev/null; then
        echo "Killed running tesseract process."
        sleep 0.5
    fi
}

# Print data directory paths after launching the app.
# Uses OSC 8 escape sequences for clickable file:// links in supported terminals.
print_data_paths() {
    local container="$HOME/Library/Containers/com.tesseract.app/Data"
    local agent_data="$container/Library/Application Support/tesse-ract/agent"
    local conversations="$agent_data/conversations"
    local bench_output="/private/tmp/tesseract-debug/benchmark"

    local ESC
    ESC=$(printf '\033')

    echo ""
    echo "Data:"
    for label_path in \
        "agent:          |$agent_data" \
        "conversations:  |$conversations" \
        "benchmarks:     |$bench_output"; do
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

cmd_build() {
    local configuration="${1:-Debug}"
    shift || true
    echo "Building tesseract ($configuration)..."
    local build_output
    local exit_code=0
    build_output=$(xcodebuild build -project "$PROJECT" -scheme "$SCHEME" \
        -configuration "$configuration" "$@" 2>&1) || exit_code=$?

    # Show errors, warnings, and the final BUILD result
    echo "$build_output" | grep -E "^(error:|warning:|Build |BUILD |\*\*)" || true

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "BUILD FAILED (exit code $exit_code)"
        # Show the last 20 lines for context on failure
        echo "$build_output" | tail -20
        return $exit_code
    fi

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
    local configuration="Release"
    cmd_build "$configuration" \
        SWIFT_COMPILATION_MODE=incremental \
        DEBUG_INFORMATION_FORMAT=dwarf \
        ONLY_ACTIVE_ARCH=YES
    echo ""
    cmd_run "$configuration"
    print_data_paths
}

cmd_dev_profile() {
    local configuration="Release"
    cmd_build "$configuration" \
        SWIFT_COMPILATION_MODE=incremental \
        DEBUG_INFORMATION_FORMAT=dwarf \
        ONLY_ACTIVE_ARCH=YES
    echo ""

    local app_path
    app_path=$(find_app "$configuration") || return 1
    kill_app
    echo "Launching $app_path with profiling..."
    open "$app_path" --args --flux2-profile --qwen3tts-profile
    echo "App launched with profiling enabled."
    print_data_paths
}

cmd_clean() {
    echo "Cleaning build..."
    xcodebuild clean -project "$PROJECT" -scheme "$SCHEME" 2>&1 | tail -1
    echo "Removing DerivedData..."
    rm -rf $DERIVED_DATA_GLOB
    echo "Clean complete."
}

cmd_log() {
    echo "Tailing tesseract logs (Ctrl-C to stop)..."
    echo ""

    # Whitelist: our subsystem + print()/NSLog from our process (empty subsystem).
    # To reveal <private> values, mark interpolations as public: \(path, privacy: .public)
    log stream \
        --predicate 'subsystem == "com.tesseract.app" OR (process == "tesseract" AND subsystem == "")' \
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

        # Extract category from [com.tesseract.app:category]
        local category=""
        if [[ "$line" =~ \[com\.tesseract\.app:([a-z]+)\] ]]; then
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
    echo "  dev         Build + kill + run using Release (the main perf workflow)"
    echo "  dev-profile Build + kill + run with profiling (FLUX2_PROFILE=1, QWEN3TTS_PROFILE=1)"
    echo "  clean       Clean build artifacts and derived data"
    echo "  log         Tail system log filtered to tesseract"
}

# --- Main ------------------------------------------------------------------

case "${1:-}" in
    build)       cmd_build ;;
    run)         cmd_run ;;
    dev)         cmd_dev ;;
    dev-profile) cmd_dev_profile ;;
    clean)       cmd_clean ;;
    log)         cmd_log ;;
    *)           usage ;;
esac
