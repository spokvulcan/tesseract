#!/usr/bin/env bash
# Dev workflow automation: build, kill, run tesseract without manual Xcode interaction.
# Usage: scripts/dev.sh <command>
#
# Commands:
#   build   Build the project (shows errors/warnings only)
#   run     Kill running app + launch the built app
#   dev     Build + kill + run (the main workflow)
#   clean   Clean build artifacts and derived data
#   log     Tail system log filtered to tesseract

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"
SCHEME="tesseract"
BUNDLE_ID="com.tesseract.app"
DERIVED_DATA_GLOB="$HOME/Library/Developer/Xcode/DerivedData/tesseract-*"

# --- Helpers ---------------------------------------------------------------

find_app() {
    local app_path
    app_path=$(find $DERIVED_DATA_GLOB -path "*/Build/Products/Debug/tesseract.app" -not -path "*/Index.noindex/*" -maxdepth 5 2>/dev/null | head -1)
    if [ -z "$app_path" ]; then
        echo "Error: tesseract.app not found in DerivedData. Run 'scripts/dev.sh build' first." >&2
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

# --- Commands --------------------------------------------------------------

cmd_build() {
    echo "Building tesseract..."
    local build_output
    local exit_code=0
    build_output=$(xcodebuild build -project "$PROJECT" -scheme "$SCHEME" 2>&1) || exit_code=$?

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
    local app_path
    app_path=$(find_app) || return 1

    kill_app
    echo "Launching $app_path ..."
    open "$app_path"
    echo "App launched."
}

cmd_dev() {
    cmd_build
    echo ""
    cmd_run
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
    echo "  build   Build the project (shows errors/warnings only)"
    echo "  run     Kill running app + launch the built app"
    echo "  dev     Build + kill + run (the main workflow)"
    echo "  clean   Clean build artifacts and derived data"
    echo "  log     Tail system log filtered to tesseract"
}

# --- Main ------------------------------------------------------------------

case "${1:-}" in
    build) cmd_build ;;
    run)   cmd_run ;;
    dev)   cmd_dev ;;
    clean) cmd_clean ;;
    log)   cmd_log ;;
    *)     usage ;;
esac
