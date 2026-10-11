# Sourced by scripts/dev.sh, scripts/gate.sh, scripts/bench.sh and
# scripts/parity-ab.sh: the running "Tesseract Agent" processes. Several
# checkouts share this Mac, and every test host, harness run and opened app
# carries that name. A run ends by itself and is waited for; an app someone
# opened (the owner's, a scratch launch, from any checkout) is the one a launch
# replaces and a measurement asks to have quit.

# One "pid<TAB>executable" line per running Tesseract Agent.
app_processes() {
    ps -axo pid=,comm= | awk '{
        pid = $1
        sub(/^ *[0-9]+ /, "")
        if ($0 ~ /\/Tesseract Agent$/) print pid "\t" $0
    }'
}

# Whether the process $1 ends by itself: a harness run (its first argument a
# flag such as --prefix-cache-e2e) or the test runner's host.
app_run_ends_by_itself() {
    case "$(ps eww -o command= -p "$1" 2>/dev/null || true)" in
        *"/Tesseract Agent --"* | *" XCTestConfigurationFilePath="* | *" XCTestSessionIdentifier="* | *" XCTestBundlePath="*)
            return 0 ;;
    esac
    return 1
}

# The apps someone opened, as app_processes lines.
opened_apps() {
    local pid executable
    while IFS=$'\t' read -r pid executable; do
        app_run_ends_by_itself "$pid" || printf '%s\t%s\n' "$pid" "$executable"
    done < <(app_processes)
}

# Quit the apps someone opened, so the launch that follows isn't the second
# instance. Runs are left to end by themselves.
quit_opened_apps() {
    local pid executable quit=0
    while IFS=$'\t' read -r pid executable; do
        kill "$pid" 2>/dev/null && quit=1
    done < <(opened_apps)
    if [ "$quit" = 1 ]; then
        echo "Quit the open Tesseract Agent."
        sleep 0.5
    fi
}

# Quit the instances running from the app bundles given: a script's own
# launches (a bench arm it timed out).
quit_apps_in() {
    local pid executable bundle
    while IFS=$'\t' read -r pid executable; do
        for bundle in "$@"; do
            case "$executable" in
                "$bundle"/*) kill "$pid" 2>/dev/null || true ;;
            esac
        done
    done < <(app_processes)
}

# Wait until no Tesseract Agent runs, for a quiet GPU and the run's own
# telemetry: a run (a test host, another gate) is waited for, up to $1
# seconds (600); an opened app fails at once, since it never ends by itself.
# Fails naming what still runs.
wait_for_quiet_app() {
    local limit="${1:-600}" waited=0 opened
    while [ -n "$(app_processes)" ]; do
        opened="$(opened_apps)"
        if [ -n "$opened" ]; then
            echo "Tesseract Agent is open: quit it, then run this again." >&2
            printf '%s\n' "$opened" | sed 's/^/  /' >&2
            return 1
        fi
        if [ "$waited" -ge "$limit" ]; then
            echo "A Tesseract Agent run still runs after ${limit}s:" >&2
            app_processes | sed 's/^/  /' >&2
            return 1
        fi
        [ "$waited" -gt 0 ] || echo "Waiting for a running Tesseract Agent run to finish..."
        sleep 5
        waited=$((waited + 5))
    done
}
