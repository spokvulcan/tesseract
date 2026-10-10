#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/test.sh [--no-build | --build-only] [--slowest N] [suite...]
#
# Runs the app's unit tests (tesseractTests) the fast way: build-for-testing,
# then test-without-building against the built .xctestrun. Going through the
# .xctestrun skips the 7–10 s xcodebuild spends loading the project on every
# -scheme invocation, so one suite takes about 3 s and the whole target about
# 25 s. Prints each failure with its messages (read from the result bundle,
# where xcodebuild's own output leaves them out), then the totals. The logs and
# the result bundle stay under DerivedData/<project>/test-runs/.
#
# A whole-target run keeps every core busy for half a minute, so at most
# TESSERACT_TEST_SLOTS (default 2) whole-target runs proceed at once, across
# every checkout on the machine, and the rest wait for a slot. More at once all
# pass (scripts/test-stress.sh), but each takes longer. Suite runs never wait.
#
#   suite       A Swift Testing suite, by its struct name (ChatSessionTests),
#               or one test as Suite/testName() with the parentheses. No suite
#               runs the whole target. A file name or a test without its
#               parentheses matches nothing, so a run that executes no tests
#               fails (exit 3) instead of passing.
#   --no-build  Reuse the last build-for-testing (iterate on one suite).
#   --build-only  Build for testing and stop.
#   --slowest N   Also list the N slowest tests. In a parallel run a test's
#                 time includes its waits for a thread.
#
# Example: scripts/test.sh ChatSessionTests AgentRunControllerTests

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$PROJECT_DIR/tesseract.xcodeproj"

SKIP_BUILD=0
BUILD_ONLY=0
SLOWEST=0
WHOLE_TARGET=0
ONLY_TESTING=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) SKIP_BUILD=1 ;;
        --build-only) BUILD_ONLY=1 ;;
        --slowest)
            SLOWEST="${2:-}"
            case "$SLOWEST" in ''|*[!0-9]*) echo "--slowest takes a count" >&2; exit 2 ;; esac
            shift ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        tesseractTests) WHOLE_TARGET=1; ONLY_TESTING+=("-only-testing:$1") ;;
        tesseractTests/*) ONLY_TESTING+=("-only-testing:$1") ;;
        *) ONLY_TESTING+=("-only-testing:tesseractTests/$1") ;;
    esac
    shift
done
if [ "$SKIP_BUILD" = 1 ] && [ "$BUILD_ONLY" = 1 ]; then
    echo "--no-build and --build-only leave nothing to do" >&2
    exit 2
fi
if [ ${#ONLY_TESTING[@]} -eq 0 ]; then
    WHOLE_TARGET=1
    ONLY_TESTING=(-only-testing:tesseractTests)
fi
SLOTS="${TESSERACT_TEST_SLOTS:-2}"

# Runs "$@" holding one of the whole-target slots: an flock on a file under
# $TMPDIR, held by the command itself (exec'd), so the kernel releases it when
# the command exits, however it exits. Says on fd 3 that it is waiting.
in_slot() {
    python3 -I -c '
import fcntl, os, sys, time
slots, command = max(int(sys.argv[1]), 1), sys.argv[2:]
directory = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tesseract-test-slots")
os.makedirs(directory, exist_ok=True)
waiting = False
while True:
    for slot in range(slots):
        fd = os.open(os.path.join(directory, f"slot-{slot}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        os.set_inheritable(fd, True)
        os.execvp(command[0], command)
    if not waiting:
        waiting = True
        try:
            os.write(3, f"waiting for a whole-target slot ({slots} at a time)...\n".encode())
        except OSError:
            pass
    time.sleep(1)
' "$SLOTS" "$@"
}

# This checkout's DerivedData: Xcode's default location, whose info.plist
# names the project it belongs to (worktrees each get their own).
derived_data() {
    local dir
    for dir in "$HOME"/Library/Developer/Xcode/DerivedData/tesseract-*; do
        [ -f "$dir/info.plist" ] || continue
        if [ "$(plutil -extract WorkspacePath raw "$dir/info.plist" 2>/dev/null)" = "$PROJECT" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

if [ "$SKIP_BUILD" = 0 ]; then
    BUILD_LOG="$(mktemp -t tesseract-build-for-testing)"
    echo "build-for-testing (log: $BUILD_LOG)..."
    if ! xcodebuild build-for-testing -project "$PROJECT" -scheme tesseract \
        -destination 'platform=macOS' -skipPackagePluginValidation > "$BUILD_LOG" 2>&1; then
        grep -E ': error: |^error: ' "$BUILD_LOG" | cut -c1-300 | awk '!seen[$0]++' | head -40 || true
        echo "BUILD FAILED"
        exit 1
    fi
fi
[ "$BUILD_ONLY" = 0 ] || exit 0

DERIVED_DATA="$(derived_data)" || {
    echo "No DerivedData for $PROJECT: run without --no-build first." >&2
    exit 1
}
XCTESTRUN="$(ls -t "$DERIVED_DATA"/Build/Products/*.xctestrun 2>/dev/null | head -1 || true)"
[ -n "$XCTESTRUN" ] || {
    echo "No .xctestrun under $DERIVED_DATA/Build/Products: run without --no-build first." >&2
    exit 1
}

# One directory per run, so concurrent runs never share a result bundle. The
# five newest finished runs are kept. A run still going is never removed (an
# agent in another terminal may be mid-run); one abandoned a day ago is.
RUNS_DIR="$DERIVED_DATA/test-runs"
mkdir -p "$RUNS_DIR"
{ ls -1t "$RUNS_DIR"/*/finished 2>/dev/null || true; } | tail -n +5 | while IFS= read -r marker; do
    rm -rf "$(dirname "$marker")"
done
find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
RUN_DIR="$(mktemp -d "$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-XXXX")"

echo "test-without-building ${ONLY_TESTING[*]} (log: $RUN_DIR/test.log)..."
START=$(date +%s)
STATUS=0
TEST=(xcodebuild test-without-building -xctestrun "$XCTESTRUN" -destination 'platform=macOS'
    -resultBundlePath "$RUN_DIR/Results.xcresult" "${ONLY_TESTING[@]}")
if [ "$WHOLE_TARGET" = 1 ]; then
    in_slot "${TEST[@]}" 3>&2 > "$RUN_DIR/test.log" 2>&1 || STATUS=$?
else
    "${TEST[@]}" > "$RUN_DIR/test.log" 2>&1 || STATUS=$?
fi
ELAPSED=$(($(date +%s) - START))

if [ -d "$RUN_DIR/Results.xcresult" ]; then
    xcrun xcresulttool get test-results tests --path "$RUN_DIR/Results.xcresult" \
        > "$RUN_DIR/tests.json" 2>/dev/null || true
fi

# Failures with their messages, then the totals. Exits 3 when nothing ran.
python3 -I - "$RUN_DIR/tests.json" "$ELAPSED" "$SLOWEST" <<'EOF' || STATUS=$?
import json, sys
path, elapsed, slowest = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    nodes = json.load(open(path)).get("testNodes", [])
except (OSError, ValueError):
    nodes = []
counts, failures, durations = {}, [], []

def walk(node, suite):
    kind = node.get("nodeType")
    if kind == "Test Suite":
        suite = node.get("name")
    if kind == "Test Case":
        result = node.get("result", "?")
        counts[result] = counts.get(result, 0) + 1
        durations.append((node.get("durationInSeconds") or 0, f"{suite}/{node.get('name')}"))
        if result == "Failed":
            messages = [c.get("name", "") for c in node.get("children", [])
                        if c.get("nodeType") == "Failure Message"]
            failures.append((f"{suite}/{node.get('name')}", messages))
    for child in node.get("children", []) or []:
        walk(child, suite)

for node in nodes:
    walk(node, None)
for name, messages in failures:
    print(f"FAILED {name}")
    for message in messages:
        print("    " + message.replace("\n", "\n    ")[:2000])
total = sum(counts.values())
for seconds, name in sorted(durations, reverse=True)[:slowest]:
    print(f"{seconds:7.2f} s  {name}")
print(f"{total} tests: {counts.get('Passed', 0)} passed, {counts.get('Failed', 0)} failed, "
      f"{counts.get('Skipped', 0)} skipped, in {elapsed} s")
if total == 0:
    print("NO TESTS RAN: name a suite (its struct name) or Suite/testName() with the parentheses.")
    sys.exit(3)
EOF

if [ "$STATUS" != 0 ] && ! grep -q 'Failure Message' "$RUN_DIR/tests.json" 2>/dev/null; then
    # The run failed without a test failure (a crash, a build mismatch): show why.
    grep -E 'error|Crash|crash|Terminated|signal' "$RUN_DIR/test.log" | cut -c1-300 | tail -20 || true
fi
touch "$RUN_DIR/finished"
exit $STATUS
