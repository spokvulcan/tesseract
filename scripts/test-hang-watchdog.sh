#!/usr/bin/env bash
set -uo pipefail

# Usage: scripts/test-hang-watchdog.sh <minutes> <out-dir> &
#
# Runs in the background beside a test run, as CI's test step starts it. If
# the shell that started it is still running after <minutes>, it samples the
# run's test host into <out-dir>, prints each thread parked in a wait under
# test code with the test frames above the wait, and stops the shell's
# xcodebuild. The step then fails with the hang's stacks in its log instead of
# running out the job's timeout. A hung run is usually a parked pool thread
# (docs/testing.md → Parallel runs share one test host). Once the shell that
# started it exits, the watchdog exits too, without a word.
#
# TEST_HOST is the test host's executable, by default the Debug app under
# ./DerivedData (CI's -derivedDataPath).

minutes="$1"
out="$2"
starter="$PPID"
host="${TEST_HOST:-$PWD/DerivedData/Build/Products/Debug/Tesseract Agent.app/Contents/MacOS/Tesseract Agent}"
deadline=$(($(date +%s) + minutes * 60))

while kill -0 "$starter" 2>/dev/null; do
    if [ "$(date +%s)" -lt "$deadline" ]; then
        sleep 10
        continue
    fi
    mkdir -p "$out"
    echo "::error::The tests are still running after $minutes minutes: sampling the test host, then stopping the run."
    for pid in $(pgrep -f -- "$host"); do
        sample "$pid" 3 -file "$out/sample-$pid.txt" >/dev/null 2>&1 || continue
        echo "Test host $pid, threads parked in test code ($out/sample-$pid.txt has every thread):"
        python3 -I - "$out/sample-$pid.txt" <<'EOF'
import re
import sys

# A thread parked for the whole sample has the same count on every frame of
# its stack, the wait at the bottom included.
WAITS = ("semaphore_wait_trap", "__psynch_cvwait", "__ulock_wait", "_dispatch_group_wait")
graph = open(sys.argv[1], encoding="utf-8", errors="replace").read()
graph = graph.split("Call graph:", 1)[-1].split("Total number in stack", 1)[0]
parked = 0
for thread in re.split(r"\n(?=    \d+ Thread_)", graph):
    lines = thread.strip("\n").splitlines()
    if not lines:
        continue
    total = lines[0].split()[0]
    stack = []
    for line in lines[1:]:
        match = re.match(r"^[\s+!:|]*(\d+) (.*)$", line)
        if match and match.group(1) == total:
            stack.append(re.sub(r"\s+\[0x[0-9a-f]+\]", "", match.group(2)))
    tests = [frame for frame in stack if "(in tesseractTests)" in frame]
    waits = [frame for frame in stack if frame.startswith(WAITS)]
    if not (tests and waits):
        continue
    parked += 1
    print("  " + lines[0].strip())
    for frame in tests + waits[-1:]:
        print("    " + frame)
if not parked:
    print("  No thread is parked in test code.")
EOF
    done
    pkill -P "$starter" xcodebuild || true
    exit 0
done
