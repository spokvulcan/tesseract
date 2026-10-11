#!/usr/bin/env bash
set -uo pipefail

# Usage: scripts/ci-xcodebuild.sh <xcodebuild action> [xcodebuild arguments...]
#
# CI's Release builds: xcodebuild through Xcode's compilation cache, which
# ci.yml restores into DerivedData/CompilationCache.noindex before the build
# and saves after it. The cache is keyed by what a compile reads, not by file
# dates, so a fresh checkout replays every compile an earlier run did with the
# same inputs: the packages, and each app file that didn't change. The output
# goes through xcbeautify when the runner has it, and the step ends with how
# many compiles the cache replayed.
#
# A replayed compile writes no index data, so the test job's Debug build stays
# outside the cache and keeps its index whole.

cache="$PWD/DerivedData/CompilationCache.noindex"
log="$(mktemp)"
filter=(cat)
command -v xcbeautify >/dev/null && filter=(xcbeautify)

xcodebuild "$@" \
    COMPILATION_CACHE_ENABLE_CACHING=YES \
    COMPILATION_CACHE_CAS_PATH="$cache" \
    COMPILATION_CACHE_LIMIT_SIZE=4G \
    2>&1 | tee "$log" | "${filter[@]}"
status=$?

replayed=$(grep -c '^Cache hit$' "$log")
compiled=$(grep -c '^Cache miss$' "$log")
size=$(du -sh "$cache" 2>/dev/null | awk '{ print $1 }')
echo "Compilation cache: $replayed compiles replayed, $compiled compiled; ${size:-0B} cached."
exit "$status"
