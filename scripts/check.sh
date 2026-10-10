#!/usr/bin/env bash
# Usage: scripts/check.sh [base]
#
# The commit gate over what changed: the pre-commit hooks of
# .pre-commit-config.yaml (swift-format in place, SwiftLint, the test-only-code
# scan) on every file changed since <base> (default HEAD) or not yet tracked,
# then the docs scan (scripts/check-docs.sh). A reformat fails the run once and
# leaves the files formatted: run it again. `scripts/check.sh main` checks the
# whole branch.
set -euo pipefail

cd "$(dirname "$0")/.."
base="${1:-HEAD}"

if ! command -v pre-commit >/dev/null; then
    echo "pre-commit is not installed: brew install pre-commit" >&2
    exit 2
fi

# Kept files only: a deleted file has nothing to check.
files=()
while IFS= read -r file; do
    [ -f "$file" ] && files+=("$file")
done < <({ git diff --name-only "$base"; git ls-files --others --exclude-standard; } | sort -u)

status=0
if [ ${#files[@]} -gt 0 ]; then
    pre-commit run --files "${files[@]}" || status=$?
else
    echo "No files changed since $base."
fi
scripts/check-docs.sh || status=$?
exit "$status"
