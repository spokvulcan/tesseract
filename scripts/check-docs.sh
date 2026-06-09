#!/usr/bin/env bash
# check-docs.sh — verify the documentation hasn't drifted from the code.
#
# Checks:
#   1. Backtick-quoted repo paths in the docs exist on disk.
#   2. Markdown links to local files resolve.
#   3. `scripts/dev.sh <subcommand>` mentions name real subcommands.
#   4. Test suites referenced via -only-testing: exist in tesseractTests/.
#   5. Swift files named in ARCHITECTURE.md exist somewhere in the source tree.
#
# No network, no build — runs in seconds, anywhere. Exits non-zero on drift.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOCS="README.md CLAUDE.md ARCHITECTURE.md DISTRIBUTION.md REVIEW.md docs/testing.md docs/agents/domain.md docs/agents/issue-tracker.md docs/agents/triage-labels.md"

# Top-level dirs whose paths the docs are expected to reference accurately.
KNOWN_DIRS="tesseract tesseractTests tesseractUITests scripts docs assets benchmarks AgentPackages Vendor .claude .agents .github"

failures=0
fail() {
    printf 'FAIL  %-18s %s\n' "$1" "$2"
    failures=$((failures + 1))
}

known_top_dir() {
    local top="${1%%/*}"
    local d
    for d in $KNOWN_DIRS; do
        [ "$top" = "$d" ] && return 0
    done
    return 1
}

# --- 1. Backtick-quoted paths ------------------------------------------------
# A backticked token is treated as a checkable path when it starts with a known
# top-level dir, or ends in a source/doc extension. Tokens with spaces,
# wildcards, or placeholders are skipped, as are build outputs.
for doc in $DOCS; do
    if [ ! -f "$doc" ]; then
        fail "$doc" "doc file itself is missing"
        continue
    fi
    while IFS= read -r token; do
        case "$token" in
            ''|*' '*|*'*'*|*'<'*|*'$'*|*'%'*|*'('*|*'…'*|-*|http*|/*|'~'*) continue ;;
            build/*) continue ;;
        esac
        if known_top_dir "$token"; then
            : # checkable
        else
            case "$token" in
                *.swift|*.md|*.sh|*.yml|*.json) : ;; # checkable by extension
                *) continue ;;
            esac
        fi
        if [ ! -e "$token" ] && [ ! -e "tesseract/$token" ]; then
            fail "$doc" "path \`$token\` not found (checked ./ and tesseract/)"
        fi
    done < <(grep -o '`[^`]\{1,\}`' "$doc" | sed 's/^`//; s/`$//' | sort -u)
done

# --- 2. Markdown links to local files ----------------------------------------
for doc in $DOCS; do
    [ -f "$doc" ] || continue
    while IFS= read -r target; do
        case "$target" in
            ''|http*|mailto:*|'#'*) continue ;;
        esac
        target="${target#./}"
        target="${target%%#*}"
        [ -z "$target" ] && continue
        if [ ! -e "$target" ] && [ ! -e "$(dirname "$doc")/$target" ]; then
            fail "$doc" "markdown link target '$target' not found"
        fi
    done < <(grep -oE '\]\([^)]+\)' "$doc" | sed 's/^](//; s/)$//' | sort -u)
done

# --- 3. dev.sh subcommands ----------------------------------------------------
valid_cmds=$(sed -n '/^case /,/^esac/p' scripts/dev.sh \
    | grep -oE '^[[:space:]]+[a-z][a-z0-9-]*\)' | tr -d ' )')
for doc in $DOCS; do
    [ -f "$doc" ] || continue
    while IFS= read -r cmd; do
        if ! printf '%s\n' "$valid_cmds" | grep -qx "$cmd"; then
            fail "$doc" "scripts/dev.sh has no '$cmd' subcommand"
        fi
    done < <(grep -ohE 'scripts/dev\.sh [a-z][a-z0-9-]*' "$doc" | awk '{print $2}' | sort -u)
done

# --- 4. Test suites named in docs/testing.md -----------------------------------
while IFS= read -r suite; do
    if ! grep -rqE "(struct|class|enum) $suite\b" tesseractTests/; then
        fail "docs/testing.md" "test suite '$suite' not found in tesseractTests/"
    fi
done < <(grep -ohE -- '-only-testing:tesseractTests/[A-Za-z0-9_]+' docs/testing.md \
    | sed 's|.*/||' | sort -u)

# --- 5. Swift files named in ARCHITECTURE.md ------------------------------------
while IFS= read -r f; do
    if [ -z "$(find tesseract tesseractTests AgentPackages -name "$f" 2>/dev/null | head -1)" ]; then
        fail "ARCHITECTURE.md" "names '$f' but no such file exists in the source tree"
    fi
done < <(grep -ohE '[A-Za-z0-9_+]+\.swift' ARCHITECTURE.md | sort -u)

# -------------------------------------------------------------------------------
echo ""
if [ "$failures" -gt 0 ]; then
    echo "check-docs: $failures drift issue(s) found."
    exit 1
fi
echo "check-docs: all references resolve. Docs match the code."
