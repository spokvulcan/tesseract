#!/usr/bin/env bash
# Fail fast if the archived app is not correctly Developer ID signed with the
# hardened runtime — catching it here beats a cryptic notarization rejection.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_PATH="${1:-$PROJECT_DIR/build/Tesseract.xcarchive/Products/Applications/Tesseract Agent.app}"

echo "Verifying signature: $APP_PATH"

codesign --verify --deep --strict --verbose=2 "$APP_PATH"

INFO=$(codesign -dvv "$APP_PATH" 2>&1)

if ! grep -q "Authority=Developer ID Application" <<<"$INFO"; then
    echo "✗ Not signed with a Developer ID Application identity:" >&2
    grep "Authority=" <<<"$INFO" >&2 || true
    exit 1
fi

if ! grep -qE "flags=.*runtime" <<<"$INFO"; then
    echo "✗ Hardened runtime flag missing — notarization would reject this." >&2
    grep "flags=" <<<"$INFO" >&2 || true
    exit 1
fi

echo "✓ Developer ID signature with hardened runtime."
