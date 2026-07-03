#!/usr/bin/env bash
# Archive the app with Developer ID signing and a build-time-injected version.
# The pbxproj's MARKETING_VERSION is a dummy on purpose — the released version
# is injected here from the release tag (ADR-0017).
#
# Usage: scripts/release/archive.sh <version> <build-number>
# Required env: APPLE_TEAM_ID
#
# Signing method is parameterized (SIGNING_METHOD, default developer-id) so a
# future Mac App Store variant can reuse this script with a different identity.

set -euo pipefail

VERSION="${1:?usage: archive.sh <version> <build-number>}"
BUILD_NUMBER="${2:?usage: archive.sh <version> <build-number>}"
: "${APPLE_TEAM_ID:?}"

SIGNING_METHOD="${SIGNING_METHOD:-developer-id}"
case "$SIGNING_METHOD" in
    developer-id) IDENTITY="Developer ID Application" ;;
    app-store)    IDENTITY="Apple Distribution" ;;
    *) echo "Unknown SIGNING_METHOD '$SIGNING_METHOD'" >&2; exit 1 ;;
esac

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE_PATH="$PROJECT_DIR/build/Tesseract.xcarchive"

echo "Archiving version $VERSION (build $BUILD_NUMBER, $SIGNING_METHOD)..."
rm -rf "$ARCHIVE_PATH"

LOG=$(mktemp)
EXIT_CODE=0
xcodebuild archive \
    -project "$PROJECT_DIR/tesseract.xcodeproj" \
    -scheme tesseract \
    -configuration Release \
    -destination 'generic/platform=macOS' \
    -archivePath "$ARCHIVE_PATH" \
    -skipPackagePluginValidation \
    ARCHS=arm64 \
    MARKETING_VERSION="$VERSION" \
    CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
    CODE_SIGN_STYLE=Manual \
    DEVELOPMENT_TEAM="$APPLE_TEAM_ID" \
    CODE_SIGN_IDENTITY="$IDENTITY" \
    OTHER_CODE_SIGN_FLAGS="--timestamp" \
    >"$LOG" 2>&1 || EXIT_CODE=$?

grep -E "^(error:|warning:|\*\* ARCHIVE)" "$LOG" || true
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ARCHIVE FAILED (exit code $EXIT_CODE)"
    tail -40 "$LOG"
    rm -f "$LOG"
    exit $EXIT_CODE
fi
rm -f "$LOG"

APP_PATH="$ARCHIVE_PATH/Products/Applications/Tesseract Agent.app"
[ -d "$APP_PATH" ] || { echo "Expected app not found at: $APP_PATH" >&2; exit 1; }
echo "Archived: $APP_PATH"
