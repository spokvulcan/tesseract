#!/usr/bin/env bash
# Package the archived app into dist/: a signed DMG (drag-to-Applications
# layout) and a dSYMs zip for symbolicating user crash reports.
#
# Usage: scripts/release/package.sh <version>

set -euo pipefail

VERSION="${1:?usage: package.sh <version>}"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE_PATH="$PROJECT_DIR/build/Tesseract.xcarchive"
APP_PATH="$ARCHIVE_PATH/Products/Applications/Tesseract Agent.app"
DIST="$PROJECT_DIR/dist"
DMG_PATH="$DIST/Tesseract-$VERSION.dmg"

[ -d "$APP_PATH" ] || { echo "Archive missing — run archive.sh first." >&2; exit 1; }

rm -rf "$DIST"
mkdir -p "$DIST"

STAGING=$(mktemp -d)
cp -R "$APP_PATH" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

echo "Creating $DMG_PATH ..."
hdiutil create \
    -volname "Tesseract Agent" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "$DMG_PATH"
rm -rf "$STAGING"

# Sign the DMG itself so Gatekeeper can evaluate the container too.
codesign --sign "Developer ID Application" --timestamp "$DMG_PATH"

echo "Zipping dSYMs ..."
DSYM_DIR="$ARCHIVE_PATH/dSYMs"
if [ -d "$DSYM_DIR" ] && [ -n "$(ls -A "$DSYM_DIR")" ]; then
    (cd "$DSYM_DIR" && zip -qr "$DIST/Tesseract-$VERSION-dSYMs.zip" .)
else
    echo "✗ No dSYMs found in archive — check DEBUG_INFORMATION_FORMAT." >&2
    exit 1
fi

ls -lh "$DIST"
