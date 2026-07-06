#!/usr/bin/env bash
# Notarize a DMG with an App Store Connect API key, staple the ticket, and
# verify Gatekeeper acceptance. Notarizing the DMG covers the app inside it.
#
# Usage: scripts/release/notarize.sh <path-to-dmg>
# Required env:
#   ASC_API_KEY_ID      App Store Connect API key ID
#   ASC_API_ISSUER_ID   App Store Connect issuer ID
#   ASC_API_KEY_P8      full contents of the .p8 private key

set -euo pipefail

DMG_PATH="${1:?usage: notarize.sh <path-to-dmg>}"
: "${ASC_API_KEY_ID:?}" "${ASC_API_ISSUER_ID:?}" "${ASC_API_KEY_P8:?}"

KEY_DIR="${RUNNER_TEMP:-$(mktemp -d)}"
KEY_PATH="$KEY_DIR/asc-api-key.p8"
printf '%s' "$ASC_API_KEY_P8" > "$KEY_PATH"
trap 'rm -f "$KEY_PATH"' EXIT

echo "Submitting $DMG_PATH for notarization ..."
SUBMIT_OUTPUT=$(xcrun notarytool submit "$DMG_PATH" \
    --key "$KEY_PATH" \
    --key-id "$ASC_API_KEY_ID" \
    --issuer "$ASC_API_ISSUER_ID" \
    --wait --timeout 30m 2>&1) || SUBMIT_FAILED=1
echo "$SUBMIT_OUTPUT"

SUBMISSION_ID=$(awk '/^  id: /{print $2; exit}' <<<"$SUBMIT_OUTPUT")

if [ "${SUBMIT_FAILED:-0}" = "1" ] || ! grep -q "status: Accepted" <<<"$SUBMIT_OUTPUT"; then
    if [ -n "$SUBMISSION_ID" ]; then
        echo "✗ Notarization not accepted — fetching the log:" >&2
        xcrun notarytool log "$SUBMISSION_ID" \
            --key "$KEY_PATH" \
            --key-id "$ASC_API_KEY_ID" \
            --issuer "$ASC_API_ISSUER_ID" >&2 || true
    elif grep -qi "agreement" <<<"$SUBMIT_OUTPUT"; then
        cat >&2 <<'EOF'
✗ Apple Developer agreement problem — notarization is blocked account-wide.
  Fix: the Account Holder signs in at https://developer.apple.com/account and
  accepts the pending agreement (also check App Store Connect → Agreements).
  No code or secrets change is needed; re-run this workflow afterwards.
EOF
    else
        echo "✗ Submission failed before Apple accepted it (no submission ID)." >&2
    fi
    exit 1
fi

echo "Stapling ticket ..."
xcrun stapler staple "$DMG_PATH"

echo "Gatekeeper assessment ..."
spctl --assess --type open --context context:primary-signature -v "$DMG_PATH"

echo "✓ Notarized, stapled, and Gatekeeper-accepted."
