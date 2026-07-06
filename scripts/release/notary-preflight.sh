#!/usr/bin/env bash
# Cheap authenticated call to the Apple notary service. Fails fast — before
# the expensive archive — when notarization would be rejected for account
# reasons: an unsigned/expired developer agreement, a revoked API key, etc.
#
# Usage: scripts/release/notary-preflight.sh
# Required env:
#   ASC_API_KEY_ID      App Store Connect API key ID
#   ASC_API_ISSUER_ID   App Store Connect issuer ID
#   ASC_API_KEY_P8      full contents of the .p8 private key

set -euo pipefail

: "${ASC_API_KEY_ID:?}" "${ASC_API_ISSUER_ID:?}" "${ASC_API_KEY_P8:?}"

KEY_DIR="${RUNNER_TEMP:-$(mktemp -d)}"
KEY_PATH="$KEY_DIR/asc-api-key.p8"
printf '%s' "$ASC_API_KEY_P8" > "$KEY_PATH"
trap 'rm -f "$KEY_PATH"' EXIT

echo "Checking notary service access ..."
if ! OUTPUT=$(xcrun notarytool history \
    --key "$KEY_PATH" \
    --key-id "$ASC_API_KEY_ID" \
    --issuer "$ASC_API_ISSUER_ID" 2>&1); then
    echo "$OUTPUT" >&2
    if grep -qi "agreement" <<<"$OUTPUT"; then
        cat >&2 <<'EOF'
✗ Apple Developer agreement problem — notarization is blocked account-wide.
  Fix: the Account Holder signs in at https://developer.apple.com/account and
  accepts the pending agreement (also check App Store Connect → Agreements).
  No code or secrets change is needed; re-run this workflow afterwards.
EOF
    else
        echo "✗ Notary service rejected the App Store Connect API key." >&2
    fi
    exit 1
fi
echo "✓ Notary service reachable and credentials accepted."
