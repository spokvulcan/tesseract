#!/usr/bin/env bash
# Import the Developer ID Application certificate into an ephemeral keychain
# on a CI runner. The canonical GitHub/Apple recipe (see docs/releasing.md).
#
# Required env:
#   MACOS_CERTIFICATE_BASE64   base64 of the Developer ID Application .p12
#   MACOS_CERTIFICATE_PASSWORD password protecting the .p12
#   KEYCHAIN_PASSWORD          arbitrary password for the throwaway keychain
#
# CI-only by design — never run against your login keychain.

set -euo pipefail

: "${MACOS_CERTIFICATE_BASE64:?}" "${MACOS_CERTIFICATE_PASSWORD:?}" "${KEYCHAIN_PASSWORD:?}"
: "${RUNNER_TEMP:?install-certificates.sh is CI-only (RUNNER_TEMP unset)}"

CERT_PATH="$RUNNER_TEMP/developer-id.p12"
KEYCHAIN_PATH="$RUNNER_TEMP/release.keychain-db"

echo -n "$MACOS_CERTIFICATE_BASE64" | base64 --decode -o "$CERT_PATH"

security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

security import "$CERT_PATH" \
    -P "$MACOS_CERTIFICATE_PASSWORD" \
    -A -t cert -f pkcs12 \
    -k "$KEYCHAIN_PATH"

# Allow codesign/notarytool to use the key without a UI prompt.
security set-key-partition-list -S apple-tool:,apple:,codesign: \
    -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

security list-keychains -d user -s "$KEYCHAIN_PATH" login.keychain-db

rm -f "$CERT_PATH"

echo "Imported signing identities:"
security find-identity -v -p codesigning "$KEYCHAIN_PATH"
