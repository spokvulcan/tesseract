#!/bin/bash
#
# Regenerate the TesseractHighlight XCFramework + Swift bindings from the
# tesseract-highlight Rust crate (ADR-0029). This script is the ONLY way the
# committed framework is produced — crate changes must land together with the
# regenerated framework in the same commit.
#
# Requires a Rust toolchain (rustup/cargo) with the aarch64-apple-darwin
# target. Day-to-day app builds and CI never need this.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$REPO_ROOT/Vendor/tesseract-highlight"
CRATE_DIR="$PKG_DIR/crate"
TARGET=aarch64-apple-darwin
LIB_NAME=tesseract_highlight

echo "==> cargo test"
(cd "$CRATE_DIR" && cargo test --quiet)

echo "==> cargo build --release ($TARGET)"
(cd "$CRATE_DIR" && cargo build --release --target "$TARGET")

BUILD_DIR="$CRATE_DIR/target/$TARGET/release"

echo "==> uniffi-bindgen (Swift)"
GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT
(cd "$CRATE_DIR" && cargo run --quiet --features uniffi/cli --bin uniffi-bindgen -- \
    generate --library "$BUILD_DIR/lib$LIB_NAME.dylib" \
    --language swift --out-dir "$GEN_DIR")

echo "==> assembling XCFramework"
HEADERS_DIR="$GEN_DIR/headers"
mkdir -p "$HEADERS_DIR"
cp "$GEN_DIR/TesseractHighlightFFI.h" "$HEADERS_DIR/"
cp "$GEN_DIR/TesseractHighlightFFI.modulemap" "$HEADERS_DIR/module.modulemap"

FRAMEWORK="$PKG_DIR/artifacts/TesseractHighlightFFI.xcframework"
rm -rf "$FRAMEWORK"
mkdir -p "$PKG_DIR/artifacts"
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/lib$LIB_NAME.a" \
    -headers "$HEADERS_DIR" \
    -output "$FRAMEWORK"

echo "==> refreshing generated Swift bindings"
mkdir -p "$PKG_DIR/Sources/TesseractHighlight/Generated"
cp "$GEN_DIR/TesseractHighlight.swift" \
    "$PKG_DIR/Sources/TesseractHighlight/Generated/TesseractHighlight.swift"

echo "==> done: $FRAMEWORK"
