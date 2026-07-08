// swift-tools-version: 6.0
//
// TesseractHighlight — syntax highlighting (syntect) and diff computation
// (similar) from the tesseract-highlight Rust crate, bridged via UniFFI
// (ADR-0029). The XCFramework under artifacts/ is a committed build product;
// regenerate it only via scripts/build-highlight-xcframework.sh.

import PackageDescription

let package = Package(
    name: "TesseractHighlight",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TesseractHighlight", targets: ["TesseractHighlight"])
    ],
    targets: [
        .binaryTarget(
            name: "TesseractHighlightFFI",
            path: "artifacts/TesseractHighlightFFI.xcframework"
        ),
        .target(
            name: "TesseractHighlight",
            dependencies: ["TesseractHighlightFFI"],
            path: "Sources/TesseractHighlight"
        ),
        .testTarget(
            name: "TesseractHighlightTests",
            dependencies: ["TesseractHighlight"],
            path: "Tests/TesseractHighlightTests"
        ),
    ]
)
