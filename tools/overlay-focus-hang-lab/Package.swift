// swift-tools-version: 6.2
//
// overlay-focus-hang-lab — the runtime harness for the 2026-09-15 main-thread
// freeze: a focusable control appearing in an AppKit-hosted SwiftUI view
// after its first layout sends SwiftUI's key-view-loop rebuild into an
// endless loop on macOS 27.0. Standalone by design: it mirrors OverlayPanel
// and the overlay's committed-beat pill but imports zero app code, so it can
// be pointed at a new OS build without the app. See RUNBOOK.md.

import PackageDescription

let package = Package(
    name: "overlay-focus-hang-lab",
    platforms: [.macOS(.v26)],
    targets: [
        .executableTarget(
            name: "overlay-focus-hang-lab",
            path: "Sources/OverlayFocusHangLab",
            swiftSettings: [.defaultIsolation(MainActor.self)]
        )
    ]
)
