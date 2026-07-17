// swift-tools-version: 6.0
//
// voice-hold-lab — the ADR-0041 runtime harness. Standalone by design: it
// exercises raw AVAudioEngine/VPIO with the same call discipline the app's
// voice hold uses, but imports zero app code — the app's unit tests replay
// the traces this lab records (tesseractTests/VoiceLabFixtures.swift), so
// calibration is regression-locked without a shared framework.
//
// v1 of this lab lived in gitignored research/ and evaporated; committed
// here so its evidence travels with the code it gates (ADR-0041 status note).

import PackageDescription

let package = Package(
    name: "voice-hold-lab",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "voice-hold-lab",
            path: "Sources/VoiceHoldLab",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
