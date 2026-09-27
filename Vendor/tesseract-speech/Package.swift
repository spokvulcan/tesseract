// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "tesseract-speech",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "TesseractSpeech", targets: ["TesseractSpeech"])
    ],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        .package(url: "https://github.com/spokvulcan/mlx-swift", revision: "6058402c676de25560051acda772e80d86d696d1"),
        // swift-transformers pinned to the spokvulcan fork (renderChatTemplate
        // carry, tesseract experiments-ledger C25). Exact-revision pin: SwiftPM
        // cannot mix revision and version requirements for one package, and
        // this is the package's only declarer in the app graph (the app and
        // MLXLMCommon import Tokenizers through it). Scheme:
        // docs/swift-transformers-fork.md.
        .package(url: "https://github.com/spokvulcan/swift-transformers", revision: "fe95f0ad9d13fdc8bf3b19848ae200e8550a875c"),
    ],
    targets: [
        // Qwen3-TTS inference, first-party since ADR-0071 (absorbed from
        // Blaizzy/mlx-audio-swift v0.1.3; provenance and license in
        // THIRD_PARTY.md). VoiceDesign and CustomVoice checkpoints only.
        .target(
            name: "Qwen3TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3TTS"
        ),
        .target(
            name: "TesseractSpeech",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/TesseractSpeech"
        ),
        .testTarget(
            name: "TesseractSpeechTests",
            dependencies: ["TesseractSpeech"],
            path: "Tests/TesseractSpeechTests"
        ),
        // Tiny random-weight models and fixed logits, no checkpoint. Needs
        // Metal: run with xcodebuild (docs/testing.md).
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Tests/Qwen3TTSTests"
        ),
        // Listening-artifact + measurement harness (NOT linked by the app):
        // drives the production engine + adapter end-to-end against real
        // weights; produces listening WAVs and the TTFA/RTF/peak-RSS numbers.
        .executableTarget(
            name: "v2-listen",
            dependencies: ["TesseractSpeech"],
            path: "Sources/Tools/v2-listen"
        ),
    ]
)
