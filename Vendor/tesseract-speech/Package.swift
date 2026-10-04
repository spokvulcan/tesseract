// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "tesseract-speech",
    // macOS 15 / iOS 18: Core ML fp16 buffers and MLComputePlan (the
    // Neural Engine codec). The app itself requires macOS 26.
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "TesseractSpeech", targets: ["TesseractSpeech"])
    ],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        .package(url: "https://github.com/spokvulcan/mlx-swift", revision: "db60fb7c6069c9d2595c2577153a92b65deddec0"),
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
        // Which MIL ops Core ML places on this machine's Neural Engine (NOT
        // linked by the app): tiny programs, compiled and planned.
        .executableTarget(
            name: "ane-lab",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Tools/ane-lab"
        ),
        // Model-level measurement harness (NOT linked by the app): memory at
        // each phase, per-component timings, and the golden code frames a
        // refactor is checked against. Real weights, below the engine.
        .executableTarget(
            name: "qwen3-tts-bench",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Tools/qwen3-tts-bench"
        ),
    ]
)
