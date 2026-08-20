// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "tesseract-speech",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "TesseractSpeech", targets: ["TesseractSpeech"])
    ],
    dependencies: [
        .package(path: "../mlx-audio-swift"),
        .package(path: "../mlx-swift-lm"),
        .package(url: "https://github.com/spokvulcan/mlx-swift", revision: "9f48e3f3a3f46058fbcd031bbe81d0b0d264ff20"),
    ],
    targets: [
        .target(
            name: "TesseractSpeech",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
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
        // Listening-artifact + measurement harness (NOT linked by the app):
        // drives the production engine + adapter end-to-end against real
        // weights; produces the morning-listen WAVs and the ADR-0037
        // precision-gate RSS numbers.
        .executableTarget(
            name: "v2-listen",
            dependencies: [
                "TesseractSpeech",
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
            ],
            path: "Sources/Tools/v2-listen"
        ),
    ]
)
