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
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: "dc43e62d7055353c7f99fa071a4e71d29dfddc44"),
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
    ]
)
