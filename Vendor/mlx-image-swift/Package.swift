// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "mlx-image-swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXImageGen", targets: ["MLXImageGen"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.3"),
        .package(url: "https://github.com/spokvulcan/mlx-swift-lm.git", branch: "feat/paroquant-support"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.6.0"),
    ],
    targets: [
        .target(
            name: "MLXImageGen",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXImageGen",
            swiftSettings: [
                .unsafeFlags(["-O"], .when(configuration: .debug)),
            ]
        ),
    ]
)
