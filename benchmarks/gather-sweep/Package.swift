// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "gather-sweep",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "/Users/owl/projects/mlx-swift")
    ],
    targets: [
        .executableTarget(
            name: "gather-sweep",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/gather-sweep"
        )
    ]
)
