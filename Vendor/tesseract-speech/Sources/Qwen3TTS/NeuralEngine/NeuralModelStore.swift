import CoreML
import CryptoKit
import Foundation

/// Where Core ML's compute plan would run a compiled model's ops.
package struct NeuralPlacement: Sendable, CustomStringConvertible {
    package var neuralEngine = 0
    package var gpu = 0
    package var cpu = 0
    package var offEngine: [String] = []
    /// How long reading the plan and loading the model took.
    package var planSeconds = 0.0
    package var loadSeconds = 0.0

    /// Every op on the Neural Engine.
    package var isNeuralEngineOnly: Bool { gpu == 0 && cpu == 0 && neuralEngine > 0 }

    package var description: String {
        "\(neuralEngine) ops on the Neural Engine, \(gpu) GPU, \(cpu) CPU"
            + (offEngine.isEmpty ? "" : " (off: \(offEngine.prefix(8).joined(separator: ", ")))")
            + String(format: " (plan %.1f s, load %.1f s)", planSeconds, loadSeconds)
    }
}

/// The Core ML models this package writes (ADR-0075, ADR-0084): built from a
/// checkpoint's own weights, compiled once and kept in a cache directory
/// under a key naming everything they were built from.
enum NeuralModelStore {
    /// A cache name: `prefix` and a digest of `identity`.
    static func key(_ prefix: String, _ identity: String) -> String {
        let digest = SHA256.hash(data: Data(identity.utf8))
        return prefix + "-" + digest.prefix(12).map { String(format: "%02x", $0) }.joined()
    }

    /// The compiled model `key` in `cacheDirectory`: when it isn't there,
    /// `build` writes the program's `main` function, and it is compiled and
    /// moved into place.
    static func compiled(
        key: String, in cacheDirectory: URL, target: MLProgramPackage.Target = .coreML7,
        metadata: [String: String], build: (MILFunctionBuilder) throws -> Void
    ) async throws -> URL {
        let compiled = cacheDirectory.appendingPathComponent("\(key).mlmodelc", isDirectory: true)
        let fm = FileManager.default
        guard !fm.fileExists(atPath: compiled.path) else { return compiled }
        try fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let package = fm.temporaryDirectory.appendingPathComponent(
            "\(key)-\(UUID().uuidString).mlpackage", isDirectory: true)
        defer { try? fm.removeItem(at: package) }
        let blobs = try MILBlobWriter(url: MLProgramPackage.weightsURL(in: package))
        let builder = MILFunctionBuilder(blobs: blobs)
        try build(builder)
        try blobs.finish()
        try MLProgramPackage.write(
            specification: MLProgramPackage.specification(builder, metadata: metadata, target: target),
            to: package)
        let temporary = try await MLModel.compileModel(at: package)
        try? fm.removeItem(at: compiled)
        try fm.moveItem(at: temporary, to: compiled)
        return compiled
    }

    /// Where `configuration`'s compute plan would run each op of `main`.
    static func placement(of compiled: URL, configuration: MLModelConfiguration) async throws
        -> NeuralPlacement
    {
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        var placement = NeuralPlacement()
        guard case .program(let program) = plan.modelStructure,
            let main = program.functions["main"]
        else { return placement }
        for operation in main.block.operations where operation.operatorName != "const" {
            guard let usage = plan.deviceUsage(for: operation) else { continue }
            switch usage.preferred {
            case .neuralEngine: placement.neuralEngine += 1
            case .gpu:
                placement.gpu += 1
                placement.offEngine.append("\(operation.operatorName)→GPU")
            case .cpu:
                placement.cpu += 1
                placement.offEngine.append("\(operation.operatorName)→CPU")
            @unknown default:
                placement.cpu += 1
            }
        }
        return placement
    }

    /// Loads `compiled` for `computeUnits`. With `requireNeuralEngine`, the
    /// compute plan must put every op on the Neural Engine first.
    static func load(
        _ compiled: URL, computeUnits: MLComputeUnits, requireNeuralEngine: Bool, what: String
    ) async throws -> (model: MLModel, placement: NeuralPlacement) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        var placement = NeuralPlacement()
        let clock = ContinuousClock()
        let started = clock.now
        if requireNeuralEngine {
            placement = try await self.placement(of: compiled, configuration: configuration)
            guard placement.isNeuralEngineOnly else {
                throw AudioGenerationError.modelNotInitialized(
                    "\(what) would not run on the Neural Engine: \(placement).")
            }
        }
        let planned = clock.now
        let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        placement.planSeconds = (planned - started) / .seconds(1)
        placement.loadSeconds = (clock.now - planned) / .seconds(1)
        return (model, placement)
    }
}
