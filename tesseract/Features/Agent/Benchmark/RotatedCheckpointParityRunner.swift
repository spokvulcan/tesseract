//
//  RotatedCheckpointParityRunner.swift
//  tesseract
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import os

/// Rotated Ternary Checkpoint acceptance gate (`--rotated-checkpoint-parity`,
/// ADR-0067): prove the production load path of a rotated pack runs the
/// Hadamard rotation.
///
/// A loader that skips the rotation yields plausible garbage, not an error, so
/// no unit test can catch it — only a comparison against an independent
/// implementation can. This runner is the Swift half: load the checkpoint
/// through `AgentEngine` exactly as the chat agent does, assert the manifest
/// modules were substituted with rotated layers, greedy-generate a fixed
/// prompt, and write the prompt and generated token ids to `latest.json`.
/// The reference half (`scripts/rotated_checkpoint_reference.py`, driven by
/// `scripts/dev.sh rotated-checkpoint-parity`) decodes the same prompt ids
/// through mlx-vlm and scores the agreement.
@MainActor
final class RotatedCheckpointParityRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("rotated-checkpoint-parity")

    /// What the reference script consumes. Token ids, never text: the
    /// comparison must not depend on two detokenizers agreeing.
    struct Report: Codable {
        let model: String
        let modelDir: String
        let prompt: String
        let promptTokens: [Int]
        let generatedTokens: [Int]
        let generatedText: String
        let rotatedLinearModules: Int
        let rotatedEmbeddingModules: Int
        let loadSeconds: Double
    }

    private struct Capture: Sendable {
        let generation: BenchmarkHarness.GreedyCapture
        let rotatedLinear: Int
        let rotatedEmbedding: Int
    }

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        setupLogging()
        defer { logFileHandle?.closeFile() }

        let modelDir = try runner.resolveModelDirectory()
        log("Rotated Ternary Checkpoint parity — model=\(runner.resolvedModelName)")
        log("model dir: \(modelDir.path)")

        let engine = AgentEngine()
        let clock = ContinuousClock()
        let start = clock.now
        try await engine.loadModel(from: modelDir, visionMode: false)
        let loadSeconds = (clock.now - start) / .seconds(1)
        log("loaded in \(Self.fmt(loadSeconds))s")

        let capture = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.inspectAndGenerate(context: context)
            }
        }
        engine.unloadModel()
        await engine.awaitPendingUnload()

        let generation = capture.generation
        log(
            "rotated modules: \(capture.rotatedLinear) linear, "
                + "\(capture.rotatedEmbedding) embedding")
        log("prompt tokens: \(generation.promptTokens.count)")
        log("generated \(generation.generatedTokens.count) tokens: \(generation.generatedTokens)")
        log("generated text: \(generation.text)")

        let report = Report(
            model: runner.resolvedModelName,
            modelDir: modelDir.path,
            prompt: BenchmarkHarness.parityPrompt,
            promptTokens: generation.promptTokens,
            generatedTokens: generation.generatedTokens,
            generatedText: generation.text,
            rotatedLinearModules: capture.rotatedLinear,
            rotatedEmbeddingModules: capture.rotatedEmbedding,
            loadSeconds: loadSeconds)
        let reportURL = reportDir.appendingPathComponent("latest.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: reportURL)
        log("report: \(reportURL.path)")

        // The rotation lives in the substituted modules; a plain Qwen3.5 load
        // of this pack would have zero of them and decode garbage. The
        // embedding must be rotated too (its rows are stored in the rotated
        // basis) and the pack ships exactly one.
        guard capture.rotatedLinear > 0, capture.rotatedEmbedding == 1 else {
            log("❌ manifest modules were not substituted — the plain load path was taken")
            log("Overall: FAIL")
            throw RotatedCheckpointParityError.rotationNotApplied(
                linear: capture.rotatedLinear, embedding: capture.rotatedEmbedding)
        }
        guard generation.generatedTokens.count == BenchmarkHarness.parityNewTokens else {
            log("❌ generation stopped early (\(generation.generatedTokens.count) tokens)")
            log("Overall: FAIL")
            throw RotatedCheckpointParityError.generationStoppedEarly(
                generation.generatedTokens.count)
        }
        log("Swift half: PASS — run the reference script to score the token agreement")
        log("Overall: PASS")
    }

    // MARK: - Inspect + generate

    /// Counts the rotated leaves the load installed, then runs the shared
    /// greedy parity generation (`BenchmarkHarness.greedyGenerate`).
    nonisolated private static func inspectAndGenerate(
        context: ModelContext
    ) async throws -> Capture {
        var rotatedLinear = 0
        var rotatedEmbedding = 0
        for (_, module) in context.model.leafModules().flattened() {
            if module is HadamardQuantizedLinear {
                rotatedLinear += 1
            } else if module is HadamardQuantizedEmbedding {
                rotatedEmbedding += 1
            }
        }
        return Capture(
            generation: try await BenchmarkHarness.greedyGenerate(context: context),
            rotatedLinear: rotatedLinear,
            rotatedEmbedding: rotatedEmbedding)
    }

    // MARK: - Logging

    private func setupLogging() {
        try? FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)
    }

    private func log(_ message: String) {
        let line = "[\(Self.timestamp())] \(message)"
        logger.info("\(line, privacy: .public)")
        if let data = (line + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    private static func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

enum RotatedCheckpointParityError: LocalizedError {
    case rotationNotApplied(linear: Int, embedding: Int)
    case generationStoppedEarly(Int)

    var errorDescription: String? {
        switch self {
        case .rotationNotApplied(let linear, let embedding):
            "Rotated Ternary Checkpoint parity: manifest modules were not substituted "
                + "(\(linear) rotated linear, \(embedding) rotated embedding)"
        case .generationStoppedEarly(let count):
            "Rotated Ternary Checkpoint parity: generation stopped after \(count) tokens"
        }
    }
}
