//
//  PreparedCheckpointParityRunner.swift
//  tesseract
//

import Foundation
import MLX
import MLXLMCommon
import os

/// Prepared Checkpoint acceptance gate (`--prepared-checkpoint-parity`):
/// prove the prepared-artifact load path preserves PARO semantics at the
/// generation level.
///
/// Protocol: delete the artifact → fresh (converting) load → greedy-generate
/// a fixed prompt → wait for the background artifact write → unload → load
/// again (must take the prepared path — asserted via the artifact staying
/// untouched in place, since a stale/corrupt fallback deletes and rewrites
/// it) → same greedy generation → token sequences must match exactly.
@MainActor
final class PreparedCheckpointParityRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("prepared-checkpoint-parity")

    nonisolated private static let prompt =
        "List the first ten prime numbers, then explain in two sentences why 1 is not prime."
    nonisolated private static let newTokens = 64
    private static let writeTimeout: Duration = .seconds(600)

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        setupLogging()
        defer { logFileHandle?.closeFile() }

        let modelDir = try runner.resolveModelDirectory()
        let artifactURL = modelDir.appendingPathComponent(
            ParoQuantPreparedCheckpoint.fileName, isDirectory: false)
        let fm = FileManager.default

        log("Prepared Checkpoint parity — model=\(runner.resolvedModelName)")

        // Phase 1: fresh, converting load (artifact removed up front).
        try? fm.removeItem(at: artifactURL)
        log("Artifact removed — fresh load will convert and background-write")
        let fresh = try await loadAndGenerate(modelDir: modelDir, label: "fresh")

        // The background write starts once the model is ready; wait for the
        // atomic rename to publish the artifact.
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: Self.writeTimeout)
        while !fm.fileExists(atPath: artifactURL.path) {
            guard clock.now < deadline else {
                log("❌ artifact was not written within \(Self.writeTimeout)")
                throw PreparedCheckpointParityError.writeTimedOut
            }
            try await Task.sleep(for: .milliseconds(500))
        }
        let writtenMtime =
            try fm.attributesOfItem(atPath: artifactURL.path)[.modificationDate] as? Date
        let writtenSize =
            (try fm.attributesOfItem(atPath: artifactURL.path)[.size] as? Int64) ?? 0
        log("Artifact published: \(writtenSize) bytes")

        // Phase 2: prepared load.
        let prepared = try await loadAndGenerate(modelDir: modelDir, label: "prepared")

        let afterMtime =
            try fm.attributesOfItem(atPath: artifactURL.path)[.modificationDate] as? Date
        let untouched = afterMtime == writtenMtime
        log(
            untouched
                ? "✅ artifact untouched — prepared path taken"
                : "❌ artifact rewritten — loader fell back to full conversion")

        let match = fresh.tokens == prepared.tokens
        if match {
            log("✅ token parity: \(fresh.tokens.count) tokens identical")
        } else {
            log("❌ token mismatch:")
            log("  fresh:    \(fresh.tokens)")
            log("  prepared: \(prepared.tokens)")
            log("  fresh text:    \(fresh.text)")
            log("  prepared text: \(prepared.text)")
        }
        log(
            "load times: fresh=\(Self.fmt(fresh.loadSeconds))s "
                + "prepared=\(Self.fmt(prepared.loadSeconds))s")
        log("sample output: \(fresh.text)")

        guard untouched, match else {
            log("Overall: FAIL")
            throw PreparedCheckpointParityError.parityFailed(
                artifactUntouched: untouched, tokensMatch: match)
        }
        log("Overall: PASS")
    }

    // MARK: - Load + generate

    private struct GenerationResult {
        let tokens: [Int]
        let text: String
        let loadSeconds: Double
    }

    private struct TokenCapture: Sendable {
        let tokens: [Int]
        let text: String
    }

    private func loadAndGenerate(modelDir: URL, label: String) async throws -> GenerationResult {
        let engine = AgentEngine()
        let clock = ContinuousClock()
        let start = clock.now
        try await engine.loadModel(from: modelDir, visionMode: false)
        let loadSeconds = (clock.now - start) / .seconds(1)
        log("[\(label)] loaded in \(Self.fmt(loadSeconds))s")

        let capture = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.greedyGenerate(context: context)
            }
        }
        engine.unloadModel()
        await engine.awaitPendingUnload()
        log("[\(label)] generated \(capture.tokens.count) tokens, engine unloaded")
        return GenerationResult(
            tokens: capture.tokens, text: capture.text, loadSeconds: loadSeconds)
    }

    /// Greedy decoding through the vendor iterator so raw token ids are
    /// compared, not detokenized text.
    nonisolated private static func greedyGenerate(
        context: ModelContext
    ) async throws -> TokenCapture {
        var parameters = AgentGenerateParameters(
            maxTokens: newTokens,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            minP: 0.0
        )
        parameters.repetitionPenalty = nil
        let genParams = LLMActor.makeGenerateParameters(from: parameters)

        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(prompt)])
        )
        var iterator = try TokenIterator(
            input: prepared, model: context.model, cache: nil, parameters: genParams
        )
        var ids: [Int] = []
        while ids.count < newTokens, let token = iterator.next() {
            ids.append(token)
        }
        return TokenCapture(tokens: ids, text: context.tokenizer.decode(tokenIds: ids))
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

enum PreparedCheckpointParityError: LocalizedError {
    case writeTimedOut
    case parityFailed(artifactUntouched: Bool, tokensMatch: Bool)

    var errorDescription: String? {
        switch self {
        case .writeTimedOut:
            "Prepared Checkpoint parity: background artifact write did not complete in time"
        case .parityFailed(let untouched, let match):
            "Prepared Checkpoint parity failed — artifactUntouched=\(untouched) tokensMatch=\(match)"
        }
    }
}
