import Foundation
import MLXLMCommon
import os

/// Minimal smoke test for the VLM path of `loadParoQuantModel`.
///
/// Exists to cover PR #164 review comment C5 — moving the `in_proj_ba` split
/// out of `Qwen35TextModel.sanitize` and into `ParoQuantLoader` means the
/// split now runs for both LLM and VLM containers. `MLXVLM/Models/Qwen35.swift`
/// doesn't split `in_proj_ba` itself, so VLM load depends entirely on the
/// loader-level split. The L3 prefix-cache E2E only exercises the LLM path;
/// this runner fills the gap.
///
/// Scope is intentionally narrow: load the model in `visionMode: true`, assert
/// the load doesn't throw (this alone validates that `update(parameters:
/// verify: [.allModelKeysSet, .shapeMismatch])` resolved every key against the
/// split weights), then unload cleanly. We don't run a generation because the
/// VLM `prepare` path expects image-aware inputs and the relevant risk from
/// C5 is load-time key resolution, not forward pass.
@MainActor
final class ParoQuantVLMSmokeRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("paroquant-vlm-smoke")

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        setupLogging()
        log("ParoQuantVLMSmoke starting — model=\(runner.resolvedModelName)")

        let modelDir = try runner.resolveModelDirectory()
        log("Loading VLM model from: \(modelDir.path)")

        let engine = AgentEngine()
        do {
            try await engine.loadModel(from: modelDir, visionMode: true)
        } catch {
            log("❌ loadModel(visionMode: true) failed: \(error)")
            throw ParoQuantVLMSmokeError.loadFailed(String(describing: error))
        }

        guard engine.isModelLoaded else {
            log("❌ engine reports not loaded after loadModel")
            throw ParoQuantVLMSmokeError.engineNotReady
        }
        log("✅ VLM model loaded — in_proj_ba split + all PARO rotation keys resolved cleanly.")

        engine.unloadModel()
        await engine.awaitPendingUnload()
        log("✅ Engine unloaded cleanly.")

        log("Overall: PASS")
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
}

enum ParoQuantVLMSmokeError: LocalizedError {
    case loadFailed(String)
    case engineNotReady

    var errorDescription: String? {
        switch self {
        case .loadFailed(let detail):
            "ParoQuant VLM smoke: loadModel(visionMode: true) failed — \(detail)"
        case .engineNotReady:
            "ParoQuant VLM smoke: engine.isModelLoaded == false after loadModel"
        }
    }
}
