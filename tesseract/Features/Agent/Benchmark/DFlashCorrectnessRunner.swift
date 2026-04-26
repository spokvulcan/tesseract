import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import os

/// Loaded-model AR-vs-DFlash equivalence gate. Runs greedy AR and greedy
/// DFlash on the same prompt and reports prefix-match length / acceptance
/// rate / speedup. Run via `--dflash-correctness` on the Tesseract CLI.
@MainActor
final class DFlashCorrectnessRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?

    private static let draftModelID = "qwen3.6-27b-dflash"

    /// Override via `--dflash-max-tokens N`.
    private let maxTokens: Int

    init(runner: BenchmarkRunner, maxTokens: Int? = nil) {
        self.runner = runner
        let arg = CommandLine.arguments
        let cliMax: Int? = {
            if let idx = arg.firstIndex(of: "--dflash-max-tokens"),
               idx + 1 < arg.count, let n = Int(arg[idx + 1]) {
                return n
            }
            return nil
        }()
        self.maxTokens = maxTokens ?? cliMax ?? 256
    }

    // MARK: - Entry point

    func run() async throws {
        setupLogging()
        log("DFlashCorrectness starting — target=\(runner.resolvedModelName), draft=\(Self.draftModelID), maxTokens=\(maxTokens)")

        let engine = AgentEngine()
        let targetDir = try runner.resolveModelDirectory()
        let draftDir = try Self.resolveDraftDirectory(modelID: Self.draftModelID)
        log("Target dir: \(targetDir.path)")
        log("Draft dir:  \(draftDir.path)")

        log("Loading target model...")
        try await engine.loadModel(from: targetDir, visionMode: false)
        log("Target loaded.")

        let maxTokensCopy = maxTokens
        let testRun = try await engine.withModelContainer { container in
            await container.perform { context in
                Self.runAllTests(
                    context: context,
                    draftDir: draftDir,
                    maxTokens: maxTokensCopy
                )
            }
        }

        for line in testRun.logs { log(line) }

        log("\n── Summary ──")
        var allPassed = true
        for check in testRun.checks {
            let mark = check.passed ? "PASS" : "FAIL"
            log("  [\(mark)] \(check.name): \(check.detail)")
            if !check.passed { allPassed = false }
        }

        try writeReport(checks: testRun.checks, allPassed: allPassed)
        log("\nOverall: \(allPassed ? "PASS" : "FAIL")")
        log("Report written to: \(reportURL.path)")
        logFileHandle?.closeFile()

        if !allPassed {
            throw DFlashCorrectnessError.verificationFailed(
                failedChecks: testRun.checks.filter { !$0.passed }.map(\.name)
            )
        }
    }

    // MARK: - Orchestration

    private struct TestRunResult: Sendable {
        let logs: [String]
        let checks: [CheckResult]
    }

    private nonisolated static func runAllTests(
        context: ModelContext,
        draftDir: URL,
        maxTokens: Int
    ) -> TestRunResult {
        var logs: [String] = []
        var checks: [CheckResult] = []

        guard let dflashTarget = context.model as? (any DFlashTarget) else {
            let msg = "Target \(type(of: context.model)) does not conform to DFlashTarget"
            logs.append(msg)
            checks.append(CheckResult(name: "targetConformsToDFlashTarget", passed: false, detail: msg))
            return TestRunResult(logs: logs, checks: checks)
        }
        logs.append("Target conforms to DFlashTarget (\(dflashTarget.dflashLayerCount) layers)")

        let draft: DFlashDraftModel
        do {
            logs.append("Loading draft from \(draftDir.path) ...")
            draft = try DFlashDraftLoader.load(directory: draftDir)
            draft.bind(
                embedTokens: dflashTarget.dflashEmbedTokens,
                lmHead: dflashTarget.dflashLMHead,
                tiesEmbedding: dflashTarget.dflashTiesEmbedding
            )
            logs.append(
                "Draft loaded: \(draft.config.numHiddenLayers) layers, "
                + "block_size=\(draft.config.blockSize), "
                + "target_layer_ids=\(draft.config.dflashConfig.targetLayerIDs), "
                + "mask_token_id=\(draft.config.dflashConfig.maskTokenID)"
            )
        } catch {
            let msg = "Failed to load draft: \(error)"
            logs.append(msg)
            checks.append(CheckResult(name: "draftLoad", passed: false, detail: msg))
            return TestRunResult(logs: logs, checks: checks)
        }

        let badIDs = draft.config.dflashConfig.targetLayerIDs.filter {
            $0 < 0 || $0 >= dflashTarget.dflashLayerCount
        }
        if !badIDs.isEmpty {
            let msg = "target_layer_ids \(badIDs) out of range [0, \(dflashTarget.dflashLayerCount))"
            logs.append(msg)
            checks.append(CheckResult(name: "targetLayerIDsInRange", passed: false, detail: msg))
            return TestRunResult(logs: logs, checks: checks)
        }

        let promptTokens = BenchmarkPrompts.deterministic(
            targetTokens: 64,
            tokenizer: context.tokenizer,
            base: BenchmarkPrompts.dflashShortBase
        )
        logs.append("Prompt tokens: \(promptTokens.count)")

        let promptArray = MLXArray(promptTokens.map { Int32($0) })
        let lmInput = LMInput(tokens: promptArray)

        let arTokens: [Int]
        let arElapsed: TimeInterval
        do {
            let start = CFAbsoluteTimeGetCurrent()
            arTokens = try runAR(
                input: lmInput,
                model: context.model,
                maxTokens: maxTokens
            )
            arElapsed = CFAbsoluteTimeGetCurrent() - start
            logs.append("AR baseline: \(arTokens.count) tokens in \(String(format: "%.2f", arElapsed))s "
                + "(\(String(format: "%.1f", Double(arTokens.count) / arElapsed)) tok/s)")
        } catch {
            let msg = "AR baseline threw: \(error)"
            logs.append(msg)
            checks.append(CheckResult(name: "arBaseline", passed: false, detail: msg))
            return TestRunResult(logs: logs, checks: checks)
        }

        let dflashTokens: [Int]
        let dflashElapsed: TimeInterval
        let acceptanceRate: Double
        do {
            let start = CFAbsoluteTimeGetCurrent()
            let result = try runDFlash(
                input: lmInput,
                model: context.model,
                draft: draft,
                maxTokens: maxTokens
            )
            dflashTokens = result.tokens
            dflashElapsed = CFAbsoluteTimeGetCurrent() - start
            acceptanceRate = result.acceptanceRate
            logs.append("DFlash:      \(dflashTokens.count) tokens in \(String(format: "%.2f", dflashElapsed))s "
                + "(\(String(format: "%.1f", Double(dflashTokens.count) / dflashElapsed)) tok/s, "
                + "acceptance=\(String(format: "%.1f%%", acceptanceRate * 100)))")
        } catch {
            let msg = "DFlash run threw: \(error)"
            logs.append(msg)
            checks.append(CheckResult(name: "dflashRun", passed: false, detail: msg))
            return TestRunResult(logs: logs, checks: checks)
        }

        // Near-AR equivalence rather than strict byte-identity: upstream MLX
        // `scaledDotProductAttention` drifts ~1 fp32 ULP between batched-with-mask
        // and AR-no-mask paths. Per-layer drift accumulates over verify rounds
        // and eventually flips a top-1 sampled token. Switch back to byte-identity
        // when MLX SDPA becomes batch-shape consistent. See
        // `docs/dflash-m2-progress-2026-04-26.md` and the `_known` drift tests
        // in `DFlashGDNRollbackTests`.
        let count = Swift.min(arTokens.count, dflashTokens.count)
        var commonPrefix = 0
        for i in 0..<count {
            if arTokens[i] != dflashTokens[i] { break }
            commonPrefix += 1
        }
        let prefixRatio = arTokens.isEmpty ? 0.0 : Double(commonPrefix) / Double(arTokens.count)
        let minPrefixRatio = 0.50
        let minAcceptance = 0.30

        let speedup = arElapsed / dflashElapsed
        let speedupStr = String(format: "%.2fx", speedup)
        let prefixPctStr = String(format: "%.1f%%", prefixRatio * 100)
        let acceptancePctStr = String(format: "%.1f%%", acceptanceRate * 100)

        let prefixOK = prefixRatio >= minPrefixRatio
        let acceptanceOK = acceptanceRate >= minAcceptance

        let prefixDetail: String
        if commonPrefix == arTokens.count && arTokens.count == dflashTokens.count {
            prefixDetail = "byte-identical: \(arTokens.count) tokens match"
        } else {
            let d = commonPrefix
            let a = arTokens[Swift.max(0, d - 4)..<Swift.min(arTokens.count, d + 4)].map(String.init).joined(separator: ",")
            let b = dflashTokens[Swift.max(0, d - 4)..<Swift.min(dflashTokens.count, d + 4)].map(String.init).joined(separator: ",")
            prefixDetail = "common prefix=\(d) (\(prefixPctStr)); first diff at \(d): AR=\(arTokens[d]) DFlash=\(dflashTokens[d]); AR window=[\(a)], DFlash window=[\(b)]"
        }
        checks.append(CheckResult(
            name: "commonPrefix",
            passed: prefixOK,
            detail: "\(prefixDetail); threshold ≥\(String(format: "%.0f%%", minPrefixRatio * 100))"
        ))

        checks.append(CheckResult(
            name: "acceptanceRate",
            passed: acceptanceOK,
            detail: "\(acceptancePctStr) (threshold ≥\(String(format: "%.0f%%", minAcceptance * 100))); speedup=\(speedupStr)"
        ))

        checks.append(CheckResult(
            name: "dflashNearAR",
            passed: prefixOK && acceptanceOK,
            detail: "prefix=\(prefixPctStr) (\(commonPrefix)/\(arTokens.count)), acceptance=\(acceptancePctStr), speedup=\(speedupStr); AR=\(formatTime(arElapsed)), DFlash=\(formatTime(dflashElapsed))"
        ))

        return TestRunResult(logs: logs, checks: checks)
    }

    // MARK: - AR baseline

    private nonisolated static func runAR(
        input: LMInput,
        model: any LanguageModel,
        maxTokens: Int
    ) throws -> [Int] {
        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0
        )
        var iterator = try TokenIterator(
            input: input, model: model, parameters: parameters
        )
        var tokens: [Int] = []
        tokens.reserveCapacity(maxTokens)
        while let t = iterator.next() {
            tokens.append(t)
            if tokens.count >= maxTokens { break }
        }
        return tokens
    }

    // MARK: - DFlash run

    private nonisolated static func runDFlash(
        input: LMInput,
        model: any LanguageModel,
        draft: DFlashDraftModel,
        maxTokens: Int
    ) throws -> (tokens: [Int], acceptanceRate: Double) {
        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0
        )
        var iterator = try DFlashTokenIterator(
            input: input,
            target: model,
            draft: draft,
            cache: nil,
            parameters: parameters,
            blockSize: draft.config.blockSize,
            maskTokenID: draft.config.dflashConfig.maskTokenID,
            targetLayerIDs: draft.config.dflashConfig.targetLayerIDs
        )
        var tokens: [Int] = []
        tokens.reserveCapacity(maxTokens)
        while let t = iterator.next() {
            tokens.append(t)
            if tokens.count >= maxTokens { break }
        }
        let possibleAccepts = Double(iterator.roundCount * (draft.config.blockSize - 1))
        let rate: Double = possibleAccepts > 0
            ? Double(iterator.acceptedDraftTokens) / possibleAccepts
            : 0
        return (tokens, rate)
    }

    // MARK: - Draft directory resolution

    @MainActor
    static func resolveDraftDirectory(modelID: String) throws -> URL {
        guard let dir = ModelDownloadManager.modelPath(for: modelID) else {
            throw DFlashCorrectnessError.draftDefinitionMissing(modelID)
        }
        guard FileManager.default.fileExists(atPath: dir.path) else {
            let repoID = ModelDefinition.all.first(where: { $0.id == modelID })?.repoID
                ?? "z-lab/Qwen3.6-27B-DFlash"
            throw DFlashCorrectnessError.draftNotDownloaded(
                "Draft model '\(modelID)' not found at \(dir.path). "
                + "Accept the gate at https://huggingface.co/\(repoID) "
                + "and download from the Models page (or via ModelDownloadManager)."
            )
        }
        return dir
    }

    // MARK: - Reporting

    private struct CheckResult: Codable, Sendable {
        let name: String
        let passed: Bool
        let detail: String
    }

    private var reportDir: URL {
        runner.activeConfig.outputDir.appendingPathComponent("dflash-correctness")
    }

    private var reportURL: URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return reportDir
            .appendingPathComponent("dflash_correctness_\(formatter.string(from: Date())).json")
    }

    private func writeReport(checks: [CheckResult], allPassed: Bool) throws {
        struct Report: Codable {
            let date: String
            let target: String
            let draft: String
            let maxTokens: Int
            let passed: Bool
            let checks: [CheckResult]
        }
        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            target: runner.resolvedModelName,
            draft: Self.draftModelID,
            maxTokens: maxTokens,
            passed: allPassed,
            checks: checks
        )
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: reportURL)
    }

    // MARK: - Logging

    private func setupLogging() {
        let dir = reportDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let logURL = dir.appendingPathComponent("log_\(formatter.string(from: Date())).txt")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)
    }

    nonisolated private static func formatTime(_ seconds: TimeInterval) -> String {
        String(format: "%.2fs", seconds)
    }

    private func log(_ message: String) {
        print(message)
        logger.info("\(message, privacy: .public)")
        if let handle = logFileHandle, let data = (message + "\n").data(using: .utf8) {
            handle.write(data)
        }
    }
}

enum DFlashCorrectnessError: Error, CustomStringConvertible {
    case verificationFailed(failedChecks: [String])
    case draftDefinitionMissing(String)
    case draftNotDownloaded(String)

    var description: String {
        switch self {
        case .verificationFailed(let names):
            return "DFlash correctness verification failed: \(names.joined(separator: ", "))"
        case .draftDefinitionMissing(let id):
            return "DFlash draft model '\(id)' not registered in ModelDefinition.all"
        case .draftNotDownloaded(let msg):
            return msg
        }
    }
}
