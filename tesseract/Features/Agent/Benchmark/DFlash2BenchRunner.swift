import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The DFlash2 perf ruler: autoregressive baseline vs the DFlash2 speculative
/// arm on the same long-context prompt, ABBA-interleaved against thermal
/// drift (the experiments-ledger discipline), decode-only timing (iterator
/// construction/prefill happens outside the timed region).
///
/// Driven via `scripts/bench.sh quick --model qwen3.8-27b --dflash2-bench`.
nonisolated struct DFlash2BenchRunner {
    let runner: BenchmarkRunner

    private struct ArmResult: Sendable {
        let arm: String
        let runIndex: Int
        let decodeSeconds: Double
        let tokens: Int
        let accepted: Int
        let proposed: Int
        var tokPerSec: Double { Double(tokens) / decodeSeconds }
    }

    @MainActor
    func run() async throws {
        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        Self.log("[dflash2-bench] loading model: \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        Self.log("[dflash2-bench] model loaded")

        guard
            let draftDir = DFlash2Support.draftDirectory(
                storageRoot: ModelDownloadManager.modelStorageURL)
        else {
            Self.log("[dflash2-bench] FATAL: draft not downloaded")
            throw DFlash2BenchError.draftMissing
        }

        // The harness runs via `open -W` (nice 0 — the ledger discipline), so
        // the report lands in the bench log file, not stdout. Write every
        // line as it is produced: a mid-bench crash must not lose the runs
        // that already finished (last time a String(format:) crash left an
        // empty log).
        let outputDir = runner.activeConfig.outputDir
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let logURL = outputDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let handle = FileHandle(forWritingAtPath: logURL.path)
        let emit: @Sendable (String) -> Void = { line in
            handle?.write(Data((line + "\n").utf8))
            Self.log(line)
        }
        defer { try? handle?.close() }

        let results = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.benchAll(context: context, draftDir: draftDir, emit: emit)
            }
        }

        // Summary: median tok/s per arm.
        emit("[dflash2-bench] === summary ===")
        let arms = ["ar"] + Self.blockSizes().map { "dflash2-bs\($0)" }
        var medians: [String: Double] = [:]
        for arm in arms {
            let rates = results.filter { $0.arm == arm }.map(\.tokPerSec).sorted()
            guard !rates.isEmpty else { continue }
            let median = rates[rates.count / 2]
            medians[arm] = median
            let accepted = results.filter { $0.arm == arm }.reduce(0) { $0 + $1.accepted }
            let proposed = results.filter { $0.arm == arm }.reduce(0) { $0 + $1.proposed }
            let accStr =
                proposed > 0
                ? String(
                    format: "  acceptance %.1f%% (%d/%d)",
                    100.0 * Double(accepted) / Double(proposed), accepted, proposed)
                : ""
            emit(
                "[dflash2-bench] \(arm.paddedToColumn) median \(String(format: "%6.1f", median)) tok/s\(accStr)"
            )
        }
        if let base = medians["ar"] {
            for arm in arms where arm != "ar" {
                if let m = medians[arm] {
                    emit(
                        "[dflash2-bench] \(arm.paddedToColumn) speedup \(String(format: "%.2f", m / base))x"
                    )
                }
            }
        }
    }

    /// The whole bench inside one Metal-affine batch: prepare the prompt,
    /// load the draft, then ABBA-interleave the arms (decode-only timing —
    /// iterator construction/prefill happens before the clock starts).
    private static func benchAll(
        context: ModelContext, draftDir: URL, emit: (String) -> Void
    ) async throws -> [ArmResult] {
        let maxNewTokens = 192
        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(buildPromptText())]))
        let promptTokens = prepared.text.tokens.dim(-1)

        let draft = try DFlash2Support.loadDrafter(directory: draftDir)

        func runAR(_ runIndex: Int) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            var iterator = try TokenIterator(
                input: prepared, model: context.model, cache: nil,
                parameters: parameters)
            let start = ContinuousClock.now
            var tokens = 0
            while iterator.next() != nil { tokens += 1 }
            let seconds = elapsedSeconds(since: start)
            return ArmResult(
                arm: "ar", runIndex: runIndex, decodeSeconds: seconds,
                tokens: tokens, accepted: 0, proposed: 0)
        }

        func runDFlash2(_ runIndex: Int, blockSize: Int) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            let cache = try context.model.newCache(parameters: parameters)
            var iterator = try DFlash2SpeculativeTokenIterator(
                input: prepared, mainModel: context.model, drafter: draft,
                mainCache: cache, parameters: parameters, blockSize: blockSize)
            let start = ContinuousClock.now
            var tokens = 0
            while iterator.next() != nil { tokens += 1 }
            let seconds = elapsedSeconds(since: start)
            if iterator.profileRoundCount > 0 {
                let rounds = Double(iterator.profileRoundCount)
                let parts = ["propose", "verify", "accept", "reconcile"].map { phase in
                    let ms = 1000 * (iterator.profilePhaseSeconds[phase] ?? 0) / rounds
                    return "\(phase)=\(String(format: "%.1f", ms))ms"
                }
                emit(
                    "[dflash2-bench] profile bs\(blockSize): \(parts.joined(separator: " ")) "
                        + "(\(iterator.profileRoundCount) rounds)")
            }
            return ArmResult(
                arm: "dflash2-bs\(blockSize)", runIndex: runIndex,
                decodeSeconds: seconds, tokens: tokens,
                accepted: iterator.acceptedCount, proposed: iterator.proposedCount)
        }

        func report(_ result: ArmResult) {
            let acc =
                result.proposed > 0
                ? " accepted=\(result.accepted)/\(result.proposed)" : ""
            emit(
                "[dflash2-bench] \(result.arm.paddedToColumn) run\(result.runIndex): "
                    + String(
                        format: "%6.1f tok/s (%d tokens in %.2fs)", result.tokPerSec, result.tokens,
                        result.decodeSeconds)
                    + acc)
        }

        var results: [ArmResult] = []
        emit(
            String(
                format: "[dflash2-bench] prompt: %d tokens; %d new per run", promptTokens,
                maxNewTokens))
        emit("[dflash2-bench] draft loaded (4-bit)")
        let blocks = Self.blockSizes()
        for round in 0..<2 {
            let a = try runAR(round * 2)
            results.append(a)
            report(a)
            for blockSize in blocks {
                let b = try runDFlash2(round, blockSize: blockSize)
                results.append(b)
                report(b)
            }
            let a2 = try runAR(round * 2 + 1)
            results.append(a2)
            report(a2)
        }
        return results
    }

    /// `--bench-blocks 3,4,5,8` overrides the default [8, 5] scan.
    private static func blockSizes() -> [Int] {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--bench-blocks"), i + 1 < args.count else {
            return [8, 5]
        }
        let parsed = args[i + 1].split(separator: ",").compactMap { Int($0) }
        return parsed.isEmpty ? [8, 5] : parsed
    }

    private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }

    private static func log(_ line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }

    /// Long-context workload: the tesseract repo's own docs plus a question
    /// (mirrors research/bench_dflash.py for cross-stack comparability).
    private static func buildPromptText() -> String {
        var parts: [String] = []
        let repo = "/Users/owl/projects/tesseract"
        for rel in ["ARCHITECTURE.md", "CONTEXT.md", "AGENTS.md"] {
            if let text = try? String(contentsOfFile: "\(repo)/\(rel)", encoding: .utf8) {
                parts.append(text)
            }
        }
        if let adrs = try? FileManager.default.contentsOfDirectory(
            atPath: "\(repo)/docs/adr")
        {
            for name in adrs.filter({ $0.hasSuffix(".md") }).sorted().prefix(12) {
                if let text = try? String(
                    contentsOfFile: "\(repo)/docs/adr/\(name)", encoding: .utf8)
                {
                    parts.append(text)
                }
            }
        }
        let joined = parts.joined(separator: "\n\n")
        // `--bench-context-mult N` tiles the prompt body N times before
        // truncation, scaling the context for the long-KV regime (decode-only
        // timing makes the repeated prefill cost irrelevant).
        var body = String(joined.prefix(24_000))
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--bench-context-mult"), i + 1 < args.count,
            let mult = Int(args[i + 1]), mult > 1
        {
            body = String(
                (0..<mult).map { _ in body }.joined(separator: "\n\n").prefix(24_000 * mult))
        }
        return body
            + "\n\nQuestion: summarize the key architectural decisions in one paragraph.\nAnswer:"
    }
}

private enum DFlash2BenchError: Error {
    case draftMissing
}

extension String {
    /// Fixed-width arm column for bench lines. Replaces `String(format: "%-12s")` —
    /// `%s` on a Swift String bridges to an object pointer, not a `char *`, and
    /// segfaulted the first bench run (DiagnosticReports 2026-08-19 22:19).
    nonisolated fileprivate var paddedToColumn: String {
        padding(toLength: 12, withPad: " ", startingAt: 0)
    }
}
