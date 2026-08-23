import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

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
        /// First 8 generated token ids — the greedy output-identity gate:
        /// every arm decodes the same prompt, so the fingerprints must match.
        /// (A prefix, not a hash: a hash would hide WHERE arms diverge.)
        let fingerprint: [Int]
        var tokPerSec: Double { Double(tokens) / decodeSeconds }
    }

    @MainActor
    func run() async throws {
        // Stage-2 pipelined-verify defaults (ledger R36). Must precede the
        // first MLX eval: the command-buffer cap is latched on first use.
        // `overwrite: 0` keeps explicit overrides from the command line.
        // - MLX_MAX_ACTIVE_TASKS=40: the 10-buffer cap re-throttles the
        //   round seam once the next verify schedules a round ahead.
        // - MLX_DYNSLICE_INPLACE=1: rolling-KV dynamic writes update rows
        //   in place (safe under the verify masks) instead of copying the
        //   full store per boundary.
        setenv("MLX_MAX_ACTIVE_TASKS", "40", 0)
        setenv("MLX_DYNSLICE_INPLACE", "1", 0)
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
        let arms =
            ["ar"]
            + Self.blockSizes().map {
                "dflash2-bs\($0.block)\($0.adaptive ? "" : "f")"
            }
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
        // Greedy output-identity gate: every arm's first-8 fingerprint must
        // equal the AR arm's (same prompt, same greedy distribution).
        if let arFingerprint = results.first(where: { $0.arm == "ar" })?.fingerprint {
            for arm in arms where arm != "ar" {
                guard let fp = results.first(where: { $0.arm == arm })?.fingerprint else {
                    continue
                }
                let diverge = zip(fp, arFingerprint).enumerated().first(where: {
                    $0.element.0 != $0.element.1
                })?.offset
                emit(
                    "[dflash2-bench] \(arm.paddedToColumn) output-identity: \(fp == arFingerprint ? "MATCH" : "DIVERGED at +\(diverge.map(String.init) ?? "?") (\(fp) vs \(arFingerprint))")"
                )
            }
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

        // Stack same-input gate+up MLP projections into one QMM each
        // (bitwise-neutral, applies to both arms via the shared model).
        if ProcessInfo.processInfo.environment["DFLASH2_STACK_GATEUP"] != "0" {
            let stacked = dflash2StackGateUpProjections(model: context.model)
            // The stacking transiently duplicated the MLP weights; the freed
            // originals sit in the buffer cache and crowd GPU residency —
            // release them before the runs.
            GPU.clearCache()
            emit("[dflash2-bench] same-input projections stacked in \(stacked) blocks")
        }

        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(buildPromptText())]))
        let promptTokens = prepared.text.tokens.dim(-1)

        let draft = try DFlash2Support.loadDrafter(directory: draftDir)
        if ProcessInfo.processInfo.environment["DFLASH2_STACK_GATEUP"] != "0",
            let draftModule = draft as? Module
        {
            let stackedDraft = dflash2StackGateUpProjections(model: draftModule)
            emit("[dflash2-bench] drafter projections stacked in \(stackedDraft) blocks")
        }

        func runAR(_ runIndex: Int) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            var iterator = try TokenIterator(
                input: prepared, model: context.model, cache: nil,
                parameters: parameters)
            let start = ContinuousClock.now
            var tokens = 0
            var fingerprint: [Int] = []
            var allTokens: [Int] = []
            while let token = iterator.next() {
                if fingerprint.count < 8 { fingerprint.append(token) }
                allTokens.append(token)
                tokens += 1
            }
            let seconds = elapsedSeconds(since: start)
            // DFLASH2_DUMP_TEXT=1: decode the AR output for quality
            // eyeballing (a trajectory shift is only acceptable if the
            // content stays coherent).
            if runIndex == 0,
                ProcessInfo.processInfo.environment["DFLASH2_DUMP_TEXT"] == "1"
            {
                emit(
                    "[dflash2-bench] ar-text run0: \(context.tokenizer.decode(tokenIds: allTokens))"
                )
            }
            return ArmResult(
                arm: "ar", runIndex: runIndex, decodeSeconds: seconds,
                tokens: tokens, accepted: 0, proposed: 0, fingerprint: fingerprint)
        }

        func runDFlash2(_ runIndex: Int, blockSize: Int, adaptive: Bool) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            let cache = try context.model.newCache(parameters: parameters)
            var iterator = try DFlash2SpeculativeTokenIterator(
                input: prepared, mainModel: context.model, drafter: draft,
                mainCache: cache, parameters: parameters, blockSize: blockSize,
                adaptiveWidth: adaptive)
            let start = ContinuousClock.now
            var tokens = 0
            var fingerprint: [Int] = []
            while let token = iterator.next() {
                if fingerprint.count < 8 { fingerprint.append(token) }
                tokens += 1
            }
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
                arm: "dflash2-bs\(blockSize)\(adaptive ? "" : "f")", runIndex: runIndex,
                decodeSeconds: seconds, tokens: tokens,
                accepted: iterator.acceptedCount, proposed: iterator.proposedCount,
                fingerprint: fingerprint)
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
            for (blockSize, adaptive) in blocks {
                let b = try runDFlash2(round, blockSize: blockSize, adaptive: adaptive)
                results.append(b)
                report(b)
            }
            let a2 = try runAR(round * 2 + 1)
            results.append(a2)
            report(a2)
        }
        return results
    }

    /// `--bench-blocks 3,4,5,8` overrides the default [8, 5] scan. A suffix
    /// `f` (e.g. `8f`) pins the width (policy off); plain numbers run the
    /// adaptive-width policy under that cap.
    private static func blockSizes() -> [(block: Int, adaptive: Bool)] {
        func parse(_ raw: String) -> [(block: Int, adaptive: Bool)] {
            raw.split(separator: ",").compactMap { token in
                let fixed = token.hasSuffix("f")
                guard let b = Int(fixed ? token.dropLast() : token) else { return nil }
                return (b, !fixed)
            }
        }
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--bench-blocks"), i + 1 < args.count else {
            return parse("8,5")
        }
        let parsed = parse(args[i + 1])
        return parsed.isEmpty ? parse("8,5") : parsed
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
    /// `DFLASH2_BENCH_PROMPT=repeat` swaps in a tiled predictable paragraph —
    /// the agent-typical high-acceptance regime the adaptive width exists for
    /// (the docs prompt is the adversarial low-acceptance one).
    private static func buildPromptText() -> String {
        if ProcessInfo.processInfo.environment["DFLASH2_BENCH_PROMPT"] == "repeat" {
            let sentence =
                "func fibonacci(_ n: Int) -> Int { n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2) }\n"
            let tiled = String(repeating: sentence, count: 400)
            return String(tiled.prefix(24_000))
                + "\n\nQuestion: rewrite the function with an iterative loop.\nAnswer:"
        }
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
