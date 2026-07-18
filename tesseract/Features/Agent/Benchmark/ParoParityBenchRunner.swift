import Foundation
import MLX
import MLXLMCommon
import os

/// Reference-parity benchmark (`--paro-parity-bench`) — wayfinder ticket #217.
///
/// Mirrors `research/paro-moe/bench_reference.py` (the harness that recorded
/// the #206 z-lab/mlx-lm baselines) against the app's own load + raw
/// generation path, so the numbers are directly comparable:
///
/// - Same prompt construction: chat-templated user turn of token-trimmed
///   filler + question, sized to each target context.
/// - Same protocol: greedy sampling, 256 new tokens, 1 short warmup, then
///   `--bench-runs` timed runs per context, MLX buffer cache cleared and
///   peak-memory counter reset before every run.
/// - Same cache config: the KV cache stays unquantized unless
///   `--bench-kv-bits` is passed (the Python reference runs fp16 KV; the
///   app's production default of 8-bit KV is a *different* config and is
///   reported as such).
///
/// Generation goes through `PrefillExecutor`/`TokenIterator` +
/// `TokenGenerationLoop` — the production raw arm (`startRawGeneration`
/// shape), deliberately bypassing the Server Completion prefix cache: its
/// mid-prefill checkpoint captures would bill snapshot copies to prefill
/// time and peak memory that the reference does not pay.
///
/// Flags: `--bench-model-id`, `--bench-output`, `--bench-contexts`
/// (default `128,8192,32768`), `--bench-runs` (default 2), `--bench-max-new`
/// (default 256), `--bench-kv-bits` (default unquantized),
/// `--bench-prefill-steps` (first entry; default production 1024).
@MainActor
final class ParoParityBenchRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private let reportStamp: String
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("paro-parity-bench")

    nonisolated private static let fillerSentence =
        "The history of computing spans mechanical calculators, vacuum tubes, "
        + "transistors, integrated circuits, and the modern era of parallel "
        + "accelerators that execute trillions of operations per second. "

    nonisolated private static let question =
        "\n\nSummarize the text above in as much detail as you can."

    init(runner: BenchmarkRunner) {
        self.runner = runner
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        self.reportStamp = formatter.string(from: Date())
    }

    // MARK: - Records

    nonisolated private struct LoadRecord: Codable {
        let model: String
        let modelID: String
        let hardware: String
        let loadSeconds: Double
        let activeAfterLoadGB: Double
        let peakDuringLoadGB: Double
    }

    nonisolated private struct RunRecord: Codable {
        let targetContext: Int
        let run: Int
        let promptTokens: Int
        let tokenizeSeconds: Double
        let appDriverPrefillSeconds: Double
        let loopPromptSeconds: Double
        let promptSeconds: Double
        let promptTPS: Double
        let generationTokens: Int
        let generateSeconds: Double
        let generationTPS: Double
        let stopReason: String
        let runPeakGB: Double
        let activeAfterGB: Double
    }

    nonisolated private struct Measurement: Sendable {
        let promptTokenCount: Int
        let tokenizeSeconds: Double
        let appDriverPrefillSeconds: Double
        let loopPromptSeconds: Double
        let generationTokenCount: Int
        let generateSeconds: Double
        let stopReason: String

        /// Total prompt-processing seconds. `loopPromptSeconds`
        /// (`GenerateCompletionInfo.promptTime`) already contains the vendor
        /// iterator's own prefill (`promptPrefillTime`, run during iterator
        /// init) plus the loop's first-token split — the 1D text path prefills
        /// entirely in there. `appDriverPrefillSeconds` adds only the
        /// app-driver share of the 2D VLM-class branch, whose
        /// `PrefillExecutor` chunks are invisible to the vendor's counters.
        var promptSeconds: Double { appDriverPrefillSeconds + loopPromptSeconds }
    }

    // MARK: - Run

    func run() async throws {
        setupLogging()

        let contexts = Self.parseIntList(flag: "--bench-contexts") ?? [128, 8192, 32_768]
        let runsPerContext = Self.parseInt(flag: "--bench-runs") ?? 2
        let maxNewTokens = Self.parseInt(flag: "--bench-max-new") ?? 256
        let kvBits = Self.parseInt(flag: "--bench-kv-bits")  // nil = reference parity (fp16 KV)
        let prefillStepSize = runner.activeConfig.prefillStepSizesOverride?.first ?? 1024

        let modelDir = try runner.resolveModelDirectory()
        log(
            "ParoParityBench starting — model=\(runner.resolvedModelName) "
                + "contexts=\(contexts) runs=\(runsPerContext) maxNew=\(maxNewTokens) "
                + "kvBits=\(kvBits.map(String.init) ?? "none") prefillStep=\(prefillStepSize)"
        )
        log("Loading LLM (text-only) from: \(modelDir.path)")

        let engine = AgentEngine()
        Memory.peakMemory = 0
        let loadStart = ContinuousClock.now
        try await engine.loadModel(from: modelDir, visionMode: false)
        let loadSeconds = loadStart.duration(to: .now).seconds
        let loadStats = await engine.llmActor.memoryStats()
        let loadRecord = LoadRecord(
            model: runner.resolvedModelName,
            modelID: runner.activeConfig.resolvedModelID,
            hardware: runner.resolvedHardwareDescription,
            loadSeconds: loadSeconds,
            activeAfterLoadGB: Double(loadStats.activeMB) / 1000,
            peakDuringLoadGB: Double(loadStats.peakMB) / 1000
        )
        log(
            "Loaded in \(Self.fmt(loadSeconds))s — active=\(Self.fmt(loadRecord.activeAfterLoadGB))GB "
                + "loadPeak=\(Self.fmt(loadRecord.peakDuringLoadGB))GB"
        )

        // Build all prompts up front (token-count calibration is untimed).
        var promptByContext: [Int: String] = [:]
        for target in [64] + contexts where promptByContext[target] == nil {
            promptByContext[target] = try await engine.llmActor.withModelContainer { container in
                try await container.perform { context in
                    try await Self.buildPromptText(context: context, targetTokens: target)
                }
            }
        }

        // Warmup: compile kernels on a short prompt, tiny generation.
        log("Warmup…")
        _ = try await measureOnce(
            engine: engine,
            userText: promptByContext[64] ?? Self.question,
            maxNewTokens: 16,
            kvBits: kvBits,
            prefillStepSize: prefillStepSize
        )

        var records: [RunRecord] = []
        for target in contexts {
            guard let promptText = promptByContext[target] else { continue }
            for runIndex in 0..<runsPerContext {
                await engine.llmActor.clearMemoryCache()
                Memory.peakMemory = 0
                let m = try await measureOnce(
                    engine: engine,
                    userText: promptText,
                    maxNewTokens: maxNewTokens,
                    kvBits: kvBits,
                    prefillStepSize: prefillStepSize
                )
                let stats = await engine.llmActor.memoryStats()
                let promptSeconds = m.promptSeconds
                let record = RunRecord(
                    targetContext: target,
                    run: runIndex,
                    promptTokens: m.promptTokenCount,
                    tokenizeSeconds: m.tokenizeSeconds,
                    appDriverPrefillSeconds: m.appDriverPrefillSeconds,
                    loopPromptSeconds: m.loopPromptSeconds,
                    promptSeconds: promptSeconds,
                    promptTPS: promptSeconds > 0 ? Double(m.promptTokenCount) / promptSeconds : 0,
                    generationTokens: m.generationTokenCount,
                    generateSeconds: m.generateSeconds,
                    generationTPS: m.generateSeconds > 0
                        ? Double(m.generationTokenCount) / m.generateSeconds : 0,
                    stopReason: m.stopReason,
                    runPeakGB: Double(stats.peakMB) / 1000,
                    activeAfterGB: Double(stats.activeMB) / 1000
                )
                records.append(record)
                log(
                    "ctx=\(target) run=\(runIndex): prompt=\(record.promptTokens) tok "
                        + "prefill=\(Self.fmt(record.promptTPS)) tok/s "
                        + "decode=\(record.generationTokens) tok @ \(Self.fmt(record.generationTPS)) tok/s "
                        + "(stop=\(record.stopReason)) peak=\(Self.fmt(record.runPeakGB))GB "
                        + "active=\(Self.fmt(record.activeAfterGB))GB "
                        + "[tokenize=\(Self.fmt(m.tokenizeSeconds))s prompt=\(Self.fmt(promptSeconds))s]"
                )
                try writeReport(
                    load: loadRecord,
                    records: records,
                    contexts: contexts,
                    kvBits: kvBits,
                    prefillStepSize: prefillStepSize,
                    maxNewTokens: maxNewTokens
                )
            }
        }

        engine.unloadModel()
        await engine.awaitPendingUnload()
        log("Report written to: \(reportURL.path)")
        log("Overall: PASS")
        logFileHandle?.closeFile()
    }

    // MARK: - Measurement

    private func measureOnce(
        engine: AgentEngine,
        userText: String,
        maxNewTokens: Int,
        kvBits: Int?,
        prefillStepSize: Int
    ) async throws -> Measurement {
        var parameters = AgentGenerateParameters(
            maxTokens: maxNewTokens,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            minP: 0.0
        )
        parameters.repetitionPenalty = nil
        parameters.kvBits = kvBits
        parameters.prefillStepSize = prefillStepSize
        let genParams = LLMActor.makeGenerateParameters(from: parameters)

        return try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.runOnce(
                    context: context, userText: userText, parameters: genParams
                )
            }
        }
    }

    /// One timed generation through the production raw arm — the same
    /// iterator construction as `LLMActor.startRawGeneration`, with the
    /// tokenize / chunked-prefill / decode phases timed separately.
    nonisolated private static func runOnce(
        context: ModelContext,
        userText: String,
        parameters: GenerateParameters
    ) async throws -> Measurement {
        let tokenizeStart = ContinuousClock.now
        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(userText)])
        )
        let tokenizeSeconds = elapsedSeconds(since: tokenizeStart)
        let promptTokenCount = prepared.text.tokens.dim(-1)

        // VLM-class (2D) prompts chunk through the app driver (ADR-0006) —
        // that share is timed here because the vendor's counters never see
        // it. 1D text-only prompts prefill inside upstream's TokenIterator
        // init, which the vendor already bills to `promptPrefillTime` (and
        // the loop folds into `GenerateCompletionInfo.promptTime`) — timing
        // construction as well would double-count the whole prefill.
        var appDriverPrefillSeconds = 0.0
        let iterator: TokenIterator
        if case .chunked(let prefillStep) = PrefillStrategy.decide(
            for: prepared, prefillStepSize: parameters.prefillStepSize
        ) {
            var cache = context.model.newCache(parameters: parameters)
            let prefillStart = ContinuousClock.now
            let warmed = try PrefillExecutor.run(
                model: context.model,
                text: prepared.text,
                cache: cache,
                prefillStepSize: prefillStep
            )
            appDriverPrefillSeconds = elapsedSeconds(since: prefillStart)
            iterator = try PrefillExecutor.makeIterator(
                model: context.model,
                fullText: prepared.text,
                remainder: warmed.remainder,
                cache: &cache,
                parameters: parameters
            )
        } else {
            iterator = try TokenIterator(
                input: prepared,
                model: context.model,
                cache: nil,
                parameters: parameters
            )
        }

        let (stream, completion) = TokenGenerationLoop.start(
            promptTokenCount: promptTokenCount,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator,
            tools: nil
        )
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let i) = event {
                info = i
            }
        }
        await completion.value

        guard let info else {
            throw ParoParityBenchError.missingCompletionInfo
        }
        return Measurement(
            promptTokenCount: promptTokenCount,
            tokenizeSeconds: tokenizeSeconds,
            appDriverPrefillSeconds: appDriverPrefillSeconds,
            loopPromptSeconds: info.promptTime,
            generationTokenCount: info.generationTokenCount,
            generateSeconds: info.generateTime,
            stopReason: String(describing: info.stopReason)
        )
    }

    // MARK: - Prompt construction (mirrors bench_reference.py build_prompt)

    nonisolated private static func buildPromptText(
        context: ModelContext,
        targetTokens: Int
    ) async throws -> String {
        let overhead = try await context.processor.prepare(
            input: UserInput(chat: [.user(question)])
        ).text.tokens.dim(-1)

        let fillerBudget = max(targetTokens - overhead, 0)
        guard fillerBudget > 0 else { return question }

        let fillerIds = context.tokenizer.encode(text: fillerSentence, addSpecialTokens: false)
        let reps = (fillerBudget / max(fillerIds.count, 1)) + 1
        let fillerText = String(repeating: fillerSentence, count: reps)
        let trimmedIds = Array(
            context.tokenizer.encode(text: fillerText, addSpecialTokens: false)
                .prefix(fillerBudget)
        )
        return context.tokenizer.decode(tokenIds: trimmedIds) + question
    }

    // MARK: - CLI parsing

    nonisolated private static func parseInt(flag: String) -> Int? {
        let args = CommandLine.arguments
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
        return Int(args[idx + 1])
    }

    nonisolated private static func parseIntList(flag: String) -> [Int]? {
        let args = CommandLine.arguments
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
        let parsed = args[idx + 1].split(separator: ",").compactMap { Int($0) }
        return parsed.isEmpty ? nil : parsed
    }

    // MARK: - Reporting

    private func writeReport(
        load: LoadRecord,
        records: [RunRecord],
        contexts: [Int],
        kvBits: Int?,
        prefillStepSize: Int,
        maxNewTokens: Int
    ) throws {
        struct Report: Codable {
            let date: String
            let load: LoadRecord
            let contexts: [Int]
            let kvBits: Int?
            let prefillStepSize: Int
            let maxNewTokens: Int
            let runs: [RunRecord]
        }
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            load: load,
            contexts: contexts,
            kvBits: kvBits,
            prefillStepSize: prefillStepSize,
            maxNewTokens: maxNewTokens,
            runs: records
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: reportURL, options: .atomic)
    }

    private var reportURL: URL {
        reportDir.appendingPathComponent("paro_parity_bench_\(reportStamp).json")
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

    nonisolated private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    nonisolated private static func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    nonisolated private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        let (seconds, attoseconds) = duration.components
        return Double(seconds) + Double(attoseconds) * 1e-18
    }
}

nonisolated enum ParoParityBenchError: LocalizedError {
    case missingCompletionInfo

    var errorDescription: String? {
        switch self {
        case .missingCompletionInfo:
            "PARO parity bench: generation stream ended without completion info"
        }
    }
}
