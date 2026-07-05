import Foundation
import MLX
import MLXLMCommon
import os

/// **Decode-shape bench** (spike phase 1 of PRD #173): lane-count curves —
/// aggregate tok/s and per-lane latency vs N — for the two decode shapes the
/// Batch Engine can host as step functions:
///
/// - `interleaved`: N independent deep-copied caches, one `[1, 1]` forward
///   per lane per round — the round-robin shape v1 ships.
/// - `batched`: one `[N, 1]` forward over a batch-N cache — the batched-
///   matmul candidate, measured at equal per-lane contexts (its best case;
///   ragged batching would only be slower).
///
/// Pre-registered verdict (PRD #173 implementation decisions): batched
/// matmul ships as a follow-up iff, at N=4 on the bench model, batched
/// aggregate tok/s ≥ 1.8× a single lane's AND per-lane latency ≤ 1.5× a
/// single lane's, at the worst measured context point, all outputs finite.
/// Interleaved ships v1 either way, so the runner records the verdict
/// instead of failing on it — only instability or setup errors exit 1.
///
/// Run via `--batch-lane-bench`; the bench model follows the standard
/// benchmark model resolution (`TESSERACT_BENCH_MODEL` / config override),
/// so the PRD's small/large × machine-size matrix is N runs of this one
/// binary.
nonisolated enum BatchLaneCurveSupport {

    enum Shape: String, Codable, CaseIterable, Sendable {
        case interleaved
        case batched
    }

    struct Measurement: Codable, Equatable, Sendable {
        let model: String
        let shape: String
        let lanes: Int
        /// Per-lane KV context at measurement start.
        let contexts: [Int]
        /// One decode round: every lane advances one token.
        let roundMs: Double
        let aggregateTokPerS: Double
        let outputsFinite: Bool
    }

    struct Verdict: Codable, Equatable, Sendable {
        let aggregateThreshold: Double
        let latencyThreshold: Double

        struct Cell: Codable, Equatable, Sendable {
            let model: String
            let context: Int
            let singleLaneTokPerS: Double
            let batchedAggregateTokPerS: Double
            let aggregateRatio: Double
            let perLaneLatencyRatio: Double
            let passed: Bool
        }

        let cells: [Cell]
        let batchedDecodeJustified: Bool
    }

    /// Pre-registered thresholds (PRD #173).
    static let aggregateThreshold = 1.8
    static let latencyThreshold = 1.5
    static let verdictLanes = 4

    static let equalContexts = [2_048, 8_192]
    static let laneCounts = [1, 2, 4, 8]
    /// Informational mixed-context interleaved curve (the v1 shape's real
    /// workload has ragged lanes; batched has no ragged counterpart here).
    static let mixedContextCycle = [2_048, 8_192]
    static let warmupRounds = 3
    static let timedRounds = 25

    /// The pre-registered decode-shape verdict over a bench run's
    /// measurements. Judged per equal-context point at N=4 against the
    /// single-lane interleaved baseline of the same context; the worst cell
    /// decides. No batched/baseline pair ⇒ no cells ⇒ not justified.
    static func verdict(measurements: [Measurement]) -> Verdict {
        var cells: [Verdict.Cell] = []
        let batchedAtN = measurements.filter {
            $0.shape == Shape.batched.rawValue
                && $0.lanes == verdictLanes
                && Set($0.contexts).count == 1
        }
        for batched in batchedAtN {
            guard let context = batched.contexts.first,
                let single = measurements.first(where: {
                    $0.model == batched.model
                        && $0.shape == Shape.interleaved.rawValue
                        && $0.lanes == 1
                        && $0.contexts == [context]
                })
            else { continue }
            let aggregateRatio = batched.aggregateTokPerS / single.aggregateTokPerS
            let latencyRatio = batched.roundMs / single.roundMs
            cells.append(
                .init(
                    model: batched.model,
                    context: context,
                    singleLaneTokPerS: single.aggregateTokPerS,
                    batchedAggregateTokPerS: batched.aggregateTokPerS,
                    aggregateRatio: aggregateRatio,
                    perLaneLatencyRatio: latencyRatio,
                    passed: aggregateRatio >= aggregateThreshold
                        && latencyRatio <= latencyThreshold
                        && batched.outputsFinite
                        && single.outputsFinite
                ))
        }
        return Verdict(
            aggregateThreshold: aggregateThreshold,
            latencyThreshold: latencyThreshold,
            cells: cells.sorted { $0.context < $1.context },
            batchedDecodeJustified: !cells.isEmpty && cells.allSatisfy(\.passed)
        )
    }
}

nonisolated enum BatchLaneCurveError: Error, CustomStringConvertible {
    case setupFailed(String)
    case unstableOutputs(String)

    var description: String {
        switch self {
        case .setupFailed(let message):
            return "batch-lane bench setup failed: \(message)"
        case .unstableOutputs(let message):
            return "batch-lane bench produced non-finite outputs: \(message)"
        }
    }
}

/// The harness: loads the bench model, builds per-shape caches from one
/// chunk-prefilled + quantized prototype (interleaved lanes restore through
/// the production deep-copy snapshot — the shipped lane storage), times
/// decode rounds, writes the JSON report, and records the pre-registered
/// verdict.
@MainActor
final class BatchLaneCurveRunner {

    private let runner: BenchmarkRunner
    private var logHandle: FileHandle?

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        setupLogging()
        defer { try? logHandle?.close() }

        log("Batch-lane decode-shape bench — model=\(runner.resolvedModelName)")
        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        log("Loading model from: \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        log("Model loaded.")

        let modelName = runner.resolvedModelName
        let outcome = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try Self.runAllCells(context: context, modelName: modelName)
            }
        }
        for line in outcome.logs { log(line) }

        let verdict = BatchLaneCurveSupport.verdict(measurements: outcome.measurements)
        for cell in verdict.cells {
            log(
                "VERDICT N=\(BatchLaneCurveSupport.verdictLanes) ctx=\(cell.context): "
                    + String(
                        format: "aggregate %.2fx (bar %.1fx), ", cell.aggregateRatio,
                        verdict.aggregateThreshold)
                    + String(
                        format: "per-lane latency %.2fx (bar %.1fx) ",
                        cell.perLaneLatencyRatio, verdict.latencyThreshold)
                    + "→ \(cell.passed ? "PASS" : "FAIL")")
        }
        log(
            "Decode-shape verdict: batched matmul "
                + (verdict.batchedDecodeJustified
                    ? "JUSTIFIED as follow-up" : "NOT justified — interleaved stands"))

        try writeReport(measurements: outcome.measurements, verdict: verdict)

        let unstable = outcome.measurements.filter { !$0.outputsFinite }
        if !unstable.isEmpty {
            throw BatchLaneCurveError.unstableOutputs(
                unstable.map { "\($0.shape) N=\($0.lanes)" }.joined(separator: ", "))
        }
    }

    // MARK: - Cells

    private struct BenchOutcome: Sendable {
        let measurements: [BatchLaneCurveSupport.Measurement]
        let logs: [String]
    }

    private nonisolated static func runAllCells(
        context: ModelContext, modelName: String
    ) throws -> BenchOutcome {
        var measurements: [BatchLaneCurveSupport.Measurement] = []
        var logs: [String] = []
        let parameters = LLMActor.makeGenerateParameters(from: AgentGenerateParameters())

        func record(_ m: BatchLaneCurveSupport.Measurement) {
            measurements.append(m)
            logs.append(
                "\(m.shape) N=\(m.lanes) ctx=\(m.contexts.map(String.init).joined(separator: "/")): "
                    + String(
                        format: "%.2f ms/round, %.1f tok/s aggregate", m.roundMs,
                        m.aggregateTokPerS))
        }

        // Equal-context cells: interleaved lanes restore from one prototype
        // snapshot (the production deep-copy), batched prefills [N, C] once.
        for contextLength in BatchLaneCurveSupport.equalContexts {
            let tokens = BenchmarkHarness.promptTokens(
                targetTokens: contextLength, tokenizer: context.tokenizer)
            logs.append("── ctx=\(contextLength) (\(tokens.count) prompt tokens) ──")
            let prototype = try prefilledQuantizedCache(
                context: context, tokens: tokens, batch: 1, parameters: parameters)
            guard
                let snapshot = HybridCacheSnapshot.capture(
                    cache: prototype, offset: tokens.count, type: .leaf)
            else {
                throw BatchLaneCurveError.setupFailed(
                    "prototype snapshot capture failed at ctx=\(contextLength)")
            }
            let decodeToken = tokens.last ?? 1

            for lanes in BatchLaneCurveSupport.laneCounts {
                let caches = try (0..<lanes).map { _ in try snapshot.restore() }
                record(
                    timeInterleavedRounds(
                        context: context, modelName: modelName, caches: caches,
                        contexts: Array(repeating: tokens.count, count: lanes),
                        decodeToken: decodeToken))
                Memory.clearCache()
            }

            for lanes in BatchLaneCurveSupport.laneCounts where lanes > 1 {
                let batchedCache = try prefilledQuantizedCache(
                    context: context, tokens: tokens, batch: lanes, parameters: parameters)
                record(
                    timeBatchedRounds(
                        context: context, modelName: modelName, cache: batchedCache,
                        lanes: lanes, contextLength: tokens.count, decodeToken: decodeToken))
                Memory.clearCache()
            }
        }

        // Mixed-context interleaved curve (informational): the v1 shape on a
        // ragged workload.
        for lanes in BatchLaneCurveSupport.laneCounts where lanes > 1 {
            var caches: [[any KVCache]] = []
            var contexts: [Int] = []
            var decodeToken = 1
            for lane in 0..<lanes {
                let cycle = BatchLaneCurveSupport.mixedContextCycle
                let contextLength = cycle[lane % cycle.count]
                let tokens = BenchmarkHarness.promptTokens(
                    targetTokens: contextLength, tokenizer: context.tokenizer)
                caches.append(
                    try prefilledQuantizedCache(
                        context: context, tokens: tokens, batch: 1, parameters: parameters))
                contexts.append(tokens.count)
                decodeToken = tokens.last ?? 1
            }
            record(
                timeInterleavedRounds(
                    context: context, modelName: modelName, caches: caches,
                    contexts: contexts, decodeToken: decodeToken))
            Memory.clearCache()
        }

        return BenchOutcome(measurements: measurements, logs: logs)
    }

    /// Chunk-prefill `tokens` (replicated `batch` rows) into a fresh cache,
    /// then quantize it exactly as the production decode boundary does.
    private nonisolated static func prefilledQuantizedCache(
        context: ModelContext,
        tokens: [Int],
        batch: Int,
        parameters: GenerateParameters
    ) throws -> [any KVCache] {
        let flat = MLXArray(tokens.map(Int32.init))
        let input: MLXArray =
            batch == 1
            ? flat
            : broadcast(
                flat.expandedDimensions(axis: 0), to: [batch, tokens.count])
        var cache = context.model.newCache(parameters: parameters)
        _ = try PrefillExecutor.run(
            model: context.model,
            text: LMInput.Text(tokens: input, mask: nil),
            cache: cache,
            prefillStepSize: parameters.prefillStepSize,
            consumeAll: true,
            evalPolicy: .checkedSynchronous
        )
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )
        return cache
    }

    /// One interleaved decode round: each lane's `[1, 1]` forward evaluated
    /// in turn — the shipped engine's round-robin decode shape.
    private nonisolated static func timeInterleavedRounds(
        context: ModelContext,
        modelName: String,
        caches: [[any KVCache]],
        contexts: [Int],
        decodeToken: Int
    ) -> BatchLaneCurveSupport.Measurement {
        let input = LMInput.Text(
            tokens: MLXArray([Int32(decodeToken)])[.newAxis], mask: nil)
        var states = [LMOutput.State?](repeating: nil, count: caches.count)
        var lastOutputs = [MLXArray?](repeating: nil, count: caches.count)

        func round() {
            for lane in caches.indices {
                let output = context.model(
                    input, cache: caches[lane], state: states[lane])
                states[lane] = output.state
                lastOutputs[lane] = output.logits
                eval(output.logits)
            }
        }

        let ms = time(round)
        let finite = lastOutputs.allSatisfy { logits in
            guard let logits else { return false }
            return MLX.all(MLX.isFinite(logits.asType(.float32))).item(Bool.self)
        }
        return BatchLaneCurveSupport.Measurement(
            model: modelName,
            shape: BatchLaneCurveSupport.Shape.interleaved.rawValue,
            lanes: caches.count,
            contexts: contexts,
            roundMs: ms,
            aggregateTokPerS: Double(caches.count) / (ms / 1000),
            outputsFinite: finite
        )
    }

    /// One batched decode round: a single `[N, 1]` forward over the batch-N
    /// cache — the batched-matmul candidate shape.
    private nonisolated static func timeBatchedRounds(
        context: ModelContext,
        modelName: String,
        cache: [any KVCache],
        lanes: Int,
        contextLength: Int,
        decodeToken: Int
    ) -> BatchLaneCurveSupport.Measurement {
        let input = LMInput.Text(
            tokens: broadcast(
                MLXArray([Int32(decodeToken)])[.newAxis], to: [lanes, 1]),
            mask: nil)
        var state: LMOutput.State?
        var lastLogits: MLXArray?

        func round() {
            let output = context.model(input, cache: cache, state: state)
            state = output.state
            lastLogits = output.logits
            eval(output.logits)
        }

        let ms = time(round)
        let finite =
            lastLogits.map { logits in
                MLX.all(MLX.isFinite(logits.asType(.float32))).item(Bool.self)
            } ?? false
        return BatchLaneCurveSupport.Measurement(
            model: modelName,
            shape: BatchLaneCurveSupport.Shape.batched.rawValue,
            lanes: lanes,
            contexts: Array(repeating: contextLength, count: lanes),
            roundMs: ms,
            aggregateTokPerS: Double(lanes) / (ms / 1000),
            outputsFinite: finite
        )
    }

    private nonisolated static func time(_ round: () -> Void) -> Double {
        for _ in 0..<BatchLaneCurveSupport.warmupRounds { round() }
        let started = Date.timeIntervalSinceReferenceDate
        for _ in 0..<BatchLaneCurveSupport.timedRounds { round() }
        let elapsed = Date.timeIntervalSinceReferenceDate - started
        return elapsed / Double(BatchLaneCurveSupport.timedRounds) * 1000
    }

    // MARK: - Plumbing

    private var reportDir: URL {
        DebugPaths.benchmark.appendingPathComponent("batch-lane-curves")
    }

    private func setupLogging() {
        try? FileManager.default.createDirectory(
            at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logHandle = try? FileHandle(forWritingTo: logURL)
    }

    private func log(_ line: String) {
        Log.agent.info("[batch-lane-bench] \(line)")
        logHandle?.write(Data((line + "\n").utf8))
    }

    private func writeReport(
        measurements: [BatchLaneCurveSupport.Measurement],
        verdict: BatchLaneCurveSupport.Verdict
    ) throws {
        struct Report: Codable {
            let date: String
            let hardware: String
            let model: String
            let warmupRounds: Int
            let timedRounds: Int
            let measurements: [BatchLaneCurveSupport.Measurement]
            let verdict: BatchLaneCurveSupport.Verdict
        }
        var hardware = utsname()
        uname(&hardware)
        let machine = withUnsafeBytes(of: &hardware.machine) { bytes in
            String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            hardware: "\(machine) / \(ProcessInfo.processInfo.physicalMemory >> 30) GiB",
            model: runner.resolvedModelName,
            warmupRounds: BatchLaneCurveSupport.warmupRounds,
            timedRounds: BatchLaneCurveSupport.timedRounds,
            measurements: measurements,
            verdict: verdict
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        try encoder.encode(report).write(
            to: reportDir.appendingPathComponent("batch_lane_curves_\(stamp).json"),
            options: .atomic)
    }
}
