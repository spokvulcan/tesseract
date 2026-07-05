import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import os

/// **Paged-KV kernel gate** (spike phase 0 of PRD #173, ADR-0023): before the
/// radix tree's RAM tier becomes refcounted KV Pages, the enabling kernel —
/// attention gathering *quantized* KV from non-contiguous pages — must beat a
/// pre-registered threshold: decode overhead vs contiguous attention ≤ 15% at
/// N ∈ {1, 4} lanes, on the small production model's shapes and ornith-35b's,
/// with no stability regressions. Fail → v1 ships deep-copied per-lane caches
/// and paged waits for a better kernel story (the Batch Engine is identical
/// either way).
///
/// This is a *kernel microbenchmark*, deliberately model-free: it builds
/// synthetic quantized KV at the exact production shapes (attention heads read
/// from each model's `config.json` on disk; `kvBits`/`kvGroupSize` from the
/// production generate-parameter defaults) and times one decode step's
/// attention per lane. Pages slice the token axis, so quantization groups —
/// which run along the head dimension — are never split (the ADR-0023
/// alignment constraint holds by construction).
///
/// Two paged candidates are measured against the contiguous baseline
/// (`quantizedScaledDotProductAttention`, the production decode op):
/// - `gatherThenAttend`: `take` the lane's pages into a contiguous copy, then
///   run the baseline op — the naive fallback shape.
/// - `fusedGatherQMM`: `gatherQuantizedMM` with the page pool as the expert
///   axis, so the gather fuses into the quantized matmul on both the score
///   and value sides — no materialized contiguous copy.
///
/// The gate verdict is taken on the best paged candidate per scenario.
nonisolated enum PagedKVKernelGateSupport {

    /// Attention shape of one model's full-attention layers, read from its
    /// `config.json` (`text_config` when nested).
    struct AttentionShape: Codable, Equatable, Sendable {
        let modelName: String
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int

        var groupHeads: Int { nHeads / nKVHeads }
    }

    /// One timed scenario: lanes with these context lengths, paged at
    /// `pageSize` tokens.
    struct Scenario: Sendable {
        let laneContexts: [Int]
        let pageSize: Int

        var id: String { "N\(laneContexts.count)-P\(pageSize)" }
    }

    enum Variant: String, Codable, CaseIterable, Sendable {
        case contiguous
        case gatherThenAttend
        case fusedGatherQMM
    }

    struct Measurement: Codable, Equatable, Sendable {
        let model: String
        let scenarioID: String
        let laneContexts: [Int]
        let pageSize: Int
        let variant: String
        let msPerDecodeRound: Double
        /// Relative to the contiguous baseline of the same scenario; 0 for
        /// the baseline itself.
        let overheadVsContiguous: Double
        /// Max |paged − contiguous| over the attention output (f32); the
        /// stability proxy alongside "no crash, all finite".
        let maxAbsDiffVsContiguous: Double
        let outputsFinite: Bool
    }

    struct GateVerdict: Codable, Equatable, Sendable {
        let thresholdOverhead: Double
        /// Per (model, N): the best paged candidate's overhead.
        struct Cell: Codable, Equatable, Sendable {
            let model: String
            let lanes: Int
            let bestVariant: String
            let bestPageSize: Int
            let bestOverhead: Double
            let passed: Bool
        }
        let cells: [Cell]
        let passed: Bool
    }

    /// Pre-registered threshold (ADR-0023): ≤ 15% decode overhead.
    static let thresholdOverhead = 0.15
    /// Context points; all divisible by every page size below.
    static let singleLaneContexts = [2_048, 8_192, 30_720]
    static let fourLaneMix = [2_048, 8_192, 8_192, 30_720]
    static let pageSizes = [64, 256]
    static let warmupRounds = 5
    static let timedRounds = 40

    /// Deterministic page placement: logical page `i` lands at physical slot
    /// `placement[i]` — a fixed-seed shuffle so the gather pattern is
    /// fragmented, the honest case for a long-lived pool.
    static func pagePlacement(totalPages: Int, seed: UInt64) -> [Int] {
        var slots = Array(0..<totalPages)
        var state = seed
        // xorshift64* — deterministic across runs, no Foundation RNG needed.
        func next() -> UInt64 {
            state ^= state >> 12
            state ^= state << 25
            state ^= state >> 27
            return state &* 0x2545_F491_4F6C_DD1D
        }
        for i in (1..<slots.count).reversed() {
            let j = Int(next() % UInt64(i + 1))
            slots.swapAt(i, j)
        }
        return slots
    }

    /// Read a model's full-attention shape from `config.json` (nested
    /// `text_config` wins — the VLM container's outer config carries no
    /// attention heads).
    static func readShape(modelDirectory: URL, modelName: String) throws -> AttentionShape {
        let url = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PagedKVKernelGateError.badConfig(url.path)
        }
        let config = (root["text_config"] as? [String: Any]) ?? root
        guard
            let nHeads = config["num_attention_heads"] as? Int,
            let nKVHeads = config["num_key_value_heads"] as? Int,
            let headDim = config["head_dim"] as? Int
        else {
            throw PagedKVKernelGateError.badConfig(url.path)
        }
        return AttentionShape(
            modelName: modelName, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim
        )
    }
}

nonisolated enum PagedKVKernelGateError: Error, CustomStringConvertible {
    case badConfig(String)
    case modelMissing(String)
    case numericMismatch(String)
    case gateFailed(String)

    var description: String {
        switch self {
        case .badConfig(let path):
            return "config.json unreadable or missing attention shape: \(path)"
        case .modelMissing(let message):
            return message
        case .numericMismatch(let message):
            return "paged attention numerics diverged from contiguous: \(message)"
        case .gateFailed(let message):
            return "kernel gate FAILED pre-registered threshold: \(message)"
        }
    }
}

/// The harness: builds fixtures, times the three variants per scenario,
/// validates numerics, writes the JSON report, and throws when the
/// pre-registered gate fails (the harness wrapper turns that into exit 1).
@MainActor
final class PagedKVKernelGateRunner {

    /// The two gate models (decision record on #151 / ADR-0023): one small
    /// production model and ornith-35b. Directory names under the app's model
    /// storage; overridable for machines with different downloads.
    private static let defaultSmallDirectory = "z-lab_Qwen3.5-4B-PARO"
    private static let defaultLargeDirectory = "leonsarmiento_Ornith-1.0-35B-4bit-mlx"

    private var logHandle: FileHandle?

    func run() async throws {
        setupLogging()
        defer { try? logHandle?.close() }

        let parameters = LLMActor.makeGenerateParameters(from: AgentGenerateParameters())
        let kvBits = parameters.kvBits ?? 8
        let kvGroupSize = parameters.kvGroupSize
        log("Paged-KV kernel gate — kvBits=\(kvBits) kvGroupSize=\(kvGroupSize)")

        let shapes = try [
            resolveShape(
                envKey: "TESSERACT_GATE_SMALL_DIR", fallback: Self.defaultSmallDirectory),
            resolveShape(
                envKey: "TESSERACT_GATE_LARGE_DIR", fallback: Self.defaultLargeDirectory),
        ]
        for shape in shapes {
            log(
                "model=\(shape.modelName) nHeads=\(shape.nHeads) "
                    + "nKVHeads=\(shape.nKVHeads) headDim=\(shape.headDim)")
        }

        var scenarios: [PagedKVKernelGateSupport.Scenario] = []
        for context in PagedKVKernelGateSupport.singleLaneContexts {
            for pageSize in PagedKVKernelGateSupport.pageSizes {
                scenarios.append(.init(laneContexts: [context], pageSize: pageSize))
            }
        }
        for pageSize in PagedKVKernelGateSupport.pageSizes {
            scenarios.append(
                .init(laneContexts: PagedKVKernelGateSupport.fourLaneMix, pageSize: pageSize))
        }

        var measurements: [PagedKVKernelGateSupport.Measurement] = []
        for shape in shapes {
            for scenario in scenarios {
                let result = try measureScenario(
                    shape: shape, scenario: scenario, kvBits: kvBits, kvGroupSize: kvGroupSize
                )
                measurements.append(contentsOf: result)
                for m in result {
                    log(
                        "\(shape.modelName) \(scenario.id) \(m.variant): "
                            + String(format: "%.3f ms/round", m.msPerDecodeRound)
                            + String(format: " overhead=%+.1f%%", m.overheadVsContiguous * 100)
                            + String(format: " maxAbsDiff=%.4f", m.maxAbsDiffVsContiguous))
                }
                Memory.clearCache()
            }
        }

        let verdict = Self.verdict(measurements: measurements)
        for cell in verdict.cells {
            log(
                "GATE \(cell.model) N=\(cell.lanes): best=\(cell.bestVariant) "
                    + "P=\(cell.bestPageSize) "
                    + String(format: "overhead=%+.1f%%", cell.bestOverhead * 100)
                    + " → \(cell.passed ? "PASS" : "FAIL")")
        }
        log("Kernel gate verdict: \(verdict.passed ? "PASS" : "FAIL")")

        try writeReport(shapes: shapes, measurements: measurements, verdict: verdict)

        if !verdict.passed {
            let failing = verdict.cells.filter { !$0.passed }
                .map { "\($0.model) N=\($0.lanes)" }
                .joined(separator: ", ")
            throw PagedKVKernelGateError.gateFailed(failing)
        }
    }

    // MARK: - Measurement

    /// One (shape, scenario) cell: builds the quantized KV fixture once,
    /// then times the three variants over full decode rounds (every lane's
    /// attention built, one eval per round — the engine's per-round shape).
    private func measureScenario(
        shape: PagedKVKernelGateSupport.AttentionShape,
        scenario: PagedKVKernelGateSupport.Scenario,
        kvBits: Int,
        kvGroupSize: Int
    ) throws -> [PagedKVKernelGateSupport.Measurement] {
        let pageSize = scenario.pageSize
        let scale = 1.0 / Float(shape.headDim).squareRoot()
        let key = MLXRandom.key(0xBA7C4)

        // Per-lane logical KV plus queries.
        struct LaneFixture {
            let context: Int
            let queries: MLXArray
            let keys: (MLXArray, MLXArray, MLXArray?)
            let values: (MLXArray, MLXArray, MLXArray?)
            /// Logical page ids into the shared pool, in token order.
            let logicalPages: [Int]
        }

        var lanes: [LaneFixture] = []
        var nextLogicalPage = 0
        var laneKeys = key
        for context in scenario.laneContexts {
            precondition(context % pageSize == 0, "contexts must be page-aligned")
            let (a, b) = MLXRandom.split(key: laneKeys)
            laneKeys = a
            let (kKey, rest) = MLXRandom.split(key: b)
            let (vKey, qKey) = MLXRandom.split(key: rest)
            let k = MLXRandom.normal(
                [1, shape.nKVHeads, context, shape.headDim], key: kKey
            ).asType(.float16)
            let v = MLXRandom.normal(
                [1, shape.nKVHeads, context, shape.headDim], key: vKey
            ).asType(.float16)
            let q = MLXRandom.normal(
                [1, shape.nHeads, 1, shape.headDim], key: qKey
            ).asType(.float16)
            let kq = quantized(k, groupSize: kvGroupSize, bits: kvBits)
            let vq = quantized(v, groupSize: kvGroupSize, bits: kvBits)
            let pages = context / pageSize
            lanes.append(
                LaneFixture(
                    context: context,
                    queries: q,
                    keys: (kq.wq, kq.scales, kq.biases),
                    values: (vq.wq, vq.scales, vq.biases),
                    logicalPages: Array(nextLogicalPage..<(nextLogicalPage + pages))
                ))
            nextLogicalPage += pages
        }
        let totalPages = nextLogicalPage

        // Shared pools. Layout A (gather-then-attend): [nKV, pages, P, cols],
        // physical page order shuffled. Layout B (fused): expert axis
        // [pages * nKV, P, cols], expert = physicalPage * nKV + kvHead.
        let placement = PagedKVKernelGateSupport.pagePlacement(
            totalPages: totalPages, seed: 0x51_0CA1)
        var inverse = [Int](repeating: 0, count: totalPages)
        for (logical, physical) in placement.enumerated() { inverse[physical] = logical }
        let inverseIndices = MLXArray(inverse.map(Int32.init))

        func poolA(_ slices: [MLXArray]) -> MLXArray {
            // slices: per lane [1, nKV, T, cols] → logical page axis, then
            // permute to physical order.
            let paged = slices.map { lane in
                lane.reshaped([
                    shape.nKVHeads, lane.dim(2) / pageSize, pageSize, lane.dim(3),
                ])
            }
            let logical = concatenated(paged, axis: 1)
            return take(logical, inverseIndices, axis: 1)
        }
        func poolB(_ pool: MLXArray) -> MLXArray {
            // [nKV, pages, P, cols] → [pages * nKV, P, cols]
            pool.transposed(1, 0, 2, 3).reshaped([
                totalPages * shape.nKVHeads, pageSize, pool.dim(3),
            ])
        }

        let kqA = poolA(lanes.map(\.keys.0))
        let ksA = poolA(lanes.map(\.keys.1))
        let kbA = poolA(lanes.map { $0.keys.2! })
        let vqA = poolA(lanes.map(\.values.0))
        let vsA = poolA(lanes.map(\.values.1))
        let vbA = poolA(lanes.map { $0.values.2! })
        let kqB = poolB(kqA)
        let ksB = poolB(ksA)
        let kbB = poolB(kbA)
        let vqB = poolB(vqA)
        let vsB = poolB(vsA)
        let vbB = poolB(vbA)
        eval(kqA, ksA, kbA, vqA, vsA, vbA, kqB, ksB, kbB, vqB, vsB, vbB)

        let physicalPages = lanes.map { $0.logicalPages.map { placement[$0] } }
        let laneIndicesA = physicalPages.map { MLXArray($0.map(Int32.init)) }
        let laneIndicesB = physicalPages.map { pages in
            MLXArray(
                pages.flatMap { page in
                    (0..<shape.nKVHeads).map { Int32(page * shape.nKVHeads + $0) }
                })
        }

        // The three round builders. Each returns one output per lane.
        func contiguousRound() -> [MLXArray] {
            lanes.map { lane in
                quantizedScaledDotProductAttention(
                    queries: lane.queries,
                    quantizedKeys: lane.keys,
                    quantizedValues: lane.values,
                    scale: scale,
                    mask: .none,
                    groupSize: kvGroupSize,
                    bits: kvBits
                )
            }
        }

        func gatherThenAttendRound() -> [MLXArray] {
            lanes.enumerated().map { index, lane in
                let idx = laneIndicesA[index]
                func gather(_ pool: MLXArray) -> MLXArray {
                    take(pool, idx, axis: 1)
                        .reshaped([1, shape.nKVHeads, lane.context, pool.dim(3)])
                }
                return quantizedScaledDotProductAttention(
                    queries: lane.queries,
                    quantizedKeys: (gather(kqA), gather(ksA), gather(kbA)),
                    quantizedValues: (gather(vqA), gather(vsA), gather(vbA)),
                    scale: scale,
                    mask: .none,
                    groupSize: kvGroupSize,
                    bits: kvBits
                )
            }
        }

        func fusedRound() -> [MLXArray] {
            lanes.enumerated().map { index, lane in
                let idx = laneIndicesB[index]
                let lanePages = lane.context / pageSize
                let experts = lanePages * shape.nKVHeads
                // Scaled queries grouped per kv head, tiled per page:
                // [experts, groupHeads, headDim].
                let grouped = (lane.queries * scale)
                    .reshaped([shape.nKVHeads, shape.groupHeads, shape.headDim])
                let x = broadcast(
                    grouped.expandedDimensions(axis: 0),
                    to: [lanePages, shape.nKVHeads, shape.groupHeads, shape.headDim]
                ).reshaped([experts, shape.groupHeads, shape.headDim])
                var scores = gatherQuantizedMM(
                    x, kqB, scales: ksB, biases: kbB,
                    rhsIndices: idx, transpose: true,
                    groupSize: kvGroupSize, bits: kvBits
                )
                scores =
                    scores
                    .reshaped([lanePages, shape.nKVHeads, shape.groupHeads, pageSize])
                    .transposed(1, 2, 0, 3)
                    .reshaped([shape.nKVHeads, shape.groupHeads, lane.context])
                let attention = softmax(scores, axis: -1)
                let weights =
                    attention
                    .reshaped([shape.nKVHeads, shape.groupHeads, lanePages, pageSize])
                    .transposed(2, 0, 1, 3)
                    .reshaped([experts, shape.groupHeads, pageSize])
                let partial = gatherQuantizedMM(
                    weights, vqB, scales: vsB, biases: vbB,
                    rhsIndices: idx, transpose: false,
                    groupSize: kvGroupSize, bits: kvBits
                )
                return
                    partial
                    .reshaped([lanePages, shape.nKVHeads, shape.groupHeads, shape.headDim])
                    .sum(axis: 0)
                    .reshaped([1, shape.nHeads, 1, shape.headDim])
            }
        }

        // Numeric validation before timing: paged variants must reproduce the
        // contiguous output (identical quantized inputs; only accumulation
        // order differs).
        let reference = contiguousRound()
        eval(reference)
        var diffs: [PagedKVKernelGateSupport.Variant: Double] = [:]
        var finite: [PagedKVKernelGateSupport.Variant: Bool] = [:]
        diffs[.contiguous] = 0
        finite[.contiguous] = reference.allSatisfy { output in
            MLX.all(MLX.isFinite(output.asType(.float32))).item(Bool.self)
        }
        for (variant, round) in [
            (PagedKVKernelGateSupport.Variant.gatherThenAttend, gatherThenAttendRound),
            (PagedKVKernelGateSupport.Variant.fusedGatherQMM, fusedRound),
        ] {
            let outputs = round()
            eval(outputs)
            var maxDiff = 0.0
            var allFinite = true
            for (out, ref) in zip(outputs, reference) {
                let diff = MLX.abs(out.asType(.float32) - ref.asType(.float32)).max()
                    .item(Float.self)
                maxDiff = max(maxDiff, Double(diff))
                allFinite =
                    allFinite
                    && MLX.all(MLX.isFinite(out.asType(.float32))).item(Bool.self)
            }
            diffs[variant] = maxDiff
            finite[variant] = allFinite
            if maxDiff > 0.05 || !allFinite {
                throw PagedKVKernelGateError.numericMismatch(
                    "\(shape.modelName) \(scenario.id) \(variant.rawValue): "
                        + "maxAbsDiff=\(maxDiff) finite=\(allFinite)")
            }
        }

        func time(_ round: () -> [MLXArray]) -> Double {
            for _ in 0..<PagedKVKernelGateSupport.warmupRounds {
                eval(round())
            }
            let started = Date.timeIntervalSinceReferenceDate
            for _ in 0..<PagedKVKernelGateSupport.timedRounds {
                eval(round())
            }
            let elapsed = Date.timeIntervalSinceReferenceDate - started
            return elapsed / Double(PagedKVKernelGateSupport.timedRounds) * 1000
        }

        let baseMs = time(contiguousRound)
        let gatherMs = time(gatherThenAttendRound)
        let fusedMs = time(fusedRound)

        func measurement(
            _ variant: PagedKVKernelGateSupport.Variant, _ ms: Double
        ) -> PagedKVKernelGateSupport.Measurement {
            PagedKVKernelGateSupport.Measurement(
                model: shape.modelName,
                scenarioID: scenario.id,
                laneContexts: scenario.laneContexts,
                pageSize: pageSize,
                variant: variant.rawValue,
                msPerDecodeRound: ms,
                overheadVsContiguous: variant == .contiguous ? 0 : (ms - baseMs) / baseMs,
                maxAbsDiffVsContiguous: diffs[variant] ?? 0,
                outputsFinite: finite[variant] ?? false
            )
        }

        return [
            measurement(.contiguous, baseMs),
            measurement(.gatherThenAttend, gatherMs),
            measurement(.fusedGatherQMM, fusedMs),
        ]
    }

    // MARK: - Verdict

    /// The pre-registered gate, applied per (model, lane count): the best
    /// paged candidate — any variant, any page size, worst context point —
    /// must stay within the threshold.
    static func verdict(
        measurements: [PagedKVKernelGateSupport.Measurement]
    ) -> PagedKVKernelGateSupport.GateVerdict {
        var cells: [PagedKVKernelGateSupport.GateVerdict.Cell] = []
        let models = Array(Set(measurements.map(\.model))).sorted()
        for model in models {
            for lanes in [1, 4] {
                let paged = measurements.filter {
                    $0.model == model && $0.laneContexts.count == lanes
                        && $0.variant != PagedKVKernelGateSupport.Variant.contiguous.rawValue
                }
                guard !paged.isEmpty else { continue }
                // Best (variant, pageSize) by its WORST context point, so one
                // fast short-context cell can't hide a slow long-context one.
                struct Candidate: Hashable {
                    let variant: String
                    let pageSize: Int
                }
                var worstByCandidate: [Candidate: Double] = [:]
                for m in paged {
                    let candidate = Candidate(variant: m.variant, pageSize: m.pageSize)
                    worstByCandidate[candidate] = max(
                        worstByCandidate[candidate] ?? -.infinity, m.overheadVsContiguous)
                }
                guard
                    let best = worstByCandidate.min(by: { $0.value < $1.value })
                else { continue }
                cells.append(
                    .init(
                        model: model,
                        lanes: lanes,
                        bestVariant: best.key.variant,
                        bestPageSize: best.key.pageSize,
                        bestOverhead: best.value,
                        passed: best.value <= PagedKVKernelGateSupport.thresholdOverhead
                    ))
            }
        }
        return PagedKVKernelGateSupport.GateVerdict(
            thresholdOverhead: PagedKVKernelGateSupport.thresholdOverhead,
            cells: cells,
            passed: !cells.isEmpty && cells.allSatisfy(\.passed)
        )
    }

    // MARK: - Plumbing

    private func resolveShape(
        envKey: String, fallback: String
    ) throws -> PagedKVKernelGateSupport.AttentionShape {
        let name = ProcessInfo.processInfo.environment[envKey] ?? fallback
        let directory = ModelDownloadManager.modelStorageURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw PagedKVKernelGateError.modelMissing(
                "gate model directory not found: \(directory.path) "
                    + "(override with \(envKey))")
        }
        return try PagedKVKernelGateSupport.readShape(
            modelDirectory: directory, modelName: name)
    }

    private var reportDir: URL {
        DebugPaths.benchmark.appendingPathComponent("paged-kv-kernel-gate")
    }

    private func setupLogging() {
        try? FileManager.default.createDirectory(
            at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logHandle = try? FileHandle(forWritingTo: logURL)
    }

    private func log(_ line: String) {
        Log.agent.info("[paged-kv-gate] \(line)")
        logHandle?.write(Data((line + "\n").utf8))
    }

    private func writeReport(
        shapes: [PagedKVKernelGateSupport.AttentionShape],
        measurements: [PagedKVKernelGateSupport.Measurement],
        verdict: PagedKVKernelGateSupport.GateVerdict
    ) throws {
        struct Report: Codable {
            let date: String
            let hardware: String
            let kvBits: Int
            let kvGroupSize: Int
            let warmupRounds: Int
            let timedRounds: Int
            let shapes: [PagedKVKernelGateSupport.AttentionShape]
            let measurements: [PagedKVKernelGateSupport.Measurement]
            let verdict: PagedKVKernelGateSupport.GateVerdict
        }
        let parameters = LLMActor.makeGenerateParameters(from: AgentGenerateParameters())
        var hardware = utsname()
        uname(&hardware)
        let machine = withUnsafeBytes(of: &hardware.machine) { bytes in
            String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            hardware: "\(machine) / \(ProcessInfo.processInfo.physicalMemory >> 30) GiB",
            kvBits: parameters.kvBits ?? 8,
            kvGroupSize: parameters.kvGroupSize,
            warmupRounds: PagedKVKernelGateSupport.warmupRounds,
            timedRounds: PagedKVKernelGateSupport.timedRounds,
            shapes: shapes,
            measurements: measurements,
            verdict: verdict
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        try encoder.encode(report).write(
            to: reportDir.appendingPathComponent("paged_kv_kernel_gate_\(stamp).json"),
            options: .atomic)
    }
}
