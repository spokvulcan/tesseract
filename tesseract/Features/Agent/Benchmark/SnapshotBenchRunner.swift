import Foundation
import MLX
import MLXLMCommon
import os

/// Snapshot copy benchmark (`--snapshot-bench`) — measures
/// `HybridCacheSnapshot.capture`/`restore` copy cost end-to-end at realistic
/// snapshot sizes, interleaving the two copy strategies (ABBA within one
/// process, per the #258 protocol — thermals otherwise dominate).
///
/// No model load: the copy mechanism is shape/dtype-driven, so a synthetic
/// cache stack reproduces the 35B-A3B's snapshot exactly (10 full-attention
/// KV layers + 30 GDN MambaCache layers at the config.json dims). Reports
/// capture/restore milliseconds, peak GB per measurement, and verifies the
/// device copy is byte-identical to the host path it replaces.
@MainActor
final class SnapshotBenchRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("snapshot-bench")

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    private struct Record: Codable {
        let contextTokens: Int
        let strategy: String
        let round: Int
        let captureMs: Double
        let restoreMs: Double
        let snapshotMB: Double
        let peakGB: Double
    }

    func run() async throws {
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)

        let contexts = [8_192, 32_768]
        let rounds = 4
        let strategies: [(String, HybridCacheSnapshot.CopyStrategy)] = [
            ("host", .host),
            ("device", .device),
        ]

        var records: [Record] = []
        var byteCheckFailures = 0

        for tokens in contexts {
            // One synthetic stack per context, shared by both strategies and
            // every iteration (source of all copies — never mutated).
            let caches = Self.makeSyntheticCaches(tokens: tokens)
            eval(caches.flatMap { $0.state })

            // Byte-equality gate: device capture vs host capture of the same
            // source must produce bit-identical snapshot arrays.
            if let devSnap = HybridCacheSnapshot.capture(
                cache: caches, offset: tokens, type: .leaf, copyStrategy: .device),
                let hostSnap = HybridCacheSnapshot.capture(
                    cache: caches, offset: tokens, type: .leaf, copyStrategy: .host)
            {
                for (d, h) in zip(devSnap.layers, hostSnap.layers) {
                    for (da, ha) in zip(d.state, h.state) {
                        if !(da .== ha).all().item(Bool.self) {
                            byteCheckFailures += 1
                        }
                    }
                }
            }
            log(
                "ctx=\(tokens): byte-equality device-vs-host "
                    + (byteCheckFailures == 0 ? "IDENTICAL" : "FAILED (\(byteCheckFailures))"))

            for round in 0..<rounds {
                for (name, strategy) in round % 2 == 0 ? strategies : strategies.reversed() {
                    Memory.peakMemory = 0
                    var captureMs = 0.0
                    var restoreMs = 0.0
                    var snapshotMB = 0.0
                    for _ in 0..<3 {
                        let t0 = ContinuousClock.now
                        guard
                            let snapshot = HybridCacheSnapshot.capture(
                                cache: caches, offset: tokens, type: .leaf, copyStrategy: strategy)
                        else { continue }
                        captureMs += snapshotBenchSeconds(t0.duration(to: .now)) * 1e3
                        snapshotMB = Double(snapshot.memoryBytes) / 1e6
                        let t1 = ContinuousClock.now
                        let restored = try snapshot.restore(copyStrategy: strategy)
                        restoreMs += snapshotBenchSeconds(t1.duration(to: .now)) * 1e3
                        eval(restored.flatMap { $0.state })
                    }
                    let peakGB = Double(Memory.peakMemory) / 1e9
                    let record = Record(
                        contextTokens: tokens,
                        strategy: name,
                        round: round,
                        captureMs: captureMs / 3,
                        restoreMs: restoreMs / 3,
                        snapshotMB: snapshotMB,
                        peakGB: peakGB
                    )
                    records.append(record)
                    log(
                        "ctx=\(tokens) \(name) round=\(round): "
                            + "capture=\(String(format: "%.2f", record.captureMs))ms "
                            + "restore=\(String(format: "%.2f", record.restoreMs))ms "
                            + "snapshot=\(String(format: "%.0f", record.snapshotMB))MB "
                            + "peak=\(String(format: "%.2f", record.peakGB))GB"
                    )
                }
            }
        }

        struct Report: Codable {
            let date: String
            let byteCheckFailures: Int
            let records: [Record]
        }
        let stamp = ISO8601DateFormatter().string(from: Date())
        let report = Report(date: stamp, byteCheckFailures: byteCheckFailures, records: records)
        let url = reportDir.appendingPathComponent("snapshot_bench_\(stamp).json")
        try JSONEncoder().encode(report).write(to: url, options: .atomic)
        log("Report written to: \(url.path)")
        log(byteCheckFailures == 0 ? "Overall: PASS" : "Overall: FAIL")
        logFileHandle?.closeFile()
        if byteCheckFailures > 0 {
            throw NSError(domain: "SnapshotBench", code: 1)
        }
    }

    /// 35B-A3B-shaped cache stack at `tokens` context: 10 full-attention KV
    /// layers (2 KV heads × 256 head_dim, f16) + 30 GDN layers (conv state
    /// [1,3,8192] f16 + recurrent state [1,32,128,128] f32), dims from
    /// config.json.
    private static func makeSyntheticCaches(tokens: Int) -> [any KVCache] {
        var caches: [any KVCache] = []
        for _ in 0..<10 {
            let cache = KVCacheSimple()
            cache.state = [
                (MLXRandom.normal([1, 2, tokens, 256]) * 0.1).asType(.float16),
                (MLXRandom.normal([1, 2, tokens, 256]) * 0.1).asType(.float16),
            ]
            cache.offset = tokens
            caches.append(cache)
        }
        for _ in 0..<30 {
            let cache = MambaCache()
            cache.state = [
                (MLXRandom.normal([1, 3, 8192]) * 0.1).asType(.float16),
                MLXRandom.normal([1, 32, 128, 128]).asType(.float32),
            ]
            cache.offset = tokens
            caches.append(cache)
        }
        return caches
    }

    private func log(_ message: String) {
        logger.info("\(message, privacy: .public)")
        if let data = (message + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }
}

private func snapshotBenchSeconds(_ duration: Duration) -> Double {
    let c = duration.components
    return Double(c.seconds) + Double(c.attoseconds) * 1e-18
}
