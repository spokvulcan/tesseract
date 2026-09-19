import CryptoKit
import Foundation
import MLX
import MLXLMCommon

/// Owner-run #532 experiment. Uses the existing SSD store and Model Session
/// boundaries; never generates a prompt or changes the live prefix-cache tier.
@MainActor
final class SSDReadBenchRunner {
    private let runner: BenchmarkRunner

    init(runner: BenchmarkRunner) { self.runner = runner }

    nonisolated private struct Plan: Codable, Sendable {
        let ownerApproval: String
        let cacheRoot: String
        let snapshotID: String
        /// Exact relative path -> SHA-256 set approved before the run.
        let segmentSHA256: [String: String]
        let maxSegmentBytes: Int
        let maxMLXBytes: Int
    }

    nonisolated private struct Record: Codable, Sendable {
        let block: Int
        let arm: SSDSnapshotReadArm
        let fileBytes: Int
        let materializedBytes: Int
        let seconds: Double
        let peakMLXBytes: Int
        let snapshotSHA256: String
        let segments: [PromptCacheTelemetryEvent]

        var gigabytesPerSecond: Double { Double(materializedBytes) / seconds / 1e9 }
    }

    nonisolated private final class SegmentEvents: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [PromptCacheTelemetryEvent] = []
        func append(_ event: PromptCacheTelemetryEvent) {
            lock.lock(); defer { lock.unlock() }
            events.append(event)
        }
        func take() -> [PromptCacheTelemetryEvent] {
            lock.lock(); defer { lock.unlock() }
            defer { events.removeAll() }
            return events
        }
    }

    func run() async throws {
        let args = CommandLine.arguments
        guard let flag = args.firstIndex(of: "--ssd-read-plan"), flag + 1 < args.count else {
            throw failure("Pass --ssd-read-plan with the owner-approved JSON plan")
        }
        let planData = try Data(contentsOf: URL(fileURLWithPath: args[flag + 1]))
        let plan = try JSONDecoder().decode(Plan.self, from: planData)
        guard !plan.ownerApproval.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            plan.maxSegmentBytes > 0, plan.maxMLXBytes > 0
        else { throw failure("Owner approval and positive byte limits are required") }

        let source = URL(fileURLWithPath: plan.cacheRoot, isDirectory: true)
        let manifest = try JSONDecoder().decode(
            SnapshotManifest.self,
            from: Data(contentsOf: source.appendingPathComponent("manifest.json")))
        guard manifest.schemaVersion == SnapshotManifestSchema.currentVersion,
            let descriptor = manifest.snapshots[plan.snapshotID],
            let partition = manifest.partitions[descriptor.partitionDigest],
            let checkpoint = HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType),
            Set(descriptor.chainFileRelativePaths) == Set(plan.segmentSHA256.keys),
            descriptor.totalBytes > 0, descriptor.totalBytes <= plan.maxSegmentBytes
        else { throw failure("Plan does not match one current, bounded Segment Chain") }
        let modelDir = try runner.resolveModelDirectory()
        guard
            try ModelFingerprint.computeFingerprint(modelDir: modelDir)
                == partition.modelFingerprint
        else { throw failure("Selected model does not match the Segment Chain fingerprint") }

        let reportDir = runner.activeConfig.outputDir
            .appendingPathComponent("ssd-read-\(UUID().uuidString)", isDirectory: true)
        let scratch = reportDir.appendingPathComponent("scratch", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try planData.write(to: reportDir.appendingPathComponent("approved-plan.json"))
        try stageChain(
            descriptor, partition: partition, source: source, scratch: scratch, plan: plan)

        let ref = SnapshotRef(
            snapshotID: descriptor.snapshotID,
            partitionDigest: descriptor.partitionDigest, tokenOffset: descriptor.tokenOffset,
            checkpointType: checkpoint, bytesOnDisk: descriptor.totalBytes)
        let config = SSDPrefixCacheConfig(
            enabled: true, rootURL: scratch,
            budgetBytes: plan.maxSegmentBytes, maxPendingBytes: 0)
        let stores = SSDSnapshotReadArm.allCases.map { arm in
            let store = SSDSnapshotStore(config: config, readArmForBenchmark: arm)
            _ = store.warmStartLoad(expectedFingerprint: partition.modelFingerprint)
            return (arm, store)
        }
        let model = LLMActor()
        do {
            // No SSD configuration, speculative drafter, or generation: only
            // this target model stays resident for all paired measurements.
            _ = try await model.loadModel(from: modelDir, visionMode: false, speculation: .off)
            let records = try await model.withModelContainer { container in
                try await container.perform { _ in
                    try Self.measure(
                        stores: stores, ref: ref, fingerprint: partition.modelFingerprint,
                        plan: plan, reportDir: reportDir)
                }
            }
            try writeReport(records, reportDir: reportDir)
            await model.unloadModel()
        } catch {
            await model.unloadModel()
            throw error
        }
    }

    /// Copy only the approved chain, never pass the owner's tier to a store
    /// whose ordinary hydration-failure cleanup may delete damaged backing.
    private func stageChain(
        _ descriptor: PersistedSnapshotDescriptor, partition: PartitionMeta,
        source: URL, scratch: URL, plan: Plan
    ) throws {
        var stagedBytes = 0
        for path in descriptor.chainFileRelativePaths {
            guard !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
                throw failure("Segment path escapes the source tier")
            }
            let input = source.appendingPathComponent(path)
            let output = scratch.appendingPathComponent(path)
            let size = try input.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size > 0, size <= plan.maxSegmentBytes - stagedBytes else {
                throw failure("Segment set exceeds the approved byte limit")
            }
            stagedBytes += size
            try FileManager.default.createDirectory(
                at: output.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: input, to: output)
            let handle = try FileHandle(forReadingFrom: output)
            defer { try? handle.close() }
            var hash = SHA256()
            while let chunk = try handle.read(upToCount: 8 * 1_024 * 1_024), !chunk.isEmpty {
                hash.update(data: chunk)
            }
            guard Self.hex(hash.finalize()) == plan.segmentSHA256[path] else {
                throw failure("Segment SHA-256 differs from the pre-registered plan: \(path)")
            }
        }
        guard stagedBytes == descriptor.totalBytes else {
            throw failure("Manifest byte total differs")
        }
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [descriptor.partitionDigest: partition],
            snapshots: [descriptor.snapshotID: descriptor])
        try JSONEncoder().encode(manifest).write(
            to: scratch.appendingPathComponent("manifest.json"))
    }

    nonisolated private static func measure(
        stores: [(SSDSnapshotReadArm, SSDSnapshotStore)],
        ref: SnapshotRef, fingerprint: String, plan: Plan, reportDir: URL
    ) throws -> [Record] {
        // One warmup per arm, then six balanced blocks (every permutation).
        // This is an SSD-only body hit; no claim of a cold OS page cache.
        let orders = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 0, 2], [0, 2, 1]]
        var expectedDigest: String?
        var records: [Record] = []
        for block in -1..<orders.count {
            for index in block < 0 ? [0, 1, 2] : orders[block] {
                Memory.clearCache()
                // Conservative allowance for host payload + MLX body + chain
                // composition. The owner also monitors process RSS/swap.
                guard ref.bytesOnDisk <= (plan.maxMLXBytes - Memory.activeMemory) / 3 else {
                    throw failure("Approved memory headroom exhausted before hydration")
                }
                Memory.peakMemory = 0
                let (arm, store) = stores[index]
                let events = SegmentEvents()
                let sink = PrefixCacheDiagnostics.addTelemetrySink { event in
                    if event.eventName == "ssdHydrateSegment", event.field("id") == ref.snapshotID {
                        events.append(event)
                    }
                }
                defer { PrefixCacheDiagnostics.removeTelemetrySink(sink) }
                let start = ContinuousClock.now
                guard
                    let snapshot = store.loadSync(
                        snapshotRef: ref, expectedFingerprint: fingerprint)
                else { throw failure("Hydration failed for \(arm.rawValue); stop the experiment") }
                eval(snapshot.layers.flatMap(\.state))
                Stream.gpu.synchronize()
                let seconds = start.duration(to: .now).seconds
                let peak = Memory.peakMemory
                guard peak <= plan.maxMLXBytes else { throw failure("Approved MLX peak exceeded") }
                let digest = snapshotDigest(snapshot)
                if let expectedDigest, expectedDigest != digest {
                    throw failure("Snapshot bytes or metadata differ across arms")
                }
                expectedDigest = digest
                let record = Record(
                    block: block, arm: arm, fileBytes: ref.bytesOnDisk,
                    materializedBytes: snapshot.memoryBytes, seconds: seconds, peakMLXBytes: peak,
                    snapshotSHA256: digest, segments: events.take())
                // Preserve completed observations even if a later arm fails.
                try JSONEncoder().encode(record).write(
                    to: reportDir.appendingPathComponent(
                        "block-\(block)-\(arm.rawValue).json"))
                if block >= 0 { records.append(record) }
            }
        }
        return records
    }

    nonisolated private static func snapshotDigest(_ snapshot: HybridCacheSnapshot) -> String {
        var hash = SHA256()
        hash.update(data: Data("offset=\(snapshot.tokenOffset)".utf8))
        for layer in snapshot.layers {
            hash.update(data: Data("\(layer.className)|\(layer.offset)|\(layer.metaState)".utf8))
            for array in layer.state {
                hash.update(data: Data("\(array.shape)|\(array.dtype)".utf8))
                hash.update(data: array.asData(access: .noCopy).data)
            }
        }
        return hex(hash.finalize())
    }

    private func writeReport(_ records: [Record], reportDir: URL) throws {
        func median(_ values: [Double]) -> Double {
            let values = values.sorted()
            return (values[values.count / 2 - 1] + values[values.count / 2]) / 2
        }
        let rates = SSDSnapshotReadArm.allCases.map { arm in
            (arm, median(records.filter { $0.arm == arm }.map(\.gigabytesPerSecond)))
        }
        let baseline = rates[0].1
        var report = "# SSD read experiment #532\n\n"
        report += "Owner plan: approved-plan.json. Model: \(runner.activeConfig.resolvedModelID).\n"
        report +=
            "Hardware: \(runner.resolvedHardwareDescription). Revision: \(runner.activeConfig.sourceRevision ?? "unrecorded").\n\n"
        report +=
            "Six balanced blocks, target loaded once, OS page cache warmed; throughput includes MLX evaluation.\n\n"
        report +=
            "| Arm | Median GB/s (decimal, materialized bytes) | Ratio to mapped |\n| --- | ---: | ---: |\n"
        for (arm, rate) in rates {
            report += String(format: "| %@ | %.3f | %.3f |\n", arm.rawValue, rate, rate / baseline)
        }
        let qualified = rates.dropFirst().filter { $0.1 / baseline >= 2 }
        report +=
            "\nQualifying arms at the pre-registered >=2x gate: \(qualified.map { $0.0.rawValue }).\n"
        report +=
            "Owner must assess memory/RSS, validity, and representative cache conditions before adoption. Production still uses mapped.\n"
        try report.write(
            to: reportDir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try JSONEncoder().encode(records).write(
            to: reportDir.appendingPathComponent("records.json"))
        Log.agent.notice("SSD read benchmark results: \(reportDir.path)")
    }

    nonisolated private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func failure(_ message: String) -> NSError {
        NSError(domain: "SSDReadBench", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    private func failure(_ message: String) -> NSError { Self.failure(message) }
}
