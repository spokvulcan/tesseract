import Foundation
import Testing

@testable import Tesseract_Agent

/// Slice #83/#84 (PRD #82): the per-completion trace record, its block
/// digests, the durable trace log, and the measured-seconds estimators.
/// All hermetic — temp dirs only, no shared state.
struct CompletionTraceTests {

    // MARK: - Block digests

    @Test func blockDigestsAreDeterministicAndPrefixStable() {
        let blockSize = 4
        let base = Array(0..<20)
        let digestsA = TraceBlockDigest.cumulativeDigests(
            forKeyPath: base, blockSize: blockSize
        )
        let digestsB = TraceBlockDigest.cumulativeDigests(
            forKeyPath: base, blockSize: blockSize
        )
        #expect(digestsA == digestsB)
        #expect(digestsA.count == 5)

        // A sequence sharing the first 8 tokens shares exactly the
        // first 2 digests, then diverges forever after.
        var forked = base
        forked[9] = 999
        let forkedDigests = TraceBlockDigest.cumulativeDigests(
            forKeyPath: forked, blockSize: blockSize
        )
        #expect(Array(forkedDigests.prefix(2)) == Array(digestsA.prefix(2)))
        #expect(forkedDigests[2] != digestsA[2])
        #expect(forkedDigests[3] != digestsA[3])
    }

    @Test func blockDigestsOmitPartialTailBlock() {
        let digests = TraceBlockDigest.cumulativeDigests(
            forKeyPath: Array(0..<7), blockSize: 4
        )
        #expect(digests.count == 1)
        #expect(TraceBlockDigest.cumulativeDigests(forKeyPath: [], blockSize: 4).isEmpty)
    }

    // MARK: - Record assembly

    private func makeRecord(
        unkeyedReason: CacheKeySpace.UnkeyedReason? = nil
    ) -> CompletionTraceRecord? {
        CompletionTraceRecord.make(
            timestamp: 1000,
            requestID: UUID(),
            modelID: "test-model",
            partitionDigest: "cafe0123",
            unkeyedReason: unkeyedReason,
            keyPath: Array(0..<600),
            admittedCheckpoints: [
                TraceAdmittedSnapshot(offset: 128, bytes: 4096, checkpointType: "system")
            ],
            admittedLeaf: TraceAdmittedSnapshot(
                offset: 600, bytes: 8192, checkpointType: "leaf"
            ),
            ramBudgetBytes: 1_000_000,
            restoredOffset: 128,
            restoredFromSSD: true,
            hitTokens: 128,
            sharedPrefixLength: 200,
            lookupSeconds: 0.01,
            restoreSeconds: 0.02,
            hydrationSeconds: 0.5,
            prefillSeconds: 1.0,
            residualPromptSeconds: 0.05,
            terminalEvictionCount: 2,
            recoveredEvictionCount: 3,
            deviceEstimates: MeasuredSecondsEstimates()
        )
    }

    @Test func unkeyedCompletionsProduceNoRecord() {
        #expect(makeRecord(unkeyedReason: .unrecognizedPlaceholderFamily) == nil)
        #expect(makeRecord() != nil)
    }

    @Test func recordCarriesTTFTIdentityAndDigests() throws {
        let record = try #require(makeRecord())
        #expect(abs(record.ttftSeconds - (0.01 + 0.02 + 1.0 + 0.05)) < 1e-12)
        #expect(record.promptTokenCount == 600)
        // 600 tokens at the production block size (256) → 2 full blocks.
        #expect(record.prefixBlockDigests.count == 600 / TraceBlockDigest.blockSize)

        // Codable round-trip is lossless.
        let data = try JSONEncoder().encode(record)
        let decoded = try JSONDecoder().decode(CompletionTraceRecord.self, from: data)
        #expect(decoded == record)
    }

    // MARK: - Trace log

    private func makeTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("trace-log-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func traceLogWritesHeaderThenRecordsAndReadsBack() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let log = CompletionTraceLog(directory: directory)
        let first = try #require(makeRecord())
        let second = try #require(makeRecord())
        log.append(first)
        log.append(second)
        log.flushForTesting()

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        let traceFile = try #require(files.first { $0.lastPathComponent.hasSuffix(".jsonl") })

        let lines = CompletionTraceLog.readLines(at: traceFile)
        #expect(lines.count == 3)
        guard case .header(let header) = try #require(lines.first) else {
            Issue.record("first line is not a header")
            return
        }
        #expect(header.schemaVersion == CompletionTraceRecord.currentSchemaVersion)
        #expect(header.blockSize == TraceBlockDigest.blockSize)

        let records = CompletionTraceLog.readRecords(at: [traceFile])
        #expect(records == [first, second])
    }

    /// `traceFiles` is the corpus: a size-cap rotation parks a day's
    /// earlier records in `.jsonl.old`, and the replay harness must see
    /// them — ordered before that day's current file, between days by
    /// day. Dropping rotated files would silently shrink the corpus on
    /// exactly the busiest days.
    @Test func traceFilesIncludeRotatedOldFilesInDayOrder() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        for name in [
            "trace-2026-06-11.jsonl",
            "trace-2026-06-10.jsonl",
            "trace-2026-06-11.jsonl.old",
            "unrelated.jsonl",
            "trace-2026-06-12.txt",
        ] {
            FileManager.default.createFile(
                atPath: directory.appendingPathComponent(name).path, contents: nil
            )
        }

        let files = CompletionTraceLog.traceFiles(in: directory)
        #expect(files.map(\.lastPathComponent) == [
            "trace-2026-06-10.jsonl",
            "trace-2026-06-11.jsonl.old",
            "trace-2026-06-11.jsonl",
        ])
    }

    @Test func traceLogReaderRejectsForeignSchema() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Hand-write a file whose header claims a future schema.
        let url = directory.appendingPathComponent("trace-fake.jsonl")
        let header = CompletionTraceLine.header(CompletionTraceHeader(
            schemaVersion: CompletionTraceRecord.currentSchemaVersion + 1,
            blockSize: TraceBlockDigest.blockSize,
            createdAt: 0
        ))
        let record = CompletionTraceLine.record(try #require(makeRecord()))
        let encoder = JSONEncoder()
        var data = try encoder.encode(header)
        data.append(0x0A)
        data.append(try encoder.encode(record))
        data.append(0x0A)
        try data.write(to: url)

        #expect(CompletionTraceLog.readRecords(at: [url]).isEmpty)
    }

    @Test func traceLogSurvivesTornTrailingLine() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let log = CompletionTraceLog(directory: directory)
        let record = try #require(makeRecord())
        log.append(record)
        log.flushForTesting()

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        let traceFile = try #require(files.first { $0.lastPathComponent.hasSuffix(".jsonl") })
        // Simulate a crash mid-append: garbage tail bytes.
        let handle = try FileHandle(forWritingTo: traceFile)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("{\"kind\":\"rec".utf8))
        try handle.close()

        #expect(CompletionTraceLog.readRecords(at: [traceFile]) == [record])
    }

    // MARK: - Measured-seconds estimators

    @Test func estimatesStartFromDocumentedColdStartDefaults() {
        let estimates = MeasuredSecondsEstimates()
        #expect(estimates.prefillFlopsPerSecond
            == MeasuredSecondsEstimates.defaultPrefillFlopsPerSecond)
        #expect(estimates.hydrationBytesPerSecond
            == MeasuredSecondsEstimates.defaultHydrationBytesPerSecond)
        #expect(estimates.prefillSampleCount == 0)
        #expect(estimates.hydrationSampleCount == 0)
    }

    @Test func firstSampleReplacesDefaultLaterSamplesAreDamped() {
        let afterFirst = MeasuredSecondsEstimates()
            .recordingPrefill(flops: 2.0e12, seconds: 1.0)
        #expect(afterFirst.prefillFlopsPerSecond == 2.0e12)
        #expect(afterFirst.prefillSampleCount == 1)

        // Second sample at 4e12 blends at the EWMA weight, never jumps.
        let afterSecond = afterFirst.recordingPrefill(flops: 4.0e12, seconds: 1.0)
        let expected = 2.0e12 * (1 - MeasuredSecondsEstimates.sampleWeight)
            + 4.0e12 * MeasuredSecondsEstimates.sampleWeight
        #expect(abs(afterSecond.prefillFlopsPerSecond - expected) < 1e3)

        let hydrated = MeasuredSecondsEstimates()
            .recordingHydration(bytes: 3_000_000_000, seconds: 2.0)
        #expect(hydrated.hydrationBytesPerSecond == 1.5e9)
        #expect(hydrated.hydrationSampleCount == 1)
    }

    @Test func estimatesIgnoreNoiseSamples() {
        let estimates = MeasuredSecondsEstimates()
            .recordingPrefill(flops: 1e12, seconds: 0)
            .recordingPrefill(flops: 0, seconds: 1)
            .recordingPrefill(flops: 1e12, seconds: 1e-6)
            .recordingHydration(bytes: 0, seconds: 1)
            .recordingHydration(bytes: 100, seconds: 0)
        #expect(estimates == MeasuredSecondsEstimates())
    }
}
