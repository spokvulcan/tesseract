import Foundation
import Testing

@testable import Tesseract_Agent

/// Slice #85 (PRD #82): the offline **trace-replay harness** — a pure
/// LRU-baseline simulation over the **Completion Trace Log**. The
/// committed fixture corpus tells an LRU-thrash story (two
/// conversations alternating under a 10 KB budget) and the golden test
/// pins its report exactly; the unit tests cover the block-granular
/// hit credit, leaf supersession, recency, and the
/// estimates-denominated TTFT-proxy.
struct TraceReplayHarnessTests {

    // MARK: - Helpers

    private var fixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/prefix-cache-trace-fixture.jsonl")
    }

    private func makeRecord(
        digests: [String],
        promptTokenCount: Int,
        partition: String = "p0",
        checkpoints: [TraceAdmittedSnapshot] = [],
        leaf: TraceAdmittedSnapshot? = nil,
        ramBudgetBytes: Int = 1_000_000,
        ttftSeconds: Double = 0.5,
        hitTokens: Int = 0,
        restoredFromSSD: Bool = false,
        hydrationSeconds: Double = 0,
        terminal: Int = 0,
        recovered: Int = 0,
        estimates: MeasuredSecondsEstimates? = MeasuredSecondsEstimates(),
        rewind: RewindTelemetry? = nil
    ) -> CompletionTraceRecord {
        CompletionTraceRecord(
            timestamp: 700_000_000,
            requestID: UUID(),
            modelID: "test-model",
            partitionDigest: partition,
            promptTokenCount: promptTokenCount,
            prefixBlockDigests: digests,
            admittedCheckpoints: checkpoints,
            admittedLeaf: leaf,
            ramBudgetBytes: ramBudgetBytes,
            restoredOffset: hitTokens,
            restoredFromSSD: restoredFromSSD,
            hitTokens: hitTokens,
            sharedPrefixLength: hitTokens,
            lookupSeconds: 0.001,
            restoreSeconds: 0,
            hydrationSeconds: hydrationSeconds,
            prefillSeconds: ttftSeconds - 0.001,
            residualPromptSeconds: 0,
            ttftSeconds: ttftSeconds,
            terminalEvictionCount: terminal,
            recoveredEvictionCount: recovered,
            deviceEstimates: estimates,
            rewind: rewind
        )
    }

    private func digestChain(_ prefix: String, _ count: Int) -> [String] {
        (1...count).map { "\(prefix)\($0)" }
    }

    // MARK: - Golden fixture

    /// The committed corpus decodes through the real reader (header
    /// gate included) and produces this exact report — byte-stable
    /// output is the whole point of the harness.
    @Test func goldenFixtureReport() {
        let records = CompletionTraceLog.readRecords(at: [fixtureURL])
        #expect(records.count == 6)

        let report = TraceReplayHarness.replay(records: records)
        #expect(report.recordCount == 6)
        #expect(report.blockSize == 256)

        // Observed: aggregated straight from the records' outcome
        // fields (all hand-checkable against the fixture).
        #expect(report.observed.ttftP50Seconds == 0.6)
        #expect(report.observed.ttftP95Seconds == 1.1)
        #expect(report.observed.totalHitTokens == 4000)
        #expect(report.observed.terminalEvictions == 2)
        #expect(report.observed.recoveredEvictions == 2)
        #expect(report.observed.ssdRestores == 2)
        #expect(abs(report.observed.totalHydrationSeconds - 0.56) < 1e-12)

        // Simulated LRU: r2 hits the r1 leaf at 5 verified blocks
        // (1280 tokens), r4 hits the r2 leaf at 7 (1792); the A/B
        // alternation under the 10 KB budget then thrashes — four
        // evictions (4000 + 3000 + 8000 + 3000 bytes), and r5/r6 both
        // miss what LRU just dropped.
        #expect(report.simulatedLRU.totalHitTokens == 3072)
        #expect(report.simulatedLRU.hitRequestCount == 2)
        #expect(report.simulatedLRU.evictionCount == 4)
        #expect(report.simulatedLRU.evictedBytes == 18000)

        #expect(report.observed.rewindEvents == 0)
        #expect(
            TraceReplayHarness.renderText(report) == """
                Trace Replay — LRU baseline (PRD #82, slice #85)
                records: 6 · block size: 256 tokens
                observed      ttft p50 0.6000s · p95 1.1000s · hit tokens 4000 · evictions 2 terminal / 2 recovered · ssd restores 2 · hydration 0.5600s
                rewinds       events 0 · size p50 0 · p95 0 · max 0 tokens
                simulated LRU ttft-proxy p50 3.9020s · p95 11.8633s · hit tokens 3072 · hit requests 2/6 · evictions 4 (all terminal — RAM-only baseline) · evicted bytes 18000
                """)
    }

    @Test func replayIsDeterministic() {
        let records = CompletionTraceLog.readRecords(at: [fixtureURL])
        let first = TraceReplayHarness.replay(records: records)
        let second = TraceReplayHarness.replay(records: records)
        #expect(first == second)
    }

    // MARK: - Rewind telemetry (issue #101)

    /// `RewindTelemetry.make` records a rewind only when the restore
    /// floor sits below the divergence, and never on a cold miss.
    @Test func rewindTelemetryDerivationFromLookupOutcome() {
        // The incident shape: divergence deep into the abandoned stretch,
        // floor at the restore base far below it.
        let rewind = RewindTelemetry.make(sharedPrefixLength: 87_495, restoredOffset: 41_897)
        #expect(rewind?.divergenceOffset == 87_495)
        #expect(rewind?.restoreFloor == 41_897)
        #expect(rewind?.rewindSize == 87_495 - 41_897)
        #expect(rewind?.isRewind == true)

        // A direct deep hit at the divergence: no rewind.
        let flat = RewindTelemetry.make(sharedPrefixLength: 5_000, restoredOffset: 5_000)
        #expect(flat?.isRewind == false)
        #expect(flat?.rewindSize == 0)

        // Cold miss: nothing to measure.
        #expect(RewindTelemetry.make(sharedPrefixLength: 5_000, restoredOffset: 0) == nil)
    }

    /// The harness rolls rewind events up over the corpus — the
    /// dashboard-facing regression signal. Non-rewind and rewind-less
    /// records are excluded from the percentile sample.
    @Test func replayAggregatesRewindEvents() {
        let records = [
            makeRecord(
                digests: digestChain("a", 4), promptTokenCount: 1100,
                rewind: RewindTelemetry(divergenceOffset: 1000, restoreFloor: 200)
            ),
            makeRecord(
                digests: digestChain("b", 4), promptTokenCount: 1100,
                rewind: RewindTelemetry(divergenceOffset: 1000, restoreFloor: 600)
            ),
            makeRecord(  // not a rewind — excluded
                digests: digestChain("c", 4), promptTokenCount: 1100,
                rewind: RewindTelemetry(divergenceOffset: 800, restoreFloor: 800)
            ),
            makeRecord(digests: digestChain("d", 4), promptTokenCount: 1100),  // no telemetry
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.observed.rewindEvents == 2)
        #expect(report.observed.maxRewindSizeTokens == 800)
        #expect(report.observed.rewindSizeP95Tokens == 800)
        #expect(report.observed.rewindSizeP50Tokens == 400)
    }

    /// A pre-#94 record (no `rewind` key) still decodes — the schema is
    /// unchanged, the field is additive-optional.
    @Test func legacyRecordWithoutRewindStillDecodes() throws {
        let legacyJSON = """
            {"timestamp":700000000,"requestID":"\(UUID().uuidString)","modelID":"m",
             "partitionDigest":"p","promptTokenCount":10,"prefixBlockDigests":[],
             "admittedCheckpoints":[],"ramBudgetBytes":1000,"restoredOffset":0,
             "restoredFromSSD":false,"hitTokens":0,"sharedPrefixLength":0,
             "lookupSeconds":0,"restoreSeconds":0,"hydrationSeconds":0,
             "prefillSeconds":0,"residualPromptSeconds":0,"ttftSeconds":0,
             "terminalEvictionCount":0,"recoveredEvictionCount":0}
            """
        let decoded = try JSONDecoder().decode(
            CompletionTraceRecord.self, from: Data(legacyJSON.utf8)
        )
        #expect(decoded.rewind == nil)
    }

    // MARK: - Hit credit

    /// A stored snapshot is credited only up to its deepest
    /// digest-verified block boundary: a leaf at token 1300 (5 full
    /// blocks) credits 1280 tokens, never 1300.
    @Test func hitCreditIsBlockAligned() {
        let records = [
            makeRecord(
                digests: digestChain("d", 5), promptTokenCount: 1300,
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 100, checkpointType: "leaf")
            ),
            makeRecord(digests: digestChain("d", 7), promptTokenCount: 1900),
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.simulatedLRU.totalHitTokens == 1280)
        #expect(report.simulatedLRU.hitRequestCount == 1)
    }

    /// Snapshots in another partition never match, even with
    /// identical digest chains.
    @Test func partitionsDoNotCrossMatch() {
        let records = [
            makeRecord(
                digests: digestChain("d", 5), promptTokenCount: 1300,
                partition: "model-a",
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 100, checkpointType: "leaf")
            ),
            makeRecord(
                digests: digestChain("d", 5), promptTokenCount: 1300,
                partition: "model-b"
            ),
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.simulatedLRU.hitRequestCount == 0)
    }

    /// A branch-point checkpoint survives deeper leaf admissions on
    /// its path and still serves shallower branches: after the leaf
    /// extends to 7 blocks, a request diverging at block 3 hits the
    /// checkpoint's 2 verified blocks.
    @Test func checkpointSurvivesLeafSupersessionAndServesBranches() {
        let chain = digestChain("d", 7)
        let records = [
            makeRecord(
                digests: Array(chain.prefix(5)), promptTokenCount: 1300,
                checkpoints: [
                    TraceAdmittedSnapshot(offset: 512, bytes: 40, checkpointType: "branchPoint")
                ],
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 60, checkpointType: "leaf")
            ),
            makeRecord(
                digests: chain, promptTokenCount: 1900,
                leaf: TraceAdmittedSnapshot(offset: 1900, bytes: 70, checkpointType: "leaf")
            ),
            makeRecord(
                digests: [chain[0], chain[1], "divergent3"], promptTokenCount: 800
            ),
        ]
        let report = TraceReplayHarness.replay(records: records)
        // r2 credits the r1 leaf (5 blocks = 1280); r3 credits the
        // checkpoint (2 blocks = 512).
        #expect(report.simulatedLRU.totalHitTokens == 1280 + 512)
        #expect(report.simulatedLRU.hitRequestCount == 2)
    }

    // MARK: - Leaf supersession

    /// A deeper leaf on the same path replaces its ancestor leaf —
    /// the freed bytes mean a budget equal to the new leaf alone
    /// triggers no eviction.
    @Test func deeperLeafSupersedesAncestorOnSamePath() {
        let records = [
            makeRecord(
                digests: digestChain("d", 5), promptTokenCount: 1300,
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 6000, checkpointType: "leaf")
            ),
            makeRecord(
                digests: digestChain("d", 7), promptTokenCount: 1900,
                leaf: TraceAdmittedSnapshot(offset: 1900, bytes: 7000, checkpointType: "leaf")
            ),
            makeRecord(digests: digestChain("d", 7), promptTokenCount: 1900, ramBudgetBytes: 7000),
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.simulatedLRU.evictionCount == 0)
    }

    /// A leaf on a different path is untouched by supersession; the
    /// same 7000-byte budget must then evict the stalest leaf.
    @Test func siblingLeafIsPreservedAndLRUEvictsStalest() {
        let records = [
            makeRecord(
                digests: digestChain("a", 5), promptTokenCount: 1300,
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 6000, checkpointType: "leaf")
            ),
            makeRecord(
                digests: digestChain("b", 7), promptTokenCount: 1900,
                leaf: TraceAdmittedSnapshot(offset: 1900, bytes: 7000, checkpointType: "leaf")
            ),
            makeRecord(digests: digestChain("b", 7), promptTokenCount: 1900, ramBudgetBytes: 7000),
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.simulatedLRU.evictionCount == 1)
        #expect(report.simulatedLRU.evictedBytes == 6000)
    }

    /// A hit refreshes the matched snapshot's recency, steering LRU
    /// away from it: after r3 touches the "a" leaf, the budget squeeze
    /// in r4 evicts the untouched "b" leaf instead.
    @Test func hitRecencyProtectsTheTouchedLeaf() {
        let records = [
            makeRecord(
                digests: digestChain("a", 5), promptTokenCount: 1300,
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 6000, checkpointType: "leaf")
            ),
            makeRecord(
                digests: digestChain("b", 5), promptTokenCount: 1300,
                leaf: TraceAdmittedSnapshot(offset: 1300, bytes: 6000, checkpointType: "leaf")
            ),
            makeRecord(digests: digestChain("a", 5), promptTokenCount: 1300),
            makeRecord(digests: digestChain("c", 1), promptTokenCount: 256, ramBudgetBytes: 6000),
        ]
        let report = TraceReplayHarness.replay(records: records)
        #expect(report.simulatedLRU.evictionCount == 1)
        // r5-equivalent probe: replay again with an extra "a" request
        // under the same squeeze — it still hits, proving "b" was the
        // victim.
        let probed = TraceReplayHarness.replay(
            records: records + [
                makeRecord(
                    digests: digestChain("a", 5), promptTokenCount: 1300,
                    ramBudgetBytes: 6000
                )
            ]
        )
        #expect(probed.simulatedLRU.hitRequestCount == 2)
    }

    // MARK: - TTFT-proxy denomination

    /// The proxy is denominated in each record's own measured
    /// estimates: doubling prefill throughput halves the proxy.
    @Test func proxyUsesEachRecordsOwnEstimates() {
        let slow = TraceReplayHarness.replay(records: [
            makeRecord(
                digests: digestChain("d", 4), promptTokenCount: 1024,
                estimates: MeasuredSecondsEstimates(prefillFlopsPerSecond: 1.0e12)
            )
        ])
        let fast = TraceReplayHarness.replay(records: [
            makeRecord(
                digests: digestChain("d", 4), promptTokenCount: 1024,
                estimates: MeasuredSecondsEstimates(prefillFlopsPerSecond: 2.0e12)
            )
        ])
        #expect(slow.simulatedLRU.ttftProxyP50Seconds > 0)
        #expect(
            abs(
                slow.simulatedLRU.ttftProxyP50Seconds
                    - 2 * fast.simulatedLRU.ttftProxyP50Seconds) < 1e-12)
    }

    /// Records predating slice #84 (no estimates) replay exactly as if
    /// they carried the estimator's cold-start defaults.
    @Test func missingEstimatesFallBackToColdStartDefaults() {
        let withNil = TraceReplayHarness.replay(records: [
            makeRecord(
                digests: digestChain("d", 4), promptTokenCount: 1024, estimates: nil
            )
        ])
        let withDefaults = TraceReplayHarness.replay(records: [
            makeRecord(
                digests: digestChain("d", 4), promptTokenCount: 1024,
                estimates: MeasuredSecondsEstimates()
            )
        ])
        #expect(
            withNil.simulatedLRU.ttftProxyP50Seconds
                == withDefaults.simulatedLRU.ttftProxyP50Seconds)
    }

    // MARK: - Degenerate corpus

    @Test func emptyCorpusYieldsZeroReport() {
        let report = TraceReplayHarness.replay(records: [])
        #expect(report.recordCount == 0)
        #expect(report.observed.ttftP50Seconds == 0)
        #expect(report.simulatedLRU.totalHitTokens == 0)
        #expect(report.simulatedLRU.evictionCount == 0)
    }
}
