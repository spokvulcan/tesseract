//
//  CompletionTraceAccumulatorTests.swift
//  tesseractTests
//
//  The trace record's derivation rules — eviction classification, the
//  restored-offset rule, admitted-snapshot projections — previously
//  lived inline in the completion drive where only a full driven
//  completion could exercise them.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CompletionTraceAccumulatorTests {

    private func makeDiagnostics() -> PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "test-model", kvBits: 8, kvGroupSize: 64)
    }

    private func makeEviction(
        terminalRefID: String?
    ) -> PrefixCacheManager.EvictionEvent {
        PrefixCacheManager.EvictionEvent(
            strategy: .utility,
            offset: 128,
            checkpointType: .leaf,
            freedBytes: 1_024,
            budgetBytes: 8_192,
            snapshotBytesAfter: 4_096,
            normalizedRecency: nil,
            normalizedFlopEfficiency: nil,
            utility: nil,
            bodyDroppedSnapshotRefID: terminalRefID,
            chainPrefixOwnerID: nil
        )
    }

    private func makeStartFacts(
        unkeyedReason: CacheKeySpace.UnkeyedReason? = nil,
        lookupReason: PrefixCacheManager.LookupReason? = .missNoEntries,
        restore: TurnRestore = .cold,
        sharedPrefixLength: Int = 0
    ) -> CompletionTraceAccumulator.StartFacts {
        CompletionTraceAccumulator.StartFacts(
            partitionDigest: "abcd1234",
            unkeyedReason: unkeyedReason,
            keyPath: [1, 2, 3],
            lookupReason: lookupReason,
            restore: restore,
            restoredFromSSD: false,
            hitTokens: 0,
            sharedPrefixLength: sharedPrefixLength,
            lookupSeconds: 0.001,
            restoreSeconds: 0.002,
            hydrationSeconds: 0,
            prefillSeconds: 0.5
        )
    }

    // MARK: - Eviction classification

    @Test
    func classifiesTerminalVersusRecoveredEvictions() {
        var trace = CompletionTraceAccumulator()
        let diagnostics = makeDiagnostics()

        // A body-drop that left a live Snapshot Ref is recovered; a drop
        // with no surviving ref is terminal (the event's own rule).
        trace.ingest(
            evictions: [
                makeEviction(terminalRefID: nil),
                makeEviction(terminalRefID: "ref-1"),
                makeEviction(terminalRefID: nil),
            ], diagnostics: diagnostics)
        trace.ingest(
            evictions: [makeEviction(terminalRefID: "ref-2")], diagnostics: diagnostics)

        #expect(trace.terminalEvictionCount == 2)
        #expect(trace.recoveredEvictionCount == 2)
    }

    // MARK: - Restored offset rule

    nonisolated private static let hit = PrefixCacheManager.LookupReason.hit(
        snapshotOffset: 342, totalTokens: 400, type: .leaf)
    nonisolated private static let handoff = TurnRestore.handoff(
        .init(leaseID: UUID(), rewindStateBytes: 0, waitSeconds: 0))

    /// The restored offset is what the turn restored: the hit's offset on a
    /// handoff or a copy, and 0 when the generation began from an empty
    /// cache. The two cold rows of a hit are the declared fix (ADR-0069's
    /// 2026-10-11 amendment): a restore that fell back cold, and a hit the
    /// plan ran cold (one covering the whole prompt), used to record the
    /// hit's offset beside `hitTokens=0`.
    @Test(
        arguments: [
            (hit, handoff, 342),
            (hit, TurnRestore.copy(nil), 342),
            (hit, TurnRestore.failedThenCold(nil), 0),
            (hit, TurnRestore.cold, 0),
            (PrefixCacheManager.LookupReason.missNoEntries, TurnRestore.cold, 0),
            (PrefixCacheManager.LookupReason.missNoSnapshotInPrefix, TurnRestore.cold, 0),
            // An Unkeyed Completion never looked up.
            (nil, TurnRestore.cold, 0),
        ] as [(PrefixCacheManager.LookupReason?, TurnRestore, Int)])
    func restoredOffsetIsWhatTheTurnRestored(
        _ reason: PrefixCacheManager.LookupReason?, _ restore: TurnRestore, _ offset: Int
    ) {
        #expect(
            CompletionTraceAccumulator.restoredOffset(for: reason, restore: restore) == offset)
    }

    /// A fallback turn's record restores nothing, so it carries no rewind
    /// either.
    @Test func aFallbackTurnsRecordRestoredNothing() throws {
        let record = try #require(
            CompletionTraceAccumulator().makeRecord(
                timestamp: 1_000,
                requestID: UUID(),
                modelID: "test-model",
                start: makeStartFacts(
                    lookupReason: Self.hit, restore: .failedThenCold(nil),
                    sharedPrefixLength: 350),
                capturedSnapshots: [],
                leafStore: nil,
                ramBudgetBytes: 8_192,
                residualPromptSeconds: 0.1,
                deviceEstimates: nil
            ))
        #expect(record.restoredOffset == 0)
        #expect(record.hitTokens == 0)
        #expect(record.rewind == nil)
    }

    // MARK: - Record derivation

    @Test
    func unkeyedCompletionYieldsNoRecord() {
        let trace = CompletionTraceAccumulator()
        let record = trace.makeRecord(
            timestamp: 1_000,
            requestID: UUID(),
            modelID: "test-model",
            start: makeStartFacts(unkeyedReason: .placeholderRunCountMismatch, lookupReason: nil),
            capturedSnapshots: [],
            leafStore: nil,
            ramBudgetBytes: 8_192,
            residualPromptSeconds: 0.1,
            deviceEstimates: nil
        )
        #expect(record == nil)
    }

    @Test
    func keyedRecordCarriesTalliesAndDerivedFields() throws {
        var trace = CompletionTraceAccumulator()
        trace.ingest(
            evictions: [makeEviction(terminalRefID: nil), makeEviction(terminalRefID: "ref")],
            diagnostics: makeDiagnostics())

        let leaf = AlphaTuner.LeafStore(storedTokens: [1, 2, 3, 4], bytes: 2_048)
        let record = try #require(
            trace.makeRecord(
                timestamp: 1_000,
                requestID: UUID(),
                modelID: "test-model",
                start: makeStartFacts(
                    lookupReason: .hit(snapshotOffset: 300, totalTokens: 400, type: .leaf),
                    restore: .copy(nil), sharedPrefixLength: 350),
                capturedSnapshots: [],
                leafStore: leaf,
                ramBudgetBytes: 8_192,
                residualPromptSeconds: 0.1,
                deviceEstimates: nil
            ))

        #expect(record.terminalEvictionCount == 1)
        #expect(record.recoveredEvictionCount == 1)
        #expect(record.restoredOffset == 300)
        #expect(record.admittedLeaf?.offset == 4)
        #expect(record.admittedLeaf?.bytes == 2_048)
        // Rewind derives from the same restored offset: 350 - 300.
        #expect(record.rewind?.rewindSize == 50)
    }

    /// A Leaf Admission classifies its own admission and hands back the
    /// tally; the request's record takes the counts without a second line.
    @Test func mergingATallyAddsItsCountsToTheRecord() throws {
        var admission = CompletionTraceAccumulator()
        admission.ingest(
            evictions: [makeEviction(terminalRefID: nil), makeEviction(terminalRefID: "ref")],
            diagnostics: makeDiagnostics())
        var request = CompletionTraceAccumulator()
        request.ingest(
            evictions: [makeEviction(terminalRefID: nil)], diagnostics: makeDiagnostics())

        request.merge(admission)

        #expect(request.terminalEvictionCount == 2)
        #expect(request.recoveredEvictionCount == 1)
    }
}
