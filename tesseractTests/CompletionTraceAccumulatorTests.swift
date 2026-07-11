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
        lookupReason: PrefixCacheManager.LookupReason = .missNoEntries,
        sharedPrefixLength: Int = 0
    ) -> CompletionTraceAccumulator.StartFacts {
        CompletionTraceAccumulator.StartFacts(
            partitionDigest: "abcd1234",
            unkeyedReason: unkeyedReason,
            keyPath: [1, 2, 3],
            lookupReason: lookupReason,
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

    @Test
    func restoredOffsetIsTheHitOffsetOrZero() {
        #expect(
            CompletionTraceAccumulator.restoredOffset(
                for: .hit(snapshotOffset: 342, totalTokens: 400, type: .leaf)) == 342)
        #expect(CompletionTraceAccumulator.restoredOffset(for: .missNoEntries) == 0)
        #expect(CompletionTraceAccumulator.restoredOffset(for: .missNoSnapshotInPrefix) == 0)
    }

    // MARK: - Record derivation

    @Test
    func unkeyedCompletionYieldsNoRecord() {
        let trace = CompletionTraceAccumulator()
        let record = trace.makeRecord(
            timestamp: 1_000,
            requestID: UUID(),
            modelID: "test-model",
            start: makeStartFacts(unkeyedReason: .placeholderRunCountMismatch),
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
                    sharedPrefixLength: 350),
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
}
