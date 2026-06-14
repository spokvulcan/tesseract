//
//  SalvageOnCancelTests.swift
//  tesseractTests
//
//  **Salvage-on-cancel** (PRD #94, issue #97): a client cancel or
//  disconnect mid-prefill admits the progress at the last completed
//  chunk RAM-only — gated by the speculative preempt-capture threshold —
//  instead of discarding every prefilled token. Covered at the
//  server-completion cancellation seam: the pure offset gate, and the
//  salvage path against a real manager (capture → admit → next lookup
//  restores at the salvaged offset).
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct SalvageOnCancelTests {

    private let threshold = SpeculativeCanonicalPrefill.minimumPreemptCaptureTokens

    // MARK: - Pure offset gate

    @Test func offsetGateRequiresThresholdPastTheRestoreBase() {
        // At threshold: admit. One short: nothing.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: 5_000 + threshold, restoreBaseOffset: 5_000,
                keyPathCount: 100_000, minimumWarmOffset: 0
            ) == 5_000 + threshold)
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: 5_000 + threshold - 1, restoreBaseOffset: 5_000,
                keyPathCount: 100_000, minimumWarmOffset: 0
            ) == nil)
    }

    @Test func offsetGateRefusesInconsistentOrUnanchorableOffsets() {
        // Past the key path — a mid-flight inconsistency must never admit.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: threshold, restoreBaseOffset: 0,
                keyPathCount: threshold - 1, minimumWarmOffset: 0
            ) == nil)
        // Inside the image prefix — unanchorable on restore.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: threshold, restoreBaseOffset: 0,
                keyPathCount: 100_000, minimumWarmOffset: threshold + 1
            ) == nil)
    }

    // MARK: - Salvage seam (capture → admit → lookup)

    private func makeWarmedCache(offset: Int) -> [any KVCache] {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, offset, 8]),
            MLXArray.zeros([1, 1, offset, 8]),
        ]
        return [kv]
    }

    private func makeFixture() -> (
        manager: PrefixCacheManager, key: CachePartitionKey,
        diagnostics: PrefixCacheDiagnostics.Context
    ) {
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 1 << 30,
            tieredStore: TieredSnapshotStore(ssdConfig: nil)
        )
        let key = CachePartitionKey(modelID: "salvage-test", kvBits: nil, kvGroupSize: 64)
        let diagnostics = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64
        )
        return (manager, key, diagnostics)
    }

    @Test func cancelledPrefillPastTheThresholdIsVisibleToTheNextLookup() async {
        let (manager, key, diagnostics) = makeFixture()
        let offset = threshold + 256
        let keyPath = (0..<(offset + 512)).map { $0 % 997 }

        await ServerCompletion.salvageCancelledPrefill(
            cache: makeWarmedCache(offset: offset),
            keySpace: .identity(keyPath: keyPath),
            restoreBaseOffset: 0,
            partitionKey: key,
            requestID: UUID(),
            prefixCache: manager,
            diagnostics: diagnostics
        )

        // A re-sent identical request restores at the salvaged offset.
        let lookup = manager.lookup(tokens: keyPath, partitionKey: key)
        guard case .hit(let snapshotOffset, _, let type) = lookup.reason else {
            Issue.record("expected RAM hit at the salvaged offset, got \(lookup.reason)")
            return
        }
        #expect(snapshotOffset == offset)
        #expect(type == .leaf)
        #expect(lookup.snapshot != nil)
    }

    @Test func progressBelowTheThresholdAdmitsNothing() async {
        let (manager, key, diagnostics) = makeFixture()
        let offset = threshold - 1
        let keyPath = Array(0..<(offset + 512))

        await ServerCompletion.salvageCancelledPrefill(
            cache: makeWarmedCache(offset: offset),
            keySpace: .identity(keyPath: keyPath),
            restoreBaseOffset: 0,
            partitionKey: key,
            requestID: UUID(),
            prefixCache: manager,
            diagnostics: diagnostics
        )

        #expect(manager.stats.snapshotCount == 0)
    }

    @Test func progressIsMeasuredFromTheRestoreBaseNotZero() async {
        // The incident's double-interrupt shape: restored at a deep
        // floor, cancelled shortly after — progress past the *base* is
        // what's gated, so a deep cache offset alone must not admit.
        let (manager, key, diagnostics) = makeFixture()
        let base = 40_000
        let offset = base + threshold - 1
        let keyPath = Array(0..<(offset + 512)).map { $0 % 997 }

        await ServerCompletion.salvageCancelledPrefill(
            cache: makeWarmedCache(offset: offset),
            keySpace: .identity(keyPath: keyPath),
            restoreBaseOffset: base,
            partitionKey: key,
            requestID: UUID(),
            prefixCache: manager,
            diagnostics: diagnostics
        )

        #expect(manager.stats.snapshotCount == 0)
    }
}
