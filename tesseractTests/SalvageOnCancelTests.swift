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

    /// A text key path of `count` tokens.
    private func textKeySpace(count: Int) -> CacheKeySpace {
        .identity(keyPath: Array(repeating: 7, count: count))
    }

    /// A key path of `count` tokens whose one image run ends at `runEnd`.
    private func imageKeySpace(count: Int, runEnd: Int) throws -> CacheKeySpace {
        let pad = 999
        var prepared = Array(repeating: 7, count: count)
        for index in (runEnd - 8)..<runEnd { prepared[index] = pad }
        return try CacheKeySpace.make(
            preparedTokens: prepared,
            images: [
                CacheKeySpace.RequestImage(
                    digest: #require(ImageDigest(rawDigest: Data(repeating: 0xAB, count: 32))),
                    positionSpan: 4)
            ],
            placeholderIdentity: ImagePlaceholderIdentity(imagePadTokenId: pad)
        ).get()
    }

    @Test func offsetGateRequiresThresholdPastTheRestoreBase() {
        // At threshold: admit. One short: nothing.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: 5_000 + threshold, restoreBaseOffset: 5_000,
                keySpace: textKeySpace(count: 100_000)
            ) == 5_000 + threshold)
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: 5_000 + threshold - 1, restoreBaseOffset: 5_000,
                keySpace: textKeySpace(count: 100_000)
            ) == nil)
    }

    @Test func offsetGateRefusesInconsistentOrUnanchorableOffsets() throws {
        // The whole key path is a restore point a later request extends.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: threshold, restoreBaseOffset: 0,
                keySpace: textKeySpace(count: threshold)
            ) == threshold)
        // Past the key path — a mid-flight inconsistency must never admit.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: threshold, restoreBaseOffset: 0,
                keySpace: textKeySpace(count: threshold - 1)
            ) == nil)
        // Inside the image prefix — unanchorable on restore.
        #expect(
            ServerCompletion.salvageableOffset(
                cacheOffset: threshold, restoreBaseOffset: 0,
                keySpace: try imageKeySpace(count: threshold + 100, runEnd: threshold + 1)
            ) == nil)
    }

    // MARK: - Salvage seam (capture → admit → lookup)

    private let sessions = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2]))

    nonisolated private static func makeWarmedCache(offset: Int) -> [any KVCache] {
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
        let telemetry = TelemetryCapture(modelID: key.modelID)
        defer { telemetry.stop() }
        let offset = threshold + 256
        let keyPath = (0..<(offset + 512)).map { $0 % 997 }

        await sessions.withSession { session in
            // The salvage captures inside the prefill's session; here the toy's.
            await ServerCompletion.salvageCancelledPrefill(
                cache: Self.makeWarmedCache(offset: offset),
                keySpace: .identity(keyPath: keyPath),
                restoreBaseOffset: 0,
                partitionKey: key,
                requestID: UUID(),
                prefixCache: manager,
                diagnostics: diagnostics,
                session: session
            )
        }

        // A re-sent identical request restores at the salvaged offset.
        let lookup = manager.lookup(tokens: keyPath, partitionKey: key)
        guard case .hit(let snapshotOffset, _, let type) = lookup.reason else {
            Issue.record("expected RAM hit at the salvaged offset, got \(lookup.reason)")
            return
        }
        #expect(snapshotOffset == offset)
        #expect(type == .leaf)
        #expect(lookup.snapshot != nil)
        // The salvaged leaf's capture line keeps its source label.
        let capture = telemetry.drain().first {
            $0.requestID == diagnostics.requestID && $0.eventName == "capture"
        }
        #expect(capture?.field("source") == "cancelledPrefillSalvage")
    }

    @Test func progressBelowTheThresholdAdmitsNothing() async {
        let (manager, key, diagnostics) = makeFixture()
        let telemetry = TelemetryCapture(modelID: key.modelID)
        defer { telemetry.stop() }
        let offset = threshold - 1
        let keyPath = Array(0..<(offset + 512))

        await sessions.withSession { session in
            // The salvage captures inside the prefill's session; here the toy's.
            await ServerCompletion.salvageCancelledPrefill(
                cache: Self.makeWarmedCache(offset: offset),
                keySpace: .identity(keyPath: keyPath),
                restoreBaseOffset: 0,
                partitionKey: key,
                requestID: UUID(),
                prefixCache: manager,
                diagnostics: diagnostics,
                session: session
            )
        }

        #expect(manager.stats.snapshotCount == 0)
        let skip = telemetry.drain().first {
            $0.requestID == diagnostics.requestID && $0.eventName == "skip"
        }
        #expect(skip?.field("stage") == "salvageOnCancel")
        #expect(skip?.field("reason") == "below-progress-threshold")
    }

    @Test func progressIsMeasuredFromTheRestoreBaseNotZero() async {
        // The incident's double-interrupt shape: restored at a deep
        // floor, cancelled shortly after — progress past the *base* is
        // what's gated, so a deep cache offset alone must not admit.
        let (manager, key, diagnostics) = makeFixture()
        let base = 40_000
        let offset = base + threshold - 1
        let keyPath = Array(0..<(offset + 512)).map { $0 % 997 }

        await sessions.withSession { session in
            // The salvage captures inside the prefill's session; here the toy's.
            await ServerCompletion.salvageCancelledPrefill(
                cache: Self.makeWarmedCache(offset: offset),
                keySpace: .identity(keyPath: keyPath),
                restoreBaseOffset: base,
                partitionKey: key,
                requestID: UUID(),
                prefixCache: manager,
                diagnostics: diagnostics,
                session: session
            )
        }

        #expect(manager.stats.snapshotCount == 0)
    }
}
