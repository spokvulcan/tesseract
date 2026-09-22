import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Model-free memory evidence for the **Cache Claim** path (#554): the MLX
/// peak around one step at a time, on small synthetic caches, after
/// resetting the process-global peak counter. The peak is shared by every
/// suite in the process, so the byte assertions hold only when this suite
/// runs alone:
///
///     TEST_RUNNER_TESSERACT_CACHE_CLAIM_MEMORY_EVIDENCE=1 xcodebuild test \
///       -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
///       -skipPackagePluginValidation -parallel-testing-enabled NO \
///       -only-testing:tesseractTests/CacheClaimMemoryEvidenceTests
///
/// Without the flag the steps still run and their functional assertions
/// hold; the measured bytes are printed either way (`CACHE_CLAIM_EVIDENCE=`).
@MainActor
@Suite(.serialized)
struct CacheClaimMemoryEvidenceTests {
    private static let assertsPeaks =
        ProcessInfo.processInfo.environment["TESSERACT_CACHE_CLAIM_MEMORY_EVIDENCE"] == "1"

    private let key = CachePartitionKey(
        modelID: "cache-claim-evidence", kvBits: nil, kvGroupSize: 64)
    private let sessions = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2]))

    /// 8,192 rows of one-head, 64-wide float32 keys and values: 4 MiB.
    private static let rows = 8_192
    /// Two float32 recurrent slots of 2 MiB each: the state exact rewind
    /// copies at check-out, 4 MiB.
    private static let recurrentElements = 524_288
    private static let recurrentBytes = 2 * recurrentElements * 4

    private func hybridLeaf() throws -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.ones([1, 1, Self.rows, 64]), MLXArray.ones([1, 1, Self.rows, 64]),
        ]
        let recurrent = MambaCache()
        recurrent.state = [
            MLXArray.ones([Self.recurrentElements]), MLXArray.ones([Self.recurrentElements]),
        ]
        recurrent.offset = Self.rows
        let owner = FinalGenerationCache([kv, recurrent])
        eval(owner.cache)
        return try #require(owner.moveSnapshot(offset: Self.rows))
    }

    /// A handed-off turn that decoded `grown` tokens and was captured by
    /// move, ready for its check-in: the request's claim (handed over, not
    /// yet redeemed), the live cache and the captured leaf.
    private struct DecodedTurn {
        let handOver: CacheClaim.HandOver
        let live: FinalGenerationCache
        let leaf: HybridCacheSnapshot
        let tokens: [Int]
    }

    private func decodedTurn(
        _ manager: PrefixCacheManager, grown: Int = 16
    ) async throws -> DecodedTurn {
        let tokens = Array(0..<Self.rows)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: try hybridLeaf(), storage: .ramOnly,
                    partitionKey: key)))
        let requested = tokens + Array(repeating: 7, count: grown)
        let (outcome, handOver) = await manager.checkOutHoldingClaim(
            .init(
                lookup: manager.lookup(tokens: requested, partitionKey: key),
                hydratedFromSSD: false, hydrationSeconds: 0),
            tokens: requested, maximumAdvance: grown + 1,
            diagnostics: .init(
                requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64),
            sessions: sessions)
        guard case .handoff(let handoff) = outcome else {
            throw EvidenceError.notHandedOff
        }
        let live = handoff.cache
        _ = live.cache[0].update(
            keys: MLXArray.zeros([1, 1, grown, 64]), values: MLXArray.zeros([1, 1, grown, 64]))
        let recurrent = try #require(live.cache[1] as? MambaCache)
        recurrent.state = [
            MLXArray.zeros([Self.recurrentElements]), MLXArray.zeros([Self.recurrentElements]),
        ]
        recurrent.offset = requested.count
        eval(live.cache)
        let leaf = try #require(live.moveSnapshot(offset: requested.count))
        return DecodedTurn(handOver: handOver, live: live, leaf: leaf, tokens: requested)
    }

    private enum EvidenceError: Error { case notHandedOff }

    /// The extension base the SSD tier would chain the leaf from: half the
    /// attention rows, so the suffix is worth writing (#517's worth-it gate).
    private static let extensionBase = SnapshotExtension(
        baseSnapshotID: "evidence-base", baseOffset: rows / 2)

    private func checkIn(
        _ handOver: CacheClaim.HandOver, _ leaf: HybridCacheSnapshot,
        from live: FinalGenerationCache, tokens: [Int]
    ) async -> CacheClaim.CheckIn {
        let sessions = sessions
        return await handOver.withClaim { claim in
            await sessions.withSession { session in
                await claim.checkIn(leaf, from: live, tokens: tokens, in: session)
            }
        }
    }

    /// Item 1 of #554: the executors check the leaf in before they extract
    /// an SSD extension payload. The check-in frees the recurrent rewind
    /// backup just before the payload copies the recurrent state, so the
    /// step's peak holds one copy of it fewer than extracting first did.
    @Test func checkingInBeforeExtractionHoldsOneFewerRecurrentCopy() async throws {
        // The order the executors used before: extract, then check in.
        let before = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let old = try await decodedTurn(before)
        let oldStart = Self.settledActiveMemory()
        let oldPayload = ServerCompletion.deferredPayload(
            for: old.leaf, extending: Self.extensionBase)
        let oldCheckIn = await checkIn(old.handOver, old.leaf, from: old.live, tokens: old.tokens)
        let oldPeak = Memory.peakMemory - oldStart
        #expect(oldCheckIn == .committed)
        #expect(oldPayload.payload.extending != nil)

        // The executors' order now: check in, then extract.
        let after = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let new = try await decodedTurn(after)
        let newStart = Self.settledActiveMemory()
        let newCheckIn = await checkIn(new.handOver, new.leaf, from: new.live, tokens: new.tokens)
        let newPayload = ServerCompletion.deferredPayload(
            for: new.leaf, extending: Self.extensionBase)
        let newPeak = Memory.peakMemory - newStart
        #expect(newCheckIn == .committed)
        #expect(newPayload.payload.totalBytes == oldPayload.payload.totalBytes)

        Self.report(
            "checkInBeforeExtraction",
            [
                "extractFirstPeakBytes": oldPeak, "checkInFirstPeakBytes": newPeak,
                "recurrentStateBytes": Self.recurrentBytes,
                "payloadBytes": newPayload.payload.totalBytes,
            ])
        if Self.assertsPeaks {
            #expect(
                oldPeak - newPeak >= Self.recurrentBytes * 9 / 10,
                "checking in first holds one recurrent copy fewer at the peak")
        }
        withExtendedLifetime((oldPayload, newPayload)) {}
    }

    /// A check-in the tree refuses rewinds in place: it rebuilds the
    /// recurrent layer from the backup it already holds and trims attention,
    /// and extracts no payload at all.
    @Test func aRefusedCheckInAllocatesNoPayload() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let turn = try await decodedTurn(manager)
        // Another body where the leaf would go: the tree refuses the check-in.
        let occupant = KVCacheSimple()
        occupant.state = [
            MLXArray.zeros([1, 1, turn.tokens.count, 1]),
            MLXArray.zeros([1, 1, turn.tokens.count, 1]),
        ]
        let occupantBody = try #require(
            HybridCacheSnapshot.capture(cache: [occupant], offset: turn.tokens.count, type: .leaf))
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: turn.tokens, snapshot: occupantBody, storage: .ramOnly,
                    partitionKey: key)))
        let start = Self.settledActiveMemory()
        let outcome = await checkIn(turn.handOver, turn.leaf, from: turn.live, tokens: turn.tokens)
        // The rewind frees the turn's own arrays, so the peak can sit below
        // where the step started; only what it rose above counts.
        let peak = max(0, Memory.peakMemory - start)
        #expect(outcome == .rewound(.refused(.occupiedDestination)))
        #expect(manager.lookup(tokens: turn.tokens, partitionKey: key).snapshot != nil)

        let rewound = try #require(
            manager.lookup(tokens: Array(0..<Self.rows) + [7], partitionKey: key).snapshot)
        let payloadBytes = ServerCompletion.deferredPayload(
            for: rewound, extending: Self.extensionBase
        ).payload.totalBytes
        Self.report("refusedCheckIn", ["peakBytes": peak, "payloadBytes": payloadBytes])
        if Self.assertsPeaks {
            #expect(peak < payloadBytes / 2, "a refused check-in extracts nothing")
        }
    }

    /// Across a claim's life the only arrays it allocates are the recurrent
    /// backup its rewind needs: a check-out moves the leaf's own objects,
    /// and the claim keeps references, not copies.
    @Test func aCheckOutAllocatesOnlyTheRecurrentBackup() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let tokens = Array(0..<Self.rows)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: try hybridLeaf(), storage: .ramOnly,
                    partitionKey: key)))
        let start = Self.settledActiveMemory()
        let (outcome, handOver) = await manager.checkOutHoldingClaim(
            .init(
                lookup: manager.lookup(tokens: tokens + [7], partitionKey: key),
                hydratedFromSSD: false, hydrationSeconds: 0),
            tokens: tokens + [7], maximumAdvance: 2,
            diagnostics: .init(
                requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64),
            sessions: sessions)
        let held = Memory.activeMemory - start
        let peak = Memory.peakMemory - start
        let handoff = try #require(outcome.handoff)
        #expect(handoff.rewindStateBytes == Self.recurrentBytes)
        #expect(await handOver.rewindAndConclude(sessions: sessions) == Self.rows)
        Stream().synchronize()
        let afterRewind = Memory.activeMemory - start

        Self.report(
            "checkOutAllocation",
            [
                "heldBytes": held, "peakBytes": peak, "afterRewindBytes": afterRewind,
                "recurrentStateBytes": Self.recurrentBytes,
            ])
        if Self.assertsPeaks {
            #expect(held <= Self.recurrentBytes + 65_536, "the backup is the only allocation")
            #expect(peak <= Self.recurrentBytes + 65_536)
            #expect(afterRewind <= 65_536, "the rewind returns the backup")
        }
    }

    /// Item 2 of #554: compaction evaluates one layer at a time. Each
    /// layer's replacement is materialized and its old arrays released
    /// before the next layer allocates, so the transient is one layer's
    /// share of the replacement, not the whole attention body's.
    @Test func compactionHoldsOneLayersReplacementAtATime() {
        // 8 layers of 2 heads × 64 dims, float16: 512 B per row. A 2,048-row
        // body that grew 20,000 rows and was trimmed back retains ~80 MB.
        let layers = 8
        let cache: [any KVCache] = (0..<layers).map { _ in
            let layer = KVCacheSimple()
            let keys = MLXArray.ones([1, 2, 2_048 + 20_000, 64], dtype: .float16)
            _ = layer.update(keys: keys, values: keys)
            layer.trim(20_000)
            return layer
        }
        eval(cache)
        let oldAddresses = cache.map { Set($0.innerState().map(backingAddress)) }
        // One layer's replacement: keys and values at the offset plus a step.
        let layerReplacementBytes = 2 * (2_048 + 256) * 2 * 64 * 2

        let start = Self.settledActiveMemory()
        let outcome = AttentionCapacityCompaction.compactIfNeeded(cache)
        let peak = max(0, Memory.peakMemory - start)

        #expect(outcome.compactedLayers == layers)
        // A replacement never shares the arrays it replaces. (A later layer
        // may reuse an earlier layer's freed buffer: that storage is gone.)
        for (layer, old) in zip(cache, oldAddresses) {
            #expect(Set(layer.innerState().map(backingAddress)).isDisjoint(with: old))
        }
        Self.report(
            "compaction",
            [
                "peakBytes": peak, "layerReplacementBytes": layerReplacementBytes,
                "wholeBodyReplacementBytes": layers * layerReplacementBytes,
                "freedBytes": outcome.freedBytes,
            ])
        if Self.assertsPeaks {
            #expect(
                peak <= 2 * layerReplacementBytes,
                "the transient is one layer's replacement, not the whole body's")
        }
    }

    /// Settle the device before a measured step: finish pending GPU work so
    /// buffers released earlier (by this test or the previous one) are gone,
    /// empty the buffer cache, and start the peak counter at what is live.
    private static func settledActiveMemory() -> Int {
        Stream().synchronize()
        Memory.clearCache()
        let active = Memory.activeMemory
        Memory.peakMemory = 0
        return active
    }

    private static func report(_ step: String, _ values: [String: Int]) {
        var record: [String: Any] = values
        record["step"] = step
        record["assertsPeaks"] = assertsPeaks
        if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
            let line = String(data: data, encoding: .utf8)
        {
            print("CACHE_CLAIM_EVIDENCE=" + line)
        }
    }
}
