import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// ADR-0064's capture-side ownership seam: real cache objects and physical
/// buffers, admitted through the real RAM/SSD tiers. Restore still copies.
@MainActor
struct LeafCaptureHandoffTests {
    private let key = CachePartitionKey(
        modelID: "capture-handoff", kvBits: nil, kvGroupSize: 64,
        modelFingerprint: String(repeating: "a", count: 64))

    private func owner(offset: Int = 8) -> FinalGenerationCache {
        let kv = KVCacheSimple()
        let count = offset * 64
        kv.state = [
            MLXArray((0..<count).map(Float.init)).reshaped([1, 1, offset, 64]),
            MLXArray.ones([1, 1, offset, 64]),
        ]
        let recurrent = MambaCache()
        recurrent.state = [MLXArray([Float(3), 4]), MLXArray([Float(5), 6])]
        eval([kv, recurrent] as [any KVCache])
        return FinalGenerationCache([kv, recurrent])
    }

    private func admit(
        _ owner: FinalGenerationCache, offset: Int = 8,
        into manager: PrefixCacheManager, tokens: [Int]? = nil,
        storage: (HybridCacheSnapshot) -> SnapshotAdmission.Storage = { _ in .ramOnly }
    ) throws {
        let leaf = try #require(owner.moveSnapshot(offset: offset))
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens ?? Array(1...offset), snapshot: leaf,
                    storage: storage(leaf), partitionKey: key)))
    }

    @Test func admittedLeafOwnsTheOriginalObjectsAndCheckpointStillCopies() throws {
        let request = owner()
        let retainedRequest = request
        weak var attention = request.cache[0] as? KVCacheSimple
        weak var recurrent = request.cache[1] as? MambaCache
        let addresses = request.cache.flatMap(\.state).map(backingAddress)
        let arrayIDs = request.cache.flatMap(\.state).map(ObjectIdentifier.init)
        let checkpoint = try #require(
            HybridCacheSnapshot.capture(
                cache: request.cache, offset: 8, type: .system))
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admit(request, into: manager)

        #expect(request.cache.isEmpty)
        #expect(retainedRequest.cache.isEmpty)
        #expect(attention != nil)
        #expect(recurrent != nil)
        do {
            let leaf = try #require(
                manager.lookup(tokens: Array(1...8), partitionKey: key).snapshot)
            #expect(leaf.layers.flatMap(\.state).map(backingAddress) == addresses)
            #expect(leaf.layers.flatMap(\.state).map(ObjectIdentifier.init) == arrayIDs)
            #expect(manager.totalSnapshotBytes == checkpoint.memoryBytes)
            for (copied, original) in zip(checkpoint.layers.flatMap(\.state), addresses) {
                #expect(backingAddress(copied) != original)
            }
        }
        #expect(manager.clearRAMTier() == checkpoint.memoryBytes)
        #expect(manager.totalSnapshotBytes == 0)
        // Only the tree owned the objects: neither the request nor the
        // independent checkpoint keeps them alive after the RAM clear.
        #expect(attention == nil)
        #expect(recurrent == nil)
    }

    @Test func copiedRestoreOfMovedHybridLeafIsBitwiseIdenticalAndIndependent() throws {
        let request = owner()
        let copied = try #require(
            HybridCacheSnapshot.capture(
                cache: request.cache, offset: 8, type: .leaf))
        let moved = try #require(request.moveSnapshot(offset: 8))
        let fromCopy = try copied.restore()
        let fromMove = try moved.restore()
        for (old, new) in zip(fromCopy, fromMove) {
            #expect(old.offset == new.offset)
            #expect(old.metaState == new.metaState)
            for (lhs, rhs) in zip(old.state, new.state) {
                #expect(lhs.asData(access: .copy).data == rhs.asData(access: .copy).data)
            }
        }
        for (restored, body) in zip(fromMove.flatMap(\.state), moved.layers.flatMap(\.state)) {
            #expect(backingAddress(restored) != backingAddress(body))
        }
        let oldKV = try #require(fromCopy[0] as? KVCacheSimple)
        let newKV = try #require(fromMove[0] as? KVCacheSimple)
        _ = oldKV.update(keys: MLXArray.ones([1, 1, 3, 64]), values: MLXArray.ones([1, 1, 3, 64]))
        _ = newKV.update(keys: MLXArray.ones([1, 1, 3, 64]), values: MLXArray.ones([1, 1, 3, 64]))
        for (lhs, rhs) in zip(oldKV.state, newKV.state) {
            #expect(lhs.asData(access: .copy).data == rhs.asData(access: .copy).data)
        }
        #expect(moved.layers[0].offset == 8)
    }

    @Test func evictionDemotesMovedLeafAndSSDStillRestoresIt() async throws {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "moved-demotion", ramBudgetBytes: 4_112,
            demotionPayloadExtractor: { ServerCompletion.extractSnapshotPayload($0) })
        defer { try? FileManager.default.removeItem(at: root) }
        let request = owner()
        weak var attention = request.cache[0] as? KVCacheSimple
        let expected = request.cache[0].state[0].asData(access: .copy).data
        try admit(request, into: manager)
        try admit(owner(), into: manager, tokens: Array(20...27))
        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 0)
        #expect(attention == nil)
        await store.flush()
        #expect(
            await waitUntil {
                if case .ssdHit = manager.lookup(tokens: Array(1...8), partitionKey: key).reason {
                    return true
                }
                return false
            })
        let resolved = await resolve(manager, offset: 8)
        let restored = try #require(resolved.lookup.snapshot).restore()
        #expect(restored[0].state[0].asData(access: .copy).data == expected)
    }

    @Test func pendingExtensionDetachesMovedBodyAndSurvivesRAMClear() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "moved-extension", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let first = owner()
        try admit(first, into: manager) { leaf in
            .ramAndSSD(ServerCompletion.extractSnapshotPayload(leaf))
        }
        #expect(first.cache.isEmpty)
        let base = try #require(manager.extensionBase(tokens: Array(1...10), partitionKey: key))
        let next = owner(offset: 10)
        let addresses = next.cache.flatMap(\.state).map(backingAddress)
        let expected = next.cache.flatMap(\.state).map { $0.asData(access: .copy).data }
        var owed: ServerCompletion.DeferredLayers?
        try admit(next, offset: 10, into: manager) { leaf in
            let deferred = ServerCompletion.deferredPayload(for: leaf, extending: base)
            #expect(deferred.payload.extending != nil)
            owed = deferred.owed
            return .ramAndSSD(deferred.payload)
        }
        #expect(next.cache.isEmpty)
        for array in try #require(owed).retainedArrays {
            #expect(!addresses.contains(backingAddress(array)))
        }
        #expect(manager.clearRAMTier() > 0)
        await gate.open()
        await store.flush()
        #expect(
            await waitUntil {
                if case .ssdHit = manager.lookup(tokens: Array(1...10), partitionKey: key).reason {
                    return true
                }
                return false
            })
        let resolved = await resolve(manager, offset: 10)
        let restored = try #require(resolved.lookup.snapshot).restore()
        #expect(restored.flatMap(\.state).map { $0.asData(access: .copy).data } == expected)
        #expect(owed?.retainedArrays.isEmpty == true)
    }

    private func resolve(_ manager: PrefixCacheManager, offset: Int) async
        -> PrefixCacheManager.Resolved
    {
        await manager.resolve(
            tokens: Array(1...offset), promptTokenCount: offset, partitionKey: key,
            modelFingerprint: key.modelFingerprint,
            diagnostics: PrefixCacheDiagnostics.Context(
                requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64))
    }
}
