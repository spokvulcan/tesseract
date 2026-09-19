import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct LeafCheckoutTests {
    private let key = CachePartitionKey(modelID: "checkout", kvBits: nil, kvGroupSize: 64)

    @Test func emptyCachesCannotBeCapturedOrAdmittedAsCompletedLeaves() async throws {
        var cache: [any KVCache] = []
        #expect(HybridCacheSnapshot.captureMoving(cache: &cache, offset: 8) == nil)
        #expect(HybridCacheSnapshot.capture(cache: [], offset: 8, type: .leaf) == nil)
        let empty = HybridCacheSnapshot(
            tokenOffset: 8, layers: [], checkpointType: .leaf, memoryBytes: 0, createdAt: .now)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let requestID = UUID()
        let admission = await ServerCompletion.admitStructuredLeaf(
            empty, storedTokens: Array(1...8), storage: .ramOnly, partitionKey: key,
            requestID: requestID, prefixCache: manager,
            diagnostics: .init(
                requestID: requestID, modelID: key.modelID, kvBits: nil, kvGroupSize: 64),
            admissionStage: "leafAdmission", captureSource: "leaf")
        #expect(!admission.survived)
        #expect(admission.store == nil)
        #expect(manager.lookup(tokens: Array(1...9), partitionKey: key).snapshot == nil)
    }

    @Test func pendingFullPayloadCopiesUntilItsArraysAreReleased() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "checkout-full", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let key = CachePartitionKey(
            modelID: "checkout-full", kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64))
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let previous = FinalGenerationCache([kv])
        let body = try #require(previous.moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        let payload = ServerCompletion.deferredPayload(for: body, extending: nil).payload
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body,
                    storage: .ramAndSSD(payload), partitionKey: key)))
        let context = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID,
            kvBits: nil, kvGroupSize: 64)
        func attempt() async -> LeafCheckout.Attempt {
            await LeafCheckout.attempt(
                resolved: .init(
                    lookup: manager.lookup(tokens: tokens + [9], partitionKey: key),
                    hydratedFromSSD: false, hydrationSeconds: 0),
                tokens: tokens + [9], maximumAdvance: 10, identityKeySpace: true,
                prefixCache: manager, context: context)
        }
        let pending = await attempt()
        #expect(pending.owner == nil)
        #expect(pending.copyReason == .pendingFullPayload)
        #expect(!body.layers.isEmpty)
        await gate.open()
        await store.flush()
        let afterWrite = try #require(await attempt().owner)
        #expect(afterWrite.cache[0] as AnyObject === kv)
        await afterWrite.rewindIfNeeded()
    }

    // MARK: - The bounded pending-payload wait (#523)

    /// One conversation, one leaf, one full payload the SSD writer still
    /// owes: the shape every second turn hits. `payload` gates the
    /// writer's materialize step so the test owns when the body arrays
    /// are released.
    @MainActor
    private struct PendingLeafScene {
        let manager: PrefixCacheManager
        let store: TieredSnapshotStore
        let root: URL
        let key: CachePartitionKey
        let kv: KVCacheSimple
        let tokens: [Int]
        let context: PrefixCacheDiagnostics.Context
        let materializing: BlockingMaterializer

        func attempt() async -> LeafCheckout.Attempt {
            await LeafCheckout.attempt(
                resolved: .init(
                    lookup: manager.lookup(tokens: tokens, partitionKey: key),
                    hydratedFromSSD: false, hydrationSeconds: 0),
                tokens: tokens, maximumAdvance: 10, identityKeySpace: true,
                prefixCache: manager, context: context)
        }

        func progress() -> PendingPayloadProgress {
            guard let snapshot = manager.lookup(tokens: tokens, partitionKey: key).snapshot
            else { return .absent }
            return manager.pendingFullPayloadProgress(
                snapshot: snapshot, tokens: tokens, partitionKey: key)
        }
    }

    private func makePendingLeafScene(
        label: String, wait: Duration,
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) throws -> PendingLeafScene {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: label, ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: writerDrainPreludeForTesting,
            evictionConfig: EvictionConfiguration(pendingFullPayloadWait: wait))
        let key = CachePartitionKey(
            modelID: label, kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64))
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let previous = FinalGenerationCache([kv])
        let body = try #require(previous.moveSnapshot(offset: 8))
        let stored = Array(1...8)
        let materializing = BlockingMaterializer()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: stored, snapshot: body,
                    storage: .ramAndSSD(
                        materializing.gating(
                            ServerCompletion.deferredPayload(for: body, extending: nil).payload)),
                    partitionKey: key)))
        return PendingLeafScene(
            manager: manager, store: store, root: root, key: key, kv: kv,
            tokens: stored + [9],
            context: PrefixCacheDiagnostics.Context(
                requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64),
            materializing: materializing)
    }

    @Test func aPayloadThatMaterializesWithinTheBoundIsWaitedOutIntoAHandoff() async throws {
        let scene = try makePendingLeafScene(label: "checkout-wait-hit", wait: .seconds(5))
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })

        let release = Task {
            try? await Task.sleep(for: .milliseconds(60))
            scene.materializing.release()
        }
        let attempt = await scene.attempt()
        _ = await release.value

        let owner = try #require(attempt.owner, "the wait should have turned the copy into a move")
        #expect(attempt.copyReason == nil)
        #expect(attempt.pendingPayloadWaitSeconds > 0)
        // The move is the real thing: the request holds the leaf's own
        // cache objects, not a restored copy of them.
        #expect(owner.cache[0] as AnyObject === scene.kv)
        await owner.rewindIfNeeded()
        await scene.store.flush()
    }

    @Test func aPayloadStillPendingAfterTheBoundCopiesAndReportsTheWaitedTime() async throws {
        let bound = Duration.milliseconds(80)
        let scene = try makePendingLeafScene(label: "checkout-wait-miss", wait: bound)
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })

        let started = ContinuousClock.now
        let attempt = await scene.attempt()
        let elapsed = ContinuousClock.now - started

        #expect(attempt.owner == nil)
        #expect(attempt.copyReason == .pendingFullPayload)
        #expect(attempt.pendingPayloadWaitSeconds >= 0.08)
        // Bounded: the request gives up and copies rather than waiting
        // out a writer that is taking its time.
        #expect(elapsed < .seconds(3))
        #expect(scene.progress() == .inProgress)

        scene.materializing.release()
        await scene.store.flush()
    }

    @Test func aQueuedPayloadIsNotWaitedForAndCopiesAtOnce() async throws {
        let gate = DrainGate()
        let scene = try makePendingLeafScene(
            label: "checkout-wait-queued", wait: .seconds(5),
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: scene.root) }
        // The writer has not started on this payload, so it has no
        // bounded completion time — waiting for it would be a guess.
        #expect(scene.progress() == .queued)

        let started = ContinuousClock.now
        let attempt = await scene.attempt()
        let elapsed = ContinuousClock.now - started

        #expect(attempt.owner == nil)
        #expect(attempt.copyReason == .pendingFullPayload)
        #expect(attempt.pendingPayloadWaitSeconds == 0)
        #expect(elapsed < .seconds(1))

        await gate.open()
        scene.materializing.release()
        await scene.store.flush()
    }

    /// The waited time rides beside the copy reason on the events a
    /// request's telemetry is read through — it is what the reason cost,
    /// and the reason keeps its name.
    @Test func theWaitedTimeRidesTheCopyReasonThroughRequestTelemetry() async throws {
        let bound = Duration.milliseconds(80)
        let scene = try makePendingLeafScene(label: "checkout-wait-telemetry", wait: bound)
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })
        let attempt = await scene.attempt()
        #expect(attempt.copyReason == .pendingFullPayload)

        let lookup = PrefixCacheDiagnostics.LookupEvent(
            reason: .hit(snapshotOffset: 8, totalTokens: 9, type: .leaf),
            promptTokens: 9, sharedPrefixLength: 9, skippedPrefillTokens: 8,
            newTokensToPrefill: 1, lookupMs: 0, restoreMs: 0, plannedCheckpoints: [],
            restoreMode: "copy", copyReason: attempt.copyReason,
            copyWaitSeconds: attempt.pendingPayloadWaitSeconds)
        let lookupFields = Dictionary(uniqueKeysWithValues: lookup.fields)
        #expect(lookupFields["copyReason"] == "pendingFullPayload")
        #expect((Double(lookupFields["copyWaitMs"] ?? "") ?? 0) >= 80)

        var report = LeafStorePhase.Report()
        report.restoreCopyReason = attempt.copyReason
        report.restoreCopyWaitSeconds = attempt.pendingPayloadWaitSeconds
        let reportFields = Dictionary(uniqueKeysWithValues: report.fields)
        #expect(reportFields["copyReason"] == "pendingFullPayload")
        #expect((Double(reportFields["copyWaitMs"] ?? "") ?? 0) >= 80)

        scene.materializing.release()
        await scene.store.flush()
    }

    @Test(arguments: [
        "system", "branch", "immutable", "rotating", "window", "untrimmable", "quantized", "image",
    ])
    func unsafeRestorePointsKeepTheCopyPath(reason: String) async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let key = CachePartitionKey(
            modelID: "fallback", kvBits: reason == "quantized" ? 8 : nil, kvGroupSize: 64)
        let layer: any KVCache
        switch reason {
        case "rotating": layer = RotatingKVCache(maxSize: 32)
        case "window": layer = AdvanceLimitedCache(limit: 5)
        case "untrimmable": layer = AdvanceLimitedCache(limit: -1)
        default: layer = KVCacheSimple()
        }
        _ = layer.update(keys: MLXArray.ones([1, 1, 8, 64]), values: MLXArray.ones([1, 1, 8, 64]))
        eval(layer)
        let previous = FinalGenerationCache([layer])
        let body = try #require(
            reason == "system" || reason == "immutable"
                ? HybridCacheSnapshot.capture(
                    cache: previous.cache, offset: 8, type: reason == "system" ? .system : .leaf)
                : previous.moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        let tree = store.getOrCreateTree(for: key)
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(body, on: node)
        if reason == "branch" { _ = tree.insertPath(tokens: tokens + [99]) }
        let resolved = PrefixCacheManager.Resolved(
            lookup: manager.lookup(tokens: tokens + [9], partitionKey: key),
            hydratedFromSSD: false, hydrationSeconds: 0)
        let attempt = await LeafCheckout.attempt(
            resolved: resolved, tokens: tokens + [9], maximumAdvance: 10,
            identityKeySpace: reason != "image", prefixCache: manager,
            context: .init(
                requestID: UUID(), modelID: key.modelID, kvBits: key.kvBits, kvGroupSize: 64))
        let expected: String
        switch reason {
        case "system", "branch": expected = "checkpoint"
        case "immutable": expected = "immutableBody"
        case "window", "untrimmable": expected = "untrimmable"
        case "image": expected = "imageKeySpace"
        case "quantized": expected = "quantized"
        default: expected = "rotating"
        }
        #expect(attempt.owner == nil)
        #expect(attempt.copyReason?.rawValue == expected)
        #expect(tree.leaseCount == 0)
        let copy = try #require(resolved.lookup.restoreCache())
        #expect(copy[0] as AnyObject !== layer as AnyObject)
        #expect(backingAddress(copy[0].state[0]) != backingAddress(body.layers[0].state[0]))
    }

    @Test func rewindRestoresRecurrentMetadataAndAttentionAfterGrowth() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let kv = KVCacheSimple()
        kv.step = 4
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let recurrent = MambaCache(leftPadding: [2])
        recurrent[1] = MLXArray([Float(3), 4])
        recurrent.prepare(lengths: [7])
        recurrent.offset = 8
        let previous = FinalGenerationCache([kv, recurrent])
        let body = try #require(previous.moveSnapshot(offset: 8))
        let expected = body.layers.map {
            ($0.metaState, $0.state.map { $0.asData(access: .copy).data })
        }
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let request = try #require(
            await LeafCheckout.attempt(
                resolved: .init(
                    lookup: manager.lookup(tokens: tokens + [9], partitionKey: key),
                    hydratedFromSSD: false, hydrationSeconds: 0),
                tokens: tokens + [9], maximumAdvance: 20, identityKeySpace: true,
                prefixCache: manager,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
            ).owner)
        #expect(request.rewindStateBytes == 8)
        _ = kv.update(
            keys: MLXArray.zeros([1, 1, 10, 64]), values: MLXArray.zeros([1, 1, 10, 64]))
        recurrent[1] = MLXArray([Float(99), 100])
        recurrent.advance(10)
        recurrent.offset = 18
        eval(request.cache)
        await request.rewindIfNeeded()
        #expect(request.cache.isEmpty)
        #expect(request.rewindStateBytes == 0)
        let restored = try #require(
            manager.lookup(tokens: tokens + [9], partitionKey: key).snapshot)
        #expect(restored.tokenOffset == 8)
        for (layer, before) in zip(restored.layers, expected) {
            #expect(layer.offset == 8)
            #expect(layer.metaState == before.0)
            #expect(layer.state.map { $0.asData(access: .copy).data } == before.1)
        }
    }

    @Test func checkoutMovesTheResidentObjectsAndCheckInReleasesTheRequest() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let previous = FinalGenerationCache([kv])
        let body = try #require(previous.moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let resolved = PrefixCacheManager.Resolved(
            lookup: manager.lookup(tokens: tokens + [9], partitionKey: key),
            hydratedFromSSD: false, hydrationSeconds: 0)
        let attempt = await LeafCheckout.attempt(
            resolved: resolved, tokens: tokens + [9], maximumAdvance: 10,
            identityKeySpace: true, prefixCache: manager,
            context: .init(requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64))
        let request = try #require(attempt.owner)
        let tree = try #require(store.tree(for: key))
        #expect(request.cache[0] as AnyObject === kv)
        #expect(body.layers.isEmpty, "retained lookup handles must release the frozen array views")
        #expect(tree.allSnapshotNodes().allSatisfy { $0.state.body == nil })
        #expect(manager.totalSnapshotBytes == 4_096)
        #expect(manager.budgetFloorBytes() == 4_096)

        _ = kv.update(keys: MLXArray.ones([1, 1, 1, 64]), values: MLXArray.ones([1, 1, 1, 64]))
        eval(kv)
        let extended = try #require(request.moveSnapshot(offset: 9))
        #expect(await request.checkIn(extended, tokens: tokens + [9]))
        #expect(request.cache.isEmpty)
        #expect(tree.leaseCount == 0)
        #expect(manager.lookup(tokens: tokens + [9], partitionKey: key).snapshotTokenOffset == 9)
        #expect(manager.totalSnapshotBytes == 4_608)
    }
}

nonisolated private final class AdvanceLimitedCache: KVCacheSimple {
    let limit: Int
    init(limit: Int) { self.limit = limit; super.init() }
    override func isTrimmable(after positions: Int) -> Bool { positions <= limit }
}
