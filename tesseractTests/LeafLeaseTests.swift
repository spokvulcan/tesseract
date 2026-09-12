import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct LeafLeaseTests {
    private let key = CachePartitionKey(
        modelID: "leaf-lease", kvBits: nil, kvGroupSize: 64,
        modelFingerprint: String(repeating: "b", count: 64))

    private func snapshot(offset: Int = 8) throws -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, offset, 64]), MLXArray.ones([1, 1, offset, 64])]
        let recurrent = MambaCache()
        recurrent.state = [MLXArray.ones([8]), MLXArray.ones([8])]
        var cache: [any KVCache] = [kv, recurrent]
        eval(cache)
        return try #require(HybridCacheSnapshot.captureMoving(cache: &cache, offset: offset))
    }

    @Test func bodyDropRefusesLeaseWithoutCopyingAndResumesAfterRewind() throws {
        let tree = TokenRadixTree()
        let tokens = Array(1...8)
        let node = tree.insertPath(tokens: tokens)
        let body = try snapshot()
        let addresses = body.layers.flatMap(\.state).map(backingAddress)
        tree.storeSnapshot(body, on: node)
        let context = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let lease = try #require(tree.beginLeafLease(on: node, context: context))

        #expect(tree.totalSnapshotBytes == 4_160)
        #expect(tree.leasedBytes == 4_160)
        #expect(tree.leaseCount == 1)
        #expect(tree.dropBody(node: node).effect == .ignored(.leased))
        #expect(node.state.body?.layers.flatMap(\.state).map(backingAddress) == addresses)
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        #expect(tree.leaseCount == 0)
        #expect(tree.leasedBytes == 0)
        #expect(tree.dropBody(node: node).droppedBodyBytes == 4_160)
        #expect(tree.totalSnapshotBytes == 0)
    }

    @Test func leaseSurvivesPinAgeOutPressureAndRAMClearUntilExplicitReturn() throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let tokens = Array(1...8)
        let body = try snapshot()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let requestID = UUID()
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: requestID, modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        manager.pinRestorePath(node: node, requestID: requestID)
        let freshTokens = Array(20...27)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: freshTokens, snapshot: snapshot(), storage: .ramOnly,
                    partitionKey: key)))
        let fresh = try #require(
            tree.findBestSnapshot(tokens: freshTokens, updateAccess: false)?.node)
        for _ in 0..<12 { manager.pinRestorePath(node: fresh, requestID: UUID()) }
        manager.completeRequest(requestID: requestID)

        try #require(manager.budgetFloorBytes() == 8_320, "the expired pin cannot end the lease")
        manager.setMemoryBudget(0)
        #expect(tree.totalSnapshotBytes == 8_320)
        #expect(manager.clearRAMTier() == 0)
        #expect(tree.leaseCount == 1)
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        #expect(manager.clearRAMTier() == 4_160)
        #expect(tree.totalSnapshotBytes == 4_160, "only the pinned fresh leaf survives")
        #expect(manager.memoryTelemetryFacts()["treeLeasedBytes"] == "0")
        #expect(manager.memoryTelemetryFacts()["treeLeaseCount"] == "0")
    }

    @Test func pendingFullWriterWaitsForLeaseEvenWhenFlushForcesDrain() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-writer", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let sink = LeafLeaseLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }
        let tokens = Array(1...8)
        let body = try snapshot()
        let deferred = ServerCompletion.deferredPayload(for: body, extending: nil)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramAndSSD(deferred.payload),
                    partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        await gate.open()
        let flush = Task { await store.flush() }
        var observed: [String] = []
        #expect(
            await waitUntil {
                observed += sink.drain()
                return deferred.payload.isMaterialized
                    || observed.contains {
                        $0.contains("event=leafLeaseDeferred") && $0.contains(lease.id.uuidString)
                    }
            })
        #expect(!deferred.payload.isMaterialized)
        #expect(!deferred.owed.retainedArrays.isEmpty)
        #expect(manager.memoryTelemetryFacts()["ssdPendingPayloadBytes"] == "4160")
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        await flush.value
        #expect(await waitUntil { node.state.committed })
        #expect(deferred.payload.isMaterialized)
        #expect(deferred.owed.retainedArrays.isEmpty)
        #expect(manager.memoryTelemetryFacts()["ssdPendingPayloadBytes"] == "0")
    }

    @Test(arguments: [false, true])
    func queuedPromotionRefusesLeaseThenDemotionProceedsAfterRelease(promoteAfterRelease: Bool)
        async throws
    {
        var extractions = 0
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-promotion", ramBudgetBytes: 1_000_000,
            demotionPayloadExtractor: {
                extractions += 1
                return ServerCompletion.extractSnapshotPayload($0)
            }, adaptiveWriteEagerness: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let tokens = Array(1...8)
        let body = try snapshot()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        // Queue promotion BEFORE acquisition to exercise its asynchronous recheck.
        for _ in 0..<SSDWriteEagernessPolicy.hitCountThreshold {
            _ = manager.lookup(tokens: tokens, partitionKey: key)
        }
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        try await Task.sleep(for: .milliseconds(100))
        #expect(extractions == 0, "promotion must not even extract a leased body's arrays")
        #expect(
            !node.ssdPromotionAttempted, "lease deferral must not consume the one-shot promotion")
        let freshTokens = Array(20...27)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: freshTokens, snapshot: snapshot(), storage: .ramOnly,
                    partitionKey: key)))
        manager.setMemoryBudget(0)
        #expect(extractions == 0, "leased body cannot be demoted")
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        if promoteAfterRelease {
            _ = manager.lookup(tokens: tokens, partitionKey: key)
            #expect(await waitUntil { extractions == 1 })
            await store.flush()
            #expect(await waitUntil { node.state.committed })
            #expect(node.state.body != nil)
            return
        }
        // Make the other leaf freshest again, then drain the released body.
        _ = manager.lookup(tokens: freshTokens, partitionKey: key)
        manager.setMemoryBudget(0)
        #expect(extractions == 1)
        #expect(node.state.body == nil)
        #expect(node.state.ref != nil, "demotion persists before the body drops")
        await store.flush()
        #expect(await waitUntil { node.state.committed })
        #expect(tree.totalSnapshotBytes == 4_160)
    }

    @Test func checkInReconcilesGrowthAndRejectsStaleOrInvalidReturns() throws {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: Array(1...8))
        tree.storeSnapshot(try snapshot(), on: node)
        let other = tree.insertPath(tokens: Array(20...27))
        tree.storeSnapshot(try snapshot(), on: other)
        let context = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let lease = try #require(tree.beginLeafLease(on: node, context: context))
        let grown = try snapshot(offset: 12)
        #expect(
            !tree.endLeafLease(
                lease, on: other, returning: grown, tokens: Array(1...12), reason: .checkIn))
        #expect(
            !tree.endLeafLease(
                lease, on: node, returning: grown, tokens: Array(20...31), reason: .checkIn))
        #expect(tree.totalSnapshotBytes == 8_320)
        #expect(tree.leasedBytes == 4_160)
        try #require(
            tree.endLeafLease(
                lease, on: node, returning: grown, tokens: Array(1...12), reason: .checkIn))
        #expect(tree.totalSnapshotBytes == 10_368)
        #expect(
            tree.totalSnapshotBytes
                == tree.allSnapshotNodes().reduce(0) { $0 + $1.state.residentBodyBytes })
        #expect(tree.snapshotCount == 2)
        #expect(tree.leaseCount == 0)
        #expect(tree.leasedBytes == 0)
        #expect(node.state.body == nil)
        let returned = try #require(
            tree.findBestSnapshot(tokens: Array(1...12), updateAccess: false)?.node)
        #expect(
            returned.state.body?.layers.flatMap(\.state).map(backingAddress)
                == grown.layers.flatMap(\.state).map(backingAddress))
        let next = try #require(tree.beginLeafLease(on: returned, context: context))
        #expect(
            !tree.endLeafLease(
                lease, on: returned, returning: grown, tokens: Array(1...12), reason: .rewind))
        #expect(tree.leaseCount == 1)
        #expect(
            tree.endLeafLease(
                next, on: returned, returning: grown, tokens: Array(1...12), reason: .rewind))
    }

    @Test func mandatoryAdmissionReportsLeaseRefusalThenPersistsAfterReturn() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TieredSnapshotStore(
            ssdConfig: SSDPrefixCacheConfig(
                enabled: true, rootURL: root, budgetBytes: 1_000_000, maxPendingBytes: 1))
        let manager = PrefixCacheManager(memoryBudgetBytes: 0, tieredStore: store)
        let tokens = Array(1...8)
        let body = try snapshot()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        let payload = ServerCompletion.deferredPayload(for: body)
        let admission = try #require(
            SnapshotAdmission.leaf(
                storedTokens: tokens, snapshot: body, storage: .ramAndSSD(payload.payload),
                partitionKey: key))
        let refused = manager.admit(admission)
        #expect(refused.leaseRefusals == [lease.id])
        #expect(node.state.ref == nil)
        #expect(!payload.payload.isMaterialized)
        #expect(node.leafLease?.id == lease.id)
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .checkIn))
        #expect(manager.admit(admission).leaseRefusals.isEmpty)
        await store.flush()
        #expect(await waitUntil { node.state.committed })
        #expect(payload.payload.isMaterialized)
        #expect(manager.memoryTelemetryFacts()["ssdPendingPayloadBytes"] == "0")
    }

    @Test func admissionCannotReplaceOrSupersedeALeasedBody() throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let tokens = Array(1...8)
        let original = try snapshot()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: original, storage: .ramOnly, partitionKey: key))
        )
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        let replacement = try snapshot()
        tree.storeSnapshot(replacement, on: node)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: replacement, storage: .ramOnly,
                    partitionKey: key)))
        #expect(
            node.state.body?.layers.flatMap(\.state).map(ObjectIdentifier.init)
                == original.layers.flatMap(\.state).map(ObjectIdentifier.init))
        let descendant = try snapshot(offset: 12)
        let diagnostics = manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...12), snapshot: descendant, storage: .ramOnly,
                    partitionKey: key)))
        #expect(diagnostics.supersededLeaves.isEmpty)
        #expect(tree.totalSnapshotBytes == 10_368)
        #expect(
            tree.endLeafLease(lease, on: node, returning: original, tokens: tokens, reason: .rewind)
        )
        let after = manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...12), snapshot: descendant, storage: .ramOnly,
                    partitionKey: key)))
        #expect(after.supersededLeaves.count == 1)
        #expect(node.state.body == nil)
        #expect(tree.totalSnapshotBytes == 6_208)
    }

    @Test func growthReturnPreservesAPendingDestinationAndItsLease() throws {
        let tree = TokenRadixTree()
        let body = try snapshot()
        let node = tree.insertPath(tokens: Array(1...8))
        tree.storeSnapshot(body, on: node)
        let destination = tree.insertPath(tokens: Array(1...12))
        tree.storeSnapshot(try snapshot(offset: 12), on: destination)
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 12)
        tree.admit(node: destination, ref: ref)
        tree.dropBody(node: destination)
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        let count = tree.nodeCount
        #expect(
            !tree.endLeafLease(
                lease, on: node, returning: try snapshot(offset: 12), tokens: Array(1...12),
                reason: .checkIn))
        #expect(destination.state.body == nil)
        #expect(destination.state.refID == ref.snapshotID)
        #expect(node.leafLease?.id == lease.id)
        #expect(tree.nodeCount == count)
        #expect(tree.totalSnapshotBytes == 4_160)
        #expect(
            tree.endLeafLease(
                lease, on: node, returning: body, tokens: Array(1...8), reason: .rewind))
    }

    @Test func refusedAcquisitionAndReturnsReportRealLeaseIdentity() throws {
        let sink = LeafLeaseLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }
        let tree = TokenRadixTree()
        let body = try snapshot()
        let node = tree.insertPath(tokens: Array(1...8))
        tree.storeSnapshot(body, on: node)
        let owner = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let contender = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let lease = try #require(tree.beginLeafLease(on: node, context: owner))
        _ = sink.drain()
        #expect(tree.beginLeafLease(on: node, context: contender) == nil)
        let refusal = try #require(sink.drain().first { $0.contains("reason=alreadyLeased") })
        #expect(refusal.contains("leaseID=\(lease.id.uuidString)"))
        #expect(refusal.contains(contender.requestID.uuidString))
        #expect(refusal.contains("activeRequestID=\(owner.requestID.uuidString)"))
        let wrongTree = TokenRadixTree()
        #expect(
            !wrongTree.endLeafLease(
                lease, on: node, returning: body, tokens: Array(1...8), reason: .rewind))
        let rejectedReturn = try #require(sink.drain().first { $0.contains("reason=wrongTree") })
        #expect(rejectedReturn.contains("leaseID=\(lease.id.uuidString)"))
        #expect(rejectedReturn.contains(owner.requestID.uuidString))
        #expect(tree.leaseCount == 1)
        #expect(
            tree.endLeafLease(
                lease, on: node, returning: body, tokens: Array(1...8), reason: .rewind))
    }

    @Test func returnThroughAnotherTreeLeavesTheOwnerAndLeaseUntouched() throws {
        let owner = TokenRadixTree()
        let wrongTree = TokenRadixTree()
        let body = try snapshot()
        let tokens = Array(1...8)
        let node = owner.insertPath(tokens: tokens)
        owner.storeSnapshot(body, on: node)
        let lease = try #require(
            owner.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        #expect(
            !wrongTree.endLeafLease(
                lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        #expect(node.leafLease?.id == lease.id)
        #expect(owner.totalSnapshotBytes == 4_160)
        #expect(owner.leaseCount == 1)
        #expect(wrongTree.totalSnapshotBytes == 0)
        #expect(wrongTree.leaseCount == 0)
        #expect(
            owner.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
    }

    @Test func growthReturnCarriesPendingWriterExclusionToTheNewNode() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-growth-writer", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let sink = LeafLeaseLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }
        let grown = try snapshot(offset: 12)
        // Model backing-capacity reuse with prefix views over the same real
        // attention buffers, rather than allocating a second cache fixture.
        let layers = grown.layers.map { layer in
            HybridCacheSnapshot.LayerState(
                className: layer.className,
                state: layer.className == "KVCache"
                    ? layer.state.map { $0[.ellipsis, 0..<8, 0...] } : layer.state,
                metaState: layer.metaState, offset: 8)
        }
        eval(layers.flatMap(\.state))
        let original = HybridCacheSnapshot(
            tokenOffset: 8, layers: layers, checkpointType: .leaf, memoryBytes: 4_160,
            createdAt: .now)
        #expect(
            original.layers[0].state.map(backingAddress)
                == grown.layers[0].state.map(backingAddress))
        let pending = ServerCompletion.deferredPayload(for: original)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...8), snapshot: original,
                    storage: .ramAndSSD(pending.payload), partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let oldNode = try #require(
            tree.findBestSnapshot(tokens: Array(1...8), updateAccess: false)?.node)
        let context = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let first = try #require(tree.beginLeafLease(on: oldNode, context: context))
        #expect(
            tree.endLeafLease(
                first, on: oldNode, returning: grown, tokens: Array(1...12), reason: .checkIn))
        let newNode = try #require(
            tree.findBestSnapshot(tokens: Array(1...12), updateAccess: false)?.node)
        let next = try #require(tree.beginLeafLease(on: newNode, context: context))
        #expect(oldNode.leafLease == nil)
        await gate.open()
        var observed: [String] = []
        #expect(
            await waitUntil {
                observed += sink.drain()
                return pending.payload.isMaterialized
                    || observed.contains {
                        $0.contains("event=leafLeaseDeferred") && $0.contains(next.id.uuidString)
                    }
            })
        #expect(!pending.payload.isMaterialized)
        #expect(tree.totalSnapshotBytes == 6_208)
        #expect(
            tree.endLeafLease(
                next, on: newNode, returning: grown, tokens: Array(1...12), reason: .rewind))
        await store.flush()
        #expect(pending.payload.isMaterialized)
    }

    @Test func writerReadInProgressRefusesAcquisitionUntilMaterialized() async throws {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-reader-wins", ramBudgetBytes: 1_000_000)
        defer { try? FileManager.default.removeItem(at: root) }
        let barrier = LeafWriterBarrier()
        defer { barrier.open() }
        let body = try snapshot()
        let deferred = ServerCompletion.deferredPayload(for: body)
        let payload = SnapshotPayload(tokenOffset: 8, checkpointType: .leaf, totalBytes: 4_160) {
            barrier.wait()
            return deferred.payload.layers
        }
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramAndSSD(payload),
                    partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let context = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        #expect(await waitUntil { barrier.started })
        #expect(tree.beginLeafLease(on: node, context: context) == nil)
        #expect(tree.leaseCount == 0)
        barrier.open()
        await store.flush()
        let lease = try #require(tree.beginLeafLease(on: node, context: context))
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
    }

    @Test func growthReturnWaitsForATombstonedDestinationReader() async throws {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-destination-reader", ramBudgetBytes: 1_000_000)
        defer { try? FileManager.default.removeItem(at: root) }
        let body = try snapshot()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...8), snapshot: body, storage: .ramOnly,
                    partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let source = try #require(tree.findBestSnapshot(tokens: Array(1...8))?.node)
        let lease = try #require(
            tree.beginLeafLease(
                on: source,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        let grown = try snapshot(offset: 12)
        let barrier = LeafWriterBarrier()
        defer { barrier.open() }
        let deferred = ServerCompletion.deferredPayload(for: grown)
        let payload = SnapshotPayload(tokenOffset: 12, checkpointType: .leaf, totalBytes: 6_208) {
            barrier.wait()
            return deferred.payload.layers
        }
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...12), snapshot: grown, storage: .ramAndSSD(payload),
                    partitionKey: key)))
        let destination = try #require(tree.findBestSnapshot(tokens: Array(1...12))?.node)
        // Keep this boundary structural after its body and backing are removed.
        tree.insertPath(tokens: Array(1...12) + [20])
        tree.insertPath(tokens: Array(1...12) + [30])
        #expect(await waitUntil { barrier.started })
        store.deleteSnapshot(snapshotID: try #require(destination.state.refID))
        tree.dropBody(node: destination)
        tree.discardSnapshotRefAfterExplicitDelete(node: destination)
        let count = tree.nodeCount
        #expect(
            !tree.endLeafLease(
                lease, on: source, returning: grown, tokens: Array(1...12), reason: .checkIn))
        #expect(source.leafLease?.id == lease.id)
        #expect(destination.state.body == nil)
        #expect(tree.nodeCount == count)
        #expect(tree.totalSnapshotBytes == 4_160)
        barrier.open()
        await store.flush()
        #expect(
            tree.endLeafLease(
                lease, on: source, returning: grown, tokens: Array(1...12), reason: .checkIn))
        #expect(tree.leaseCount == 0)
        #expect(tree.totalSnapshotBytes == 6_208)
    }

    @Test(arguments: [false, true])
    func writerFailureReleasesReadClaim(ioFailure: Bool) async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-writer-failure", ramBudgetBytes: 1_000_000,
            ssdBudgetBytes: ioFailure ? 1_000_000 : 1,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let body = try snapshot()
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body,
                    storage: .ramAndSSD(ServerCompletion.extractSnapshotPayload(body)),
                    partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        if ioFailure {
            // Replace only this test's scratch directory with a file to force
            // directory creation to fail at the real writer's I/O boundary.
            try FileManager.default.removeItem(at: root)
            try Data().write(to: root)
        }
        await gate.open()
        await store.flush()
        #expect(await waitUntil { store.pendingSnapshotRefIDs.isEmpty })
        #expect(node.state.ref == nil)
        #expect(manager.memoryTelemetryFacts()["ssdPendingPayloadBytes"] == "0")
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
    }

    @Test func leasedBaseDefersItsSuffixButNotUnrelatedWrites() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "lease-base-order", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let body = try snapshot()
        let full = ServerCompletion.deferredPayload(for: body)
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramAndSSD(full.payload),
                    partitionKey: key)))
        let tree = try #require(store.tree(for: key))
        let node = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
        let lease = try #require(
            tree.beginLeafLease(
                on: node,
                context: .init(
                    requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)))
        let base = try #require(manager.extensionBase(tokens: Array(1...12), partitionKey: key))
        let grown = try snapshot(offset: 12)
        let suffix = ServerCompletion.deferredPayload(for: grown, extending: base)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(1...12), snapshot: grown,
                    storage: .ramAndSSD(suffix.payload), partitionKey: key)))
        let unrelated = try snapshot()
        let independent = ServerCompletion.deferredPayload(for: unrelated)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: Array(20...27), snapshot: unrelated,
                    storage: .ramAndSSD(independent.payload), partitionKey: key)))
        await gate.open()
        #expect(await waitUntil { independent.payload.isMaterialized })
        #expect(!full.payload.isMaterialized)
        #expect(!suffix.payload.isMaterialized)
        #expect(
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: .rewind))
        await store.flush()
        #expect(await waitUntil { store.pendingSnapshotRefIDs.isEmpty })
        #expect(full.payload.isMaterialized)
        #expect(suffix.payload.isMaterialized)
        #expect(manager.memoryTelemetryFacts()["ssdPendingPayloadBytes"] == "0")
    }
}

private nonisolated final class LeafWriterBarrier: @unchecked Sendable {
    private let lock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var entered = false
    var started: Bool { lock.withLock { entered } }
    func wait() {
        lock.withLock { entered = true }
        _ = semaphore.wait(timeout: .now() + 5)
    }
    func open() { semaphore.signal() }
}

private nonisolated final class LeafLeaseLineSink: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    var handler: @Sendable (String) -> Void {
        { [self] line in lock.withLock { lines.append(line) } }
    }
    func drain() -> [String] {
        lock.withLock {
            defer { lines.removeAll() }
            return lines
        }
    }
}
