//
//  ChainPrefixRestoreTests.swift
//  tesseractTests
//
//  **Chain-Prefix Restore** (docs/adr/0012, issue #96): a node whose
//  leaf entry a **Leaf Extension Admission** consumed at writer commit
//  keeps a tree-side restore point into the owning chain — the boundary
//  stays restorable from the chain's leading **Snapshot Segment**s
//  instead of going dark, which is what froze the incident's restore
//  floor one interrupt behind.
//
//  Three layers, mirroring the production seams:
//  1. Tree — hittability, self-heal keeps pointed nodes, clear heals.
//  2. Store/router — fold-time conversion, transitive re-own, eager
//     clears on owner death, prefix hydration composition.
//  3. Manager/policy — the chainPrefixHit lookup channel, promote, and
//     ADR-0011 recovery-cost pricing by prefix bytes.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

private let testFingerprint = String(repeating: "a", count: 64)

// MARK: - 1. Tree: hittability + self-heal guards

@MainActor
struct ChainPrefixRestoreTreeTests {

    private func makePoint(
        owner: String = "owner-1",
        boundary: Int = 5,
        bytes: Int = 1_024
    ) -> ChainPrefixRestorePoint {
        ChainPrefixRestorePoint(
            ownerSnapshotID: owner,
            boundaryOffset: boundary,
            prefixBytes: bytes,
            checkpointType: .leaf,
            partitionDigest: "deadbeef"
        )
    }

    @Test func pointedNodeIsHittableOnlyWhenRefsAreIncluded() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.attachChainPrefixRestorePoint(node: node, point: makePoint())

        // Body-less and ref-less: invisible to the RAM-only walk,
        // hittable alongside committed refs.
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3, 4, 5, 9]) == nil)
        let hit = tree.findBestSnapshot(
            tokens: [1, 2, 3, 4, 5, 9], includeSnapshotRefs: true
        )
        #expect(hit?.node === node)
        #expect(hit?.sharedPrefixLength == 5)
    }

    @Test func dropRefKeepsPointedNodeInTree() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3])
        tree.attachChainPrefixRestorePoint(node: node, point: makePoint(boundary: 3))
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 3)
        node.state = .pendingDropped(ref)
        let countBefore = tree.nodeCount

        // State 3 → empty would self-heal the node away; the point must
        // keep it addressable.
        let effect = tree.dropRef(node: node, expectedID: ref.snapshotID)

        #expect(effect == .becameEmpty)
        #expect(node.state.isEmpty)
        #expect(node.parent != nil)
        #expect(tree.nodeCount == countBefore)
    }

    @Test func clearingThePointSelfHealsAnEmptyNode() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3])
        tree.attachChainPrefixRestorePoint(node: node, point: makePoint(boundary: 3))
        let countBefore = tree.nodeCount

        tree.clearChainPrefixRestorePoint(node: node)

        #expect(node.chainPrefixRestorePoint == nil)
        #expect(tree.nodeCount < countBefore)
        #expect(
            tree.findBestSnapshot(
                tokens: [1, 2, 3], includeSnapshotRefs: true
            ) == nil)
    }

    @Test func conversionDiscardsRefAndKeepsBoundaryAddressable() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3, 4, 5])
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 5)
        node.state = .ssdOnly(ref)

        let point = tree.convertConsumedBaseToChainPrefixRestorePoint(
            node: node, ownerSnapshotID: "head-1"
        )

        #expect(point.ownerSnapshotID == "head-1")
        #expect(point.boundaryOffset == 5)
        #expect(point.prefixBytes == ref.bytesOnDisk)
        #expect(node.state.isEmpty)
        #expect(node.chainPrefixRestorePoint == point)
        #expect(node.parent != nil)
    }
}

// MARK: - 2. Store/router: fold conversion, re-own, eager clears, hydration

@MainActor
struct ChainPrefixRestoreRouterTests {

    private typealias Fixture = (
        root: URL,
        store: TieredSnapshotStore,
        tree: TokenRadixTree,
        key: CachePartitionKey
    )

    private func makeFixture() -> Fixture {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("chain-prefix-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = TieredSnapshotStore(
            ssdConfig: SSDPrefixCacheConfig(
                enabled: true,
                rootURL: root,
                budgetBytes: 1_000_000,
                maxPendingBytes: 10_000_000
            ))
        let key = CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint
        )
        store.registerPartition(
            PartitionMeta(
                modelID: key.modelID,
                modelFingerprint: testFingerprint,
                kvBits: key.kvBits,
                kvGroupSize: key.kvGroupSize,
                createdAt: 100_000,
                schemaVersion: SnapshotManifestSchema.currentVersion
            ), for: key)
        return (root, store, store.getOrCreateTree(for: key), key)
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func waitUntil(
        timeout: Duration = .seconds(5),
        _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let start = ContinuousClock.now
        while ContinuousClock.now - start < timeout {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    /// Admit + commit a leaf at `tokens`, returning its node and ref.
    /// `extending` drives the writer's fold against an earlier leaf.
    private func admitCommittedLeaf(
        _ fixture: Fixture,
        tokens: [Int],
        bytes: Int,
        extending: SnapshotExtension? = nil
    ) async throws -> (node: RadixTreeNode, ref: SnapshotRef) {
        let node = fixture.tree.insertPath(tokens: tokens)
        fixture.tree.storeSnapshot(
            PrefixCacheTestFixtures.makeUniformSnapshot(offset: tokens.count, type: .leaf),
            on: node
        )
        let ref = try #require(
            fixture.store.admitSnapshot(
                node: node,
                tree: fixture.tree,
                partitionKey: fixture.key,
                pathFromRoot: tokens,
                snapshot: HybridCacheSnapshot(
                    tokenOffset: tokens.count,
                    layers: [],
                    checkpointType: .leaf,
                    memoryBytes: 0,
                    createdAt: ContinuousClock().now
                ),
                payload: PrefixCacheTestFixtures.makeLeafPayload(
                    bytes: bytes, tokenOffset: tokens.count, extending: extending
                )
            ))
        await fixture.store.flush()
        let committed = await waitUntil { node.state.committed }
        #expect(committed, "leaf at \(tokens.count) must commit")
        return (node, ref)
    }

    /// The incident shape in miniature: base leaf at offset 5, fold at
    /// offset 9 consuming it.
    private func makeFoldedPair(
        _ fixture: Fixture
    ) async throws -> (base: RadixTreeNode, head: RadixTreeNode, headRef: SnapshotRef) {
        let basePath = Array(1...5)
        let (baseNode, baseRef) = try await admitCommittedLeaf(
            fixture, tokens: basePath, bytes: 800
        )
        fixture.tree.dropBody(node: baseNode)
        #expect(baseNode.state.label == "ssdOnly")

        let (headNode, headRef) = try await admitCommittedLeaf(
            fixture, tokens: Array(1...9), bytes: 200,
            extending: SnapshotExtension(
                baseSnapshotID: baseRef.snapshotID, baseOffset: 5
            )
        )
        let converted = await waitUntil { baseNode.chainPrefixRestorePoint != nil }
        #expect(converted, "fold commit must convert the base ref into a restore point")
        return (baseNode, headNode, headRef)
    }

    @Test func foldConvertsConsumedBaseIntoRestorePoint() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }

        let (baseNode, _, headRef) = try await makeFoldedPair(fixture)

        let point = try #require(baseNode.chainPrefixRestorePoint)
        #expect(point.ownerSnapshotID == headRef.snapshotID)
        #expect(point.boundaryOffset == 5)
        #expect(point.prefixBytes == 800)
        #expect(point.checkpointType == .leaf)
        // The base's own identity is gone, its state empty — but the
        // boundary stays addressable on a divergent future, which is the
        // rewind geometry the incident could not serve.
        #expect(baseNode.state.isEmpty)
        let hit = fixture.tree.findBestSnapshot(
            tokens: [1, 2, 3, 4, 5, 99, 98], includeSnapshotRefs: true
        )
        #expect(hit?.node === baseNode)
    }

    @Test func lookupSurfacesChainPrefixHitAtTheBoundary() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 1_000_000, tieredStore: fixture.store
        )

        let (_, headNode, headRef) = try await makeFoldedPair(fixture)

        // A divergent future past the boundary — the post-interrupt
        // steering shape — resolves at the restore point.
        let rewind = manager.lookup(
            tokens: [1, 2, 3, 4, 5, 99, 98], partitionKey: fixture.key
        )
        guard case .chainPrefixHit(let ctx) = rewind.reason else {
            Issue.record("expected chainPrefixHit, got \(rewind.reason)")
            return
        }
        #expect(ctx.point.ownerSnapshotID == headRef.snapshotID)
        #expect(rewind.snapshotTokenOffset == 5)

        // The straight-line future still prefers the head's own body.
        let extend = manager.lookup(
            tokens: Array(1...9) + [10], partitionKey: fixture.key
        )
        guard case .hit(let offset, _, _) = extend.reason else {
            Issue.record("expected RAM hit on the head, got \(extend.reason)")
            return
        }
        #expect(offset == 9)
        #expect(extend.snapshot != nil)
        _ = headNode
    }

    @Test func deletingTheOwnerChainClearsDependentPoints() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }

        let (baseNode, _, headRef) = try await makeFoldedPair(fixture)
        #expect(baseNode.chainPrefixRestorePoint != nil)

        fixture.store.deleteSnapshot(snapshotID: headRef.snapshotID)

        #expect(baseNode.chainPrefixRestorePoint == nil)
        // Nothing left on the node — the lookup degrades shallower
        // instead of resolving a dangling point.
        #expect(
            fixture.tree.findBestSnapshot(
                tokens: [1, 2, 3, 4, 5, 99], includeSnapshotRefs: true
            ) == nil)
    }

    @Test func lruCutOfTheOwnerClearsDependentsThroughTheDropCallback() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }

        let (baseNode, _, headRef) = try await makeFoldedPair(fixture)

        // The SSD tier's eviction of the owner reaches the router as a
        // drop callback against a committed resident — dependents clear
        // through the same eager backing-loss plumbing as the ref.
        fixture.store.markSnapshotRefDropped(
            id: headRef.snapshotID, reason: .evictedByLRU
        )

        #expect(baseNode.chainPrefixRestorePoint == nil)
    }

    @Test func aSecondFoldReownsPointsTransitively() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }

        let (baseNode, headNode, headRef) = try await makeFoldedPair(fixture)
        fixture.tree.dropBody(node: headNode)
        #expect(headNode.state.label == "ssdOnly")

        let (_, head2Ref) = try await admitCommittedLeaf(
            fixture, tokens: Array(1...12), bytes: 100,
            extending: SnapshotExtension(
                baseSnapshotID: headRef.snapshotID, baseOffset: 9
            )
        )
        let reowned = await waitUntil {
            baseNode.chainPrefixRestorePoint?.ownerSnapshotID == head2Ref.snapshotID
        }

        // The first boundary's point follows the chain to its new head,
        // and the consumed first head leaves a point of its own.
        #expect(reowned, "the second fold must re-own the first boundary's point")
        #expect(baseNode.chainPrefixRestorePoint?.boundaryOffset == 5)
        let headPoint = try #require(headNode.chainPrefixRestorePoint)
        #expect(headPoint.ownerSnapshotID == head2Ref.snapshotID)
        #expect(headPoint.boundaryOffset == 9)

        // Both boundaries sit on the new owner's segment grid.
        let descriptor = try #require(
            fixture.store.ssdStoreForTesting?
                .residentDescriptorForTesting(id: head2Ref.snapshotID)
        )
        #expect(descriptor.inheritedSegments.map(\.tokenOffset) == [5, 9])
    }

    // MARK: - Warm start reconstruction (issue #99)

    /// Simulate a restart: open a fresh store + manager over the same SSD
    /// root and warm-start it. Callers flush (and optionally corrupt the
    /// manifest) first.
    private func restartedManager(
        _ fixture: Fixture
    ) async throws -> (manager: PrefixCacheManager, store: TieredSnapshotStore) {
        let store = TieredSnapshotStore(
            ssdConfig: SSDPrefixCacheConfig(
                enabled: true,
                rootURL: fixture.root,
                budgetBytes: 1_000_000,
                maxPendingBytes: 10_000_000
            ))
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 1_000_000, tieredStore: store
        )
        try await manager.warmStart(modelFingerprint: testFingerprint)
        return (manager, store)
    }

    @Test func warmStartReconstructsRestorePointsFromTheManifest() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let (_, _, headRef) = try await makeFoldedPair(fixture)
        await fixture.store.flush()

        let (manager, _) = try await restartedManager(fixture)

        // The rewind floor survives the restart: a divergent future past
        // the consumed base's extent resolves at the reconstructed point.
        let rewind = manager.lookup(
            tokens: [1, 2, 3, 4, 5, 99, 98], partitionKey: fixture.key
        )
        guard case .chainPrefixHit(let ctx) = rewind.reason else {
            Issue.record("expected chainPrefixHit after restart, got \(rewind.reason)")
            return
        }
        #expect(ctx.point.ownerSnapshotID == headRef.snapshotID)
        #expect(ctx.point.boundaryOffset == 5)
        #expect(ctx.point.prefixBytes == 800)
        #expect(ctx.point.checkpointType == .leaf)
        #expect(rewind.snapshotTokenOffset == 5)

        // The straight-line future still resolves through the head's own
        // restored SSD ref, deeper than the boundary.
        let straight = manager.lookup(
            tokens: Array(1...9) + [10], partitionKey: fixture.key
        )
        guard case .ssdHit(let headCtx) = straight.reason else {
            Issue.record("expected ssdHit on the head, got \(straight.reason)")
            return
        }
        #expect(headCtx.snapshotRef.snapshotID == headRef.snapshotID)
    }

    @Test func warmStartDirectoryWalkRebuildReconstructsRestorePoints() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let (_, _, headRef) = try await makeFoldedPair(fixture)
        await fixture.store.flush()

        // Corrupt the manifest so warm start takes the directory-walk
        // rebuild — the points must come back from per-file headers alone.
        let manifestURL = fixture.root.appendingPathComponent("manifest.json")
        try Data("not json".utf8).write(to: manifestURL)

        let (manager, _) = try await restartedManager(fixture)

        let rewind = manager.lookup(
            tokens: [1, 2, 3, 4, 5, 99, 98], partitionKey: fixture.key
        )
        guard case .chainPrefixHit(let ctx) = rewind.reason else {
            Issue.record("expected chainPrefixHit after rebuild, got \(rewind.reason)")
            return
        }
        #expect(ctx.point.ownerSnapshotID == headRef.snapshotID)
        #expect(ctx.point.boundaryOffset == 5)
    }

    @Test func warmStartCondemnedChainsGetNoRestorePoints() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let (_, _, headRef) = try await makeFoldedPair(fixture)
        await fixture.store.flush()

        // Break the chain: delete the inherited base segment's file, then
        // corrupt the manifest so the rebuild must judge the chain by its
        // on-disk links. The head is condemned — no ref, no points.
        let descriptor = try #require(
            fixture.store.ssdStoreForTesting?
                .residentDescriptorForTesting(id: headRef.snapshotID)
        )
        let basePath = try #require(descriptor.inheritedSegments.first?.fileRelativePath)
        try FileManager.default.removeItem(
            at: fixture.root.appendingPathComponent(basePath)
        )
        try Data("not json".utf8).write(
            to: fixture.root.appendingPathComponent("manifest.json")
        )

        let (manager, store) = try await restartedManager(fixture)

        let rewind = manager.lookup(
            tokens: [1, 2, 3, 4, 5, 99, 98], partitionKey: fixture.key
        )
        if case .chainPrefixHit = rewind.reason {
            Issue.record("condemned chain must not surface a restore point")
        }
        #expect(
            store.tree(for: fixture.key)?.findBestSnapshot(
                tokens: Array(1...9), includeSnapshotRefs: true
            ) == nil)
    }

    @Test func warmStartReconstructedPointsClearWhenTheOwnerDies() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let (_, _, headRef) = try await makeFoldedPair(fixture)
        await fixture.store.flush()

        let (manager, store) = try await restartedManager(fixture)
        let rewind = manager.lookup(
            tokens: [1, 2, 3, 4, 5, 99, 98], partitionKey: fixture.key
        )
        guard case .chainPrefixHit(let ctx) = rewind.reason else {
            Issue.record("expected chainPrefixHit after restart, got \(rewind.reason)")
            return
        }

        // The reconstructed dependents index must reach the point when
        // the owner dies — same eager clearing as the in-session channel.
        store.deleteSnapshot(snapshotID: headRef.snapshotID)
        #expect(ctx.node.chainPrefixRestorePoint == nil)
        #expect(
            store.tree(for: fixture.key)?.findBestSnapshot(
                tokens: [1, 2, 3, 4, 5, 99], includeSnapshotRefs: true
            ) == nil)
    }
}

// MARK: - 3. Prefix hydration composition (store level, real arrays)

struct ChainPrefixHydrationTests {

    private func makeScratch() -> (config: SSDPrefixCacheConfig, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("chain-prefix-hydrate-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (
            SSDPrefixCacheConfig(
                enabled: true,
                rootURL: root,
                budgetBytes: 1_000_000,
                maxPendingBytes: 10_000_000
            ), root
        )
    }

    @Test func prefixHydrationComposesOnlyTheLeadingSegments() async throws {
        let (config, root) = makeScratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: .milliseconds(20),
            onCommit: { _ in },
            onDrop: { _, _ in }
        )
        let digest = "abcd1234"
        store.registerPartition(
            PartitionMeta(
                modelID: "mlx-community/Qwen3-4B-4bit",
                modelFingerprint: testFingerprint,
                kvBits: 8,
                kvGroupSize: 64,
                createdAt: 100_000,
                schemaVersion: SnapshotManifestSchema.currentVersion
            ), digest: digest)

        func descriptor(
            id: String, bytes: Int, tokenOffset: Int, segmentBaseOffset: Int = 0
        ) -> PersistedSnapshotDescriptor {
            PersistedSnapshotDescriptor(
                snapshotID: id,
                partitionDigest: digest,
                pathFromRoot: Array(1...tokenOffset),
                tokenOffset: tokenOffset,
                checkpointType: "leaf",
                bytes: bytes,
                segmentBaseOffset: segmentBaseOffset,
                createdAt: 100_000,
                lastAccessAt: 0,
                fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                    snapshotID: id, partitionDigest: digest
                ),
                schemaVersion: SnapshotManifestSchema.currentVersion
            )
        }

        // The base capture at offset 4: positions 0..<4 of the full KV
        // state, plus a distinct recurrent state (zeros).
        let count = 1 * 2 * 7 * 8
        let keysFull = MLXArray((0..<count).map { Float($0) }).reshaped([1, 2, 7, 8])
        let valuesFull = MLXArray((0..<count).map { Float($0) + 0.5 }).reshaped([1, 2, 7, 8])

        let kvBase = KVCacheSimple()
        kvBase.state = [
            keysFull[.ellipsis, ..<4, 0...],
            valuesFull[.ellipsis, ..<4, 0...],
        ]
        let mambaBase = MambaCache()
        mambaBase.state = [MLXArray.zeros([1, 3, 16]), MLXArray.zeros([1, 4, 8, 8])]
        let baseSnapshot = HybridCacheSnapshot.capture(
            cache: [kvBase, mambaBase], offset: 4, type: .leaf
        )!

        let kvHead = KVCacheSimple()
        kvHead.state = [keysFull, valuesFull]
        let mambaHead = MambaCache()
        mambaHead.state = [MLXArray.ones([1, 3, 16]), MLXArray.ones([1, 4, 8, 8])]
        let headSnapshot = HybridCacheSnapshot.capture(
            cache: [kvHead, mambaHead], offset: 7, type: .leaf
        )!

        let basePayload = ServerCompletion.extractSnapshotPayload(baseSnapshot)
        guard
            case .accepted = store.tryEnqueue(
                payload: basePayload,
                descriptor: descriptor(id: "base", bytes: basePayload.totalBytes, tokenOffset: 4)
            )
        else {
            Issue.record("base enqueue rejected")
            return
        }
        await store.flushAsync()

        let headPayload = ServerCompletion.extractSnapshotPayload(
            headSnapshot,
            extending: SnapshotExtension(baseSnapshotID: "base", baseOffset: 4)
        )
        guard
            case .accepted = store.tryEnqueue(
                payload: headPayload,
                descriptor: descriptor(
                    id: "head", bytes: headPayload.totalBytes,
                    tokenOffset: 7, segmentBaseOffset: 4
                )
            )
        else {
            Issue.record("extension enqueue rejected")
            return
        }
        await store.flushAsync()

        let point = ChainPrefixRestorePoint(
            ownerSnapshotID: "head",
            boundaryOffset: 4,
            prefixBytes: basePayload.totalBytes,
            checkpointType: .leaf,
            partitionDigest: digest
        )
        let restored = try #require(
            store.loadSyncPrefix(
                point: point, expectedFingerprint: testFingerprint
            ))

        // The composed body is the *base* capture exactly: the sliced
        // layer carries only the leading extent, and the whole-state
        // layer takes the last *included* segment's copy — the recurrent
        // state as of the boundary, not the head's deeper one.
        #expect(restored.tokenOffset == 4)
        let restoredKV = restored.layers[0]
        #expect(restoredKV.state[0].shape == [1, 2, 4, 8])
        #expect(
            restoredKV.state[0].asData(access: .copy).data
                == keysFull[.ellipsis, ..<4, 0...].asData(access: .copy).data
        )
        let restoredMamba = restored.layers[1]
        #expect(
            restoredMamba.state[0].asData(access: .copy).data
                == MLXArray.zeros([1, 3, 16]).asData(access: .copy).data
        )
    }

    @Test func staleBoundaryOffTheSegmentGridMissesCleanly() async throws {
        let (config, root) = makeScratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: .milliseconds(20),
            onCommit: { _ in },
            onDrop: { _, _ in }
        )

        // Unknown owner and off-grid boundary both return nil without
        // condemning anything — the caller clears the point and the next
        // lookup degrades shallower.
        let point = ChainPrefixRestorePoint(
            ownerSnapshotID: "ghost",
            boundaryOffset: 4,
            prefixBytes: 100,
            checkpointType: .leaf,
            partitionDigest: "abcd1234"
        )
        #expect(
            store.loadSyncPrefix(
                point: point, expectedFingerprint: testFingerprint
            ) == nil)
    }
}

// MARK: - 4. Manager/policy: promote + recovery-cost pricing

@MainActor
struct ChainPrefixRestorePolicyTests {

    @Test func promoteChainPrefixStoresTheBodyAndKeepsThePoint() {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let key = CachePartitionKey(modelID: "m", kvBits: nil, kvGroupSize: 64)
        let tree = store.getOrCreateTree(for: key)
        let node = tree.insertPath(tokens: [1, 2, 3, 4, 5])
        let point = ChainPrefixRestorePoint(
            ownerSnapshotID: "owner",
            boundaryOffset: 5,
            prefixBytes: 4_096,
            checkpointType: .leaf,
            partitionDigest: "deadbeef"
        )
        tree.attachChainPrefixRestorePoint(node: node, point: point)

        manager.promoteChainPrefix(
            node: node,
            snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 5, type: .leaf),
            partitionKey: key
        )

        // The body lands RAM-only — the backing stays borrowed from the
        // owner chain — and the point survives for the next eviction.
        #expect(node.state.label == "ramOnly")
        #expect(node.chainPrefixRestorePoint == point)
    }

    @Test func recoveryCostPricesPointedNodesAsPrefixHydration() {
        let tree = TokenRadixTree()
        // Two same-shape, same-recency bodies deep in a conversation:
        // one chain-prefix-backed, one bare. Under the recovery-cost
        // blend the backed body must be the cheaper victim — hydrating
        // its prefix bytes costs far less than re-prefilling 4k tokens.
        let pointed = tree.insertPath(tokens: Array(1...4_096))
        tree.storeSnapshot(
            PrefixCacheTestFixtures.makeUniformSnapshot(offset: 4_096, type: .leaf),
            on: pointed
        )
        tree.attachChainPrefixRestorePoint(
            node: pointed,
            point: ChainPrefixRestorePoint(
                ownerSnapshotID: "owner",
                boundaryOffset: 4_096,
                prefixBytes: 1 << 20,
                checkpointType: .leaf,
                partitionDigest: "deadbeef"
            ))
        let bare = tree.insertPath(tokens: Array(5_001...9_096))
        tree.storeSnapshot(
            PrefixCacheTestFixtures.makeUniformSnapshot(offset: 4_096, type: .leaf),
            on: bare
        )
        let now = ContinuousClock.now
        pointed.lastAccessTime = now
        bare.lastAccessTime = now

        let scores = EvictionPolicy.computeScores(
            candidates: [pointed, bare],
            now: now,
            config: EvictionConfiguration(alpha: 1.0)
        )

        // Min-max over two candidates: the cheap-to-recover body
        // normalizes to 0, the terminal one to 1.
        #expect(scores[0].normalizedFlopEfficiency < scores[1].normalizedFlopEfficiency)
        #expect(scores[0].utility < scores[1].utility)
    }
}
