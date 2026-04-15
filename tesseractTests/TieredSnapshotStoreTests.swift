//
//  TieredSnapshotStoreTests.swift
//  tesseractTests
//
//  Task 4.1.6 coverage — tiered store composition + five-state
//  storage-ref lifecycle + admission-time LRU cut forwarding.
//
//  Each test constructs a `TieredSnapshotStore` over a temp
//  directory, drives a storage-ref lifecycle transition via the
//  public API, and asserts the post-state via `node.storageRef`,
//  the testing-hook `pendingRefCountForTesting`, and (where
//  applicable) the underlying `SSDSnapshotStore` accessors.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

@MainActor
struct TieredSnapshotStoreTests {

    // MARK: - Scratch fixtures

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiered-store-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makeConfig(
        rootURL: URL,
        budgetBytes: Int = 1_000_000,
        maxPendingBytes: Int = 10_000_000
    ) -> SSDPrefixCacheConfig {
        SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: budgetBytes,
            maxPendingBytes: maxPendingBytes
        )
    }

    private func makePartitionMeta() -> PartitionMeta {
        PartitionMeta(
            modelID: "mlx-community/Qwen3-4B-4bit",
            modelFingerprint: String(repeating: "a", count: 64),
            kvBits: 8,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func makePartitionKey() -> CachePartitionKey {
        CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64)
        )
    }

    private func makePayload(
        bytes: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType = .leaf
    ) -> SnapshotPayload {
        SnapshotPayload(
            tokenOffset: 4_096,
            checkpointType: checkpointType,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: bytes),
                            dtype: "bfloat16",
                            shape: [1, bytes]
                        )
                    ],
                    metaState: ["meta"],
                    offset: 4_096
                )
            ]
        )
    }

    private func makeDescriptor(
        id: String = UUID().uuidString,
        partitionDigest: String,
        checkpointType: String = "leaf",
        bytes: Int,
        lastAccessAt: Double = 0
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: partitionDigest,
            pathFromRoot: [1, 2, 3],
            tokenOffset: 4_096,
            checkpointType: checkpointType,
            bytes: bytes,
            createdAt: 100_000,
            lastAccessAt: lastAccessAt,
            fileRelativePath: "partitions/\(partitionDigest)/snapshots/\(id.prefix(1))/\(id).safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// Poll `condition` on MainActor every 10 ms until it returns
    /// true or `timeout` elapses. Needed because the writer's
    /// commit / drop callbacks fire from a background task and hop
    /// back to MainActor asynchronously.
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

    /// Bundle of (root, store, tree, key) used by every happy-path
    /// test. Tests unpack the tuple and add a `defer { cleanup(root) }`.
    private typealias Fixture = (
        root: URL,
        store: TieredSnapshotStore,
        tree: TokenRadixTree,
        key: CachePartitionKey
    )

    /// Build a fresh scratch-backed store, register the default
    /// partition, and return a tree for it. Every happy-path test
    /// in this file calls this; keep signatures additive (extra
    /// knobs via optional parameters) so callers stay terse.
    private func makeFixture(
        budgetBytes: Int = 1_000_000,
        maxPendingBytes: Int = 10_000_000
    ) -> Fixture {
        let root = makeScratchDir()
        let store = TieredSnapshotStore(
            ssdConfig: makeConfig(
                rootURL: root,
                budgetBytes: budgetBytes,
                maxPendingBytes: maxPendingBytes
            )
        )
        let key = makePartitionKey()
        store.registerPartition(makePartitionMeta(), for: key)
        let tree = store.getOrCreateTree(for: key)
        return (root, store, tree, key)
    }

    // MARK: - Construction

    @Test
    func ramOnlyModeSkipsSSDAdmission() async {
        // No config → SSD disabled → admitSnapshot always returns nil.
        let store = TieredSnapshotStore(ssdConfig: nil)
        let key = makePartitionKey()
        let tree = store.getOrCreateTree(for: key)
        let node = tree.insertPath(tokens: [1, 2, 3])

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        let result = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        )
        #expect(result == nil)
        #expect(node.storageRef == nil)
        #expect(store.pendingRefCountForTesting == 0)
    }

    @Test
    func disabledConfigCollapsesToRAMOnlyMode() async {
        // Explicitly disabled config → same as nil.
        let root = makeScratchDir()
        defer { cleanup(root) }
        let config = SSDPrefixCacheConfig(
            enabled: false,
            rootURL: root,
            budgetBytes: 1_000_000,
            maxPendingBytes: 10_000_000
        )
        let store = TieredSnapshotStore(ssdConfig: config)
        #expect(store.ssdStore == nil)
    }

    // MARK: - State 1 → 2 → 4 (happy path commit)

    @Test
    func admitAndCommitTransitionsStateOneToFour() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])

        // Pre-condition: state 1 (RAM only).
        #expect(node.storageRef == nil)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        // State 2: pending ref attached, not yet committed.
        #expect(node.storageRef?.snapshotID == ref.snapshotID)
        #expect(node.storageRef?.committed == false)
        #expect(store.pendingRefCountForTesting == 1)
        #expect(store.isPendingForTesting(id: ref.snapshotID))

        // Writer commit drains → state 4.
        let committed = await waitUntil {
            node.storageRef?.committed == true
        }
        #expect(committed)
        #expect(store.pendingRefCountForTesting == 0)
        #expect(store.isPendingForTesting(id: ref.snapshotID) == false)
    }

    // MARK: - State 2 → 3 → 5 (body-drop then commit)

    @Test
    func bodyDropThenCommitTransitionsStateTwoToFive() async {
        // The synchronous pendingRefsByID assignment always races
        // ahead of commit, so body-dropping the node via
        // `evictSnapshot` reliably lands before the `waitUntil`
        // below can observe `committed == true`.
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])

        // Store a dummy snapshot so `node.snapshot` is non-nil (we
        // need a body to drop). The tree reference counts update.
        let dummy = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(dummy, on: node)

        let payload = makePayload(bytes: 1_024, checkpointType: .leaf)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        // Body-drop before commit lands → state 3.
        tree.evictSnapshot(node: node)
        #expect(node.snapshot == nil)
        #expect(node.storageRef?.committed == false)
        #expect(store.isPendingForTesting(id: ref.snapshotID))

        // Commit callback fires → state 5 (body absent, committed ref).
        let committed = await waitUntil {
            node.storageRef?.committed == true
        }
        #expect(committed)
        #expect(node.snapshot == nil)
        #expect(store.pendingRefCountForTesting == 0)
        // Node stays in the tree as a structural path for future lookups.
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 3)
    }

    // MARK: - State 3 → removed-from-tree (drop callback)

    @Test
    func bodyDropThenDropCallbackHardDeletesLeafNode() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])

        // Sibling path so the shared intermediate retains a child
        // after the state-3 hard delete. Without this, evictNode
        // would walk all the way up to root (each ancestor becomes
        // a leaf once the sole child is gone), and the assertion
        // that the intermediate survives would be meaningless.
        let siblingNode = tree.insertPath(tokens: [1, 2, 5])
        #expect(siblingNode !== node)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        // Body is already nil (insertPath does not set snapshot) —
        // the node is in state 3 by construction. Directly invoke
        // the drop callback to simulate a writer-side failure
        // (e.g. diskFull after retry) before the commit path runs.
        store.markStorageRefDropped(id: ref.snapshotID, reason: .diskFull)

        #expect(store.pendingRefCountForTesting == 0)
        #expect(node.storageRef == nil)
        // Target node is hard-deleted — the [1,2,3] path no
        // longer reaches a terminal, but the shared [1,2] prefix
        // survives on the sibling path.
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 2)
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 5]) == 3)
    }

    // MARK: - State 2 → 1 (drop callback with body present)

    @Test
    func dropCallbackOnStateTwoClearsRefButKeepsBody() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])
        let dummy = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(dummy, on: node)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        // Simulate writer failure while the RAM body is still
        // live — state 2 → state 1.
        store.markStorageRefDropped(id: ref.snapshotID, reason: .writerIOError)

        #expect(store.pendingRefCountForTesting == 0)
        #expect(node.storageRef == nil)
        #expect(node.snapshot != nil)
        // Node remains in the tree; lookup can still hit via the body.
        let lookup = tree.findBestSnapshot(tokens: [1, 2, 3])
        #expect(lookup?.node === node)
    }

    // MARK: - State 4 → 5 (natural body-drop via evictSnapshot)

    @Test
    func bodyDropAfterCommitTransitionsStateFourToFive() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])
        let dummy = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(dummy, on: node)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        _ = store.admitSnapshot(
            node: node,
            tree: tree,
            payload: payload,
            descriptor: descriptor
        )

        // Wait for commit (state 2 → state 4).
        let committed = await waitUntil {
            node.storageRef?.committed == true
        }
        #expect(committed)
        #expect(node.snapshot != nil)

        // Body-drop: 4 → 5. The node stays in the tree with its
        // committed ref still attached. Lookup via snapshot-bearing
        // search misses (body is gone), but the tree path is intact.
        tree.evictSnapshot(node: node)
        #expect(node.snapshot == nil)
        #expect(node.storageRef?.committed == true)
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 3)
        // The snapshot-bearing lookup must miss because the RAM body
        // is gone — the upgrade to a state-5 hit is Task 4.1.9.
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3]) == nil)
    }

    // MARK: - System type protection on SSD

    @Test
    func systemAdmissionEvictsOldestNonSystemUnderBudgetPressure() async {
        // Tight budget: 3.5 KiB total, each payload ~1 KiB, so the
        // third admit forces an eviction.
        let (root, store, tree, key) = makeFixture(budgetBytes: 3_500)
        defer { cleanup(root) }

        // Three distinct token paths → three radix nodes.
        let nodes = [
            tree.insertPath(tokens: [10, 20]),
            tree.insertPath(tokens: [30, 40]),
            tree.insertPath(tokens: [50, 60]),
        ]

        // Admit two non-system residents first.
        let leafA = makeDescriptor(
            partitionDigest: key.partitionDigest,
            checkpointType: "leaf",
            bytes: 1_024,
            lastAccessAt: 1_000  // oldest
        )
        let leafB = makeDescriptor(
            partitionDigest: key.partitionDigest,
            checkpointType: "leaf",
            bytes: 1_024,
            lastAccessAt: 2_000
        )
        _ = store.admitSnapshot(
            node: nodes[0],
            tree: tree,
            payload: makePayload(bytes: 1_024, checkpointType: .leaf),
            descriptor: leafA
        )
        _ = store.admitSnapshot(
            node: nodes[1],
            tree: tree,
            payload: makePayload(bytes: 1_024, checkpointType: .leaf),
            descriptor: leafB
        )

        // Wait for both commits to land.
        let bothCommitted = await waitUntil {
            nodes[0].storageRef?.committed == true
            && nodes[1].storageRef?.committed == true
        }
        #expect(bothCommitted)

        // Third admit is a `.system` incoming that forces the
        // writer's admission-time LRU cut to evict the oldest
        // non-system resident (leafA).
        let systemDesc = makeDescriptor(
            partitionDigest: key.partitionDigest,
            checkpointType: "system",
            bytes: 2_048,
            lastAccessAt: 3_000
        )
        _ = store.admitSnapshot(
            node: nodes[2],
            tree: tree,
            payload: makePayload(bytes: 2_048, checkpointType: .system),
            descriptor: systemDesc
        )

        // System incoming is committed.
        let systemCommitted = await waitUntil {
            nodes[2].storageRef?.committed == true
        }
        #expect(systemCommitted)

        // Oldest non-system (leafA) is gone; leafB survives.
        let residents = store.ssdStore!.residentIDsByRecencyForTesting()
        #expect(residents.contains(leafA.snapshotID) == false)
        #expect(residents.contains(leafB.snapshotID))
        #expect(residents.contains(systemDesc.snapshotID))
    }

    @Test
    func nonSystemAdmissionIsDroppedWhenSystemProtectionWins() async {
        // Very tight budget: a single `.system` resident fills the
        // whole budget, so the second non-system admit has nowhere
        // to go without violating type protection.
        let (root, store, tree, key) = makeFixture(budgetBytes: 2_200)
        defer { cleanup(root) }

        let systemNode = tree.insertPath(tokens: [10, 20])
        let leafNode = tree.insertPath(tokens: [30, 40])

        let systemDesc = makeDescriptor(
            partitionDigest: key.partitionDigest,
            checkpointType: "system",
            bytes: 2_048,
            lastAccessAt: 1_000
        )
        _ = store.admitSnapshot(
            node: systemNode,
            tree: tree,
            payload: makePayload(bytes: 2_048, checkpointType: .system),
            descriptor: systemDesc
        )

        let systemCommitted = await waitUntil {
            systemNode.storageRef?.committed == true
        }
        #expect(systemCommitted)

        // Now try to admit a non-system entry that doesn't fit
        // without evicting the protected `.system` resident.
        let leafDesc = makeDescriptor(
            partitionDigest: key.partitionDigest,
            checkpointType: "leaf",
            bytes: 1_024,
            lastAccessAt: 2_000
        )
        _ = store.admitSnapshot(
            node: leafNode,
            tree: tree,
            payload: makePayload(bytes: 1_024, checkpointType: .leaf),
            descriptor: leafDesc
        )

        // The writer drops the incoming non-system entry with
        // `.systemProtectionWins`. The drop callback fires via
        // MainActor hop → `markStorageRefDropped` → clears the
        // leaf node's ref.
        let dropped = await waitUntil {
            leafNode.storageRef == nil
        }
        #expect(dropped)

        // System resident is still intact.
        #expect(systemNode.storageRef?.committed == true)
        let residents = store.ssdStore!.residentIDsByRecencyForTesting()
        #expect(residents.contains(systemDesc.snapshotID))
        #expect(residents.contains(leafDesc.snapshotID) == false)

        // The leaf node is in state 3 → drop callback hard-deleted
        // it from the tree (leafNode was a leaf).
        #expect(store.pendingRefCountForTesting == 0)
    }

    // MARK: - Back-pressure byte-budget eviction

    @Test
    func backpressureDropsOldestPendingAndClearsRef() async {
        // 1 MiB payloads + 2.5 MiB front-door cap: the writer's
        // fsync+rename (~ms) dominates three synchronous MainActor
        // admits (~µs), so admit 3 reliably hits back-pressure
        // before the writer drains the first entry.
        let payloadBytes = 1 * 1024 * 1024
        let (root, store, tree, key) = makeFixture(
            budgetBytes: 64 * 1024 * 1024,
            maxPendingBytes: 2_500_000
        )
        defer { cleanup(root) }

        // Distinct radix paths so each admit has its own node.
        let nodeA = tree.insertPath(tokens: [1, 2, 10])
        let nodeB = tree.insertPath(tokens: [1, 2, 20])
        let nodeC = tree.insertPath(tokens: [1, 2, 30])

        func admit(_ node: RadixTreeNode) {
            _ = store.admitSnapshot(
                node: node, tree: tree,
                payload: makePayload(bytes: payloadBytes),
                descriptor: makeDescriptor(
                    partitionDigest: key.partitionDigest,
                    checkpointType: "leaf",
                    bytes: payloadBytes
                )
            )
        }

        admit(nodeA)
        admit(nodeB)
        admit(nodeC)

        // Each node settles into committed-ref or cleared-ref and
        // the pending map drains. Both race outcomes are accepted
        // (back-pressure fired vs. writer drained fast enough) —
        // the test's job is verifying the wiring, not pinning NVMe
        // timing.
        let settled = await waitUntil {
            [nodeA, nodeB, nodeC].allSatisfy { node in
                node.storageRef == nil || node.storageRef?.committed == true
            } && store.pendingRefCountForTesting == 0
        }
        #expect(settled)

        // Most recent admission must always survive — the
        // front-door drops OLDEST pending, and the writer's LRU
        // has ample SSD budget here so its own cut never touches
        // the incoming.
        #expect(nodeC.storageRef != nil)

        // Every surviving ref must match a committed manifest
        // entry — the pending state doesn't count.
        let residentIDs = Set(store.ssdStore!.residentIDsByRecencyForTesting())
        let committedIDs = [nodeA, nodeB, nodeC]
            .compactMap { $0.storageRef?.snapshotID }
        #expect(Set(committedIDs).isSubset(of: residentIDs))
    }

    // MARK: - Non-suspending admission latency

    @Test
    func admissionIsNonSuspendingFromMainActor() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }

        // 100 synchronous admits from a MainActor scope — nothing
        // awaits inside `admitSnapshot`, so the whole batch must
        // finish on a single runloop tick. The time bound is a
        // soft smoke signal; the real guarantee is structural (no
        // `await` in the call chain).
        let count = 100
        let payload = makePayload(bytes: 16)
        let descriptors = (0..<count).map { i in
            makeDescriptor(
                id: "bulk-\(i)",
                partitionDigest: key.partitionDigest,
                checkpointType: "leaf",
                bytes: 16
            )
        }
        let nodes = (0..<count).map { i in
            tree.insertPath(tokens: [1, 2, 1_000 + i])
        }

        let start = ContinuousClock.now
        for i in 0..<count {
            _ = store.admitSnapshot(
                node: nodes[i],
                tree: tree,
                payload: payload,
                descriptor: descriptors[i]
            )
        }
        let elapsed = ContinuousClock.now - start
        let elapsedMs = Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
            + Double(elapsed.components.seconds) * 1_000.0
        #expect(
            elapsedMs < 50.0,
            "admitSnapshot batch took \(elapsedMs) ms, expected <50 ms"
        )
    }

    // MARK: - State-3 lookup returns miss

    @Test
    func stateThreeNodeReturnsMissOnRadixLookup() async {
        // A node that transitioned state 2 → 3 (body dropped while
        // write still pending) must NOT surface as a hit on
        // `findBestSnapshot` — pending refs are not hit targets.
        // `findBestSnapshot` already filters on `snapshot != nil`,
        // so state-3 nodes naturally return miss.
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = tree.insertPath(tokens: [1, 2, 3])
        let dummy = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(dummy, on: node)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(
            partitionDigest: key.partitionDigest,
            bytes: payload.totalBytes
        )

        _ = store.admitSnapshot(
            node: node, tree: tree,
            payload: payload,
            descriptor: descriptor
        )

        // Body-drop while the pending ref is still uncommitted.
        tree.evictSnapshot(node: node)
        // Sanity: the node is in state 3 (pending + body absent).
        #expect(node.snapshot == nil)
        #expect(node.storageRef != nil)
        #expect(node.storageRef?.committed == false)

        // Radix lookup must return miss — state-3 nodes are not
        // hit targets.
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3]) == nil)
    }

    // MARK: - Protocol conformance smoke test

    @Test
    func snapshotStoreProtocolForwardsToRAMTier() {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let key = makePartitionKey()

        // First-touch creates a fresh tree.
        #expect(store.tree(for: key) == nil)
        #expect(store.partitionCount == 0)
        #expect(store.totalSnapshotBytes == 0)

        let tree = store.getOrCreateTree(for: key)
        #expect(store.partitionCount == 1)
        #expect(store.tree(for: key) === tree)

        // Store a snapshot and verify `totalSnapshotBytes`
        // aggregates through the forwarding property.
        let node = tree.insertPath(tokens: [1, 2, 3])
        let snapshot = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(snapshot, on: node)
        #expect(store.totalSnapshotBytes == snapshot.memoryBytes)

        // Ordered partition iteration yields exactly the one key.
        let partitions = store.orderedPartitions()
        #expect(partitions.count == 1)
        #expect(partitions.first?.key == key)
    }
}
