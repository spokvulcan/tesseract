//
//  TieredSnapshotStoreTests.swift
//  tesseractTests
//
//  Coverage for the tiered store as an SSD **router**: admission routes
//  through `tree.admit`, the writer's commit / drop callbacks route
//  through `tree.commitRef` / `tree.dropRef`, and node removal is the
//  tree's self-heal. The store never mutates node state itself.
//
//  Each test constructs a `TieredSnapshotStore` over a temp directory,
//  drives a lifecycle transition via the public API, and asserts the
//  post-state via `node.state` queries, the `pendingRefCountForTesting`
//  hook, and (where applicable) the underlying `SSDSnapshotStore`.
//
//  Note: `admit` requires a resident RAM body (states 1/2/4) — in
//  production every admission follows a `storeSnapshot`. Tests therefore
//  seed a body via `insertWithBody` before admitting, and reach the
//  body-less states (3/5) through `dropBody`.
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

    /// Insert a path and attach a RAM body so the node is in state 1
    /// (`ramOnly`) — the precondition `admit` requires.
    @discardableResult
    private func insertWithBody(
        _ tree: TokenRadixTree,
        tokens: [Int],
        type: HybridCacheSnapshot.CheckpointType = .leaf
    ) -> RadixTreeNode {
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(
            PrefixCacheTestFixtures.makeUniformSnapshot(offset: node.tokenOffset, type: type),
            on: node
        )
        return node
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

    private typealias Fixture = (
        root: URL,
        store: TieredSnapshotStore,
        tree: TokenRadixTree,
        key: CachePartitionKey
    )

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
        let node = insertWithBody(tree, tokens: [1, 2, 3])

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
        #expect(node.state.ref == nil)
        #expect(store.pendingRefCountForTesting == 0)
    }

    @Test
    func disabledConfigCollapsesToRAMOnlyMode() async {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let config = SSDPrefixCacheConfig(
            enabled: false,
            rootURL: root,
            budgetBytes: 1_000_000,
            maxPendingBytes: 10_000_000
        )
        let store = TieredSnapshotStore(ssdConfig: config)
        #expect(store.ssdStoreForTesting == nil)
    }

    // MARK: - State 1 → 2 → 4 (happy path commit)

    @Test
    func admitAndCommitTransitionsStateOneToFour() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        // Pre-condition: state 1 (RAM only, no ref).
        #expect(node.state.ref == nil)

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
        #expect(node.state.refID == ref.snapshotID)
        #expect(!node.state.committed)
        #expect(store.pendingRefCountForTesting == 1)
        #expect(store.isPendingForTesting(id: ref.snapshotID))

        // Writer commit drains → state 4.
        let committed = await waitUntil { node.state.committed }
        #expect(committed)
        #expect(store.pendingRefCountForTesting == 0)
        #expect(store.isPendingForTesting(id: ref.snapshotID) == false)
    }

    // MARK: - State 2 → 3 → 5 (body-drop then commit)

    @Test
    func bodyDropThenCommitTransitionsStateTwoToFive() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

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

        // Body-drop before commit lands → state 3 (pendingDropped).
        tree.dropBody(node: node)
        #expect(node.state.body == nil)
        #expect(!node.state.committed)
        #expect(store.isPendingForTesting(id: ref.snapshotID))

        // Commit callback fires → state 5 (body absent, committed ref).
        let committed = await waitUntil { node.state.committed }
        #expect(committed)
        #expect(node.state.body == nil)
        #expect(store.pendingRefCountForTesting == 0)
        // Node stays in the tree as a structural path for future lookups.
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 3)
    }

    // MARK: - State 3 → removed-from-tree (drop callback)

    @Test
    func bodyDropThenDropCallbackHardDeletesLeafNode() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        // Sibling path so the shared intermediate retains a child after
        // the state-3 hard delete.
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

        // Drop the body → state 3, then simulate a writer-side drop
        // (e.g. diskFull after retry) before commit runs.
        tree.dropBody(node: node)
        store.markSnapshotRefDropped(id: ref.snapshotID, reason: .diskFull)

        #expect(store.pendingRefCountForTesting == 0)
        #expect(node.state.ref == nil)
        // Target node is hard-deleted via self-heal — the [1,2,3] path no
        // longer reaches a terminal, but the shared [1,2] prefix survives.
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 2)
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 5]) == 3)
    }

    // MARK: - State 2 → 1 (drop callback with body present)

    @Test
    func dropCallbackOnStateTwoClearsRefButKeepsBody() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

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

        // Simulate writer failure while the RAM body is still live —
        // state 2 → state 1.
        store.markSnapshotRefDropped(id: ref.snapshotID, reason: .writerIOError)

        #expect(store.pendingRefCountForTesting == 0)
        #expect(node.state.ref == nil)
        #expect(node.state.body != nil)
        // Node remains in the tree; lookup can still hit via the body.
        let lookup = tree.findBestSnapshot(tokens: [1, 2, 3])
        #expect(lookup?.node === node)
    }

    // MARK: - State 4 → 5 (natural body-drop via dropBody)

    @Test
    func bodyDropAfterCommitTransitionsStateFourToFive() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

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
        let committed = await waitUntil { node.state.committed }
        #expect(committed)
        #expect(node.state.body != nil)

        // Body-drop: 4 → 5. The node stays with its committed ref
        // attached. A RAM-only lookup misses (body gone); the path stays.
        tree.dropBody(node: node)
        #expect(node.state.body == nil)
        #expect(node.state.committed)
        #expect(tree.findSharedPrefixLength(tokens: [1, 2, 3]) == 3)
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3]) == nil)
    }

    // MARK: - System type protection on SSD

    @Test
    func systemAdmissionEvictsOldestNonSystemUnderBudgetPressure() async {
        let (root, store, tree, key) = makeFixture(budgetBytes: 3_500)
        defer { cleanup(root) }

        let nodes = [
            insertWithBody(tree, tokens: [10, 20]),
            insertWithBody(tree, tokens: [30, 40]),
            insertWithBody(tree, tokens: [50, 60], type: .system),
        ]

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

        let bothCommitted = await waitUntil {
            nodes[0].state.committed && nodes[1].state.committed
        }
        #expect(bothCommitted)

        // Third admit is `.system` and forces the writer's admission-time
        // LRU cut to evict the oldest non-system resident (leafA).
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

        let systemCommitted = await waitUntil { nodes[2].state.committed }
        #expect(systemCommitted)

        let residents = store.ssdStoreForTesting!.residentIDsByRecencyForTesting()
        #expect(residents.contains(leafA.snapshotID) == false)
        #expect(residents.contains(leafB.snapshotID))
        #expect(residents.contains(systemDesc.snapshotID))
    }

    @Test
    func nonSystemAdmissionIsDroppedWhenSystemProtectionWins() async {
        let (root, store, tree, key) = makeFixture(budgetBytes: 2_200)
        defer { cleanup(root) }

        let systemNode = insertWithBody(tree, tokens: [10, 20], type: .system)
        let leafNode = insertWithBody(tree, tokens: [30, 40])

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

        let systemCommitted = await waitUntil { systemNode.state.committed }
        #expect(systemCommitted)

        // Admit a non-system entry that does not fit without evicting the
        // protected `.system` resident.
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

        // The writer drops the incoming with `.systemProtectionWins`. The
        // drop callback routes through `tree.dropRef`: the leaf had a RAM
        // body, so it settles to state 1 (ref cleared, body kept).
        let dropped = await waitUntil { leafNode.state.ref == nil }
        #expect(dropped)
        #expect(leafNode.state.body != nil)

        // System resident is still intact.
        #expect(systemNode.state.committed)
        let residents = store.ssdStoreForTesting!.residentIDsByRecencyForTesting()
        #expect(residents.contains(systemDesc.snapshotID))
        #expect(residents.contains(leafDesc.snapshotID) == false)
        #expect(store.pendingRefCountForTesting == 0)
    }

    // MARK: - Back-pressure byte-budget eviction

    @Test
    func backpressureDropsOldestPendingAndClearsRef() async {
        let payloadBytes = 1 * 1024 * 1024
        let (root, store, tree, key) = makeFixture(
            budgetBytes: 64 * 1024 * 1024,
            maxPendingBytes: 2_500_000
        )
        defer { cleanup(root) }

        let nodeA = insertWithBody(tree, tokens: [1, 2, 10])
        let nodeB = insertWithBody(tree, tokens: [1, 2, 20])
        let nodeC = insertWithBody(tree, tokens: [1, 2, 30])

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

        // Each node settles into committed-ref or cleared-ref and the
        // pending map drains. Both race outcomes are accepted.
        let settled = await waitUntil {
            [nodeA, nodeB, nodeC].allSatisfy { node in
                node.state.ref == nil || node.state.committed
            } && store.pendingRefCountForTesting == 0
        }
        #expect(settled)

        // Most recent admission must survive — the front-door drops
        // OLDEST pending and SSD budget is ample.
        #expect(nodeC.state.ref != nil)

        // Every surviving committed ref must match a manifest entry.
        let residentIDs = Set(store.ssdStoreForTesting!.residentIDsByRecencyForTesting())
        let committedIDs = [nodeA, nodeB, nodeC]
            .compactMap { $0.state.committed ? $0.state.refID : nil }
        #expect(Set(committedIDs).isSubset(of: residentIDs))
    }

    // MARK: - Non-suspending admission latency

    @Test
    func admissionIsNonSuspendingFromMainActor() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }

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
        // Bodies are seeded before the timed loop, so they do not count.
        let nodes = (0..<count).map { i in
            insertWithBody(tree, tokens: [1, 2, 1_000 + i])
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
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

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

        // Body-drop while the pending ref is still uncommitted → state 3.
        tree.dropBody(node: node)
        #expect(node.state.body == nil)
        #expect(node.state.ref != nil)
        #expect(!node.state.committed)

        // Radix lookup must miss — state-3 nodes are not hit targets.
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3]) == nil)
    }

    // MARK: - Forgiving callbacks (stale / duplicate / committed-resident)

    /// A commit callback that arrives after the ref was dropped is a
    /// logged no-op — it must not resurrect the cleared ref.
    @Test
    func commitAfterDropIsLoggedNoOp() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node, tree: tree,
            payload: makePayload(bytes: 1_024),
            descriptor: makeDescriptor(partitionDigest: key.partitionDigest, bytes: 1_024)
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        // Drop the pending ref (state 2 → 1), then a late commit arrives.
        store.markSnapshotRefDropped(id: ref.snapshotID, reason: .writerIOError)
        store.markSnapshotRefCommitted(id: ref.snapshotID)

        #expect(node.state.ref == nil)       // not resurrected
        #expect(node.state.body != nil)
        #expect(store.pendingRefCountForTesting == 0)
    }

    /// A duplicate commit callback for an already-committed id is a
    /// logged no-op.
    @Test
    func duplicateCommitIsLoggedNoOp() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node, tree: tree,
            payload: makePayload(bytes: 1_024),
            descriptor: makeDescriptor(partitionDigest: key.partitionDigest, bytes: 1_024)
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        let committed = await waitUntil { node.state.committed }
        #expect(committed)

        // Fire commit again — already committed, not in pending map.
        store.markSnapshotRefCommitted(id: ref.snapshotID)
        #expect(node.state.committed)
        #expect(store.pendingRefCountForTesting == 0)
    }

    /// A drop callback for a committed resident whose id is no longer in
    /// the pending map (`.evictedByLRU` / `.hydrationFailure`) is a logged
    /// no-op: it leaves the tree untouched. The stale committed ref is
    /// cleared lazily through `tree.clearCommittedSnapshotRefAfterHydrationFailure`.
    @Test
    func committedResidentDropIsLoggedNoOpThenClearedLazily() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        guard case .accepted(let ref) = store.admitSnapshot(
            node: node, tree: tree,
            payload: makePayload(bytes: 1_024),
            descriptor: makeDescriptor(partitionDigest: key.partitionDigest, bytes: 1_024)
        ) else {
            Issue.record("admitSnapshot did not accept")
            return
        }

        let committed = await waitUntil { node.state.committed }
        #expect(committed)
        tree.dropBody(node: node)            // state 4 → 5 (ssdOnly)
        #expect(store.pendingRefCountForTesting == 0)

        // Committed-resident drop: id is not in the pending map → no-op.
        store.markSnapshotRefDropped(id: ref.snapshotID, reason: .hydrationFailure)
        #expect(node.state.ref != nil)       // tree untouched
        #expect(node.state.committed)

        // Lazy clear on the next lookup (node supplied by the failing
        // hydration) removes the stale ref and self-heals the leaf.
        tree.clearCommittedSnapshotRefAfterHydrationFailure(node: node)
        #expect(node.state.ref == nil)
        #expect(node.parent == nil)
    }

    // MARK: - Re-admission supersession (SSD-orphan bug fix)

    /// Admitting a second snapshot onto a node that already holds a
    /// committed ref must delete the superseded SSD backing before
    /// installing the new pending ref — otherwise the old write orphans a
    /// file + manifest entry.
    @Test
    func reAdmissionDeletesSupersededSSDBacking() async {
        let (root, store, tree, key) = makeFixture()
        defer { cleanup(root) }
        let node = insertWithBody(tree, tokens: [1, 2, 3])

        guard case .accepted(let firstRef) = store.admitSnapshot(
            node: node, tree: tree,
            payload: makePayload(bytes: 1_024),
            descriptor: makeDescriptor(partitionDigest: key.partitionDigest, bytes: 1_024)
        ) else {
            Issue.record("first admitSnapshot did not accept")
            return
        }
        let firstCommitted = await waitUntil { node.state.committed }
        #expect(firstCommitted)
        #expect(store.ssdStoreForTesting!.residentIDsByRecencyForTesting().contains(firstRef.snapshotID))

        // Re-admit over the committed node (state 4). The node still has a
        // RAM body, so admit applies and supersedes firstRef.
        guard case .accepted(let secondRef) = store.admitSnapshot(
            node: node, tree: tree,
            payload: makePayload(bytes: 1_024),
            descriptor: makeDescriptor(partitionDigest: key.partitionDigest, bytes: 1_024)
        ) else {
            Issue.record("second admitSnapshot did not accept")
            return
        }

        #expect(node.state.refID == secondRef.snapshotID)
        #expect(store.isPendingForTesting(id: secondRef.snapshotID))
        #expect(store.isPendingForTesting(id: firstRef.snapshotID) == false)
        // The superseded backing is gone — no orphan.
        #expect(store.ssdStoreForTesting!.residentIDsByRecencyForTesting().contains(firstRef.snapshotID) == false)

        let secondCommitted = await waitUntil { node.state.committed }
        #expect(secondCommitted)
        let residents = Set(store.ssdStoreForTesting!.residentIDsByRecencyForTesting())
        #expect(residents.contains(secondRef.snapshotID))
        #expect(residents.contains(firstRef.snapshotID) == false)
    }

    // MARK: - Protocol conformance smoke test

    @Test
    func snapshotStoreProtocolForwardsToRAMTier() {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let key = makePartitionKey()

        #expect(store.tree(for: key) == nil)
        #expect(store.partitionCount == 0)
        #expect(store.totalSnapshotBytes == 0)

        let tree = store.getOrCreateTree(for: key)
        #expect(store.partitionCount == 1)
        #expect(store.tree(for: key) === tree)

        let node = tree.insertPath(tokens: [1, 2, 3])
        let snapshot = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 3,
            type: .leaf
        )
        tree.storeSnapshot(snapshot, on: node)
        #expect(store.totalSnapshotBytes == snapshot.memoryBytes)

        let partitions = store.orderedPartitions()
        #expect(partitions.count == 1)
        #expect(partitions.first?.key == key)
    }
}
