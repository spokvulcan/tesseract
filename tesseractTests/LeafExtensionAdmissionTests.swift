//
//  LeafExtensionAdmissionTests.swift
//  tesseractTests
//
//  Tests for **Leaf Extension Admission** (docs/adr/0010): the SSD leaf
//  store admitting only the suffix past an SSD-backed ancestor leaf and
//  taking ownership of that ancestor's **Segment Chain**, instead of
//  re-writing the full KV state every turn (issue #78).
//
//  Four layers, mirroring the production seams:
//  1. Extraction — `ServerCompletion.extractSnapshotPayload(_:extending:)`
//     suffix slicing + the worth-it gate (no store).
//  2. Ledger — the transfer protocol: shield, fold, consuming commit,
//     chain removal, rebuild head detection (no writer scaffold).
//  3. Store — front-door validation, writer-side fold-at-commit, chain
//     hydration composition, shield release on terminal paths.
//  4. Manager — the three supersession modes (transferred / deleted /
//     preserved) and `extensionBase` resolution.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Shared fixtures (file-scope, used by suites 2–4)

private let testFingerprint = String(repeating: "a", count: 64)
private let testDigest = "abcd1234"

private func makeScratchDir() -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("leaf-ext-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func cleanup(_ url: URL) {
    try? FileManager.default.removeItem(at: url)
}

private func makePartitionMeta() -> PartitionMeta {
    PartitionMeta(
        modelID: "mlx-community/Qwen3-4B-4bit",
        modelFingerprint: testFingerprint,
        kvBits: 8,
        kvGroupSize: 64,
        createdAt: 100_000,
        schemaVersion: SnapshotManifestSchema.currentVersion
    )
}

private func makeDescriptor(
    id: String = UUID().uuidString,
    bytes: Int = 1_000,
    tokenOffset: Int = 5,
    segmentBaseOffset: Int = 0,
    inheritedSegments: [SnapshotSegment] = [],
    lastAccessAt: Double = 0
) -> PersistedSnapshotDescriptor {
    PersistedSnapshotDescriptor(
        snapshotID: id,
        partitionDigest: testDigest,
        pathFromRoot: Array(1...tokenOffset),
        tokenOffset: tokenOffset,
        checkpointType: "leaf",
        bytes: bytes,
        segmentBaseOffset: segmentBaseOffset,
        inheritedSegments: inheritedSegments,
        createdAt: 100_000,
        lastAccessAt: lastAccessAt,
        fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
            snapshotID: id,
            partitionDigest: testDigest
        ),
        schemaVersion: SnapshotManifestSchema.currentVersion
    )
}

/// Single-layer leaf payload whose one KV array carries `bytes` raw
/// bytes; `extending` marks it as a suffix segment past the base.
private func makePayload(
    bytes: Int,
    tokenOffset: Int,
    extending: SnapshotExtension? = nil
) -> SnapshotPayload {
    SnapshotPayload(
        tokenOffset: tokenOffset,
        checkpointType: .leaf,
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
                offset: tokenOffset,
                suffixBaseOffset: extending?.baseOffset
            )
        ],
        extending: extending
    )
}

// MARK: - 1. Extraction: suffix slicing + worth-it gate

struct LeafExtensionExtractionTests {

    /// A `KVCacheSimple` snapshot whose token axis matches its capture
    /// offset, so the layer qualifies for suffix slicing. Values are
    /// position-dependent so slice equality checks are byte-exact.
    private func makeSliceableSnapshot(
        tokenOffset: Int,
        heads: Int = 2,
        headDim: Int = 8
    ) -> HybridCacheSnapshot {
        let count = 1 * heads * tokenOffset * headDim
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray((0..<count).map { Float($0) }).reshaped([1, heads, tokenOffset, headDim]),
            MLXArray((0..<count).map { Float($0) + 0.5 }).reshaped([1, heads, tokenOffset, headDim]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: tokenOffset, type: .leaf)!
    }

    private func makeMambaOnlySnapshot(tokenOffset: Int) -> HybridCacheSnapshot {
        let mamba = MambaCache()
        mamba.state = [
            MLXArray.zeros([1, 3, 128]),
            MLXArray.zeros([1, 8, 16, 32]),
        ]
        return HybridCacheSnapshot.capture(cache: [mamba], offset: tokenOffset, type: .leaf)!
    }

    private func sliceData(
        _ array: MLXArray, from base: Int, to offset: Int
    ) -> Data {
        array[.ellipsis, base..<offset, 0...].asData(access: .copy).data
    }

    @Test func slicesSliceableLayersPastBase() throws {
        let snapshot = makeSliceableSnapshot(tokenOffset: 8)
        let extending = SnapshotExtension(baseSnapshotID: "base-1", baseOffset: 5)

        let payload = ServerCompletion.extractSnapshotPayload(snapshot, extending: extending)

        #expect(payload.extending == extending)
        #expect(payload.tokenOffset == 8)
        let layer = try #require(payload.layers.first)
        #expect(layer.suffixBaseOffset == 5)
        // Layer offset stays the absolute capture offset — only the
        // carried token range shrinks.
        #expect(layer.offset == 8)
        for (arrayPayload, original) in zip(layer.state, snapshot.layers[0].state) {
            #expect(arrayPayload.shape == [1, 2, 3, 8])
            #expect(arrayPayload.data == sliceData(original, from: 5, to: 8))
        }
    }

    @Test func nonSliceableLayersRideWhole() throws {
        // The KV layer dominates the byte total so the worth-it gate
        // passes; the small Mamba layer must still ride whole.
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.ones([1, 2, 6, 64]),
            MLXArray.ones([1, 2, 6, 64]),
        ]
        let mamba = MambaCache()
        mamba.state = [
            MLXArray.ones([1, 3, 4]),
            MLXArray.ones([1, 2, 2, 2]),
        ]
        let snapshot = HybridCacheSnapshot.capture(
            cache: [kv, mamba], offset: 6, type: .leaf
        )!
        let extending = SnapshotExtension(baseSnapshotID: "base-2", baseOffset: 2)

        let payload = ServerCompletion.extractSnapshotPayload(snapshot, extending: extending)

        #expect(payload.extending == extending)
        let kvLayer = try #require(payload.layers.first)
        #expect(kvLayer.suffixBaseOffset == 2)
        #expect(kvLayer.state.allSatisfy { $0.shape == [1, 2, 4, 64] })
        let mambaLayer = try #require(payload.layers.last)
        #expect(mambaLayer.suffixBaseOffset == nil)
        #expect(mambaLayer.state[0].shape == [1, 3, 4])
        #expect(mambaLayer.state[1].shape == [1, 2, 2, 2])
    }

    @Test func worthItGateFallsBackToFullWhenSavingsTooSmall() throws {
        // Suffix would be 19/20 = 95% of the full payload — past the
        // 90% gate, so the extension is abandoned and the payload
        // admits full.
        let snapshot = makeSliceableSnapshot(tokenOffset: 20)
        let extending = SnapshotExtension(baseSnapshotID: "base-3", baseOffset: 1)

        let payload = ServerCompletion.extractSnapshotPayload(snapshot, extending: extending)

        #expect(payload.extending == nil)
        let layer = try #require(payload.layers.first)
        #expect(layer.suffixBaseOffset == nil)
        #expect(layer.state.allSatisfy { $0.shape == [1, 2, 20, 8] })
    }

    @Test func worthItGateFallsBackToFullWhenNothingIsSliceable() throws {
        // Every layer rides whole, so an "extension" would re-write
        // 100% of the bytes — the gate must reject it.
        let snapshot = makeMambaOnlySnapshot(tokenOffset: 64)
        let extending = SnapshotExtension(baseSnapshotID: "base-4", baseOffset: 32)

        let payload = ServerCompletion.extractSnapshotPayload(snapshot, extending: extending)

        #expect(payload.extending == nil)
        #expect(payload.layers.allSatisfy { $0.suffixBaseOffset == nil })
    }

    @Test func invalidBaseOffsetsAreIgnored() {
        let snapshot = makeSliceableSnapshot(tokenOffset: 8)
        for badOffset in [0, 8, 9, -3] {
            let payload = ServerCompletion.extractSnapshotPayload(
                snapshot,
                extending: SnapshotExtension(baseSnapshotID: "base-5", baseOffset: badOffset)
            )
            #expect(payload.extending == nil, "baseOffset \(badOffset) must admit full")
        }
    }
}

// MARK: - 2. Ledger: shield, fold, consuming commit, rebuild

struct LeafExtensionLedgerTests {

    private func makeLedgerWithPartition(
        budgetBytes: Int = 1_000_000,
        root: URL
    ) -> SnapshotLedger {
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: budgetBytes,
            manifestDebounce: .milliseconds(20)
        )
        ledger.registerPartition(makePartitionMeta(), digest: testDigest)
        return ledger
    }

    @Test func beginTransferValidatesBaseReachability() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        // Unknown base, not queued: rejected, no shield.
        #expect(!ledger.beginExtensionTransfer(baseID: "ghost", baseIsQueuedOrInFlight: false))
        #expect(ledger.transferringBaseIDsForTesting().isEmpty)

        // Unknown base but still in the writer's queue: allowed (FIFO
        // settles it before the extension).
        #expect(ledger.beginExtensionTransfer(baseID: "queued", baseIsQueuedOrInFlight: true))
        #expect(ledger.transferringBaseIDsForTesting() == ["queued"])
        ledger.releaseExtensionTransfer(baseID: "queued")
        #expect(ledger.transferringBaseIDsForTesting().isEmpty)

        // Resident base: allowed.
        let base = makeDescriptor(id: "base-resident")
        ledger.seedDescriptorForTesting(base)
        #expect(ledger.beginExtensionTransfer(baseID: "base-resident", baseIsQueuedOrInFlight: false))
        #expect(ledger.transferringBaseIDsForTesting() == ["base-resident"])
    }

    @Test func transferShieldExcludesBaseFromLRUCutUntilReleased() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        // Budget fits exactly two 400-byte residents.
        let ledger = makeLedgerWithPartition(budgetBytes: 1_000, root: root)

        // Base is the LRU-oldest entry — the natural victim.
        let base = makeDescriptor(id: "base", bytes: 400, lastAccessAt: 1)
        let other = makeDescriptor(id: "other", bytes: 400, lastAccessAt: 2)
        ledger.seedDescriptorForTesting(base)
        ledger.seedDescriptorForTesting(other)
        #expect(ledger.beginExtensionTransfer(baseID: "base", baseIsQueuedOrInFlight: false))

        // Admitting 400 more requires one eviction; the shield must
        // divert the cut past the older base onto `other`.
        let (decision, evicted) = ledger.admit(makeDescriptor(id: "incoming", bytes: 400))
        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == ["other"])
        #expect(ledger.residentDescriptorForTesting(id: "base") != nil)

        // Released, the base is an ordinary victim again.
        ledger.releaseExtensionTransfer(baseID: "base")
        let (_, evictedAfter) = ledger.admit(makeDescriptor(id: "incoming-2", bytes: 1_000))
        #expect(evictedAfter.map(\.snapshotID) == ["base"])
    }

    @Test func prepareFoldedDescriptorFoldsBaseChain() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        // The base is itself a chain head: one inherited segment below
        // its own file. The fold must append the base's own file after
        // its inherited chain, preserving shallow→deep order.
        let grandSegment = SnapshotSegment(
            baseOffset: 0,
            tokenOffset: 3,
            fileRelativePath: "partitions/\(testDigest)/snapshots/0/grand.safetensors",
            bytes: 300
        )
        let base = makeDescriptor(
            id: "base", bytes: 200, tokenOffset: 5,
            segmentBaseOffset: 3, inheritedSegments: [grandSegment]
        )
        ledger.seedDescriptorForTesting(base)

        let extensionDescriptor = makeDescriptor(
            id: "head", bytes: 100, tokenOffset: 9, segmentBaseOffset: 5
        )
        let folded = try #require(
            ledger.prepareFoldedDescriptor(extensionDescriptor, baseID: "base")
        )
        #expect(folded.snapshotID == "head")
        #expect(folded.inheritedSegments == [grandSegment, base.ownSegment])
        #expect(folded.totalBytes == 300 + 200 + 100)
        #expect(folded.chainFileRelativePaths == [
            grandSegment.fileRelativePath,
            base.fileRelativePath,
            extensionDescriptor.fileRelativePath,
        ])

        // Slice boundary disagrees with the base's offset → no fold.
        let misaligned = makeDescriptor(
            id: "head-2", bytes: 100, tokenOffset: 9, segmentBaseOffset: 4
        )
        #expect(ledger.prepareFoldedDescriptor(misaligned, baseID: "base") == nil)

        // Unknown base → no fold.
        #expect(ledger.prepareFoldedDescriptor(extensionDescriptor, baseID: "ghost") == nil)
    }

    @Test func commitConsumingBaseFoldsAtomically() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let base = makeDescriptor(id: "base", bytes: 800, tokenOffset: 5)
        ledger.seedDescriptorForTesting(base)
        #expect(ledger.beginExtensionTransfer(baseID: "base", baseIsQueuedOrInFlight: false))
        #expect(ledger.currentSSDBytesForTesting() == 800)

        let extensionDescriptor = makeDescriptor(
            id: "head", bytes: 200, tokenOffset: 9, segmentBaseOffset: 5
        )
        let folded = try #require(
            ledger.prepareFoldedDescriptor(extensionDescriptor, baseID: "base")
        )

        #expect(ledger.commit(folded, consumingBase: "base"))

        // Base entry gone, head owns the chain total, shield released —
        // and no instant in between double-counted (end-state check).
        #expect(ledger.residentDescriptorForTesting(id: "base") == nil)
        #expect(ledger.currentSSDBytesForTesting() == 1_000)
        #expect(ledger.transferringBaseIDsForTesting().isEmpty)

        let chain = try #require(ledger.chainForHydration(id: "head"))
        #expect(chain.tokenOffset == 9)
        #expect(chain.fileURLs.map(\.lastPathComponent) == [
            "base.safetensors", "head.safetensors",
        ])

        // Removing the folded head frees the whole chain.
        let removed = try #require(ledger.remove(id: "head"))
        #expect(removed.fileURLs.count == 2)
        #expect(ledger.currentSSDBytesForTesting() == 0)
    }

    @Test func commitConsumingMissingBaseVetoes() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let head = makeDescriptor(id: "head", bytes: 200, tokenOffset: 9, segmentBaseOffset: 5)
        #expect(!ledger.commit(head, consumingBase: "ghost"))
        #expect(ledger.residentDescriptorForTesting(id: "head") == nil)
        #expect(ledger.currentSSDBytesForTesting() == 0)
    }

    @Test func rebuildKeepsChainHeadAndDropsBaseEntry() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // On-disk state after a committed extension: the base's file
        // (whose own embedded descriptor still claims headship — it was
        // written before the extension existed) plus the head's file
        // whose descriptor lists the base file as inherited. The
        // rebuild must keep only the head, with the base's file
        // surviving as its inherited segment.
        let meta = makePartitionMeta()
        let metaDir = root
            .appendingPathComponent("partitions")
            .appendingPathComponent(testDigest)
        try FileManager.default.createDirectory(at: metaDir, withIntermediateDirectories: true)
        try JSONEncoder().encode(meta)
            .write(to: metaDir.appendingPathComponent("_meta.json"))

        let base = makeDescriptor(id: "base-1", bytes: 64, tokenOffset: 4)
        let head = makeDescriptor(
            id: "head-1", bytes: 64, tokenOffset: 7,
            segmentBaseOffset: 4, inheritedSegments: [base.ownSegment]
        )
        for descriptor in [base, head] {
            try writeContainerFile(descriptor: descriptor, rootURL: root)
        }
        try Data("{not-valid-json".utf8)
            .write(to: root.appendingPathComponent("manifest.json"))

        let ledger = SnapshotLedger(
            rootURL: root, budgetBytes: 1_000_000, manifestDebounce: .milliseconds(20)
        )
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID) == ["head-1"]
        )
        let restored = try #require(ledger.residentDescriptorForTesting(id: "head-1"))
        #expect(restored.inheritedSegments == [base.ownSegment])
        #expect(ledger.currentSSDBytesForTesting() == restored.totalBytes)
        // The base's file is part of the head's chain — never orphaned.
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent(base.fileRelativePath).path
        ))
    }

    /// Minimal container file with the descriptor embedded in the
    /// header — what the rebuild walk recovers descriptors from.
    private func writeContainerFile(
        descriptor: PersistedSnapshotDescriptor,
        rootURL: URL
    ) throws {
        let payload = makePayload(
            bytes: descriptor.bytes,
            tokenOffset: descriptor.tokenOffset
        )
        let data = try encodePlaceholderContainer(payload: payload, descriptor: descriptor)
        let fileURL = rootURL.appendingPathComponent(descriptor.fileRelativePath)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: fileURL)
    }
}

// MARK: - 3. Store: front door, writer fold, hydration composition

struct LeafExtensionStoreTests {

    private func makeConfig() -> (SSDPrefixCacheConfig, URL) {
        let rootURL = makeScratchDir()
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: 1_000_000,
            maxPendingBytes: 10_000_000
        )
        return (config, rootURL)
    }

    private func makeStoreWithPartition(
        config: SSDPrefixCacheConfig,
        onCommit: @escaping @Sendable (String) -> Void = { _ in },
        onDrop: @escaping @Sendable (String, SSDDropReason) -> Void = { _, _ in },
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) -> SSDSnapshotStore {
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: .milliseconds(20),
            onCommit: onCommit,
            onDrop: onDrop,
            writerDrainPreludeForTesting: writerDrainPreludeForTesting
        )
        store.registerPartition(makePartitionMeta(), digest: testDigest)
        return store
    }

    @Test func enqueueRejectsExtensionWithUnavailableBase() {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        let extending = SnapshotExtension(baseSnapshotID: "ghost", baseOffset: 4)
        let result = store.tryEnqueue(
            payload: makePayload(bytes: 256, tokenOffset: 8, extending: extending),
            descriptor: makeDescriptor(id: "head", bytes: 256, tokenOffset: 8, segmentBaseOffset: 4)
        )

        #expect(result == .rejectedExtensionBaseUnavailable)
        #expect(store.pendingCountForTesting() == 0)
        #expect(store.transferringBaseIDsForTesting().isEmpty)
    }

    @Test func writerFoldsBaseChainAtCommit() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        let basePayload = makePayload(bytes: 800, tokenOffset: 4)
        let base = makeDescriptor(id: "base", bytes: 800, tokenOffset: 4)
        guard case .accepted = store.tryEnqueue(payload: basePayload, descriptor: base) else {
            Issue.record("base enqueue rejected")
            return
        }
        await store.flushAsync()

        let extending = SnapshotExtension(baseSnapshotID: "base", baseOffset: 4)
        let headPayload = makePayload(bytes: 200, tokenOffset: 9, extending: extending)
        let head = makeDescriptor(id: "head", bytes: 200, tokenOffset: 9, segmentBaseOffset: 4)
        guard case .accepted = store.tryEnqueue(payload: headPayload, descriptor: head) else {
            Issue.record("extension enqueue rejected")
            return
        }
        #expect(store.transferringBaseIDsForTesting() == ["base"])
        await store.flushAsync()

        // Folded: head owns the chain, base entry consumed, shield off.
        #expect(store.residentDescriptorForTesting(id: "base") == nil)
        #expect(store.transferringBaseIDsForTesting().isEmpty)
        let folded = try #require(store.residentDescriptorForTesting(id: "head"))
        #expect(folded.inheritedSegments.map(\.fileRelativePath) == [base.fileRelativePath])
        #expect(folded.totalBytes == 1_000)
        #expect(store.currentSSDBytesForTesting() == 1_000)

        // Both segment files exist on disk.
        for path in [base.fileRelativePath, head.fileRelativePath] {
            #expect(FileManager.default.fileExists(
                atPath: root.appendingPathComponent(path).path
            ))
        }
    }

    @Test func deletingPendingExtensionReleasesShield() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        // Hold the writer at its prelude so both items stay observably
        // queued; the gate releases it for the final drain.
        let gate = Gate()
        let store = makeStoreWithPartition(
            config: config,
            writerDrainPreludeForTesting: { await gate.wait() }
        )

        guard case .accepted = store.tryEnqueue(
            payload: makePayload(bytes: 800, tokenOffset: 4),
            descriptor: makeDescriptor(id: "base", bytes: 800, tokenOffset: 4)
        ) else {
            Issue.record("base enqueue rejected")
            return
        }
        // Base is queued (writer blocked), so the extension is accepted
        // on the queued/in-flight rule.
        guard case .accepted = store.tryEnqueue(
            payload: makePayload(
                bytes: 200, tokenOffset: 9,
                extending: SnapshotExtension(baseSnapshotID: "base", baseOffset: 4)
            ),
            descriptor: makeDescriptor(id: "head", bytes: 200, tokenOffset: 9, segmentBaseOffset: 4)
        ) else {
            Issue.record("extension enqueue rejected")
            return
        }
        #expect(store.transferringBaseIDsForTesting() == ["base"])

        // Deleting the still-queued extension must release the shield.
        store.deleteSnapshot(snapshotID: "head")
        #expect(store.transferringBaseIDsForTesting().isEmpty)
        #expect(store.pendingCountForTesting() == 1)

        gate.open()
        await store.flushAsync()

        // The base commits alone, un-superseded.
        #expect(store.residentDescriptorForTesting(id: "base") != nil)
        #expect(store.residentDescriptorForTesting(id: "head") == nil)
        #expect(store.currentSSDBytesForTesting() == 800)
    }

    /// Async latch for holding the writer loop at its test prelude.
    nonisolated final class Gate: @unchecked Sendable {
        private let lock = NSLock()
        private var isOpen = false

        func open() {
            lock.lock()
            isOpen = true
            lock.unlock()
        }

        private func check() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return isOpen
        }

        func wait() async {
            while !check() {
                try? await Task.sleep(for: .milliseconds(5))
            }
        }
    }

    @Test func hydrationComposesChainAcrossSegments() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        // Real arrays: the full leaf state at offset 7, with the base
        // capture being its first 4 token positions — exactly what two
        // consecutive turns of the same conversation produce.
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
        // Distinct recurrent state: hydration must surface the head's
        // copy (whole-state layers are last-segment-wins).
        mambaHead.state = [MLXArray.ones([1, 3, 16]), MLXArray.ones([1, 4, 8, 8])]
        let headSnapshot = HybridCacheSnapshot.capture(
            cache: [kvHead, mambaHead], offset: 7, type: .leaf
        )!

        // Extract through the production slicing path.
        let basePayload = ServerCompletion.extractSnapshotPayload(baseSnapshot)
        let base = makeDescriptor(id: "base", bytes: basePayload.totalBytes, tokenOffset: 4)
        guard case .accepted = store.tryEnqueue(payload: basePayload, descriptor: base) else {
            Issue.record("base enqueue rejected")
            return
        }
        await store.flushAsync()

        let headPayload = ServerCompletion.extractSnapshotPayload(
            headSnapshot,
            extending: SnapshotExtension(baseSnapshotID: "base", baseOffset: 4)
        )
        #expect(headPayload.extending != nil)
        let head = makeDescriptor(
            id: "head", bytes: headPayload.totalBytes, tokenOffset: 7, segmentBaseOffset: 4
        )
        guard case .accepted = store.tryEnqueue(payload: headPayload, descriptor: head) else {
            Issue.record("extension enqueue rejected")
            return
        }
        await store.flushAsync()

        let ref = SnapshotRef(
            snapshotID: "head",
            partitionDigest: testDigest,
            tokenOffset: 7,
            checkpointType: .leaf,
            bytesOnDisk: headPayload.totalBytes
        )
        let restored = try #require(store.loadSync(
            snapshotRef: ref,
            expectedFingerprint: testFingerprint
        ))

        #expect(restored.tokenOffset == 7)
        #expect(restored.layers.count == 2)

        // Sliced layer: base segment ⧺ suffix segment == the full state.
        // (`capture` normalizes the class name to the savePromptCache
        // wire convention — `KVCacheSimple` rides as "KVCache".)
        let restoredKV = restored.layers[0]
        #expect(restoredKV.className == "KVCache")
        #expect(restoredKV.state[0].shape == [1, 2, 7, 8])
        #expect(
            restoredKV.state[0].asData(access: .copy).data
                == keysFull.asData(access: .copy).data
        )
        #expect(
            restoredKV.state[1].asData(access: .copy).data
                == valuesFull.asData(access: .copy).data
        )

        // Whole-state layer: the head's copy wins, the base's is unused.
        let restoredMamba = restored.layers[1]
        #expect(restoredMamba.className == "MambaCache")
        #expect(
            restoredMamba.state[0].asData(access: .copy).data
                == MLXArray.ones([1, 3, 16]).asData(access: .copy).data
        )
    }

    @Test func hydrationFailureDropsWholeChain() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        guard case .accepted = store.tryEnqueue(
            payload: makePayload(bytes: 800, tokenOffset: 4),
            descriptor: makeDescriptor(id: "base", bytes: 800, tokenOffset: 4)
        ) else {
            Issue.record("base enqueue rejected")
            return
        }
        await store.flushAsync()
        let extending = SnapshotExtension(baseSnapshotID: "base", baseOffset: 4)
        let head = makeDescriptor(id: "head", bytes: 200, tokenOffset: 9, segmentBaseOffset: 4)
        guard case .accepted = store.tryEnqueue(
            payload: makePayload(bytes: 200, tokenOffset: 9, extending: extending),
            descriptor: head
        ) else {
            Issue.record("extension enqueue rejected")
            return
        }
        await store.flushAsync()
        let folded = try #require(store.residentDescriptorForTesting(id: "head"))

        // Sabotage one inherited segment file; hydration must fail and
        // sweep the entire chain — entry and every remaining file.
        let baseFileURL = root.appendingPathComponent(
            folded.inheritedSegments[0].fileRelativePath
        )
        try FileManager.default.removeItem(at: baseFileURL)

        let ref = SnapshotRef(
            snapshotID: "head",
            partitionDigest: testDigest,
            tokenOffset: 9,
            checkpointType: .leaf,
            bytesOnDisk: folded.totalBytes
        )
        #expect(store.loadSync(snapshotRef: ref, expectedFingerprint: testFingerprint) == nil)
        #expect(store.residentDescriptorForTesting(id: "head") == nil)
        #expect(store.currentSSDBytesForTesting() == 0)
        #expect(!FileManager.default.fileExists(
            atPath: root.appendingPathComponent(folded.fileRelativePath).path
        ))
    }
}

// MARK: - 4. Manager: supersession policy + extension-base resolution

@MainActor
struct LeafExtensionSupersessionPolicyTests {

    private func makePartitionKey() -> CachePartitionKey {
        CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint
        )
    }

    private typealias Fixture = (
        root: URL,
        manager: PrefixCacheManager,
        store: TieredSnapshotStore,
        key: CachePartitionKey
    )

    private func makeFixture() -> Fixture {
        let root = makeScratchDir()
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: 1_000_000,
            maxPendingBytes: 10_000_000
        )
        let store = TieredSnapshotStore(ssdConfig: config)
        let key = makePartitionKey()
        store.registerPartition(makePartitionMeta(), for: key)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 10_000_000,
            tieredStore: store
        )
        return (root, manager, store, key)
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

    /// Admit an SSD-backed ancestor leaf at `tokens` and wait for its
    /// ref to commit. Returns the committed snapshot ID.
    private func admitCommittedAncestor(
        _ fixture: Fixture,
        tokens: [Int]
    ) async throws -> String {
        let admission = try #require(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                offset: tokens.count, type: .leaf
            ),
            storage: .ramAndSSD(makePayload(bytes: 1_024, tokenOffset: tokens.count)),
            partitionKey: fixture.key,
            requestID: UUID()
        ))
        fixture.manager.admit(admission)
        // The pending ref appears synchronously; commitment (a manifest
        // entry) lands when the background writer settles.
        let tree = try #require(fixture.store.tree(for: fixture.key))
        let refID = try #require(
            tree.deepestRefBearingLeaf(tokens: tokens + [999])?.snapshotID
        )
        let ssdStore = try #require(fixture.store.ssdStoreForTesting)
        let committed = await waitUntil {
            ssdStore.residentDescriptorForTesting(id: refID) != nil
        }
        #expect(committed, "ancestor leaf never committed")
        return refID
    }

    @Test func extensionBaseResolvesDeepestRefBearingAncestor() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let ancestorTokens = Array(1...5)

        // No SSD-backed leaf yet → no base.
        #expect(fixture.manager.extensionBase(
            tokens: Array(1...9), partitionKey: fixture.key
        ) == nil)

        let baseID = try await admitCommittedAncestor(fixture, tokens: ancestorTokens)

        let base = try #require(fixture.manager.extensionBase(
            tokens: Array(1...9), partitionKey: fixture.key
        ))
        #expect(base.baseSnapshotID == baseID)
        #expect(base.baseOffset == 5)

        // Strictness: the leaf is never its own base.
        #expect(fixture.manager.extensionBase(
            tokens: ancestorTokens, partitionKey: fixture.key
        ) == nil)
    }

    @Test func ramOnlyAdmissionPreservesAncestorBacking() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let ancestorTokens = Array(1...5)
        let baseID = try await admitCommittedAncestor(fixture, tokens: ancestorTokens)

        let descendant = try #require(SnapshotAdmission.leaf(
            storedTokens: Array(1...8),
            snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 8, type: .leaf),
            storage: .ramOnly,
            partitionKey: fixture.key,
            requestID: UUID()
        ))
        let diagnostics = fixture.manager.admit(descendant)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].mode == .preserved)
        #expect(diagnostics.supersededLeaves[0].bodyDroppedSnapshotRefID == baseID)

        // The ancestor's SSD entry survives and remains the next
        // turn's extension base (the descendant has no SSD copy).
        #expect(
            fixture.store.ssdStoreForTesting?.residentDescriptorForTesting(id: baseID) != nil
        )
        let nextBase = try #require(fixture.manager.extensionBase(
            tokens: Array(1...12), partitionKey: fixture.key
        ))
        #expect(nextBase.baseSnapshotID == baseID)
    }

    @Test func fullSSDAdmissionDeletesAncestorBacking() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let ancestorTokens = Array(1...5)
        let baseID = try await admitCommittedAncestor(fixture, tokens: ancestorTokens)

        let descendant = try #require(SnapshotAdmission.leaf(
            storedTokens: Array(1...8),
            snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 8, type: .leaf),
            storage: .ramAndSSD(makePayload(bytes: 1_024, tokenOffset: 8)),
            partitionKey: fixture.key,
            requestID: UUID()
        ))
        let diagnostics = fixture.manager.admit(descendant)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].mode == .deleted)
        #expect(
            fixture.store.ssdStoreForTesting?.residentDescriptorForTesting(id: baseID) == nil
        )
    }

    @Test func extensionAdmissionTransfersAncestorBacking() async throws {
        let fixture = makeFixture()
        defer { cleanup(fixture.root) }
        let ancestorTokens = Array(1...5)
        let baseID = try await admitCommittedAncestor(fixture, tokens: ancestorTokens)

        let extending = SnapshotExtension(baseSnapshotID: baseID, baseOffset: 5)
        let descendant = try #require(SnapshotAdmission.leaf(
            storedTokens: Array(1...8),
            snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 8, type: .leaf),
            storage: .ramAndSSD(makePayload(bytes: 256, tokenOffset: 8, extending: extending)),
            partitionKey: fixture.key,
            requestID: UUID()
        ))
        let diagnostics = fixture.manager.admit(descendant)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].mode == .transferred)

        // After the writer settles, the head owns the folded chain.
        let ssdStore = try #require(fixture.store.ssdStoreForTesting)
        let folded = await waitUntil {
            ssdStore.residentDescriptorForTesting(id: baseID) == nil
                && ssdStore.transferringBaseIDsForTesting().isEmpty
        }
        #expect(folded, "extension fold never committed")
        let tree = try #require(fixture.store.tree(for: fixture.key))
        let headRef = try #require(tree.deepestRefBearingLeaf(tokens: Array(1...12)))
        let head = try #require(
            ssdStore.residentDescriptorForTesting(id: headRef.snapshotID)
        )
        #expect(head.inheritedSegments.count == 1)
        #expect(head.segmentBaseOffset == 5)
    }
}
