import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// One per-layer kind for snapshot layers (#521): sliceable attention or
/// whole-state, derived once from the vendor cache class and the array
/// shapes when a layer state is built — captured, moved, deserialized or
/// hydrated from a **Segment Chain** — and read by every consumer that
/// slices, composes, trims or rebuilds. No behavior change: the extraction
/// edge, chain composition, check-out eligibility and **Leaf Rewind** must
/// decide exactly as their local class matches did.
struct SnapshotLayerKindTests {

    typealias Kind = HybridCacheSnapshot.LayerState.Kind

    // MARK: - Fixtures: one cache per vendor class the snapshot supports

    /// Every vendor cache class `HybridCacheSnapshot` captures, named by
    /// the `savePromptCache` wire convention (`KVCacheSimple` rides as
    /// `"KVCache"`). Each builds a cache holding eight tokens' state.
    enum VendorClass: String, CaseIterable {
        case simple = "KVCache"
        case quantized = "QuantizedKVCache"
        case rotating = "RotatingKVCache"
        case chunked = "ChunkedKVCache"
        case arrays = "ArraysCache"
        case mamba = "MambaCache"

        /// Attention state slices along the token axis; recurrent,
        /// rotating and chunked state rides whole.
        var expectedKind: Kind {
            switch self {
            case .simple, .quantized: .sliceableAttention
            case .rotating, .chunked, .arrays, .mamba: .wholeState
            }
        }

        func make(tokens: Int) -> any KVCache {
            switch self {
            case .simple:
                let cache = KVCacheSimple()
                cache.state = [
                    MLXArray.ones([1, 2, tokens, 16]), MLXArray.ones([1, 2, tokens, 16]),
                ]
                return cache
            case .quantized:
                // Four arrays (no biases): packed keys/values plus scales.
                let cache = QuantizedKVCache(groupSize: 64, bits: 8)
                cache.state = [
                    MLXArray.zeros([1, 1, tokens, 16]), MLXArray.zeros([1, 1, tokens, 2]),
                    MLXArray.zeros([1, 1, tokens, 16]), MLXArray.zeros([1, 1, tokens, 2]),
                ]
                cache.offset = tokens
                return cache
            case .rotating:
                let cache = RotatingKVCache(maxSize: 32)
                _ = cache.update(
                    keys: MLXArray.ones([1, 1, tokens, 16]),
                    values: MLXArray.ones([1, 1, tokens, 16]))
                eval(cache)
                return cache
            case .chunked:
                let cache = ChunkedKVCache(chunkSize: 16)
                cache.state = [
                    MLXArray.ones([1, 1, tokens, 16]), MLXArray.ones([1, 1, tokens, 16]),
                ]
                return cache
            case .arrays:
                let cache = ArraysCache(size: 2)
                cache.state = [MLXArray.zeros([1, 4, 16]), MLXArray.zeros([1, 2, 8, 8])]
                cache.offset = tokens
                return cache
            case .mamba:
                let cache = MambaCache()
                cache.state = [MLXArray.zeros([1, 3, 16]), MLXArray.zeros([1, 4, 8, 8])]
                cache.offset = tokens
                return cache
            }
        }
    }

    // MARK: - Derivation from the vendor cache class

    @Test(arguments: VendorClass.allCases)
    func captureDerivesTheKindFromTheVendorCacheClass(vendorClass: VendorClass) throws {
        let snapshot = try #require(
            HybridCacheSnapshot.capture(
                cache: [vendorClass.make(tokens: 8)], offset: 8, type: .leaf))
        let layer = try #require(snapshot.layers.first)
        #expect(layer.className == vendorClass.rawValue)
        #expect(layer.kind == vendorClass.expectedKind)
    }

    @Test func aMovedLeafBodyCarriesTheSameKinds() throws {
        var cache: [any KVCache] = [
            VendorClass.simple.make(tokens: 8),
            VendorClass.mamba.make(tokens: 8),
            VendorClass.rotating.make(tokens: 8),
        ]
        let moved = try #require(HybridCacheSnapshot.captureMoving(cache: &cache, offset: 8))
        #expect(moved.layers.map(\.kind) == [.sliceableAttention, .wholeState, .wholeState])
    }

    // MARK: - The shape guard

    /// Ways an attention layer's arrays can fail to cover
    /// `[0..<snapshotOffset]` along the token axis at dim −2.
    enum ShapeFault: String, CaseIterable {
        /// The layer's own offset is behind the snapshot's: its arrays
        /// end before the position the snapshot claims.
        case layerBehindTheSnapshotOffset
        /// The layer claims the snapshot's offset but its token axis is
        /// shorter.
        case tokenAxisShort
        /// Arrays with no separable token axis.
        case flatArrays
        /// No arrays at all.
        case emptyState
    }

    @Test(arguments: ShapeFault.allCases, [VendorClass.simple, VendorClass.quantized])
    func theShapeGuardKeepsAMisShapedAttentionLayerWholeState(
        fault: ShapeFault, vendorClass: VendorClass
    ) throws {
        let className = vendorClass.rawValue
        let layer: HybridCacheSnapshot.LayerState
        switch fault {
        case .layerBehindTheSnapshotOffset:
            // Through the real class, so the guard runs against the
            // snapshot's offset rather than the layer's own.
            let snapshot = try #require(
                HybridCacheSnapshot.capture(
                    cache: [vendorClass.make(tokens: 8)], offset: 12, type: .leaf))
            layer = snapshot.layers[0]
            #expect(layer.offset == 8)
        case .tokenAxisShort:
            layer = HybridCacheSnapshot.LayerState(
                className: className,
                state: [MLXArray.zeros([1, 2, 6, 16]), MLXArray.zeros([1, 2, 6, 16])],
                metaState: [], offset: 8, snapshotOffset: 8)
        case .flatArrays:
            layer = HybridCacheSnapshot.LayerState(
                className: className,
                state: [MLXArray.zeros([8, 16]), MLXArray.zeros([8, 16])],
                metaState: [], offset: 8, snapshotOffset: 8)
        case .emptyState:
            layer = HybridCacheSnapshot.LayerState(
                className: className, state: [], metaState: [], offset: 8, snapshotOffset: 8)
        }
        #expect(layer.kind == .wholeState)

        // Control: the same class with arrays that cover the offset.
        let sound = HybridCacheSnapshot.LayerState(
            className: className,
            state: [MLXArray.zeros([1, 2, 8, 16]), MLXArray.zeros([1, 2, 8, 16])],
            metaState: [], offset: 8, snapshotOffset: 8)
        #expect(sound.kind == .sliceableAttention)
    }

    // MARK: - Layers built off the live cache carry the kind too

    @Test func deserializedLayersCarryTheKind() throws {
        let snapshot = try #require(
            HybridCacheSnapshot.capture(
                cache: [VendorClass.simple.make(tokens: 8), VendorClass.mamba.make(tokens: 8)],
                offset: 8, type: .leaf))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("layer-kind-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try snapshot.serialize(
            to: url, metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"])
        let restored = try HybridCacheSnapshot.deserialize(from: url, expectedFingerprint: "fp")
        #expect(restored.layers.map(\.kind) == [.sliceableAttention, .wholeState])
    }

    /// A two-segment **Segment Chain** (a full base plus a **Leaf
    /// Extension Admission** suffix) composes per layer — the attention
    /// suffix appended, the recurrent copy reset — and the composed
    /// layers come back with their kind.
    @Test func aHydratedSegmentChainCarriesTheKind() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("layer-kind-chain-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fingerprint = String(repeating: "a", count: 64)
        let digest = "abcd1234"
        let store = SSDSnapshotStore(
            config: SSDPrefixCacheConfig(
                enabled: true, rootURL: root, budgetBytes: 1_000_000,
                maxPendingBytes: 10_000_000),
            manifestDebounce: .milliseconds(20))
        store.registerPartition(
            PartitionMeta(
                modelID: "layer-kind", modelFingerprint: fingerprint, kvBits: nil,
                kvGroupSize: 64, createdAt: 100_000,
                schemaVersion: SnapshotManifestSchema.currentVersion),
            digest: digest)
        func descriptor(
            id: String, bytes: Int, tokenOffset: Int, segmentBaseOffset: Int = 0
        ) -> PersistedSnapshotDescriptor {
            PersistedSnapshotDescriptor(
                snapshotID: id, partitionDigest: digest,
                pathFromRoot: Array(1...tokenOffset), tokenOffset: tokenOffset,
                checkpointType: "leaf", bytes: bytes, segmentBaseOffset: segmentBaseOffset,
                createdAt: 100_000, lastAccessAt: 0,
                fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                    snapshotID: id, partitionDigest: digest),
                schemaVersion: SnapshotManifestSchema.currentVersion)
        }

        let base = try #require(
            HybridCacheSnapshot.capture(
                cache: [VendorClass.simple.make(tokens: 4), VendorClass.mamba.make(tokens: 4)],
                offset: 4, type: .leaf))
        let head = try #require(
            HybridCacheSnapshot.capture(
                cache: [VendorClass.simple.make(tokens: 7), VendorClass.mamba.make(tokens: 7)],
                offset: 7, type: .leaf))

        let basePayload = ServerCompletion.extractSnapshotPayload(base)
        guard
            case .accepted = store.tryEnqueue(
                payload: basePayload,
                descriptor: descriptor(id: "base", bytes: basePayload.totalBytes, tokenOffset: 4))
        else {
            Issue.record("base enqueue rejected")
            return
        }
        await store.flushAsync()
        let headPayload = ServerCompletion.extractSnapshotPayload(
            head, extending: SnapshotExtension(baseSnapshotID: "base", baseOffset: 4))
        #expect(headPayload.extending != nil)
        guard
            case .accepted = store.tryEnqueue(
                payload: headPayload,
                descriptor: descriptor(
                    id: "head", bytes: headPayload.totalBytes, tokenOffset: 7,
                    segmentBaseOffset: 4))
        else {
            Issue.record("extension enqueue rejected")
            return
        }
        await store.flushAsync()

        let hydrated = try #require(
            store.loadSync(
                snapshotRef: SnapshotRef(
                    snapshotID: "head", partitionDigest: digest, tokenOffset: 7,
                    checkpointType: .leaf, bytesOnDisk: headPayload.totalBytes),
                expectedFingerprint: fingerprint))
        #expect(hydrated.layers.map(\.kind) == [.sliceableAttention, .wholeState])
        #expect(hydrated.layers[0].state[0].shape == [1, 2, 7, 16])
    }

    // MARK: - Consumers read the kind

    /// The extraction edge slices by kind, not by class: an attention
    /// layer the shape guard kept whole rides whole in an extension payload
    /// beside a sliceable layer of the same class.
    @Test func theExtractionEdgeSlicesByKind() throws {
        let sliceable = VendorClass.simple.make(tokens: 6)
        let behind = VendorClass.simple.make(tokens: 4)
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: [sliceable, behind], offset: 6, type: .leaf))
        #expect(snapshot.layers.map(\.kind) == [.sliceableAttention, .wholeState])

        let payload = ServerCompletion.extractSnapshotPayload(
            snapshot, extending: SnapshotExtension(baseSnapshotID: "base", baseOffset: 2))
        #expect(payload.extending != nil)
        let layers = payload.layers
        #expect(layers.count == 2)
        #expect(layers[0].suffixBaseOffset == 2)
        #expect(layers[0].state.allSatisfy { $0.shape == [1, 2, 4, 16] })
        #expect(layers[1].suffixBaseOffset == nil)
        #expect(layers[1].state.allSatisfy { $0.shape == [1, 2, 4, 16] })
    }

    /// Check-out eligibility reads the kind of the moved body: a
    /// whole-state layer that is not recurrent cannot promise a **Leaf
    /// Rewind**, so the hit restores by copy with the reason a plain
    /// attention layer reports — `untrimmable`, as before the kind.
    @MainActor
    @Test func checkOutEligibilityReadsTheKindOfAMovedBody() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let key = CachePartitionKey(modelID: "layer-kind", kvBits: nil, kvGroupSize: 64)
        let previous = FinalGenerationCache([VendorClass.simple.make(tokens: 8)])
        let body = try #require(previous.moveSnapshot(offset: 12))
        #expect(body.layers[0].kind == .wholeState)
        let tokens = Array(1...12)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let (outcome, handOver) = await manager.checkOutHoldingClaim(
            .init(
                lookup: manager.lookup(tokens: tokens + [13], partitionKey: key),
                hydratedFromSSD: false, hydrationSeconds: 0),
            tokens: tokens + [13],
            diagnostics: .init(
                requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64))
        guard case .copy(let copy) = outcome else {
            Issue.record("a non-recurrent whole-state layer must restore by copy")
            return
        }
        #expect(copy.reason == .untrimmable)
        #expect(copy.refusal == .untrimmable)
        #expect(!body.layers.isEmpty, "a refused check-out leaves the body in the tree")
        #expect(await handOver.rewindAndConclude() == nil)
    }
}
