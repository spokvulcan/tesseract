import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// TurboQuant layers in the prefix cache (the **KV Scheme**, #603): a
/// compressed layer captures as sliceable attention, restores bitwise,
/// moves into a Leaf Handoff, serves a prefix view and survives the SSD
/// round trip.
struct TurboQuantSnapshotTests {

    /// A TurboQuant layer holding `rows` positions: `rows - 1` raw rows, then
    /// one decode step that compresses them and appends its own.
    static func turboLayer(keyBits: Int, rows: Int, seed: UInt64 = 7) -> TurboQuantKVCache {
        let cache = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
        let (keys, values) = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([1, 2, rows - 1, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 2, rows - 1, 128]).asType(.bfloat16)
            )
        }
        _ = cache.update(keys: keys, values: values)
        _ = step(cache, seed: seed + 1)
        return cache
    }

    /// One decode step through the compressed path.
    @discardableResult
    static func step(_ cache: TurboQuantKVCache, seed: UInt64) -> MLXArray {
        let (q, k, v) = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([1, 4, 1, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 2, 1, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 2, 1, 128]).asType(.bfloat16)
            )
        }
        let out = cache.compressedAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(128).squareRoot(), mask: .none)
        eval(out)
        return out
    }

    static func bytes(_ cache: any KVCache) -> [Data] {
        cache.state.map { $0.asData(access: .copy).data }
    }

    @Test(arguments: [0, 8])
    func aCompressedLayerCapturesAsSliceableAttentionAndRestoresBitwise(keyBits: Int) throws {
        let live = Self.turboLayer(keyBits: keyBits, rows: 300)
        #expect(live.isCompressed)
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: [live], offset: 300, type: .leaf))
        let layer = snapshot.layers[0]
        #expect(layer.className == "TurboQuantKVCache")
        #expect(layer.kind == .sliceableAttention)
        #expect(layer.metaState.count == 6)

        let restored = try #require(try snapshot.restore().first as? TurboQuantKVCache)
        #expect(restored.isCompressed)
        #expect(restored.offset == 300)
        #expect(restored.keyBits == keyBits && restored.valueBits == 4)
        #expect(restored.keyGroupSize == live.keyGroupSize)
        #expect(Self.bytes(restored) == Self.bytes(live))

        // Both keep decoding identically, and the restore owns its buffers.
        let a = Self.step(live, seed: 99)
        let b = Self.step(restored, seed: 99)
        #expect(a.asData(access: .copy).data == b.asData(access: .copy).data)
        let again = try #require(try snapshot.restore().first)
        #expect(again.offset == 300)
    }

    @Test func aRawPhaseLayerRoundTrips() throws {
        let raw = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4)
        _ = raw.update(
            keys: MLXRandom.normal([1, 2, 40, 128]).asType(.bfloat16),
            values: MLXRandom.normal([1, 2, 40, 128]).asType(.bfloat16))
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: [raw], offset: 40, type: .system))
        #expect(snapshot.layers[0].kind == .sliceableAttention)
        let restored = try #require(try snapshot.restore().first as? TurboQuantKVCache)
        #expect(!restored.isCompressed)
        #expect(Self.bytes(restored) == Self.bytes(raw))
    }

    /// A TurboQuant leaf is a moved body Leaf Handoff can check out: its
    /// attention trims back for **Leaf Rewind** like a plain layer.
    @Test func aTurboQuantLeafMovesAndChecksOut() throws {
        var cache: [any KVCache] = [Self.turboLayer(keyBits: 8, rows: 64)]
        #expect(HybridCacheSnapshot.canCaptureMoving(cache: cache))
        let moved = try #require(HybridCacheSnapshot.captureMoving(cache: &cache, offset: 64))
        #expect(moved.checkoutRefusal(maximumAdvance: 8) == nil)
        #expect(!moved.canCompress, "a TurboQuant body is never warm-compressed")
    }

    @Test func aPrefixViewRestoresTheLeafsPrefix() throws {
        let live = Self.turboLayer(keyBits: 0, rows: 120)
        let atView = Self.bytes(live)
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: [live], offset: 120, type: .branchPoint, prefixView: true))
        for seed in UInt64(200)..<205 { Self.step(live, seed: seed) }
        let leaf = try #require(
            HybridCacheSnapshot.capture(cache: [live], offset: 125, type: .leaf))
        let restored = try #require(try view.restore(backingLeaf: leaf).first)
        #expect(restored.offset == 120)
        #expect(Self.bytes(restored) == atView)
    }

    @Test func aStateTheSetterWouldDropIsARestoreError() {
        let live = Self.turboLayer(keyBits: 8, rows: 32)
        let state = live.state
        let wrongCount = HybridCacheSnapshot(
            tokenOffset: 32,
            layers: [
                .init(
                    className: "TurboQuantKVCache", state: Array(state.prefix(4)),
                    metaState: live.metaState, offset: 32)
            ],
            checkpointType: .leaf, memoryBytes: 0, createdAt: .now)
        #expect(throws: HybridCacheSnapshot.RestoreError.self) { _ = try wrongCount.restore() }
        var meta = live.metaState
        meta[4] = "seed"
        let badMeta = HybridCacheSnapshot(
            tokenOffset: 32,
            layers: [
                .init(className: "TurboQuantKVCache", state: state, metaState: meta, offset: 32)
            ],
            checkpointType: .leaf, memoryBytes: 0, createdAt: .now)
        #expect(throws: HybridCacheSnapshot.RestoreError.self) { _ = try badMeta.restore() }
    }

    static func plainLayer(rows: Int, seed: UInt64 = 11) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let (keys, values) = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([1, 2, rows, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 2, rows, 128]).asType(.bfloat16)
            )
        }
        _ = cache.update(keys: keys, values: values)
        return cache
    }

    /// A scheme partition's Stored Form: full-precision attention captures
    /// compressed, exactly as the scheme compresses the same rows.
    @Test(arguments: KVScheme.allCases)
    func aSchemePartitionStoresFullPrecisionAttentionCompressed(_ scheme: KVScheme) throws {
        let plain = Self.plainLayer(rows: 300)
        let snapshot = try #require(
            HybridCacheSnapshot.capture(
                cache: [plain], offset: 300, type: .system, storedForm: scheme))
        #expect(snapshot.layers[0].className == "TurboQuantKVCache")
        #expect(snapshot.layers[0].kind == .sliceableAttention)
        #expect(snapshot.memoryBytes < plain.state.reduce(0) { $0 + $1.nbytes })
        let restored = try #require(try snapshot.restore().first as? TurboQuantKVCache)
        #expect(restored.isCompressed && restored.keyBits == scheme.keyBits)

        let expected = scheme.makeCache()
        _ = expected.update(keys: plain.state[0], values: plain.state[1])
        expected.compress()
        #expect(Self.bytes(restored) == Self.bytes(expected))
        // The live layer is untouched.
        #expect(type(of: plain) == KVCacheSimple.self && plain.offset == 300)
    }

    /// A branch point captured during the full-precision prefill restores
    /// from the turn's converted, compressed leaf.
    @Test func aViewCapturedBeforeConversionRestoresFromTheCompressedLeaf() throws {
        var live: [any KVCache] = [Self.plainLayer(rows: 100)]
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: live, offset: 100, type: .branchPoint, prefixView: true,
                storedForm: .turbo0v4))
        #expect(view.isPrefixView && view.layers[0].className == "TurboQuantKVCache")
        let expected = KVScheme.turbo0v4.makeCache()
        _ = expected.update(keys: live[0].state[0], values: live[0].state[1])
        expected.compress()

        HybridCacheSnapshot.storeInForm(&live, scheme: .turbo0v4)
        let converted = try #require(live[0] as? TurboQuantKVCache)
        #expect(converted.isCompressed)
        for seed in UInt64(300)..<304 { Self.step(converted, seed: seed) }
        let leaf = try #require(
            HybridCacheSnapshot.capture(cache: live, offset: 104, type: .leaf))
        let restored = try #require(try view.restore(backingLeaf: leaf).first)
        #expect(restored.offset == 100)
        #expect(Self.bytes(restored) == Self.bytes(expected))
    }

    @Test(arguments: [0, 8])
    func aTurboQuantLeafSurvivesTheSSDRoundTrip(keyBits: Int) async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbo-ssd-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = SSDSnapshotStore(
            config: SSDPrefixCacheConfig(
                enabled: true, rootURL: root, budgetBytes: 100_000_000,
                maxPendingBytes: 100_000_000))
        let fingerprint = String(repeating: "a", count: 64)
        store.registerPartition(
            PartitionMeta(
                modelID: "m", modelFingerprint: fingerprint, kvBits: nil, kvGroupSize: 64,
                createdAt: 1, schemaVersion: SnapshotManifestSchema.currentVersion,
                kvScheme: keyBits == 8 ? "turbo8v4" : "turbo0v4"),
            digest: "abcd1234")

        let live = Self.turboLayer(keyBits: keyBits, rows: 200)
        let leaf = try #require(
            HybridCacheSnapshot.capture(cache: [live], offset: 200, type: .leaf))
        let payload = SnapshotPayload.extract(leaf)
        let id = UUID().uuidString
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: id, partitionDigest: "abcd1234", pathFromRoot: [1], tokenOffset: 200,
            checkpointType: "leaf", bytes: payload.totalBytes, createdAt: 1, lastAccessAt: 1,
            fileRelativePath: "partitions/abcd1234/snapshots/\(id.prefix(1))/\(id).safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion)
        guard case .accepted(let ref) = store.tryEnqueue(payload: payload, descriptor: descriptor)
        else {
            Issue.record("admission failed")
            return
        }
        await store.flushAsync()
        let loaded = try #require(
            store.loadSync(snapshotRef: ref, expectedFingerprint: fingerprint))
        let restored = try #require(try loaded.restore().first as? TurboQuantKVCache)
        #expect(restored.isCompressed)
        #expect(restored.offset == 200)
        #expect(Self.bytes(restored) == Self.bytes(live))
    }
}
