import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// LFM3's cache layout through the snapshot paths: a `MambaCache` per short
/// convolution (its offset stays 0; the attention layers carry the token
/// count), a `RotatingKVCache` per sliding-window layer and a `KVCacheSimple`
/// per global one. A snapshot taken after the ring wrapped must restore, by
/// copy and from the SSD form, to a cache that continues exactly as the live
/// one does.
struct LFM3CacheSnapshotTests {

    /// LFM3's shapes scaled down: a 4-token window (256 in the checkpoint),
    /// 2 KV heads of 8 dimensions, a width-3 convolution over 16 channels.
    private static let window = 4
    private static let convState = 2
    private static let channels = 16

    private static func liveCache() -> [any KVCache] {
        [MambaCache(), RotatingKVCache(maxSize: window), KVCacheSimple()]
    }

    /// Advance every layer over `tokens` as the model does: the convolution
    /// keeps its last `kernel - 1` inputs and moves no offset, attention
    /// appends a key and a value per token.
    private static func feed(_ cache: [any KVCache], tokens: Range<Int>) {
        let count = tokens.count
        let base = Float(tokens.lowerBound)
        let conv = cache[0] as! MambaCache  // swiftlint:disable:this force_cast
        let inputs = MLXArray((0..<(count * channels)).map { base + Float($0) / 64 })
            .reshaped([1, count, channels])
        let previous = conv[0] ?? MLXArray.zeros([1, convState, channels])
        let joined = concatenated([previous, inputs], axis: 1)
        conv[0] = joined[0..., (joined.dim(1) - convState)..., 0...]
        conv.advance(count)
        for layer in cache[1...] {
            let keys = MLXArray((0..<(2 * count * 8)).map { base + Float($0) / 128 })
                .reshaped([1, 2, count, 8])
            _ = layer.update(keys: keys, values: keys + 1)
        }
        eval(cache.flatMap(\.state))
    }

    /// A prefill chunk past the window, then single-token steps that wrap
    /// the ring: the state a leaf holds after a short turn.
    private static func wrappedCache() -> [any KVCache] {
        let cache = liveCache()
        feed(cache, tokens: 0..<6)
        feed(cache, tokens: 6..<7)
        feed(cache, tokens: 7..<8)
        return cache
    }

    private static func expectSameState(
        _ lhs: [any KVCache], _ rhs: [any KVCache],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(lhs.count == rhs.count, sourceLocation: sourceLocation)
        for (index, (a, b)) in zip(lhs, rhs).enumerated() {
            #expect(
                String(describing: type(of: a)) == String(describing: type(of: b)),
                "layer \(index)", sourceLocation: sourceLocation)
            #expect(a.offset == b.offset, "layer \(index)", sourceLocation: sourceLocation)
            #expect(a.metaState == b.metaState, "layer \(index)", sourceLocation: sourceLocation)
            #expect(
                a.state.map(\.shape) == b.state.map(\.shape), "layer \(index)",
                sourceLocation: sourceLocation)
            for (x, y) in zip(a.state, b.state) {
                #expect(
                    x.asData(access: .copy).data == y.asData(access: .copy).data,
                    "layer \(index)", sourceLocation: sourceLocation)
            }
        }
    }

    @Test func captureRecordsEachLayerAsLFM3HoldsIt() throws {
        let cache = Self.wrappedCache()
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: cache, offset: 8, type: .leaf))

        #expect(snapshot.layers.map(\.className) == ["MambaCache", "RotatingKVCache", "KVCache"])
        #expect(snapshot.layers.map(\.offset) == [0, 8, 8])
        #expect(snapshot.layers.map(\.kind) == [.wholeState, .wholeState, .sliceableAttention])
        // The pinned ring records its origin and layout and no rewind
        // reserve: the 7-value form the restore accepts.
        #expect(snapshot.layers[1].metaState.count == 7)
    }

    @Test func copyRestoreContinuesLikeTheLiveCache() throws {
        let live = Self.wrappedCache()
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: live, offset: 8, type: .leaf))
        let restored = try snapshot.restore()
        Self.expectSameState(live, restored)

        Self.feed(live, tokens: 8..<9)
        Self.feed(restored, tokens: 8..<9)
        Self.expectSameState(live, restored)
    }

    @Test func ssdRoundTripContinuesLikeTheLiveCache() throws {
        let live = Self.wrappedCache()
        let snapshot = try #require(
            HybridCacheSnapshot.capture(cache: live, offset: 8, type: .leaf))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LFM3CacheSnapshotTests-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try snapshot.serialize(
            to: url, metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"])
        let loaded = try HybridCacheSnapshot.deserialize(from: url, expectedFingerprint: "fp")
        #expect(loaded.layers.map(\.offset) == [0, 8, 8])
        let restored = try loaded.restore()

        Self.feed(live, tokens: 8..<9)
        Self.feed(restored, tokens: 8..<9)
        Self.expectSameState(live, restored)
    }

    /// A ring cannot give its prefix back, so a moved LFM3 leaf is never
    /// checked out: the turn restores it by copy instead.
    @Test func aMovedLeafIsRefusedCheckOutForItsRing() throws {
        var cache = Self.wrappedCache()
        let snapshot = try #require(HybridCacheSnapshot.captureMoving(cache: &cache, offset: 8))
        #expect(snapshot.checkoutRefusal(maximumAdvance: 16) == .rotating)
    }
}
