import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

struct HybridCacheSnapshotTests {

    // MARK: - Capture Tests

    @Test func captureStoresAllLayerStates() throws {
        // 32-layer Qwen3.5 hybrid: 24 Mamba + 8 attention (every 4th layer)
        var cache: [any KVCache] = []
        for i in 0..<32 {
            if i % 4 == 3 {
                let kv = KVCacheSimple()
                kv.state = [
                    MLXArray.zeros([1, 1, 10, 64]),
                    MLXArray.zeros([1, 1, 10, 64]),
                ]
                cache.append(kv)
            } else {
                let mamba = MambaCache()
                mamba.state = [
                    MLXArray.zeros([1, 3, 14336]),
                    MLXArray.zeros([1, 64, 128, 192]),
                ]
                cache.append(mamba)
            }
        }

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: cache, offset: 10, type: .system))

        #expect(snapshot.layers.count == 32)
        #expect(snapshot.tokenOffset == 10)
    }

    @Test func captureDeepCopiesArrays() throws {
        let kv = KVCacheSimple()
        let keys = MLXArray.ones([1, 1, 5, 64])
        let values = MLXArray.ones([1, 1, 5, 64])
        kv.state = [keys, values]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 5, type: .leaf))

        kv.state = [MLXArray.zeros([1, 1, 8, 64]), MLXArray.zeros([1, 1, 8, 64])]

        #expect(snapshot.layers[0].state[0].dim(2) == 5)
        #expect(snapshot.layers[0].state[1].dim(2) == 5)
    }

    @Test func captureRecordsCorrectClassName() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]

        let mamba = MambaCache()
        mamba.state = [MLXArray.zeros([1, 3, 14336]), MLXArray.zeros([1, 64, 128, 192])]

        let quantized = QuantizedKVCache(groupSize: 64, bits: 8)

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [kv, mamba, quantized], offset: 4, type: .system
        ))

        #expect(snapshot.layers[0].className == "KVCache")
        #expect(snapshot.layers[1].className == "MambaCache")
        #expect(snapshot.layers[2].className == "QuantizedKVCache")
    }

    @Test func captureWithMixedCacheTypes() throws {
        let simple = KVCacheSimple()
        simple.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]

        let quantized = QuantizedKVCache(groupSize: 64, bits: 8)

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [simple, quantized], offset: 4, type: .system
        ))

        #expect(snapshot.layers[0].className == "KVCache")
        #expect(snapshot.layers[1].className == "QuantizedKVCache")
    }

    @Test func memoryBytesMatchesSumOfTensorSizes() throws {
        let kv = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 10, 64])
        let values = MLXArray.zeros([1, 1, 10, 64])
        kv.state = [keys, values]

        let mamba = MambaCache()
        mamba.state = [MLXArray.zeros([1, 3, 100]), MLXArray.zeros([1, 64, 16, 24])]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv, mamba], offset: 10, type: .leaf))

        let expectedBytes = kv.state.reduce(0) { $0 + $1.nbytes }
            + mamba.state.reduce(0) { $0 + $1.nbytes }

        #expect(snapshot.memoryBytes == expectedBytes)
    }

    @Test func captureReturnsNilForCacheList() {
        let sub1 = KVCacheSimple()
        let sub2 = MambaCache()
        let cacheList = CacheList(sub1, sub2)

        let snapshot = HybridCacheSnapshot.capture(cache: [cacheList], offset: 0, type: .system)
        #expect(snapshot == nil)
    }

    @Test func captureReturnsNilWhenAnyCacheLayerIsCacheList() {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]
        let cacheList = CacheList(KVCacheSimple(), MambaCache())

        // One supported layer + one CacheList → entire capture returns nil
        let snapshot = HybridCacheSnapshot.capture(cache: [kv, cacheList], offset: 4, type: .system)
        #expect(snapshot == nil)
    }

    // MARK: - Restore Tests

    @Test func restoreCreatesKVCacheSimple() throws {
        let original = KVCacheSimple()
        original.state = [MLXArray.zeros([1, 1, 20, 64]), MLXArray.zeros([1, 1, 20, 64])]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [original], offset: 20, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored.count == 1)
        #expect(restored[0] is KVCacheSimple)
        #expect(restored[0].offset == 20)
    }

    @Test func restoreCreatesQuantizedKVCache() throws {
        let quantized = QuantizedKVCache(groupSize: 64, bits: 8)

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [quantized], offset: 0, type: .system))
        let restored = snapshot.restore()

        #expect(restored.count == 1)
        let restoredQ = restored[0] as? QuantizedKVCache
        #expect(restoredQ != nil)
        #expect(restoredQ?.groupSize == 64)
        #expect(restoredQ?.bits == 8)
    }

    @Test func restoreCreatesRotatingKVCache() throws {
        let rotating = RotatingKVCache(maxSize: 512)
        rotating.state = [MLXArray.zeros([1, 1, 10, 64]), MLXArray.zeros([1, 1, 10, 64])]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [rotating], offset: 10, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored.count == 1)
        #expect(restored[0] is RotatingKVCache)
        #expect(restored[0].maxSize == 512)
    }

    @Test func restoreCreatesMambaCache() throws {
        let mamba = MambaCache()
        mamba.state = [MLXArray.zeros([1, 3, 14336]), MLXArray.zeros([1, 64, 128, 192])]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [mamba], offset: 100, type: .system))
        let restored = snapshot.restore()

        #expect(restored.count == 1)
        #expect(restored[0] is MambaCache)
        #expect(restored[0].state.count == 2)
        #expect(restored[0].state[0].shape == [1, 3, 14336])
        #expect(restored[0].state[1].shape == [1, 64, 128, 192])
    }

    @Test func restoreCreatesChunkedKVCache() throws {
        let chunked = ChunkedKVCache()
        chunked.state = [MLXArray.zeros([1, 1, 10, 64]), MLXArray.zeros([1, 1, 10, 64])]
        chunked.metaState = ["256", "5"]  // chunkSize=256, startPosition=5

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [chunked], offset: 10, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored.count == 1)
        #expect(restored[0] is ChunkedKVCache)
        #expect(restored[0].metaState == ["256", "5"])
    }

    @Test func roundTripCaptureRestorePreservesState() throws {
        let kv = KVCacheSimple()
        let keys = MLXArray(Array(stride(from: Float(0), to: 640, by: 1)))
            .reshaped([1, 1, 10, 64])
        let values = MLXArray(Array(stride(from: Float(640), to: 1280, by: 1)))
            .reshaped([1, 1, 10, 64])
        kv.state = [keys, values]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 10, type: .leaf))
        let restored = snapshot.restore()

        let restoredKeys = restored[0].state[0]
        let restoredValues = restored[0].state[1]
        #expect(allClose(keys, restoredKeys, atol: 0).all().item(Bool.self))
        #expect(allClose(values, restoredValues, atol: 0).all().item(Bool.self))
    }

    @Test func restoredCacheIsIsolatedFromSnapshot() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 5, 64]), MLXArray.ones([1, 1, 5, 64])]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 5, type: .leaf))
        var restored = snapshot.restore()

        restored[0].state = [MLXArray.zeros([1, 1, 8, 64]), MLXArray.zeros([1, 1, 8, 64])]

        #expect(snapshot.layers[0].state[0].dim(2) == 5)
    }

    @Test func restoredKVCacheSimpleOffsetMatchesKeys() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 42, 64]), MLXArray.zeros([1, 1, 42, 64])]
        #expect(kv.offset == 42)

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 42, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored[0].offset == 42)
        #expect(restored[0].state[0].dim(2) == 42)
    }

    @Test func restoredQuantizedKVCacheHasCorrectBits() throws {
        let quantized = QuantizedKVCache(groupSize: 32, bits: 4)

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [quantized], offset: 0, type: .system))
        let restored = snapshot.restore()

        let restoredQ = restored[0] as? QuantizedKVCache
        #expect(restoredQ != nil)
        #expect(restoredQ?.groupSize == 32)
        #expect(restoredQ?.bits == 4)
    }

    @Test func restoreTrimmedChunkedKVCachePreservesOffset() throws {
        let chunked = ChunkedKVCache(chunkSize: 16)
        chunked.state = [MLXArray.ones([1, 1, 10, 64]), MLXArray.ones([1, 1, 10, 64])]
        chunked.metaState = ["16", "500"]
        chunked.offset = 510

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [chunked], offset: 510, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored[0] is ChunkedKVCache)
        #expect(restored[0].offset == 510)
        #expect(restored[0].metaState == ["16", "500"])
    }

    @Test func restoreCreatesArraysCache() throws {
        let arrays = ArraysCache(size: 3)
        arrays.state = [
            MLXArray.zeros([1, 4, 64]),
            MLXArray.ones([1, 4, 64]),
            MLXArray.zeros([1, 2, 32]),
        ]
        arrays.offset = 7

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [arrays], offset: 7, type: .leaf))
        let restored = snapshot.restore()

        #expect(restored[0] is ArraysCache)
        #expect(restored[0].offset == 7)
        #expect(restored[0].state.count == 3)
        #expect(restored[0].state[0].shape == [1, 4, 64])
        #expect(restored[0].state[2].shape == [1, 2, 32])
    }
}
