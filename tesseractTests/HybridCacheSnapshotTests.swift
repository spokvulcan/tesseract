import Foundation
import MLX
import MLXLMCommon
import Testing

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

    // MARK: - Safetensors persistence (Task 4.1.3)

    /// Allocate a fresh per-test safetensors file under the system temp
    /// directory and delete it when the closure exits. Used by the
    /// serialize/deserialize round-trip tests below.
    private func withTempSnapshotURL<T>(_ body: (URL) throws -> T) rethrows -> T {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("HybridCacheSnapshotTests-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        return try body(url)
    }

    @Test func serializeRoundTripPreservesKVCacheSimpleContent() throws {
        let kv = KVCacheSimple()
        let keys = MLXArray(Array(stride(from: Float(0), to: 640, by: 1)))
            .reshaped([1, 1, 10, 64])
        let values = MLXArray(Array(stride(from: Float(640), to: 1280, by: 1)))
            .reshaped([1, 1, 10, 64])
        kv.state = [keys, values]

        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 10, type: .leaf))

        try withTempSnapshotURL { url in
            try snapshot.serialize(
                to: url,
                metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp-abc"]
            )

            let restored = try HybridCacheSnapshot.deserialize(
                from: url,
                expectedFingerprint: "fp-abc"
            )

            #expect(restored.tokenOffset == 10)
            #expect(restored.checkpointType == .leaf)
            #expect(restored.layers.count == 1)
            #expect(restored.layers[0].className == "KVCache")
            #expect(restored.layers[0].offset == 10)
            #expect(allClose(keys, restored.layers[0].state[0], atol: 0).all().item(Bool.self))
            #expect(allClose(values, restored.layers[0].state[1], atol: 0).all().item(Bool.self))
        }
    }

    @Test func serializeRoundTripAllCheckpointTypes() throws {
        let cases: [HybridCacheSnapshot.CheckpointType] = [
            .system, .lastMessageBoundary, .leaf, .branchPoint,
        ]
        for ckptType in cases {
            let kv = KVCacheSimple()
            kv.state = [
                MLXArray.zeros([1, 1, 4, 64]),
                MLXArray.zeros([1, 1, 4, 64]),
            ]
            let snapshot = try #require(HybridCacheSnapshot.capture(
                cache: [kv], offset: 4, type: ckptType
            ))
            try withTempSnapshotURL { url in
                try snapshot.serialize(
                    to: url,
                    metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"]
                )
                let restored = try HybridCacheSnapshot.deserialize(
                    from: url, expectedFingerprint: "fp"
                )
                #expect(restored.checkpointType == ckptType,
                        "round-trip failed for \(ckptType)")
            }
        }
    }

    @Test func serializeRoundTripPreservesChunkedKVCacheAbsoluteOffset() throws {
        // ChunkedKVCache is the one cache type where
        // `cache.state = ...` inside loadPromptCache resets offset to
        // `keys.dim(2)` (the truncated chunk count) rather than the
        // caller's absolute prompt position. The per-layer offset
        // metadata written by serialize must restore the absolute value
        // so warm-start hydration matches capture semantics.
        let chunked = ChunkedKVCache(chunkSize: 16)
        chunked.state = [
            MLXArray.ones([1, 1, 10, 64]),
            MLXArray.ones([1, 1, 10, 64]),
        ]
        chunked.metaState = ["16", "500"]
        chunked.offset = 510

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [chunked], offset: 510, type: .leaf
        ))

        try withTempSnapshotURL { url in
            try snapshot.serialize(
                to: url,
                metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"]
            )
            let restored = try HybridCacheSnapshot.deserialize(
                from: url, expectedFingerprint: "fp"
            )
            #expect(restored.tokenOffset == 510)
            #expect(restored.layers[0].offset == 510)
            #expect(restored.layers[0].metaState == ["16", "500"])
        }
    }

    @Test func serializeRoundTripIsBitwiseIdenticalAfterRecapture() throws {
        // Bitwise round-trip per Task 4.1.3:
        // capture → serialize → deserialize → restore → capture → compare
        // every LayerState.state array byte-for-byte.
        let kv = KVCacheSimple()
        let keys = MLXArray(Array(stride(from: Float(0), to: 640, by: 1)))
            .reshaped([1, 1, 10, 64])
        let values = MLXArray(Array(stride(from: Float(640), to: 1280, by: 1)))
            .reshaped([1, 1, 10, 64])
        kv.state = [keys, values]

        let mamba = MambaCache()
        mamba.state = [
            MLXArray(Array(stride(from: Float(0), to: 300, by: 1))).reshaped([1, 3, 100]),
            MLXArray(Array(stride(from: Float(0), to: Float(1_024), by: 1))).reshaped([1, 4, 16, 16]),
        ]

        let snap1 = try #require(HybridCacheSnapshot.capture(
            cache: [kv, mamba], offset: 10, type: .system
        ))

        try withTempSnapshotURL { url in
            try snap1.serialize(
                to: url,
                metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"]
            )
            let snap2 = try HybridCacheSnapshot.deserialize(
                from: url, expectedFingerprint: "fp"
            )

            // Hydrate snap2 into live caches, re-capture on top — mirrors
            // the warm-start hydration + continued-prefill path.
            let hydrated = snap2.restore()
            let snap3 = try #require(HybridCacheSnapshot.capture(
                cache: hydrated, offset: snap2.tokenOffset, type: snap2.checkpointType
            ))

            #expect(snap1.layers.count == snap3.layers.count)
            for (a, b) in zip(snap1.layers, snap3.layers) {
                #expect(a.className == b.className)
                #expect(a.metaState == b.metaState)
                #expect(a.offset == b.offset)
                #expect(a.state.count == b.state.count)
                for (arrA, arrB) in zip(a.state, b.state) {
                    #expect(arrA.shape == arrB.shape)
                    #expect(arrA.dtype == arrB.dtype)
                    #expect(arrA.asData() == arrB.asData(),
                            "layer state bytes mismatch after round-trip")
                }
            }
        }
    }

    @Test func deserializeThrowsOnFingerprintMismatch() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]
        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 4, type: .system))

        try withTempSnapshotURL { url in
            try snapshot.serialize(
                to: url,
                metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "expected"]
            )
            #expect(throws: HybridCacheSnapshot.SerializationError.self) {
                _ = try HybridCacheSnapshot.deserialize(
                    from: url, expectedFingerprint: "different"
                )
            }
        }
    }

    @Test func deserializeThrowsOnMissingFingerprint() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]
        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 4, type: .system))

        try withTempSnapshotURL { url in
            try snapshot.serialize(to: url, metadata: [:])
            #expect(throws: HybridCacheSnapshot.SerializationError.self) {
                _ = try HybridCacheSnapshot.deserialize(
                    from: url, expectedFingerprint: "anything"
                )
            }
        }
    }

    @Test func serializePreservesCallerMetadataUnderNonReservedKeys() throws {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 4, 64]), MLXArray.zeros([1, 1, 4, 64])]
        let snapshot = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 4, type: .leaf))

        try withTempSnapshotURL { url in
            try snapshot.serialize(
                to: url,
                metadata: [
                    "user_note": "hello world",
                    "other_key": "42",
                    HybridCacheSnapshot.MetadataKey.fingerprint: "fp",
                ]
            )

            let (_, md) = try loadPromptCache(url: url)
            #expect(md["user_note"] == "hello world")
            #expect(md["other_key"] == "42")
            #expect(md[HybridCacheSnapshot.MetadataKey.fingerprint] == "fp")
            #expect(md[HybridCacheSnapshot.MetadataKey.checkpointType] == "leaf")
            #expect(md[HybridCacheSnapshot.MetadataKey.tokenOffset] == "4")
        }
    }

    @Test func serializeRoundTripPreservesQuantizedKVCacheGroupSizeAndBits() throws {
        // Regression for P1: deserialize previously read metaState via the
        // live cache's getter after loadPromptCache, but loadPromptCache
        // constructs `QuantizedKVCache()` with default groupSize/bits and
        // the type-specific metaState setter only restores offset — so
        // non-default quantization settings silently collapsed back to
        // 64/8 across the round-trip. The reserved-key metaState mirror
        // in serialize/deserialize pins the captured values.
        let quantized = QuantizedKVCache(groupSize: 32, bits: 4)
        // QuantizedKVCache.state setter requires count 4 or 6. Shapes
        // are arbitrary; the safetensors layer round-trips bytes without
        // interpreting quantization meaning. `offset == keys.dim(2)`
        // avoids the getter's trim branch.
        quantized.state = [
            MLXArray.zeros([1, 1, 8, 16]),
            MLXArray.zeros([1, 1, 8, 2]),
            MLXArray.zeros([1, 1, 8, 16]),
            MLXArray.zeros([1, 1, 8, 2]),
        ]
        quantized.offset = 8

        let expectedMetaState = quantized.metaState
        // Sanity: the captured metaState must carry the non-default
        // quantization params at the positions consumed by
        // HybridCacheSnapshot.makeQuantizedCache (index 2 / 3).
        #expect(expectedMetaState.count == 4)
        #expect(expectedMetaState[2] == "32")
        #expect(expectedMetaState[3] == "4")

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [quantized], offset: 8, type: .leaf
        ))
        #expect(snapshot.layers[0].className == "QuantizedKVCache")
        #expect(snapshot.layers[0].metaState == expectedMetaState)

        try withTempSnapshotURL { url in
            try snapshot.serialize(
                to: url,
                metadata: [HybridCacheSnapshot.MetadataKey.fingerprint: "fp"]
            )
            let restored = try HybridCacheSnapshot.deserialize(
                from: url, expectedFingerprint: "fp"
            )

            #expect(restored.layers.count == 1)
            #expect(restored.layers[0].className == "QuantizedKVCache")
            #expect(restored.layers[0].metaState == expectedMetaState)

            let hydrated = restored.restore()
            let q = try #require(hydrated[0] as? QuantizedKVCache)
            #expect(q.groupSize == 32)
            #expect(q.bits == 4)
        }
    }

    @Test func threadAffinityContractDocCommentIsPinned() throws {
        // The thread-affinity contract must appear on both serialize and
        // deserialize doc comments so future refactors can't silently drop
        // it. Exact phrase matched here is the same one the plan specifies
        // for the contract (see Task 4.1.3 in
        // docs/marconi-hybrid-prefix-cache-implementation-plan.md).
        let testFile = URL(fileURLWithPath: #filePath)
        let projectRoot = testFile
            .deletingLastPathComponent()   // tesseractTests
            .deletingLastPathComponent()   // project root
        let vendorFile = projectRoot
            .appendingPathComponent("Vendor")
            .appendingPathComponent("mlx-swift-lm")
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXLMCommon")
            .appendingPathComponent("HybridCacheSnapshot.swift")

        let source = try String(contentsOf: vendorFile, encoding: .utf8)

        let phrase = "must be called from inside `container.perform`"
        let occurrences = source.components(separatedBy: phrase).count - 1
        #expect(
            occurrences >= 2,
            """
            Thread-affinity contract phrase must appear in the doc comments \
            for both serialize and deserialize — found \(occurrences) \
            occurrence(s) at \(vendorFile.path).
            """
        )
    }
}
