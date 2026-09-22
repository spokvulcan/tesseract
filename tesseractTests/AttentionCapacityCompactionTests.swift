import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// #534: retained attention capacity is compacted on Leaf Rewind and
/// check-in when it exceeds the measured threshold.
@Suite(.serialized)
struct AttentionCapacityCompactionTests {
    /// A small full-attention layout: 2 KV heads × 64 dims, float16, so one
    /// row is 2 × 2 × 64 × 2 bytes = 512 bytes per layer.
    private func makeCache(layers: Int, rows: Int, step: Int = 256) -> [any KVCache] {
        (0..<layers).map { _ in
            let cache = KVCacheSimple()
            cache.step = step
            let keys = MLXArray.ones([1, 2, rows, 64], dtype: .float16)
            _ = cache.update(keys: keys, values: keys)
            return cache
        }
    }

    private func addresses(_ cache: [any KVCache]) -> [UInt] {
        cache.flatMap { $0.innerState() }.map { array in
            array.asData(access: .noCopyIfContiguous).data.withUnsafeBytes {
                UInt(bitPattern: $0.baseAddress)
            }
        }
    }

    @Test func rewindAfterALongGenerationCompactsToTheOffsetPlusOneStep() {
        // 8 layers × 512 B/row: a 2,048-row prompt is a 8 MB body, whose
        // quarter (2 MB) is the threshold; 20,000 trimmed rows retain ~80 MB.
        let cache = makeCache(layers: 8, rows: 2_048 + 20_000)
        for layer in cache { layer.trim(20_000) }
        // Capacity is what the vendor's growth granules rounded up to.
        let capacity = cache[0].innerState()[0].dim(2)
        #expect(capacity >= 2_048 + 20_000)
        let before = AttentionCapacityCompaction.measure(cache)
        #expect(before.bodyBytes == 8 * 512 * 2_048)
        #expect(before.retainedBytes == 8 * 512 * (capacity - 2_048))
        #expect(before.thresholdBytes == before.bodyBytes / 4)
        let oldAddresses = addresses(cache)

        let outcome = AttentionCapacityCompaction.compactIfNeeded(cache)

        #expect(outcome.compactedLayers == 8)
        #expect(outcome.freedBytes == 8 * 512 * (capacity - 2_048 - 256))
        for layer in cache {
            #expect(layer.offset == 2_048)
            let arrays = layer.innerState()
            #expect(arrays.map { $0.dim(2) } == [2_048 + 256, 2_048 + 256])
            #expect(layer.state.map { $0.dim(2) } == [2_048, 2_048])
            #expect(layer.state.allSatisfy { $0.asType(.float32).sum().item(Float.self) == Float(2 * 2_048 * 64) })
        }
        #expect(Set(addresses(cache)).isDisjoint(with: Set(oldAddresses)))
        let after = AttentionCapacityCompaction.measure(cache)
        #expect(after.retainedBytes == 8 * 512 * 256)
    }

    @Test func belowTheThresholdNothingChanges() {
        let cache = makeCache(layers: 8, rows: 2_048 + 100)
        for layer in cache { layer.trim(100) }
        let before = AttentionCapacityCompaction.measure(cache)
        #expect(before.retainedBytes < before.thresholdBytes)
        let oldAddresses = addresses(cache)
        let outcome = AttentionCapacityCompaction.compactIfNeeded(cache)
        #expect(outcome.compactedLayers == 0)
        #expect(outcome.freedBytes == 0)
        #expect(addresses(cache) == oldAddresses)
        #expect(cache.allSatisfy { $0.innerState().allSatisfy { $0.dim(2) >= 2_048 + 100 } })
    }

    @Test func thresholdIsTheSmallerOfAQuarterBodyAnd64MB() {
        #expect(AttentionCapacityCompaction.threshold(bodyBytes: 100 * 1_048_576) == 25 * 1_048_576)
        #expect(
            AttentionCapacityCompaction.threshold(bodyBytes: 2_000 * 1_048_576) == 64 * 1_048_576)
        #expect(AttentionCapacityCompaction.maximumThresholdBytes == 64 * 1_048_576)
    }

    @Test func wholeStateAndQuantizedLayersAreUntouched() throws {
        let cache = makeCache(layers: 2, rows: 2_048 + 20_000)
        for layer in cache { layer.trim(20_000) }
        let quantized = try (makeCache(layers: 1, rows: 512)[0] as! KVCacheSimple).toQuantized(
            groupSize: 64, bits: 8)
        let mamba = MambaCache()
        let mixed: [any KVCache] = cache + [quantized, mamba]
        let outcome = AttentionCapacityCompaction.compactIfNeeded(mixed)
        #expect(outcome.compactedLayers == 2)
        #expect(quantized.offset == 512)
    }
}
