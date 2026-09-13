import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite(.serialized)
struct CacheStateBytesTests {
    @Test func detectsExactBytesIncludingSignedZero() {
        let cache = MambaCache()
        cache.state = [MLXArray([Float(0)])]
        let original = CacheStateBytes([cache])
        #expect(original == CacheStateBytes([cache]))
        cache.state = [MLXArray([-Float.zero])]
        #expect(original != CacheStateBytes([cache]))
    }

    @Test func detectsLayerAndTensorCountsMetadataShapeAndDtype() {
        let cache = MambaCache()
        cache.state = [MLXArray([Float(1), 2])]
        let original = CacheStateBytes([cache])
        #expect(original != CacheStateBytes([]))
        #expect(original != CacheStateBytes([cache, cache]))
        cache.offset = 1
        #expect(original != CacheStateBytes([cache]))
        cache.offset = 0
        cache.state = [MLXArray([Float(1), 2]).reshaped([1, 2])]
        #expect(original != CacheStateBytes([cache]))
        cache.state = [MLXArray([Int32(1), 2])]
        #expect(original != CacheStateBytes([cache]))
        cache.state = [MLXArray([Float(1), 2]), MLXArray([Float(3)])]
        #expect(original != CacheStateBytes([cache]))
    }

    @Test func evidenceRemainsIndependentOfLaterMutation() {
        let cache = MambaCache()
        cache.state = [MLXArray([Float(1), 2])]
        let original = CacheStateBytes([cache])
        let bytes = original.layers[0].tensors[0].bytes
        cache.state = [MLXArray([Float(9), 10])]
        #expect(original.layers[0].tensors[0].bytes == bytes)
        #expect(original != CacheStateBytes([cache]))
    }
}
