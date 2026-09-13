import Foundation
import MLX
import MLXLMCommon

/// Exact, host-owned evidence for the opt-in correctness harness. Never used
/// by production telemetry: reading these bytes evaluates and copies arrays.
nonisolated struct CacheStateBytes: Equatable, Sendable {
    struct Tensor: Equatable, Sendable {
        let shape: [Int]
        let dtype: String
        let bytes: Data
    }

    struct Layer: Equatable, Sendable {
        let kind: String
        let offset: Int
        let metadata: [String]
        let tensors: [Tensor]
    }

    let layers: [Layer]
    var byteCount: Int { layers.reduce(0) { $0 + $1.tensors.reduce(0) { $0 + $1.bytes.count } } }

    init(_ cache: [any KVCache]) {
        layers = cache.map { layer in
            Layer(
                kind: HybridCacheSnapshot.classNameForCache(layer)
                    ?? String(reflecting: type(of: layer)),
                offset: layer.offset, metadata: layer.metaState,
                tensors: layer.state.map {
                    Tensor(
                        shape: $0.shape, dtype: String(describing: $0.dtype),
                        bytes: $0.asData(access: .copy).data)
                })
        }
    }
}
