import Foundation
import MLX
import MLXLMCommon

@testable import Tesseract_Agent

/// Shared snapshot factories for prefix-cache test files. Centralizes
/// construction so eviction tests across `PrefixCacheManagerTests` and
/// `EvictionPolicyTests` produce the same shapes.
@MainActor
enum PrefixCacheTestFixtures {

    /// Build a `KVCacheSimple`-backed snapshot whose `memoryBytes` does
    /// **not** depend on `offset`. Eviction tests need same-size snapshots
    /// to keep "evict exactly N snapshots" budgets predictable; the
    /// offset-scaled `KVCacheSimple([1, 1, max(offset, 1), 64])` shape used
    /// by other helpers makes the budget math fragile.
    static func makeUniformSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .system,
        length: Int = 16
    ) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, length, 64]),
            MLXArray.zeros([1, 1, length, 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }
}
