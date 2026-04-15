import Foundation
import MLX
import MLXLMCommon

@testable import Tesseract_Agent

/// Shared snapshot factories and reset hooks for prefix-cache test
/// files. Centralizes construction so eviction tests across
/// `PrefixCacheManagerTests`, `EvictionPolicyTests`, and
/// `AlphaTunerTests` produce the same shapes and reset the same global
/// state.
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

    /// Fabricate a `SnapshotStorageRef` for tests that exercise the
    /// storageRef-aware radix / eviction guards without plumbing
    /// through the full `SSDSnapshotStore` write pipeline. The
    /// `bytesOnDisk` value is arbitrary — the guard logic only
    /// branches on `storageRef != nil` and `committed`.
    static func makeStorageRef(
        committed: Bool = true,
        type: HybridCacheSnapshot.CheckpointType = .leaf,
        tokenOffset: Int = 0
    ) -> SnapshotStorageRef {
        SnapshotStorageRef(
            snapshotID: UUID().uuidString,
            partitionDigest: "deadbeef",
            tokenOffset: tokenOffset,
            checkpointType: type,
            bytesOnDisk: 1024,
            lastAccessTime: .now,
            committed: committed
        )
    }

    /// Reset `EvictionPolicy`'s static state to defaults. Tests that
    /// mutate `alpha` or `modelProfile` should call this in a `defer`.
    static func resetPolicyDefaults() {
        EvictionPolicy.alpha = 0.0
        EvictionPolicy.modelProfile = .qwen35_4B_PARO
    }

    /// Build a leaf-only `AlphaTuner.RequestRecord` (no mid-prefill
    /// captures). Used by `AlphaTunerTests` state-machine tests where
    /// the snapshot mix doesn't matter.
    static func makeLeafOnlyRecord(
        partitionKey: CachePartitionKey = CachePartitionKey(
            modelID: "alpha-tuner-test", kvBits: nil, kvGroupSize: 64
        ),
        promptTokens: [Int],
        storedTokens: [Int],
        leafBytes: Int = 4096
    ) -> AlphaTuner.RequestRecord {
        AlphaTuner.RequestRecord(
            partitionKey: partitionKey,
            promptTokens: promptTokens,
            midPrefillSnapshots: [],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: storedTokens,
                bytes: leafBytes
            )
        )
    }
}
