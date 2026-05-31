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

    /// Fabricate a `SnapshotRef` for tests that exercise the
    /// state-aware radix / eviction logic without plumbing through the
    /// full `SSDSnapshotStore` write pipeline. The `bytesOnDisk` value
    /// is arbitrary. The write phase (pending vs committed) is no longer
    /// a field on the ref — it is the owning `SnapshotState` case, so
    /// callers attach the ref via `node.state = .ssdOnly(ref)` etc.
    static func makeRef(
        type: HybridCacheSnapshot.CheckpointType = .leaf,
        tokenOffset: Int = 0
    ) -> SnapshotRef {
        SnapshotRef(
            snapshotID: UUID().uuidString,
            partitionDigest: "deadbeef",
            tokenOffset: tokenOffset,
            checkpointType: type,
            bytesOnDisk: 1024
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

@MainActor
extension PrefixCacheManager {
    @discardableResult
    func admitCheckpoints(
        promptTokens: [Int],
        capturedSnapshots: [HybridCacheSnapshot],
        storage: SnapshotAdmission.Storage = .ramOnly,
        partitionKey: CachePartitionKey,
        requestID: UUID? = nil
    ) -> StoreDiagnostics {
        guard !capturedSnapshots.isEmpty else {
            return StoreDiagnostics(evictions: [], supersededLeaves: [], stats: stats)
        }

        let candidates = capturedSnapshots.map { snapshot in
            return SnapshotAdmission.CheckpointCandidate(
                snapshot: snapshot,
                storage: storage
            )
        }

        guard let admission = SnapshotAdmission.checkpoints(
            fullPromptTokens: promptTokens,
            candidates: candidates,
            partitionKey: partitionKey,
            requestID: requestID
        ) else {
            preconditionFailure("Invalid checkpoint Snapshot Admission in test fixture")
        }

        return admit(admission)
    }

    @discardableResult
    func admitLeaf(
        storedTokens: [Int],
        leafSnapshot: HybridCacheSnapshot,
        leafPayload: SnapshotPayload? = nil,
        partitionKey: CachePartitionKey,
        requestID: UUID? = nil
    ) -> StoreDiagnostics {
        let storage: SnapshotAdmission.Storage =
            leafPayload.map(SnapshotAdmission.Storage.ramAndSSD) ?? .ramOnly
        guard let admission = SnapshotAdmission.leaf(
            storedTokens: storedTokens,
            snapshot: leafSnapshot,
            storage: storage,
            partitionKey: partitionKey,
            requestID: requestID
        ) else {
            preconditionFailure("Invalid leaf Snapshot Admission in test fixture")
        }

        return admit(admission)
    }
}
