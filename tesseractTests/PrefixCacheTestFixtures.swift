import Foundation
import MLX
import MLXLMCommon

@testable import Tesseract_Agent

/// Shared snapshot factories for prefix-cache test files. Centralizes
/// construction so eviction tests across `PrefixCacheManagerTests`,
/// `EvictionPolicyTests`, and `AlphaTunerTests` produce the same shapes.
/// There is no shared global state to reset — eviction inputs travel as
/// **Eviction Configuration** values passed by each test.
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

    /// Single-layer leaf payload whose one KV array carries `bytes` raw
    /// bytes; `extending` marks it as a suffix segment past the base.
    /// For tests that only need the payload's byte accounting, never
    /// its tensor content. `nonisolated` — pure value construction,
    /// callable from the nonisolated store-level suites.
    nonisolated static func makeLeafPayload(
        bytes: Int,
        tokenOffset: Int = 10,
        extending: SnapshotExtension? = nil
    ) -> SnapshotPayload {
        SnapshotPayload(
            tokenOffset: tokenOffset,
            checkpointType: .leaf,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: bytes),
                            dtype: "bfloat16",
                            shape: [1, bytes]
                        )
                    ],
                    metaState: ["meta"],
                    offset: tokenOffset,
                    suffixBaseOffset: extending?.baseOffset
                )
            ],
            extending: extending
        )
    }

    /// Scratch-rooted SSD-enabled `TieredSnapshotStore` +
    /// `PrefixCacheManager` pair. The caller owns the returned root and
    /// should `defer`-delete it; partitions are registered by the
    /// caller (each suite uses its own key).
    static func makeSSDBackedManager(
        label: String,
        ramBudgetBytes: Int,
        ssdBudgetBytes: Int = 10_000_000,
        demotionPayloadExtractor: ((HybridCacheSnapshot) -> SnapshotPayload?)? = nil
    ) -> (manager: PrefixCacheManager, store: TieredSnapshotStore, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = TieredSnapshotStore(ssdConfig: SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: ssdBudgetBytes,
            maxPendingBytes: 10_000_000
        ))
        let manager = PrefixCacheManager(
            memoryBudgetBytes: ramBudgetBytes,
            tieredStore: store,
            demotionPayloadExtractor: demotionPayloadExtractor
        )
        return (manager, store, root)
    }

    /// Admit a uniform-size RAM leaf at `tokens`. The shared shorthand
    /// for "one conversation turn landed" across the eviction /
    /// demotion / pressure / counters suites.
    @discardableResult
    static func admitUniformLeaf(
        _ manager: PrefixCacheManager,
        tokens: [Int],
        partitionKey: CachePartitionKey,
        storage: SnapshotAdmission.Storage = .ramOnly,
        endOfTurn: Bool = true,
        requestID: UUID? = nil
    ) -> PrefixCacheManager.StoreDiagnostics {
        manager.admit(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: makeUniformSnapshot(offset: tokens.count, type: .leaf),
            storage: storage,
            partitionKey: partitionKey,
            requestID: requestID,
            endOfTurn: endOfTurn
        )!)
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

/// Test-only gate for pausing `SSDSnapshotStore`'s detached writer (via
/// `writerDrainPreludeForTesting`) until a test has finished building
/// the pending-queue state it wants to assert against.
actor DrainGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen {
            return
        }

        await withCheckedContinuation { continuation in
            if isOpen {
                continuation.resume()
                return
            }
            waiters.append(continuation)
        }
    }

    func open() {
        if isOpen {
            return
        }
        isOpen = true
        let currentWaiters = waiters
        waiters.removeAll()

        currentWaiters.forEach { $0.resume() }
    }
}

extension SnapshotAdmission.CheckpointCandidate {
    static func ramOnly(
        _ snapshot: HybridCacheSnapshot
    ) -> SnapshotAdmission.CheckpointCandidate {
        SnapshotAdmission.CheckpointCandidate(
            snapshot: snapshot,
            storage: .ramOnly
        )
    }

    static func ramAndSSD(
        _ snapshot: HybridCacheSnapshot,
        payload: SnapshotPayload
    ) -> SnapshotAdmission.CheckpointCandidate {
        SnapshotAdmission.CheckpointCandidate(
            snapshot: snapshot,
            storage: .ramAndSSD(payload)
        )
    }
}
