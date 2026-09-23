import Foundation
import MLX
import MLXLMCommon

@testable import Tesseract_Agent

/// Poll `condition` on MainActor until true or timeout — the SSD
/// writer's commit/drop callbacks hop back to MainActor asynchronously.
/// Shared by every suite that awaits a writer-side transition.
@MainActor
func waitUntil(
    timeout: Duration = .seconds(5),
    _ condition: @MainActor () -> Bool
) async -> Bool {
    let start = ContinuousClock.now
    while ContinuousClock.now - start < timeout {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}

/// Physical address of an `MLXArray`'s backing buffer. Two arrays that
/// share a Metal allocation report the same address; independent copies
/// report different ones. `asData(access: .noCopy)` (after `eval`) wraps
/// `mlx_array_data_uint8` directly, so the `Data`'s base address is the
/// real backing pointer — not a fresh copy.
///
/// This is the only thing that distinguishes a deep copy from a
/// copy-on-write *alias*: `MLXArray` is a reference type, and an
/// `array[.ellipsis]` slice shares the source's buffer until a mutation
/// forces a copy. A value-isolation test (mutate one, read the other) can
/// **not** catch the alias because COW preserves the un-mutated party's
/// values either way — see MLX's own `testCopyEllipsis`. Only the physical
/// address discriminates. Shared by the buffer-isolation tests in
/// `HybridCacheSnapshotTests` and the pending-payload tests in
/// `LeafExtensionAdmissionTests`.
nonisolated func backingAddress(_ array: MLXArray) -> UInt {
    array.asData(access: .noCopy).data.withUnsafeBytes {
        UInt(bitPattern: $0.baseAddress)
    }
}

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
        tokenOffset: Int = 0,
        bytesOnDisk: Int = 1024
    ) -> SnapshotRef {
        SnapshotRef(
            snapshotID: UUID().uuidString,
            partitionDigest: "deadbeef",
            tokenOffset: tokenOffset,
            checkpointType: type,
            bytesOnDisk: bytesOnDisk
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
        demotionPayloadExtractor: ((HybridCacheSnapshot) -> SnapshotPayload?)? = nil,
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil,
        activityGate: StorageActivityGate? = nil,
        adaptiveWriteEagerness: Bool = false,
        evictionConfig: EvictionConfiguration = EvictionConfiguration()
    ) -> (manager: PrefixCacheManager, store: TieredSnapshotStore, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = TieredSnapshotStore(
            ssdConfig: SSDPrefixCacheConfig(
                enabled: true,
                rootURL: root,
                budgetBytes: ssdBudgetBytes,
                maxPendingBytes: 10_000_000
            ),
            activityGate: activityGate,
            writerDrainPreludeForTesting: writerDrainPreludeForTesting
        )
        let manager = PrefixCacheManager(
            memoryBudgetBytes: ramBudgetBytes,
            evictionConfig: evictionConfig,
            tieredStore: store,
            demotionPayloadExtractor: demotionPayloadExtractor,
            adaptiveWriteEagerness: adaptiveWriteEagerness
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
        requestID: UUID? = nil,
        source: LeafStorePhase.Report.Source? = nil,
        maximumAdvance: Int = .max
    ) -> PrefixCacheManager.StoreDiagnostics {
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: tokens,
                snapshot: makeUniformSnapshot(offset: tokens.count, type: .leaf),
                storage: storage,
                partitionKey: partitionKey,
                requestID: requestID,
                endOfTurn: endOfTurn,
                source: source,
                maximumAdvance: maximumAdvance
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

extension TryEnqueueResult {
    nonisolated var isAccepted: Bool {
        if case .accepted = self { return true }
        return false
    }
}

/// Parks the SSD writer inside one payload's **Deferred Payload
/// Extraction** materialize step until the test releases it — the only
/// place a test can hold a payload *in the writer's hands* rather than
/// merely queued, which is the distinction `pendingPayloadProgress`
/// draws and the bounded pending-payload wait (#523) turns on.
///
/// The gate is a semaphore because `SnapshotPayload.materialize()` is
/// synchronous on the writer's task; the wait is bounded so a regression
/// cannot hang the suite.
nonisolated final class BlockingMaterializer: @unchecked Sendable {
    private let gate = DispatchSemaphore(value: 0)

    /// Let the parked materializer finish. Safe to call before the writer
    /// has reached the gate — the signal is remembered.
    func release() { gate.signal() }

    /// Wrap `inner` so the host copy parks before reading its layers.
    /// Everything `inner` retains — for a full payload, the body's own
    /// arrays — stays retained for the duration, so a **Leaf Checkout**
    /// keeps reporting `pendingFullPayload` while the gate is shut.
    func gating(_ inner: SnapshotPayload) -> SnapshotPayload {
        let gate = self.gate
        return SnapshotPayload(
            tokenOffset: inner.tokenOffset,
            checkpointType: inner.checkpointType,
            extending: inner.extending,
            totalBytes: inner.totalBytes,
            materialize: {
                _ = gate.wait(timeout: .now() + 5)
                return inner.layers
            })
    }

    /// A stand-alone gated payload for store-level tests that need the
    /// writer parked but have no body to keep alive.
    func payload(bytes: Int, tokenOffset: Int = 4_096) -> SnapshotPayload {
        let gate = self.gate
        return SnapshotPayload(
            tokenOffset: tokenOffset,
            checkpointType: .leaf,
            totalBytes: bytes,
            materialize: {
                _ = gate.wait(timeout: .now() + 5)
                return [
                    SnapshotPayload.LayerPayload(
                        className: "KVCache",
                        state: [
                            SnapshotPayload.ArrayPayload(
                                data: Data(repeating: 0xAB, count: bytes),
                                dtype: "bfloat16", shape: [1, bytes])
                        ],
                        metaState: ["meta"],
                        offset: tokenOffset)
                ]
            })
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

extension PrefixCacheManager {
    /// Resolve for a fresh request **Cache Claim** and hand the claim over
    /// unredeemed: it keeps the lane and the Restore Pins resolution added
    /// until `handOver.withClaim { _ in }` concludes it. The claim belongs
    /// to `diagnostics.requestID`.
    func resolveHoldingClaim(
        tokens: [Int],
        partitionKey: CachePartitionKey,
        diagnostics: PrefixCacheDiagnostics.Context,
        modelFingerprint: String? = nil,
        transientBoundary: HybridCacheSnapshot? = nil,
        sessions: any ModelSessionProviding = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [1, 2])),
        tripwire: CacheClaim.Tripwire = .standard
    ) async -> (resolved: Resolved, handOver: CacheClaim.HandOver) {
        let (resolved, handOver) = await CacheClaim.withRequestClaim(
            context: diagnostics, prefixCache: self, sessions: sessions, memory: nil,
            tripwire: tripwire
        ) { claim in
            await resolve(
                tokens: tokens, promptTokenCount: tokens.count, partitionKey: partitionKey,
                modelFingerprint: modelFingerprint, diagnostics: diagnostics,
                transientBoundary: transientBoundary, for: claim)
        }
        return (resolved, handOver)
    }
}

extension PrefixCacheManager {
    /// Check `resolved` out for a fresh request **Cache Claim** inside the
    /// toy Model Session, and hand the claim over unredeemed with whatever
    /// it holds: a handoff keeps its lease until the claim rewinds or checks
    /// in.
    func checkOutHoldingClaim(
        _ resolved: Resolved,
        tokens: [Int],
        maximumAdvance: Int = 10,
        identityKeySpace: Bool = true,
        diagnostics: PrefixCacheDiagnostics.Context,
        sessions: any ModelSessionProviding = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [1, 2])),
        tripwire: CacheClaim.Tripwire = .standard
    ) async -> (outcome: CacheClaim.RestoreOutcome, handOver: CacheClaim.HandOver) {
        let (outcome, handOver) = await CacheClaim.withRequestClaim(
            context: diagnostics, prefixCache: self, sessions: sessions, memory: nil,
            tripwire: tripwire
        ) { claim in
            await sessions.withSession { session in
                await claim.checkOut(
                    resolved, tokens: tokens, maximumAdvance: maximumAdvance,
                    identityKeySpace: identityKeySpace, in: session)
            }
        }
        return (outcome, handOver)
    }
}

extension CacheClaim.HandOver {
    /// Redeem the hand-over, rewind whatever the claim still leases in the
    /// toy Model Session, and conclude: the rewound leaf's offset, `nil`
    /// when nothing was leased.
    @discardableResult
    func rewindAndConclude(
        sessions: any ModelSessionProviding = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [1, 2]))
    ) async -> Int? {
        await withClaim { claim in
            await sessions.withSession { session in await claim.rewind(in: session) }
        }
    }
}
