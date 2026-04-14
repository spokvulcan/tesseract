import Foundation
import MLX
import MLXLMCommon

/// Adaptive `EvictionPolicy.alpha` tuner — paper/repo-aligned Marconi
/// retrospective tuning. Tunes once per process and writes the result
/// back to `EvictionPolicy.alpha`. Continuous retuning is out of scope.
///
/// Tie-break rule for the grid search: highest cumulative
/// parent-relative FLOPs saved wins, breaking ties on cached-token
/// count, then on first-in-iteration order under strict `>`.
@MainActor
final class AlphaTuner {

    /// Snapshot of one mid-prefill capture, recorded for replay.
    struct SnapshotMetadata: Sendable, Equatable {
        let offset: Int
        let bytes: Int
        let type: HybridCacheSnapshot.CheckpointType
    }

    /// Leaf store metadata. `nil` on the request record means the leaf
    /// store was skipped (e.g. normalization-trim guard, no reusable
    /// cache, capture failure) — that's still a recordable request.
    struct LeafStore: Sendable, Equatable {
        let storedTokens: [Int]
        let bytes: Int
    }

    /// One request's worth of replay data. Captures everything the cache
    /// did for the request, regardless of which subset of stores
    /// completed: the lookup tokens, every mid-prefill snapshot
    /// (`.system` / `.branchPoint`), and the leaf store if any.
    struct RequestRecord: Sendable, Equatable {
        let partitionKey: CachePartitionKey
        let promptTokens: [Int]
        let midPrefillSnapshots: [SnapshotMetadata]
        let leafStore: LeafStore?
    }

    /// One snapshot from the production cache state at bootstrap-window
    /// start. Used to reseed each replay with the post-first-eviction
    /// state instead of an empty cache.
    struct InventoryEntry: Sendable, Equatable {
        let partitionKey: CachePartitionKey
        let path: [Int]
        let offset: Int
        let bytes: Int
        let type: HybridCacheSnapshot.CheckpointType
        let lastAccessTime: ContinuousClock.Instant
    }

    enum Phase: Equatable, Sendable {
        case waitingForFirstEviction
        case bootstrapping
        case tuned
    }

    /// Number of requests to record post-first-eviction =
    /// `requestsBeforeFirstEviction * bootstrapMultiplier`, then clamped
    /// to `[minimumBootstrapWindow, maximumBootstrapWindow]`.
    ///
    /// The Marconi paper specifies `5–15`; the reference repo uses `5`.
    /// Tesseract is a single-user local agent where one process rarely
    /// sees hundreds of requests, so a 5× window is unreachable in
    /// practice — most sessions exit before the tuner ever transitions
    /// out of `.bootstrapping`. We use `1×` instead, with the floor
    /// `minimumBootstrapWindow = 10` providing a safety net of distinct
    /// hit/miss outcomes.
    ///
    /// `1×` is safe because `TokenRadixTree.collectEligible` protects
    /// `.system` snapshots from utility scoring, so the worst-case
    /// during bootstrap (alpha = 0) is no longer "lose the stable prefix
    /// and stall for minutes" — it's "evict `.leaf`/`.branchPoint`
    /// slightly less optimally than tuned alpha would." Minimizing time
    /// to tuned alpha matters more than tuned-alpha precision: alpha=0.5
    /// vs alpha=0.7 is a small gap, but alpha=0 (untuned) vs
    /// alpha=anything-positive is a large gap.
    static let bootstrapMultiplier = 1

    /// `[0.0, 0.1, ..., 2.0]` — 21 values. Integer-multiply form is
    /// immune to float-accumulation drift.
    static let alphaCandidates: [Double] = (0...20).map { Double($0) * 0.1 }

    /// Without this floor, a workload that triggers eviction on the very
    /// first request would tune against an empty window. Raised to 10 so
    /// the grid search has at least a handful of distinct hit/miss
    /// outcomes to score against, even when the first eviction fires
    /// almost immediately on a tight budget.
    static let minimumBootstrapWindow = 10

    /// Cap on the bootstrap window. Without this, a workload that takes
    /// 200+ requests before its first eviction (very loose budget) would
    /// require 600+ post-bootstrap requests to tune — far longer than a
    /// typical Tesseract session. Capping at 60 keeps the window
    /// reachable in one sitting while preserving enough signal for the
    /// grid search to discriminate between alpha candidates.
    static let maximumBootstrapWindow = 60

    private(set) var phase: Phase = .waitingForFirstEviction
    private(set) var requestsBeforeFirstEviction: Int = 0
    private(set) var bootstrapTarget: Int = 0
    private var bootstrapWindow: [RequestRecord] = []
    private var startingInventory: [InventoryEntry] = []
    private var inventoryCaptureTime: ContinuousClock.Instant?

    var bootstrapWindowCount: Int { bootstrapWindow.count }

    // MARK: - State machine hooks

    /// Called by `PrefixCacheManager.recordRequest` after every request
    /// finishes its lifecycle (post-store, including paths where the
    /// leaf store was skipped). Drives the phase machine.
    func recordRequest(_ record: RequestRecord) {
        switch phase {
        case .waitingForFirstEviction:
            requestsBeforeFirstEviction += 1
        case .bootstrapping:
            bootstrapWindow.append(record)
            if bootstrapWindow.count >= bootstrapTarget {
                runGridSearch()
            }
        case .tuned:
            break
        }
    }

    /// Called by `PrefixCacheManager.evictToFitBudget` after the first
    /// drain-triggering request finishes its stores. The manager passes
    /// the post-request snapshot inventory so each replay starts from
    /// the same cache state the production process keeps using before
    /// the next request arrives.
    func notifyFirstEviction(startingInventory: [InventoryEntry]) {
        guard case .waitingForFirstEviction = phase else { return }
        bootstrapTarget = min(
            max(
                requestsBeforeFirstEviction * Self.bootstrapMultiplier,
                Self.minimumBootstrapWindow
            ),
            Self.maximumBootstrapWindow
        )
        self.startingInventory = startingInventory
        self.inventoryCaptureTime = .now
        phase = .bootstrapping
        Log.agent.info(
            "AlphaTuner: bootstrap started — requestsBeforeFirstEviction="
            + "\(requestsBeforeFirstEviction) windowSize=\(bootstrapTarget) "
            + "inventoryEntries=\(startingInventory.count)"
        )
    }

    // MARK: - Grid search

    /// Sandbox memory budget for one replay pass. Includes inventory
    /// bytes so the seeded cache state isn't immediately drained on
    /// the first replay step. `internal` so tests can replay the
    /// window manually with the same budget the grid search uses.
    var simBudget: Int { simBudget(for: bootstrapWindow) }

    /// Sandbox budget for a specific replay window. Includes the bytes
    /// from both seeded inventory and request-recorded stores.
    func simBudget(for records: [RequestRecord]) -> Int {
        let inventoryBytes = startingInventory.reduce(0) { $0 + $1.bytes }
        let windowSnapshotBytes = records.reduce(0) {
            $0 + ($1.leafStore?.bytes ?? 0) + $1.midPrefillSnapshots.reduce(0) { $0 + $1.bytes }
        }
        return max((inventoryBytes + windowSnapshotBytes) / 2, 1)
    }

    private func runGridSearch() {
        let records = bootstrapWindow
        let budget = simBudget(for: records)

        var bestAlpha: Double = 0.0
        var bestFlops: Double = -.infinity
        var bestHitTokens: Int = -1

        for candidate in Self.alphaCandidates {
            let result = replayWindow(alpha: candidate, simBudget: budget, records: records)
            if result.flopsSaved > bestFlops
                || (result.flopsSaved == bestFlops && result.hitTokens > bestHitTokens)
            {
                bestFlops = result.flopsSaved
                bestHitTokens = result.hitTokens
                bestAlpha = candidate
            }
        }

        // The replay loop above leaves `EvictionPolicy.alpha` set to the
        // last candidate; writing the winner here is the explicit
        // contract. No `defer` restore — that would clobber the value we
        // just chose.
        EvictionPolicy.alpha = bestAlpha
        phase = .tuned
        let windowSize = bootstrapWindow.count
        let inventorySize = startingInventory.count
        bootstrapWindow = []
        startingInventory = []
        inventoryCaptureTime = nil
        Log.agent.info(
            "AlphaTuner: tuned alpha=\(bestAlpha) "
            + "(flopsSaved=\(bestFlops) hitTokens=\(bestHitTokens) "
            + "windowSize=\(windowSize) inventorySize=\(inventorySize))"
        )
    }

    /// Replay the recorded window against a fresh sandboxed
    /// `PrefixCacheManager` (no `AlphaTuner`, so recursion stops here).
    /// The cache is first seeded from `startingInventory` so each
    /// candidate alpha sees the same starting state the production
    /// process actually had at first eviction. `internal` so tests
    /// can compute per-alpha scores and verify the grid search picks
    /// the maximum.
    func replayWindow(
        alpha: Double, simBudget: Int
    ) -> (flopsSaved: Double, hitTokens: Int) {
        replayWindow(alpha: alpha, simBudget: simBudget, records: bootstrapWindow)
    }

    /// Replay an explicit request window against the current seeded
    /// inventory. `internal` so tests can compute the expected winner
    /// on the exact same full record set the production grid search
    /// will evaluate.
    func replayWindow(
        alpha: Double, simBudget: Int, records: [RequestRecord]
    ) -> (flopsSaved: Double, hitTokens: Int) {
        EvictionPolicy.alpha = alpha
        let simCache = PrefixCacheManager(memoryBudgetBytes: simBudget, alphaTuner: nil)

        // Reseed: shift each entry's lastAccessTime by `now - capture`
        // so the relative recency between inventory snapshots is
        // preserved at replay time. Without the shift, every restored
        // snapshot would look uniformly stale and the recency term
        // would lose all signal.
        let now: ContinuousClock.Instant = .now
        let timeShift: Duration = inventoryCaptureTime.map { now - $0 } ?? .zero
        for entry in startingInventory {
            let stub = Self.makeReplaySnapshot(
                tokenOffset: entry.offset, bytes: entry.bytes, type: entry.type
            )
            simCache.restoreSnapshot(
                path: entry.path,
                snapshot: stub,
                partitionKey: entry.partitionKey,
                lastAccessTime: entry.lastAccessTime + timeShift
            )
        }

        var totalFlops: Double = 0
        var totalHitTokens: Int = 0

        for record in records {
            let lookup = simCache.lookup(
                tokens: record.promptTokens, partitionKey: record.partitionKey
            )
            if let snapshot = lookup.snapshot {
                totalFlops += EvictionPolicy.parentRelativeFlops(
                    nodeOffset: snapshot.tokenOffset,
                    parentOffset: 0
                )
                totalHitTokens += snapshot.tokenOffset
            }

            // Reproduce the production stores: mid-prefill captures
            // first (under the prompt path), then the leaf if any
            // (under the post-response path).
            if !record.midPrefillSnapshots.isEmpty {
                let captures = record.midPrefillSnapshots.map { meta in
                    Self.makeReplaySnapshot(
                        tokenOffset: meta.offset, bytes: meta.bytes, type: meta.type
                    )
                }
                simCache.storeSnapshots(
                    promptTokens: record.promptTokens,
                    capturedSnapshots: captures,
                    partitionKey: record.partitionKey
                )
            }
            if let leafStore = record.leafStore {
                let leafSnap = Self.makeReplaySnapshot(
                    tokenOffset: leafStore.storedTokens.count,
                    bytes: leafStore.bytes,
                    type: .leaf
                )
                simCache.storeLeaf(
                    storedTokens: leafStore.storedTokens,
                    leafSnapshot: leafSnap,
                    partitionKey: record.partitionKey
                )
            }
        }

        return (totalFlops, totalHitTokens)
    }

    /// Synthetic snapshot whose `memoryBytes` matches the recorded size.
    /// Eviction only inspects `memoryBytes` — the zero-filled MLX arrays
    /// are never read. Each `KVCacheSimple` snapshot stores two
    /// `[1, 1, length, 64]` float32 arrays, so `length =
    /// bytes / (2 * 64 * 4)`.
    static func makeReplaySnapshot(
        tokenOffset: Int,
        bytes: Int,
        type: HybridCacheSnapshot.CheckpointType
    ) -> HybridCacheSnapshot {
        let length = max(1, bytes / (2 * 64 * 4))
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, length, 64]),
            MLXArray.zeros([1, 1, length, 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: tokenOffset, type: type)!
    }
}
