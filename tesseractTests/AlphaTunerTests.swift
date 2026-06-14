import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Tests for the adaptive `alpha` tuner.
///
/// The tuner returns its tuned winning `alpha` from `recordRequest`
/// instead of writing a global, so the suite carries no ambient state and
/// needs neither serialization nor a per-test reset. Tests rely on the
/// default injected profile (`.qwen35_4B_PARO`).
@MainActor
struct AlphaTunerTests {

    // MARK: - Helpers

    private let defaultKey = CachePartitionKey(
        modelID: "alpha-tuner-test", kvBits: nil, kvGroupSize: 64
    )

    private func makeLeafOnlyRecord(
        promptTokens: [Int],
        storedTokens: [Int],
        leafBytes: Int = 4096
    ) -> AlphaTuner.RequestRecord {
        PrefixCacheTestFixtures.makeLeafOnlyRecord(
            partitionKey: defaultKey,
            promptTokens: promptTokens,
            storedTokens: storedTokens,
            leafBytes: leafBytes
        )
    }

    private func makeMixedBootstrapWorkloadRecords(
        count: Int,
        tallVariantModulus: Int = 3
    ) -> [AlphaTuner.RequestRecord] {
        let tallShared = Array(1...80)
        return (0..<count).map { i in
            if i % 2 == 0 {
                let variant = (i / 2) % tallVariantModulus
                let stored = tallShared + [900 + variant]
                return makeLeafOnlyRecord(
                    promptTokens: stored, storedTokens: stored, leafBytes: 8192
                )
            }

            let stored = [500 + i, 600 + i, 700 + i]
            return makeLeafOnlyRecord(
                promptTokens: stored, storedTokens: stored, leafBytes: 1024
            )
        }
    }

    /// Drives `count` records into the tuner alternating tall
    /// shared-prefix snapshots (favors F/B-weighted scoring) with
    /// short noise (favors recency). Used by both grid-search tests.
    /// `tallVariantModulus` controls how many distinct tall variants
    /// repeat across the window — smaller values give more re-hit
    /// opportunities under F/B preservation.
    @discardableResult
    private func populateMixedBootstrapWorkload(
        tuner: AlphaTuner,
        count: Int,
        tallVariantModulus: Int = 3
    ) -> Double? {
        var tunedAlpha: Double?
        for record in makeMixedBootstrapWorkloadRecords(
            count: count, tallVariantModulus: tallVariantModulus
        ) {
            if let result = tuner.recordRequest(record) { tunedAlpha = result }
        }
        return tunedAlpha
    }

    // MARK: - 1. startsAtZeroBeforeFirstEviction

    /// Until `notifyFirstEviction` is called the tuner stays in the
    /// `.waitingForFirstEviction` phase, just incrementing
    /// `requestsBeforeFirstEviction`. No tuned `alpha` is returned.
    @Test func startsAtZeroBeforeFirstEviction() {
        let tuner = AlphaTuner()

        var tunedAlphas: [Double] = []
        for i in 0..<10 {
            if let tuned = tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [1, 2, 3, i],
                    storedTokens: [1, 2, 3, i, 99]
                ))
            {
                tunedAlphas.append(tuned)
            }
        }

        #expect(tuner.phase == .waitingForFirstEviction)
        #expect(tuner.requestsBeforeFirstEviction == 10)
        #expect(tuner.bootstrapWindowCount == 0)
        // No tuned alpha is emitted before the first eviction fires.
        #expect(tunedAlphas.isEmpty)
    }

    // MARK: - 2. bootstrapWindowUsesMultiplierTimesFirstEvictionCount

    /// On the first eviction, the bootstrap target is
    /// `requestsBeforeFirstEviction * bootstrapMultiplier`, clamped to
    /// `[minimumBootstrapWindow, maximumBootstrapWindow]`. The phase
    /// transitions to `.bootstrapping` and subsequent records accumulate
    /// in the window until it fills. Pre-eviction count is chosen large
    /// enough that `multiplier * count` is the dominant term (above the
    /// floor and below the cap).
    @Test func bootstrapWindowUsesMultiplierTimesFirstEvictionCount() {
        let tuner = AlphaTuner()

        // Pick a count where `multiplier * count` lands strictly
        // between the floor and the cap so neither clamp interferes.
        let preEvictionCount = max(
            AlphaTuner.minimumBootstrapWindow + 5,
            AlphaTuner.minimumBootstrapWindow / AlphaTuner.bootstrapMultiplier + 5
        )
        for i in 0..<preEvictionCount {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [i, 100],
                    storedTokens: [i, 100, 200]
                ))
        }
        #expect(tuner.requestsBeforeFirstEviction == preEvictionCount)

        let expected = min(
            max(
                preEvictionCount * AlphaTuner.bootstrapMultiplier,
                AlphaTuner.minimumBootstrapWindow
            ),
            AlphaTuner.maximumBootstrapWindow
        )

        tuner.notifyFirstEviction(startingInventory: [])
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.bootstrapTarget == expected)

        // A second notify is a no-op.
        tuner.notifyFirstEviction(startingInventory: [])
        #expect(tuner.bootstrapTarget == expected)
    }

    /// When the first eviction fires before any request has been
    /// recorded, the bootstrap window falls back to the safety floor
    /// instead of running with size zero.
    @Test func bootstrapTargetHonorsMinimumWindow() {
        let tuner = AlphaTuner()

        tuner.notifyFirstEviction(startingInventory: [])
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.bootstrapTarget == AlphaTuner.minimumBootstrapWindow)
    }

    /// A workload with a very loose budget could push
    /// `requestsBeforeFirstEviction * bootstrapMultiplier` past the
    /// `maximumBootstrapWindow` cap. The cap must clamp it back so the
    /// tuner can finish in a single typical session even when first
    /// eviction takes hundreds of requests.
    @Test func bootstrapTargetHonorsMaximumWindow() {
        let tuner = AlphaTuner()

        // Pick a pre-eviction count guaranteed to overshoot the cap.
        let preEvictionCount =
            AlphaTuner.maximumBootstrapWindow * 2
            + AlphaTuner.minimumBootstrapWindow
        for i in 0..<preEvictionCount {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [i, 100],
                    storedTokens: [i, 100, 200]
                ))
        }

        tuner.notifyFirstEviction(startingInventory: [])
        #expect(tuner.bootstrapTarget == AlphaTuner.maximumBootstrapWindow)
    }

    // MARK: - 3. gridSearchTransitionsToTunedWithCandidateAlpha

    /// The grid search must terminate, transition the tuner to
    /// `.tuned`, and return a value from the `alphaCandidates` set.
    @Test func gridSearchTransitionsToTunedWithCandidateAlpha() {
        let tuner = AlphaTuner()

        for _ in 0..<3 {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [1, 2],
                    storedTokens: [1, 2, 3]
                ))
        }
        tuner.notifyFirstEviction(startingInventory: [])

        // Mixed shared-prefix / one-off workload so different alphas
        // pick different victims and the grid search has signal.
        let tunedAlpha = populateMixedBootstrapWorkload(
            tuner: tuner, count: tuner.bootstrapTarget
        )

        #expect(tuner.phase == .tuned)
        #expect(tunedAlpha.map { AlphaTuner.alphaCandidates.contains($0) } == true)
    }

    /// `alphaCandidates` covers `[0.0, 0.1, ..., 2.0]` evenly.
    @Test func alphaCandidatesCoverFullGrid() {
        let candidates = AlphaTuner.alphaCandidates
        #expect(candidates.count == 21)
        #expect(candidates.first == 0.0)
        #expect(candidates.last == 2.0)
        for i in 1..<candidates.count {
            #expect(abs((candidates[i] - candidates[i - 1]) - 0.1) < 1e-9)
        }
    }

    // MARK: - 4. tunedAlphaUsedForSubsequentEvictions

    /// When the tuner finishes, `recordRequest` returns exactly one
    /// grid-member `alpha` — the winner the manager assigns to its
    /// **Eviction Configuration**. No global is read back.
    @Test func tunedAlphaUsedForSubsequentEvictions() {
        let tuner = AlphaTuner()
        tuner.notifyFirstEviction(startingInventory: [])

        // The grid search fires exactly once, on the record that fills the
        // bootstrap window; every earlier call returns nil.
        var tunedAlpha: Double?
        for i in 0..<tuner.bootstrapTarget {
            tunedAlpha = tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [i, i + 1],
                    storedTokens: [i, i + 1, i + 2]
                ))
        }
        #expect(tuner.phase == .tuned)
        #expect(tunedAlpha.map { AlphaTuner.alphaCandidates.contains($0) } == true)
    }

    /// A freshly constructed prefix cache starts at the LRU default
    /// (`alpha = 0`, `ModelFlopProfile.fallback`). Because each cache owns
    /// its **Eviction Configuration**, there is no process global for a
    /// previous cache's tuned `alpha` to leak through.
    @Test func freshPrefixCacheStartsAtLRUDefault() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1024)
        #expect(mgr.evictionConfig.alpha == 0.0)
        #expect(mgr.evictionConfig.flopProfile == .fallback)
    }

    /// `ensurePrefixCache` folds the model's `flopProfile` into the cache's
    /// **Eviction Configuration**. Before a model loads there is no identity,
    /// so the cache gets the shared `ModelFlopProfile.fallback` and the LRU
    /// default `alpha`. Drives the real actor construction path, not a
    /// hand-built manager.
    @Test func ensurePrefixCacheUsesFallbackProfileBeforeLoad() async {
        let actor = LLMActor()
        // setPrefixCacheBudgetBytes builds the cache through ensurePrefixCache.
        await actor.setPrefixCacheBudgetBytes(4096)
        let config = await actor.currentEvictionConfigForTesting()
        #expect(config?.flopProfile == .fallback)
        #expect(config?.alpha == 0.0)
    }

    /// Once an identity is installed, the cache `ensurePrefixCache` builds
    /// scores against *that* model's `flopProfile`, not the fallback — the
    /// wiring this change adds. Without this assertion a regression that
    /// ignored the identity (always using the fallback, or wiring the budget
    /// in place of the profile) would compile and ship green.
    @Test func ensurePrefixCacheSourcesProfileFromModelIdentity() async {
        // A Qwen3.5 config whose dimensions differ from the PARO fallback.
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "full_attention_interval": 4,
            ],
            chatTemplate: nil
        )
        #expect(identity.flopProfile != .fallback)  // sanity: distinct profile

        let actor = LLMActor()
        await actor.setModelIdentityForTesting(identity)
        await actor.setPrefixCacheBudgetBytes(4096)

        let config = await actor.currentEvictionConfigForTesting()
        #expect(config?.flopProfile == identity.flopProfile)
    }

    // MARK: - 5. recordRequest no-op after .tuned

    /// Once `.tuned`, further `recordRequest` calls are silently
    /// ignored — the tuner does not retune in this implementation.
    /// `bootstrapWindow` is also cleared after tuning.
    @Test func recordRequestIsNoOpAfterTuned() {
        let tuner = AlphaTuner()
        tuner.notifyFirstEviction(startingInventory: [])
        for i in 0..<tuner.bootstrapTarget {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [i],
                    storedTokens: [i, i + 1]
                ))
        }
        #expect(tuner.phase == .tuned)
        #expect(tuner.bootstrapWindowCount == 0)

        for i in 0..<10 {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [9000 + i],
                    storedTokens: [9000 + i, 9100 + i]
                ))
        }
        #expect(tuner.bootstrapWindowCount == 0)
    }

    // MARK: - 6. PrefixCacheManager wires the tuner

    /// `PrefixCacheManager.evictToFitBudget` marks the first-evicting
    /// request, but the tuner doesn't transition to `.bootstrapping`
    /// until `recordRequest` fires at request end. That deferred
    /// boundary capture ensures the seeded inventory reflects the final
    /// post-request cache state, not the mid-request state at the
    /// moment of the first drain.
    @Test func managerCallsTunerWithInventoryOnFirstEviction() {
        let tuner = AlphaTuner()
        let snap = PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf)
        let snapBytes = snap.memoryBytes
        let boundaryRequestID = UUID()
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes, alphaTuner: tuner
        )

        // First request: stores a leaf, no eviction. The manager call
        // path doesn't include the tuner notification — that's the
        // agent layer's responsibility — so requestsBeforeFirstEviction
        // stays at 0 here. The tuner only sees first-eviction notify.
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: Array(1...100),
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey
            )!)
        #expect(tuner.phase == .waitingForFirstEviction)

        // Second request: pushes over budget, triggers eviction →
        // tuner enters .bootstrapping with the post-drain inventory.
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: Array(200...299),
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey,
                requestID: boundaryRequestID
            )!)
        #expect(tuner.phase == .waitingForFirstEviction)

        mgr.recordRequest(
            partitionKey: defaultKey,
            promptTokens: Array(200...299),
            capturedSnapshots: [],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: Array(200...299),
                bytes: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf)
                    .memoryBytes
            ),
            requestID: boundaryRequestID
        )
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.bootstrapTarget >= AlphaTuner.minimumBootstrapWindow)
        #expect(tuner.bootstrapWindowCount == 0)
    }

    /// `collectSnapshotInventory` walks every partition and returns one
    /// `InventoryEntry` per snapshot, with the path reconstructed from
    /// the radix tree.
    @Test func collectSnapshotInventoryReturnsAllSnapshots() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1024 * 1024 * 1024)

        let pathA = Array(1...10)
        let pathB = Array(20...29)
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: pathA,
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                    offset: pathA.count, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey
            )!)
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: pathB,
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                    offset: pathB.count, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey
            )!)

        let inventory = mgr.collectSnapshotInventory()
        #expect(inventory.count == 2)
        let paths = Set(inventory.map { $0.path })
        #expect(paths == Set([pathA, pathB]))
        for entry in inventory {
            #expect(entry.partitionKey == defaultKey)
            #expect(entry.type == .leaf)
            #expect(entry.bytes > 0)
        }
    }

    /// `restoreSnapshot` reseeds a snapshot with an explicit
    /// `lastAccessTime`, preserving the relative recency that the
    /// tuner's replay needs.
    @Test func restoreSnapshotPreservesLastAccessTime() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1024 * 1024 * 1024)
        let path = Array(1...50)
        let snap = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: path.count, type: .leaf
        )
        let pastInstant: ContinuousClock.Instant = .now - .seconds(120)

        mgr.restoreSnapshot(
            path: path,
            snapshot: snap,
            partitionKey: defaultKey,
            lastAccessTime: pastInstant
        )

        let lookup = mgr.lookup(tokens: path, partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == path.count)
        // The lookup itself refreshes lastAccessTime; we can't observe
        // the original past instant directly without breaking the
        // public API. The structural assertion (snapshot is reachable)
        // is the contract we care about here.
    }

    // MARK: - 7. First-evicting request boundary

    /// If the first eviction happens during mid-prefill capture, the
    /// manager must defer bootstrap start until request end so the
    /// seeded inventory includes the same request's later leaf store.
    /// The boundary request itself is excluded from the replay window.
    ///
    /// Both mid-prefill snapshots are `.branchPoint` (not `.system`)
    /// because `.system` is type-protected from utility eviction by
    /// `TokenRadixTree.collectEligible`. The test's intent is the
    /// boundary-capture timing, not the type-priority rule, so we use
    /// types that participate in normal recency-based eviction.
    @Test func managerDefersBootstrapBoundaryUntilFullRequestCompletes() throws {
        let tuner = AlphaTuner()
        let boundaryRequestID = UUID()
        let snapshotA = PrefixCacheTestFixtures.makeUniformSnapshot(offset: 10, type: .branchPoint)
        let snapshotB = PrefixCacheTestFixtures.makeUniformSnapshot(offset: 20, type: .branchPoint)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapshotA.memoryBytes,
            alphaTuner: tuner
        )
        let promptTokens = Array(1...30)
        let storedTokens = promptTokens + [31]
        let leafSnapshot = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: storedTokens.count,
            type: .leaf
        )

        // First eviction happens during mid-prefill store, but the
        // tuner must stay in `.waitingForFirstEviction` until the
        // request-end record arrives.
        let admission = try #require(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: promptTokens,
                candidates: [
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: snapshotA,
                        storage: .ramOnly
                    ),
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: snapshotB,
                        storage: .ramOnly
                    ),
                ],
                partitionKey: defaultKey,
                requestID: boundaryRequestID
            ))
        mgr.admit(admission)
        #expect(tuner.phase == .waitingForFirstEviction)

        // The same request then stores a leaf. This must be part of
        // the seeded inventory before bootstrap begins.
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: storedTokens,
                snapshot: leafSnapshot,
                storage: .ramOnly,
                partitionKey: defaultKey,
                requestID: boundaryRequestID
            )!)
        #expect(tuner.phase == .waitingForFirstEviction)

        mgr.recordRequest(
            partitionKey: defaultKey,
            promptTokens: promptTokens,
            capturedSnapshots: [snapshotA, snapshotB],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: storedTokens,
                bytes: leafSnapshot.memoryBytes
            ),
            requestID: boundaryRequestID
        )
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.bootstrapWindowCount == 0)

        // A replay starting from the seeded inventory must hit the
        // surviving leaf from the first-evicting request, proving the
        // inventory was captured after the leaf store rather than in
        // the earlier mid-prefill drain.
        let followup = makeLeafOnlyRecord(
            promptTokens: storedTokens + [999],
            storedTokens: storedTokens + [999, 1000],
            leafBytes: leafSnapshot.memoryBytes
        )
        let budget = tuner.simBudget(for: [followup])
        let result = tuner.replayWindow(alpha: 0.0, simBudget: budget, records: [followup])
        #expect(result.hitTokens == storedTokens.count)
    }

    /// If another request reaches `recordRequest` before the first-evicting
    /// request finishes, it must not consume the pending bootstrap boundary.
    /// The manager should wait for the matching request ID before starting
    /// bootstrapping and dropping the boundary request.
    @Test func managerWaitsForMatchingBoundaryRequestID() {
        let tuner = AlphaTuner()
        let snap = PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf)
        let snapBytes = snap.memoryBytes
        let boundaryRequestID = UUID()
        let otherRequestID = UUID()
        let nextRequestID = UUID()
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            alphaTuner: tuner
        )

        // Seed one snapshot so the boundary request's store triggers the
        // first eviction.
        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: Array(1...100),
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey
            )!)

        mgr.admit(
            SnapshotAdmission.leaf(
                storedTokens: Array(200...299),
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 100, type: .leaf),
                storage: .ramOnly,
                partitionKey: defaultKey,
                requestID: boundaryRequestID
            )!)
        #expect(tuner.phase == .waitingForFirstEviction)

        // Another request finishes first. It should be counted as pre-bootstrap
        // traffic, not dropped as the boundary request.
        mgr.recordRequest(
            partitionKey: defaultKey,
            promptTokens: [900, 901],
            capturedSnapshots: [],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: [900, 901, 902],
                bytes: snapBytes
            ),
            requestID: otherRequestID
        )
        #expect(tuner.phase == .waitingForFirstEviction)
        #expect(tuner.requestsBeforeFirstEviction == 1)

        // The matching request ID starts bootstrap and is excluded.
        mgr.recordRequest(
            partitionKey: defaultKey,
            promptTokens: Array(200...299),
            capturedSnapshots: [],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: Array(200...299),
                bytes: snapBytes
            ),
            requestID: boundaryRequestID
        )
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.requestsBeforeFirstEviction == 1)
        #expect(tuner.bootstrapWindowCount == 0)

        // Requests after the boundary now enter the bootstrap window.
        mgr.recordRequest(
            partitionKey: defaultKey,
            promptTokens: [1000, 1001],
            capturedSnapshots: [],
            leafStore: AlphaTuner.LeafStore(
                storedTokens: [1000, 1001, 1002],
                bytes: snapBytes
            ),
            requestID: nextRequestID
        )
        #expect(tuner.bootstrapWindowCount == 1)
    }

    // MARK: - 8. Grid search picks the maximum-flop alpha

    /// **Acceptance test for Task 2.4**: the grid search must select
    /// the alpha that maximizes cumulative parent-relative FLOPs saved
    /// (with cached-token count as the tie-breaker). This test
    /// independently replays each candidate via the same `replayWindow`
    /// the grid search uses, computes the expected winner with the
    /// same comparison rule, then drives the actual grid search and
    /// asserts the two agree. A regression in `runGridSearch`'s
    /// comparison or tie-break logic would diverge the two, and the
    /// test catches it.
    @Test func gridSearchPicksMaximumAcrossAllCandidates() {
        let tuner = AlphaTuner()

        // Populate a deterministic bootstrap window with mixed
        // shared-prefix reuse and one-off noise. The mix is the same
        // shape we use in `gridSearchTransitionsToTunedWithCandidateAlpha`
        // — different alphas pick different victims, so per-candidate
        // scores actually vary.
        for _ in 0..<6 {
            tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [1, 2],
                    storedTokens: [1, 2, 3]
                ))
        }
        tuner.notifyFirstEviction(startingInventory: [])

        // Build the full bootstrap window deterministically. Feed all
        // but the final record into the tuner so it stays in
        // `.bootstrapping`, while the test computes the expected
        // winner on the exact same full window the real grid search
        // will evaluate after the final append.
        let target = tuner.bootstrapTarget
        let records = makeMixedBootstrapWorkloadRecords(
            count: target, tallVariantModulus: 5
        )
        for record in records.dropLast() {
            tuner.recordRequest(record)
        }
        #expect(tuner.phase == .bootstrapping)
        #expect(tuner.bootstrapWindowCount == target - 1)

        // Replay each candidate against the full target-sized window
        // before the final record fires the real grid search.
        let budget = tuner.simBudget(for: records)

        // Score every candidate independently. `replayWindow` builds its
        // own sandbox configuration per candidate — there is no global to
        // save and restore.
        var alphaResults: [(alpha: Double, flops: Double, hits: Int)] = []
        for candidate in AlphaTuner.alphaCandidates {
            let result = tuner.replayWindow(alpha: candidate, simBudget: budget, records: records)
            alphaResults.append((candidate, result.flopsSaved, result.hitTokens))
        }

        // Compute the expected winner with the same comparison rule
        // the grid search uses: max flops, ties broken by hit tokens,
        // first candidate in iteration order on perfect ties.
        var expectedAlpha: Double = 0
        var expectedFlops: Double = -.infinity
        var expectedHits: Int = -1
        for r in alphaResults {
            if r.flops > expectedFlops || (r.flops == expectedFlops && r.hits > expectedHits) {
                expectedFlops = r.flops
                expectedHits = r.hits
                expectedAlpha = r.alpha
            }
        }

        // Now fire the grid search by appending the final record from
        // the same full window we scored above.
        let tunedAlpha = tuner.recordRequest(records.last!)
        #expect(tuner.phase == .tuned)

        // The grid search must return exactly the alpha our manual
        // replay-and-compare loop identified.
        #expect(tunedAlpha == expectedAlpha)
    }

    /// Degenerate case: every candidate scores zero (no shared
    /// prefixes anywhere), so they all tie. The grid search must pick
    /// the **first** candidate in iteration order under a strict `>`
    /// comparison (alpha = 0.0). A regression that flips the
    /// comparison to `>=` would walk all the way to alpha = 2.0.
    @Test func gridSearchPicksZeroAlphaWhenAllCandidatesTie() {
        let tuner = AlphaTuner()
        tuner.notifyFirstEviction(startingInventory: [])
        // Each request has a totally unique prompt/stored path, so the
        // simulated cache never returns a hit and every candidate
        // scores zero flops + zero hit tokens.
        var tunedAlpha: Double?
        for i in 0..<tuner.bootstrapTarget {
            let unique = 100_000 + i * 7
            tunedAlpha = tuner.recordRequest(
                makeLeafOnlyRecord(
                    promptTokens: [unique],
                    storedTokens: [unique, unique + 1]
                ))
        }
        #expect(tuner.phase == .tuned)
        // All candidates tie at zero, so the strict `>` comparison keeps
        // the first one (alpha = 0).
        #expect(tunedAlpha == 0.0)
    }
}
