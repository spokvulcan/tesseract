import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// The pure decision table behind **Snapshot Resolution** (issue #400):
/// `SnapshotResolutionLadder` maps each step's facts to a decision value with no
/// manager, tree, ledger, or **Snapshot Hydrating** handle in sight. Where
/// `SnapshotResolutionTests` proves the composition *through* `resolve()` (the
/// behavior-preservation net), these assert the ladder's rows directly — every
/// gate outcome, every post-hydration fork, and the `Resolved` value shape each
/// decision resolves to.
///
/// `@MainActor` only because the `HybridCacheSnapshot` fixtures touch MLX; the
/// ladder itself is `nonisolated` and every function under test is a pure static.
@MainActor
@Suite struct SnapshotResolutionLadderTests {

    private let key = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)

    private func makeSnapshot(offset: Int, type: HybridCacheSnapshot.CheckpointType = .leaf)
        -> HybridCacheSnapshot
    {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    /// An `initial` lookup carrying a distinctive `sharedPrefixLength` and
    /// `divergence` so the builders' carry-through is observable. The two hit
    /// builders read only those two fields off `initial` — never its reason —
    /// so a plain miss reason keeps the fixture free of node/context plumbing.
    private func initialLookup(sharedPrefixLength: Int = 12) -> PrefixCacheManager.LookupResult {
        PrefixCacheManager.LookupResult(
            snapshot: nil,
            partitionKey: key,
            snapshotTokenOffset: 8,
            sharedPrefixLength: sharedPrefixLength,
            reason: .missNoSnapshotInPrefix,
            divergence: PrefixDivergenceProbe(offset: 3, deepestAbandonedOffset: 10)
        )
    }

    // MARK: - Step A: the Hydration Gate

    @Test func gateAdmitsHydrationRegardlessOfAnyResidentAlternative() {
        let withAlt = SnapshotResolutionLadder.gateOutcome(
            admitsHydration: true, hasAlternativeBody: true
        )
        let withoutAlt = SnapshotResolutionLadder.gateOutcome(
            admitsHydration: true, hasAlternativeBody: false
        )
        #expect(withAlt == .hydrate)
        #expect(withoutAlt == .hydrate)
    }

    @Test func gateServesTheAlternativeWhenRecomputeIsCheaperAndABodyIsResident() {
        let outcome = SnapshotResolutionLadder.gateOutcome(
            admitsHydration: false, hasAlternativeBody: true
        )
        #expect(outcome == .serveAlternative)
    }

    @Test func gateMissesWhenRecomputeIsCheaperAndNothingIsResident() {
        let outcome = SnapshotResolutionLadder.gateOutcome(
            admitsHydration: false, hasAlternativeBody: false
        )
        #expect(outcome == .fallbackMiss)
    }

    // MARK: - Step B: the post-hydration decision

    @Test func aMaterializedBodyIsAlwaysAHydratedHit() {
        // Success wins over the interruption flag and is kind-agnostic — a body
        // in hand is a hit, no clear.
        let ssd = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: true, interrupted: false, kind: .ssd
        )
        let ssdEvenIfInterruptFlagSet = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: true, interrupted: true, kind: .ssd
        )
        let chain = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: true, interrupted: false, kind: .chainPrefix
        )
        #expect(ssd == .hydratedHit)
        #expect(ssdEvenIfInterruptFlagSet == .hydratedHit)
        #expect(chain == .hydratedHit)
    }

    @Test func anInterruptedReadIsACleanMissThatClearsNothing() {
        // Interrupted (PRD #149 item 7) — the same decision for both kinds,
        // because the backing must stay intact either way.
        let ssd = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: false, interrupted: true, kind: .ssd
        )
        let chain = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: false, interrupted: true, kind: .chainPrefix
        )
        #expect(ssd == .interruptedMiss)
        #expect(chain == .interruptedMiss)
    }

    @Test func aFailedSSDReadStrikesTheCommittedRef() {
        let outcome = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: false, interrupted: false, kind: .ssd
        )
        // **Committed Ref Cleanup** — not **Explicit Ref Discard** (a
        // supersession-side edge, unreachable from resolve).
        #expect(outcome == .failedCleanup(.committedRef))
    }

    @Test func aFailedChainPrefixComposeClearsThePoint() {
        let outcome = SnapshotResolutionLadder.hydrationOutcome(
            succeeded: false, interrupted: false, kind: .chainPrefix
        )
        #expect(outcome == .failedCleanup(.chainPrefixPoint))
    }

    // MARK: - Resolved value builders

    @Test func hydratedHitRewritesToAPlainHitAndMarksSSD() {
        let initial = initialLookup(sharedPrefixLength: 12)
        let body = makeSnapshot(offset: 8, type: .system)
        let resolved = SnapshotResolutionLadder.hydratedHit(
            body, initial: initial, promptTokenCount: 20, partitionKey: key,
            hydrateSeconds: 0.5
        )
        #expect(resolved.hydratedFromSSD == true)
        #expect(resolved.hydrationSeconds == 0.5)
        #expect(resolved.wasChainPrefixRestore == false)
        #expect(resolved.lookup.snapshot?.tokenOffset == 8)
        // Carried through from `initial`, not recomputed.
        #expect(resolved.lookup.sharedPrefixLength == 12)
        #expect(resolved.lookup.divergence?.offset == 3)
        guard case .hit(let offset, let total, let type) = resolved.lookup.reason else {
            Issue.record("expected a .hit reason, got \(resolved.lookup.reason)")
            return
        }
        #expect(offset == 8)
        #expect(total == 20)
        #expect(type == .system)
    }

    @Test func hydratedHitCarriesTheChainPrefixRestoreMarker() {
        let resolved = SnapshotResolutionLadder.hydratedHit(
            makeSnapshot(offset: 8), initial: initialLookup(), promptTokenCount: 20,
            partitionKey: key, hydrateSeconds: 0.25, wasChainPrefixRestore: true
        )
        // The erased Chain-Prefix / Think-Strip Rewind signal (issue #101) rides
        // the marker after the reason is rewritten to `.hit`.
        #expect(resolved.wasChainPrefixRestore == true)
        if case .hit = resolved.lookup.reason {
        } else {
            Issue.record("expected the marker to accompany a rewritten .hit reason")
        }
    }

    @Test func missAfterFailedHydrationDegradesToOffsetZeroButKeepsSSDProvenance() {
        let initial = initialLookup(sharedPrefixLength: 12)
        let resolved = SnapshotResolutionLadder.missAfterFailedHydration(
            initial: initial, partitionKey: key
        )
        #expect(resolved.lookup.snapshot == nil)
        #expect(resolved.lookup.snapshotTokenOffset == 0)
        // A hydration was attempted — provenance stays true — but its failed
        // time is not a cost a future hit pays.
        #expect(resolved.hydratedFromSSD == true)
        #expect(resolved.hydrationSeconds == 0)
        // The `initial` lookup's depth + divergence survive the degrade.
        #expect(resolved.lookup.sharedPrefixLength == 12)
        #expect(resolved.lookup.divergence?.offset == 3)
        if case .missNoSnapshotInPrefix = resolved.lookup.reason {
        } else {
            Issue.record("expected missNoSnapshotInPrefix, got \(resolved.lookup.reason)")
        }
    }

    @Test func gateFallbackHitServesTheResidentBodyWithoutSSDProvenance() {
        let body = makeSnapshot(offset: 4, type: .branchPoint)
        let divergence = PrefixDivergenceProbe(offset: 2, deepestAbandonedOffset: 9)
        let resolved = SnapshotResolutionLadder.gateFallbackHit(
            body: body, partitionKey: key, promptTokenCount: 16, treeMatchDepth: 7,
            recordedHitID: "snap-abc", divergence: divergence
        )
        // The recompute alternative served as an ordinary RAM hit — no SSD hydration.
        #expect(resolved.hydratedFromSSD == false)
        #expect(resolved.hydrationSeconds == 0)
        #expect(resolved.lookup.snapshot?.tokenOffset == 4)
        #expect(resolved.lookup.recordedHitSnapshotID == "snap-abc")
        // The gate re-walk's depth is the shared-prefix length here (not `initial`).
        #expect(resolved.lookup.sharedPrefixLength == 7)
        #expect(resolved.lookup.divergence?.offset == 2)
        guard case .hit(let offset, let total, let type) = resolved.lookup.reason else {
            Issue.record("expected a .hit reason, got \(resolved.lookup.reason)")
            return
        }
        #expect(offset == 4)
        #expect(total == 16)
        #expect(type == .branchPoint)
    }

    @Test func gateFallbackMissCarriesTheGateWalkDepthWithoutSSDProvenance() {
        let divergence = PrefixDivergenceProbe(offset: 1, deepestAbandonedOffset: 5)
        let resolved = SnapshotResolutionLadder.gateFallbackMiss(
            partitionKey: key, treeMatchDepth: 7, divergence: divergence
        )
        #expect(resolved.lookup.snapshot == nil)
        #expect(resolved.lookup.snapshotTokenOffset == 0)
        // The gate skipped hydration entirely — provenance is false.
        #expect(resolved.hydratedFromSSD == false)
        #expect(resolved.lookup.sharedPrefixLength == 7)
        #expect(resolved.lookup.divergence?.offset == 1)
        if case .missNoSnapshotInPrefix = resolved.lookup.reason {
        } else {
            Issue.record("expected missNoSnapshotInPrefix, got \(resolved.lookup.reason)")
        }
    }
}
