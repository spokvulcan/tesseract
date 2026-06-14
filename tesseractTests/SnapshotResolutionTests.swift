import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of **Snapshot Resolution** — the single lookup + hydrate-if-SSD
/// home shared by the main prefill and the canonical-leaf fallback.
///
/// These exercise the composition that runs without disk: a RAM hit and a
/// miss pass through `resolve` unchanged, with no hydration. The SSD-hydration
/// branch (promote on success, Committed Ref Cleanup on failure) is covered by
/// the PrefixCacheManager state-5 tests (`lookup` → `.ssdHit`, `promote`,
/// `clearCommittedSnapshotRefAfterHydrationFailure`) that `resolve` composes,
/// plus the loaded-model E2E net — ADR-0001 keeps the SSD store concrete, so
/// there is no in-memory hydration seam to fake at this layer.
@MainActor
@Suite struct SnapshotResolutionTests {

    private let key = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)

    private var diagnostics: PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
    }

    private func makeManager() -> PrefixCacheManager {
        PrefixCacheManager(memoryBudgetBytes: 100 * 1024 * 1024)
    }

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

    @Test func resolveSurfacesAMissWithoutHydrationWhenNothingIsCached() async {
        let manager = makeManager()
        let resolved = await SnapshotResolution.resolve(
            tokens: [1, 2, 3, 4],
            promptTokenCount: 4,
            partitionKey: key,
            modelFingerprint: nil,
            prefixCache: manager,
            diagnostics: diagnostics
        )
        #expect(resolved.hydratedFromSSD == false)
        #expect(resolved.lookup.snapshot == nil)
        if case .hit = resolved.lookup.reason { Issue.record("expected a miss, got a hit") }
    }

    @Test func resolveSurfacesARamHitWithoutHydration() async {
        let manager = makeManager()
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let snapshot = makeSnapshot(offset: tokens.count)
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: tokens, snapshot: snapshot, storage: .ramOnly, partitionKey: key
            )!)

        let resolved = await SnapshotResolution.resolve(
            tokens: tokens,
            promptTokenCount: tokens.count,
            partitionKey: key,
            modelFingerprint: nil,
            prefixCache: manager,
            diagnostics: diagnostics
        )

        #expect(resolved.hydratedFromSSD == false)
        guard case .hit = resolved.lookup.reason else {
            Issue.record("expected a hit, got \(resolved.lookup.reason)")
            return
        }
        #expect(resolved.lookup.snapshot?.tokenOffset == tokens.count)
    }

    // MARK: - alignmentLookup (the SSD-timing rule)

    /// A `.hit` lookup for the alignment tests — `alignmentLookup` keys off
    /// `hydratedFromSSD`, not the lookup's contents, so any hit will do.
    private func ramHitLookup(offset: Int = 8) -> PrefixCacheManager.LookupResult {
        let snapshot = makeSnapshot(offset: offset)
        return PrefixCacheManager.LookupResult(
            snapshot: snapshot,
            partitionKey: key,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: snapshot.tokenOffset,
            reason: .hit(
                snapshotOffset: snapshot.tokenOffset, totalTokens: offset * 2,
                type: snapshot.checkpointType)
        )
    }

    @Test func alignmentLookupIsTheLookupWhenNotHydratedFromSSD() {
        let lookup = ramHitLookup()
        let resolved = SnapshotResolution.Resolved(
            lookup: lookup, hydratedFromSSD: false, hydrationSeconds: 0
        )
        // A RAM hit aligns checkpoint planning against itself.
        #expect(resolved.alignmentLookup != nil)
        #expect(resolved.alignmentLookup?.snapshotTokenOffset == lookup.snapshotTokenOffset)
    }

    @Test func alignmentLookupIsNilForAnSSDHydratedHit() {
        let resolved = SnapshotResolution.Resolved(
            lookup: ramHitLookup(), hydratedFromSSD: true, hydrationSeconds: 0
        )
        // An SSD-hydrated hit must align against nothing: it matches the pre-carve
        // ordering that planned against the unhydrated `.ssdHit`, which never merged
        // an alignment branch-point. Flipping this rule would silently resume it.
        #expect(resolved.alignmentLookup == nil)
    }
}
