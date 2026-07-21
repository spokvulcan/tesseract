import Foundation
import MLXLMCommon

/// The pure decision ladder behind **Snapshot Resolution**'s lookup-then-hydrate
/// state machine (issue #400). Resolution stays manager-owned — `resolve()` is
/// still the single read-side door, and the **Prefix Cache Manager** still
/// performs every `await`, tree/ledger mutation, hydration call, pin placement,
/// and telemetry emission. What moves here is only the *choosing*: each step's
/// facts map to a decision value, and the manager reads that value and performs
/// the decided effects in order.
///
/// This is the reducer/performer split ADR-0033 (and ADR-0049/0050/0051) name —
/// "phases return values; the module owns effects" — applied to the read side.
/// It cannot recreate the retired "hydrator" that reached back across the
/// manager's mutation seam (CONTEXT.md → **Snapshot Resolution** _Avoid_): the
/// ladder holds **no references at all** — no manager, tree, ledger, node, or
/// **Snapshot Hydrating** handle. Every input is a plain fact or value the
/// manager extracted; every output is a decision value or a `Sendable`
/// `Resolved`/`LookupResult` the manager returns unchanged.
///
/// `nonisolated` (against the module's default MainActor isolation) so the
/// manager can consult it from `resolve`'s Metal-affine `container.perform`
/// scope, off the MainActor, exactly where the hydration read runs (ADR-0001).
nonisolated enum SnapshotResolutionLadder {

    // MARK: - Step A: the Hydration Gate

    /// The **Hydration Gate** outcome (PRD #149 item 7) once the manager has
    /// priced the body-less hit against the deepest resident RAM body on the
    /// path (the recompute alternative). Three-valued: attempt-hydration when
    /// the disk read is worth it, else serve the resident alternative as an
    /// ordinary hit, else — nothing resident — a clean miss. The skipped hit's
    /// backing is untouched either way; it stays hittable for a deeper future
    /// request whose recompute span prices the other way.
    enum GateOutcome: Equatable {
        /// Hydrating the body-less hit beats recomputing — proceed to the
        /// off-main `loadSync` / `loadSyncPrefix` read.
        case hydrate
        /// Recompute is cheaper and a resident body is on the path — serve it
        /// as an ordinary hit (access bumped, savings counted, restore path
        /// pinned).
        case serveAlternative
        /// Recompute is cheaper but no body is resident — a clean miss.
        case fallbackMiss
    }

    /// `hydrationGateAdmits` (the `EvictionPolicy` cost compare, computed by the
    /// manager) plus whether a resident RAM body was peeked on the path fold to
    /// the gate outcome. Pure: the manager owns the peek and the effects.
    static func gateOutcome(admitsHydration: Bool, hasAlternativeBody: Bool) -> GateOutcome {
        guard !admitsHydration else { return .hydrate }
        return hasAlternativeBody ? .serveAlternative : .fallbackMiss
    }

    // MARK: - Step B: the post-hydration decision

    /// Which faulted node the manager strikes after a *real* hydration failure,
    /// so the caller's re-lookup degrades to the next-shallower resident body
    /// instead of re-attempting a broken read. The typed cleanup edge is the
    /// decision value; the manager performs it on the MainActor.
    ///
    /// Both edges share the "eager backing-loss" semantics: `loadSync` /
    /// `loadSyncPrefix` already deleted the on-disk backing, so clearing
    /// orphans nothing. This is *not* **Explicit Ref Discard**
    /// (`discardSnapshotRefAfterExplicitDelete`) — that strict edge belongs to
    /// leaf supersession (an already-deleted backing on the *write* side) and
    /// is unreachable from resolve. See the report note on issue #400.
    enum Cleanup: Equatable {
        /// **Committed Ref Cleanup** (CONTEXT.md) — the `.ssdHit`'s state-5
        /// `loadSync` failed; strike the committed **Snapshot Ref**.
        case committedRef
        /// The **Chain-Prefix Restore** (ADR-0012) compose failed; clear the
        /// *point* (not a ref), which borrows the owner chain's segments.
        case chainPrefixPoint
    }

    /// A body-less hit's two hydratable kinds, so one decision function serves
    /// both the `.ssdHit` and `.chainPrefixHit` performers.
    enum HydrationKind: Equatable {
        case ssd
        case chainPrefix
    }

    /// The post-hydration decision: the off-main read's result folds to a
    /// hit, an interrupted miss, or a real-failure cleanup.
    enum HydrationOutcome: Equatable {
        /// Body materialized — perform the load-bearing success ordering
        /// (bump SSD recency, promote the body, fold the measurement), then
        /// return the hydrated hit.
        case hydratedHit
        /// Interrupted (PRD #149 item 7), not failed: the backing is intact,
        /// the node stays hittable for the next caller — surface a clean miss
        /// and clear nothing.
        case interruptedMiss
        /// Real failure (missing file / fingerprint mismatch / decode error):
        /// strike the faulted node via the typed cleanup edge, then re-look-up.
        case failedCleanup(Cleanup)
    }

    /// Fold the off-main read result + interruption state + hit kind to the
    /// post-hydration decision. On success the interruption is irrelevant (a
    /// materialized body is always a hit); a failure is a real cleanup unless
    /// the read was interrupted, whose backing must stay intact.
    static func hydrationOutcome(
        succeeded: Bool,
        interrupted: Bool,
        kind: HydrationKind
    ) -> HydrationOutcome {
        if succeeded { return .hydratedHit }
        if interrupted { return .interruptedMiss }
        switch kind {
        case .ssd: return .failedCleanup(.committedRef)
        case .chainPrefix: return .failedCleanup(.chainPrefixPoint)
        }
    }

    // MARK: - Resolved value builders (pure)

    /// The `Resolved` a hydrated hit resolves to (was `hydratedHit`). Rewrites
    /// the body-less hit's reason to a plain `.hit` before telemetry sees it —
    /// the `wasChainPrefixRestore` marker carries the erased Chain-Prefix /
    /// Think-Strip Rewind signal (issue #101) forward instead. Carries the
    /// `initial` lookup's `sharedPrefixLength` and `divergence` through
    /// unchanged. Pure value transform — builds a `Sendable` `Resolved`, holds
    /// nothing.
    static func hydratedHit(
        _ hydrated: HybridCacheSnapshot,
        initial: PrefixCacheManager.LookupResult,
        promptTokenCount: Int,
        partitionKey: CachePartitionKey,
        hydrateSeconds: TimeInterval,
        wasChainPrefixRestore: Bool = false
    ) -> PrefixCacheManager.Resolved {
        PrefixCacheManager.Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: hydrated,
                partitionKey: partitionKey,
                snapshotTokenOffset: hydrated.tokenOffset,
                sharedPrefixLength: initial.sharedPrefixLength,
                reason: .hit(
                    snapshotOffset: hydrated.tokenOffset,
                    totalTokens: promptTokenCount,
                    type: hydrated.checkpointType
                ),
                divergence: initial.divergence
            ),
            hydratedFromSSD: true,
            hydrationSeconds: hydrateSeconds,
            wasChainPrefixRestore: wasChainPrefixRestore
        )
    }

    /// The `Resolved` a failed (or interrupted) hydration resolves to (was
    /// `missAfterFailedHydration`). Degrades to a clean miss at offset 0 while
    /// carrying the `initial` lookup's `sharedPrefixLength` and `divergence`.
    /// `hydratedFromSSD: true` — a hydration was attempted; its failed time is
    /// not a cost a future hit would pay, so `hydrationSeconds` stays 0.
    static func missAfterFailedHydration(
        initial: PrefixCacheManager.LookupResult,
        partitionKey: CachePartitionKey
    ) -> PrefixCacheManager.Resolved {
        PrefixCacheManager.Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: nil,
                partitionKey: partitionKey,
                snapshotTokenOffset: 0,
                sharedPrefixLength: initial.sharedPrefixLength,
                reason: .missNoSnapshotInPrefix,
                divergence: initial.divergence
            ),
            hydratedFromSSD: true,
            hydrationSeconds: 0
        )
    }

    /// The `Resolved` the gate's `serveAlternative` resolves to: the peeked
    /// resident body served as a real `.hit`. The manager bumps the node's
    /// access and pins it before calling this; the pure part is the value
    /// shape (offset, reason, the just-recorded SSD hit id, the fresh
    /// divergence from the gate's re-walk).
    static func gateFallbackHit(
        body: HybridCacheSnapshot,
        partitionKey: CachePartitionKey,
        promptTokenCount: Int,
        treeMatchDepth: Int,
        recordedHitID: String?,
        divergence: PrefixDivergenceProbe?
    ) -> PrefixCacheManager.Resolved {
        PrefixCacheManager.Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: body,
                partitionKey: partitionKey,
                snapshotTokenOffset: body.tokenOffset,
                sharedPrefixLength: treeMatchDepth,
                reason: .hit(
                    snapshotOffset: body.tokenOffset,
                    totalTokens: promptTokenCount,
                    type: body.checkpointType
                ),
                recordedHitSnapshotID: recordedHitID,
                divergence: divergence
            ),
            hydratedFromSSD: false,
            hydrationSeconds: 0
        )
    }

    /// The `Resolved` the gate's `fallbackMiss` resolves to: recompute is
    /// cheaper but nothing resident on the path — a clean miss carrying the
    /// gate re-walk's `treeMatchDepth` and `divergence`. `hydratedFromSSD:
    /// false` — no hydration was attempted, the gate skipped it.
    static func gateFallbackMiss(
        partitionKey: CachePartitionKey,
        treeMatchDepth: Int,
        divergence: PrefixDivergenceProbe?
    ) -> PrefixCacheManager.Resolved {
        PrefixCacheManager.Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: nil,
                partitionKey: partitionKey,
                snapshotTokenOffset: 0,
                sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix,
                divergence: divergence
            ),
            hydratedFromSSD: false,
            hydrationSeconds: 0
        )
    }
}
