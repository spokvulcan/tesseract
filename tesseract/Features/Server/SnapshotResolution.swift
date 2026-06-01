import Foundation
import MLXLMCommon

/// Resolves a token path to the best usable cached prefix under a
/// `CachePartitionKey`: the radix **lookup** plus lazy **SSD hydration**, in
/// one place. The read-side counterpart to **Snapshot Admission**. It owns this
/// lookup-then-hydrate-if-SSD sequence for the main prefill path; the
/// canonical-leaf fallback still hydrates inline via
/// `LLMActor.hydrateSSDLookupIfNeeded` and is the next caller to converge here.
///
/// Surfaces only `.hit` or a miss — the `.ssdHit` hydration intermediate is
/// consumed internally (promote on success, Committed Ref Cleanup on failure).
///
/// Crosses isolation exactly as **Snapshot Admission** does: the lookup and the
/// promote/clear hops run on the MainActor, but `loadSync` stays off-MainActor
/// inside the caller's `container.perform` scope so a disk read never stalls the
/// UI (ADR-0001). `nonisolated`, driven from inside the Metal-affine scope.
nonisolated enum SnapshotResolution {

    /// The resolved lookup plus whether it was materialized from an SSD-only
    /// committed ref.
    struct Resolved: Sendable {
        let lookup: PrefixCacheManager.LookupResult
        let hydratedFromSSD: Bool

        /// The lookup to align checkpoint planning against, or `nil` to skip the
        /// alignment merge. An SSD-hydrated hit aligns against nothing: the
        /// pre-resolution ordering planned against the unhydrated `.ssdHit`,
        /// which never merged an alignment branch-point, so hydrating one must
        /// not start aligning. Keeps that SSD-timing rule here, where the
        /// hydration story is known, rather than at the planning call site.
        var alignmentLookup: PrefixCacheManager.LookupResult? {
            hydratedFromSSD ? nil : lookup
        }
    }

    static func resolve(
        tokens: [Int],
        promptTokenCount: Int,
        partitionKey: CachePartitionKey,
        modelFingerprint: String?,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async -> Resolved {
        let initial = await MainActor.run {
            prefixCache.lookup(tokens: tokens, partitionKey: partitionKey)
        }

        // Only `.ssdHit` (state 5 — committed ref, no body) needs hydration, and
        // only when we have a fingerprint to validate the on-disk body against.
        guard case .ssdHit(let ctx) = initial.reason, let fingerprint = modelFingerprint else {
            if let id = initial.recordedHitSnapshotID {
                diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: id))
            }
            return Resolved(lookup: initial, hydratedFromSSD: false)
        }

        // Materialize the body from disk on this Metal-affine thread (ADR-0001).
        let hydrateStart = Date.timeIntervalSinceReferenceDate
        let hydrated = ctx.ssdStore.loadSync(
            snapshotRef: ctx.snapshotRef, expectedFingerprint: fingerprint
        )
        let hydrateSeconds = Date.timeIntervalSinceReferenceDate - hydrateStart

        guard let hydrated else {
            // Hydration failed: `loadSync` already removed the on-disk file; the
            // forgiving clear removes the now-bodyless node so checkpoint planning
            // against the settled tree re-captures the lost checkpoint cold.
            await MainActor.run {
                prefixCache.clearCommittedSnapshotRefAfterHydrationFailure(
                    node: ctx.node, partitionKey: partitionKey
                )
            }
            return Resolved(
                lookup: PrefixCacheManager.LookupResult(
                    snapshot: nil,
                    partitionKey: partitionKey,
                    snapshotTokenOffset: 0,
                    sharedPrefixLength: initial.sharedPrefixLength,
                    reason: .missNoSnapshotInPrefix
                ),
                hydratedFromSSD: true
            )
        }

        diagnostics.log(PrefixCacheDiagnostics.SSDHitEvent(
            id: ctx.snapshotRef.snapshotID, hydrateMs: hydrateSeconds
        ))
        await MainActor.run {
            ctx.ssdStore.recordHit(id: ctx.snapshotRef.snapshotID)
            prefixCache.promote(node: ctx.node, snapshot: hydrated, partitionKey: partitionKey)
        }
        diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: ctx.snapshotRef.snapshotID))

        return Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: hydrated,
                partitionKey: partitionKey,
                snapshotTokenOffset: hydrated.tokenOffset,
                sharedPrefixLength: initial.sharedPrefixLength,
                reason: .hit(
                    snapshotOffset: hydrated.tokenOffset,
                    totalTokens: promptTokenCount,
                    type: hydrated.checkpointType
                )
            ),
            hydratedFromSSD: true
        )
    }
}
