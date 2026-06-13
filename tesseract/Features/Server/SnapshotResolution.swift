import Foundation
import MLXLMCommon

/// Resolves a token path to the best usable cached prefix under a
/// `CachePartitionKey`: the radix **lookup** plus lazy **SSD hydration**, in
/// one place. The read-side counterpart to **Snapshot Admission**. It owns this
/// lookup-then-hydrate-if-SSD sequence for both callers — the main prefill path
/// and the canonical-leaf fallback, which drives `resolve` from inside its own
/// `container.perform` exactly as the main path does.
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
        /// Wall-clock seconds `loadSync` spent materializing the body
        /// (`0` for RAM hits, misses, and failed hydrations — a failure
        /// surfaces as a miss and its time is not a hydration cost a
        /// future hit would pay). Feeds the per-completion trace record.
        let hydrationSeconds: TimeInterval

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

        guard let fingerprint = modelFingerprint else {
            if let id = initial.recordedHitSnapshotID {
                diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: id))
            }
            return Resolved(lookup: initial, hydratedFromSSD: false, hydrationSeconds: 0)
        }

        // Two body-less hit kinds need hydration — `.ssdHit` (state 5,
        // committed own ref) and `.chainPrefixHit` (ADR-0012, backed by the
        // owning chain's leading segments).
        switch initial.reason {
        case .ssdHit(let ctx):
            return await resolveSSDHit(
                ctx, initial: initial, promptTokenCount: promptTokenCount,
                partitionKey: partitionKey, fingerprint: fingerprint,
                prefixCache: prefixCache, diagnostics: diagnostics
            )
        case .chainPrefixHit(let ctx):
            return await resolveChainPrefixHit(
                ctx, initial: initial, promptTokenCount: promptTokenCount,
                partitionKey: partitionKey, fingerprint: fingerprint,
                prefixCache: prefixCache, diagnostics: diagnostics
            )
        default:
            if let id = initial.recordedHitSnapshotID {
                diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: id))
            }
            return Resolved(lookup: initial, hydratedFromSSD: false, hydrationSeconds: 0)
        }
    }

    private static func resolveSSDHit(
        _ ctx: SSDHitContext,
        initial: PrefixCacheManager.LookupResult,
        promptTokenCount: Int,
        partitionKey: CachePartitionKey,
        fingerprint: String,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async -> Resolved {
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
            return missAfterFailedHydration(initial: initial, partitionKey: partitionKey)
        }

        diagnostics.log(PrefixCacheDiagnostics.SSDHitEvent(
            id: ctx.snapshotRef.snapshotID, hydrateMs: hydrateSeconds
        ))
        let hydratedBytes = ctx.snapshotRef.bytesOnDisk
        await MainActor.run {
            ctx.ssdStore.recordHit(id: ctx.snapshotRef.snapshotID)
            prefixCache.promote(node: ctx.node, snapshot: hydrated, partitionKey: partitionKey)
            // Fold the observed hydration into the rolling bytes/s
            // estimate — a real measured operation, never a constant.
            prefixCache.recordHydrationMeasurement(
                bytes: hydratedBytes, seconds: hydrateSeconds
            )
        }
        diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: ctx.snapshotRef.snapshotID))

        return hydratedHit(
            hydrated, initial: initial, promptTokenCount: promptTokenCount,
            partitionKey: partitionKey, hydrateSeconds: hydrateSeconds
        )
    }

    /// **Chain-Prefix Restore** hydration (ADR-0012): compose the owning
    /// chain's leading segments into a body at the boundary. Mirrors
    /// `resolveSSDHit` — same Metal-affine read, same measured-seconds
    /// fold; the owner chain's recency bumps (a prefix hit keeps the
    /// whole chain hot), and a failed compose clears the *point*, not a
    /// committed ref, degrading the next lookup to the next-shallower
    /// backing.
    private static func resolveChainPrefixHit(
        _ ctx: ChainPrefixHitContext,
        initial: PrefixCacheManager.LookupResult,
        promptTokenCount: Int,
        partitionKey: CachePartitionKey,
        fingerprint: String,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async -> Resolved {
        let hydrateStart = Date.timeIntervalSinceReferenceDate
        let hydrated = ctx.ssdStore.loadSyncPrefix(
            point: ctx.point, expectedFingerprint: fingerprint
        )
        let hydrateSeconds = Date.timeIntervalSinceReferenceDate - hydrateStart

        guard let hydrated else {
            await MainActor.run {
                prefixCache.clearChainPrefixRestorePointAfterHydrationFailure(
                    node: ctx.node, partitionKey: partitionKey
                )
            }
            return missAfterFailedHydration(initial: initial, partitionKey: partitionKey)
        }

        diagnostics.log(PrefixCacheDiagnostics.SSDHitEvent(
            id: ctx.point.ownerSnapshotID, hydrateMs: hydrateSeconds
        ))
        await MainActor.run {
            ctx.ssdStore.recordHit(id: ctx.point.ownerSnapshotID)
            prefixCache.promoteChainPrefix(
                node: ctx.node, snapshot: hydrated, partitionKey: partitionKey
            )
            prefixCache.recordHydrationMeasurement(
                bytes: ctx.point.prefixBytes, seconds: hydrateSeconds
            )
        }
        diagnostics.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: ctx.point.ownerSnapshotID))

        return hydratedHit(
            hydrated, initial: initial, promptTokenCount: promptTokenCount,
            partitionKey: partitionKey, hydrateSeconds: hydrateSeconds
        )
    }

    private static func missAfterFailedHydration(
        initial: PrefixCacheManager.LookupResult,
        partitionKey: CachePartitionKey
    ) -> Resolved {
        Resolved(
            lookup: PrefixCacheManager.LookupResult(
                snapshot: nil,
                partitionKey: partitionKey,
                snapshotTokenOffset: 0,
                sharedPrefixLength: initial.sharedPrefixLength,
                reason: .missNoSnapshotInPrefix
            ),
            hydratedFromSSD: true,
            hydrationSeconds: 0
        )
    }

    private static func hydratedHit(
        _ hydrated: HybridCacheSnapshot,
        initial: PrefixCacheManager.LookupResult,
        promptTokenCount: Int,
        partitionKey: CachePartitionKey,
        hydrateSeconds: TimeInterval
    ) -> Resolved {
        Resolved(
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
            hydratedFromSSD: true,
            hydrationSeconds: hydrateSeconds
        )
    }
}
