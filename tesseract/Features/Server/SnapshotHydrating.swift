import Foundation

/// **Snapshot Hydrating** — the narrow off-main handle that **Snapshot
/// Resolution** depends on to materialize a body-absent snapshot from SSD.
/// Carries only the three members resolution calls off the MainActor:
/// `loadSync`, `loadSyncPrefix`, and `recordHit`. Satisfied in production by
/// the concrete `SSDSnapshotStore` and in tests by an in-memory peer — the
/// second adapter that made this seam real (ADR-0001).
///
/// A consumer needing broader SSD access reaches for the concrete store
/// instead; widening this handle before a second caller needs a member is the
/// shape to avoid. The off-MainActor `loadSync` discipline (a disk read never
/// stalls the UI) lives at this seam: `loadSync`/`loadSyncPrefix` are
/// `nonisolated`, so **Snapshot Resolution** can call them from inside the
/// caller's Metal-affine `container.perform` scope without hopping to the
/// MainActor.
nonisolated protocol SnapshotHydrating: Sendable {
    /// Materialize the body for a committed own ref (state 5) from disk.
    /// `nonisolated` so the read stays off the MainActor (ADR-0001). Returns
    /// `nil` on any failure (missing file / fingerprint mismatch / decode
    /// error); the on-disk backing is removed on failure.
    nonisolated func loadSync(
        snapshotRef: SnapshotRef,
        expectedFingerprint: String
    ) -> HybridCacheSnapshot?

    /// Compose the owning chain's leading segments into a body for a
    /// **Chain-Prefix Restore** point (ADR-0012). Same isolation contract as
    /// `loadSync`: off-MainActor read, `nil` on failure.
    nonisolated func loadSyncPrefix(
        point: ChainPrefixRestorePoint,
        expectedFingerprint: String
    ) -> HybridCacheSnapshot?

    /// Bump an SSD-tier entry's recency so hot entries never look stale to the
    /// SSD eviction policy. `nonisolated` — the one recency answer every caller
    /// reaches through this handle, so resolution no longer touches the concrete
    /// store for it (the `recordHit` inconsistency ADR-0001 named is healed here).
    nonisolated func recordHit(id: String)
}
