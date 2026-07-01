---
status: accepted
---

# SSD hydration handle stays off the MainActor

Historical context: `PrefixCacheManager.lookup` originally vended an
`SSDHitContext` carrying the concrete `nonisolated SSDSnapshotStore` — not a
narrow hydration interface — so `LLMActor` could run `loadSync` off the
MainActor inside `container.perform` when hydrating a body-absent `ssdOnly`
(state-5) node. We deliberately did **not** route that read through
`TieredSnapshotStore`'s sealed interface the way we did for `recordHit`,
`flush`, and `warmStartLoad`, because a `@MainActor` store method would force
the disk read onto the main actor and stall the UI during hydration.

## Consequences

Sealed, as anticipated below. The **Snapshot Resolution** deepening introduced
the second adapter this section named as the trigger — an in-memory hydrator
fake for tests — so the narrow `nonisolated SnapshotHydrating` seam is now real,
not hypothetical. `SSDHitContext` and `ChainPrefixHitContext` carry that handle
(vending `loadSync`, `loadSyncPrefix`, and `recordHit`) instead of the concrete
`SSDSnapshotStore`; the off-MainActor `loadSync` discipline above is unchanged.
A consumer needing broader SSD access still reaches for the concrete store, so
the seam stays minimal rather than widened speculatively. (The original
"architecture review should not seal this leak" directive is retired — it was
conditional on the seam being one-adapter, which no longer holds.)
