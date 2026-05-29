---
status: accepted
---

# SSD hydration handle stays off the MainActor

`PrefixCacheManager.lookup` vends an `SSDHitContext` that carries the concrete
`nonisolated SSDSnapshotStore` — not a narrow hydration interface — so `LLMActor`
can run `loadSync` off the MainActor inside `container.perform` when hydrating a
body-absent `ssdOnly` (state-5) node. We deliberately did **not** route this
through `TieredSnapshotStore`'s sealed interface the way we did for `recordHit`,
`flush`, and `warmStartLoad`, because a `@MainActor` store method would force the
disk read onto the main actor and stall the UI during hydration.

## Consequences

The single remaining place the SSD store escapes the `TieredSnapshotStore` seam is
`makeSSDHitContext` / the `SSDHitContext.ssdStore` field. An architecture review
that finds this leak should treat it as intentional and not "seal" it.

If we ever do want it sealed, the move is a narrow `nonisolated SnapshotHydrating`
interface that vends just `loadSync` (and `recordHit`), so LLMActor depends on the
handle instead of the concrete store. That is a one-adapter (hence hypothetical)
seam today — only worth introducing once a second adapter exists, e.g. an
in-memory hydrator fake for tests.
