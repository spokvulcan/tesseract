# ADR-0055: Warm start is a pure plan — the ledger performs it

- Status: Accepted
- Date: 2026-07-22
- Relates to: ADR-0049 (eviction candidate selection as a pure policy — the
  same one-decision move one lifecycle stage later), ADR-0010/0011/0012/0019
  (SSD tier design), PRD #150 (stale-partition GC), issue #397

## Context

`SnapshotLedger.commitRestoredManifest` — the shared seed step for both the
normal `manifest.json` load and the corrupt-manifest directory-walk rebuild —
braided three *pure* warm-start decisions through the ledger lock, the
debounced persist, and the detached file cleanup:

1. the **Stale-Partition GC** cut — the `tierMostRecentUse` anchor fold over
   current-version, matching-fingerprint partitions and the anchor-*relative*
   staleness test (a tier ages relative to its freshest partition's use stamp,
   never the wall clock), including the subtle rule that a legacy nil-stamped
   partition, about to be grace-stamped, must **not** count toward the anchor —
   letting it read as "used now" would inflate the anchor and reclaim a
   genuinely-stamped sibling at the migration launch;
2. legacy **grace-stamping** — a nil `lastUsedAt` stamped to `now` and kept;
3. the **dead-descriptor drop + persist choice** — the schema-version and
   `CheckpointType(wireString:)` round-trip filter, the `seedBytes` fold, and
   the `persistAfter = persistManifestAfter || mutatedRestoredMeta ||
   anyInvalidated` derivation.

Every test of these decisions (`SnapshotLedgerTests`) paid a JSON-to-disk round
trip through a real ledger; the anchor rule — exactly the kind of subtle policy
that inverts on a one-line change — had no pure test at all. The on-disk layout
literal `partitions/{digest}/snapshots/{shard}/{name}` was retyped between the
rebuild walk and the detached cleanup, a second silent-divergence risk.

## Decision

One pure module, **`WarmStartPlanner`**, holds the three decisions as a
value-in / value-out derivation; the ledger keeps every effect.

- **`WarmStartPlanner.plan(loaded:expectedFingerprint:now:…​persistManifestAfter:)`**
  — `nonisolated`, sibling shape to `EvictionCandidatePolicy` (ADR-0049), with
  `now` promoted to a parameter so the whole decision is pure over its inputs.
  It returns a **`WarmStartPlan`**: the manifest to install, `seedBytes`, the
  `persistNeeded` flag, the file deletions as **root-relative paths**, and the
  reused `WarmStartOutcome` (valid + invalidated partitions with typed reasons)
  — the plan wraps the existing outcome rather than duplicating it.
- `commitRestoredManifest` becomes: read + decode (I/O, upstream) → `plan`
  (pure) → install `manifest` + `seedBytes` under the ledger lock → schedule the
  debounced persist iff `persistNeeded` → delete the plan's paths in one
  detached sweep. No behavior change — the same manifests, stamps, reclaims, and
  deletions as before.
- The layout literal collapses into **`SnapshotDiskLayout`** (`nonisolated`
  enum): the partition-directory and `snapshots/{shard}/{name}` derivations used
  by the rebuild walk, the plan's deletion paths, `writePartitionMetaFile`, and
  `PersistedSnapshotDescriptor.relativeFilePath` — one source of truth for the
  on-disk shape.

## Consequences

- The anchor rule is now a decision table (`WarmStartPlannerTests`): an idle
  tier ages together (old stamps survive when the whole tier is old), an
  abandoned variant of a still-active model ages out, warm start never refreshes
  a kept stamp, a legacy grace-stamp does not inflate the anchor and reclaim a
  sibling, plus the fingerprint / stale-schema / unknown-checkpoint-type drops,
  the persist-choice rows, and the `seedBytes` fold — none of it paying a disk
  round trip.
- `SnapshotLedgerTests` keeps its role unchanged as the disk-integration layer:
  it is the proof that the ledger *performs* the plan (reads the file, installs
  the manifest, persists, deletes) — no longer doubling as the only proof of the
  decisions themselves.
- A latent quirk is preserved deliberately: a descriptor dropped only for a dead
  checkpoint type / stale descriptor schema does not by itself force a persist,
  so the stale entry is re-read and re-dropped next launch. Behavior-preserving
  by contract; the plan makes it a named, tested row rather than an accident.
- The two effect-free lifecycle cuts — warm-start seeding and eviction
  selection — now sit as sibling pure modules, where the SSD tier's policy
  surface is visible and reviewable in one place.
