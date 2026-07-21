//
//  WarmStartPlanner.swift
//  tesseract
//
//  The **Warm-Start Plan** — the pure rebuild decisions the
//  **Snapshot Ledger** makes when it seeds the in-memory manifest from
//  a loaded (or directory-walk-rebuilt) `manifest.json`. Lifted out of
//  `SnapshotLedger.commitRestoredManifest` so the three warm-start
//  decisions live as one value-in/value-out derivation instead of
//  braided through the ledger lock, the debounced persist, and the
//  detached file cleanup (ADR-0055; sibling to `EvictionCandidatePolicy`
//  / ADR-0049, one lifecycle stage earlier).
//
//  The three decisions, all pure over the loaded manifest + the model's
//  expected fingerprint + a single injected `now`:
//
//  1. the **Stale-Partition GC** cut — the `tierMostRecentUse` anchor
//     fold over current-version, matching-fingerprint partitions, and
//     the anchor-*relative* staleness test (a tier ages relative to its
//     freshest partition's use stamp, never the wall clock; see
//     `CONTEXT.md` "Stale-Partition GC");
//  2. legacy **grace-stamping** — a nil `lastUsedAt` is stamped to `now`
//     and kept, and it must NOT count toward the anchor above (grace-
//     stamping a legacy partition and letting it read as "used now"
//     would inflate the anchor and reclaim a genuinely-stamped sibling
//     at the migration launch — the wall-clock regression the relative
//     rule exists to prevent);
//  3. the **dead-descriptor drop + persist choice** — the schema-version
//     and `CheckpointType(wireString:)` round-trip filter over the
//     snapshots, the `seedBytes` fold, and the `persistNeeded` derivation.
//
//  The ledger keeps every effect: it reads + decodes the file, installs
//  the plan's manifest under the lock, schedules the debounced persist
//  per the plan's flag, and performs the plan's deletions in its detached
//  cleanup task. `nonisolated` — no ledger state, callable from the
//  off-MainActor warm-start path without a hop.
//

import Foundation

// MARK: - On-disk layout

/// Single source of truth for the SSD tier's relative on-disk layout:
/// the `partitions/{digest}/snapshots/{shard}/{name}` shape and the
/// partition-directory derivation. Both the corrupt-manifest rebuild
/// walk and the warm-start cleanup route through here, and the
/// **Warm-Start Plan**'s deletion paths are built from it, so the layout
/// literal has one home instead of being retyped at each call site.
/// `PersistedSnapshotDescriptor.relativeFilePath` is the descriptor-side
/// consumer (it derives the shard byte from the snapshot ID).
nonisolated enum SnapshotDiskLayout {
    /// Top-level directory under the tier root holding every partition.
    static let partitionsRoot = "partitions"

    /// Relative path from the tier root to one partition's directory.
    static func partitionDirectory(digest: String) -> String {
        "\(partitionsRoot)/\(digest)"
    }

    /// Relative path to a partition's `snapshots/` subtree.
    static func snapshotsDirectory(digest: String) -> String {
        "\(partitionDirectory(digest: digest))/snapshots"
    }

    /// Relative path to one `.safetensors` file, given the digest, its
    /// shard byte, and the file name — the path the rebuild walk records
    /// for every file it sees on disk.
    static func snapshotFile(digest: String, shard: String, name: String) -> String {
        "\(snapshotsDirectory(digest: digest))/\(shard)/\(name)"
    }
}

// MARK: - Warm-start plan

/// The pure result of `WarmStartPlanner.plan`: everything the
/// **Snapshot Ledger** needs to seed itself and clean up, with no I/O of
/// its own. The ledger installs `manifest` + `seedBytes` under its lock,
/// schedules a persist iff `persistNeeded`, deletes `deletions` (each a
/// path relative to the tier root) in its detached cleanup, and returns
/// `outcome` to `PrefixCacheManager.warmStart`.
nonisolated struct WarmStartPlan: Sendable {
    /// The manifest to install as the in-memory authority — only the kept
    /// partitions (grace stamps applied) and the kept descriptors.
    let manifest: SnapshotManifest

    /// Chain-total bytes of the kept descriptors — seeds `currentSSDBytes`.
    let seedBytes: Int

    /// Whether the install must schedule a debounced manifest persist
    /// this session. `persistManifestAfter || mutatedRestoredMeta ||
    /// anyInvalidated`: a grace stamp or a reclaim must reach disk (an
    /// unpersisted grace stamp restarts the GC clock every launch; an
    /// unpersisted reclaim resurfaces dangling descriptors on the next
    /// read), and the rebuild path forces it to overwrite the just-renamed
    /// corrupt `manifest.json`.
    let persistNeeded: Bool

    /// Files and directories to delete, each a path relative to the tier
    /// root — invalidated partition directories plus the dead descriptors'
    /// chain files. The ledger joins each to `rootURL` and removes it off
    /// the hot path.
    let deletions: [String]

    /// The partitioned view `PrefixCacheManager.warmStart` iterates —
    /// the reused `WarmStartOutcome` (valid partitions + invalidated
    /// partitions with typed reasons), so the plan carries no parallel
    /// copy of that value.
    let outcome: WarmStartOutcome
}

// MARK: - Warm-start planner

/// Pure derivation of the **Warm-Start Plan** from a loaded manifest.
/// Sibling shape to `EvictionCandidatePolicy` (ADR-0049): value in, value
/// out, `now` promoted to a parameter so the whole decision is pure over
/// its inputs and testable as a decision table (`WarmStartPlannerTests`)
/// rather than a JSON-to-disk round-trip.
nonisolated enum WarmStartPlanner {

    /// Plan the seed from `loaded`.
    ///
    /// - Parameters:
    ///   - loaded: the decoded manifest (from `manifest.json` or the
    ///     directory-walk rebuild).
    ///   - expectedFingerprint: the currently-loaded model's fingerprint;
    ///     partitions whose persisted `modelFingerprint` differs are
    ///     invalidated.
    ///   - now: `Date().timeIntervalSinceReferenceDate` at the call site
    ///     — the grace stamp for legacy nil-`lastUsedAt` metas. Never a
    ///     staleness input (staleness is anchor-relative).
    ///   - currentSchemaVersion: the schema version a kept meta/descriptor
    ///     must match; anything else is a hand-edited / partially-upgraded
    ///     file and is dropped.
    ///   - maxUnusedAge: the **Stale-Partition GC** use-gap. A partition
    ///     whose stamp trails the tier anchor by more than this is reclaimed.
    ///   - persistManifestAfter: force a persist regardless of mutations
    ///     (the rebuild path passes `true` to overwrite the renamed corrupt
    ///     manifest; the normal load path passes `false`).
    ///   - isKnownCheckpointType: whether a descriptor's wire checkpoint
    ///     type still round-trips. Injected as a value so the drop rule is
    ///     testable; defaults to the real `CheckpointType(wireString:)` gate.
    static func plan(
        loaded: SnapshotManifest,
        expectedFingerprint: String,
        now: TimeInterval,
        currentSchemaVersion: Int = SnapshotManifestSchema.currentVersion,
        maxUnusedAge: TimeInterval = SSDStalePartitionPolicy.maxUnusedAge,
        persistManifestAfter: Bool,
        isKnownCheckpointType: (String) -> Bool = {
            HybridCacheSnapshot.CheckpointType(wireString: $0) != nil
        }
    ) -> WarmStartPlan {
        var restored = SnapshotManifest.empty()
        var invalidated: [WarmStartOutcome.InvalidatedPartition] = []

        // Chain-total bytes per partition, so each reclaim can report what
        // it returned to the budget.
        var bytesByPartition: [String: Int] = [:]
        for descriptor in loaded.snapshots.values {
            bytesByPartition[descriptor.partitionDigest, default: 0] += descriptor.totalBytes
        }
        func invalidate(
            _ digest: String,
            _ meta: PartitionMeta,
            _ reason: WarmStartOutcome.PartitionInvalidationReason
        ) {
            invalidated.append(
                WarmStartOutcome.InvalidatedPartition(
                    digest: digest,
                    modelID: meta.modelID,
                    bytes: bytesByPartition[digest] ?? 0,
                    reason: reason
                ))
        }

        // True when a legacy meta was grace-stamped below — the stamp must
        // reach disk this session or the GC clock never starts.
        var mutatedRestoredMeta = false

        // Stale-Partition GC (PRD #150): staleness is measured against the
        // freshest valid partition's use stamp, not the wall clock — an
        // idle week must not reclaim the whole tier. Anchor on *real*
        // stamps of current-version, matching-fingerprint partitions only:
        // a legacy nil-stamped partition is about to be grace-stamped, and
        // letting it count as "used now" would inflate the anchor and
        // reclaim a genuinely-stamped sibling at the migration launch — the
        // wall-clock regression the relative rule exists to prevent. No
        // stamps anywhere → no anchor → nothing reclaimed this launch.
        let tierMostRecentUse = loaded.partitions.values
            .filter {
                $0.schemaVersion == currentSchemaVersion
                    && $0.modelFingerprint == expectedFingerprint
            }
            .compactMap(\.lastUsedAt)
            .max()

        for (digest, meta) in loaded.partitions {
            // Stale `PartitionMeta` inside a current-version manifest
            // signals a hand-edited or partially upgraded file — drop it
            // rather than reattach under stale canonicalization.
            guard meta.schemaVersion == currentSchemaVersion else {
                invalidate(digest, meta, .schemaStale)
                continue
            }
            guard meta.modelFingerprint == expectedFingerprint else {
                invalidate(digest, meta, .fingerprintChanged)
                continue
            }
            // A legacy meta without a stamp is grace-stamped to "now" — the
            // clock starts here, it does not retroactively reclaim
            // long-lived caches.
            if let lastUsed = meta.lastUsedAt {
                if let anchor = tierMostRecentUse,
                    anchor - lastUsed > maxUnusedAge
                {
                    invalidate(digest, meta, .staleUnused)
                    continue
                }
                restored.partitions[digest] = meta
            } else {
                var graced = meta
                graced.lastUsedAt = now
                restored.partitions[digest] = graced
                mutatedRestoredMeta = true
            }
        }

        var descriptorsByDigest: [String: [PersistedSnapshotDescriptor]] = [:]
        var deadDescriptorFiles: [String] = []
        for (id, desc) in loaded.snapshots {
            guard restored.partitions[desc.partitionDigest] != nil else { continue }
            // Same rationale as the `PartitionMeta` filter above.
            guard desc.schemaVersion == currentSchemaVersion else {
                deadDescriptorFiles.append(contentsOf: desc.chainFileRelativePaths)
                continue
            }
            // Drop descriptors whose wire-format checkpoint type no longer
            // decodes — `PrefixCacheManager.warmStart` would skip them
            // silently otherwise, leaving their bytes stranded in
            // `currentSSDBytes`.
            guard isKnownCheckpointType(desc.checkpointType) else {
                deadDescriptorFiles.append(contentsOf: desc.chainFileRelativePaths)
                continue
            }
            restored.snapshots[id] = desc
            descriptorsByDigest[desc.partitionDigest, default: []].append(desc)
        }

        let seedBytes = restored.snapshots.values.reduce(0) { $0 + $1.totalBytes }

        // Reclaims and grace stamps must reach disk even on the normal
        // manifest-load path (which otherwise defers persistence to the
        // next mutation): an unpersisted grace stamp restarts the GC clock
        // every launch, and an unpersisted reclaim resurfaces the
        // partition's dangling descriptors on the next read.
        let persistNeeded =
            persistManifestAfter || mutatedRestoredMeta || !invalidated.isEmpty

        let validPartitions: [WarmStartOutcome.Partition] = restored.partitions.map {
            digest, meta in
            WarmStartOutcome.Partition(
                digest: digest,
                meta: meta,
                descriptors: (descriptorsByDigest[digest] ?? [])
                    .sorted { $0.snapshotID < $1.snapshotID }
            )
        }
        .sorted { $0.digest < $1.digest }

        // Invalidated partition directories first, then the dead
        // descriptors' chain files (under kept partitions). The two sets
        // are disjoint — an invalidated partition is never kept — so the
        // ledger's single detached sweep deletes exactly what today's two
        // sweeps did.
        let deletions =
            invalidated.map { SnapshotDiskLayout.partitionDirectory(digest: $0.digest) }
            + deadDescriptorFiles

        let outcome = WarmStartOutcome(
            validPartitions: validPartitions,
            invalidated: invalidated.sorted { $0.digest < $1.digest }
        )

        return WarmStartPlan(
            manifest: restored,
            seedBytes: seedBytes,
            persistNeeded: persistNeeded,
            deletions: deletions,
            outcome: outcome
        )
    }
}
