//
//  WarmStartPlannerTests.swift
//  tesseractTests
//
//  Decision table for the **Warm-Start Plan** (`WarmStartPlanner`,
//  ADR-0055): the pure rebuild decisions the **Snapshot Ledger** makes
//  when seeding itself from a loaded manifest. No disk — a decoded
//  `SnapshotManifest` value + an expected fingerprint + a fixed `now`
//  go in, a `WarmStartPlan` comes out. The disk-round-trip integration
//  layer stays in `SnapshotLedgerTests`; this file names each decision.
//
//  The subtle one is the **Stale-Partition GC** anchor rule: staleness
//  is measured relative to the tier's freshest use stamp, never the wall
//  clock, and a legacy grace-stamped partition must not inflate that
//  anchor. Both directions are pinned below.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct WarmStartPlannerTests {

    // MARK: - Fixtures

    private let fingerprint = String(repeating: "a", count: 64)
    private let otherFingerprint = String(repeating: "b", count: 64)
    private let digestA = "aaaa1111"
    private let digestB = "bbbb2222"
    private var currentSchema: Int { SnapshotManifestSchema.currentVersion }

    private func meta(
        fingerprint fp: String? = nil,
        schemaVersion: Int? = nil,
        lastUsedAt: Double? = nil,
        modelID: String = "test-model"
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: modelID,
            modelFingerprint: fp ?? fingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100,
            schemaVersion: schemaVersion ?? SnapshotManifestSchema.currentVersion,
            lastUsedAt: lastUsedAt
        )
    }

    private func descriptor(
        id: String = UUID().uuidString,
        digest: String,
        bytes: Int = 1000,
        type: HybridCacheSnapshot.CheckpointType = .leaf,
        checkpointTypeOverride: String? = nil,
        schemaVersion: Int? = nil,
        inheritedSegments: [SnapshotSegment] = []
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: digest,
            pathFromRoot: [1, 2],
            tokenOffset: 2,
            checkpointType: checkpointTypeOverride ?? type.wireString,
            bytes: bytes,
            inheritedSegments: inheritedSegments,
            createdAt: 100,
            lastAccessAt: 0,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: id, partitionDigest: digest
            ),
            schemaVersion: schemaVersion ?? SnapshotManifestSchema.currentVersion
        )
    }

    private func manifest(
        partitions: [String: PartitionMeta],
        snapshots: [PersistedSnapshotDescriptor] = []
    ) -> SnapshotManifest {
        SnapshotManifest(
            schemaVersion: currentSchema,
            partitions: partitions,
            snapshots: Dictionary(uniqueKeysWithValues: snapshots.map { ($0.snapshotID, $0) })
        )
    }

    // MARK: - Anchor-relative staleness (Stale-Partition GC)

    /// An idle tier ages *together*: when every partition's stamp is old
    /// but they sit within the gap of one another, the freshest is the
    /// anchor and the whole tier survives — the wall clock racing far
    /// ahead of the stamps must never reclaim anything.
    @Test
    func idleTierAgesTogetherSoOldStampsSurvive() {
        let loaded = manifest(partitions: [
            digestA: meta(lastUsedAt: 1000),
            digestB: meta(lastUsedAt: 1500),  // freshest → anchor
        ])
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 10_000_000,  // wall clock ages ahead — must not bite
            maxUnusedAge: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions.keys.sorted() == [digestA, digestB])
        #expect(plan.outcome.invalidated.isEmpty)
    }

    /// An abandoned kv-config / template variant of a *still-active*
    /// model (same fingerprint, different digest) ages out: its stamp
    /// trails the active sibling's anchor by more than the gap.
    @Test
    func abandonedVariantOfActiveModelAgesOut() {
        let active = 100_000.0  // anchor
        let loaded = manifest(
            partitions: [
                digestA: meta(lastUsedAt: active),
                digestB: meta(lastUsedAt: active - 2000),  // beyond the gap
            ],
            snapshots: [descriptor(digest: digestB, bytes: 777)]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: active,
            maxUnusedAge: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions.keys.sorted() == [digestA])
        #expect(plan.outcome.invalidated.map(\.digest) == [digestB])
        #expect(plan.outcome.invalidated.first?.reason == .staleUnused)
        // The reclaim reports the bytes it returned to the budget.
        #expect(plan.outcome.invalidated.first?.bytes == 777)
        // The whole partition directory is swept; the dangling descriptor's
        // files live under it and are not listed separately.
        #expect(
            plan.deletions == [SnapshotDiskLayout.partitionDirectory(digest: digestB)]
        )
        #expect(plan.seedBytes == 0)
        #expect(plan.persistNeeded == true)  // a reclaim must reach disk
    }

    /// Warm start itself is never "use": a kept partition's `lastUsedAt`
    /// survives verbatim — it is not refreshed to `now`.
    @Test
    func warmStartNeverRefreshesAKeptStamp() {
        let stamp = 500_000.0
        let loaded = manifest(partitions: [digestA: meta(lastUsedAt: stamp)])
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 9_999_999,
            maxUnusedAge: SSDStalePartitionPolicy.maxUnusedAge,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions[digestA]?.lastUsedAt == stamp)
        #expect(plan.persistNeeded == false)  // nothing mutated, nothing reclaimed
    }

    /// The anchor-inflation guard: a legacy nil-stamped partition is
    /// grace-stamped to `now` and kept, but it must NOT count toward the
    /// anchor — otherwise it would pull the anchor to `now` and reclaim a
    /// genuinely-stamped, older sibling at the migration launch.
    @Test
    func graceStampDoesNotInflateAnchorAndReclaimSibling() {
        let loaded = manifest(partitions: [
            digestA: meta(lastUsedAt: nil),  // legacy — grace-stamped below
            digestB: meta(lastUsedAt: 1000),  // genuinely stamped, old
        ])
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 10_000,  // if the legacy stamp inflated the anchor, B would age out
            maxUnusedAge: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions.keys.sorted() == [digestA, digestB])
        #expect(plan.outcome.invalidated.isEmpty)  // sibling NOT reclaimed
        #expect(plan.manifest.partitions[digestA]?.lastUsedAt == 10_000)  // grace-stamped
        #expect(plan.manifest.partitions[digestB]?.lastUsedAt == 1000)  // untouched
        #expect(plan.persistNeeded == true)  // the grace stamp must reach disk
    }

    // MARK: - Fingerprint / schema / checkpoint-type filters

    @Test
    func fingerprintMismatchPartitionIsInvalidatedWithReason() {
        let desc = descriptor(digest: digestA, bytes: 1234)
        let loaded = manifest(
            partitions: [digestA: meta(fingerprint: otherFingerprint)],
            snapshots: [desc]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions.isEmpty)
        #expect(plan.outcome.invalidated.map(\.digest) == [digestA])
        #expect(plan.outcome.invalidated.first?.reason == .fingerprintChanged)
        #expect(plan.outcome.invalidated.first?.bytes == 1234)
        #expect(plan.seedBytes == 0)
        #expect(plan.deletions == [SnapshotDiskLayout.partitionDirectory(digest: digestA)])
        #expect(plan.persistNeeded == true)
    }

    @Test
    func staleSchemaPartitionIsDroppedWithSchemaStaleReason() {
        let loaded = manifest(
            partitions: [digestA: meta(schemaVersion: currentSchema - 1)]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.partitions.isEmpty)
        #expect(plan.outcome.invalidated.map(\.reason) == [.schemaStale])
    }

    /// A descriptor stamped at a prior schema inside an otherwise
    /// current partition is dropped to a dead file — its bytes are not
    /// seeded, and its chain files are scheduled for deletion.
    @Test
    func staleSchemaDescriptorDropsToDeadFileNotSeeded() {
        let valid = descriptor(id: "v-1", digest: digestA, bytes: 1000)
        let stale = descriptor(
            id: "s-1", digest: digestA, bytes: 333, schemaVersion: currentSchema - 1
        )
        let loaded = manifest(
            partitions: [digestA: meta(lastUsedAt: 500)],
            snapshots: [valid, stale]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.snapshots.keys.sorted() == ["v-1"])
        #expect(plan.seedBytes == 1000)
        #expect(plan.deletions == stale.chainFileRelativePaths)
        #expect(plan.outcome.invalidated.isEmpty)  // a dead descriptor is not a partition reclaim
    }

    /// An unknown wire checkpoint type is dropped to a dead file so its
    /// bytes never leak into the budget. Dead-file cleanup alone does not
    /// force a persist — the behavior the ledger preserves today.
    @Test
    func unknownCheckpointTypeDescriptorDropsToDeadFileNotSeeded() {
        let valid = descriptor(id: "v-1", digest: digestA, bytes: 1000)
        let unknown = descriptor(
            id: "u-1", digest: digestA, bytes: 777, checkpointTypeOverride: "no-such-type"
        )
        let loaded = manifest(
            partitions: [digestA: meta(lastUsedAt: 500)],
            snapshots: [valid, unknown]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 1000,
            persistManifestAfter: false
        )
        #expect(plan.manifest.snapshots.keys.sorted() == ["v-1"])
        #expect(plan.seedBytes == 1000)
        #expect(plan.deletions == unknown.chainFileRelativePaths)
        #expect(plan.persistNeeded == false)
    }

    // MARK: - Persist-choice derivation

    /// `persistManifestAfter` (the rebuild path) forces a persist even
    /// when nothing was mutated or reclaimed; the normal load path of the
    /// same clean manifest does not.
    @Test
    func persistNeededIsForcedByTheRebuildPath() {
        let loaded = manifest(partitions: [digestA: meta(lastUsedAt: 500)])
        let normal = WarmStartPlanner.plan(
            loaded: loaded, expectedFingerprint: fingerprint,
            now: 1000, persistManifestAfter: false
        )
        let rebuild = WarmStartPlanner.plan(
            loaded: loaded, expectedFingerprint: fingerprint,
            now: 1000, persistManifestAfter: true
        )
        #expect(normal.persistNeeded == false)
        #expect(rebuild.persistNeeded == true)
    }

    // MARK: - seedBytes fold

    /// `seedBytes` folds chain totals (own + inherited) of the kept
    /// descriptors only.
    @Test
    func seedBytesFoldsChainTotalsOfKeptDescriptors() {
        let seg = SnapshotSegment(
            baseOffset: 0,
            tokenOffset: 2,
            fileRelativePath: SnapshotDiskLayout.snapshotFile(
                digest: digestA, shard: "b", name: "base.safetensors"
            ),
            bytes: 400
        )
        let head = descriptor(id: "h-1", digest: digestA, bytes: 600, inheritedSegments: [seg])
        let solo = descriptor(id: "s-1", digest: digestA, bytes: 250)
        let loaded = manifest(
            partitions: [digestA: meta(lastUsedAt: 500)],
            snapshots: [head, solo]
        )
        let plan = WarmStartPlanner.plan(
            loaded: loaded,
            expectedFingerprint: fingerprint,
            now: 1000,
            persistManifestAfter: false
        )
        // head chain total = 600 own + 400 inherited; solo = 250.
        #expect(plan.seedBytes == 600 + 400 + 250)
    }
}
