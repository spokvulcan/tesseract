//
//  SnapshotLedgerTests.swift
//  tesseractTests
//
//  Unit tests for the **Snapshot Ledger** — the in-memory authority
//  over the SSD prefix-cache tier carved out of `SSDSnapshotStore`.
//  These construct the ledger standalone against a temp root, with no
//  writer task, wakeup stream, or commit/drop callbacks, and assert
//  external behaviour only: on-disk state + a fingerprint in, manifest /
//  outcome out; a sequence of `admit` / `commit` / `recordHit` /
//  `remove` / `removeOrTombstone` in, resident set + evictions +
//  tombstone effects out. Lock internals and private field shapes are
//  never asserted.
//
//  The recovery / LRU / byte-accounting / tombstone cases mirror the
//  corresponding `SSDSnapshotStoreTests` and `WarmStartTests` cases but
//  shed the writer scaffold — that shedding is the whole point of the
//  carve.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

// Large test suite — splitting deferred (evolving MVP, see CLAUDE.md).
// swiftlint:disable:next type_body_length
struct SnapshotLedgerTests {

    // MARK: - Fixtures

    private let testFingerprint = String(repeating: "a", count: 64)
    private let testDigest = "abcd1234"

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ledger-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makeLedger(
        budgetBytes: Int = 1_000_000,
        root: URL
    ) -> SnapshotLedger {
        SnapshotLedger(
            rootURL: root,
            budgetBytes: budgetBytes,
            manifestDebounce: .milliseconds(20)
        )
    }

    private func makePartitionMeta(
        fingerprint: String? = nil,
        schemaVersion: Int = SnapshotManifestSchema.currentVersion
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: "test-model",
            modelFingerprint: fingerprint ?? testFingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: schemaVersion
        )
    }

    private func makeDescriptor(
        id: String = UUID().uuidString,
        partition: String? = nil,
        type: HybridCacheSnapshot.CheckpointType = .leaf,
        bytes: Int = 1000,
        lastAccessAt: Double = 0,
        pathFromRoot: [Int] = [1, 2, 3],
        tokenOffset: Int? = nil,
        inheritedSegments: [SnapshotSegment] = [],
        schemaVersion: Int = SnapshotManifestSchema.currentVersion,
        checkpointTypeOverride: String? = nil
    ) -> PersistedSnapshotDescriptor {
        let digest = partition ?? testDigest
        return PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: digest,
            pathFromRoot: pathFromRoot,
            tokenOffset: tokenOffset ?? pathFromRoot.count,
            checkpointType: checkpointTypeOverride ?? type.wireString,
            bytes: bytes,
            inheritedSegments: inheritedSegments,
            createdAt: 100_000,
            lastAccessAt: lastAccessAt,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: id,
                partitionDigest: digest
            ),
            schemaVersion: schemaVersion
        )
    }

    /// A ledger with the default test partition registered, so `commit`
    /// / `seedDescriptorForTesting` satisfy the manifest invariant.
    private func makeLedgerWithPartition(
        budgetBytes: Int = 1_000_000,
        root: URL
    ) -> SnapshotLedger {
        let ledger = makeLedger(budgetBytes: budgetBytes, root: root)
        ledger.registerPartition(makePartitionMeta(), digest: testDigest)
        return ledger
    }

    // MARK: - On-disk fixtures (no writer scaffold)

    private func writeManifest(_ manifest: SnapshotManifest, rootURL: URL) throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: rootURL.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func writeCorruptManifest(rootURL: URL) throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try Data("{not-valid-json".utf8)
            .write(to: rootURL.appendingPathComponent("manifest.json"))
    }

    /// Write a partition's `_meta.json` sidecar the directory-walk
    /// rebuild reads for the fingerprint.
    private func writeMetaFile(_ meta: PartitionMeta, digest: String, rootURL: URL) throws {
        let dir =
            rootURL
            .appendingPathComponent("partitions")
            .appendingPathComponent(digest)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(meta)
        try data.write(to: dir.appendingPathComponent("_meta.json"))
    }

    /// Write a real placeholder-container `.safetensors` file at the
    /// descriptor's sharded path via the neutral `encodePlaceholderContainer`
    /// — the same bytes the store's writer would produce, but with no
    /// writer task. The header carries the descriptor the rebuild recovers.
    private func writeContainerFile(
        descriptor: PersistedSnapshotDescriptor,
        rootURL: URL
    ) throws {
        let payload = SnapshotPayload(
            tokenOffset: descriptor.tokenOffset,
            checkpointType: HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType
            ) ?? .leaf,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: descriptor.bytes),
                            dtype: "bfloat16",
                            shape: [1, descriptor.bytes]
                        )
                    ],
                    metaState: ["meta"],
                    offset: descriptor.tokenOffset
                )
            ]
        )
        let data = try encodePlaceholderContainer(payload: payload, descriptor: descriptor)
        let fileURL = rootURL.appendingPathComponent(descriptor.fileRelativePath)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: fileURL)
    }

    // MARK: - Byte accounting

    @Test
    func byteBudgetTracksCommitAndRemove() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        #expect(ledger.residency().bytes == 0)

        let a = makeDescriptor(bytes: 1000)
        #expect(ledger.commit(a) == true)
        #expect(ledger.residency().bytes == 1000)

        let b = makeDescriptor(bytes: 2500)
        #expect(ledger.commit(b) == true)
        #expect(ledger.residency().bytes == 3500)
        #expect(ledger.residencyStats().snapshotCount == 2)

        let evicted = ledger.remove(id: a.snapshotID)
        #expect(evicted?.snapshotID == a.snapshotID)
        #expect(evicted?.fileURLs == [root.appendingPathComponent(a.fileRelativePath)])
        #expect(ledger.residency().bytes == 2500)
        #expect(ledger.residencyStats().snapshotCount == 1)

        // Removing an absent ID is a nil no-op that does not strand bytes.
        #expect(ledger.remove(id: "nonexistent") == nil)
        #expect(ledger.residency().bytes == 2500)
    }

    // MARK: - Recency ordering + recordHit

    @Test
    func recencyOrderFollowsLastAccessAndRecordHitBumps() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let a = makeDescriptor(bytes: 100, lastAccessAt: 1)
        let b = makeDescriptor(bytes: 100, lastAccessAt: 2)
        let c = makeDescriptor(bytes: 100, lastAccessAt: 3)
        [a, b, c].forEach { ledger.seedDescriptorForTesting($0) }

        #expect(
            ledger.residency().idsByRecency
                == [a.snapshotID, b.snapshotID, c.snapshotID]
        )

        let before = ledger.residency().lastAccessAt(id: a.snapshotID)
        ledger.recordHit(id: a.snapshotID)
        let after = ledger.residency().lastAccessAt(id: a.snapshotID)
        #expect(after > before)

        // The bumped entry is now the most-recent, so it sorts last.
        #expect(
            ledger.residency().idsByRecency
                == [b.snapshotID, c.snapshotID, a.snapshotID]
        )
    }

    @Test
    func recordHitForUnknownIDIsNoOp() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        ledger.recordHit(id: "nonexistent")
        #expect(ledger.residencyStats().snapshotCount == 0)
    }

    // MARK: - Type-protected LRU admission cut

    @Test
    func admitEvictsOldestNonSystemFirstAndProtectsSystem() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 3000, root: root)

        let system = makeDescriptor(type: .system, bytes: 1000, lastAccessAt: 1)
        let leaf = makeDescriptor(type: .leaf, bytes: 1000, lastAccessAt: 2)
        ledger.seedDescriptorForTesting(system)
        ledger.seedDescriptorForTesting(leaf)
        #expect(ledger.residency().bytes == 2000)

        // Incoming needs 1500; only 1000 free. Pass 1 evicts the oldest
        // non-system (the leaf); the system resident is protected.
        let incoming = makeDescriptor(type: .leaf, bytes: 1500)
        let (decision, evicted) = ledger.admit(incoming)

        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == [leaf.snapshotID])
        let resident = ledger.residency().idsByRecency
        #expect(resident.contains(system.snapshotID))
        #expect(!resident.contains(leaf.snapshotID))
        // `admit` only evicts; the incoming's bytes land at `commit`.
        #expect(ledger.residency().bytes == 1000)
    }

    @Test
    func admitDropsNonSystemIncomingWhenOnlySystemResidentsRemain() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 2000, root: root)

        let system = makeDescriptor(type: .system, bytes: 1500, lastAccessAt: 1)
        ledger.seedDescriptorForTesting(system)

        let incoming = makeDescriptor(type: .leaf, bytes: 1000)
        let (decision, evicted) = ledger.admit(incoming)

        #expect(decision == .drop(.systemProtectionWins))
        #expect(evicted.isEmpty)
        #expect(ledger.residency().bytes == 1500)
    }

    @Test
    func admitLaterallyEvictsOldestSystemForSystemIncoming() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 2000, root: root)

        let oldSystem = makeDescriptor(type: .system, bytes: 1500, lastAccessAt: 1)
        ledger.seedDescriptorForTesting(oldSystem)

        let incoming = makeDescriptor(type: .system, bytes: 1000)
        let (decision, evicted) = ledger.admit(incoming)

        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == [oldSystem.snapshotID])
        #expect(ledger.residency().bytes == 0)
    }

    @Test
    func admitDropsExceedsBudgetWhenSystemPayloadLargerThanBudget() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 1000, root: root)

        // A single system payload bigger than the whole budget: no
        // eviction can create room, and it is system so it reaches the
        // pass-2 lateral branch that resolves to `.exceedsBudget`.
        let incoming = makeDescriptor(type: .system, bytes: 2000)
        let (decision, evicted) = ledger.admit(incoming)

        #expect(decision == .drop(.exceedsBudget))
        #expect(evicted.isEmpty)
    }

    @Test
    func admitUnderBudgetAdmitsWithoutEviction() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 10_000, root: root)
        ledger.seedDescriptorForTesting(makeDescriptor(type: .leaf, bytes: 1000, lastAccessAt: 1))

        let incoming = makeDescriptor(type: .leaf, bytes: 500)
        let (decision, evicted) = ledger.admit(incoming)

        #expect(decision == .admit)
        #expect(evicted.isEmpty)
    }

    @Test
    func retryAfterDiskFullEvictsOldestEligibleResident() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 10_000, root: root)

        let leaf1 = makeDescriptor(type: .leaf, bytes: 1000, lastAccessAt: 1)
        let leaf2 = makeDescriptor(type: .leaf, bytes: 1000, lastAccessAt: 2)
        ledger.seedDescriptorForTesting(leaf1)
        ledger.seedDescriptorForTesting(leaf2)

        let victim = ledger.retryAfterDiskFull(makeDescriptor(type: .leaf, bytes: 500))
        #expect(victim?.snapshotID == leaf1.snapshotID)
        #expect(ledger.residency().bytes == 1000)
    }

    // MARK: - In-flight-delete tombstone protocol

    @Test
    func removeOrTombstoneThenCommitVetoesTheInFlightWrite() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let d = makeDescriptor(bytes: 1000)
        // Delete before commit: not resident → tombstone, returns nil.
        #expect(ledger.removeOrTombstone(id: d.snapshotID) == nil)
        // The in-flight write's commit is vetoed by the tombstone.
        #expect(ledger.commit(d) == false)
        #expect(ledger.residencyStats().snapshotCount == 0)
        #expect(ledger.residency().bytes == 0)
    }

    @Test
    func removeOrTombstoneReturnsResidentWhenAlreadyCommitted() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let d = makeDescriptor(bytes: 1000)
        #expect(ledger.commit(d) == true)

        let evicted = ledger.removeOrTombstone(id: d.snapshotID)
        #expect(evicted?.snapshotID == d.snapshotID)
        #expect(evicted?.fileURLs == [root.appendingPathComponent(d.fileRelativePath)])
        #expect(ledger.residency().bytes == 0)
        #expect(ledger.residencyStats().snapshotCount == 0)
    }

    @Test
    func consumeTombstoneClearsItSoASubsequentCommitSucceeds() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)

        let id = UUID().uuidString
        _ = ledger.removeOrTombstone(id: id)

        // The writer's pre-write skip consumes the tombstone once.
        #expect(ledger.consumeTombstone(id: id) == true)
        #expect(ledger.consumeTombstone(id: id) == false)

        // With the tombstone consumed, a fresh write of the same ID is
        // no longer vetoed.
        let d = makeDescriptor(id: id, bytes: 100)
        #expect(ledger.commit(d) == true)
        #expect(ledger.residency().bytes == 100)
    }

    // MARK: - Partition registration + persistence

    @Test
    func registerPartitionMakesItQueryableAndWritesMetaSidecar() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)

        #expect(ledger.hasPartition(digest: testDigest) == false)
        #expect(ledger.partitionFingerprint(digest: testDigest) == nil)

        ledger.registerPartition(makePartitionMeta(), digest: testDigest)

        #expect(ledger.hasPartition(digest: testDigest) == true)
        #expect(ledger.partitionFingerprint(digest: testDigest) == testFingerprint)
        #expect(ledger.residencyStats().partitionCount == 1)

        // The `_meta.json` sidecar the directory-walk rebuild depends on
        // is written synchronously, carrying the registered fingerprint.
        let metaURL =
            root
            .appendingPathComponent("partitions")
            .appendingPathComponent(testDigest)
            .appendingPathComponent("_meta.json")
        let metaData = try Data(contentsOf: metaURL)
        let decoded = try JSONDecoder().decode(PartitionMeta.self, from: metaData)
        #expect(decoded.modelFingerprint == testFingerprint)
    }

    @Test
    func persistNowWritesManifestThatRoundTrips() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(root: root)
        let d = makeDescriptor(bytes: 4096)
        #expect(ledger.commit(d) == true)

        ledger.persistNow()

        // The committed descriptor and its partition are on disk in a
        // manifest that decodes back to the same residency.
        let manifestURL = root.appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let decoded = try JSONDecoder().decode(SnapshotManifest.self, from: data)
        #expect(decoded.partitions[testDigest] != nil)
        #expect(decoded.snapshots[d.snapshotID]?.bytes == 4096)
    }

    // MARK: - Warm-start recovery (on-disk → outcome, no writer scaffold)

    @Test
    func warmStartOnMissingManifestReturnsEmpty() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)

        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.isEmpty)
        #expect(outcome.invalidatedPartitionDigests.isEmpty)
        #expect(ledger.residency().bytes == 0)
    }

    @Test
    func warmStartRestoresValidManifestPartitionsDescriptorsAndBytes() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let d1 = makeDescriptor(id: "aaa-1", bytes: 1000)
        let d2 = makeDescriptor(id: "bbb-2", bytes: 2000)
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [testDigest: makePartitionMeta()],
            snapshots: [d1.snapshotID: d1, d2.snapshotID: d2]
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.count == 1)
        #expect(outcome.validPartitions.first?.digest == testDigest)
        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID)
                == [d1.snapshotID, d2.snapshotID]
        )
        #expect(ledger.residency().bytes == 3000)
        // Post-seed the partition is queryable through the live interface.
        #expect(ledger.hasPartition(digest: testDigest) == true)
        #expect(ledger.partitionFingerprint(digest: testDigest) == testFingerprint)
    }

    @Test
    func warmStartExcludesFingerprintMismatchPartitionAndReportsItInvalidated() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let otherFingerprint = String(repeating: "b", count: 64)
        let desc = makeDescriptor(bytes: 1000)
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [testDigest: makePartitionMeta(fingerprint: otherFingerprint)],
            snapshots: [desc.snapshotID: desc]
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        // The live model's fingerprint differs from the persisted one.
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.isEmpty)
        #expect(outcome.invalidatedPartitionDigests == [testDigest])
        #expect(ledger.residency().bytes == 0)
        #expect(ledger.hasPartition(digest: testDigest) == false)
    }

    @Test
    func warmStartDropsDanglingDescriptorWhosePartitionIsUnregistered() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // `dangling` points at a digest with no partition entry; `valid`
        // points at the registered one. Only `valid` survives, and only
        // its bytes are seeded — the dangling 500 are not stranded.
        let dangling = makeDescriptor(partition: "deadbeef", bytes: 500)
        let valid = makeDescriptor(partition: testDigest, bytes: 1000)
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [testDigest: makePartitionMeta()],
            snapshots: [dangling.snapshotID: dangling, valid.snapshotID: valid]
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.count == 1)
        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID)
                == [valid.snapshotID]
        )
        #expect(ledger.residency().bytes == 1000)
    }

    @Test
    func warmStartDropsUnknownCheckpointTypeDescriptorWithoutStrandingBytes() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // A wire-format checkpoint type that no longer decodes must be
        // dropped — otherwise `warmStart` skips it silently and its bytes
        // leak into the SSD budget forever.
        let valid = makeDescriptor(bytes: 1000)
        let unknown = makeDescriptor(bytes: 777, checkpointTypeOverride: "no-such-type")
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [testDigest: makePartitionMeta()],
            snapshots: [valid.snapshotID: valid, unknown.snapshotID: unknown]
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID)
                == [valid.snapshotID]
        )
        #expect(ledger.residency().bytes == 1000)
    }

    @Test
    func warmStartDropsStaleSchemaDescriptorWithoutStrandingBytes() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // A descriptor stamped at a prior schema version inside an
        // otherwise current-version manifest cannot be reattached safely;
        // it is dropped and its bytes are not seeded.
        let valid = makeDescriptor(bytes: 1000)
        let stale = makeDescriptor(
            bytes: 333,
            schemaVersion: SnapshotManifestSchema.currentVersion - 1
        )
        let manifest = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [testDigest: makePartitionMeta()],
            snapshots: [valid.snapshotID: valid, stale.snapshotID: stale]
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID)
                == [valid.snapshotID]
        )
        #expect(ledger.residency().bytes == 1000)
    }

    @Test
    func schemaMismatchedManifestIsBackedUpAndStartsFresh() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // A manifest written at schema v2 against the current (newer)
        // version: back it up under `manifest.v2.bak` and start empty.
        let desc = makeDescriptor(bytes: 1000, schemaVersion: 2)
        let staleManifest = SnapshotManifest(
            schemaVersion: 2,
            partitions: [testDigest: makePartitionMeta(schemaVersion: 2)],
            snapshots: [desc.snapshotID: desc]
        )
        try writeManifest(staleManifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.isEmpty)
        #expect(ledger.residency().bytes == 0)
        // The mismatched manifest is archived, not left to be re-read.
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: root.appendingPathComponent("manifest.v2.bak").path))
        #expect(!fm.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
    }

    @Test
    func corruptManifestRebuildsDescriptorsFromHeadersAndArchivesIt() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        // Stand up a partition sidecar + a real container file on disk,
        // then corrupt `manifest.json`. The rebuild walks the directory,
        // reads `_meta.json` for the fingerprint, and recovers the
        // descriptor from the container header — no writer scaffold.
        try writeMetaFile(makePartitionMeta(), digest: testDigest, rootURL: root)
        let descriptor = makeDescriptor(id: "c0ffee-1", bytes: 2048)
        try writeContainerFile(descriptor: descriptor, rootURL: root)
        try writeCorruptManifest(rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.count == 1)
        #expect(outcome.validPartitions.first?.digest == testDigest)
        #expect(
            outcome.validPartitions.first?.descriptors.map(\.snapshotID)
                == [descriptor.snapshotID]
        )
        #expect(ledger.residency().bytes == 2048)

        // The corrupt manifest was renamed aside for forensics rather
        // than left in place to fail decode again next start.
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        #expect(contents.contains { $0.hasPrefix("manifest.corrupt.") })
    }

    // MARK: - Descriptor schema factory (schema owned by the ledger)

    @Test
    func makeDescriptorStampsSchemaFromDomainInputs() {
        let key = CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint
        )
        let snapshot = HybridCacheSnapshot(
            tokenOffset: 1234,
            layers: [],
            checkpointType: .system,
            memoryBytes: 0,
            createdAt: ContinuousClock().now
        )

        let descriptor = SnapshotLedger.makeDescriptor(
            partitionKey: key,
            pathFromRoot: [7, 8, 9],
            snapshot: snapshot,
            payloadBytes: 4096
        )

        // The schema fields are derived from the domain inputs…
        #expect(descriptor.partitionDigest == key.partitionDigest)
        #expect(descriptor.pathFromRoot == [7, 8, 9])
        #expect(descriptor.tokenOffset == 1234)
        #expect(descriptor.checkpointType == HybridCacheSnapshot.CheckpointType.system.wireString)
        #expect(descriptor.bytes == 4096)
        #expect(descriptor.schemaVersion == SnapshotManifestSchema.currentVersion)
        // …and the sharded path is derived from the minted ID, so the
        // file lands where `loadSync` / the rebuild walk expect it.
        #expect(
            descriptor.fileRelativePath
                == PersistedSnapshotDescriptor.relativeFilePath(
                    snapshotID: descriptor.snapshotID,
                    partitionDigest: key.partitionDigest
                )
        )

        // Each call mints a fresh identity — two captures of the same
        // path are distinct on-disk snapshots, never a silent overwrite.
        let other = SnapshotLedger.makeDescriptor(
            partitionKey: key,
            pathFromRoot: [7, 8, 9],
            snapshot: snapshot,
            payloadBytes: 4096
        )
        #expect(descriptor.snapshotID != other.snapshotID)
    }

    // MARK: - Terminal-loss cut scoring (PRD #82 slice #89)

    /// At `α = 0` (the construction default) the cut is byte-identical
    /// to the pre-ADR-0011 type-protected LRU: the stalest non-system
    /// resident goes first, no matter how tall its chain is.
    @Test
    func cutAtAlphaZeroIsPlainLRU() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 2_500, root: root)
        let now = Date().timeIntervalSinceReferenceDate

        let tallStale = makeDescriptor(
            id: "tall-stale", bytes: 1_000,
            lastAccessAt: now - 3_600, tokenOffset: 50_000
        )
        let shortFresh = makeDescriptor(
            id: "short-fresh", bytes: 1_000,
            lastAccessAt: now - 1, tokenOffset: 100
        )
        ledger.seedDescriptorForTesting(tallStale)
        ledger.seedDescriptorForTesting(shortFresh)

        let (decision, evicted) = ledger.admit(makeDescriptor(bytes: 1_000))
        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == ["tall-stale"])
    }

    /// At `α > 0` terminal-loss utility takes over: the stale chain
    /// whose re-prefill seconds per byte dwarf the fresh short one's is
    /// shielded, and the cut victimizes the cheap-to-recreate resident
    /// instead — the exact inversion of the LRU pick above.
    @Test
    func cutAtPositiveAlphaShieldsExpensiveChains() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 2_500, root: root)
        let now = Date().timeIntervalSinceReferenceDate

        let tallStale = makeDescriptor(
            id: "tall-stale", bytes: 1_000,
            lastAccessAt: now - 3_600, tokenOffset: 50_000
        )
        let shortFresh = makeDescriptor(
            id: "short-fresh", bytes: 1_000,
            lastAccessAt: now - 1, tokenOffset: 100
        )
        ledger.seedDescriptorForTesting(tallStale)
        ledger.seedDescriptorForTesting(shortFresh)

        let (decision, evicted) = ledger.admit(
            makeDescriptor(bytes: 1_000),
            scoring: EvictionConfiguration(alpha: 2.0)
        )
        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == ["short-fresh"])
        #expect(ledger.residency().descriptor(id: "tall-stale") != nil)
    }

    /// The density denominator is the **Segment Chain** total, never
    /// the head's own file. Two residents with identical offsets and
    /// recency: the chain head whose inherited segments make its total
    /// large has the *lower* re-prefill density and is the victim —
    /// scoring the own-file bytes instead would invert the pick.
    @Test
    func cutScoresChainTotalsNotOwnSegmentBytes() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 14_500, root: root)
        let now = Date().timeIntervalSinceReferenceDate

        let singleFile = makeDescriptor(
            id: "single-file", bytes: 4_000,
            lastAccessAt: now - 60, tokenOffset: 8_000
        )
        let chainHead = makeDescriptor(
            id: "chain-head", bytes: 1_000,
            lastAccessAt: now - 60, tokenOffset: 8_000,
            inheritedSegments: [
                SnapshotSegment(
                    baseOffset: 0,
                    tokenOffset: 6_000,
                    fileRelativePath: "partitions/\(testDigest)/snapshots/0/base.safetensors",
                    bytes: 9_000
                )
            ]
        )
        ledger.seedDescriptorForTesting(singleFile)
        ledger.seedDescriptorForTesting(chainHead)

        let (decision, evicted) = ledger.admit(
            makeDescriptor(bytes: 1_000),
            scoring: EvictionConfiguration(alpha: 2.0)
        )
        #expect(decision == .admit)
        #expect(evicted.map(\.snapshotID) == ["chain-head"])
    }

    /// Hard protection is score-proof: a `.system` resident with the
    /// worst conceivable utility (ancient AND cheap per byte) still
    /// wins against a non-system incoming at `α > 0` — the incoming
    /// drops.
    @Test
    func systemChainsAreNeverCutRegardlessOfScore() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedgerWithPartition(budgetBytes: 1_500, root: root)
        let now = Date().timeIntervalSinceReferenceDate

        ledger.seedDescriptorForTesting(
            makeDescriptor(
                id: "system-chain", type: .system, bytes: 1_000,
                lastAccessAt: now - 86_400, tokenOffset: 100
            ))

        let (decision, evicted) = ledger.admit(
            makeDescriptor(bytes: 1_000, tokenOffset: 50_000),
            scoring: EvictionConfiguration(alpha: 5.0)
        )
        #expect(decision == .drop(.systemProtectionWins))
        #expect(evicted.isEmpty)
        #expect(ledger.residency().descriptor(id: "system-chain") != nil)
    }
}
