//
//  StalePartitionGCTests.swift
//  tesseractTests
//
//  Stale-partition GC + invalidation visibility (PRD #150, phase 3).
//  Ledger-level: `lastUsedAt` stamping through `registerPartition` /
//  `recordHit`, the warm-start staleness cut, and the legacy-meta
//  grace stamp. Manager-level: the `ssdPartitionInvalidated`
//  diagnostics event a reclaim emits — the visibility fix for the
//  2026-07-04 silent invalidation.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct StalePartitionGCTests {

    // MARK: - Fixtures

    private let testFingerprint = String(repeating: "a", count: 64)
    private let otherFingerprint = String(repeating: "b", count: 64)
    private let testDigest = "abcd1234"

    private var now: Double { Date().timeIntervalSinceReferenceDate }

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("stale-gc-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makeLedger(root: URL) -> SnapshotLedger {
        SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20)
        )
    }

    private func makePartitionMeta(
        fingerprint: String? = nil,
        modelID: String = "test-model",
        createdAt: Double = 100_000,
        lastUsedAt: Double? = nil
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: modelID,
            modelFingerprint: fingerprint ?? testFingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: createdAt,
            schemaVersion: SnapshotManifestSchema.currentVersion,
            lastUsedAt: lastUsedAt
        )
    }

    private func makeDescriptor(
        id: String = UUID().uuidString,
        partition: String,
        bytes: Int = 1000
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: partition,
            pathFromRoot: [1, 2, 3],
            tokenOffset: 3,
            checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
            bytes: bytes,
            createdAt: 100_000,
            lastAccessAt: 100_000,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: id,
                partitionDigest: partition
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func writeManifest(_ manifest: SnapshotManifest, rootURL: URL) throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: rootURL.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func readManifest(rootURL: URL) throws -> SnapshotManifest {
        let data = try Data(contentsOf: rootURL.appendingPathComponent("manifest.json"))
        return try JSONDecoder().decode(SnapshotManifest.self, from: data)
    }

    // MARK: - PartitionMeta wire compatibility

    /// A pre-PRD-#150 `_meta.json` (no `lastUsedAt` key) still decodes —
    /// the field is optional-with-nil, no schema bump.
    @Test func partitionMetaDecodesWithoutLastUsedAt() throws {
        let legacyJSON = """
            {"modelID":"m","modelFingerprint":"f","kvGroupSize":64,
             "createdAt":1,"schemaVersion":\(SnapshotManifestSchema.currentVersion)}
            """
        let meta = try JSONDecoder().decode(
            PartitionMeta.self, from: Data(legacyJSON.utf8)
        )
        #expect(meta.lastUsedAt == nil)
    }

    @Test func partitionMetaRoundTripsLastUsedAt() throws {
        let meta = makePartitionMeta(lastUsedAt: 123_456)
        let data = try JSONEncoder().encode(meta)
        let decoded = try JSONDecoder().decode(PartitionMeta.self, from: data)
        #expect(decoded.lastUsedAt == 123_456)
    }

    // MARK: - registerPartition use stamping

    /// A fresh registration (no prior entry) stamps `lastUsedAt` to now
    /// while keeping the caller's `createdAt`.
    @Test func registerStampsLastUsedAtOnFreshMeta() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)
        let before = now

        ledger.registerPartition(makePartitionMeta(), digest: testDigest)

        let stored = ledger.residency().partitionMeta(digest: testDigest)
        #expect(stored?.createdAt == 100_000)
        #expect((stored?.lastUsedAt ?? 0) >= before)
    }

    /// A same-identity re-registration (the admission path minting a
    /// fresh meta with `createdAt = now`) preserves the stored
    /// `createdAt` and bumps only `lastUsedAt`.
    @Test func registerMergePreservesCreatedAtAndBumpsLastUsedAt() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)
        let stale = now - 7 * 24 * 3600

        ledger.registerPartition(
            makePartitionMeta(createdAt: 100_000, lastUsedAt: stale),
            digest: testDigest
        )
        let before = now
        ledger.registerPartition(
            makePartitionMeta(createdAt: now),
            digest: testDigest
        )

        let stored = ledger.residency().partitionMeta(digest: testDigest)
        #expect(stored?.createdAt == 100_000)
        #expect((stored?.lastUsedAt ?? 0) >= before)
    }

    /// Re-stamps are throttled: a `lastUsedAt` fresher than the refresh
    /// interval is left alone, so per-admission registration does not
    /// become a sidecar write storm.
    @Test func registerThrottlesFreshRestamp() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)
        let recent = now - 60

        ledger.registerPartition(
            makePartitionMeta(lastUsedAt: recent),
            digest: testDigest
        )
        ledger.registerPartition(
            makePartitionMeta(createdAt: now),
            digest: testDigest
        )

        let stored = ledger.residency().partitionMeta(digest: testDigest)
        #expect(stored?.lastUsedAt == recent)
    }

    /// A value-equal re-registration (warm start replaying the persisted
    /// meta) is a no-op: a warm start alone is not "use".
    @Test func registerValueEqualMetaDoesNotBump() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)
        let stale = now - 7 * 24 * 3600
        let meta = makePartitionMeta(lastUsedAt: stale)

        ledger.registerPartition(meta, digest: testDigest)
        ledger.registerPartition(meta, digest: testDigest)

        #expect(ledger.residency().partitionMeta(digest: testDigest)?.lastUsedAt == stale)
    }

    // MARK: - recordHit use stamping

    /// An SSD hit refreshes the owning partition's use stamp — a
    /// read-heavy partition must not age out because nothing wrote to it.
    @Test func recordHitBumpsPartitionLastUsed() {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let ledger = makeLedger(root: root)
        let stale = now - 7 * 24 * 3600
        ledger.registerPartition(
            makePartitionMeta(lastUsedAt: stale),
            digest: testDigest
        )
        let descriptor = makeDescriptor(partition: testDigest)
        ledger.seedDescriptorForTesting(descriptor)

        let before = now
        ledger.recordHit(id: descriptor.snapshotID)

        #expect(
            (ledger.residency().partitionMeta(digest: testDigest)?.lastUsedAt ?? 0) >= before
        )
    }

    // MARK: - Warm-start staleness cut

    /// A partition whose use stamp sits more than
    /// `SSDStalePartitionPolicy.maxUnusedAge` behind the tier's freshest
    /// is reclaimed at warm start: descriptors leave the manifest, its
    /// bytes return to the budget, and the outcome reports reason +
    /// bytes. The fresh partition under the same fingerprint survives.
    @Test func warmStartReclaimsStalePartition() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let staleDigest = "deadbeef"
        let freshDigest = "cafebabe"
        let freshUse = now - 24 * 3600

        var manifest = SnapshotManifest.empty()
        manifest.partitions[staleDigest] = makePartitionMeta(
            lastUsedAt: freshUse - SSDStalePartitionPolicy.maxUnusedAge - 24 * 3600
        )
        manifest.partitions[freshDigest] = makePartitionMeta(
            lastUsedAt: freshUse
        )
        let staleDescriptor = makeDescriptor(partition: staleDigest, bytes: 3000)
        let freshDescriptor = makeDescriptor(partition: freshDigest, bytes: 5000)
        manifest.snapshots[staleDescriptor.snapshotID] = staleDescriptor
        manifest.snapshots[freshDescriptor.snapshotID] = freshDescriptor
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.map(\.digest) == [freshDigest])
        #expect(outcome.invalidated.count == 1)
        #expect(outcome.invalidated.first?.digest == staleDigest)
        #expect(outcome.invalidated.first?.reason == .staleUnused)
        #expect(outcome.invalidated.first?.bytes == 3000)
        #expect(outcome.invalidated.first?.modelID == "test-model")
        #expect(ledger.residency().bytes == 5000)
    }

    /// Staleness is relative to the tier's freshest use, not the wall
    /// clock: a tier that simply sat idle (every stamp old, but aged
    /// *together*) is never reclaimed — coming back from a long break
    /// must not cost the whole cache.
    @Test func warmStartKeepsIdleTierIntact() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let digestA = "deadbeef"
        let digestB = "cafebabe"
        let idleAnchor = now - 60 * 24 * 3600

        var manifest = SnapshotManifest.empty()
        manifest.partitions[digestA] = makePartitionMeta(lastUsedAt: idleAnchor)
        manifest.partitions[digestB] = makePartitionMeta(
            lastUsedAt: idleAnchor - SSDStalePartitionPolicy.maxUnusedAge + 3600
        )
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.count == 2)
        #expect(outcome.invalidated.isEmpty)
    }

    /// A mixed tier — one partition with a real (old) stamp, one legacy
    /// nil-stamped sibling — must not reclaim the stamped partition at
    /// the migration launch: the sibling's grace stamp is "the clock
    /// starts here", not evidence of use, so it cannot inflate the
    /// staleness anchor (the code-review catch on `lastUsedAt ?? now`).
    @Test func legacySiblingDoesNotInflateTheStalenessAnchor() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let stampedDigest = "deadbeef"
        let legacyDigest = "cafebabe"

        var manifest = SnapshotManifest.empty()
        manifest.partitions[stampedDigest] = makePartitionMeta(
            lastUsedAt: now - SSDStalePartitionPolicy.maxUnusedAge - 3 * 24 * 3600
        )
        manifest.partitions[legacyDigest] = makePartitionMeta(lastUsedAt: nil)
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.invalidated.isEmpty)
        #expect(Set(outcome.validPartitions.map(\.digest)) == [stampedDigest, legacyDigest])
    }

    /// A legacy meta without a `lastUsedAt` stamp survives warm start
    /// (no retroactive reclaim), is grace-stamped to now, and the stamp
    /// is persisted — otherwise the GC clock would restart every launch.
    @Test func warmStartGraceStampsLegacyMeta() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        var manifest = SnapshotManifest.empty()
        manifest.partitions[testDigest] = makePartitionMeta(lastUsedAt: nil)
        let descriptor = makeDescriptor(partition: testDigest, bytes: 2000)
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let before = now
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.map(\.digest) == [testDigest])
        #expect(outcome.invalidated.isEmpty)
        let stamped = ledger.residency().partitionMeta(digest: testDigest)?.lastUsedAt
        #expect((stamped ?? 0) >= before)

        ledger.persistNow()
        let persisted = try readManifest(rootURL: root)
        #expect(persisted.partitions[testDigest]?.lastUsedAt == stamped)
    }

    /// Fingerprint invalidation carries the reason and the reclaimed
    /// byte count on the outcome — the inputs the visible panel event
    /// is built from.
    @Test func warmStartFingerprintInvalidationCarriesReasonAndBytes() throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        var manifest = SnapshotManifest.empty()
        manifest.partitions[testDigest] = makePartitionMeta(
            fingerprint: otherFingerprint,
            modelID: "other-model",
            lastUsedAt: now - 60
        )
        let descriptor = makeDescriptor(partition: testDigest, bytes: 7000)
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let ledger = makeLedger(root: root)
        let outcome = ledger.seedFromWarmStart(expectedFingerprint: testFingerprint)

        #expect(outcome.validPartitions.isEmpty)
        #expect(outcome.invalidated.count == 1)
        #expect(outcome.invalidated.first?.reason == .fingerprintChanged)
        #expect(outcome.invalidated.first?.bytes == 7000)
        #expect(outcome.invalidated.first?.modelID == "other-model")
        #expect(ledger.residency().bytes == 0)
    }

    // MARK: - Manager-level event emission

    private final class DiagnosticsSink: @unchecked Sendable {
        private let lock = NSLock()
        private var _lines: [String] = []

        func lines(matching keyword: String) -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return _lines.filter { $0.contains(keyword) }
        }

        var handler: @Sendable (String) -> Void {
            { [weak self] line in
                guard let self else { return }
                self.lock.lock()
                self._lines.append(line)
                self.lock.unlock()
            }
        }
    }

    /// `PrefixCacheManager.warmStart` renders each reclaimed partition
    /// as an `ssdPartitionInvalidated` event (reason + modelID + bytes)
    /// and keeps the legacy `fingerprintMismatch` line for the
    /// fingerprint reason only — the stale-GC reclaim must not
    /// masquerade as a fingerprint problem.
    @MainActor
    @Test func warmStartEmitsPartitionInvalidatedEvents() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }
        let sink = DiagnosticsSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }

        let staleDigest = "deadbeef"
        let mismatchDigest = "facefeed"
        let freshDigest = "cafebabe"
        var manifest = SnapshotManifest.empty()
        // The fresh partition anchors the tier's most-recent-use clock,
        // making `staleDigest` reclaimable under the relative rule.
        manifest.partitions[freshDigest] = makePartitionMeta(lastUsedAt: now - 60)
        manifest.partitions[staleDigest] = makePartitionMeta(
            modelID: "stale-model",
            lastUsedAt: now - SSDStalePartitionPolicy.maxUnusedAge - 24 * 3600
        )
        manifest.partitions[mismatchDigest] = makePartitionMeta(
            fingerprint: otherFingerprint,
            modelID: "changed-model",
            lastUsedAt: now - 60
        )
        manifest.snapshots = [:]
        try writeManifest(manifest, rootURL: root)

        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: 1_000_000,
            maxPendingBytes: 1_000_000
        )
        let tieredStore = TieredSnapshotStore(ssdConfig: config)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: 100_000_000,
            tieredStore: tieredStore
        )
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        let staleLines = sink.lines(matching: "event=ssdPartitionInvalidated")
            .filter { $0.contains("digest=\(staleDigest)") }
        #expect(staleLines.count == 1)
        #expect(staleLines.first?.contains("reason=staleUnused") == true)
        #expect(staleLines.first?.contains("modelID=stale-model") == true)

        let mismatchLines = sink.lines(matching: "event=ssdPartitionInvalidated")
            .filter { $0.contains("digest=\(mismatchDigest)") }
        #expect(mismatchLines.count == 1)
        #expect(mismatchLines.first?.contains("reason=fingerprintChanged") == true)

        // Legacy line: fingerprint reason only.
        let legacyLines = sink.lines(matching: "event=fingerprintMismatch")
        #expect(legacyLines.contains { $0.contains("partition=\(mismatchDigest)") })
        #expect(!legacyLines.contains { $0.contains("partition=\(staleDigest)") })
    }
}
