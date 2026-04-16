//
//  WarmStartTests.swift
//  tesseractTests
//
//  PrefixCacheManager.warmStart + SSDSnapshotStore.warmStartLoad.
//  Verifies manifest-only restart: a fresh store + manager pair reads
//  `manifest.json` from disk, reattaches `storageRef`-only nodes to
//  the radix tree, and seeds `currentSSDBytes` from the valid
//  descriptor set. Fingerprint mismatch partitions are excluded from
//  both the tree and the byte count.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

@MainActor
struct WarmStartTests {

    // MARK: - Fixtures

    private let testFingerprint = String(repeating: "a", count: 64)
    private let otherFingerprint = String(repeating: "b", count: 64)

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("warmstart-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makePartitionMeta(
        fingerprint: String,
        modelID: String = "test-model"
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: modelID,
            modelFingerprint: fingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func makePartitionKey(
        fingerprint: String,
        modelID: String = "test-model"
    ) -> CachePartitionKey {
        CachePartitionKey(
            modelID: modelID,
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: fingerprint
        )
    }

    private func makeDescriptor(
        snapshotID: String = UUID().uuidString,
        partitionDigest: String,
        pathFromRoot: [Int],
        tokenOffset: Int,
        bytes: Int = 4096,
        checkpointType: HybridCacheSnapshot.CheckpointType = .leaf
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: snapshotID,
            partitionDigest: partitionDigest,
            pathFromRoot: pathFromRoot,
            tokenOffset: tokenOffset,
            checkpointType: checkpointType.wireString,
            bytes: bytes,
            createdAt: 100_000,
            lastAccessAt: 100_000,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: snapshotID,
                partitionDigest: partitionDigest
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// Write a manifest to disk at `rootURL/manifest.json`. Tests use
    /// this to seed the warm-start directory with a known starting
    /// state before the store reads it.
    private func writeManifest(_ manifest: SnapshotManifest, rootURL: URL) throws {
        try FileManager.default.createDirectory(
            at: rootURL,
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder().encode(manifest)
        let manifestURL = rootURL.appendingPathComponent("manifest.json")
        try data.write(to: manifestURL, options: .atomic)
    }

    private func makeManager(
        rootURL: URL,
        budgetBytes: Int = 100_000_000
    ) -> (PrefixCacheManager, SSDSnapshotStore) {
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: budgetBytes,
            maxPendingBytes: 10_000_000
        )
        let tieredStore = TieredSnapshotStore(ssdConfig: config)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: 100_000_000,
            tieredStore: tieredStore
        )
        return (mgr, tieredStore.ssdStore!)
    }

    // MARK: - Tests

    /// Warm start on a clean directory: no manifest, nothing to
    /// restore, `currentSSDBytes` starts at zero.
    @Test func warmStartOnEmptyDirectoryIsNoOp() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == 0)
        #expect(mgr.stats.snapshotCount == 0)
    }

    /// Warm start reads a manifest with three descriptors, reattaches
    /// each node as state 5, and seeds `currentSSDBytes` to the
    /// summed descriptor size. A subsequent lookup on each path
    /// surfaces `.ssdHit`.
    @Test func warmStartRestoresTreeAndSeedsBytes() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let digest = CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint
        ).partitionDigest

        let descriptors = [
            makeDescriptor(
                partitionDigest: digest,
                pathFromRoot: Array(1...10),
                tokenOffset: 10,
                bytes: 4096
            ),
            makeDescriptor(
                partitionDigest: digest,
                pathFromRoot: Array(1...20),
                tokenOffset: 20,
                bytes: 8192
            ),
            makeDescriptor(
                partitionDigest: digest,
                pathFromRoot: Array(1...30),
                tokenOffset: 30,
                bytes: 12_288
            ),
        ]
        let totalBytes = descriptors.reduce(0) { $0 + $1.bytes }

        var manifest = SnapshotManifest.empty()
        manifest.partitions[digest] = makePartitionMeta(fingerprint: testFingerprint)
        for d in descriptors {
            manifest.snapshots[d.snapshotID] = d
        }
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == totalBytes)

        let partitionKey = makePartitionKey(fingerprint: testFingerprint)
        for descriptor in descriptors {
            let result = mgr.lookup(
                tokens: descriptor.pathFromRoot,
                partitionKey: partitionKey
            )
            guard case .ssdHit(let ctx) = result.reason else {
                #expect(Bool(false), "Expected .ssdHit for \(descriptor.pathFromRoot.count) tokens, got \(result.reason)")
                continue
            }
            #expect(ctx.storageRef.snapshotID == descriptor.snapshotID)
            #expect(ctx.storageRef.tokenOffset == descriptor.tokenOffset)
        }
    }

    /// Warm start excludes partitions whose `modelFingerprint` does
    /// not match the caller's expected value. The invalidated
    /// partition's bytes must NOT count against `currentSSDBytes`,
    /// and its descriptors must NOT appear in the restored tree.
    @Test func warmStartExcludesFingerprintMismatchPartition() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let validDigest = CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint
        ).partitionDigest
        let invalidDigest = CachePartitionKey(
            modelID: "other-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: otherFingerprint
        ).partitionDigest

        let validDescriptor = makeDescriptor(
            partitionDigest: validDigest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )
        let invalidDescriptor = makeDescriptor(
            partitionDigest: invalidDigest,
            pathFromRoot: Array(100...110),
            tokenOffset: 10,
            bytes: 99_999
        )

        var manifest = SnapshotManifest.empty()
        manifest.partitions[validDigest] = makePartitionMeta(
            fingerprint: testFingerprint
        )
        manifest.partitions[invalidDigest] = makePartitionMeta(
            fingerprint: otherFingerprint,
            modelID: "other-model"
        )
        manifest.snapshots[validDescriptor.snapshotID] = validDescriptor
        manifest.snapshots[invalidDescriptor.snapshotID] = invalidDescriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == validDescriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)

        // The valid descriptor's path still reaches an SSD hit…
        let validKey = makePartitionKey(fingerprint: testFingerprint)
        let validResult = mgr.lookup(
            tokens: validDescriptor.pathFromRoot,
            partitionKey: validKey
        )
        if case .ssdHit = validResult.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected .ssdHit for valid path, got \(validResult.reason)")
        }

        // …while the invalidated partition is not registered at all.
        let invalidKey = makePartitionKey(
            fingerprint: otherFingerprint,
            modelID: "other-model"
        )
        let invalidResult = mgr.lookup(
            tokens: invalidDescriptor.pathFromRoot,
            partitionKey: invalidKey
        )
        if case .missNoEntries = invalidResult.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected missNoEntries for invalidated partition, got \(invalidResult.reason)")
        }
    }

    /// Warm start returns early when the SSD tier is disabled —
    /// `PrefixCacheManager` gracefully no-ops even though the caller
    /// passes a valid fingerprint.
    @Test func warmStartWithoutSSDTierIsNoOp() async throws {
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: 10_000_000,
            tieredStore: TieredSnapshotStore(ssdConfig: nil)
        )
        try await mgr.warmStart(modelFingerprint: testFingerprint)
        #expect(mgr.stats.partitionCount == 0)
    }

    /// An old manifest schema is a hard invalidation point: warm start
    /// archives `manifest.json`, removes stale partition directories,
    /// and starts from an empty cache.
    @Test func warmStartArchivesSchemaMismatchedManifestAndRemovesPartitions() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let digest = makePartitionKey(fingerprint: testFingerprint).partitionDigest
        let snapshotID = UUID().uuidString
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: snapshotID,
            partitionDigest: digest,
            pathFromRoot: [1, 2, 3],
            tokenOffset: 3,
            checkpointType: "lastMessageBoundary",
            bytes: 4096,
            createdAt: 100_000,
            lastAccessAt: 100_000,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: snapshotID,
                partitionDigest: digest
            ),
            schemaVersion: 2
        )
        let staleMeta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: testFingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: 2
        )
        let staleManifest = SnapshotManifest(
            schemaVersion: 2,
            partitions: [digest: staleMeta],
            snapshots: [descriptor.snapshotID: descriptor]
        )
        try writeManifest(staleManifest, rootURL: root)

        let stalePartitionDir = root
            .appendingPathComponent("partitions")
            .appendingPathComponent(digest)
        try FileManager.default.createDirectory(
            at: stalePartitionDir,
            withIntermediateDirectories: true
        )

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == 0)
        #expect(mgr.stats.partitionCount == 0)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.v2.bak").path))
        #expect(await waitUntil {
            !FileManager.default.fileExists(
                atPath: root.appendingPathComponent("partitions").path
            )
        })
    }

    /// A corrupt manifest with no on-disk snapshots still works:
    /// the file is renamed to `manifest.corrupt.{ts}.json`, the
    /// directory-walk rebuild runs against an empty `partitions/`
    /// directory, and warm start proceeds with zero bytes. Fresh
    /// admissions can then register partitions from scratch.
    @Test func warmStartHandlesCorruptManifestWithNoSnapshotsOnDisk() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let manifestURL = root.appendingPathComponent("manifest.json")
        try Data("{this-is-not-valid-json".utf8).write(to: manifestURL)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == 0)
        #expect(!FileManager.default.fileExists(atPath: manifestURL.path))
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        #expect(contents.contains { $0.hasPrefix("manifest.corrupt.") })
    }

    /// Corrupt `manifest.json` with real snapshot files on disk:
    /// warm start walks `partitions/*/_meta.json` + every
    /// `.safetensors` header, reconstructs the descriptor set,
    /// seeds `currentSSDBytes`, and persists a fresh `manifest.json`
    /// so the next restart does not re-walk.
    @Test func warmStartRebuildsFromDirectoryWalkAfterCorruption() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: 100_000_000,
            maxPendingBytes: 10_000_000
        )
        let partitionKey = makePartitionKey(fingerprint: testFingerprint)
        let digest = partitionKey.partitionDigest
        let meta = makePartitionMeta(fingerprint: testFingerprint)

        // Phase 1: drive the real writer path so the on-disk state
        // matches what production would produce — `_meta.json` +
        // `partitions/{digest}/snapshots/*/*.safetensors` + a
        // first-cut `manifest.json`.
        let tracker = RebuildCallbackTracker()
        let writerStore = SSDSnapshotStore(
            config: config,
            manifestDebounce: .milliseconds(10),
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )
        writerStore.registerPartition(meta, digest: digest)

        let payload = makeWriterPayload(bytes: 256)
        let descriptor = makeDescriptor(
            partitionDigest: digest,
            pathFromRoot: [1, 2, 3, 4, 5],
            tokenOffset: 5,
            bytes: 256
        )
        guard case .accepted = writerStore.tryEnqueue(
            payload: payload, descriptor: descriptor
        ) else {
            #expect(Bool(false), "tryEnqueue rejected")
            return
        }
        _ = await waitUntil { tracker.committed.contains(descriptor.snapshotID) }
        writerStore.flushManifestForTesting()

        // Phase 2: corrupt `manifest.json`, leaving snapshot files
        // and `_meta.json` intact.
        let manifestURL = root.appendingPathComponent("manifest.json")
        #expect(FileManager.default.fileExists(atPath: manifestURL.path))
        try Data("{not-valid-json".utf8).write(to: manifestURL)

        // Phase 3: warm start a NEW manager against the same
        // directory. It must rebuild the manifest from the headers.
        let (mgr, newStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(newStore.currentSSDBytesForTesting() == descriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)

        let lookup = mgr.lookup(
            tokens: descriptor.pathFromRoot,
            partitionKey: partitionKey
        )
        guard case .ssdHit(let ctx) = lookup.reason else {
            #expect(Bool(false), "Expected .ssdHit after rebuild, got \(lookup.reason)")
            return
        }
        #expect(ctx.storageRef.snapshotID == descriptor.snapshotID)
        #expect(ctx.storageRef.tokenOffset == descriptor.tokenOffset)
    }

    /// Descriptors whose wire-format `checkpointType` no longer
    /// decodes (older enum case removed, forward compat break) must
    /// be dropped before seeding `currentSSDBytes`. Otherwise the
    /// manager silently skips them on restore while their bytes leak
    /// into the budget forever.
    @Test func warmStartSkipsDescriptorsWithUnknownCheckpointType() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let partitionKey = makePartitionKey(fingerprint: testFingerprint)
        let digest = partitionKey.partitionDigest

        var bogus = SnapshotManifest.empty()
        bogus.partitions[digest] = makePartitionMeta(fingerprint: testFingerprint)
        let validDescriptor = makeDescriptor(
            partitionDigest: digest,
            pathFromRoot: Array(1...5),
            tokenOffset: 5,
            bytes: 1024,
            checkpointType: .leaf
        )
        bogus.snapshots[validDescriptor.snapshotID] = validDescriptor

        let deadID = UUID().uuidString
        bogus.snapshots[deadID] = PersistedSnapshotDescriptor(
            snapshotID: deadID,
            partitionDigest: digest,
            pathFromRoot: Array(10...15),
            tokenOffset: 5,
            checkpointType: "deprecated-future-type",
            bytes: 9_999_999,
            createdAt: 100_000,
            lastAccessAt: 100_000,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: deadID,
                partitionDigest: digest
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        try writeManifest(bogus, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        // Only the valid descriptor's bytes are seeded; the dead
        // one is excluded from both the budget and the restored tree.
        #expect(ssdStore.currentSSDBytesForTesting() == validDescriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)

        // The valid path still hits SSD.
        let validLookup = mgr.lookup(
            tokens: validDescriptor.pathFromRoot,
            partitionKey: partitionKey
        )
        if case .ssdHit = validLookup.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected .ssdHit for valid descriptor, got \(validLookup.reason)")
        }
    }

    // MARK: - TriAttention digest-mismatch invariant

    /// A partition written under a TriAttention digest must be dropped
    /// at warm start, not reattached under the reconstructed `.dense`
    /// key — otherwise a dense lookup would hydrate TriAttention state.
    @Test func warmStartDropsPartitionWhenOnDiskDigestMismatches() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let triAttentionKey = CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint,
            triAttention: .triAttention(
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "aaa"
                ),
                implementationVersion: .v1
            )
        )
        let triDigest = triAttentionKey.partitionDigest
        let denseKey = makePartitionKey(fingerprint: testFingerprint)
        #expect(triDigest != denseKey.partitionDigest)

        let descriptor = makeDescriptor(
            partitionDigest: triDigest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )

        var manifest = SnapshotManifest.empty()
        manifest.partitions[triDigest] = makePartitionMeta(fingerprint: testFingerprint)
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, _) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(mgr.stats.partitionCount == 0)

        let denseResult = mgr.lookup(
            tokens: descriptor.pathFromRoot,
            partitionKey: denseKey
        )
        #expect(denseResult.snapshot == nil)
        if case .hit = denseResult.reason {
            #expect(Bool(false), "dense lookup should not hit a TriAttention-digest partition")
        }
        if case .ssdHit = denseResult.reason {
            #expect(Bool(false), "dense lookup should not resolve an ssdHit against TriAttention state")
        }
    }

    /// v5 round-trip: a TriAttention partition whose `PartitionMeta`
    /// carries the matching identity reattaches under the same key
    /// rather than being dropped as a digest mismatch. Pinned because
    /// the dense-only gate that lived in `PrefixCacheManager.storeLeaf`
    /// for v4 was removed when v5 added the `triAttention` field —
    /// regressing this test would silently re-introduce the gate.
    @Test func warmStartReattachesTriAttentionPartitionWhenMetaCarriesIdentity() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let triIdentity: TriAttentionPartitionIdentity = .triAttention(
            budgetTokens: 12_000,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                rawValue: "aaa"
            ),
            implementationVersion: .v1
        )
        let triKey = CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: testFingerprint,
            triAttention: triIdentity
        )
        let triDigest = triKey.partitionDigest
        let triMeta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: testFingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: SnapshotManifestSchema.currentVersion,
            triAttention: triIdentity
        )

        let descriptor = makeDescriptor(
            partitionDigest: triDigest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )
        var manifest = SnapshotManifest.empty()
        manifest.partitions[triDigest] = triMeta
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == descriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)

        // Lookup under the reconstructed TriAttention key surfaces the
        // restored ref…
        let triLookup = mgr.lookup(
            tokens: descriptor.pathFromRoot,
            partitionKey: triKey
        )
        guard case .ssdHit(let ctx) = triLookup.reason else {
            #expect(Bool(false), "Expected .ssdHit for TriAttention key, got \(triLookup.reason)")
            return
        }
        #expect(ctx.storageRef.snapshotID == descriptor.snapshotID)

        // …while a dense lookup at the same path remains a miss because
        // the partition is isolated by digest.
        let denseLookup = mgr.lookup(
            tokens: descriptor.pathFromRoot,
            partitionKey: makePartitionKey(fingerprint: testFingerprint)
        )
        #expect(denseLookup.snapshot == nil)
        if case .ssdHit = denseLookup.reason {
            #expect(Bool(false), "Dense lookup must not resolve to a TriAttention ref")
        }
    }

    /// v5 wipe gate: a partition meta written under v4 (or any
    /// schema-version other than the current one) must be invalidated
    /// at warm-start, even if the top-level manifest version matches.
    /// Defends the descriptor-level filter added to
    /// `commitRestoredManifest` against accidental removal.
    @Test func warmStartDropsPartitionMetaWithStaleSchemaVersion() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let denseKey = makePartitionKey(fingerprint: testFingerprint)
        let digest = denseKey.partitionDigest
        // Manifest itself is v5, but the per-partition meta is v4.
        let staleMeta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: testFingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 100_000,
            schemaVersion: 4
        )
        let descriptor = makeDescriptor(
            partitionDigest: digest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )
        var manifest = SnapshotManifest.empty()
        manifest.partitions[digest] = staleMeta
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(mgr.stats.partitionCount == 0)
        #expect(ssdStore.currentSSDBytesForTesting() == 0)
    }

    /// Same wipe gate, but for a `PersistedSnapshotDescriptor` whose
    /// schemaVersion is stale even though its enclosing partition meta
    /// is current. Pinned because the descriptor-level filter is the
    /// only safeguard against a hand-edited manifest mixing v4 and v5
    /// descriptors under a current partition.
    @Test func warmStartDropsDescriptorWithStaleSchemaVersion() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let denseKey = makePartitionKey(fingerprint: testFingerprint)
        let digest = denseKey.partitionDigest

        let validDescriptor = makeDescriptor(
            partitionDigest: digest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )
        let staleID = UUID().uuidString
        let staleDescriptor = PersistedSnapshotDescriptor(
            snapshotID: staleID,
            partitionDigest: digest,
            pathFromRoot: Array(20...25),
            tokenOffset: 6,
            checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
            bytes: 9_999_999,
            createdAt: 100_000,
            lastAccessAt: 100_000,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: staleID,
                partitionDigest: digest
            ),
            schemaVersion: 4
        )

        var manifest = SnapshotManifest.empty()
        manifest.partitions[digest] = makePartitionMeta(fingerprint: testFingerprint)
        manifest.snapshots[validDescriptor.snapshotID] = validDescriptor
        manifest.snapshots[staleDescriptor.snapshotID] = staleDescriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        // Only the v5 descriptor's bytes are seeded; the stale-schema
        // entry is dropped before the budget is summed.
        #expect(ssdStore.currentSSDBytesForTesting() == validDescriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)
    }

    @Test func warmStartAcceptsDenseDigestMatch() async throws {
        let root = makeScratchDir()
        defer { cleanup(root) }

        let denseKey = makePartitionKey(fingerprint: testFingerprint)
        let digest = denseKey.partitionDigest
        let descriptor = makeDescriptor(
            partitionDigest: digest,
            pathFromRoot: Array(1...10),
            tokenOffset: 10,
            bytes: 4096
        )

        var manifest = SnapshotManifest.empty()
        manifest.partitions[digest] = makePartitionMeta(fingerprint: testFingerprint)
        manifest.snapshots[descriptor.snapshotID] = descriptor
        try writeManifest(manifest, rootURL: root)

        let (mgr, ssdStore) = makeManager(rootURL: root)
        try await mgr.warmStart(modelFingerprint: testFingerprint)

        #expect(ssdStore.currentSSDBytesForTesting() == descriptor.bytes)
        #expect(mgr.stats.partitionCount == 1)
    }

    // MARK: - Rebuild-path fixtures

    private func makeWriterPayload(bytes: Int) -> SnapshotPayload {
        SnapshotPayload(
            tokenOffset: 5,
            checkpointType: .leaf,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: bytes),
                            dtype: "bfloat16",
                            shape: [bytes / 2]
                        )
                    ],
                    metaState: ["meta"],
                    offset: 5
                )
            ]
        )
    }

    private func waitUntil(
        timeout: Duration = .seconds(2),
        _ condition: @Sendable () async -> Bool
    ) async -> Bool {
        let start = ContinuousClock.now
        while ContinuousClock.now - start < timeout {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    private final class RebuildCallbackTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var _committed: Set<String> = []

        var committed: Set<String> {
            lock.lock(); defer { lock.unlock() }
            return _committed
        }

        var onCommit: @Sendable (String) -> Void {
            { [weak self] id in
                guard let self else { return }
                self.lock.lock()
                self._committed.insert(id)
                self.lock.unlock()
            }
        }

        var onDrop: @Sendable (String, SSDDropReason) -> Void {
            { _, _ in }
        }
    }
}
