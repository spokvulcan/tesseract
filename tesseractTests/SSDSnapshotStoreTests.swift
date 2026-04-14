//
//  SSDSnapshotStoreTests.swift
//  tesseractTests
//
//  Unit tests for the SSD prefix-cache writer skeleton. Cover the
//  front-door admission rules (byte budget, drop-oldest-pending,
//  single-payload-too-large), writer FIFO ordering, atomic
//  temp-rename file creation, commit/drop callback ordering,
//  admission-time type-protected LRU cut (including the asymmetric
//  system-protection rule), `recordHit` recency bump, and the
//  debounced manifest persist.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct SSDSnapshotStoreTests {

    // MARK: - Scratch config + payload helpers

    private func makeConfig(
        budgetBytes: Int = 1_000_000,
        maxPendingBytes: Int = 10_000_000
    ) -> (SSDPrefixCacheConfig, URL) {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-store-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: budgetBytes,
            maxPendingBytes: maxPendingBytes
        )
        return (config, rootURL)
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makePayload(bytes: Int) -> SnapshotPayload {
        SnapshotPayload(
            tokenOffset: 4_096,
            checkpointType: .system,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: bytes),
                            dtype: "bfloat16",
                            shape: [1, bytes]
                        )
                    ],
                    metaState: ["meta"],
                    offset: 4_096
                )
            ]
        )
    }

    private func makeDescriptor(
        id: String = UUID().uuidString,
        partition: String = "abcd1234",
        checkpointType: String = "leaf",
        bytes: Int = 1_024,
        lastAccessAt: Double = 0
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: partition,
            pathFromRoot: [1, 2, 3],
            tokenOffset: 4_096,
            checkpointType: checkpointType,
            bytes: bytes,
            createdAt: 100_000,
            lastAccessAt: lastAccessAt,
            fileRelativePath: "partitions/\(partition)/snapshots/\(id.prefix(1))/\(id).safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func makePartitionMeta(
        modelID: String = "mlx-community/Qwen3-4B-4bit",
        fingerprint: String = String(repeating: "a", count: 64)
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: modelID,
            modelFingerprint: fingerprint,
            kvBits: 8,
            kvGroupSize: 64,
            sessionAffinity: nil,
            createdAt: 100_000,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// Spins up a store and registers the default test partition
    /// (`"abcd1234"`) so enqueue tests produce manifests that
    /// satisfy the `SnapshotManifest` invariant (every snapshot's
    /// partition digest must be present in `manifest.partitions`).
    private func makeStoreWithPartition(
        config: SSDPrefixCacheConfig,
        manifestDebounce: Duration = .milliseconds(20),
        onCommit: @escaping @Sendable (String) -> Void = { _ in },
        onDrop: @escaping @Sendable (String, SSDDropReason) -> Void = { _, _ in }
    ) -> SSDSnapshotStore {
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: manifestDebounce,
            onCommit: onCommit,
            onDrop: onDrop
        )
        store.registerPartition(makePartitionMeta(), digest: "abcd1234")
        return store
    }

    // MARK: - Callback tracker

    /// Thread-safe collector of commit/drop events. Tests poll on
    /// this via `waitUntil` to observe the writer's asynchronous
    /// completion without guessing at sleep durations.
    final class CallbackTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var _committed: [String] = []
        private var _dropped: [(id: String, reason: SSDDropReason)] = []

        var committed: [String] {
            lock.lock(); defer { lock.unlock() }
            return _committed
        }

        var dropped: [(id: String, reason: SSDDropReason)] {
            lock.lock(); defer { lock.unlock() }
            return _dropped
        }

        var onCommit: @Sendable (String) -> Void {
            { [weak self] id in
                guard let self else { return }
                self.lock.lock()
                self._committed.append(id)
                self.lock.unlock()
            }
        }

        var onDrop: @Sendable (String, SSDDropReason) -> Void {
            { [weak self] id, reason in
                guard let self else { return }
                self.lock.lock()
                self._dropped.append((id: id, reason: reason))
                self.lock.unlock()
            }
        }
    }

    /// Poll `condition` every 10 ms until it returns true or
    /// `timeout` elapses. Returns true if the condition was met.
    /// Used to observe asynchronous writer completion — the alternative
    /// is hard-coded sleeps, which race under CI load.
    private func waitUntil(
        timeout: Duration = .seconds(5),
        _ condition: @escaping @Sendable () -> Bool
    ) async -> Bool {
        let start = ContinuousClock.now
        while ContinuousClock.now - start < timeout {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    // MARK: - Front door: happy path and rejection

    @Test
    func tryEnqueueAcceptsPayloadUnderBudget() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)

        guard case .accepted(let ref) = result else {
            Issue.record("expected .accepted, got \(result)")
            return
        }
        #expect(ref.snapshotID == descriptor.snapshotID)
        #expect(ref.committed == false)
        #expect(ref.bytesOnDisk == payload.totalBytes)

        let committed = await waitUntil {
            tracker.committed.contains(descriptor.snapshotID)
        }
        #expect(committed)
        #expect(tracker.dropped.isEmpty)
    }

    @Test
    func tryEnqueueRejectsPayloadLargerThanCap() {
        let (config, root) = makeConfig(maxPendingBytes: 1_024)
        defer { cleanup(root) }
        let store = SSDSnapshotStore(config: config, manifestDebounce: .milliseconds(20))

        let payload = makePayload(bytes: 4_096)
        let descriptor = makeDescriptor(bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        #expect(result == .rejectedTooLargeForBudget)
    }

    @Test
    func tryEnqueueRejectsInvalidCheckpointType() {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = SSDSnapshotStore(config: config)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(checkpointType: "not-a-real-type", bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        #expect(result == .rejectedInvalidCheckpointType)
    }

    @Test
    func tryEnqueueRejectsUnregisteredPartition() {
        // Enforces the `SnapshotManifest` invariant by construction:
        // a descriptor whose partition digest has not been
        // registered via `registerPartition` is refused at the
        // front door rather than persisted as a dangling entry
        // that warm start would later drop.
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = SSDSnapshotStore(config: config)

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(partition: "unregistered", bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        #expect(result == .rejectedUnregisteredPartition)

        // After registering the partition, the same enqueue must
        // succeed — the rejection is not sticky.
        store.registerPartition(makePartitionMeta(), digest: "unregistered")
        let retry = store.tryEnqueue(payload: payload, descriptor: descriptor)
        if case .accepted = retry {
            // expected
        } else {
            Issue.record("expected .accepted after registration, got \(retry)")
        }
    }

    // MARK: - Back-pressure

    @Test
    func tryEnqueueDropsOldestPendingOnBurstOverflow() async {
        // Cap pending bytes so two 600-byte payloads can live in
        // the queue but a third forces the oldest out. We block
        // the writer from draining by pointing at a root URL
        // whose parent directory is writable — the writer will
        // try to process items, so we race. To avoid that race,
        // we make each payload 600 bytes and the cap 1_200 bytes,
        // then enqueue 3 rapidly and snapshot before the writer
        // finishes. The race is inherent to a "drop under burst"
        // test, so we check EITHER the pending count drop OR the
        // drop-callback fire.
        let (config, root) = makeConfig(
            budgetBytes: 10_000_000,
            maxPendingBytes: 1_200
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payloadA = makePayload(bytes: 600)
        let payloadB = makePayload(bytes: 600)
        let payloadC = makePayload(bytes: 600)
        let descA = makeDescriptor(id: "aaa", bytes: 600)
        let descB = makeDescriptor(id: "bbb", bytes: 600)
        let descC = makeDescriptor(id: "ccc", bytes: 600)

        let resultA = store.tryEnqueue(payload: payloadA, descriptor: descA)
        let resultB = store.tryEnqueue(payload: payloadB, descriptor: descB)
        let resultC = store.tryEnqueue(payload: payloadC, descriptor: descC)

        #expect(resultA != .rejectedTooLargeForBudget)
        #expect(resultB != .rejectedTooLargeForBudget)
        #expect(resultC != .rejectedTooLargeForBudget)

        // Wait until both drops and commits have fully drained so
        // we can inspect the final set of dropped IDs. `aaa` and
        // `bbb` might BOTH drop (if C arrives before A/B commit)
        // or only `aaa` drops (if the writer ran fast). Either
        // way `ccc` must survive because it's the most recent.
        let settled = await waitUntil {
            let committed = Set(tracker.committed)
            let dropped = Set(tracker.dropped.map(\.id))
            return committed.union(dropped).count >= 3
        }
        #expect(settled)

        let committed = Set(tracker.committed)
        let dropped = Set(tracker.dropped.map(\.id))
        #expect(committed.contains("ccc") || dropped.contains("ccc") == false)
        // `aaa` should have been the first victim of back-pressure
        // if ANY drop happened. Verify the drops (if any) carry
        // the correct reason.
        for (id, reason) in tracker.dropped {
            #expect(reason == .backpressureOldest, "unexpected drop reason for \(id): \(reason)")
        }
    }

    // MARK: - Writer: files, FIFO, atomic rename

    @Test
    func writerCreatesFileOnCommit() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 2_048)
        let descriptor = makeDescriptor(id: "file-test-00000000", bytes: payload.totalBytes)
        _ = store.tryEnqueue(payload: payload, descriptor: descriptor)

        let committed = await waitUntil {
            tracker.committed.contains(descriptor.snapshotID)
        }
        #expect(committed)

        let fileURL = root
            .appendingPathComponent("partitions")
            .appendingPathComponent(descriptor.partitionDigest)
            .appendingPathComponent("snapshots")
            .appendingPathComponent("f")
            .appendingPathComponent("\(descriptor.snapshotID).safetensors")
        #expect(FileManager.default.fileExists(atPath: fileURL.path))

        let tempURL = fileURL.appendingPathExtension("tmp")
        #expect(FileManager.default.fileExists(atPath: tempURL.path) == false)
    }

    @Test
    func writerProcessesCommittedPayloadsInEnqueueOrder() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let ids = (0..<5).map { "order-\($0)-00000000-1111-2222-3333-444444444444" }
        for id in ids {
            let payload = makePayload(bytes: 512)
            let descriptor = makeDescriptor(id: id, bytes: payload.totalBytes)
            _ = store.tryEnqueue(payload: payload, descriptor: descriptor)
        }

        let done = await waitUntil { tracker.committed.count == ids.count }
        #expect(done)
        #expect(tracker.committed == ids)
    }

    @Test
    func writerWritesFileAtomicallyViaTempRename() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 4_096)
        let descriptor = makeDescriptor(id: "atomic-test-000", bytes: payload.totalBytes)
        _ = store.tryEnqueue(payload: payload, descriptor: descriptor)

        let committed = await waitUntil {
            tracker.committed.contains(descriptor.snapshotID)
        }
        #expect(committed)

        // Atomic invariant: after commit, the final file exists
        // AND the temp file has been removed. A crash during
        // write would leave the temp file, never a half-written
        // final.
        let finalURL = root
            .appendingPathComponent("partitions/\(descriptor.partitionDigest)/snapshots/a/\(descriptor.snapshotID).safetensors")
        let tempURL = finalURL.appendingPathExtension("tmp")
        #expect(FileManager.default.fileExists(atPath: finalURL.path))
        #expect(FileManager.default.fileExists(atPath: tempURL.path) == false)

        // File is non-empty and starts with our 8-byte header
        // length prefix — pin the placeholder container format.
        let data = try! Data(contentsOf: finalURL)
        #expect(data.count > 8)
        let headerLength: UInt64 = data.withUnsafeBytes { buffer in
            buffer.load(fromByteOffset: 0, as: UInt64.self).littleEndian
        }
        #expect(headerLength > 0)
        #expect(Int(headerLength) + 8 <= data.count)
    }

    // MARK: - Admission-time LRU cut

    @Test
    func admissionEvictsOldestNonSystemResident() async {
        // Budget just fits two 500-byte entries. Fill with two
        // leaves, wait for both to commit, then add a third leaf
        // → oldest gets evicted.
        let (config, root) = makeConfig(
            budgetBytes: 1_100,
            maxPendingBytes: 10_000_000
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let leafA = makeDescriptor(id: "aaa-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: leafA)

        let committedA = await waitUntil { tracker.committed.contains(leafA.snapshotID) }
        #expect(committedA)

        // Introduce a small gap so the second commit gets a
        // strictly later `lastAccessAt` than the first.
        try? await Task.sleep(for: .milliseconds(10))

        let leafB = makeDescriptor(id: "bbb-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: leafB)

        let committedB = await waitUntil { tracker.committed.contains(leafB.snapshotID) }
        #expect(committedB)
        #expect(store.currentSSDBytesForTesting() == 1_000)

        try? await Task.sleep(for: .milliseconds(10))

        // Third leaf forces the admission cut. Oldest non-system
        // is leafA.
        let leafC = makeDescriptor(id: "ccc-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: leafC)

        let committedC = await waitUntil { tracker.committed.contains(leafC.snapshotID) }
        #expect(committedC)

        // A should be gone; B and C should still reside.
        let resident = Set(store.residentIDsByRecencyForTesting())
        #expect(resident.contains(leafA.snapshotID) == false)
        #expect(resident.contains(leafB.snapshotID))
        #expect(resident.contains(leafC.snapshotID))

        // File for A should be gone too.
        let aURL = root.appendingPathComponent(
            "partitions/\(leafA.partitionDigest)/snapshots/a/\(leafA.snapshotID).safetensors"
        )
        #expect(FileManager.default.fileExists(atPath: aURL.path) == false)
    }

    @Test
    func admissionDropsNonSystemIncomingWhenOnlySystemResident() async {
        // Budget fits one 500-byte entry. Fill with a system
        // entry, then try to add a leaf — the leaf must be
        // dropped with `.systemProtectionWins`, and the system
        // resident must stay intact.
        let (config, root) = makeConfig(
            budgetBytes: 600,
            maxPendingBytes: 10_000_000
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let sys = makeDescriptor(id: "sys-proto", checkpointType: "system", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: sys)

        let committedSys = await waitUntil { tracker.committed.contains(sys.snapshotID) }
        #expect(committedSys)

        let leaf = makeDescriptor(id: "leaf-loser", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: leaf)

        // Wait for the writer to process and drop.
        let dropped = await waitUntil {
            tracker.dropped.contains(where: { $0.id == leaf.snapshotID })
        }
        #expect(dropped)
        let dropReason = tracker.dropped.first(where: { $0.id == leaf.snapshotID })?.reason
        #expect(dropReason == .systemProtectionWins)

        // System resident is still there, unmodified.
        let resident = Set(store.residentIDsByRecencyForTesting())
        #expect(resident.contains(sys.snapshotID))
        #expect(resident.contains(leaf.snapshotID) == false)
    }

    @Test
    func admissionEvictionFiresEvictedByLRUDropCallback() async {
        // Committed residents that are displaced by a later
        // admission's LRU cut must fire `onDrop(.evictedByLRU)`.
        // Downstream consumers attach committed `storageRef`s to
        // radix nodes; without this callback, those refs would
        // keep pointing at a deleted file and surface stale SSD
        // hits on subsequent lookups.
        let (config, root) = makeConfig(
            budgetBytes: 1_100,
            maxPendingBytes: 10_000_000
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let victim = makeDescriptor(id: "victim-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: victim)
        _ = await waitUntil { tracker.committed.contains(victim.snapshotID) }
        #expect(tracker.dropped.isEmpty)

        try? await Task.sleep(for: .milliseconds(10))

        let filler = makeDescriptor(id: "filler-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: filler)
        _ = await waitUntil { tracker.committed.contains(filler.snapshotID) }

        try? await Task.sleep(for: .milliseconds(10))

        // Third commit forces the admission cut; `victim` is the
        // oldest non-system resident and must be evicted.
        let incoming = makeDescriptor(id: "incoming-leaf", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: incoming)
        _ = await waitUntil { tracker.committed.contains(incoming.snapshotID) }

        let dropFired = await waitUntil {
            tracker.dropped.contains { $0.id == victim.snapshotID && $0.reason == .evictedByLRU }
        }
        #expect(dropFired)

        // The evicted resident's file must also be gone from disk.
        // This is the observable half of "file I/O happens outside
        // the front-door lock" — if delete were skipped or deferred
        // forever, this assertion would fail.
        let victimURL = root.appendingPathComponent(
            "partitions/\(victim.partitionDigest)/snapshots/v/\(victim.snapshotID).safetensors"
        )
        let deleted = await waitUntil {
            FileManager.default.fileExists(atPath: victimURL.path) == false
        }
        #expect(deleted)
    }

    @Test
    func admissionEvictionOnlyPathPersistsManifestWithoutCommit() async {
        // The eviction-only path — admission cut evicts residents
        // but then drops the incoming — must still schedule a
        // debounced manifest persist. Without this, the on-disk
        // manifest stays stale until some unrelated later mutation
        // happens to reschedule. Exercises the "evict leaf, drop
        // oversized leaf with .systemProtectionWins" flow.
        let (config, root) = makeConfig(
            budgetBytes: 1_100,
            maxPendingBytes: 10_000_000
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let leafVictim = makeDescriptor(id: "leaf-victim", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: leafVictim)
        _ = await waitUntil { tracker.committed.contains(leafVictim.snapshotID) }

        let systemKept = makeDescriptor(id: "sys-kept", checkpointType: "system", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: systemKept)
        _ = await waitUntil { tracker.committed.contains(systemKept.snapshotID) }

        // Oversized leaf: after evicting leafVictim the free space
        // is 1000 bytes (budget 1100 minus 500-byte system resident
        // plus 500 reclaimed), but the incoming needs 1300 bytes
        // to fit on top of the remaining system resident, so
        // admission drops it with .systemProtectionWins. leafVictim
        // is gone from the manifest regardless — that's the
        // eviction-only mutation the persist fix has to observe.
        let oversized = makeDescriptor(id: "oversized", checkpointType: "leaf", bytes: 800)
        _ = store.tryEnqueue(payload: makePayload(bytes: 800), descriptor: oversized)
        let dropped = await waitUntil {
            tracker.dropped.contains(where: { $0.id == oversized.snapshotID && $0.reason == .systemProtectionWins })
        }
        #expect(dropped)
        #expect(
            tracker.dropped.contains(where: { $0.id == leafVictim.snapshotID && $0.reason == .evictedByLRU })
        )

        // Do NOT call `flushManifestForTesting()`. The test is
        // specifically verifying that the debounced persist fires
        // on its own after an eviction-only path, without needing
        // a later unrelated mutation to reschedule it.
        let manifestURL = root.appendingPathComponent("manifest.json")
        let persisted = await waitUntil(timeout: .seconds(5)) {
            guard FileManager.default.fileExists(atPath: manifestURL.path) else {
                return false
            }
            guard
                let data = try? Data(contentsOf: manifestURL),
                let decoded = try? JSONDecoder().decode(SnapshotManifest.self, from: data)
            else {
                return false
            }
            return decoded.snapshots[leafVictim.snapshotID] == nil
                && decoded.snapshots[systemKept.snapshotID] != nil
        }
        #expect(persisted)
    }

    @Test
    func admissionAllowsSystemIncomingToLaterallyEvictOldestSystem() async {
        // Budget fits one 500-byte entry. Fill with a system
        // entry. A second system entry must laterally evict the
        // first (protection is preserved across the set).
        let (config, root) = makeConfig(
            budgetBytes: 600,
            maxPendingBytes: 10_000_000
        )
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let sysOld = makeDescriptor(id: "sys-old", checkpointType: "system", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: sysOld)

        let committedOld = await waitUntil { tracker.committed.contains(sysOld.snapshotID) }
        #expect(committedOld)

        try? await Task.sleep(for: .milliseconds(10))

        let sysNew = makeDescriptor(id: "sys-new", checkpointType: "system", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: sysNew)

        let committedNew = await waitUntil { tracker.committed.contains(sysNew.snapshotID) }
        #expect(committedNew)

        let resident = Set(store.residentIDsByRecencyForTesting())
        #expect(resident.contains(sysOld.snapshotID) == false)
        #expect(resident.contains(sysNew.snapshotID))
    }

    // MARK: - recordHit

    @Test
    func recordHitBumpsLastAccessAtOnCommittedDescriptor() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let first = makeDescriptor(id: "recency-a", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: first)
        _ = await waitUntil { tracker.committed.contains(first.snapshotID) }

        try? await Task.sleep(for: .milliseconds(10))

        let second = makeDescriptor(id: "recency-b", checkpointType: "leaf", bytes: 500)
        _ = store.tryEnqueue(payload: makePayload(bytes: 500), descriptor: second)
        _ = await waitUntil { tracker.committed.contains(second.snapshotID) }

        // Before the bump: second is the most recent, so resident
        // order (ascending lastAccessAt) is [first, second].
        let before = store.residentIDsByRecencyForTesting()
        #expect(before == [first.snapshotID, second.snapshotID])

        try? await Task.sleep(for: .milliseconds(10))
        store.recordHit(id: first.snapshotID)

        // After the bump: first's `lastAccessAt` moved past
        // second's, so the order flips.
        let after = store.residentIDsByRecencyForTesting()
        #expect(after == [second.snapshotID, first.snapshotID])
    }

    @Test
    func recordHitForUnknownIDIsNoOp() {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = SSDSnapshotStore(config: config)

        // Does not throw, does not crash, does not schedule a
        // persist. Just returns.
        store.recordHit(id: "never-seen-before")
        #expect(store.residentIDsByRecencyForTesting().isEmpty)
    }

    // MARK: - registerPartition

    @Test
    func registerPartitionUpsertsIntoManifest() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = SSDSnapshotStore(config: config, manifestDebounce: .milliseconds(20))

        let original = makePartitionMeta(fingerprint: String(repeating: "a", count: 64))
        store.registerPartition(original, digest: "abcd1234")
        store.flushManifestForTesting()

        let manifestURL = root.appendingPathComponent("manifest.json")
        let exists = await waitUntil {
            FileManager.default.fileExists(atPath: manifestURL.path)
        }
        #expect(exists)

        var decoded = try! JSONDecoder().decode(
            SnapshotManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        #expect(decoded.partitions["abcd1234"] == original)
        #expect(decoded.snapshots.isEmpty)

        // Re-registering the same digest with a fresh fingerprint
        // must overwrite the stored metadata — this is the
        // fingerprint-rotation path and it has to be idempotent.
        let rotated = makePartitionMeta(fingerprint: String(repeating: "b", count: 64))
        store.registerPartition(rotated, digest: "abcd1234")
        store.flushManifestForTesting()

        decoded = try! JSONDecoder().decode(
            SnapshotManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        #expect(decoded.partitions["abcd1234"] == rotated)
        #expect(decoded.partitions["abcd1234"]?.modelFingerprint != original.modelFingerprint)
    }

    // MARK: - Manifest persistence

    @Test
    func manifestIsPersistedAfterCommit() async {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let descriptor = makeDescriptor(id: "manifest-persist", checkpointType: "system", bytes: 1_024)
        _ = store.tryEnqueue(payload: makePayload(bytes: 1_024), descriptor: descriptor)
        _ = await waitUntil { tracker.committed.contains(descriptor.snapshotID) }

        // Force-flush so we don't race the 20 ms debounce.
        store.flushManifestForTesting()

        let manifestURL = root.appendingPathComponent("manifest.json")
        let exists = await waitUntil {
            FileManager.default.fileExists(atPath: manifestURL.path)
        }
        #expect(exists)

        let data = try! Data(contentsOf: manifestURL)
        let decoded = try! JSONDecoder().decode(SnapshotManifest.self, from: data)
        #expect(decoded.isSchemaCompatible)
        #expect(decoded.snapshots[descriptor.snapshotID]?.snapshotID == descriptor.snapshotID)
        // Load-bearing invariant: every snapshot's partition digest
        // must resolve to a `PartitionMeta` in the same manifest —
        // otherwise warm start drops the descriptor as a dangling
        // reference.
        #expect(decoded.partitions[descriptor.partitionDigest] != nil)
    }

    @Test
    func manifestFileUsesAtomicTempRename() async {
        // The persist writer should never leave a `manifest.json.tmp`
        // behind on a successful write.
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let descriptor = makeDescriptor(id: "atomic-manifest", checkpointType: "leaf", bytes: 1_024)
        _ = store.tryEnqueue(payload: makePayload(bytes: 1_024), descriptor: descriptor)
        _ = await waitUntil { tracker.committed.contains(descriptor.snapshotID) }

        store.flushManifestForTesting()

        let tempURL = root.appendingPathComponent("manifest.json.tmp")
        let finalURL = root.appendingPathComponent("manifest.json")
        _ = await waitUntil { FileManager.default.fileExists(atPath: finalURL.path) }
        #expect(FileManager.default.fileExists(atPath: tempURL.path) == false)
    }

    // MARK: - Non-suspending admission

    @Test
    func tryEnqueueDoesNotSuspendUnderBusyMainActor() async {
        // Proves the plan's load-bearing property: `tryEnqueue`
        // can be called synchronously from a MainActor closure
        // that never yields to the runtime. If tryEnqueue were
        // accidentally made `async`, this test would fail to
        // compile because the call site wouldn't be synchronous.
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        // Run inside an explicit MainActor.run — same shape as the
        // production LLMActor store call sites. The block is
        // synchronous; no await inside the closure.
        let result: TryEnqueueResult = await MainActor.run {
            let payload = self.makePayload(bytes: 256)
            let descriptor = self.makeDescriptor(bytes: 256)
            return store.tryEnqueue(payload: payload, descriptor: descriptor)
        }
        #expect(result != .rejectedTooLargeForBudget)
        #expect(result != .rejectedInvalidCheckpointType)
    }
}
