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
import MLX
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

    // MARK: - loadSync round trip + hydration failures

    /// Build a committed `SnapshotStorageRef` from the pending ref
    /// returned by `tryEnqueue`. The writer's commit callback flips
    /// the ref's `committed` bit in production via
    /// `TieredSnapshotStore.markStorageRefCommitted` — here we
    /// reconstruct the committed shape directly because the test
    /// talks to the store without an intermediate tier.
    private func committedRef(
        from ref: SnapshotStorageRef
    ) -> SnapshotStorageRef {
        SnapshotStorageRef(
            snapshotID: ref.snapshotID,
            partitionDigest: ref.partitionDigest,
            tokenOffset: ref.tokenOffset,
            checkpointType: ref.checkpointType,
            bytesOnDisk: ref.bytesOnDisk,
            lastAccessTime: ref.lastAccessTime,
            committed: true
        )
    }

    /// Build a payload whose shape / dtype / byte count are
    /// internally consistent so `MLXArray(data:, shape:, dtype:)`
    /// accepts the slice without tripping an MLX precondition.
    /// The existing `makePayload(bytes:)` deliberately uses a mismatched
    /// synthetic shape for writer-only tests where the bytes never
    /// reach MLX; `loadSync` needs a real tensor layout.
    private func makeRoundTripPayload(
        elementCount: Int = 16
    ) -> (SnapshotPayload, PersistedSnapshotDescriptor) {
        let dtypeWireName = "bfloat16"
        let itemSize = 2  // bfloat16
        let byteCount = elementCount * itemSize
        let payload = SnapshotPayload(
            tokenOffset: 4_096,
            checkpointType: .system,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: byteCount),
                            dtype: dtypeWireName,
                            shape: [elementCount]
                        )
                    ],
                    metaState: ["meta"],
                    offset: 4_096
                )
            ]
        )
        let descriptor = makeDescriptor(bytes: byteCount)
        return (payload, descriptor)
    }

    private func descriptorWithDFlashDraft(
        targetPayload: SnapshotPayload,
        draftPayload: SnapshotPayload,
        checkpointType: String = "leaf"
    ) -> PersistedSnapshotDescriptor {
        let id = UUID().uuidString
        let partition = "abcd1234"
        return PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: partition,
            pathFromRoot: [1, 2, 3],
            tokenOffset: targetPayload.tokenOffset,
            checkpointType: checkpointType,
            bytes: targetPayload.totalBytes + draftPayload.totalBytes,
            dflashDraftBytes: draftPayload.totalBytes,
            createdAt: 100_000,
            lastAccessAt: 0,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: id,
                partitionDigest: partition
            ),
            dflashDraftFileRelativePath: PersistedSnapshotDescriptor.relativeDFlashDraftFilePath(
                snapshotID: id,
                partitionDigest: partition
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// End-to-end round trip: enqueue a payload via `tryEnqueue`,
    /// wait for the writer to commit, then call `loadSync` and verify
    /// the reconstructed `HybridCacheSnapshot` matches the original
    /// payload (layer count, class name, shape, meta state).
    @Test
    func loadSyncReadsBackCommittedSnapshot() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let (payload, descriptor) = makeRoundTripPayload(elementCount: 128)
        let fingerprint = makePartitionMeta().modelFingerprint

        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }

        let committed = await waitUntil {
            tracker.committed.contains(pending.snapshotID)
        }
        #expect(committed)

        let snapshot = store.loadSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: fingerprint
        )
        #expect(snapshot != nil)
        guard let snapshot else { return }

        #expect(snapshot.tokenOffset == descriptor.tokenOffset)
        #expect(snapshot.checkpointType == pending.checkpointType)
        #expect(snapshot.layers.count == payload.layers.count)

        for (layerIndex, layerPayload) in payload.layers.enumerated() {
            let layerState = snapshot.layers[layerIndex]
            #expect(layerState.className == layerPayload.className)
            #expect(layerState.offset == layerPayload.offset)
            #expect(layerState.metaState == layerPayload.metaState)
            #expect(layerState.state.count == layerPayload.state.count)
            for (arrayIndex, arrayPayload) in layerPayload.state.enumerated() {
                let mlxArray = layerState.state[arrayIndex]
                #expect(mlxArray.shape == arrayPayload.shape)
            }
        }
    }

    @Test
    func loadBundleSyncRestoresDFlashDraftCompanion() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let (targetPayload, _) = makeRoundTripPayload(elementCount: 128)
        let (draftPayload, _) = makeRoundTripPayload(elementCount: 64)
        let descriptor = descriptorWithDFlashDraft(
            targetPayload: targetPayload,
            draftPayload: draftPayload
        )
        let fingerprint = makePartitionMeta().modelFingerprint

        let result = store.tryEnqueue(
            payload: targetPayload,
            dflashDraftPayload: draftPayload,
            descriptor: descriptor
        )
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }

        let committed = await waitUntil {
            tracker.committed.contains(pending.snapshotID)
        }
        #expect(committed)
        #expect(store.currentSSDBytesForTesting() == descriptor.bytes)

        let bundle = store.loadBundleSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: fingerprint
        )
        #expect(bundle?.snapshot.tokenOffset == descriptor.tokenOffset)
        #expect(bundle?.dflashDraftSnapshot?.tokenOffset == descriptor.tokenOffset)
        #expect(bundle?.dflashDraftSnapshot?.memoryBytes == draftPayload.totalBytes)

        let draftURL = root.appendingPathComponent(
            PersistedSnapshotDescriptor.relativeDFlashDraftFilePath(
                snapshotID: descriptor.snapshotID,
                partitionDigest: descriptor.partitionDigest
            )
        )
        #expect(FileManager.default.fileExists(atPath: draftURL.path))
    }

    @Test
    func loadBundleSyncMissingDFlashDraftDegradesToTargetOnly() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let (targetPayload, _) = makeRoundTripPayload(elementCount: 128)
        let (draftPayload, _) = makeRoundTripPayload(elementCount: 64)
        let descriptor = descriptorWithDFlashDraft(
            targetPayload: targetPayload,
            draftPayload: draftPayload
        )
        let fingerprint = makePartitionMeta().modelFingerprint

        let result = store.tryEnqueue(
            payload: targetPayload,
            dflashDraftPayload: draftPayload,
            descriptor: descriptor
        )
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }
        _ = await waitUntil { tracker.committed.contains(pending.snapshotID) }

        let draftURL = root.appendingPathComponent(
            PersistedSnapshotDescriptor.relativeDFlashDraftFilePath(
                snapshotID: descriptor.snapshotID,
                partitionDigest: descriptor.partitionDigest
            )
        )
        try FileManager.default.removeItem(at: draftURL)

        let bundle = store.loadBundleSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: fingerprint
        )
        #expect(bundle?.snapshot.tokenOffset == descriptor.tokenOffset)
        #expect(bundle?.dflashDraftSnapshot == nil)
        #expect(store.currentSSDBytesForTesting() == targetPayload.totalBytes)
        #expect(!tracker.dropped.contains { $0.reason == .hydrationFailure })
    }

    @Test
    func loadBundleSyncCorruptDFlashDraftDegradesToTargetOnly() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let (targetPayload, _) = makeRoundTripPayload(elementCount: 128)
        let (draftPayload, _) = makeRoundTripPayload(elementCount: 64)
        let descriptor = descriptorWithDFlashDraft(
            targetPayload: targetPayload,
            draftPayload: draftPayload
        )
        let fingerprint = makePartitionMeta().modelFingerprint

        let result = store.tryEnqueue(
            payload: targetPayload,
            dflashDraftPayload: draftPayload,
            descriptor: descriptor
        )
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }
        _ = await waitUntil { tracker.committed.contains(pending.snapshotID) }

        let draftURL = root.appendingPathComponent(
            PersistedSnapshotDescriptor.relativeDFlashDraftFilePath(
                snapshotID: descriptor.snapshotID,
                partitionDigest: descriptor.partitionDigest
            )
        )
        try Data("corrupt draft companion".utf8).write(to: draftURL)

        let bundle = store.loadBundleSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: fingerprint
        )
        #expect(bundle?.snapshot.tokenOffset == descriptor.tokenOffset)
        #expect(bundle?.dflashDraftSnapshot == nil)
        #expect(store.currentSSDBytesForTesting() == targetPayload.totalBytes)
        #expect(!FileManager.default.fileExists(atPath: draftURL.path))
        #expect(!tracker.dropped.contains { $0.reason == .hydrationFailure })
    }

    @Test
    func admissionEvictionDeletesDFlashDraftCompanionAndAccountsBytes() async throws {
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

        let victimTarget = makePayload(bytes: 400)
        let victimDraft = makePayload(bytes: 200)
        let victim = descriptorWithDFlashDraft(
            targetPayload: victimTarget,
            draftPayload: victimDraft
        )
        _ = store.tryEnqueue(
            payload: victimTarget,
            dflashDraftPayload: victimDraft,
            descriptor: victim
        )
        #expect(await waitUntil { tracker.committed.contains(victim.snapshotID) })
        #expect(store.currentSSDBytesForTesting() == victim.bytes)

        let victimURL = root.appendingPathComponent(victim.fileRelativePath)
        let victimDraftURL = root.appendingPathComponent(
            victim.dflashDraftFileRelativePath ?? ""
        )
        #expect(FileManager.default.fileExists(atPath: victimURL.path))
        #expect(FileManager.default.fileExists(atPath: victimDraftURL.path))

        try? await Task.sleep(for: .milliseconds(10))

        let filler = makeDescriptor(
            id: "filler-dflash-evict",
            checkpointType: "leaf",
            bytes: 400
        )
        _ = store.tryEnqueue(payload: makePayload(bytes: 400), descriptor: filler)
        #expect(await waitUntil { tracker.committed.contains(filler.snapshotID) })

        try? await Task.sleep(for: .milliseconds(10))

        let incoming = makeDescriptor(
            id: "incoming-dflash-evict",
            checkpointType: "leaf",
            bytes: 400
        )
        _ = store.tryEnqueue(payload: makePayload(bytes: 400), descriptor: incoming)
        #expect(await waitUntil { tracker.committed.contains(incoming.snapshotID) })

        #expect(await waitUntil {
            tracker.dropped.contains {
                $0.id == victim.snapshotID && $0.reason == .evictedByLRU
            }
        })
        #expect(await waitUntil {
            !FileManager.default.fileExists(atPath: victimURL.path)
                && !FileManager.default.fileExists(atPath: victimDraftURL.path)
        })
        #expect(store.currentSSDBytesForTesting() == filler.bytes + incoming.bytes)
    }

    @Test
    func loadBundleSyncFingerprintMismatchDeletesDFlashDraftCompanion() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let (targetPayload, _) = makeRoundTripPayload(elementCount: 128)
        let (draftPayload, _) = makeRoundTripPayload(elementCount: 64)
        let descriptor = descriptorWithDFlashDraft(
            targetPayload: targetPayload,
            draftPayload: draftPayload
        )

        let result = store.tryEnqueue(
            payload: targetPayload,
            dflashDraftPayload: draftPayload,
            descriptor: descriptor
        )
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }
        _ = await waitUntil { tracker.committed.contains(pending.snapshotID) }

        let targetURL = root.appendingPathComponent(descriptor.fileRelativePath)
        let draftURL = root.appendingPathComponent(
            descriptor.dflashDraftFileRelativePath ?? ""
        )
        #expect(FileManager.default.fileExists(atPath: targetURL.path))
        #expect(FileManager.default.fileExists(atPath: draftURL.path))

        let bundle = store.loadBundleSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: String(repeating: "z", count: 64)
        )
        #expect(bundle == nil)

        #expect(await waitUntil {
            tracker.dropped.contains {
                $0.id == pending.snapshotID && $0.reason == .hydrationFailure
            }
        })
        #expect(!FileManager.default.fileExists(atPath: targetURL.path))
        #expect(!FileManager.default.fileExists(atPath: draftURL.path))
        #expect(store.currentSSDBytesForTesting() == 0)
    }

    /// Fingerprint mismatch: `loadSync` returns nil, removes the
    /// descriptor from the manifest, deletes the on-disk file, and
    /// fires `.hydrationFailure` via the drop callback.
    @Test
    func loadSyncFingerprintMismatchDropsDescriptor() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 128)
        let descriptor = makeDescriptor(bytes: 128)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }
        _ = await waitUntil { tracker.committed.contains(pending.snapshotID) }

        let snapshot = store.loadSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: String(repeating: "z", count: 64)
        )
        #expect(snapshot == nil)

        let dropped = await waitUntil {
            tracker.dropped.contains {
                $0.id == pending.snapshotID && $0.reason == .hydrationFailure
            }
        }
        #expect(dropped)
        #expect(store.lastAccessAtForTesting(id: pending.snapshotID) == -1)

        let relative = PersistedSnapshotDescriptor.relativeFilePath(
            snapshotID: pending.snapshotID,
            partitionDigest: pending.partitionDigest
        )
        let fileURL = root.appendingPathComponent(relative)
        #expect(!FileManager.default.fileExists(atPath: fileURL.path))
    }

    /// File deleted externally between commit and hydration:
    /// `loadSync` returns nil, drops the descriptor, and fires
    /// `.hydrationFailure`.
    @Test
    func loadSyncFileMissingDropsDescriptor() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 128)
        let descriptor = makeDescriptor(bytes: 128)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)
        guard case .accepted(let pending) = result else {
            #expect(Bool(false), "tryEnqueue rejected: \(result)")
            return
        }
        _ = await waitUntil { tracker.committed.contains(pending.snapshotID) }

        let relative = PersistedSnapshotDescriptor.relativeFilePath(
            snapshotID: pending.snapshotID,
            partitionDigest: pending.partitionDigest
        )
        let fileURL = root.appendingPathComponent(relative)
        try FileManager.default.removeItem(at: fileURL)

        let fingerprint = makePartitionMeta().modelFingerprint
        let snapshot = store.loadSync(
            storageRef: committedRef(from: pending),
            expectedFingerprint: fingerprint
        )
        #expect(snapshot == nil)

        let dropped = await waitUntil {
            tracker.dropped.contains {
                $0.id == pending.snapshotID && $0.reason == .hydrationFailure
            }
        }
        #expect(dropped)
    }

    // MARK: - flushAsync

    /// `flushAsync` returns only after the writer has drained every
    /// pending item AND persisted a manifest that reflects the new
    /// state. A fresh store pointed at the same rootURL must then
    /// read the committed descriptor via `warmStartLoad` without
    /// relying on the debounced persist timer.
    @Test
    func flushAsyncDrainsWriterAndPersistsManifest() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        // Long debounce so a naive implementation (wait-for-commit-
        // without-persist) would fail the warm-start assertion.
        let store = makeStoreWithPartition(
            config: config,
            manifestDebounce: .seconds(60)
        )

        let payload = makePayload(bytes: 256)
        let descriptor = makeDescriptor(bytes: 256)
        guard case .accepted = store.tryEnqueue(
            payload: payload, descriptor: descriptor
        ) else {
            #expect(Bool(false), "tryEnqueue rejected")
            return
        }

        await store.flushAsync()

        let manifestURL = root.appendingPathComponent("manifest.json")
        #expect(FileManager.default.fileExists(atPath: manifestURL.path))
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(SnapshotManifest.self, from: data)
        #expect(manifest.snapshots[descriptor.snapshotID] != nil)
        #expect(manifest.partitions["abcd1234"] != nil)

        // Writer-side byte accounting also reflects the committed
        // descriptor — pending queue is drained and the item is now
        // resident.
        #expect(store.currentSSDBytesForTesting() == descriptor.bytes)
        #expect(store.pendingCountForTesting() == 0)
    }

    /// `flushAsync` on an empty store wakes the writer for a no-op
    /// drain, resumes the waiter, and returns. The conservative
    /// timeout protects against a heavily loaded CI box without
    /// hiding a real hang.
    @Test
    func flushAsyncOnEmptyStoreReturnsWithoutBlocking() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let store = makeStoreWithPartition(config: config)

        let start = ContinuousClock.now
        await store.flushAsync()
        let elapsed = ContinuousClock.now - start

        #expect(elapsed < .seconds(5))
    }

    // MARK: - Diagnostic events (Task 4.1.12)

    /// Test-only sink for `PrefixCacheDiagnostics` events. The
    /// store's writer task is detached and emits events from off-main
    /// threads, so the sink must be lock-protected. Tests poll on
    /// `lines(matching:)` via `waitUntil` so the writer's commit /
    /// drop callbacks have time to land before the assertion runs.
    private final class DiagnosticsSink: @unchecked Sendable {
        private let lock = NSLock()
        private var _lines: [String] = []

        var lines: [String] {
            lock.lock(); defer { lock.unlock() }
            return _lines
        }

        func lines(matching keyword: String) -> [String] {
            lock.lock(); defer { lock.unlock() }
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

    /// Test-only gate for pausing the detached writer until a test
    /// has finished building the pending-queue state it wants to
    /// assert against.
    private actor DrainGate {
        private var isOpen = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func wait() async {
            if isOpen {
                return
            }

            await withCheckedContinuation { continuation in
                if isOpen {
                    continuation.resume()
                    return
                }
                waiters.append(continuation)
            }
        }

        func open() {
            if isOpen {
                return
            }
            isOpen = true
            let currentWaiters = waiters
            waiters.removeAll()

            currentWaiters.forEach { $0.resume() }
        }
    }

    /// Install a sink for the duration of a single test. Returns the
    /// sink so the test can inspect its captured lines, and a closure
    /// the test must defer-call to remove the sink from the registry.
    /// Multiple tests can share the registry concurrently — each
    /// test's sink sees every emitted event, but tests filter by
    /// per-test snapshot IDs (assigned via `makeDescriptor` UUIDs or
    /// fixed-string IDs unique to the test) so cross-test leakage is
    /// ignored. The defer-uninstall keeps the registry small even
    /// under repeated test runs.
    private func makeSink() -> (DiagnosticsSink, @Sendable () -> Void) {
        let sink = DiagnosticsSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        return (sink, { PrefixCacheDiagnostics.removeTestSink(handle) })
    }

    @Test
    func ssdAdmitAcceptedFiresAfterCommit() async throws {
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let payload = makePayload(bytes: 1_024)
        let descriptor = makeDescriptor(bytes: payload.totalBytes)
        _ = store.tryEnqueue(payload: payload, descriptor: descriptor)

        let landed = await waitUntil { tracker.committed.contains(descriptor.snapshotID) }
        #expect(landed)

        let admitLines = sink.lines(matching: "event=ssdAdmit")
            .filter { $0.contains("id=\(descriptor.snapshotID)") }
        #expect(admitLines.count == 1)
        #expect(admitLines.first?.contains("outcome=accepted") == true)

        let commitLines = sink.lines(matching: "event=storageRefCommit")
            .filter { $0.contains("id=\(descriptor.snapshotID)") }
        #expect(commitLines.count == 1)
    }

    @Test
    func ssdAdmitFiresDroppedTooLargeForBudgetSynchronously() {
        let (config, root) = makeConfig(maxPendingBytes: 1_024)
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let store = makeStoreWithPartition(config: config)

        let payload = makePayload(bytes: 8_192)
        let descriptor = makeDescriptor(bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: descriptor)

        #expect(result == .rejectedTooLargeForBudget)

        let admitLines = sink.lines(matching: "event=ssdAdmit")
            .filter { $0.contains("id=\(descriptor.snapshotID)") }
        #expect(admitLines.count == 1)
        #expect(admitLines.first?.contains("outcome=droppedTooLargeForBudget") == true)
    }

    @Test
    func ssdAdmitFiresDroppedSystemProtectionWinsWhenEligibleSetEmpty() async throws {
        // Tiny budget; seed it full with a `.system` descriptor so
        // the non-`.system` eligible set is empty. A subsequent
        // non-system enqueue must hit the `.systemProtectionWins`
        // branch in `admitUnderBudget` and surface the matching
        // `ssdAdmit` outcome.
        let (config, root) = makeConfig(budgetBytes: 4_096)
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        let systemSeed = makeDescriptor(
            id: "system-seed",
            checkpointType: "system",
            bytes: 4_096,
            lastAccessAt: 100
        )
        store.seedDescriptorForTesting(systemSeed)

        let payload = makePayload(bytes: 2_048)
        let leaf = makeDescriptor(checkpointType: "leaf", bytes: payload.totalBytes)
        let result = store.tryEnqueue(payload: payload, descriptor: leaf)
        guard case .accepted = result else {
            Issue.record("expected front-door .accepted, got \(result)")
            return
        }

        let dropped = await waitUntil {
            tracker.dropped.contains { $0.id == leaf.snapshotID && $0.reason == .systemProtectionWins }
        }
        #expect(dropped)

        let admitLines = sink.lines(matching: "event=ssdAdmit")
            .filter { $0.contains("id=\(leaf.snapshotID)") }
        #expect(admitLines.count == 1)
        #expect(admitLines.first?.contains("outcome=droppedSystemProtectionWins") == true)

        // Symmetric guard: the seeded `.system` resident must not
        // have been evicted to make room. No `ssdEvictAtAdmission`
        // event ever names it as the victim.
        let evictLines = sink.lines(matching: "event=ssdEvictAtAdmission")
        #expect(!evictLines.contains { $0.contains("victimID=system-seed") })

        // And the matching storageRefDropCallback companion fired
        // with the same reason — the writer's drop path always
        // emits the lifecycle event before invoking onDrop.
        let dropCallbackLines = sink.lines(matching: "event=storageRefDropCallback")
            .filter { $0.contains("id=\(leaf.snapshotID)") }
        #expect(dropCallbackLines.count == 1)
        #expect(dropCallbackLines.first?.contains("reason=systemProtectionWins") == true)
    }

    @Test
    func ssdEvictAtAdmissionFiresOncePerVictim() async throws {
        // Tiny budget filled with two non-system residents, then
        // enqueue a non-system payload large enough to displace
        // both. The writer must fire one `ssdEvictAtAdmission`
        // event per resident plus exactly one `ssdAdmit(accepted)`
        // for the incoming.
        let (config, root) = makeConfig(budgetBytes: 4_096)
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let tracker = CallbackTracker()
        let store = makeStoreWithPartition(
            config: config,
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop
        )

        store.seedDescriptorForTesting(makeDescriptor(
            id: "old-leaf-1", checkpointType: "leaf", bytes: 2_048, lastAccessAt: 1
        ))
        store.seedDescriptorForTesting(makeDescriptor(
            id: "old-leaf-2", checkpointType: "leaf", bytes: 2_048, lastAccessAt: 2
        ))

        let payload = makePayload(bytes: 4_096)
        let incoming = makeDescriptor(checkpointType: "leaf", bytes: payload.totalBytes)
        _ = store.tryEnqueue(payload: payload, descriptor: incoming)

        let landed = await waitUntil { tracker.committed.contains(incoming.snapshotID) }
        #expect(landed)

        let evictLines = sink.lines(matching: "event=ssdEvictAtAdmission")
            .filter { $0.contains("incomingID=\(incoming.snapshotID)") }
        #expect(evictLines.count == 2)
        #expect(evictLines.contains { $0.contains("victimID=old-leaf-1") })
        #expect(evictLines.contains { $0.contains("victimID=old-leaf-2") })

        // Each evicted resident also fired its lifecycle drop
        // callback with `.evictedByLRU`.
        let evictDropCallbacks = sink.lines(matching: "event=storageRefDropCallback")
            .filter { $0.contains("reason=evictedByLRU") }
        #expect(evictDropCallbacks.count >= 2)
    }

    @Test
    func storageRefDropCallbackFiresOnBackpressureBumpedItems() async throws {
        // Tight pending byte cap — first payload occupies it
        // entirely, second payload bumps the first via
        // drop-oldest-pending. The bumped item must surface as
        // both `ssdAdmit(droppedByteBudget)` and
        // `storageRefDropCallback(backpressureOldest)`.
        let (config, root) = makeConfig(
            budgetBytes: 8_192,
            maxPendingBytes: 6_000
        )
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let tracker = CallbackTracker()
        let drainGate = DrainGate()
        // Pause the detached writer so the second enqueue sees the
        // first item still pending and deterministically triggers
        // the drop-oldest path.
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: .seconds(60),
            onCommit: tracker.onCommit,
            onDrop: tracker.onDrop,
            writerDrainPreludeForTesting: { await drainGate.wait() }
        )
        defer { Task { await drainGate.open() } }
        store.registerPartition(makePartitionMeta(), digest: "abcd1234")
        let firstPayload = makePayload(bytes: 4_096)
        let firstDescriptor = makeDescriptor(
            id: "first", checkpointType: "leaf", bytes: firstPayload.totalBytes
        )
        let firstResult = store.tryEnqueue(payload: firstPayload, descriptor: firstDescriptor)
        guard case .accepted = firstResult else {
            Issue.record("expected first .accepted, got \(firstResult)")
            return
        }

        // Second payload pushes pendingBytes over the cap and
        // bumps the oldest.
        let secondPayload = makePayload(bytes: 4_096)
        let secondDescriptor = makeDescriptor(
            id: "second", checkpointType: "leaf", bytes: secondPayload.totalBytes
        )
        let secondResult = store.tryEnqueue(payload: secondPayload, descriptor: secondDescriptor)
        guard case .accepted = secondResult else {
            Issue.record("expected second .accepted, got \(secondResult)")
            return
        }
        await drainGate.open()

        // The bumped lifecycle event fires synchronously from the
        // front door (the call site runs the callback after
        // releasing the lock); no need to wait on the writer.
        let bumpedAdmit = sink.lines(matching: "event=ssdAdmit")
            .filter { $0.contains("id=first") && $0.contains("outcome=droppedByteBudget") }
        #expect(bumpedAdmit.count == 1)

        let bumpedCallback = sink.lines(matching: "event=storageRefDropCallback")
            .filter { $0.contains("id=first") && $0.contains("reason=backpressureOldest") }
        #expect(bumpedCallback.count == 1)
    }

    @Test
    func ssdMissFiresOnFingerprintMismatchHydration() {
        // Seed a manifest with a partition whose fingerprint does
        // not match the caller's expected value; loadSync must
        // fire `ssdMiss(reason=fingerprintMismatch)` from inside
        // its terminal failure branch.
        let (config, root) = makeConfig()
        defer { cleanup(root) }
        let (sink, uninstall) = makeSink()
        defer { uninstall() }

        let store = SSDSnapshotStore(config: config)
        store.registerPartition(
            makePartitionMeta(fingerprint: String(repeating: "a", count: 64)),
            digest: "abcd1234"
        )
        let descriptor = makeDescriptor(
            id: "snap-mismatch",
            checkpointType: "leaf",
            bytes: 2_048,
            lastAccessAt: 1
        )
        store.seedDescriptorForTesting(descriptor)

        let storageRef = SnapshotStorageRef(
            snapshotID: descriptor.snapshotID,
            partitionDigest: descriptor.partitionDigest,
            tokenOffset: descriptor.tokenOffset,
            checkpointType: .leaf,
            bytesOnDisk: descriptor.bytes,
            lastAccessTime: .now,
            committed: true
        )

        let result = store.loadSync(
            storageRef: storageRef,
            expectedFingerprint: String(repeating: "b", count: 64)
        )
        #expect(result == nil)

        let missLines = sink.lines(matching: "event=ssdMiss")
            .filter { $0.contains("id=\(descriptor.snapshotID)") }
        #expect(missLines.count == 1)
        #expect(missLines.first?.contains("reason=fingerprintMismatch") == true)
    }
}
