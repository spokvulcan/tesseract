//
//  SSDWriteEagernessTests.swift
//  tesseractTests
//
//  Adaptive write eagerness (ADR-0019, PRD #150): the pure deferral
//  policy, the manager-level defer/promote flow (skip redundant SSD
//  copies while RAM is comfortable; reuse earns a deferred-class
//  promotion write), and the writer-side Storage Activity Gate
//  scheduling (deferred items wait out hydration/prefill windows,
//  flushes force-drain them, extension bases are promoted to
//  non-deferrable so base-before-suffix ordering survives).
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Policy

struct SSDWriteEagernessPolicyTests {

    @Test func defersColdNodeWhileRAMComfortable() {
        #expect(
            SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .branchPoint,
                nodeHitCount: 0, residentBytes: 50, budgetBytes: 100, bandRetreating: false
            ))
        #expect(
            SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .leaf,
                nodeHitCount: 1, residentBytes: 75, budgetBytes: 100, bandRetreating: false
            ))
    }

    @Test func writesThroughOnceHitCountProvesReuse() {
        #expect(
            !SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .branchPoint,
                nodeHitCount: SSDWriteEagernessPolicy.hitCountThreshold,
                residentBytes: 0, budgetBytes: 100, bandRetreating: false
            ))
    }

    @Test func writesThroughAboveComfortFraction() {
        #expect(
            !SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .branchPoint,
                nodeHitCount: 0, residentBytes: 76, budgetBytes: 100, bandRetreating: false
            ))
    }

    @Test func writesThroughWhileBandRetreats() {
        #expect(
            !SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .branchPoint,
                nodeHitCount: 0, residentBytes: 0, budgetBytes: 100, bandRetreating: true
            ))
    }

    /// `.system` is never deferrable, even cold under a fully
    /// comfortable RAM budget — see the `mayDefer` doc (issue #165).
    @Test func systemCheckpointNeverDeferred() {
        #expect(
            !SSDWriteEagernessPolicy.mayDefer(
                checkpointType: .system,
                nodeHitCount: 0, residentBytes: 0, budgetBytes: 100, bandRetreating: false
            ))
    }
}

// MARK: - Manager defer/promote flow

@MainActor
struct SSDWriteEagernessTests {

    private var key: CachePartitionKey {
        CachePartitionKey(
            modelID: "write-eagerness-test", kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "f", count: 64)
        )
    }

    private func admitSSDCheckpoint(
        _ manager: PrefixCacheManager, prefixTokens: [Int],
        type: HybridCacheSnapshot.CheckpointType = .branchPoint
    ) {
        manager.admit(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: prefixTokens + [9_999],
                candidates: [
                    .ramAndSSD(
                        PrefixCacheTestFixtures.makeUniformSnapshot(
                            offset: prefixTokens.count, type: type
                        ),
                        payload: PrefixCacheTestFixtures.makeLeafPayload(
                            bytes: 2_000, tokenOffset: prefixTokens.count
                        )
                    )
                ],
                partitionKey: key
            )!)
    }

    private func nodeState(
        _ store: TieredSnapshotStore, tokens: [Int]
    ) -> SnapshotState? {
        store.tree(for: key)?
            .findBestSnapshot(tokens: tokens, updateAccess: false, includeSnapshotRefs: true)?
            .0.state
    }

    /// The core deferral: a cold checkpoint write-through under a
    /// comfortable RAM budget skips the SSD write entirely — no ref,
    /// no SSD resident, one `eagernessDeferrals` count, one
    /// `ssdWriteDeferred` event.
    @Test func coldCheckpointDeferredWhileRAMComfortable() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-defer",
            ramBudgetBytes: 100_000_000,
            adaptiveWriteEagerness: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let sink = RecordingLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }

        admitSSDCheckpoint(manager, prefixTokens: Array(1...10))
        await store.flush()

        let state = nodeState(store, tokens: Array(1...10))
        #expect(state?.ref == nil)
        #expect(state?.body != nil)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 1)
        #expect(store.ssdResidency()!.idsByRecency.isEmpty)
        #expect(sink.drain().contains { $0.contains("event=ssdWriteDeferred") })
    }

    /// A cold `.system` checkpoint lands on SSD even with eagerness on
    /// and RAM fully comfortable (issue #165, prefix-cache-e2e
    /// `requestX3_stable_prefix_reused_across_users`).
    @Test func coldSystemCheckpointWritesThroughDespiteComfort() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-system-writethrough",
            ramBudgetBytes: 100_000_000,
            adaptiveWriteEagerness: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        admitSSDCheckpoint(manager, prefixTokens: Array(1...10), type: .system)
        await store.flush()

        let committed = await waitUntil {
            self.nodeState(store, tokens: Array(1...10))?.committed == true
        }
        #expect(committed)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 0)
    }

    /// The guarantee-class end-of-turn leaf is never deferred: it lands
    /// on SSD even with eagerness on and RAM fully comfortable.
    @Test func endOfTurnLeafNeverDeferred() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-guarantee",
            ramBudgetBytes: 100_000_000,
            adaptiveWriteEagerness: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: tokens,
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                    offset: tokens.count, type: .leaf
                ),
                storage: .ramAndSSD(
                    PrefixCacheTestFixtures.makeLeafPayload(
                        bytes: 2_000, tokenOffset: tokens.count
                    )),
                partitionKey: key
            )!)
        await store.flush()

        let committed = await waitUntil {
            self.nodeState(store, tokens: tokens)?.committed == true
        }
        #expect(committed)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 0)
    }

    /// A node that already proved reuse (hit count at threshold) writes
    /// through at admission — deferral is only for cold nodes.
    @Test func provenHotNodeWritesThroughAtAdmission() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-hot-node",
            ramBudgetBytes: 100_000_000,
            adaptiveWriteEagerness: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let prefix = Array(1...10)
        // Seed the node RAM-only, then hit it to the threshold.
        manager.admit(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: prefix + [9_999],
                candidates: [
                    .ramOnly(
                        PrefixCacheTestFixtures.makeUniformSnapshot(
                            offset: prefix.count, type: .branchPoint
                        ))
                ],
                partitionKey: key
            )!)
        for _ in 0..<SSDWriteEagernessPolicy.hitCountThreshold {
            _ = manager.lookup(tokens: prefix, partitionKey: key)
        }

        admitSSDCheckpoint(manager, prefixTokens: prefix)
        await store.flush()

        let committed = await waitUntil {
            self.nodeState(store, tokens: prefix)?.committed == true
        }
        #expect(committed)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 0)
    }

    /// The promotion loop: a deferred checkpoint that gets hit to the
    /// threshold earns its deferred-class SSD write from the lookup
    /// path — once, ever (`ssdPromotionAttempted` latches).
    @Test func hotLookupPromotesDeferredNodeOnce() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-promotion",
            ramBudgetBytes: 100_000_000,
            demotionPayloadExtractor: { snapshot in
                PrefixCacheTestFixtures.makeLeafPayload(
                    bytes: 2_000, tokenOffset: snapshot.tokenOffset
                )
            },
            adaptiveWriteEagerness: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let sink = RecordingLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }

        let prefix = Array(1...10)
        admitSSDCheckpoint(manager, prefixTokens: prefix)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 1)

        for _ in 0..<SSDWriteEagernessPolicy.hitCountThreshold {
            _ = manager.lookup(tokens: prefix, partitionKey: key)
        }

        // The promotion admits on a follow-up MainActor task; wait for
        // the ref, then drain the writer for the durable commit.
        let promoted = await waitUntil {
            self.nodeState(store, tokens: prefix)?.ref != nil
        }
        #expect(promoted)
        await store.flush()
        let committed = await waitUntil {
            self.nodeState(store, tokens: prefix)?.committed == true
        }
        #expect(committed)
        #expect(manager.cumulativeCounters.eagernessPromotions == 1)

        let lines = sink.drain()
        #expect(lines.contains { $0.contains("event=ssdWritePromoted") })
        // The endurance ledger sees the promotion as a deferred-class
        // accepted write.
        #expect(
            lines.contains {
                $0.contains("event=ssdAdmit") && $0.contains("outcome=accepted")
                    && $0.contains("writeClass=deferred")
            })

        // Further hits never re-promote.
        _ = manager.lookup(tokens: prefix, partitionKey: key)
        await store.flush()
        #expect(manager.cumulativeCounters.eagernessPromotions == 1)
    }

    /// Eagerness off (the default, and every pre-existing fixture):
    /// checkpoint write-throughs land on SSD exactly as before.
    @Test func disabledEagernessKeepsWriteThrough() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "eagerness-disabled",
            ramBudgetBytes: 100_000_000
        )
        defer { try? FileManager.default.removeItem(at: root) }

        admitSSDCheckpoint(manager, prefixTokens: Array(1...10))
        await store.flush()

        let committed = await waitUntil {
            self.nodeState(store, tokens: Array(1...10))?.committed == true
        }
        #expect(committed)
        #expect(manager.cumulativeCounters.eagernessDeferrals == 0)
    }

    private final class RecordingLineSink: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []

        var handler: @Sendable (String) -> Void {
            { [weak self] line in
                guard let self else { return }
                self.lock.lock()
                self.lines.append(line)
                self.lock.unlock()
            }
        }

        func drain() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            let copy = lines
            lines.removeAll()
            return copy
        }
    }
}

// MARK: - Writer-side gate scheduling

@MainActor
struct StorageActivityGateSchedulingTests {

    private func makeConfig() -> (SSDPrefixCacheConfig, URL) {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("gate-sched-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        return (
            SSDPrefixCacheConfig(
                enabled: true, rootURL: rootURL,
                budgetBytes: 1_000_000, maxPendingBytes: 10_000_000
            ), rootURL
        )
    }

    private func makeDescriptor(
        id: String = UUID().uuidString,
        bytes: Int = 1_024,
        tokenOffset: Int = 4_096,
        segmentBaseOffset: Int = 0
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: "abcd1234",
            pathFromRoot: [1, 2, 3],
            tokenOffset: tokenOffset,
            checkpointType: "leaf",
            bytes: bytes,
            segmentBaseOffset: segmentBaseOffset,
            createdAt: 100_000,
            lastAccessAt: 0,
            fileRelativePath: "partitions/abcd1234/snapshots/\(id.prefix(1))/\(id).safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func makeStore(
        config: SSDPrefixCacheConfig,
        gate: StorageActivityGate,
        onCommit: @escaping @Sendable (SSDCommitInfo) -> Void
    ) -> SSDSnapshotStore {
        let store = SSDSnapshotStore(
            config: config,
            manifestDebounce: .milliseconds(20),
            activityGate: gate,
            onCommit: onCommit
        )
        store.registerPartition(
            PartitionMeta(
                modelID: "gate-sched-test",
                modelFingerprint: String(repeating: "a", count: 64),
                kvBits: 8,
                kvGroupSize: 64,
                createdAt: 100_000,
                schemaVersion: SnapshotManifestSchema.currentVersion
            ), digest: "abcd1234")
        return store
    }

    private final class CommitTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var _committed: [String] = []
        var committed: [String] {
            lock.lock()
            defer { lock.unlock() }
            return _committed
        }
        func record(_ id: String) {
            lock.lock()
            _committed.append(id)
            lock.unlock()
        }
    }

    /// A deferrable item holds while the gate is busy and lands as
    /// soon as the gate quiets down (via the delayed re-wake).
    @Test func deferrableWriteWaitsOutBusyGate() async {
        let (config, root) = makeConfig()
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = StorageActivityGate()
        let tracker = CommitTracker()
        let store = makeStore(config: config, gate: gate) { tracker.record($0.snapshotID) }

        gate.hydrationDidBegin()
        let result = store.tryEnqueue(
            payload: PrefixCacheTestFixtures.makeLeafPayload(bytes: 1_024),
            descriptor: makeDescriptor(id: "deferred-1"),
            deferrable: true
        )
        guard case .accepted = result else {
            Issue.record("expected accepted, got \(result)")
            return
        }

        // Longer than the writer's re-check interval: the hold is a
        // scheduling decision, not a race.
        try? await Task.sleep(for: .milliseconds(700))
        #expect(tracker.committed.isEmpty)

        gate.hydrationDidEnd()
        let landed = await waitUntil { tracker.committed == ["deferred-1"] }
        #expect(landed)
    }

    /// A busy gate never holds non-deferrable traffic.
    @Test func writeThroughIgnoresBusyGate() async {
        let (config, root) = makeConfig()
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = StorageActivityGate()
        let tracker = CommitTracker()
        let store = makeStore(config: config, gate: gate) { tracker.record($0.snapshotID) }

        gate.prefillDidBegin()
        defer { gate.prefillDidEnd() }
        _ = store.tryEnqueue(
            payload: PrefixCacheTestFixtures.makeLeafPayload(bytes: 1_024),
            descriptor: makeDescriptor(id: "through-1")
        )

        let landed = await waitUntil { tracker.committed == ["through-1"] }
        #expect(landed)
    }

    /// `flushAsync` (unload / benchmark drains) force-drains deferred
    /// items regardless of gate state — durability outranks bandwidth.
    @Test func flushForceDrainsDeferredWhileGateBusy() async {
        let (config, root) = makeConfig()
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = StorageActivityGate()
        let tracker = CommitTracker()
        let store = makeStore(config: config, gate: gate) { tracker.record($0.snapshotID) }

        gate.hydrationDidBegin()
        defer { gate.hydrationDidEnd() }
        _ = store.tryEnqueue(
            payload: PrefixCacheTestFixtures.makeLeafPayload(bytes: 1_024),
            descriptor: makeDescriptor(id: "deferred-flush"),
            deferrable: true
        )

        await store.flushAsync()
        #expect(tracker.committed == ["deferred-flush"])
    }

    /// An extension naming a still-queued deferrable base promotes the
    /// base to non-deferrable: base-before-suffix ordering survives a
    /// busy gate instead of the extension jumping its own base.
    @Test func extensionPromotesQueuedDeferrableBase() async {
        let (config, root) = makeConfig()
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = StorageActivityGate()
        let tracker = CommitTracker()
        let store = makeStore(config: config, gate: gate) { tracker.record($0.snapshotID) }

        gate.hydrationDidBegin()
        defer { gate.hydrationDidEnd() }
        _ = store.tryEnqueue(
            payload: PrefixCacheTestFixtures.makeLeafPayload(bytes: 1_024, tokenOffset: 10),
            descriptor: makeDescriptor(id: "base-1", tokenOffset: 10),
            deferrable: true
        )
        _ = store.tryEnqueue(
            payload: PrefixCacheTestFixtures.makeLeafPayload(
                bytes: 512, tokenOffset: 20,
                extending: SnapshotExtension(baseSnapshotID: "base-1", baseOffset: 10)
            ),
            descriptor: makeDescriptor(id: "ext-1", tokenOffset: 20, segmentBaseOffset: 10)
        )

        // Both land while the gate stays busy, base first.
        let landed = await waitUntil { tracker.committed.count == 2 }
        #expect(landed)
        #expect(tracker.committed == ["base-1", "ext-1"])
    }
}
