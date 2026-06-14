import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Slice #87 (PRD #82): **Snapshot Demotion** — evict-to-fit persists an
/// unbacked victim to SSD before dropping its RAM body, so the loss is
/// recovered (next hit pays a hydration) instead of terminal (next hit
/// pays a re-prefill). Includes the flagged invariant: a demotion write
/// never refreshes the ledger's recency. Hermetic — every test gets its
/// own scratch SSD root.
@MainActor
struct SnapshotDemotionTests {

    private let key = CachePartitionKey(
        modelID: "demotion-test",
        kvBits: nil,
        kvGroupSize: 64,
        modelFingerprint: String(repeating: "d", count: 64)
    )

    /// SSD-enabled manager with the production-shaped demotion extractor
    /// (the **Server Completion** module's extraction edge) injected.
    private func makeDemotingManager(
        memoryBudgetBytes: Int,
        ssdBudgetBytes: Int = 10_000_000,
        extractor: ((HybridCacheSnapshot) -> SnapshotPayload?)? = { snapshot in
            ServerCompletion.extractSnapshotPayload(snapshot)
        }
    ) -> (manager: PrefixCacheManager, store: TieredSnapshotStore, root: URL) {
        PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "snapshot-demotion",
            ramBudgetBytes: memoryBudgetBytes,
            ssdBudgetBytes: ssdBudgetBytes,
            demotionPayloadExtractor: extractor
        )
    }

    private func admitLeaf(
        _ manager: PrefixCacheManager,
        tokens: [Int]
    ) {
        PrefixCacheTestFixtures.admitUniformLeaf(manager, tokens: tokens, partitionKey: key)
    }

    /// Poll `condition` on MainActor until true or timeout — the writer's
    /// commit callback hops back to MainActor asynchronously.
    private func waitUntil(
        timeout: Duration = .seconds(5),
        _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let start = ContinuousClock.now
        while ContinuousClock.now - start < timeout {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    // MARK: - Demote-don't-drop

    /// The headline behavior: budget pressure on an unbacked victim ends
    /// with the node SSD-backed and hittable — counted recovered — and a
    /// committed write that the next lookup surfaces as `.ssdHit`.
    @Test func evictionDemotesUnbackedVictimToSSD() async {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let (manager, store, root) = makeDemotingManager(memoryBudgetBytes: snapBytes)
        defer { try? FileManager.default.removeItem(at: root) }

        let victimTokens = Array(1...10)
        admitLeaf(manager, tokens: victimTokens)
        // Second leaf overflows the one-snapshot budget; the older first
        // leaf is the victim and demotes instead of vanishing.
        admitLeaf(manager, tokens: Array(20...29))

        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 0)
        #expect(manager.totalSnapshotBytes <= snapBytes)

        // The demoted node survives in the tree with a pending ref
        // (state 3) until the writer commits it to state 5 (ssdOnly).
        let tree = store.tree(for: key)!
        await store.ssdStoreForTesting!.flushAsync()
        let committed = await waitUntil {
            if case .ssdHit = manager.lookup(
                tokens: victimTokens, partitionKey: key
            ).reason {
                return true
            }
            return false
        }
        #expect(committed, "demoted victim should become an SSD hit after commit")
        // The demoted leaf is also the next turn's extension base —
        // `deepestRefBearingLeaf` is strict-ancestor, so probe past it.
        #expect(tree.deepestRefBearingLeaf(tokens: victimTokens + [99]) != nil)
    }

    /// SSD-disabled (RAM-only store) behavior matches today: the drop is
    /// terminal and the extractor is never consulted.
    @Test func terminalDropWhenSSDUnavailable() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        var extractorCalls = 0
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: TieredSnapshotStore(ssdConfig: nil),
            demotionPayloadExtractor: { _ in
                extractorCalls += 1
                return nil
            }
        )

        let victimTokens = Array(1...10)
        admitLeaf(manager, tokens: victimTokens)
        admitLeaf(manager, tokens: Array(20...29))

        #expect(extractorCalls == 0)
        #expect(manager.cumulativeCounters.terminalEvictions == 1)
        #expect(manager.cumulativeCounters.recoveredEvictions == 0)
        #expect(manager.lookup(tokens: victimTokens, partitionKey: key).snapshot == nil)
    }

    /// No extractor injected (test/replay caches) also means terminal —
    /// the pre-demotion behavior is the default, not a regression risk.
    @Test func terminalDropWithoutExtractor() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let (manager, _, root) = makeDemotingManager(
            memoryBudgetBytes: snapBytes, extractor: nil
        )
        defer { try? FileManager.default.removeItem(at: root) }

        admitLeaf(manager, tokens: Array(1...10))
        admitLeaf(manager, tokens: Array(20...29))

        #expect(manager.cumulativeCounters.terminalEvictions == 1)
        #expect(manager.cumulativeCounters.recoveredEvictions == 0)
    }

    /// An already-backed victim (live ref) skips the demotion write — the
    /// body drop alone is the recovery, and a duplicate SSD copy would
    /// supersede-delete the one that exists.
    @Test func backedVictimSkipsDemotionWrite() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        var extractorCalls = 0
        let (manager, store, root) = makeDemotingManager(
            memoryBudgetBytes: snapBytes,
            extractor: { _ in
                extractorCalls += 1
                return nil
            }
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let victimTokens = Array(1...10)
        admitLeaf(manager, tokens: victimTokens)
        let tree = store.tree(for: key)!
        let (node, _) = tree.findBestSnapshot(
            tokens: victimTokens, updateAccess: false
        )!
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: victimTokens.count)
        tree.admit(node: node, ref: ref)
        tree.commitRef(node: node, expectedID: ref.snapshotID)

        admitLeaf(manager, tokens: Array(20...29))

        #expect(extractorCalls == 0)
        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(node.state.refID == ref.snapshotID)
    }

    // MARK: - The flagged invariant: demotion never refreshes recency

    /// End-to-end: a demoted body's ledger entry carries the node's real
    /// (stale) `lastAccessAt`, not the commit time — while an ordinary
    /// write-through admission is stamped fresh at commit. Refreshing
    /// demotions would make every pressure event invert the SSD tier's
    /// recency signal.
    @Test func demotionPreservesStaleRecencyInLedger() async {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let (manager, store, root) = makeDemotingManager(memoryBudgetBytes: snapBytes)
        defer { try? FileManager.default.removeItem(at: root) }

        let victimTokens = Array(1...10)
        admitLeaf(manager, tokens: victimTokens)
        let tree = store.tree(for: key)!
        let (node, _) = tree.findBestSnapshot(
            tokens: victimTokens, updateAccess: false
        )!
        // Make the victim's recency unmistakably ancient.
        node.lastAccessTime = .now - .seconds(3_600)

        admitLeaf(manager, tokens: Array(20...29))
        guard let demotedID = node.state.refID else {
            Issue.record("victim was not demoted")
            return
        }

        await store.ssdStoreForTesting!.flushAsync()
        let committed = await waitUntil { node.state.committed }
        #expect(committed)

        let now = Date().timeIntervalSinceReferenceDate
        let demotedAccessAt = store.ssdStoreForTesting!
            .lastAccessAtForTesting(id: demotedID)
        #expect(demotedAccessAt > 0)
        #expect(
            now - demotedAccessAt > 3_000,
            "demotion must carry the node's stale recency, got age \(now - demotedAccessAt)s"
        )
    }

    /// Ledger-level pin of the same invariant at the commit seam:
    /// `refreshRecency: false` preserves the descriptor's stamp,
    /// the default re-stamps it.
    @Test func ledgerCommitHonorsRefreshRecencyFlag() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("demotion-ledger-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(50)
        )

        let staleStamp = Date().timeIntervalSinceReferenceDate - 7_200
        func makeStaleDescriptor() -> PersistedSnapshotDescriptor {
            SnapshotLedger.makeDescriptor(
                partitionKey: key,
                pathFromRoot: Array(1...10),
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                    offset: 10, type: .leaf
                ),
                payloadBytes: 1_024,
                lastAccessAt: staleStamp
            )
        }

        let demoted = makeStaleDescriptor()
        #expect(ledger.commit(demoted, refreshRecency: false))
        #expect(ledger.lastAccessAtForTesting(id: demoted.snapshotID) == staleStamp)

        let ordinary = makeStaleDescriptor()
        #expect(ledger.commit(ordinary))
        let refreshed = ledger.lastAccessAtForTesting(id: ordinary.snapshotID)
        #expect(Date().timeIntervalSinceReferenceDate - refreshed < 60)
    }
}
