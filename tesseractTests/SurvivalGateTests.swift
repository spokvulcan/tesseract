import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Slice #90 (PRD #82): the **Survival Gate** — SSD writes happen only
/// when the incoming chain would survive the eviction its own admission
/// triggers. End-of-turn leaves bypass; a gated demotion terminal-drops;
/// a gated leaf degrades to RAM-only with supersession *preserve*; an
/// unfilled ledger admits everything. Hermetic — per-test scratch SSD
/// roots.
@MainActor
struct SurvivalGateTests {

    private let fingerprint = String(repeating: "f", count: 64)

    private var key: CachePartitionKey {
        CachePartitionKey(
            modelID: "survival-gate-test",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: fingerprint
        )
    }

    private func makeFixture(
        ramBudgetBytes: Int,
        ssdBudgetBytes: Int,
        extractor: ((HybridCacheSnapshot) -> SnapshotPayload?)? = nil
    ) -> (manager: PrefixCacheManager, store: TieredSnapshotStore, root: URL) {
        PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "survival-gate",
            ramBudgetBytes: ramBudgetBytes,
            ssdBudgetBytes: ssdBudgetBytes,
            demotionPayloadExtractor: extractor
        )
    }

    private func makePayload(bytes: Int) -> SnapshotPayload {
        PrefixCacheTestFixtures.makeLeafPayload(bytes: bytes)
    }

    /// Register the test partition with the SSD tier and seed one warm
    /// committed resident so the ledger is full and the gate has
    /// something to compare against.
    private func seedWarmResident(
        _ store: TieredSnapshotStore,
        bytes: Int
    ) {
        let meta = PartitionMeta(
            modelID: key.modelID,
            modelFingerprint: fingerprint,
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: Date().timeIntervalSinceReferenceDate,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        store.registerPartition(meta, for: key)
        let id = UUID().uuidString
        store.ssdStoreForTesting!.seedDescriptorForTesting(
            PersistedSnapshotDescriptor(
                snapshotID: id,
                partitionDigest: key.partitionDigest,
                pathFromRoot: Array(100...120),
                tokenOffset: 1_000,
                checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
                bytes: bytes,
                createdAt: Date().timeIntervalSinceReferenceDate,
                lastAccessAt: Date().timeIntervalSinceReferenceDate,
                fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                    snapshotID: id,
                    partitionDigest: key.partitionDigest
                ),
                schemaVersion: SnapshotManifestSchema.currentVersion
            ))
    }

    @discardableResult
    private func admitLeaf(
        _ manager: PrefixCacheManager,
        tokens: [Int],
        storage: SnapshotAdmission.Storage = .ramOnly,
        endOfTurn: Bool = true
    ) -> PrefixCacheManager.StoreDiagnostics {
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: tokens, partitionKey: key,
            storage: storage, endOfTurn: endOfTurn
        )
    }

    // MARK: - Demotion gating

    /// The headline case: a cold (ancient-recency) demotion against a
    /// full ledger of warmer chains skips the SSD write entirely — no
    /// pending write, no churn — and the RAM drop settles terminal.
    @Test func coldDemotionAgainstWarmerChainsSkipsTheWrite() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: snapBytes,
            ssdBudgetBytes: 10_000,
            extractor: { _ in self.makePayload(bytes: 5_000) }
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 9_000)

        let victimTokens = Array(1...10)
        _ = admitLeaf(manager, tokens: victimTokens)
        let tree = store.tree(for: key)!
        let (victim, _) = tree.findBestSnapshot(
            tokens: victimTokens, updateAccess: false
        )!
        victim.lastAccessTime = .now - .seconds(3_600)

        _ = admitLeaf(manager, tokens: Array(20...29))

        #expect(manager.cumulativeCounters.survivalGateSkips == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 1)
        #expect(manager.cumulativeCounters.recoveredEvictions == 0)
        #expect(store.pendingRefCountForTesting == 0)
        #expect(manager.lookup(tokens: victimTokens, partitionKey: key).snapshot == nil)
    }

    /// A demotion that *would* survive (the ledger is unfilled) writes
    /// through exactly as in slice #87 — the gate is inert without
    /// contention.
    @Test func demotionAgainstUnfilledLedgerStillWrites() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: snapBytes,
            ssdBudgetBytes: 1_000_000,
            extractor: { _ in self.makePayload(bytes: 5_000) }
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 9_000)

        _ = admitLeaf(manager, tokens: Array(1...10))
        _ = admitLeaf(manager, tokens: Array(20...29))

        #expect(manager.cumulativeCounters.survivalGateSkips == 0)
        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 0)
    }

    // MARK: - Leaf bypass and leaf gating

    /// End-of-turn leaf admissions always write through — even one the
    /// gate would reject (here: bigger than the whole SSD budget, so
    /// the simulation can never fit it). The pending ref lands; the
    /// writer's own cut remains the final authority later.
    @Test func endOfTurnLeafBypassesTheGate() {
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 1_000
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 900)

        let tokens = Array(1...10)
        _ = admitLeaf(
            manager, tokens: tokens,
            storage: .ramAndSSD(makePayload(bytes: 5_000))
        )

        #expect(manager.cumulativeCounters.survivalGateSkips == 0)
        let tree = store.tree(for: key)!
        let (node, _) = tree.findBestSnapshot(tokens: tokens, updateAccess: false)!
        #expect(node.state.ref != nil, "end-of-turn leaf must enqueue despite the gate")
    }

    /// A gated-out (non-end-of-turn) leaf degrades to RAM-only and the
    /// ancestor's SSD backing is preserved — it remains the warm-start
    /// fallback and the next turn's extension base.
    @Test func gatedOutLeafDegradesToRAMOnlyWithPreserve() {
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 1_000
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 900)

        // Ancestor leaf with a committed SSD backing on the same path.
        let ancestorTokens = Array(1...10)
        _ = admitLeaf(manager, tokens: ancestorTokens)
        let tree = store.tree(for: key)!
        let (ancestor, _) = tree.findBestSnapshot(
            tokens: ancestorTokens, updateAccess: false
        )!
        let ancestorRef = PrefixCacheTestFixtures.makeRef(tokenOffset: ancestorTokens.count)
        tree.admit(node: ancestor, ref: ancestorRef)
        tree.commitRef(node: ancestor, expectedID: ancestorRef.snapshotID)

        let leafTokens = Array(1...20)
        let diagnostics = admitLeaf(
            manager, tokens: leafTokens,
            storage: .ramAndSSD(makePayload(bytes: 5_000)),
            endOfTurn: false
        )

        #expect(manager.cumulativeCounters.survivalGateSkips == 1)
        #expect(
            diagnostics.supersededLeaves.map(\.mode)
                == [PrefixCacheManager.LeafSupersession.Mode.preserved])
        #expect(ancestor.state.refID == ancestorRef.snapshotID)
        let (leaf, _) = tree.findBestSnapshot(tokens: leafTokens, updateAccess: false)!
        #expect(leaf.state.ref == nil, "gated-out leaf stays RAM-only")
        #expect(leaf.state.body != nil)
    }

    // MARK: - Checkpoint gating and cold start

    /// A checkpoint write-through that cannot survive (bigger than the
    /// SSD budget) skips its write — the RAM body stays, only the SSD
    /// copy is forgone — and the skip is counted.
    @Test func gatedCheckpointSkipsWriteAndKeepsRAMBody() {
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 1_000
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 900)

        let tokens = Array(1...10)
        manager.admit(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: tokens + [999],
                candidates: [
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                            offset: tokens.count, type: .branchPoint
                        ),
                        storage: .ramAndSSD(makePayload(bytes: 5_000))
                    )
                ],
                partitionKey: key
            )!)

        #expect(manager.cumulativeCounters.survivalGateSkips == 1)
        #expect(store.pendingRefCountForTesting == 0)
        let result = manager.lookup(tokens: tokens, partitionKey: key)
        #expect(result.snapshot != nil, "the RAM body is untouched by a gate skip")
    }

    /// An unfilled ledger admits everything: the same checkpoint
    /// write-through proceeds untouched when there is room — cold-start
    /// behavior is identical to today.
    @Test func unfilledLedgerAdmitsCheckpointWrites() {
        let (manager, store, root) = makeFixture(
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 1_000_000
        )
        defer { try? FileManager.default.removeItem(at: root) }
        seedWarmResident(store, bytes: 900)

        let tokens = Array(1...10)
        manager.admit(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: tokens + [999],
                candidates: [
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                            offset: tokens.count, type: .branchPoint
                        ),
                        storage: .ramAndSSD(makePayload(bytes: 5_000))
                    )
                ],
                partitionKey: key
            )!)

        #expect(manager.cumulativeCounters.survivalGateSkips == 0)
        #expect(store.pendingRefCountForTesting == 1)
    }
}
