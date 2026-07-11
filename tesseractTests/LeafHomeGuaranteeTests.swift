import Foundation
import Testing

@testable import Tesseract_Agent

/// Regression suite for issue #148 (ADR-0019): the newest turn must never
/// be lost. Each test reproduces one of the audited 2026-07-04 defects —
/// the fresh leaf evicted by its own admission, floor-less drains, and
/// in-flight restore paths left unprotected — and pins the corrected
/// behavior: the **Budget Floor** is honored on every drain, and its
/// membership is the in-flight restore pins plus the single
/// most-recently-extended leaf (**Leaf Home Guarantee**).
@MainActor
struct LeafHomeGuaranteeTests {

    private let key = CachePartitionKey(
        modelID: "leaf-home-guarantee-test", kvBits: nil, kvGroupSize: 64
    )

    private var snapBytes: Int {
        PrefixCacheTestFixtures.makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
    }

    private func admitSystem(_ manager: PrefixCacheManager, tokens: [Int]) {
        manager.admit(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: tokens + [9_999],
                candidates: [
                    .ramOnly(
                        PrefixCacheTestFixtures.makeUniformSnapshot(
                            offset: tokens.count, type: .system
                        ))
                ],
                partitionKey: key
            )!)
    }

    /// Defect 1 (2026-07-04 audit, `PrefixCacheManager.swift:986` before
    /// the fix): `admit` drained without the floor, so under a collapsed
    /// budget the eviction loop's last victim was the leaf the admission
    /// had just captured — observable as `capturedThenEvicted`. Now the
    /// admission drain protects the freshest leaf and the drain falls
    /// back to the type-shielded `.system` body instead.
    @Test func freshLeafSurvivesItsOwnAdmissionDrain() {
        // Budget holds exactly one uniform snapshot — the collapsed-band
        // shape from the incident (budget below system + leaf).
        let manager = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        admitSystem(manager, tokens: Array(1...10))
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(20...39), partitionKey: key)  // stale leaf, offset 20

        let diagnostics = PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(40...69), partitionKey: key)  // fresh leaf, offset 30

        // The fresh leaf was NOT evicted by its own admission…
        let freshLeafEvicted = diagnostics.evictions.contains { event in
            event.checkpointType == .leaf && event.offset == 30
        }
        #expect(!freshLeafEvicted)
        #expect(manager.lookup(tokens: Array(40...69), partitionKey: key).snapshot != nil)
        // …and the budget invariant still holds: everything else drained.
        #expect(manager.stats.snapshotCount == 1)
        #expect(manager.totalSnapshotBytes <= manager.memoryBudgetBytes)
    }

    /// The floor is honored on every drain — the explicit zero-budget
    /// override included. Only the freshest leaf survives a full drain;
    /// nothing else does.
    @Test func zeroBudgetDrainKeepsTheFreshestLeaf() {
        let manager = PrefixCacheManager(memoryBudgetBytes: snapBytes * 10)
        admitSystem(manager, tokens: Array(1...10))
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(20...29), partitionKey: key)
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(40...49), partitionKey: key)

        manager.setMemoryBudget(0)

        #expect(manager.stats.snapshotCount == 1)
        #expect(manager.lookup(tokens: Array(40...49), partitionKey: key).snapshot != nil)
    }

    /// An in-flight request's pinned restore path is a floor member: no
    /// drain may evict the body the request restored from, until
    /// `completeRequest` releases the pin (ADR-0019).
    @Test func pinnedRestorePathSurvivesDrainsUntilReleased() {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 10, tieredStore: store
        )
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(20...39), partitionKey: key)  // restore base
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(40...49), partitionKey: key)  // freshest leaf

        let tree = store.tree(for: key)!
        let (baseNode, _) = tree.findBestSnapshot(
            tokens: Array(20...39), updateAccess: false
        )!
        let requestID = UUID()
        manager.pinRestorePath(node: baseNode, requestID: requestID)
        #expect(manager.budgetFloorBytes() == snapBytes * 2, "pin + freshest leaf")

        manager.setMemoryBudget(0)
        #expect(manager.lookup(tokens: Array(20...39), partitionKey: key).snapshot != nil)
        #expect(manager.lookup(tokens: Array(40...49), partitionKey: key).snapshot != nil)

        manager.completeRequest(requestID: requestID)
        manager.setMemoryBudget(0)
        #expect(manager.lookup(tokens: Array(20...39), partitionKey: key).snapshot == nil)
        #expect(manager.lookup(tokens: Array(40...49), partitionKey: key).snapshot != nil)
    }

    /// End-to-end pin plumbing: `resolve(pinningRestorePathFor:)` pins
    /// the hit node for the requesting completion, and the pin holds
    /// through a pressure collapse that would otherwise evict it.
    @Test func resolvePinsTheHitNodeForTheRequest() async {
        let pressure = InMemoryMemoryPressureSource()
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 10, pressureSource: pressure
        )
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(20...39), partitionKey: key)  // will be resolved
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(40...49), partitionKey: key)

        let requestID = UUID()
        let diagnostics = PrefixCacheDiagnostics.Context(
            requestID: requestID, modelID: "leaf-home-guarantee-test",
            kvBits: nil, kvGroupSize: 64
        )
        let resolved = await manager.resolve(
            tokens: Array(20...39),
            promptTokenCount: 20,
            partitionKey: key,
            modelFingerprint: nil,
            diagnostics: diagnostics,
            pinningRestorePathFor: requestID
        )
        #expect(resolved.lookup.snapshot != nil)

        // A later turn makes a different leaf the freshest — the resolved
        // node is now protected only by its pin.
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(60...79), partitionKey: key)

        pressure.send(.critical)
        #expect(manager.lookup(tokens: Array(20...39), partitionKey: key).snapshot != nil)

        // Lookups bump recency — re-touch the newest leaf so it, not the
        // just-asserted pinned node, is the freshest-leaf floor member
        // for the second collapse.
        #expect(manager.lookup(tokens: Array(60...79), partitionKey: key).snapshot != nil)

        manager.completeRequest(requestID: requestID)
        pressure.send(.critical)
        #expect(manager.lookup(tokens: Array(20...39), partitionKey: key).snapshot == nil)
        #expect(manager.lookup(tokens: Array(60...79), partitionKey: key).snapshot != nil)
    }

    // MARK: - Enqueue-before-delete (defects 3+4: supersession must not
    // outrun durability)

    private var ssdKey: CachePartitionKey {
        CachePartitionKey(
            modelID: "leaf-home-guarantee-ssd", kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "e", count: 64)
        )
    }

    private func admitSSDLeaf(
        _ manager: PrefixCacheManager,
        tokens: [Int],
        payloadBytes: Int
    ) -> PrefixCacheManager.StoreDiagnostics {
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: tokens,
                snapshot: PrefixCacheTestFixtures.makeUniformSnapshot(
                    offset: tokens.count, type: .leaf
                ),
                storage: .ramAndSSD(
                    PrefixCacheTestFixtures.makeLeafPayload(
                        bytes: payloadBytes, tokenOffset: tokens.count
                    )),
                partitionKey: ssdKey
            )!)
    }

    /// Defect 4 repro (2026-07-04 audit): the old supersede-first
    /// ordering deleted the ancestor backing before the new write was
    /// even enqueued, so a writer-side rejection lost the new leaf AND
    /// the previous turn's warm-start fallback. Now the deletion waits
    /// for the commit: a dropped write leaves the ancestor SSD-resident
    /// and hittable.
    @Test func droppedFullWritePreservesSupersededAncestorBacking() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "leaf-home-guarantee",
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 10_000
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let ancestorTokens = Array(1...10)
        _ = admitSSDLeaf(manager, tokens: ancestorTokens, payloadBytes: 2_000)
        await store.flush()
        let tree = store.tree(for: ssdKey)!
        let ancestorID = tree.findBestSnapshot(
            tokens: ancestorTokens, updateAccess: false, includeSnapshotRefs: true
        )!.0.state.refID!

        // The descendant's payload exceeds the whole SSD budget — the
        // writer must drop it (`exceedsBudget`), and that drop must not
        // cost the ancestor its backing.
        let diagnostics = admitSSDLeaf(
            manager, tokens: Array(1...15), payloadBytes: 20_000
        )
        #expect(diagnostics.supersededLeaves.map(\.mode) == [.deferredDelete])
        await store.flush()
        let settled = await waitUntil { store.pendingSnapshotRefIDs.isEmpty }
        #expect(settled)

        // The ancestor is still the SSD resident and still hittable.
        let residents = store.ssdResidency()!.idsByRecency
        #expect(residents == [ancestorID])
        let ancestorLookup = manager.lookup(tokens: ancestorTokens, partitionKey: ssdKey)
        guard case .ssdHit = ancestorLookup.reason else {
            Issue.record("expected ssdHit, got \(ancestorLookup.reason)")
            return
        }
        // The new leaf's RAM body also survives (its home is RAM).
        #expect(manager.lookup(tokens: Array(1...15), partitionKey: ssdKey).snapshot != nil)
    }

    /// The happy path of the same invariant: the superseded ancestor's
    /// backing is deleted exactly when the replacing write commits.
    @Test func committedFullWriteDeletesSupersededAncestorBacking() async {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "leaf-home-guarantee",
            ramBudgetBytes: 100_000_000,
            ssdBudgetBytes: 10_000
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let ancestorTokens = Array(1...10)
        _ = admitSSDLeaf(manager, tokens: ancestorTokens, payloadBytes: 2_000)
        await store.flush()
        let tree = store.tree(for: ssdKey)!
        let ancestorID = tree.findBestSnapshot(
            tokens: ancestorTokens, updateAccess: false, includeSnapshotRefs: true
        )!.0.state.refID!

        let diagnostics = admitSSDLeaf(
            manager, tokens: Array(1...15), payloadBytes: 2_000
        )
        #expect(diagnostics.supersededLeaves.map(\.mode) == [.deferredDelete])
        await store.flush()

        // Commit lands on MainActor asynchronously; the deferred
        // deletion rides it.
        let ancestorGone = await waitUntil {
            !store.ssdResidency()!.idsByRecency.contains(ancestorID)
        }
        #expect(ancestorGone)
        let residents = store.ssdResidency()!.idsByRecency
        #expect(residents.count == 1)
        #expect(manager.lookup(tokens: Array(1...15), partitionKey: ssdKey).snapshot != nil)
    }
}
