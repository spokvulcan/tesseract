import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of **Snapshot Resolution** — the **Prefix Cache Manager**'s single
/// read-side entry: the radix lookup plus lazy SSD hydration, in one place.
/// Every test drives `PrefixCacheManager.resolve(...)` — the same seam callers
/// cross — never past it into the privatized helpers.
///
/// The SSD-hydration composition (retry cap, re-lookup-after-clear fallback,
/// clear-on-fail vs promote-on-success ordering) is asserted here through
/// `resolve` with an `InMemorySnapshotHydrating` peer — no loaded model, no
/// temp directory, no concrete `SSDSnapshotStore` (ADR-0001's narrow seam, now
/// real). The loaded-model E2E net still proves the composition end-to-end;
/// these move the subtle-path assertions down to a fast tier.
@MainActor
@Suite struct SnapshotResolutionTests {

    @Test(arguments: [6, 8], [false, true])
    func storedAndTransientViewsChooseAndPinThePreferredWarmOrFullBacker(
        warmOffset: Int, transient: Bool
    ) async throws {
        let sessions = ToyModelSessionProvider(
            model: ToyLanguageModel(script: Array(1...12), headDim: 64))
        let (view, warm, full) = try await sessions.withSession { session in
            func capture(_ offset: Int, type: HybridCacheSnapshot.CheckpointType) throws
                -> HybridCacheSnapshot
            {
                let cache = KVCacheSimple()
                cache.state = [
                    MLXArray.ones([1, 1, offset, 64]), MLXArray.ones([1, 1, offset, 64]),
                ]
                let recurrent = MambaCache()
                recurrent.state = [MLXArray([Float(offset)])]
                return try #require(
                    session.captureSnapshot(cache: [cache, recurrent], offset: offset, type: type))
            }
            return try (
                capture(4, type: .branchPoint),
                session.compress(capture(warmOffset, type: .leaf), bits: 8), capture(8, type: .leaf)
            )
        }
        let tier = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: tier)
        let prefix = [1, 2, 3, 4]
        if !transient {
            manager.restoreSnapshot(
                path: prefix, snapshot: view, partitionKey: key, lastAccessTime: .now)
        }
        manager.restoreSnapshot(
            path: prefix + Array(repeating: 20, count: warmOffset - 4), snapshot: warm,
            partitionKey: key, lastAccessTime: .now)
        manager.restoreSnapshot(
            path: prefix + [30, 30, 30, 30], snapshot: full,
            partitionKey: key, lastAccessTime: .now - .seconds(2))
        let context = diagnostics
        let (resolved, handOver) = await manager.resolveHoldingClaim(
            tokens: prefix + [99], partitionKey: key, diagnostics: context,
            transientBoundary: transient ? view : nil)
        let selected = warmOffset == 6 ? warm : full
        #expect(resolved.lookup.backingLeaf?.bodyID == selected.bodyID)
        // The hit is served through the backer, which earns it (ADR-0068).
        let backerNode = try #require(
            tier.getOrCreateTree(for: key).allSnapshotNodes().first {
                $0.state.body?.bodyID == selected.bodyID
            })
        #expect(backerNode.hitCount == 1)
        if !transient {
            let telemetry = manager.makeTelemetrySnapshot()
            #expect(telemetry.viewOnlyBytes == 4)
            #expect(telemetry.warmSnapshotBytes == warm.memoryBytes)
            #expect(telemetry.hotSnapshotBytes == full.memoryBytes)
            #expect(
                telemetry.hotSnapshotBytes + telemetry.warmSnapshotBytes + telemetry.viewOnlyBytes
                    == manager.totalSnapshotBytes)
        }
        // The view restores by copy, and the pins hold until the claim
        // that resolved it concludes.
        await handOver.withClaim { claim in
            let outcome = await sessions.withSession { session in
                await claim.checkOut(
                    resolved, tokens: prefix + [99], maximumAdvance: 1, identityKeySpace: true,
                    in: session)
            }
            guard case .copy(let copy) = outcome else {
                Issue.record("a Prefix-View Checkpoint must restore by copy")
                return
            }
            #expect(copy.reason == .checkpoint)
            #expect(copy.refusal == .prefixView)
            _ = manager.clearRAMTier()
            let remaining = tier.getOrCreateTree(for: key).allSnapshotNodes()
                .compactMap(\.state.body)
                .filter { !$0.isPrefixView }
            #expect(remaining.map(\.bodyID) == [selected.bodyID])
        }
        _ = manager.clearRAMTier()
        #expect(manager.totalSnapshotBytes == 0)
    }

    @Test func finalRequestSettlementRetiresViewsWhoseLeafWasNeverStored() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        var handOvers: [CacheClaim.HandOver] = []
        for _ in 0..<2 {
            let (_, handOver) = await manager.resolveHoldingClaim(
                tokens: Array(1...8), partitionKey: key, diagnostics: diagnostics)
            handOvers.append(handOver)
        }
        let recurrent = MambaCache()
        recurrent.state = [MLXArray([Float(42)])]
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: [recurrent], offset: 4, type: .branchPoint, prefixView: true))
        manager.restoreSnapshot(
            path: Array(1...4), snapshot: view, partitionKey: key, lastAccessTime: .now)
        await handOvers[0].withClaim { _ in }
        #expect(manager.totalSnapshotBytes == 4)
        // The remaining request skips/fails leaf storage. No lookup visits
        // the view again: completion must release its non-evictable state.
        await handOvers[1].withClaim { _ in }
        #expect(manager.totalSnapshotBytes == 0)
        let tree = store.getOrCreateTree(for: key)
        #expect(tree.snapshotCount == 0)
        #expect(tree.nodeCount == 1)
    }

    @Test func viewResolutionPinsBothNodesAndRestoresItsOwnWholeState() async throws {
        let manager = makeManager()
        func hybrid(offset: Int, value: Float, view: Bool = false) throws -> HybridCacheSnapshot {
            let attention = KVCacheSimple()
            attention.state = [
                MLXArray.ones([1, 1, offset, 64]), MLXArray.ones([1, 1, offset, 64]),
            ]
            let recurrent = MambaCache()
            recurrent.state = [MLXArray([value])]
            return try #require(
                HybridCacheSnapshot.capture(
                    cache: [attention, recurrent], offset: offset,
                    type: view ? .branchPoint : .leaf, prefixView: view))
        }
        let view = try hybrid(offset: 4, value: 42, view: true)
        let nearest = try hybrid(offset: 6, value: 99)
        manager.restoreSnapshot(
            path: [1, 2, 3, 4], snapshot: view, partitionKey: key, lastAccessTime: .now)
        manager.restoreSnapshot(
            path: [1, 2, 3, 4, 20, 21], snapshot: nearest,
            partitionKey: key, lastAccessTime: .now - .seconds(1))
        manager.restoreSnapshot(
            path: Array(1...8), snapshot: try hybrid(offset: 8, value: 100),
            partitionKey: key, lastAccessTime: .now)
        let fork = [1, 2, 3, 4, 90]
        let (resolved, handOver) = await manager.resolveHoldingClaim(
            tokens: fork, partitionKey: key, diagnostics: diagnostics)
        #expect(resolved.lookup.backingLeaf?.tokenOffset == 6)
        #expect(resolved.lookup.snapshot?.memoryBytes == 4)
        let telemetry = manager.makeTelemetrySnapshot()
        #expect(telemetry.viewOnlyBytes == 4)
        let checkpoint = try #require(
            telemetry.trees.flatMap(\.nodes).first { $0.tokenOffset == 4 })
        #expect(checkpoint.hasSnapshot)
        #expect(checkpoint.checkpointKind == "prefixView")
        let restored = try #require(resolved.lookup.restoreCache())
        #expect(restored[0].offset == 4)
        #expect(restored[1].state[0].item(Float.self) == 42)
        let ownedAddresses = Set(
            (view.layers + nearest.layers).flatMap(\.state).map(backingAddress))
        #expect(
            restored.flatMap(\.state).allSatisfy { !ownedAddresses.contains(backingAddress($0)) })
        _ = manager.clearRAMTier()
        #expect(manager.totalSnapshotBytes == 3080)
        #expect(manager.lookup(tokens: fork, partitionKey: key).snapshot?.tokenOffset == 4)
        await handOver.withClaim { _ in }
        _ = manager.clearRAMTier()
        #expect(manager.totalSnapshotBytes == 0)
    }

    private let key = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)
    private let fingerprint = "fp"

    private var diagnostics: PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
    }

    private func makeManager() -> PrefixCacheManager {
        PrefixCacheManager(memoryBudgetBytes: 100 * 1024 * 1024)
    }

    private func makeSnapshot(offset: Int, type: HybridCacheSnapshot.CheckpointType = .leaf)
        -> HybridCacheSnapshot
    {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    /// Plant a body-absent `ssdOnly` (state-5) node at `path` via the warm-start
    /// ref-attach seam — the only production path that lands a committed ref
    /// without an SSD tier (no temp directory). Returns the fabricated ref so a
    /// test can program the peer's outcome for its id.
    @discardableResult
    private func plantSSDOnly(
        _ manager: PrefixCacheManager, path: [Int], offset: Int
    ) -> SnapshotRef {
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: offset)
        manager.restoreSnapshotRef(
            path: path, snapshotRef: ref, partitionKey: key, lastAccessTime: .now
        )
        return ref
    }

    // MARK: - Without hydration (passthrough)

    @Test func resolveSurfacesAMissWithoutHydrationWhenNothingIsCached() async {
        let manager = makeManager()
        let resolved = await manager.resolve(
            tokens: [1, 2, 3, 4],
            promptTokenCount: 4,
            partitionKey: key,
            modelFingerprint: nil,
            diagnostics: diagnostics
        )
        #expect(resolved.hydratedFromSSD == false)
        #expect(resolved.lookup.snapshot == nil)
        if case .hit = resolved.lookup.reason { Issue.record("expected a miss, got a hit") }
    }

    @Test func resolveSurfacesARamHitWithoutHydration() async {
        let manager = makeManager()
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let snapshot = makeSnapshot(offset: tokens.count)
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: tokens, snapshot: snapshot, storage: .ramOnly, partitionKey: key
            )!)

        let resolved = await manager.resolve(
            tokens: tokens,
            promptTokenCount: tokens.count,
            partitionKey: key,
            modelFingerprint: nil,
            diagnostics: diagnostics
        )

        #expect(resolved.hydratedFromSSD == false)
        guard case .hit = resolved.lookup.reason else {
            Issue.record("expected a hit, got \(resolved.lookup.reason)")
            return
        }
        #expect(resolved.lookup.snapshot?.tokenOffset == tokens.count)
    }

    // MARK: - alignmentLookup (the SSD-timing rule)

    /// A `.hit` lookup for the alignment tests — `alignmentLookup` keys off
    /// `hydratedFromSSD`, not the lookup's contents, so any hit will do.
    private func ramHitLookup(offset: Int = 8) -> PrefixCacheManager.LookupResult {
        let snapshot = makeSnapshot(offset: offset)
        return PrefixCacheManager.LookupResult(
            snapshot: snapshot,
            partitionKey: key,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: snapshot.tokenOffset,
            reason: .hit(
                snapshotOffset: snapshot.tokenOffset, totalTokens: offset * 2,
                type: snapshot.checkpointType)
        )
    }

    @Test func alignmentLookupIsTheLookupWhenNotHydratedFromSSD() {
        let lookup = ramHitLookup()
        let resolved = PrefixCacheManager.Resolved(
            lookup: lookup, hydratedFromSSD: false, hydrationSeconds: 0
        )
        // A RAM hit aligns checkpoint planning against itself.
        #expect(resolved.alignmentLookup != nil)
        #expect(resolved.alignmentLookup?.snapshotTokenOffset == lookup.snapshotTokenOffset)
    }

    @Test func alignmentLookupIsNilForAnSSDHydratedHit() {
        let resolved = PrefixCacheManager.Resolved(
            lookup: ramHitLookup(), hydratedFromSSD: true, hydrationSeconds: 0
        )
        // An SSD-hydrated hit must align against nothing: it matches the pre-carve
        // ordering that planned against the unhydrated `.ssdHit`, which never merged
        // an alignment branch-point. Flipping this rule would silently resume it.
        #expect(resolved.alignmentLookup == nil)
    }

    // MARK: - SSD-hydration composition (through the InMemory peer)

    @Test func resolveHydratesAnSSDHitAndPromotesIt() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        let path = [1, 2, 3, 4, 5, 6, 7, 8]
        let ref = plantSSDOnly(manager, path: path, offset: 8)
        peer.programSuccess(id: ref.snapshotID, body: makeSnapshot(offset: 8))

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        #expect(resolved.hydratedFromSSD == true)
        guard case .hit = resolved.lookup.reason else {
            Issue.record("expected a hit, got \(resolved.lookup.reason)")
            return
        }
        #expect(resolved.lookup.snapshot?.tokenOffset == 8)
        // The load-bearing success ordering: recency bumped (recordHit) for the
        // hydrated id, exactly once.
        #expect(peer.recordHitCalls == [ref.snapshotID])
        // Promote landed: a follow-up lookup now hits the resident body, not an
        // `.ssdHit` — the hydration folded the body into the tree.
        let relookup = manager.lookup(tokens: path, partitionKey: key)
        guard case .hit = relookup.reason else {
            Issue.record("expected the promoted body to hit, got \(relookup.reason)")
            return
        }
        #expect(relookup.snapshot?.tokenOffset == 8)
    }

    @Test func resolveRetriesAfterAFailedHydrationUntilAShallowerNodeSucceeds() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        // A chain of state-5 nodes at increasing depth. Inserted shallow→deep
        // so each extends the path without disturbing the prior ref.
        let ref4 = plantSSDOnly(manager, path: [1, 2, 3, 4], offset: 4)
        let ref6 = plantSSDOnly(manager, path: [1, 2, 3, 4, 5, 6], offset: 6)
        let ref8 = plantSSDOnly(manager, path: [1, 2, 3, 4, 5, 6, 7, 8], offset: 8)
        // The two deepest fail (clearing each); the shallowest succeeds.
        peer.programFailure(id: ref8.snapshotID)
        peer.programFailure(id: ref6.snapshotID)
        peer.programSuccess(id: ref4.snapshotID, body: makeSnapshot(offset: 4))

        let resolved = await manager.resolve(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8], promptTokenCount: 8, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        // Two failures then a success still surfaces a hit — the retry cap was
        // not exhausted (attempt went 0→1→2, success on the third).
        #expect(resolved.hydratedFromSSD == true)
        guard case .hit = resolved.lookup.reason else {
            Issue.record("expected a hit after retries, got \(resolved.lookup.reason)")
            return
        }
        #expect(resolved.lookup.snapshot?.tokenOffset == 4)
        // Hydrations were attempted deep→shallow, one per cleared node.
        #expect(peer.loadSyncCalls == [ref8.snapshotID, ref6.snapshotID, ref4.snapshotID])
        // Only the success bumped recency.
        #expect(peer.recordHitCalls == [ref4.snapshotID])
    }

    @Test func resolveSurfacesAMissAfterThreeConsecutiveHydrationFailures() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        let ref4 = plantSSDOnly(manager, path: [1, 2, 3, 4], offset: 4)
        let ref6 = plantSSDOnly(manager, path: [1, 2, 3, 4, 5, 6], offset: 6)
        let ref8 = plantSSDOnly(manager, path: [1, 2, 3, 4, 5, 6, 7, 8], offset: 8)
        // Every node on the path fails — three strikes hits the cap.
        peer.programFailure(id: ref8.snapshotID)
        peer.programFailure(id: ref6.snapshotID)
        peer.programFailure(id: ref4.snapshotID)

        let resolved = await manager.resolve(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8], promptTokenCount: 8, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        // After three consecutive failures the cap surfaces a clean miss rather
        // than looping the clear storm.
        #expect(resolved.lookup.snapshot == nil)
        if case .hit = resolved.lookup.reason { Issue.record("expected a miss") }
        #expect(peer.loadSyncCalls.count == 3)
    }

    @Test func resolveStopsAClearStormAtTheRetryCap() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        // Five state-5 nodes on the path — more than the cap. All fail.
        let path: [Int] = Array(1...10)
        let ref2 = plantSSDOnly(manager, path: Array(path[0..<2]), offset: 2)
        let ref4 = plantSSDOnly(manager, path: Array(path[0..<4]), offset: 4)
        let ref6 = plantSSDOnly(manager, path: Array(path[0..<6]), offset: 6)
        let ref8 = plantSSDOnly(manager, path: Array(path[0..<8]), offset: 8)
        let ref10 = plantSSDOnly(manager, path: path, offset: 10)
        [ref2, ref4, ref6, ref8, ref10].forEach { peer.programFailure(id: $0.snapshotID) }

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        // The cap stops the loop after three hydrations, not five — the two
        // shallowest nodes are never attempted.
        #expect(resolved.lookup.snapshot == nil)
        #expect(peer.loadSyncCalls.count == 3)
    }

    @Test func resolveFallsBackToAShallowerResidentBodyAfterAFailedHydration() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        // A RAM-resident body at offset 4, then a deeper state-5 node at 8.
        let ramPath = [1, 2, 3, 4]
        manager.admit(
            SnapshotAdmission.leaf(
                storedTokens: ramPath, snapshot: makeSnapshot(offset: 4),
                storage: .ramOnly, partitionKey: key
            )!)
        let ref8 = plantSSDOnly(manager, path: [1, 2, 3, 4, 5, 6, 7, 8], offset: 8)
        peer.programFailure(id: ref8.snapshotID)

        let resolved = await manager.resolve(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8], promptTokenCount: 8, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        // The deep ssdHit failed and cleared; the re-lookup fell back to the
        // next-shallower resident body instead of degrading straight to a miss.
        #expect(resolved.hydratedFromSSD == false)
        guard case .hit = resolved.lookup.reason else {
            Issue.record("expected a fallback hit, got \(resolved.lookup.reason)")
            return
        }
        #expect(resolved.lookup.snapshot?.tokenOffset == 4)
        #expect(peer.loadSyncCalls == [ref8.snapshotID])
        // A failed hydration calls neither recordHit nor promote.
        #expect(peer.recordHitCalls == [])
    }

    @Test func aFailedHydrationCallsNeitherRecordHitNorPromote() async {
        let manager = makeManager()
        let peer = InMemorySnapshotHydrating()
        manager.setSnapshotHydratingForTesting(peer)
        let path = [1, 2, 3, 4]
        let ref = plantSSDOnly(manager, path: path, offset: 4)
        peer.programFailure(id: ref.snapshotID)

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        // A lone failed hydration surfaces as a clean miss after the clear +
        // re-lookup (the re-lookup finds nothing hittable on the path).
        #expect(resolved.lookup.snapshot == nil)
        if case .hit = resolved.lookup.reason { Issue.record("expected a miss") }
        // The ordering invariant on failure: recency is not bumped, the body is
        // not promoted — a broken file must never look hot.
        #expect(peer.recordHitCalls == [])
        // The node was forgivingly cleared: a re-lookup misses cleanly rather
        // than re-attempting hydration on the broken backing.
        let relookup = manager.lookup(tokens: path, partitionKey: key)
        #expect(relookup.snapshot == nil)
    }
}
