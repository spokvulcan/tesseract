import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The **Cache Claim** through its interface (#554, ADR-0069), on the real
/// manager and radix tree with tiny caches: what Snapshot Resolution adds
/// to a claim, how each owner scope concludes it, the copy-only claim a
/// Speculative Canonical Prefill pass holds, and the tripwire.
@MainActor
@Suite(.serialized)
struct CacheClaimTests {
    private let key = CachePartitionKey(modelID: "cache-claim", kvBits: nil, kvGroupSize: 64)

    private func context(modelID: String = "cache-claim") -> PrefixCacheDiagnostics.Context {
        .init(requestID: UUID(), modelID: modelID, kvBits: nil, kvGroupSize: 64)
    }

    private func admitLeaf(_ manager: PrefixCacheManager, tokens: [Int]) throws {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.ones([1, 1, tokens.count, 64]), MLXArray.ones([1, 1, tokens.count, 64]),
        ]
        let body = try #require(FinalGenerationCache([kv]).moveSnapshot(offset: tokens.count))
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
    }

    // MARK: - Resolution opens the claim

    @Test func aMissHoldsOnlyItsLane() async {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let (resolved, handOver) = await manager.resolveHoldingClaim(
            tokens: [1, 2, 3], partitionKey: key, diagnostics: context())
        #expect(resolved.lookup.snapshot == nil)
        #expect(manager.requestHoldings == .init(lanes: 1, pinnedRequests: 0, leases: 0))
        await handOver.withClaim { _ in }
        #expect(manager.requestHoldings == .none)
    }

    @Test func aHitHoldsItsLaneAndPinsItsRestorePath() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let (resolved, handOver) = await manager.resolveHoldingClaim(
            tokens: Array(1...9), partitionKey: key, diagnostics: context())
        #expect(resolved.lookup.snapshot?.tokenOffset == 8)
        #expect(manager.requestHoldings == .init(lanes: 1, pinnedRequests: 1, leases: 0))
        await handOver.withClaim { _ in }
        #expect(manager.requestHoldings == .none)
    }

    // MARK: - Owner scopes conclude

    @Test func aStartThatThrowsConcludesItsClaimInTheScope() async {
        struct StartFailed: Error {}
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let context = context()
        await #expect(throws: StartFailed.self) {
            _ = try await CacheClaim.withRequestClaim(
                context: context, prefixCache: manager, memory: nil
            ) { claim in
                _ = await manager.resolve(
                    tokens: [1, 2, 3], promptTokenCount: 3, partitionKey: key,
                    modelFingerprint: nil, diagnostics: context, for: claim)
                #expect(manager.requestHoldings.lanes == 1)
                throw StartFailed()
            }
        }
        #expect(manager.requestHoldings == .none)
    }

    /// The drive redeems the hand-over around its whole body; resolving
    /// there pins into the same claim, and the scope's exit lets go of it
    /// all — even when the drive task was cancelled first, which stream
    /// termination does on every normal finish.
    @Test func theDriveConcludesTheClaimItRedeemsEvenWhenCancelled() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let modelID = "cache-claim-drive-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let context = context(modelID: modelID)
        let memory = RequestMemoryTelemetry(context: context)
        let (_, handOver) = await CacheClaim.withRequestClaim(
            context: context, prefixCache: manager, memory: memory
        ) { claim in
            _ = await manager.resolve(
                tokens: [1, 2, 3], promptTokenCount: 3, partitionKey: key,
                modelFingerprint: nil, diagnostics: context, for: claim)
        }
        let started = AsyncStream<Void>.makeStream()
        let drive = Task {
            await handOver.withClaim { claim in
                _ = await manager.resolve(
                    tokens: Array(1...9), promptTokenCount: 9, partitionKey: key,
                    modelFingerprint: nil, diagnostics: context, for: claim)
                started.continuation.yield()
                while !Task.isCancelled { await Task.yield() }
            }
        }
        for await _ in started.stream { break }
        #expect(manager.requestHoldings == .init(lanes: 1, pinnedRequests: 1, leases: 0))
        drive.cancel()
        await drive.value
        #expect(manager.requestHoldings == .none)
        let releases = capture.drain().filter {
            $0.eventName == "requestMemory" && $0.field("phase") == "releasingRequest"
                && $0.field("sampleKind") == "phaseBegin"
        }
        #expect(releases.count == 1, "exactly one conclusion")
    }

    // MARK: - Check-out

    /// The toy Model Session the steps run in: their `session` argument is
    /// the witness that they run at a quiescent point inside it.
    private let sessions = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2]))

    private func checkOut(
        _ resolved: PrefixCacheManager.Resolved, tokens: [Int], maximumAdvance: Int = 10,
        identityKeySpace: Bool = true, manager: PrefixCacheManager,
        context: PrefixCacheDiagnostics.Context? = nil,
        tripwire: CacheClaim.Tripwire = .standard
    ) async -> (outcome: CacheClaim.RestoreOutcome, handOver: CacheClaim.HandOver) {
        await manager.checkOutHoldingClaim(
            resolved, tokens: tokens, maximumAdvance: maximumAdvance,
            identityKeySpace: identityKeySpace, diagnostics: context ?? self.context(),
            sessions: sessions, tripwire: tripwire)
    }

    @discardableResult
    private func rewindAndConclude(_ handOver: CacheClaim.HandOver) async -> Int? {
        await handOver.rewindAndConclude(sessions: sessions)
    }

    private func resolved(
        _ manager: PrefixCacheManager, _ tokens: [Int], key: CachePartitionKey? = nil
    ) -> PrefixCacheManager.Resolved {
        .init(
            lookup: manager.lookup(tokens: tokens, partitionKey: key ?? self.key),
            hydratedFromSSD: false, hydrationSeconds: 0)
    }

    @Test(arguments: [false, true], [nil, 8] as [Int?])
    func aPrefixViewCopiesAsACheckpointWhateverThePartition(
        identityKeySpace: Bool, kvBits: Int?
    ) async throws {
        let layer = KVCacheSimple()
        layer.state = [MLXArray.ones([1, 1, 4, 64]), MLXArray.ones([1, 1, 4, 64])]
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: [layer], offset: 4, type: .branchPoint, prefixView: true))
        let partition = CachePartitionKey(
            modelID: "view-copy-reason", kvBits: kvBits, kvGroupSize: 64)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let (outcome, handOver) = await checkOut(
            .init(
                lookup: .init(
                    snapshot: view, partitionKey: partition, snapshotTokenOffset: 4,
                    sharedPrefixLength: 4,
                    reason: .hit(snapshotOffset: 4, totalTokens: 5, type: .branchPoint)),
                hydratedFromSSD: false, hydrationSeconds: 0),
            tokens: Array(1...5), identityKeySpace: identityKeySpace, manager: manager)
        guard case .copy(let copy) = outcome else {
            Issue.record("a prefix view must restore by copy")
            return
        }
        #expect(copy.reason == .checkpoint)
        #expect(copy.refusal == .prefixView)
        #expect(await rewindAndConclude(handOver) == nil)
    }

    @Test func aPendingFullPayloadCopiesUntilItsArraysAreReleased() async throws {
        let gate = DrainGate()
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: "claim-checkout-full", ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: root) }
        let key = CachePartitionKey(
            modelID: "claim-checkout-full", kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64))
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let body = try #require(FinalGenerationCache([kv]).moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        let payload = ServerCompletion.deferredPayload(for: body, extending: nil).payload
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body,
                    storage: .ramAndSSD(payload), partitionKey: key)))

        let (pending, first) = await checkOut(
            resolved(manager, tokens + [9], key: key), tokens: tokens + [9], manager: manager)
        #expect(pending.copy?.reason == .pendingFullPayload)
        #expect(pending.copy?.refusal == .pendingFullPayload)
        #expect(!body.layers.isEmpty)
        await rewindAndConclude(first)

        await gate.open()
        await store.flush()
        let (afterWrite, second) = await checkOut(
            resolved(manager, tokens + [9], key: key), tokens: tokens + [9], manager: manager)
        let handoff = try #require(afterWrite.handoff)
        #expect(handoff.cache.cache[0] as AnyObject === kv)
        #expect(await rewindAndConclude(second) == 8)
    }

    /// A lease the tree refuses because the leaf is already leased reads
    /// `pendingFullPayload` as it always did, and now says why.
    @Test func anAlreadyLeasedLeafCopiesAndSaysSo() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        try admitLeaf(manager, tokens: Array(1...8))
        let tree = try #require(store.tree(for: key))
        let node = try #require(
            tree.findBestSnapshot(tokens: Array(1...8), updateAccess: false)?.node)
        let holder = try #require(tree.beginLeafLease(on: node, context: context()))

        let (outcome, handOver) = await checkOut(
            resolved(manager, Array(1...9)), tokens: Array(1...9), manager: manager)
        let copy = try #require(outcome.copy)
        #expect(copy.reason == .pendingFullPayload)
        #expect(copy.refusal == .alreadyLeased)
        #expect(await rewindAndConclude(handOver) == nil)
        let body = try #require(node.state.body)
        #expect(
            tree.endLeafLease(
                holder, on: node, returning: body, tokens: Array(1...8), reason: .rewind))
    }

    // MARK: - The bounded Pending-Payload Wait (#523)

    /// One conversation, one leaf, one full payload the SSD writer still
    /// owes: the shape every second turn hits. `payload` gates the
    /// writer's materialize step so the test owns when the body arrays
    /// are released.
    @MainActor
    private struct PendingLeafScene {
        let manager: PrefixCacheManager
        let store: TieredSnapshotStore
        let root: URL
        let key: CachePartitionKey
        let kv: KVCacheSimple
        let tokens: [Int]
        let materializing: BlockingMaterializer

        var resolved: PrefixCacheManager.Resolved {
            .init(
                lookup: manager.lookup(tokens: tokens, partitionKey: key),
                hydratedFromSSD: false, hydrationSeconds: 0)
        }

        func progress() -> PendingPayloadProgress {
            guard let snapshot = manager.lookup(tokens: tokens, partitionKey: key).snapshot
            else { return .absent }
            return manager.pendingFullPayloadProgress(
                snapshot: snapshot, tokens: tokens, partitionKey: key)
        }
    }

    private func makePendingLeafScene(
        label: String, wait: Duration,
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) throws -> PendingLeafScene {
        let (manager, store, root) = PrefixCacheTestFixtures.makeSSDBackedManager(
            label: label, ramBudgetBytes: 1_000_000,
            writerDrainPreludeForTesting: writerDrainPreludeForTesting,
            evictionConfig: EvictionConfiguration(pendingFullPayloadWait: wait))
        let key = CachePartitionKey(
            modelID: label, kvBits: nil, kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64))
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let body = try #require(FinalGenerationCache([kv]).moveSnapshot(offset: 8))
        let stored = Array(1...8)
        let materializing = BlockingMaterializer()
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: stored, snapshot: body,
                    storage: .ramAndSSD(
                        materializing.gating(
                            ServerCompletion.deferredPayload(for: body, extending: nil).payload)),
                    partitionKey: key)))
        return PendingLeafScene(
            manager: manager, store: store, root: root, key: key, kv: kv,
            tokens: stored + [9], materializing: materializing)
    }

    @Test func aPayloadThatMaterializesWithinTheBoundIsWaitedOutIntoAHandoff() async throws {
        let scene = try makePendingLeafScene(label: "claim-wait-hit", wait: .seconds(5))
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })

        let release = Task {
            try? await Task.sleep(for: .milliseconds(60))
            scene.materializing.release()
        }
        let (outcome, handOver) = await checkOut(
            scene.resolved, tokens: scene.tokens, manager: scene.manager)
        _ = await release.value

        let handoff = try #require(
            outcome.handoff, "the wait should have turned the copy into a move")
        #expect(handoff.waitSeconds > 0)
        // The move is the real thing: the request holds the leaf's own
        // cache objects, not a restored copy of them.
        #expect(handoff.cache.cache[0] as AnyObject === scene.kv)
        #expect(await rewindAndConclude(handOver) == 8)
        await scene.store.flush()
    }

    @Test func aPayloadStillPendingAfterTheBoundCopiesAndReportsTheWaitedTime() async throws {
        let bound = Duration.milliseconds(80)
        let scene = try makePendingLeafScene(label: "claim-wait-miss", wait: bound)
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })

        let started = ContinuousClock.now
        let (outcome, handOver) = await checkOut(
            scene.resolved, tokens: scene.tokens, manager: scene.manager)
        let elapsed = ContinuousClock.now - started

        let copy = try #require(outcome.copy)
        #expect(copy.reason == .pendingFullPayload)
        #expect(copy.refusal == .pendingFullPayload)
        #expect(copy.waitSeconds >= 0.08)
        // Bounded: the request gives up and copies rather than waiting
        // out a writer that is taking its time.
        #expect(elapsed < .seconds(3))
        #expect(scene.progress() == .inProgress)
        await rewindAndConclude(handOver)

        scene.materializing.release()
        await scene.store.flush()
    }

    @Test func aQueuedPayloadIsNotWaitedForAndCopiesAtOnce() async throws {
        let gate = DrainGate()
        let scene = try makePendingLeafScene(
            label: "claim-wait-queued", wait: .seconds(5),
            writerDrainPreludeForTesting: { await gate.wait() })
        defer { try? FileManager.default.removeItem(at: scene.root) }
        // The writer has not started on this payload, so it has no
        // bounded completion time — waiting for it would be a guess.
        #expect(scene.progress() == .queued)

        let started = ContinuousClock.now
        let (outcome, handOver) = await checkOut(
            scene.resolved, tokens: scene.tokens, manager: scene.manager)
        let elapsed = ContinuousClock.now - started

        let copy = try #require(outcome.copy)
        #expect(copy.reason == .pendingFullPayload)
        #expect(copy.waitSeconds == 0)
        #expect(elapsed < .seconds(1))
        await rewindAndConclude(handOver)

        await gate.open()
        scene.materializing.release()
        await scene.store.flush()
    }

    /// The waited time and the precise refusal ride beside the copy reason
    /// on the events a request's telemetry is read through — the reason
    /// keeps its name and its value.
    @Test func theRefusalAndTheWaitedTimeRideTheCopyReasonThroughTelemetry() async throws {
        let bound = Duration.milliseconds(80)
        let scene = try makePendingLeafScene(label: "claim-wait-telemetry", wait: bound)
        defer { try? FileManager.default.removeItem(at: scene.root) }
        #expect(await waitUntil { scene.progress() == .inProgress })
        let (outcome, handOver) = await checkOut(
            scene.resolved, tokens: scene.tokens, manager: scene.manager)
        let copy = try #require(outcome.copy)
        await rewindAndConclude(handOver)

        let lookup = PrefixCacheDiagnostics.LookupEvent(
            reason: .hit(snapshotOffset: 8, totalTokens: 9, type: .leaf),
            promptTokens: 9, sharedPrefixLength: 9, skippedPrefillTokens: 8,
            newTokensToPrefill: 1, lookupMs: 0, restoreMs: 0, plannedCheckpoints: [],
            restoreMode: "copy", copyReason: copy.reason, copyRefusal: copy.refusal,
            copyWaitSeconds: copy.waitSeconds)
        let lookupFields = Dictionary(uniqueKeysWithValues: lookup.fields)
        #expect(lookupFields["copyReason"] == "pendingFullPayload")
        #expect(lookupFields["copyRefusal"] == "pendingFullPayload")
        #expect((Double(lookupFields["copyWaitMs"] ?? "") ?? 0) >= 80)

        var report = LeafStorePhase.Report()
        report.restoreCopyReason = copy.reason
        report.restoreCopyRefusal = copy.refusal
        report.restoreCopyWaitSeconds = copy.waitSeconds
        let reportFields = Dictionary(uniqueKeysWithValues: report.fields)
        #expect(reportFields["copyReason"] == "pendingFullPayload")
        #expect(reportFields["copyRefusal"] == "pendingFullPayload")
        #expect((Double(reportFields["copyWaitMs"] ?? "") ?? 0) >= 80)

        scene.materializing.release()
        await scene.store.flush()
    }

    // MARK: - The refusal ladder

    @Test(arguments: [
        "system", "branch", "immutable", "rotating", "window", "untrimmable", "quantized", "image",
    ])
    func unsafeRestorePointsKeepTheCopyPath(reason: String) async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let key = CachePartitionKey(
            modelID: "fallback", kvBits: reason == "quantized" ? 8 : nil, kvGroupSize: 64)
        let layer: any KVCache
        switch reason {
        case "rotating": layer = RotatingKVCache(maxSize: 32)
        case "window": layer = AdvanceLimitedCache(limit: 5)
        case "untrimmable": layer = AdvanceLimitedCache(limit: -1)
        default: layer = KVCacheSimple()
        }
        _ = layer.update(keys: MLXArray.ones([1, 1, 8, 64]), values: MLXArray.ones([1, 1, 8, 64]))
        eval(layer)
        let previous = FinalGenerationCache([layer])
        let body = try #require(
            reason == "system" || reason == "immutable"
                ? HybridCacheSnapshot.capture(
                    cache: previous.cache, offset: 8, type: reason == "system" ? .system : .leaf)
                : previous.moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        let tree = store.getOrCreateTree(for: key)
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(body, on: node)
        if reason == "branch" { _ = tree.insertPath(tokens: tokens + [99]) }
        let resolved = resolved(manager, tokens + [9], key: key)
        let (outcome, handOver) = await checkOut(
            resolved, tokens: tokens + [9], identityKeySpace: reason != "image",
            manager: manager)
        let expected: (reason: String, refusal: String)
        switch reason {
        case "system": expected = ("checkpoint", "notLeafCheckpoint")
        case "branch": expected = ("checkpoint", "notResidentLeaf")
        case "immutable": expected = ("immutableBody", "immutableBody")
        case "window", "untrimmable": expected = ("untrimmable", "untrimmable")
        case "image": expected = ("imageKeySpace", "imageKeySpace")
        case "quantized": expected = ("quantized", "quantizedPartition")
        default: expected = ("rotating", "rotating")
        }
        let copy = try #require(outcome.copy)
        #expect(copy.reason.rawValue == expected.reason)
        #expect(copy.refusal.rawValue == expected.refusal)
        #expect(tree.leaseCount == 0)
        #expect(await rewindAndConclude(handOver) == nil)
        let restored = try #require(resolved.lookup.restoreCache())
        #expect(restored[0] as AnyObject !== layer as AnyObject)
        #expect(backingAddress(restored[0].state[0]) != backingAddress(body.layers[0].state[0]))
    }

    // MARK: - Rewind and check-in

    @Test func aRewindRestoresRecurrentMetadataAndAttentionAfterGrowth() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let kv = KVCacheSimple()
        kv.step = 4
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let recurrent = MambaCache(leftPadding: [2])
        recurrent[1] = MLXArray([Float(3), 4])
        recurrent.prepare(lengths: [7])
        recurrent.offset = 8
        let body = try #require(FinalGenerationCache([kv, recurrent]).moveSnapshot(offset: 8))
        let expected = body.layers.map {
            ($0.metaState, $0.state.map { $0.asData(access: .copy).data })
        }
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let (outcome, handOver) = await checkOut(
            resolved(manager, tokens + [9]), tokens: tokens + [9], maximumAdvance: 20,
            manager: manager)
        let handoff = try #require(outcome.handoff)
        #expect(handoff.rewindStateBytes == 8)
        _ = kv.update(
            keys: MLXArray.zeros([1, 1, 10, 64]), values: MLXArray.zeros([1, 1, 10, 64]))
        recurrent[1] = MLXArray([Float(99), 100])
        recurrent.advance(10)
        recurrent.offset = 18
        eval(handoff.cache.cache)

        #expect(await rewindAndConclude(handOver) == 8)
        #expect(handoff.cache.cache.isEmpty)
        let restored = try #require(
            manager.lookup(tokens: tokens + [9], partitionKey: key).snapshot)
        #expect(restored.tokenOffset == 8)
        for (layer, before) in zip(restored.layers, expected) {
            #expect(layer.offset == 8)
            #expect(layer.metaState == before.0)
            #expect(layer.state.map { $0.asData(access: .copy).data } == before.1)
        }
        #expect(manager.requestHoldings == .none)
    }

    @Test func aHandoffMovesTheResidentObjectsAndCheckInCommitsTheGrownLeaf() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let body = try #require(FinalGenerationCache([kv]).moveSnapshot(offset: 8))
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let (outcome, handOver) = await checkOut(
            resolved(manager, tokens + [9]), tokens: tokens + [9], manager: manager)
        let handoff = try #require(outcome.handoff)
        let tree = try #require(store.tree(for: key))
        #expect(handoff.cache.cache[0] as AnyObject === kv)
        #expect(body.layers.isEmpty, "retained lookup handles must release the frozen array views")
        #expect(tree.allSnapshotNodes().allSatisfy { $0.state.body == nil })
        #expect(manager.totalSnapshotBytes == 4_096)
        #expect(manager.budgetFloorBytes() == 4_096)

        _ = kv.update(keys: MLXArray.ones([1, 1, 1, 64]), values: MLXArray.ones([1, 1, 1, 64]))
        eval(kv)
        let extended = try #require(handoff.cache.moveSnapshot(offset: 9))
        let sessions = sessions
        let checkIn = await handOver.withClaim { claim in
            await sessions.withSession { session in
                await claim.checkIn(
                    extended, from: handoff.cache, tokens: tokens + [9], in: session)
            }
        }
        #expect(checkIn == .committed)
        #expect(handoff.cache.cache.isEmpty)
        #expect(tree.leaseCount == 0)
        #expect(manager.lookup(tokens: tokens + [9], partitionKey: key).snapshotTokenOffset == 9)
        #expect(manager.totalSnapshotBytes == 4_608)
    }

    /// A check-in the tree refuses (another body already sits where the
    /// leaf would go) takes the objects back and rewinds them in the same
    /// step: the original leaf is in the tree again, unchanged.
    @Test func aRefusedCheckInRewindsTheLeafInTheSameStep() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
        let body = try #require(FinalGenerationCache([kv]).moveSnapshot(offset: 8))
        let original = body.layers.flatMap(\.state).map { $0.asData(access: .copy).data }
        let tokens = Array(1...8)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key)))
        let (outcome, handOver) = await checkOut(
            resolved(manager, tokens + [9]), tokens: tokens + [9], manager: manager)
        let handoff = try #require(outcome.handoff)
        try admitLeaf(manager, tokens: tokens + [9])

        _ = kv.update(keys: MLXArray.zeros([1, 1, 1, 64]), values: MLXArray.zeros([1, 1, 1, 64]))
        eval(kv)
        let extended = try #require(handoff.cache.moveSnapshot(offset: 9))
        let sessions = sessions
        let checkIn = await handOver.withClaim { claim in
            await sessions.withSession { session in
                await claim.checkIn(
                    extended, from: handoff.cache, tokens: tokens + [9], in: session)
            }
        }
        #expect(checkIn == .rewound(.refused(.occupiedDestination)))
        #expect(handoff.cache.cache.isEmpty)
        let tree = try #require(store.tree(for: key))
        #expect(tree.leaseCount == 0)
        let back = try #require(tree.findBestSnapshot(tokens: tokens, updateAccess: false))
        #expect(back.node.tokenOffset == 8)
        #expect(
            back.node.state.body?.layers.flatMap(\.state).map { $0.asData(access: .copy).data }
                == original)
        #expect(manager.requestHoldings == .none)
    }

    /// A request cancelled before its check-in keeps the old leaf for the
    /// resend: the step rewinds instead of committing.
    @Test func aCancelledCheckInRewindsTheLeaf() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        try admitLeaf(manager, tokens: Array(1...8))
        let (outcome, handOver) = await checkOut(
            resolved(manager, Array(1...9)), tokens: Array(1...9), manager: manager)
        let handoff = try #require(outcome.handoff)
        let extended = try #require(handoff.cache.moveSnapshot(offset: 8))
        let sessions = sessions
        let drive = Task {
            await handOver.withClaim { claim in
                withUnsafeCurrentTask { $0?.cancel() }
                return await sessions.withSession { session in
                    await claim.checkIn(
                        extended, from: handoff.cache, tokens: Array(1...8), in: session)
                }
            }
        }
        #expect(await drive.value == .rewound(.cancelled))
        #expect(try #require(store.tree(for: key)).leaseCount == 0)
        #expect(manager.lookup(tokens: Array(1...9), partitionKey: key).snapshotTokenOffset == 8)
    }

    // MARK: - The copy-only claim

    /// A Speculative Canonical Prefill pass pins what it restores from and
    /// holds a lane while its scope runs. Its claim type has no check-out
    /// step, so it can never take the leaf the next request needs.
    @Test func aCopyOnlyClaimPinsAndHoldsALaneUntilItsScopeEnds() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let context = context()
        await CacheClaim.withCopyOnlyClaim(context: context, prefixCache: manager) { claim in
            let resolved = await manager.resolve(
                tokens: Array(1...9), promptTokenCount: 9, partitionKey: key,
                modelFingerprint: nil, diagnostics: context, for: claim)
            #expect(resolved.lookup.snapshot?.tokenOffset == 8)
            #expect(manager.requestHoldings == .init(lanes: 1, pinnedRequests: 1, leases: 0))
        }
        #expect(manager.requestHoldings == .none)
    }

    // MARK: - The tripwire

    /// Release-build behaviour, in every build: a hand-over the drive never
    /// redeems is a leak. The tripwire reports it and lets go of the pins
    /// and the lane, instead of the reserve pricing that lane until restart.
    @Test func aHandOverNeverRedeemedTripsAndReleasesThePinsAndTheLane() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let modelID = "cache-claim-dropped-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        _ = await manager.resolveHoldingClaim(
            tokens: Array(1...9), partitionKey: key, diagnostics: context(modelID: modelID),
            tripwire: .reporting)

        #expect(await waitUntil { manager.requestHoldings == .none })
        let event = try #require(
            await Self.tripwireEvent(capture), "the tripwire names the leaked claim")
        #expect(event.field("violation") == "droppedUnconcluded")
        #expect(event.field("owner") == "inTransit")
        #expect(event.field("lane") == "true")
        #expect(event.field("pins") == "1")
    }

    @Test func aHandOverRedeemedTwiceTripsAndConcludesOnce() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        let modelID = "cache-claim-twice-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let context = context(modelID: modelID)
        let memory = RequestMemoryTelemetry(context: context)
        let (_, handOver) = await CacheClaim.withRequestClaim(
            context: context, prefixCache: manager, memory: memory, tripwire: .reporting
        ) { claim in
            _ = await manager.resolve(
                tokens: [1, 2, 3], promptTokenCount: 3, partitionKey: key,
                modelFingerprint: nil, diagnostics: context, for: claim)
        }
        await handOver.withClaim { _ in }
        await handOver.withClaim { _ in }

        let events = await Self.eventsThroughTripwire(capture)
        let event = try #require(events.last)
        #expect(event.field("violation") == "handOverRedeemedTwice")
        #expect(manager.requestHoldings == .none)
        let releases = events.filter {
            $0.eventName == "requestMemory" && $0.field("phase") == "releasingRequest"
                && $0.field("sampleKind") == "phaseBegin"
        }
        #expect(releases.count == 1)
    }

    /// A step on a claim that already concluded adds nothing: resolving for
    /// it pins nothing and holds no lane, and the tripwire says so.
    @Test func aStepAfterTheConclusionTripsAndHoldsNothing() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let modelID = "cache-claim-late-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let context = context(modelID: modelID)
        let (_, handOver) = await CacheClaim.withRequestClaim(
            context: context, prefixCache: manager, memory: nil, tripwire: .reporting
        ) { _ in }
        let escaped = await handOver.withClaim { claim in claim }

        let resolved = await manager.resolve(
            tokens: Array(1...9), promptTokenCount: 9, partitionKey: key,
            modelFingerprint: nil, diagnostics: context, for: escaped)
        #expect(resolved.lookup.snapshot?.tokenOffset == 8, "resolution itself still answers")
        let event = try #require(await Self.tripwireEvent(capture))
        #expect(event.field("violation") == "stepAfterConclusion")
        #expect(await waitUntil { manager.requestHoldings == .none })
    }

    /// A hand-over dropped while the claim holds a lease: the tripwire lets
    /// go of the pins and the lane, cannot return the lease without the
    /// session, and names it.
    @Test func theTripwireNamesALeaseItCannotReturn() async throws {
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        try admitLeaf(manager, tokens: Array(1...8))
        let modelID = "cache-claim-leased-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let resolved = resolved(manager, Array(1...9))
        func checkOutAndDropTheHandOver() async -> UUID? {
            let (outcome, _) = await checkOut(
                resolved, tokens: Array(1...9), manager: manager,
                context: context(modelID: modelID), tripwire: .reporting)
            return outcome.handoff?.leaseID
        }
        let leaseID = await checkOutAndDropTheHandOver()
        let event = try #require(await Self.tripwireEvent(capture))
        #expect(event.field("violation") == "droppedUnconcluded")
        #expect(event.field("heldLeaseID") == leaseID?.uuidString)
        #expect(event.field("heldLeaseOffset") == "8")
        #expect(await waitUntil { manager.requestHoldings.lanes == 0 })
        #expect(manager.requestHoldings.leases == 1, "no exact return is possible here")
    }

    @Test func aSecondCheckOutTripsAndCopiesNothing() async throws {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000)
        try admitLeaf(manager, tokens: Array(1...8))
        let modelID = "cache-claim-second-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let resolved = resolved(manager, Array(1...9))
        let sessions = sessions
        let (outcomes, handOver) = await CacheClaim.withRequestClaim(
            context: context(modelID: modelID), prefixCache: manager, memory: nil,
            tripwire: .reporting
        ) { claim in
            await sessions.withSession { session in
                let first = await claim.checkOut(
                    resolved, tokens: Array(1...9), maximumAdvance: 10, identityKeySpace: true,
                    in: session)
                let second = await claim.checkOut(
                    resolved, tokens: Array(1...9), maximumAdvance: 10, identityKeySpace: true,
                    in: session)
                return (first.handoff != nil, second.handoff == nil && second.copy == nil)
            }
        }
        #expect(outcomes == (true, true))
        let event = try #require(await Self.tripwireEvent(capture))
        #expect(event.field("violation") == "secondCheckOut")
        #expect(await rewindAndConclude(handOver) == 8)
    }

    /// The debug guard's input: both session providers mark the task as
    /// inside the Model Session, so an owner scope opened there traps
    /// instead of deadlocking on the session's lock at its conclusion.
    @Test func bothSessionProvidersMarkTheModelSession() async throws {
        #expect(!ModelSessionScope.isInside)
        let provider = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2]))
        #expect(await provider.withSession { _ in ModelSessionScope.isInside })
        let container = ContainerModelSessionProvider(container: provider.container)
        #expect(await container.withSession { _ in ModelSessionScope.isInside })
        #expect(!ModelSessionScope.isInside)
    }

    // MARK: - Helpers

    private static func tripwireEvent(_ capture: TelemetryCapture) async
        -> PromptCacheTelemetryEvent?
    {
        await eventsThroughTripwire(capture).last { $0.eventName == "cacheClaimTripwire" }
    }

    /// Every captured event up to and including the first tripwire event,
    /// which the tripwire logs from its MainActor release.
    private static func eventsThroughTripwire(
        _ capture: TelemetryCapture
    ) async -> [PromptCacheTelemetryEvent] {
        let deadline = ContinuousClock.now + .seconds(5)
        var seen: [PromptCacheTelemetryEvent] = []
        while ContinuousClock.now < deadline {
            seen += capture.drain()
            if let index = seen.firstIndex(where: { $0.eventName == "cacheClaimTripwire" }) {
                return Array(seen[...index])
            }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return seen
    }
}

extension CacheClaim.RestoreOutcome {
    var copy: CacheClaim.Copy? {
        if case .copy(let copy) = self { return copy }
        return nil
    }

    var handoff: CacheClaim.Handoff? {
        if case .handoff(let handoff) = self { return handoff }
        return nil
    }
}

nonisolated private final class AdvanceLimitedCache: KVCacheSimple {
    let limit: Int
    init(limit: Int) { self.limit = limit; super.init() }
    override func isTrimmable(after positions: Int) -> Bool { positions <= limit }
}
