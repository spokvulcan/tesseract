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
