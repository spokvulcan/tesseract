import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Every way a keyed request can end, and what each must give back (#554).
/// A request holds a lane in the **Active-Inference Reserve**, its **Restore
/// Pins** and, after a **Leaf Handoff**, a **Leaf Lease**. Whatever the
/// exit — a completed handoff, copy or cold turn, a startup cancel by the
/// caller or by the unload drain, a decode cancel, a prefill that fails
/// after the leaf was taken, a refused check-in, a think-stripping turn or
/// the direct-turn guard — the drive's end must leave:
///
/// 1. the leaf back: no lease, and the next request hits it at its offset;
/// 2. no pins and no lane for the request;
/// 3. exactly one release of the request.
///
/// Each case runs the real **Server Completion** module on the toy **Model
/// Session** and waits for the drive to finish before asserting anything.
@MainActor
struct ServerCompletionExitMatrixTests {
    typealias Replay = EmittedPathSynthesizedReplayTests

    private static func parameters(maxTokens: Int? = nil) -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        if let maxTokens { parameters.maxTokens = maxTokens }
        return parameters
    }

    private static let firstRequest = Replay.conversation([Replay.user("hi")])
    private static let warmRequest = Replay.conversation([
        Replay.user("hi"), Replay.assistant("hello world"), Replay.user("more"),
    ])

    // MARK: - Completed exits

    @Test func aCompletedHandoffChecksItsLeafIn() async throws {
        let session = Replay.Session()
        _ = try await session.turn(Self.firstRequest)
        let turn = try await session.turn(Self.warmRequest, text: "again")
        #expect(turn.leafStore["restoreMode"] == "handoff", turn.account)
        #expect(turn.leafStore["copyRefusal"] == nil, turn.account)
        #expect(turn.event("lookup")?.field("copyRefusal") == nil)
        await Self.expectReleased(session.fixture, turn.events)

        let next = try await session.turn(
            Replay.conversation(
                Self.warmRequest.messages + [Replay.assistant("again"), Replay.user("next")]))
        #expect(next.cached == Self.leafOffset(turn), next.account)
    }

    @Test func aCompletedCopyStoresItsOwnLeaf() async throws {
        let session = Replay.Session()
        _ = try await session.turn(Self.firstRequest)
        session.fixture.cacheAdmin.setLeafCheckoutDisabled(true)
        let turn = try await session.turn(Self.warmRequest, text: "again")
        #expect(turn.leafStore["restoreMode"] == "copy", turn.account)
        #expect(turn.leafStore["copyReason"] == "checkoutDisabled", turn.account)
        // The precise refusal rides beside the unchanged copy reason.
        #expect(turn.leafStore["copyRefusal"] == "checkoutDisabled", turn.account)
        #expect(turn.event("lookup")?.field("copyRefusal") == "checkoutDisabled")
        #expect(
            turn.events.first {
                $0.eventName == "requestMemory" && $0.field("phase") == "restored"
            }?.field("restoreCopyRefusal") == "checkoutDisabled")
        await Self.expectReleased(session.fixture, turn.events)

        session.fixture.cacheAdmin.setLeafCheckoutDisabled(false)
        let next = try await session.turn(
            Replay.conversation(
                Self.warmRequest.messages + [Replay.assistant("again"), Replay.user("next")]))
        #expect(next.cached == Self.leafOffset(turn), next.account)
    }

    @Test func aCompletedColdTurnStoresItsLeaf() async throws {
        let session = Replay.Session()
        let turn = try await session.turn(Self.firstRequest)
        #expect(turn.leafStore["restoreMode"] == "cold", turn.account)
        await Self.expectReleased(session.fixture, turn.events)

        let next = try await session.turn(Self.warmRequest, text: "again")
        #expect(next.cached == Self.leafOffset(turn), next.account)
    }

    // MARK: - Rewound exits

    /// A cancel that lands in the warm prefill, by the caller or by the
    /// unload drain: the start throws, and the leaf it took goes back.
    @Test(arguments: [false, true])
    func aStartupCancelRewindsTheLeaf(drain: Bool) async throws {
        let gate = ForwardGate(threshold: 0, armed: false)
        let session = Replay.Session(onForward: gate.onForward)
        let first = try await session.turn(Self.firstRequest)
        let request = Replay.conversation([
            Replay.user("hi"), Replay.assistant("hello world"),
            Replay.user(String(repeating: "more ", count: 400)),
        ])
        _ = session.capture.drain()
        gate.arm()
        let starting = Task {
            try await session.fixture.start(
                conversation: request, parameters: Self.parameters(),
                renderContext: Replay.preserving)
        }
        await gate.reached()
        let draining: Task<Void, Never>?
        if drain {
            draining = await Self.beginDrain(session.fixture, on: session.fixture.actor)
        } else {
            draining = nil
            starting.cancel()
        }
        gate.open()
        do {
            let handle = try await starting.value
            handle.cancel()
            await handle.waitForCompletion()
            Issue.record("a cancelled prefill must throw")
        } catch is CancellationError {}
        await draining?.value
        let events = session.capture.drain()
        #expect(Self.terminal(events)?.field("outcome") == "cancelledDuringStart")
        #expect(Self.leafRewinds(events) == 1)
        await Self.expectReleased(session.fixture, events)

        let resend = try await session.turn(request, text: "again")
        #expect(resend.cached == Self.leafOffset(first), resend.account)
        #expect(resend.leafStore["restoreMode"] == "handoff", resend.account)
    }

    @Test func aDecodeCancelRewindsTheLeaf() async throws {
        let prompt = try EmittedPathToyTokenizer().applyChatTemplate(
            messages: Self.warmRequest.promptMessages, tools: nil,
            additionalContext: Replay.preserving.additionalContext())
        let gate = ForwardGate(threshold: prompt.count + 8, armed: false)
        let session = Replay.Session(onForward: gate.onForward)
        let first = try await session.turn(Self.firstRequest)
        _ = session.capture.drain()
        gate.arm()
        session.queue.enqueue(
            session.completion(thinking: "plan", text: String(repeating: "x", count: 128)))
        let handle = try await session.fixture.start(
            conversation: Self.warmRequest, parameters: Self.parameters(),
            renderContext: Replay.preserving)
        await gate.reached()
        handle.cancel()
        gate.open()
        for try await _ in handle.stream {}
        await handle.waitForCompletion()
        let events = session.capture.drain()
        #expect(Self.terminal(events)?.field("outcome") == "cancelled")
        #expect(Self.leafRewinds(events) == 1)
        // The claim's conclusion rewinds after the stream has finished, still
        // inside the drive: finishingStream, then rewindingLeaf, then
        // releasingRequest.
        let order = ["finishingStream", "rewindingLeaf", "releasingRequest"].map { phase in
            events.firstIndex { Self.isPhaseBegin($0, phase) }
        }
        #expect(order.allSatisfy { $0 != nil })
        #expect(order.compactMap { $0 } == order.compactMap { $0 }.sorted())
        await Self.expectReleased(session.fixture, events)

        let resend = try await session.turn(Self.warmRequest, text: "again")
        #expect(resend.cached == Self.leafOffset(first), resend.account)
        #expect(resend.leafStore["restoreMode"] == "handoff", resend.account)
    }

    /// The failure a request can actually meet after it took the leaf: its
    /// suffix prefill throws (a checked MLX error). Decode itself has no
    /// failure exit — the stream loop never throws, and the token loop
    /// always ends with its completion info.
    @Test func aPrefillFailureAfterTheHandoffRewindsTheLeaf() async throws {
        let fault = ToyPrefillFault()
        let session = Replay.Session(prefillFault: fault)
        let first = try await session.turn(Self.firstRequest)
        _ = session.capture.drain()
        fault.arm()
        await #expect(throws: ToyPrefillFault.Injected.self) {
            _ = try await session.fixture.start(
                conversation: Self.warmRequest, parameters: Self.parameters(),
                renderContext: Replay.preserving)
        }
        let events = session.capture.drain()
        #expect(
            events.first { $0.eventName == "lookup" }?.field("restoreMode") == "handoff")
        #expect(Self.terminal(events)?.field("outcome") == "startFailed")
        #expect(Self.leafRewinds(events) == 1)
        await Self.expectReleased(session.fixture, events)

        let resend = try await session.turn(Self.warmRequest, text: "again")
        #expect(resend.cached == Self.leafOffset(first), resend.account)
        #expect(resend.leafStore["restoreMode"] == "handoff", resend.account)
    }

    /// The tree refuses the check-in when another body already sits where
    /// the turn's leaf would go. The request keeps no leaf of its own and
    /// gives the one it took back.
    @Test func aRefusedCheckInRewindsTheLeaf() async throws {
        let prompt = try EmittedPathToyTokenizer().applyChatTemplate(
            messages: Self.warmRequest.promptMessages, tools: nil,
            additionalContext: Replay.preserving.additionalContext())
        let gate = ForwardGate(threshold: prompt.count + 8, armed: false)
        let session = Replay.Session(onForward: gate.onForward)
        let first = try await session.turn(Self.firstRequest)
        _ = session.capture.drain()
        let generated = session.completion(
            thinking: "plan", text: String(repeating: "x", count: 32))
        session.queue.enqueue(generated)
        gate.arm()
        let handle = try await session.fixture.start(
            conversation: Self.warmRequest, parameters: Self.parameters(),
            renderContext: Replay.preserving)
        await gate.reached()
        let destination = prompt + generated + [session.queue.eosTokenId]
        let manager = try #require(session.fixture.cacheAdmin.current)
        let key = CachePartitionKey(
            modelID: session.modelID, kvBits: nil, kvGroupSize: 64,
            modelFingerprint: session.fingerprint,
            templateContextDigest: Self.warmRequest.templateContextDigest)
        manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: destination, snapshot: try Self.body(offset: destination.count),
                    storage: .ramOnly, partitionKey: key)))
        gate.open()
        for try await _ in handle.stream {}
        await handle.waitForCompletion()
        let events = session.capture.drain()
        #expect(
            events.contains {
                $0.eventName == "leafLeaseRefused" && $0.field("reason") == "occupiedDestination"
            })
        #expect(
            events.contains {
                $0.eventName == "leafStore" && $0.field("skip") == "lease-return-refused"
            })
        // The check-in comes before the payload is extracted, so a refused
        // one never reaches the admission phase.
        #expect(!events.contains { Self.isPhaseBegin($0, "admittingLeaf") })
        #expect(Self.leafRewinds(events) == 1)
        await Self.expectReleased(session.fixture, events)

        let resend = try await session.turn(Self.warmRequest, text: "again")
        #expect(resend.cached == Self.leafOffset(first), resend.account)
    }

    /// A think-stripping template at a user boundary: the boundary route
    /// checks the leaf in as its views' Backing Leaf, stores the canonical
    /// leaf from the boundary, then lets the checked-in leaf go (ADR-0068).
    @Test func aThinkStrippingTurnChecksItsLeafInForTheCanonicalLeaf() async throws {
        let session = Replay.Session()
        _ = try await session.turn(
            Replay.conversation([Replay.user("hi")], context: .canonical), context: .canonical)
        let request = Replay.conversation(
            [Replay.user("hi"), Replay.assistant("hello world"), Replay.user("more")],
            context: .canonical)
        let turn = try await session.turn(request, text: "again", context: .canonical)
        #expect(turn.leafStore["restoreMode"] == "handoff", turn.account)
        #expect(turn.leafStore["path"] == "boundary", turn.account)
        #expect(
            turn.events.contains {
                $0.eventName == "leafLeaseEnd" && $0.field("reason") == "checkIn"
            }, turn.account)
        #expect(Self.leafRewinds(turn.events) == 0)
        await Self.expectReleased(session.fixture, turn.events)

        let next = try await session.turn(
            Replay.conversation(
                request.messages + [Replay.assistant("again"), Replay.user("next")],
                context: .canonical),
            context: .canonical)
        #expect(next.cached == Self.leafOffset(turn), next.account)
    }

    /// A non-thinking direct turn that a structural guard keeps off the live
    /// path: it returns the leaf it took and stores nothing else.
    @Test func theDirectTurnGuardReturnsTheLeaf() async throws {
        let tokenizer = ToySequencingTokenizer()
        let first = HTTPPrefixCacheConversation(
            systemPrompt: nil, messages: [.init(role: .user, content: "Hi")])
        let next = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Hi"), .assistant(content: "Hello!"),
                .init(role: .user, content: "More?"),
            ])
        let render = try tokenizer.applyChatTemplate(
            messages: next.promptMessages, tools: nil, additionalContext: nil)
        let modelID = "exit-direct-guard-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: render + Array("Sure.".utf8).map(Int.init)),
                tokenizer: tokenizer),
            modelID: modelID)
        let firstHandle = try await fixture.start(
            conversation: first, parameters: Self.parameters())
        #expect(try await collectServerText(firstHandle).text == "Hello!")
        _ = capture.drain()

        let handle = try await fixture.start(
            conversation: next, parameters: Self.parameters(maxTokens: 0))
        #expect(handle.cachedTokenCount > 0)
        #expect(try await collectServerText(handle).text.isEmpty)
        let events = capture.drain()
        #expect(events.last { $0.eventName == "leafStore" }?.field("source") == "rewind")
        #expect(Self.leafRewinds(events) == 1)
        await Self.expectReleased(fixture, events)

        let resend = try await fixture.start(conversation: next, parameters: Self.parameters())
        #expect(resend.cachedTokenCount == handle.cachedTokenCount)
        #expect(try await collectServerText(resend).text == "Sure.")
    }

    // MARK: - Helpers

    /// Facts 2 and 3 of every exit, read once the drive has finished: no
    /// lane, no pins and no lease for the request, and one conclusion of
    /// its Cache Claim — and the terminal sample agrees about the lease.
    private static func expectReleased(
        _ fixture: ServerCompletionFixture, _ events: [PromptCacheTelemetryEvent],
        sourceLocation: SourceLocation = #_sourceLocation
    ) async {
        // A finished turn may have scheduled a Speculative Canonical
        // Prefill, which holds a claim of its own while it runs.
        await fixture.module.preemptSpeculativePrefill(on: fixture.actor)
        #expect(
            fixture.cacheAdmin.requestHoldings == PrefixCacheManager.RequestHoldings.none,
            sourceLocation: sourceLocation)
        #expect(terminal(events)?.field("treeLeaseCount") == "0", sourceLocation: sourceLocation)
        // The request's Cache Claim concluded once: one release.
        let releases = events.filter { isPhaseBegin($0, "releasingRequest") }
        #expect(releases.count == 1, "exactly one conclusion", sourceLocation: sourceLocation)
    }

    private static func terminal(
        _ events: [PromptCacheTelemetryEvent]
    ) -> PromptCacheTelemetryEvent? {
        events.last { $0.eventName == "requestMemory" && $0.field("sampleKind") == "terminal" }
    }

    private static func isPhaseBegin(_ event: PromptCacheTelemetryEvent, _ phase: String) -> Bool {
        event.eventName == "requestMemory" && event.field("phase") == phase
            && event.field("sampleKind") == "phaseBegin"
    }

    private static func leafRewinds(_ events: [PromptCacheTelemetryEvent]) -> Int {
        events.count { $0.eventName == "leafLeaseEnd" && $0.field("reason") == "rewind" }
    }

    /// The offset of the leaf a turn stored.
    private static func leafOffset(_ turn: Replay.Turn) -> Int? {
        turn.leafStore["leafOffset"].flatMap(Int.init)
    }

    /// A stand-in body at `offset`, for staging a competing admission.
    private static func body(offset: Int) throws -> HybridCacheSnapshot {
        let layer = KVCacheSimple()
        layer.state = [MLXArray.zeros([1, 1, offset, 4]), MLXArray.zeros([1, 1, offset, 4])]
        return try #require(
            HybridCacheSnapshot.capture(cache: [layer], offset: offset, type: .leaf))
    }

    private static func beginDrain(
        _ fixture: ServerCompletionFixture, on actor: isolated LLMActor
    ) -> Task<Void, Never> {
        // Runs on the actor through its first suspension, so the drain has
        // bumped its generation before the test opens the gate.
        Task.immediate { await fixture.module.drainActiveCompletion(on: actor) }
    }
}
