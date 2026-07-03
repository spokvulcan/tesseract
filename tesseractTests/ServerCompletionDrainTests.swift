import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The unload drain contract of the **Server Completion** module —
/// ADR-0015's in-actor backstop, pinned at the module seam with genuine
/// occupants (PRD #137, PR C): a live toy-backed completion parked at a
/// `ForwardGate`, and a genuinely scheduled **Speculative Canonical
/// Prefill** over the toy container. `drainActiveCompletion` — the exact
/// call `LLMActor.unloadModel` makes — must cancel-and-await the live
/// work, concurrent drains must all wait (not skip past a slot another
/// drain is already awaiting), and stale natural-finish clears must never
/// drop a newer occupant.
@MainActor
struct ServerCompletionDrainTests {

    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        return parameters
    }

    /// A fixture whose toy model is parked mid-decode at the gate: the
    /// completion is genuinely live — registered by the module's own
    /// `start`, its model thread blocked — when the test drains.
    private static func gatedLiveCompletion() async throws -> (
        fixture: ServerCompletionFixture, gate: ForwardGate,
        handle: HTTPServerGenerationStart, scriptedText: String
    ) {
        let tokenizer = ToySequencingTokenizer()
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Hi")]
        )
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let scriptedText = String(repeating: "x", count: 64)
        let gate = ForwardGate(threshold: render.count + 8)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: render + Array(scriptedText.utf8).map(Int.init),
                    onForward: gate.onForward
                ),
                tokenizer: tokenizer
            )
        )
        let handle = try await fixture.start(
            conversation: conversation, parameters: parameters()
        )
        await gate.reached()
        return (fixture, gate, handle, scriptedText)
    }

    @Test
    func drainCancelsAndAwaitsTheActiveCompletion() async throws {
        let (fixture, gate, handle, scriptedText) = try await Self.gatedLiveCompletion()

        let drainFinished = AsyncFlag()
        let drainTask = Task {
            await fixture.drain()
            await drainFinished.set()
        }

        // The drain must not return while the completion is still running.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await drainFinished.get() == false)

        gate.open()
        await drainTask.value
        #expect(await drainFinished.get())

        // The drain cancelled the live drive: the stream ended early, well
        // short of the scripted completion.
        var text = ""
        for try await event in handle.stream {
            if case .text(let chunk) = event { text += chunk }
        }
        #expect(text.utf8.count < scriptedText.utf8.count)
    }

    @Test
    func concurrentDrainsBothAwaitTheActiveCompletion() async throws {
        let (fixture, gate, _, _) = try await Self.gatedLiveCompletion()

        let firstFinished = AsyncFlag()
        let secondFinished = AsyncFlag()
        let firstDrain = Task {
            await fixture.drain()
            await firstFinished.set()
        }
        let secondDrain = Task {
            await fixture.drain()
            await secondFinished.set()
        }

        // Neither drain may return while the completion is still running —
        // the second must not slip past a slot the first is already awaiting.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await firstFinished.get() == false)
        #expect(await secondFinished.get() == false)

        gate.open()
        await firstDrain.value
        await secondDrain.value
        #expect(await firstFinished.get())
        #expect(await secondFinished.get())
    }

    @Test
    func naturalFinishLeavesTheDrainNothingToAwait() async throws {
        let tokenizer = ToySequencingTokenizer()
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Hi")]
        )
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: render + Array("Done".utf8).map(Int.init)),
                tokenizer: tokenizer
            )
        )
        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        var text = ""
        for try await event in handle.stream {
            if case .text(let chunk) = event { text += chunk }
        }
        await handle.waitForCompletion()

        // The completion finished naturally (full scripted text, no cancel),
        // so the drain returns instead of parking.
        #expect(text == "Done")
        await fixture.drain()
    }

    @Test
    func staleFinishDoesNotDropANewerSlot() async throws {
        let (fixture, gate, handle, scriptedText) = try await Self.gatedLiveCompletion()

        // A finisher for some other (older) request must not clear the slot:
        // the drain must still find, cancel, and await the live completion.
        await fixture.module.clearFinishedCompletion(UUID(), on: fixture.actor)

        let drainFinished = AsyncFlag()
        let drainTask = Task {
            await fixture.drain()
            await drainFinished.set()
        }
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await drainFinished.get() == false)

        gate.open()
        await drainTask.value

        var text = ""
        for try await event in handle.stream {
            if case .text(let chunk) = event { text += chunk }
        }
        #expect(text.utf8.count < scriptedText.utf8.count)
    }

    // MARK: - Speculative prefill drain contract (ADR-0009)

    /// A fixture with a genuinely scheduled **Speculative Canonical
    /// Prefill** parked mid-span at the gate. One completed turn first,
    /// over a conversation with a system prompt: the pass only *extends* a
    /// resolved boundary strictly inside its admit path, a turn's own
    /// canonical leaf always ends past the future-shared prefix, and the
    /// planner's `.system` checkpoint at the stable-prefix boundary is the
    /// mid-path snapshot the pass restores from. The drain after the turn
    /// clears the slot and leaves the drain generation deterministically
    /// at 1, so the schedule's quiescence guard passes; the gate arms only
    /// then, so it catches the pass's own extension forwards, not the
    /// turn's.
    private static func gatedLiveSpeculativePass() async throws -> (
        fixture: ServerCompletionFixture, gate: ForwardGate
    ) {
        let tokenizer = ToySequencingTokenizer()
        let opener = HTTPPrefixCacheMessage(
            role: .user, content: String(repeating: "a", count: 4000))
        // Long enough that the stable-prefix detector's ≥⅓-of-prompt ratio
        // guard accepts the system boundary, so the planner captures the
        // `.system` checkpoint the pass later restores from.
        let turn1 = HTTPPrefixCacheConversation(
            systemPrompt: String(repeating: "s", count: 3000), messages: [opener])
        let render1 = try tokenizer.applyChatTemplate(
            messages: turn1.promptMessages, tools: nil, additionalContext: nil
        )
        let gate = ForwardGate(threshold: 1024, armed: false)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: render1 + Array("OK".utf8).map(Int.init),
                    onForward: gate.onForward
                ),
                tokenizer: tokenizer
            )
        )
        let handle = try await fixture.start(
            conversation: turn1, parameters: parameters()
        )
        for try await _ in handle.stream {}
        await handle.waitForCompletion()
        await fixture.drain()  // clears the slot; drain generation 0 → 1
        gate.arm()

        let stored = turn1.appendingAssistant(.assistant(content: "OK"))
        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: []),
            partitionKey: CachePartitionKey(
                modelID: "toy/model", kvBits: nil, kvGroupSize: 64
            ),
            prefillStepSize: 1024,
            ssdEnabled: false,
            seedsPositionAnchor: false,
            canonicalLeafOffset: 0,
            diagnostics: PrefixCacheDiagnostics.Context(
                requestID: UUID(), modelID: "toy/model", kvBits: nil, kvGroupSize: 64
            )
        )
        await fixture.module.scheduleSpeculativePrefill(
            seed: seed,
            container: fixture.provider.container,
            entryDrainGeneration: 1,
            on: fixture.actor
        )
        await gate.reached()
        return (fixture, gate)
    }

    @Test
    func drainCancelsAndAwaitsTheSpeculativePrefill() async throws {
        let (fixture, gate) = try await Self.gatedLiveSpeculativePass()

        let drainFinished = AsyncFlag()
        let drainTask = Task {
            await fixture.drain()
            await drainFinished.set()
        }
        // The drain must not return while the pass is still mid-chunk.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await drainFinished.get() == false)

        gate.open()
        await drainTask.value
        #expect(await drainFinished.get())
    }

    @Test
    func staleSpeculativeFinishDoesNotDropANewerSlot() async throws {
        let (fixture, gate) = try await Self.gatedLiveSpeculativePass()

        // A finisher for some other (older) pass must not clear the slot:
        // the drain must still find, cancel, and await the live pass.
        await fixture.module.clearFinishedSpeculativePrefill(UUID(), on: fixture.actor)

        let drainFinished = AsyncFlag()
        let drainTask = Task {
            await fixture.drain()
            await drainFinished.set()
        }
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await drainFinished.get() == false)

        gate.open()
        await drainTask.value
        #expect(await drainFinished.get())
    }

    @Test
    func preemptAwaitsTheSpeculativePrefillSettle() async throws {
        let (fixture, gate) = try await Self.gatedLiveSpeculativePass()

        let preemptFinished = AsyncFlag()
        let preemptTask = Task {
            await fixture.module.preemptSpeculativePrefill(on: fixture.actor)
            await preemptFinished.set()
        }
        // The preempting entry's wait covers the pass's settle: it must not
        // return while the pass is still mid-chunk.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await preemptFinished.get() == false)

        gate.open()
        await preemptTask.value
        #expect(await preemptFinished.get())
    }
}

private actor AsyncFlag {
    private var value = false

    func set() {
        value = true
    }

    func get() -> Bool {
        value
    }
}
