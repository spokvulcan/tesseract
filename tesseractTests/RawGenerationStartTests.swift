import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The **Raw Generation Start** module (ADR-0016 amendment) driven through
/// the **Model Session** seam over the toy model — the agent chat turn and
/// both thinking-continuation shapes run the same script with no weights:
/// tokenize through the session's one authority, the progress-event
/// sequence, the **Prefill Strategy** route, the loop start, the handle
/// wrap. These paths had no test reach before the seam.
@MainActor
struct RawGenerationStartTests {

    /// Progress events, as the Activity surfaces would receive them.
    @MainActor
    private final class EventLog {
        var events: [ServerInferenceProgressEvent] = []
    }

    /// Forward offsets the toy model reports, in order — the observable
    /// difference between the chunked and single-shot prefill routes.
    private final class ForwardLog: @unchecked Sendable {
        private let lock = NSLock()
        private var _offsets: [Int] = []
        var offsets: [Int] { lock.withLock { _offsets } }
        func record(_ offset: Int) {
            lock.withLock { _offsets.append(offset) }
        }
    }

    private static func parameters(prefillStepSize: Int? = nil) -> GenerateParameters {
        var agentParameters = AgentGenerateParameters()
        agentParameters.temperature = 0
        agentParameters.kvBits = nil
        var parameters = LLMActor.makeGenerateParameters(from: agentParameters)
        if let prefillStepSize {
            parameters.prefill = .init(stepSize: prefillStepSize)
        }
        return parameters
    }

    private static func bytes(_ text: String) -> [Int] {
        Array(text.utf8).map(Int.init)
    }

    private static let messages: [Message] = [["role": "user", "content": "Hi"]]

    /// Run the module inside one toy session and drain the stream.
    private static func run(
        provider: ToyModelSessionProvider,
        prompt: sending RawGenerationPrompt,
        parameters: GenerateParameters,
        modelFingerprint: String? = nil,
        log: EventLog? = nil,
        onStarted: (@Sendable (HTTPServerRawGenerationStart) async -> Void)? = nil
    ) async throws -> String {
        let handler: ServerInferenceProgressHandler?
        if let log {
            handler = { @MainActor event in log.events.append(event) }
        } else {
            handler = nil
        }
        let start = try await provider.withSession(nonSendable: prompt) { session, prompt in
            try await RawGenerationStart.start(
                session: session,
                prompt: prompt,
                tools: nil,
                parameters: parameters,
                modelFingerprint: modelFingerprint,
                progressHandler: handler
            )
        }
        await onStarted?(start)
        var text = ""
        for await event in start.stream {
            if case .chunk(let chunk) = event { text += chunk }
        }
        await start.waitForCompletion()
        return text
    }

    // MARK: - Fresh turn

    /// The agent chat turn: the toy decodes its scripted completion, and the
    /// module reports the lookup and prefill events in order with the
    /// prompt token count the tokenizer produced. No speculation badge: the
    /// toy session has no drafter.
    @Test func freshTurnDecodesTheScriptAndReportsProgress() async throws {
        let tokenizer = ToySequencingTokenizer()
        let render = try tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil)
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: render + Self.bytes("Done")),
            tokenizer: tokenizer
        )
        let log = EventLog()

        let text = try await Self.run(
            provider: provider,
            prompt: .fresh(UserInput(messages: Self.messages)),
            parameters: Self.parameters(),
            log: log
        )

        #expect(text == "Done")
        #expect(provider.recorder.verbs == [.prepare, .makeRawDecodeIterator])
        let kinds = log.events.map { event -> String in
            switch event {
            case .cacheLookupStarted: "lookupStarted"
            case .cacheLookupFinished: "lookupFinished"
            case .prefillStarted: "prefillStarted"
            case .prefillFinished: "prefillFinished"
            case .speculationEngaged: "speculationEngaged"
            }
        }
        #expect(kinds == ["lookupStarted", "lookupFinished", "prefillStarted", "prefillFinished"])
        guard case .cacheLookupFinished(let lookup) = log.events[1] else {
            Issue.record("expected a lookup-finished event")
            return
        }
        #expect(lookup.reason == RawGenerationStart.freshLookupReason)
        #expect(lookup.promptTokens == render.count)
        #expect(lookup.cachedTokens == 0)
        guard case .prefillFinished(let prefill) = log.events[3] else {
            Issue.record("expected a prefill-finished event")
            return
        }
        #expect(prefill.newTokensToPrefill == render.count)
        #expect(prefill.prefillMs != nil)
    }

    // MARK: - Continuations

    /// The server's continuation shape: the captured token list extended
    /// with the hand-off encoded as plain text — the prompt count is the
    /// original plus the appended tokens, and decode picks up right after
    /// the hand-off.
    @Test func continuationFromTokensAppendsTheHandoff() async throws {
        let tokenizer = ToySequencingTokenizer()
        let render = try tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil)
        let handoff = "</think>"
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: render + Self.bytes(handoff) + Self.bytes("After")),
            tokenizer: tokenizer
        )
        let log = EventLog()

        let text = try await Self.run(
            provider: provider,
            prompt: .continuation(base: .tokens(render, ndim: 1), handoff: handoff),
            parameters: Self.parameters(),
            log: log
        )

        #expect(text == "After")
        // No tokenize verb: the base is already tokens.
        #expect(provider.recorder.verbs == [.makeRawDecodeIterator])
        guard case .cacheLookupFinished(let lookup) = log.events[1] else {
            Issue.record("expected a lookup-finished event")
            return
        }
        #expect(lookup.reason == RawGenerationStart.continuationLookupReason)
        #expect(lookup.promptTokens == render.count + handoff.utf8.count)
    }

    /// The two continuation shapes are one arm: re-tokenizing the original
    /// input yields the same prompt, the same stream, and the same count as
    /// the captured token list. This is the drift guard the old hand copies
    /// needed, expressed as the module's contract.
    @Test func continuationFromInputMatchesContinuationFromTokens() async throws {
        let tokenizer = ToySequencingTokenizer()
        let render = try tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil)
        let handoff = "</think>"
        let script = render + Self.bytes(handoff) + Self.bytes("Same")
        let fromTokensLog = EventLog()
        let fromInputLog = EventLog()

        let fromTokens = try await Self.run(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script), tokenizer: tokenizer),
            prompt: .continuation(base: .tokens(render, ndim: 1), handoff: handoff),
            parameters: Self.parameters(),
            log: fromTokensLog
        )
        let fromInputProvider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: script), tokenizer: tokenizer)
        let fromInput = try await Self.run(
            provider: fromInputProvider,
            prompt: .continuation(
                base: .input(UserInput(messages: Self.messages)), handoff: handoff),
            parameters: Self.parameters(),
            log: fromInputLog
        )

        #expect(fromTokens == "Same")
        #expect(fromInput == fromTokens)
        #expect(fromInputProvider.recorder.verbs == [.prepare, .makeRawDecodeIterator])
        guard case .cacheLookupFinished(let a) = fromTokensLog.events[1],
            case .cacheLookupFinished(let b) = fromInputLog.events[1]
        else {
            Issue.record("expected lookup-finished events")
            return
        }
        #expect(a.promptTokens == b.promptTokens)
    }

    // MARK: - Prefill route

    /// A 2D text-only prompt longer than one step takes the chunked route
    /// (ADR-0044): the toy sees forwards at each chunk boundary before the
    /// iterator's prime forward, where a single-shot prompt sees one.
    @Test func longTwoDimensionalPromptChunksThroughTheAppDriver() async throws {
        let tokenizer = ToySequencingTokenizer()
        let base = Self.bytes(String(repeating: "a", count: 20))
        let forwards = ForwardLog()
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(
                script: base + Self.bytes("!") + Self.bytes("ok"),
                onForward: forwards.record),
            tokenizer: tokenizer
        )

        let text = try await Self.run(
            provider: provider,
            prompt: .continuation(base: .tokens(base, ndim: 2), handoff: "!"),
            parameters: Self.parameters(prefillStepSize: 8)
        )

        #expect(text == "ok")
        // 21 prompt tokens at step 8: chunks at 0 and 8, then the remainder
        // primes the iterator at 16 — decode forwards follow from 21.
        #expect(Array(forwards.offsets.prefix(3)) == [0, 8, 16])
    }

    /// The same prompt as a flat 1D list goes single-shot: one forward over
    /// the whole prompt inside the vendor iterator's init.
    @Test func flatPromptPrefillsSingleShot() async throws {
        let tokenizer = ToySequencingTokenizer()
        let base = Self.bytes(String(repeating: "a", count: 20))
        let forwards = ForwardLog()
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(
                script: base + Self.bytes("!") + Self.bytes("ok"),
                onForward: forwards.record),
            tokenizer: tokenizer
        )

        let text = try await Self.run(
            provider: provider,
            prompt: .continuation(base: .tokens(base, ndim: 1), handoff: "!"),
            parameters: Self.parameters(prefillStepSize: 8)
        )

        #expect(text == "ok")
        #expect(Array(forwards.offsets.prefix(2)) == [0, 21])
    }

    // MARK: - Cancellation

    /// The wrapped handle's `cancel` stops the loop mid-decode and
    /// `waitForCompletion` returns once the model is no longer touched.
    @Test func cancelStopsGenerationAndCompletionSettles() async throws {
        let tokenizer = ToySequencingTokenizer()
        let render = try tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil)
        let scripted = String(repeating: "x", count: 64)
        let gate = ForwardGate(threshold: render.count + 8)
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(
                script: render + Self.bytes(scripted), onForward: gate.onForward),
            tokenizer: tokenizer
        )

        let text = try await Self.run(
            provider: provider,
            prompt: .fresh(UserInput(messages: Self.messages)),
            parameters: Self.parameters()
        ) { start in
            await gate.reached()
            start.cancel()
            gate.open()
        }

        #expect(text.count < scripted.count)
    }

    // MARK: - Tokenize authority

    /// A flat-token model with a rendering tokenizer under a known
    /// fingerprint tokenizes through the **Render+Token Cache**: no
    /// `prepare` verb runs, and the prompt count equals the fused render.
    /// Without a fingerprint the same request falls back to the processor.
    @Test func textOnlyRequestTokenizesThroughTheRenderTokenCache() async throws {
        let tokenizer = GreedyTokenizer(pieces: [
            "<|im_start|>", "<|im_end|>", "assistant", "user", "system", "\n", "Hi",
        ])
        let truth = try tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil)
        let cachedLog = EventLog()
        let cachedProvider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: truth),
            tokenizer: tokenizer,
            reportsFlatTextTokens: true
        )
        _ = try await Self.run(
            provider: cachedProvider,
            prompt: .fresh(UserInput(messages: Self.messages)),
            parameters: Self.parameters(),
            modelFingerprint: "raw-start-\(UUID().uuidString)",
            log: cachedLog
        )
        #expect(cachedProvider.recorder.verbs == [.makeRawDecodeIterator])
        guard case .cacheLookupFinished(let lookup) = cachedLog.events[1] else {
            Issue.record("expected a lookup-finished event")
            return
        }
        #expect(lookup.promptTokens == truth.count)

        let uncachedProvider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: truth),
            tokenizer: tokenizer,
            reportsFlatTextTokens: true
        )
        _ = try await Self.run(
            provider: uncachedProvider,
            prompt: .fresh(UserInput(messages: Self.messages)),
            parameters: Self.parameters(),
            modelFingerprint: nil
        )
        #expect(uncachedProvider.recorder.verbs == [.prepare, .makeRawDecodeIterator])
    }
}
