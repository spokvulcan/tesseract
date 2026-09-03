import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The **Raw Generation Start** module (ADR-0016 amendment) driven through
/// the **Model Session** seam over the toy model — the agent chat turn and
/// both thinking-continuation shapes run the same script with no weights:
/// tokenize through the session's agent-edge verb, the progress-event
/// sequence, the **Prefill Strategy** route, the loop start, the handle
/// wrap. These paths had no test reach before the seam.
@MainActor
struct RawGenerationStartTests {

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

    /// A toy session whose model believes in `script`, over the sequencing
    /// tokenizer unless a test needs a rendering one.
    private static func toy(
        script: [Int],
        tokenizer: any Tokenizer = ToySequencingTokenizer(),
        onForward: (@Sendable (Int) -> Void)? = nil,
        reportsFlatTextTokens: Bool = false
    ) -> ToyModelSessionProvider {
        ToyModelSessionProvider(
            model: ToyLanguageModel(script: script, onForward: onForward),
            tokenizer: tokenizer,
            reportsFlatTextTokens: reportsFlatTextTokens
        )
    }

    /// The chat-template render of the one-message conversation.
    private static func render(_ tokenizer: any Tokenizer = ToySequencingTokenizer()) throws
        -> [Int]
    {
        try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    /// Run the module inside one toy session and drain the stream.
    private static func run(
        provider: ToyModelSessionProvider,
        prompt: sending RawGenerationPrompt,
        parameters: GenerateParameters,
        modelFingerprint: String? = nil,
        log: ProgressEventLog? = nil,
        onStarted: (@Sendable (HTTPServerRawGenerationStart) async -> Void)? = nil
    ) async throws -> String {
        let handler: ServerInferenceProgressHandler?
        if let log {
            handler = { event in log.append(event) }
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
        let render = try Self.render()
        let provider = Self.toy(script: render + Self.bytes("Done"))
        let log = ProgressEventLog()

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
        #expect(lookup.reason == "standardGenerationNoPrefixCache")
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
        let render = try Self.render()
        let handoff = "</think>"
        let provider = Self.toy(script: render + Self.bytes(handoff) + Self.bytes("After"))
        let log = ProgressEventLog()

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
        #expect(lookup.reason == "thinkingContinuationNoPrefixCache")
        #expect(lookup.promptTokens == render.count + handoff.utf8.count)
    }

    /// The two continuation shapes are one arm: re-tokenizing the original
    /// input yields the same prompt, the same stream, and the same count as
    /// the captured token list. This is the drift guard the old hand copies
    /// needed, expressed as the module's contract.
    @Test func continuationFromInputMatchesContinuationFromTokens() async throws {
        let render = try Self.render()
        let handoff = "</think>"
        let script = render + Self.bytes(handoff) + Self.bytes("Same")
        let fromTokensLog = ProgressEventLog()
        let fromInputLog = ProgressEventLog()

        let fromTokens = try await Self.run(
            provider: Self.toy(script: script),
            prompt: .continuation(base: .tokens(render, ndim: 1), handoff: handoff),
            parameters: Self.parameters(),
            log: fromTokensLog
        )
        let fromInputProvider = Self.toy(script: script)
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

    /// The **Prefill Strategy** route (ADR-0044), observed through the toy's
    /// forward offsets: a 2D text-only prompt longer than one step chunks
    /// through the app driver — forwards at each chunk boundary, then the
    /// remainder priming the iterator — while the same prompt as a flat 1D
    /// list goes single-shot, one forward over the whole prompt inside the
    /// vendor iterator's init. 21 prompt tokens at step 8.
    @Test(arguments: [(ndim: 2, forwards: [0, 8, 16]), (ndim: 1, forwards: [0, 21])])
    func prefillRouteFollowsThePromptRank(ndim: Int, forwards expected: [Int]) async throws {
        let base = Self.bytes(String(repeating: "a", count: 20))
        let forwards = ForwardLog()
        let provider = Self.toy(
            script: base + Self.bytes("!") + Self.bytes("ok"), onForward: forwards.onForward)

        let text = try await Self.run(
            provider: provider,
            prompt: .continuation(base: .tokens(base, ndim: ndim), handoff: "!"),
            parameters: Self.parameters(prefillStepSize: 8)
        )

        #expect(text == "ok")
        #expect(Array(forwards.offsets.prefix(expected.count)) == expected)
    }

    // MARK: - Cancellation

    /// The wrapped handle's `cancel` stops the loop mid-decode and
    /// `waitForCompletion` returns once the model is no longer touched.
    @Test func cancelStopsGenerationAndCompletionSettles() async throws {
        let render = try Self.render()
        let scripted = String(repeating: "x", count: 64)
        let gate = ForwardGate(threshold: render.count + 8)
        let provider = Self.toy(script: render + Self.bytes(scripted), onForward: gate.onForward)

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
        let truth = try Self.render(tokenizer)
        let cachedLog = ProgressEventLog()
        let cachedProvider = Self.toy(
            script: truth, tokenizer: tokenizer, reportsFlatTextTokens: true)
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

        let uncachedProvider = Self.toy(
            script: truth, tokenizer: tokenizer, reportsFlatTextTokens: true)
        _ = try await Self.run(
            provider: uncachedProvider,
            prompt: .fresh(UserInput(messages: Self.messages)),
            parameters: Self.parameters(),
            modelFingerprint: nil
        )
        #expect(uncachedProvider.recorder.verbs == [.prepare, .makeRawDecodeIterator])
    }
}
