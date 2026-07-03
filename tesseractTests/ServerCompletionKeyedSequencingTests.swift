import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// A minimal chat-template tokenizer whose renders are *row-consistent* with
/// what generation physically leaves in the KV cache: every message renders
/// as `[roleMark] + content-bytes + [EOT]`, the generation prompt is the
/// assistant's own role mark, and EOT doubles as the EOS the model emits.
/// A stored render (prompt + assistant turn, no generation prompt) is then
/// token-identical to the rows the drive produced — prompt, completion
/// bytes, forwarded EOS — so leaf offsets line up exactly, the property the
/// real ChatML template + real tokenizer pair provides in production.
nonisolated struct ToySequencingTokenizer: Tokenizer {
    static let eotTokenId = 300
    /// Assistant role mark == generation prompt: opening the assistant turn
    /// in history renders the same token generation started from.
    static let assistantMarkTokenId = 301
    static let systemMarkTokenId = 310
    static let userMarkTokenId = 311
    static let toolMarkTokenId = 313

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        // Lossy byte decode; marker ids (≥ 256) are dropped like specials.
        // swiftlint:disable:next optional_data_string_conversion
        String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
    }
    func tokenize(text: String) -> [String] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eot>" }
    var eosTokenId: Int? { Self.eotTokenId }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    private func roleMark(_ role: String) -> Int {
        switch role {
        case "assistant": Self.assistantMarkTokenId
        case "system": Self.systemMarkTokenId
        case "tool": Self.toolMarkTokenId
        default: Self.userMarkTokenId
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            tokens.append(roleMark(message["role"] as? String ?? "user"))
            tokens += encode(text: message["content"] as? String ?? "", addSpecialTokens: false)
            tokens.append(Self.eotTokenId)
        }
        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        if addGenerationPrompt {
            tokens.append(Self.assistantMarkTokenId)
        }
        return tokens
    }
}

/// PR B gating coverage (PRD #137, ADR-0016): the keyed spine — Snapshot
/// Resolution → restore → suffix prefill → drive → leaf capture → Snapshot
/// Admission — and the cancellation orderings, all through the module's
/// public entry with the toy-model-backed **Model Session**.
@Suite struct ServerCompletionKeyedSequencingTests {

    private static func conversation(
        _ messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: nil, messages: messages)
    }

    @MainActor
    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        // The toy's head dimension is far below any quantization group size;
        // KV quantization stays a recorded no-op verb in these suites.
        parameters.kvBits = nil
        return parameters
    }

    /// The keyed spine, cold then warm. Round 1 (cold miss) must run
    /// prepare → newCache → chunked prefill → (no-op) quantize → decode
    /// iterator, then capture the post-generation leaf. Round 2 extends the
    /// conversation, must resolve the admitted leaf, restore it *before*
    /// prefilling only the suffix, and decode the scripted continuation —
    /// proving the restored rows landed where the script expects them.
    @Test func keyedSpineRestoresAdmittedLeafAndPrefillsOnlyTheSuffix() async throws {
        let tokenizer = ToySequencingTokenizer()
        let round1 = Self.conversation([HTTPPrefixCacheMessage(role: .user, content: "Hi")])
        let round2 = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: "Hi"),
            .assistant(content: "Hello!"),
            HTTPPrefixCacheMessage(role: .user, content: "More?"),
        ])

        // The toy believes in the full two-round transcript: round 2's
        // render (whose prefix is round 1's render, then round 1's
        // completion + EOS) followed by round 2's completion.
        let render1 = try tokenizer.applyChatTemplate(
            messages: round1.promptMessages, tools: nil, additionalContext: nil
        )
        let render2 = try tokenizer.applyChatTemplate(
            messages: round2.promptMessages, tools: nil, additionalContext: nil
        )
        #expect(Array(render2.prefix(render1.count)) == render1)
        let script = render2 + Array("Sure.".utf8).map(Int.init)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script),
                tokenizer: tokenizer
            )
        )

        // -- Round 1: cold.
        let handle1 = try await fixture.start(
            conversation: round1, parameters: Self.parameters()
        )
        #expect(handle1.cachedTokenCount == 0)
        let (text1, info1) = try await collectServerText(handle1)
        #expect(text1 == "Hello!")
        #expect(try #require(info1).promptTokenCount == render1.count)
        let round1Verbs = fixture.provider.recorder.verbs
        #expect(
            round1Verbs == [
                .prepare, .newCache, .prefill, .quantizeKVCache, .makeDecodeIterator,
                .captureSnapshot,
            ]
        )

        // Row-consistency: the drive left prompt + completion + forwarded
        // EOS in the cache — exactly the stored render's length.
        let storedTokens1 = try tokenizer.applyChatTemplate(
            messages: round1.appendingAssistant(.assistant(content: "Hello!")).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        #expect(storedTokens1.count == render1.count + "Hello!".utf8.count + 1)

        // -- Round 2: warm. The admitted leaf must be restored, then only
        // the suffix prefilled.
        let handle2 = try await fixture.start(
            conversation: round2, parameters: Self.parameters()
        )
        #expect(handle2.cachedTokenCount == storedTokens1.count)
        let (text2, info2) = try await collectServerText(handle2)
        #expect(text2 == "Sure.")
        #expect(try #require(info2).promptTokenCount == render2.count)
        let round2Verbs = Array(fixture.provider.recorder.verbs.dropFirst(round1Verbs.count))
        #expect(
            round2Verbs == [
                .prepare, .restore, .prefill, .quantizeKVCache, .makeDecodeIterator,
                .captureSnapshot,
            ]
        )

        await fixture.drain()
    }

    /// **Salvage-on-cancel** (issue #97): a cancel landing between prefill
    /// chunks must admit the completed progress as a RAM-only leaf and
    /// release the in-flight start, so a re-sent request resumes from the
    /// salvaged offset instead of the restore floor.
    @Test func cancelMidPrefillSalvagesProgressAndReleasesTheStart() async throws {
        let tokenizer = ToySequencingTokenizer()
        let longContent = String(repeating: "a", count: 4000)
        let conversation = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: longContent)
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let script = render + Array("OK".utf8).map(Int.init)

        // Pause the second prefill chunk (offset 1024, prefillStepSize 1024)
        // so the cancel deterministically lands at the following chunk
        // boundary — past the salvage progress threshold of 2,048 tokens.
        let gate = ForwardGate(threshold: 1024)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script, onForward: gate.onForward),
                tokenizer: tokenizer
            )
        )

        let startTask = Task {
            try await fixture.start(conversation: conversation, parameters: Self.parameters())
        }
        await gate.reached()
        startTask.cancel()
        gate.open()

        guard case .failure(let error) = await startTask.result else {
            Issue.record("cancelled start must throw, not return a handle")
            return
        }
        #expect(error is CancellationError)

        // The registry released the in-flight start: the unload drain has
        // nothing to park on.
        await fixture.drain()

        // The re-sent request resumes from the salvaged chunk boundary and
        // completes the scripted response.
        let retry = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        #expect(retry.cachedTokenCount >= 2048)
        #expect(retry.cachedTokenCount < render.count)
        let (text, _) = try await collectServerText(retry)
        #expect(text == "OK")
        await fixture.drain()
    }

    /// Abort during the drive (decode underway): the stream must end, the
    /// registry slot must be released to the drain, and nothing may be
    /// admitted for the aborted turn — a re-sent request starts cold.
    @Test func abortDuringDriveReleasesSlotAndAdmitsNothing() async throws {
        let tokenizer = ToySequencingTokenizer()
        let conversation = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: "Hi")
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let completion = String(repeating: "x", count: 64)
        let script = render + Array(completion.utf8).map(Int.init)

        // Pause decode a few tokens in, cancel while generation is live.
        let gate = ForwardGate(threshold: render.count + 8)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script, onForward: gate.onForward),
                tokenizer: tokenizer
            )
        )

        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        await gate.reached()
        handle.cancel()
        gate.open()
        // Consume whatever was emitted; the stream must terminate.
        for try await _ in handle.stream {}
        await handle.waitForCompletion()

        // Slot released — the drain returns instead of parking.
        await fixture.drain()

        // Nothing was admitted for the aborted turn: the re-send is cold and
        // completes the full scripted response.
        let retry = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        #expect(retry.cachedTokenCount == 0)
        let (text, _) = try await collectServerText(retry)
        #expect(text == completion)
        await fixture.drain()
    }
}
