import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The DFlash2 arm run end to end on the toy (ADR-0079): the real vendor
/// iterator over the toy target and a scripted drafter, through both
/// consumers of the **Speculation Plan**. Greedy rounds are lossless, so the
/// text always equals the toy's script; a drafter that misses exercises
/// partial acceptance and the rewind of undrained drafts. Before the toy
/// could carry a drafter, no test reached either speculative arm.
@MainActor
struct SpeculativeDecodeToyTests {

    private static func bytes(_ text: String) -> [Int] {
        Array(text.utf8).map(Int.init)
    }

    private static func conversation(
        _ messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: nil, messages: messages)
    }

    private static func parameters(kvBits: Int? = nil) -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = kvBits
        return parameters
    }

    private static func engagedArms(_ log: ProgressEventLog) -> [SpeculativeArm] {
        log.events.compactMap { event in
            if case .speculationEngaged(let arm) = event { return arm }
            return nil
        }
    }

    // MARK: - Raw Generation Start

    @Test(arguments: [0, 3])
    func rawStartDecodesThroughTheDFlash2Plan(missEvery: Int) async throws {
        let tokenizer = ToySequencingTokenizer()
        let messages: [Message] = [["role": "user", "content": "Hi"]]
        let render = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let reply = "Hello from the draft, verified by the target."
        let model = ToyLanguageModel(script: render + Self.bytes(reply))
        let provider = ToyModelSessionProvider(
            model: model, tokenizer: tokenizer,
            speculation: .scriptedDFlash2(over: model, missEvery: missEvery))
        let log = ProgressEventLog()
        let parameters = LLMActor.makeGenerateParameters(from: Self.parameters())

        let start = try await provider.withSession(
            nonSendable: RawGenerationPrompt.fresh(
                UserInput(messages: messages), renderContext: .canonical)
        ) { session, prompt in
            try await RawGenerationStart.start(
                session: session, prompt: prompt, tools: nil, parameters: parameters,
                modelFingerprint: nil, progressHandler: { event in log.append(event) })
        }
        var text = ""
        var info: GenerateCompletionInfo?
        for await event in start.stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .info(let value): info = value
            default: break
            }
        }
        await start.waitForCompletion()

        #expect(text == reply)
        #expect(provider.recorder.verbs == [.prepare, .newCache, .makeSpeculativeDecodeIterator])
        #expect(Self.engagedArms(log) == [.dflash2])
        let proposed = try #require(info?.proposedDraftTokens)
        let accepted = try #require(info?.acceptedDraftTokens)
        #expect(proposed > 0)
        if missEvery == 0 {
            #expect(accepted == proposed)
        } else {
            #expect(accepted < proposed)
        }
    }

    // MARK: - Server Completion

    /// The keyed spine under DFlash2, cold then warm: the cold turn engages
    /// the arm and stores the leaf the speculative cache holds (the drafts
    /// the loop never drained are rewound first); the warm turn restores it
    /// and speculates again over the restored cache (ADR-0059).
    @Test(arguments: [0, 3])
    func keyedTurnsSpeculateColdThenWarmOverTheStoredLeaf(missEvery: Int) async throws {
        let tokenizer = ToySequencingTokenizer()
        let round1 = Self.conversation([HTTPPrefixCacheMessage(role: .user, content: "Hi")])
        let round2 = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: "Hi"),
            .assistant(content: "Hello!"),
            HTTPPrefixCacheMessage(role: .user, content: "More?"),
        ])
        let render1 = try tokenizer.applyChatTemplate(
            messages: round1.promptMessages, tools: nil, additionalContext: nil)
        let render2 = try tokenizer.applyChatTemplate(
            messages: round2.promptMessages, tools: nil, additionalContext: nil)
        let model = ToyLanguageModel(script: render2 + Self.bytes("Sure."))
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: model, tokenizer: tokenizer,
                speculation: .scriptedDFlash2(over: model, missEvery: missEvery)),
            modelID: "speculative-spine-\(UUID())")

        let log1 = ProgressEventLog()
        let handle1 = try await fixture.start(
            conversation: round1, parameters: Self.parameters(), progress: log1)
        #expect(handle1.cachedTokenCount == 0)
        let (text1, info1) = try await collectServerText(handle1)
        #expect(text1 == "Hello!")
        #expect((info1?.draftTokensProposed ?? 0) > 0)
        #expect(Self.engagedArms(log1) == [.dflash2])
        let round1Verbs = fixture.provider.recorder.verbs
        #expect(round1Verbs.contains(.makeSpeculativeDecodeIterator))
        #expect(!round1Verbs.contains(.makeDecodeIterator))

        // The leaf covers the prompt and the whole reply. Its last row is the
        // stop token when a round verified it, and not when the stop token
        // was the round's bonus, which has no row yet.
        let storedTokens1 = try tokenizer.applyChatTemplate(
            messages: round1.appendingAssistant(.assistant(content: "Hello!")).promptMessages,
            tools: nil, additionalContext: ["add_generation_prompt": false])
        let log2 = ProgressEventLog()
        let handle2 = try await fixture.start(
            conversation: round2, parameters: Self.parameters(), progress: log2)
        #expect(handle2.cachedTokenCount >= render1.count + "Hello!".utf8.count)
        #expect(handle2.cachedTokenCount <= storedTokens1.count)
        let (text2, info2) = try await collectServerText(handle2)
        #expect(text2 == "Sure.")
        #expect((info2?.draftTokensProposed ?? 0) > 0)
        #expect(Self.engagedArms(log2) == [.dflash2])
        let round2Verbs = Array(fixture.provider.recorder.verbs.dropFirst(round1Verbs.count))
        #expect(round2Verbs.contains(.makeSpeculativeDecodeIterator))
        #expect(!round2Verbs.contains(.makeDecodeIterator))
        await fixture.drain()
    }

    /// The split sits past the deepest capture, so a thinking turn keeps the
    /// boundary snapshots its canonical leaf is synthesized from, and the
    /// canonical leaf is stored by restoring the boundary: speculation and
    /// the prefix cache compound on the traffic ADR-0056's amendment had to
    /// park (ADR-0059).
    @Test func aThinkingTurnKeepsItsBoundarySnapshotsUnderDFlash2() async throws {
        let tokenizer = FakeParoThinkingTokenizer()
        let conversation = Self.conversation([
            .init(role: .user, content: String(repeating: "a", count: 200))
        ])
        let prompt = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let reply = "reasoning\n</think>\n\n" + String(repeating: "b", count: 40)
        let model = ToyLanguageModel(script: prompt + Self.bytes(reply))
        let modelID = "speculative-boundary-\(UUID())"
        let telemetry = TelemetryCapture(modelID: modelID)
        defer { telemetry.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: model, tokenizer: tokenizer, speculation: .scriptedDFlash2(over: model)),
            modelID: modelID)
        var parameters = Self.parameters()
        parameters.prefillStepSize = 64

        let handle = try await fixture.start(conversation: conversation, parameters: parameters)
        let (_, info) = try await collectServerText(handle)
        #expect((info?.draftTokensProposed ?? 0) > 0)
        await fixture.drain()

        let events = telemetry.drain()
        let prefilled = try #require(
            events.first {
                $0.eventName == "requestMemory" && $0.field("phase") == "prefilled"
                    && $0.field("boundaryCheckpointOffsets") != nil
            })
        #expect(prefilled.field("boundaryCheckpointOffsets") != "none")
        #expect(
            !events.contains {
                $0.eventName == "skip" && $0.field("reason") == "no-canonical-restore-boundary"
            })
        let leafStore = try #require(events.first { $0.eventName == "leafStore" })
        #expect(leafStore.field("mode") == HTTPLeafStoreMode.canonicalUserLeaf.rawValue)
        #expect(fixture.provider.recorder.verbs.contains(.restore))
    }

    /// A quantized-KV request keeps ordinary decoding with the draft
    /// resident: the plan refuses it, on the keyed path as on the raw one.
    @Test func aQuantizedKVTurnDecodesWithoutSpeculation() async throws {
        let tokenizer = ToySequencingTokenizer()
        let conversation = Self.conversation([HTTPPrefixCacheMessage(role: .user, content: "Hi")])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        // headDim 64 so the real quantizer can replace the attention caches.
        let model = ToyLanguageModel(script: render + Self.bytes("Hello!"), headDim: 64)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: model, tokenizer: tokenizer, speculation: .scriptedDFlash2(over: model)),
            modelID: "speculative-quantized-\(UUID())")
        let log = ProgressEventLog()

        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters(kvBits: 4), progress: log)
        let (text, info) = try await collectServerText(handle)
        #expect(text == "Hello!")
        #expect(info?.draftTokensProposed == nil)
        #expect(Self.engagedArms(log).isEmpty)
        let verbs = fixture.provider.recorder.verbs
        #expect(verbs.contains(.makeDecodeIterator))
        #expect(!verbs.contains(.makeSpeculativeDecodeIterator))
        await fixture.drain()
    }
}
