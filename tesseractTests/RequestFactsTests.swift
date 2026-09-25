import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Request Keying derives every per-request fact once (ADR-0070), for a
/// keyed request and an unkeyed one alike, on the toy Model Session.
struct RequestFactsTests {

    static let readTool: ToolSpec = [
        "type": "function",
        "function": ["name": "read", "parameters": ["type": "object"]] as [String: any Sendable],
    ]

    @Test func keyedRequestCarriesTheFactsDerivedOnce() async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]), tokenizer: FakeChatMLTokenizer())
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "sys", messages: [HTTPPrefixCacheMessage(role: .user, content: "hi")])
        let request = try await provider.withSession { session in
            var parameters = GenerateParameters(kvBits: 8, temperature: 0)
            parameters.prefill = PrefillParameters(stepSize: nil)
            guard
                case .keyed(let request, _) = try await RequestKeyingPhase.run(
                    session: session, conversation: conversation,
                    canonicalTools: [Self.readTool], renderContext: .canonical,
                    parameters: parameters, modelID: "toy/model", modelFingerprint: nil,
                    imageKeying: nil, ssdEnabled: true)
            else { throw ToyRequestKeying.NotKeyedAsExpected() }
            return request
        }
        let facts = request.facts
        #expect(facts.promptTokens == request.keySpace.keyPath)
        #expect(facts.generationPrompt.thinkBlock == .opens)
        // The one `?? 512`, and the decode parameters' one quantization reset.
        #expect(facts.prefillStepSize == 512)
        #expect(facts.decodeParameters.kvBits == nil)
        #expect(facts.partitionKey.kvBits == 8)
        #expect(facts.ssdEnabled)
        #expect(facts.toolsDefined)
        #expect(facts.toolCallFormat == .json)
        // Defined tools predict a tool call; a stop turn reads the think block.
        #expect(facts.predictedLeafStoreMode == .directToolLeaf)
        #expect(facts.leafStoreMode(emittedToolCalls: false) == .canonicalUserLeaf)
    }

    @Test func unkeyedRequestCarriesTheSameFacts() async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]), tokenizer: ToySequencingTokenizer(thinking: true))
        let facts = try await provider.withSession { session in
            try await ToyRequestKeying.unkeyedRequest(in: session, userText: "Hi").request.facts
        }
        // Checked against the prompt tokens, which end with the open block.
        #expect(facts.generationPrompt.thinkBlock == .opens)
        #expect(facts.promptTokens.last == ToySequencingTokenizer.assistantMarkTokenId)
        #expect(facts.prefillStepSize == 512)
        #expect(!facts.toolsDefined)
        #expect(!facts.ssdEnabled)
    }
}
