import Foundation
import Testing
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Leaf Admission Builder**'s reusable-prefix probe: the
/// GPU-free routing core that finds the shared token path a future continuation
/// can hydrate. Driven by a byte-level fake tokenizer — no model.
@Suite struct LeafAdmissionBuilderTests {

    /// One UTF-8 byte ⇒ one token; `applyChatTemplate` renders the ChatML
    /// envelope and never appends a generation prompt when told not to.
    private struct FakeTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map(Int.init)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
        }
        func tokenize(text: String) -> [String] { [] }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        var bosToken: String? { nil }
        var bosTokenId: Int? { nil }
        var eosToken: String? { "<|im_end|>" }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var rendered = ""
            for message in messages {
                let role = message["role"] as? String ?? ""
                let content = message["content"] as? String ?? ""
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
            if addGenerationPrompt {
                rendered += "<|im_start|>assistant\n"
            }
            return encode(text: rendered, addSpecialTokens: false)
        }
    }

    private let tokenizer = FakeTokenizer()

    private func conversation(
        systemPrompt: String? = "You are helpful.",
        messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: systemPrompt, messages: messages)
    }

    private func render(_ conversation: HTTPPrefixCacheConversation) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
    }

    @Test func userTurnProbeReturnsTheStoredRenderWithoutTheProbeContinuation() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "question"),
            HTTPPrefixCacheMessage(role: .assistant, content: "answer"),
        ])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer
        )
        // The reusable prefix is exactly the stored turn's own render — the
        // synthetic user continuation is excluded.
        #expect(prefix == (try render(stored)))
    }

    @Test func toolResultProbeReturnsTheStoredRenderWithoutTheProbeContinuation() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "call the tool"),
            HTTPPrefixCacheMessage(role: .assistant, content: "calling"),
        ])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .toolResult,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer
        )
        #expect(prefix == (try render(stored)))
    }

    @Test func probeDivergesToNilWhenThereIsNoCommonPrefix() throws {
        // An empty conversation renders to nothing, so the stored render and the
        // probe-extended render share no prefix.
        let empty = HTTPPrefixCacheConversation(systemPrompt: nil, messages: [])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: empty,
            toolSpecs: nil,
            tokenizer: tokenizer
        )
        #expect(prefix == nil)
    }
}
