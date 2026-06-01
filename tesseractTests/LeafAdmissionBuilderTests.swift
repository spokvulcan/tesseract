import Foundation
import Testing
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Leaf Admission Builder**'s reusable-prefix probe: the
/// GPU-free routing core that finds the shared token path a future continuation
/// can hydrate. Driven by a byte-level fake tokenizer — no model.
@Suite struct LeafAdmissionBuilderTests {

    private let tokenizer = FakeChatMLTokenizer()

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
