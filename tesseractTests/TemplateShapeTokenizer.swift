import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// A rendering test tokenizer whose chat template takes one of the shapes the
/// **Generation Prompt** measurement has to tell apart (ADR-0070). Byte-level
/// (one UTF-8 byte per token) unless the shape merges across the append
/// point. A class, so the probe memo can key on the instance when the model
/// fingerprint is unknown, and it counts its renders for the probe-cost
/// tests.
nonisolated final class TemplateShapeTokenizer: ChatTemplateRendering, @unchecked Sendable {

    enum Shape: Sendable {
        /// The Qwen3.8 shape: an open think block by default, a closed empty
        /// one with `enable_thinking: false`.
        case thinkingByDefault
        /// The Qwen3.5-0.8B shape: a closed empty block by default, an open
        /// one only when `enable_thinking: true` is asked for.
        case thinkingWhenAsked
        /// A template that is not ChatML-shaped: Gemma's
        /// `<start_of_turn>model` header, no think tag.
        case gemma
        /// `add_generation_prompt` appends nothing.
        case appendsNothing
        /// The vocabulary merges the newline closing the last turn with the
        /// `<` opening the prompt into one token.
        case mergesAcrossAppend
        /// The prompt names the conversation's message count, so a probe of
        /// one message measures a prompt a longer request never feeds.
        case conversationDependent
        /// The template throws on a one-message conversation.
        case throwsOnProbe
        /// Adding the prompt also rewrites the last user turn.
        case rewritesHistory
    }

    struct ProbeRejected: Error {}

    let shape: Shape

    /// The id the merging vocabulary spells `"\n<"` with.
    static let mergedTokenID = 1_000

    private let lock = NSLock()
    private var renders = 0

    init(_ shape: Shape) {
        self.shape = shape
    }

    /// Template renders so far.
    var renderCount: Int { lock.withLock { renders } }

    /// The generation prompt this template appends under `additionalContext`
    /// for a conversation of `messageCount` messages.
    func generationPrompt(
        additionalContext: [String: any Sendable]? = nil, messageCount: Int = 1
    ) -> String {
        let enableThinking = additionalContext?["enable_thinking"] as? Bool
        switch shape {
        case .thinkingByDefault, .mergesAcrossAppend, .throwsOnProbe, .rewritesHistory:
            return enableThinking == false
                ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                : "<|im_start|>assistant\n<think>\n"
        case .thinkingWhenAsked:
            return enableThinking == true
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        case .gemma:
            return "<start_of_turn>model\n"
        case .appendsNothing:
            return ""
        case .conversationDependent:
            return "<|im_start|>assistant \(messageCount)\n"
        }
    }

    func renderChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        lock.withLock { renders += 1 }
        if shape == .throwsOnProbe, messages.count < 2 { throw ProbeRejected() }
        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        var rendered = ""
        for (index, message) in messages.enumerated() {
            let role = message["role"] as? String ?? "user"
            var content = message["content"] as? String ?? ""
            if shape == .rewritesHistory, addGenerationPrompt, index == messages.count - 1 {
                content += " (answer now)"
            }
            if shape == .gemma {
                let turn = role == "assistant" ? "model" : role
                rendered += "<start_of_turn>\(turn)\n\(content)<end_of_turn>\n"
            } else {
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
        }
        if addGenerationPrompt {
            rendered += generationPrompt(
                additionalContext: additionalContext, messageCount: messages.count)
        }
        return rendered
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: try renderChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext),
            addSpecialTokens: false)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let bytes = Array(text.utf8)
        var ids: [Int] = []
        var offset = 0
        while offset < bytes.count {
            if shape == .mergesAcrossAppend, offset + 1 < bytes.count,
                bytes[offset] == UInt8(ascii: "\n"), bytes[offset + 1] == UInt8(ascii: "<")
            {
                ids.append(Self.mergedTokenID)
                offset += 2
                continue
            }
            ids.append(Int(bytes[offset]))
            offset += 1
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let bytes = tokenIds.flatMap { id -> [UInt8] in
            id == Self.mergedTokenID ? Array("\n<".utf8) : [UInt8(truncatingIfNeeded: id)]
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }
}
