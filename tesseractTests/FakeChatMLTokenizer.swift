import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// A byte-level ChatML test tokenizer (one UTF-8 byte ⇒ one token) shared by the
/// tokenizer-affine Server suites (`PrefillPlannerTests`, `LeafAdmissionBuilderTests`).
///
/// `applyChatTemplate` reproduces the `<|im_start|>role\ncontent<|im_end|>\n`
/// envelope, optionally renders a `<tools>count</tools>` marker, and — unless
/// `add_generation_prompt` is false — appends the generation prompt (think or
/// non-think per `promptStartsThinking`).
///
/// `StablePrefixDetectorTests` keeps its own `MockTokenizer`: it renders tools as
/// `[tools:...]` and has no generation-prompt support, so it is a genuinely
/// different template, not a copy of this one.
struct FakeChatMLTokenizer: Tokenizer {
    var promptStartsThinking = true

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
        if let tools, !tools.isEmpty {
            rendered += "<tools>\(tools.count)</tools>\n"
        }
        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        if addGenerationPrompt {
            rendered += Self.generationPrompt(thinking: promptStartsThinking)
        }
        return encode(text: rendered, addSpecialTokens: false)
    }

    static func generationPrompt(thinking: Bool) -> String {
        thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n"
    }
}
