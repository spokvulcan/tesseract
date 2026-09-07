import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// The Emitted Path suites' tokenizer: a greedy longest-match vocabulary
/// (special tokens and a few scenario words as whole pieces, printable
/// ASCII as single-character pieces, so every id fits the toy vocabulary)
/// behind a Qwen3.8-shaped thinking template:
///
/// - a system block that carries the reasoning-effort sentence (ADR-0060:
///   the effort changes the render from token 0) whether or not thinking
///   is enabled — `enable_thinking` changes only the generation prompt,
///   the Qwen3-family shape spec #471 decision 23 names;
/// - assistant turns rendered with their `<think>` block — reasoning or
///   empty — after the last user message, or everywhere under
///   `preserve_thinking`; stripped to bare content before it otherwise;
/// - the generation prompt `<|im_start|>assistant\n<think>\n`, or the
///   closed empty block when thinking is disabled;
/// - `<|im_end|>` closing every turn as one token — the end-of-turn
///   marker the index keys on — followed by a newline.
struct EmittedPathToyTokenizer: ChatTemplateRendering {
    static let endOfTurn = "<|im_end|>"
    static let thinkingGenerationPrompt = "<|im_start|>assistant\n<think>\n"
    static let closedThinkingGenerationPrompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// Scenario words that encode as single tokens, so the same text can
    /// be emitted under different splits (`KNI` versus `K` + `NI`).
    static let words = [
        "hello", "world", "there", "plan", "probe", "summary", "more", "again",
        "KNI", "KN", "NI", "K", "N", "I",
    ]

    /// An image part renders as this run in place, the Qwen-VL shape.
    static let imagePad = "<|image_pad|>"
    static let imagePadRunLength = 4

    static let pieces: [String] = {
        let specials = [
            "<|im_start|>", endOfTurn, "<think>", "</think>", "\n",
            "user", "assistant", "system", "tool",
            "<|vision_start|>", "<|vision_end|>", imagePad,
        ]
        let ascii = (32...126).map { String(UnicodeScalar(UInt8($0))) }
        return specials + words + ascii
    }()

    let inner = GreedyTokenizer(pieces: Self.pieces)

    var bosToken: String? { inner.bosToken }
    var eosToken: String? { Self.endOfTurn }
    var unknownToken: String? { inner.unknownToken }

    /// The `<|im_end|>` id: the model's stop token and the index's marker.
    var endOfTurnID: Int { inner.convertTokenToId(Self.endOfTurn)! }
    /// The placeholder id an image part renders as (`image_token_id`).
    var imagePadID: Int { inner.convertTokenToId(Self.imagePad)! }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }

    /// Both generation prompts the template can emit, as ids.
    var generationPrompts: [[Int]] {
        [Self.thinkingGenerationPrompt, Self.closedThinkingGenerationPrompt].map {
            encode(text: $0, addSpecialTokens: false)
        }
    }

    func renderChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        let enableThinking = (additionalContext?["enable_thinking"] as? Bool) ?? true
        let preserveThinking = (additionalContext?["preserve_thinking"] as? Bool) ?? false
        let effort = (additionalContext?["reasoning_effort"] as? String) ?? "xhigh"

        var rendered = ""
        var body = messages[...]
        var system = ""
        if let first = body.first, first["role"] as? String == "system" {
            system = Self.text(of: first)
            body = body.dropFirst()
        }
        system += (system.isEmpty ? "" : "\n") + "Think with \(effort) effort."
        if !system.isEmpty {
            rendered += "<|im_start|>system\n\(system)\(Self.endOfTurn)\n"
        }
        let lastUserIndex = body.lastIndex { ($0["role"] as? String) == "user" }
        for (index, message) in zip(body.indices, body) {
            let role = message["role"] as? String ?? "user"
            let content = Self.text(of: message)
            guard role == "assistant" else {
                rendered += "<|im_start|>\(role)\n\(content)\(Self.endOfTurn)\n"
                continue
            }
            let keepsThinking = preserveThinking || index > (lastUserIndex ?? -1)
            if keepsThinking {
                let reasoning = message["reasoning_content"] as? String ?? ""
                rendered +=
                    "<|im_start|>assistant\n<think>\n\(reasoning)\n</think>\n\n"
                    + "\(content)\(Self.endOfTurn)\n"
            } else {
                rendered += "<|im_start|>assistant\n\(content)\(Self.endOfTurn)\n"
            }
        }
        if (additionalContext?["add_generation_prompt"] as? Bool) != false {
            rendered +=
                enableThinking
                ? Self.thinkingGenerationPrompt : Self.closedThinkingGenerationPrompt
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

    /// A message's text: a string content, or a content array's parts in
    /// order — text as is, an image as its framed placeholder run.
    private static func text(of message: [String: any Sendable]) -> String {
        if let content = message["content"] as? String { return content }
        guard let parts = message["content"] as? [[String: any Sendable]] else { return "" }
        return parts.map { part in
            switch part["type"] as? String {
            case "image":
                "<|vision_start|>"
                    + String(repeating: imagePad, count: imagePadRunLength)
                    + "<|vision_end|>"
            default:
                part["text"] as? String ?? ""
            }
        }.joined()
    }
}

/// A response-conversion fault between the model and the client: the
/// tokenizer the stream detokenizes through corrupts one decode — the first
/// whose ids include `targetID` — then decodes faithfully again, so the
/// text the client streamed (and echoes back) differs from what the model
/// fed while the Leaf Store's fidelity replay of the fed ids sees the truth.
final class FaultyStreamTokenizer: ChatTemplateRendering, @unchecked Sendable {
    let inner: EmittedPathToyTokenizer
    let targetID: Int
    let corrupt: @Sendable (String) -> String
    private let lock = NSLock()
    private var armed = true
    private(set) var corruptions = 0

    init(
        inner: EmittedPathToyTokenizer, targetID: Int,
        corrupt: @escaping @Sendable (String) -> String
    ) {
        self.inner = inner
        self.targetID = targetID
        self.corrupt = corrupt
    }

    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let text = inner.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        let fires = lock.withLock {
            guard armed, tokenIds.contains(targetID) else { return false }
            armed = false
            corruptions += 1
            return true
        }
        return fires ? corrupt(text) : text
    }

    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }

    func renderChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        try inner.renderChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try inner.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
