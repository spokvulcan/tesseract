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

    /// Mirrors the PARO think-strip: when set, assistant turns that precede a
    /// later user message render with their `<think>…</think>` block removed
    /// — the template keeps thinks only in the span after the last user
    /// message. Off by default so the existing tokenizer-affine suites keep
    /// byte-identical renders.
    var stripsThinkBeforeLastUser = false

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        // Lossy UTF-8 decode is intentional in this test fake.
        // swiftlint:disable:next optional_data_string_conversion
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

    /// Render-space image markers, mirroring the Qwen-VL framed single-pad
    /// shape. Outside the byte range so they can never collide with text
    /// tokens, and distinct from key-space pseudo-tokens (negative).
    static let visionStartTokenId = 999_000
    static let imagePadTokenId = 999_001
    static let visionEndTokenId = 999_002

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // Token-level assembly so image entries can render as single marker
        // tokens; pure-string contents produce byte-identical renders to the
        // previous string-concatenation implementation.
        var tokens: [Int] = []
        let lastUserIndex = messages.lastIndex { ($0["role"] as? String) == "user" }
        for (index, message) in messages.enumerated() {
            let role = message["role"] as? String ?? ""
            tokens += encode(text: "<|im_start|>\(role)\n", addSpecialTokens: false)
            if let parts = message["content"] as? [[String: any Sendable]] {
                for part in parts {
                    if part["type"] as? String == "image" {
                        tokens += [
                            Self.visionStartTokenId, Self.imagePadTokenId, Self.visionEndTokenId,
                        ]
                    } else {
                        tokens += encode(
                            text: part["text"] as? String ?? "", addSpecialTokens: false)
                    }
                }
            } else {
                var content = message["content"] as? String ?? ""
                if stripsThinkBeforeLastUser,
                    role == "assistant",
                    let lastUserIndex, index < lastUserIndex
                {
                    content = Self.strippingThink(content)
                }
                tokens += encode(text: content, addSpecialTokens: false)
            }
            tokens += encode(text: "<|im_end|>\n", addSpecialTokens: false)
        }
        if let tools, !tools.isEmpty {
            tokens += encode(text: "<tools>\(tools.count)</tools>\n", addSpecialTokens: false)
        }
        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        if addGenerationPrompt {
            tokens += encode(
                text: Self.generationPrompt(thinking: promptStartsThinking),
                addSpecialTokens: false
            )
        }
        return tokens
    }

    static func generationPrompt(thinking: Bool) -> String {
        thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n"
    }

    /// Remove one `<think>…</think>` block (plus a trailing newline) from an
    /// assistant turn, the way thinking templates rewrite history once a
    /// later user message exists.
    private static func strippingThink(_ content: String) -> String {
        guard let start = content.range(of: "<think>"),
            let end = content.range(of: "</think>")
        else { return content }
        var stripped = content
        stripped.removeSubrange(start.lowerBound..<end.upperBound)
        if stripped.hasPrefix("\n") { stripped.removeFirst() }
        return stripped
    }

    /// A **Cache Key Space** over a synthetic prepared sequence — one
    /// `imagePadTokenId` run per conversation image at the given run length,
    /// keyed by the conversation's own image digests. The tokenizer-affine
    /// suites' stand-in for a real processor prepare: `translate` only reads
    /// the image table, so the prepared text around the runs is arbitrary.
    static func keySpace(
        for conversation: HTTPPrefixCacheConversation,
        runLengths: [Int]
    ) throws -> CacheKeySpace {
        var prepared: [Int] = [1]
        var images: [CacheKeySpace.RequestImage] = []
        for (image, runLength) in zip(conversation.images, runLengths) {
            prepared += Array(repeating: imagePadTokenId, count: runLength) + [2]
            images.append(CacheKeySpace.RequestImage(digest: image.digest, positionSpan: runLength))
        }
        return try CacheKeySpace.make(
            preparedTokens: prepared,
            images: images,
            placeholderIdentity: ImagePlaceholderIdentity(imagePadTokenId: imagePadTokenId)
        ).get()
    }
}
