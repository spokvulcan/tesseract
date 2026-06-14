import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// A byte-level test tokenizer whose chat template mirrors the Qwen3.5-PARO
/// thinking template's *render semantics* — the parts the canonical-echo
/// fidelity contracts depend on:
///
/// - `last_query_index`: the last `user` message whose content is not a
///   `<tool_response>…</tool_response>` wrapper.
/// - Assistant turns after it render `<think>\n{reasoning}\n</think>\n\n` +
///   content; turns at-or-before it render content only (**Think-Strip
///   Rewind** semantics).
/// - Reasoning comes from the `reasoning_content` field when present, else is
///   extracted from a `<think>…</think>` block embedded in content — both
///   forms must render identically.
/// - Tool calls re-render as `<tool_call>\n<function=name>\n<parameter=…>`
///   blocks with parameters in **sorted key order**, mirroring swift-jinja's
///   sorted dictionary conversion (`Value(any:)`).
/// - Tool results render inside a `user`-role `<tool_response>` wrapper.
///
/// `FakeChatMLTokenizer` stays untouched: its byte-exact renders anchor the
/// planner/leaf-builder suites; this fake exists for the fidelity contracts.
struct FakeParoThinkingTokenizer: Tokenizer {
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

        if let tools, !tools.isEmpty {
            rendered += "<|im_start|>system\n<tools>\(tools.count)</tools>"
            if let system = messages.first, system["role"] as? String == "system" {
                let content = (system["content"] as? String ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !content.isEmpty { rendered += "\n\n" + content }
            }
            rendered += "<|im_end|>\n"
        } else if let system = messages.first, system["role"] as? String == "system" {
            let content = (system["content"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            rendered += "<|im_start|>system\n" + content + "<|im_end|>\n"
        }

        // The Preserve-Thinking Render (issue #98): with the flag set, every
        // assistant turn keeps its think block — modeled by treating the
        // strip horizon as "before the conversation", exactly how Qwen3.6's
        // template neutralizes its last-query scan.
        let preserveThinking =
            (additionalContext?[TemplateRenderFlag.preserveThinking.rawValue] as? Bool) ?? false
        let lastQueryIndex = preserveThinking ? -1 : Self.lastQueryIndex(of: messages)

        for (index, message) in messages.enumerated() {
            let role = message["role"] as? String ?? ""
            let content = (message["content"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            switch role {
            case "system":
                continue  // rendered above; mid-conversation systems out of scope
            case "user":
                rendered += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "tool":
                let previousIsTool = index > 0 && (messages[index - 1]["role"] as? String) == "tool"
                if !previousIsTool { rendered += "<|im_start|>user" }
                rendered +=
                    "\n<tool_response>\n" + (message["content"] as? String ?? "")
                    + "\n</tool_response>"
                let nextIsTool =
                    index + 1 < messages.count
                    && (messages[index + 1]["role"] as? String) == "tool"
                if !nextIsTool { rendered += "<|im_end|>\n" }
            case "assistant":
                let (reasoning, strippedContent) = Self.splitReasoning(
                    content: content,
                    reasoningField: message["reasoning_content"] as? String
                )
                rendered += "<|im_start|>assistant\n"
                if index > lastQueryIndex {
                    rendered += "<think>\n" + reasoning + "\n</think>\n\n"
                }
                rendered += strippedContent
                if let toolCalls = message["tool_calls"] as? [[String: any Sendable]] {
                    for (callIndex, call) in toolCalls.enumerated() {
                        let function = call["function"] as? [String: any Sendable] ?? [:]
                        let name = function["name"] as? String ?? ""
                        if callIndex == 0 {
                            rendered += strippedContent.isEmpty ? "" : "\n\n"
                        } else {
                            rendered += "\n"
                        }
                        rendered += "<tool_call>\n<function=\(name)>\n"
                        let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
                        for key in arguments.keys.sorted() {
                            rendered +=
                                "<parameter=\(key)>\n\(arguments[key].map { "\($0)" } ?? "")\n</parameter>\n"
                        }
                        rendered += "</function>\n</tool_call>"
                    }
                }
                rendered += "<|im_end|>\n"
            default:
                break
            }
        }

        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        if addGenerationPrompt {
            rendered += "<|im_start|>assistant\n<think>\n"
        }
        return encode(text: rendered, addSpecialTokens: false)
    }

    /// The template's reverse scan: index of the last real user query.
    private static func lastQueryIndex(of messages: [[String: any Sendable]]) -> Int {
        for index in messages.indices.reversed() {
            guard messages[index]["role"] as? String == "user" else { continue }
            let content = (messages[index]["content"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !(content.hasPrefix("<tool_response>") && content.hasSuffix("</tool_response>")) {
                return index
            }
        }
        return messages.count - 1
    }

    /// Field-first reasoning resolution with embedded-`<think>` fallback,
    /// mirroring template lines 90–99.
    private static func splitReasoning(
        content: String,
        reasoningField: String?
    ) -> (reasoning: String, content: String) {
        if let reasoningField {
            return (reasoningField.trimmingCharacters(in: .whitespacesAndNewlines), content)
        }
        guard let closeRange = content.range(of: "</think>", options: .backwards) else {
            return ("", content)
        }
        var reasoning = String(content[..<closeRange.lowerBound])
        if let openRange = reasoning.range(of: "<think>") {
            reasoning = String(reasoning[openRange.upperBound...])
        }
        var stripped = String(content[closeRange.upperBound...])
        while stripped.hasPrefix("\n") { stripped.removeFirst() }
        return (
            reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            stripped
        )
    }
}
