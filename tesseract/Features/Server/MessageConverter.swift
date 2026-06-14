//
//  MessageConverter.swift
//  tesseract
//

import Foundation
import MLXLMCommon
import Tokenizers

nonisolated enum HTTPPrefixCacheEligibility: Sendable, Equatable, CustomStringConvertible {
    case eligible(HTTPPrefixCacheConversation)
    case nonTextSystemMessage(index: Int)
    case nonTextUserMessage(index: Int)
    case nonTextAssistantMessage(index: Int)
    case nonTextToolMessage(index: Int)

    var conversation: HTTPPrefixCacheConversation? {
        switch self {
        case .eligible(let conversation):
            conversation
        default:
            nil
        }
    }

    var description: String {
        switch self {
        case .eligible(let conversation):
            let lastRole = conversation.lastMessage.map { "\($0.role)" } ?? "none"
            return "eligible(messages=\(conversation.messages.count), lastRole=\(lastRole))"
        case .nonTextSystemMessage(let index):
            return "nonTextSystemMessage(index=\(index))"
        case .nonTextUserMessage(let index):
            return "nonTextUserMessage(index=\(index))"
        case .nonTextAssistantMessage(let index):
            return "nonTextAssistantMessage(index=\(index))"
        case .nonTextToolMessage(let index):
            return "nonTextToolMessage(index=\(index))"
        }
    }
}

/// Converts between OpenAI-compatible API types and internal `LLMMessage` types.
enum MessageConverter {

    // MARK: - Messages

    /// One normalization of an OpenAI request into both downstream shapes:
    /// the `LLMMessage` history the engine prompts with, and the prefix-cache
    /// eligibility decision carrying the conversation value. Built in a
    /// single walk so each image payload is base64-decoded exactly once —
    /// the cache shape wraps the same decoded bytes (CoW `Data`) the
    /// attachment carries — and so the two shapes can never disagree on the
    /// text they extract from the same message.
    struct NormalizedRequest {
        let systemPrompt: String?
        let messages: [LLMMessage]
        let prefixCacheEligibility: HTTPPrefixCacheEligibility
    }

    /// Leading system messages (before any user/assistant/tool turn) are
    /// concatenated into the system prompt; mid-conversation system messages
    /// are preserved in-place. The eligibility side records the FIRST
    /// incompatible message (video/audio, undecodable images, non-text
    /// content outside user messages) while the `LLMMessage` side keeps
    /// converting with its lenient drop semantics — an ineligible request
    /// still serves on the standard route.
    static func normalizeRequest(
        _ messages: [OpenAI.ChatMessage],
        tools: [OpenAI.ToolDefinition]? = nil,
        templateContextDigest: String = HTTPPrefixCacheConversation.defaultTemplateContextDigest
    ) -> NormalizedRequest {
        let reorderedMessages = reorderToolResultMessages(messages)
        var systemPrompt: String?
        var llmMessages: [LLMMessage] = []
        var cacheMessages: [HTTPPrefixCacheMessage] = []
        var firstIneligibility: HTTPPrefixCacheEligibility?
        var hasNonSystemMessage = false

        func markIneligible(_ reason: HTTPPrefixCacheEligibility) {
            if firstIneligibility == nil { firstIneligibility = reason }
        }

        for (index, message) in reorderedMessages.enumerated() {
            switch message.role {
            case .system:
                let (text, imageParts) = splitContent(message.content)
                if !imageParts.isEmpty {
                    markIneligible(.nonTextSystemMessage(index: index))
                }
                if !hasNonSystemMessage {
                    systemPrompt = systemPrompt.map { $0 + "\n\n" + text } ?? text
                } else {
                    llmMessages.append(.system(content: text))
                    cacheMessages.append(.init(role: .system, content: text))
                }

            case .user:
                hasNonSystemMessage = true
                let (text, imageParts) = splitContent(message.content)
                var attachments: [ImageAttachment] = []
                var cacheImages: [HTTPPrefixCacheImage] = []
                var cacheable = true
                for part in imageParts {
                    guard let attachment = convertImageContent(part) else {
                        // The engine path drops an unparseable part; the cache
                        // must not key a payload it cannot attribute.
                        cacheable = false
                        continue
                    }
                    attachments.append(attachment)
                    if attachment.ciImage != nil {
                        cacheImages.append(HTTPPrefixCacheImage(data: attachment.data))
                    } else {
                        cacheable = false
                    }
                }
                llmMessages.append(.user(content: text, images: attachments))
                if cacheable {
                    cacheMessages.append(.init(role: .user, content: text, images: cacheImages))
                } else {
                    markIneligible(.nonTextUserMessage(index: index))
                }

            case .assistant:
                hasNonSystemMessage = true
                let (text, imageParts) = splitContent(message.content)
                if !imageParts.isEmpty {
                    markIneligible(.nonTextAssistantMessage(index: index))
                }
                let infos = convertAssistantToolCalls(message.tool_calls)
                llmMessages.append(
                    .assistant(
                        content: text,
                        reasoning: message.resolvedReasoningContent,
                        toolCalls: infos?.isEmpty == false ? infos : nil
                    ))
                cacheMessages.append(
                    .assistant(
                        content: text,
                        reasoning: message.resolvedReasoningContent,
                        toolCalls: (infos ?? []).map {
                            HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                        }
                    ))

            case .tool:
                hasNonSystemMessage = true
                let (text, imageParts) = splitContent(message.content)
                if !imageParts.isEmpty {
                    markIneligible(.nonTextToolMessage(index: index))
                }
                llmMessages.append(
                    .toolResult(
                        toolCallId: message.tool_call_id ?? "", content: text
                    ))
                cacheMessages.append(.init(role: .tool, content: text))
            }
        }

        let eligibility =
            firstIneligibility
            ?? .eligible(
                HTTPPrefixCacheConversation(
                    systemPrompt: systemPrompt,
                    messages: cacheMessages,
                    toolDefinitionsDigest: toolDefinitionDigest(tools),
                    templateContextDigest: templateContextDigest
                ))
        return NormalizedRequest(
            systemPrompt: systemPrompt,
            messages: reorderToolResults(llmMessages),
            prefixCacheEligibility: eligibility
        )
    }

    /// Extract system prompt and convert OpenAI messages to internal `LLMMessage` values.
    static func convertMessages(_ messages: [OpenAI.ChatMessage]) -> (
        systemPrompt: String?, messages: [LLMMessage]
    ) {
        let normalized = normalizeRequest(messages)
        return (normalized.systemPrompt, normalized.messages)
    }

    /// Normalize an OpenAI-style request into the canonical conversation shape
    /// used by the HTTP prefix cache.
    ///
    /// Returns `nil` for requests with content the shape cannot carry (video/
    /// audio, undecodable images, non-text content outside user messages).
    static func normalizeConversation(
        _ messages: [OpenAI.ChatMessage],
        tools: [OpenAI.ToolDefinition]? = nil
    ) -> HTTPPrefixCacheConversation? {
        analyzePrefixCacheEligibility(messages, tools: tools).conversation
    }

    /// Returns the HTTP prefix-cache eligibility decision with the first incompatible
    /// content reason preserved for debug logging.
    static func analyzePrefixCacheEligibility(
        _ messages: [OpenAI.ChatMessage],
        tools: [OpenAI.ToolDefinition]? = nil,
        templateContextDigest: String = HTTPPrefixCacheConversation.defaultTemplateContextDigest
    ) -> HTTPPrefixCacheEligibility {
        normalizeRequest(
            messages, tools: tools, templateContextDigest: templateContextDigest
        ).prefixCacheEligibility
    }

    // MARK: - Tool Result Reordering

    /// Reorder tool result messages to match the order of `tool_calls` in the
    /// preceding assistant message.
    ///
    /// Qwen3.5's chat template matches tool results to calls positionally (the
    /// `Chat.Message.tool` type has no `tool_call_id` field). If a client sends
    /// results out of order, the model would see result B attached to call A.
    /// This method uses `tool_call_id` to align results before the IDs are lost
    /// during conversion to `Chat.Message`.
    static func reorderToolResults(_ messages: [LLMMessage]) -> [LLMMessage] {
        var result = messages
        var i = 0
        while i < result.count {
            // Find an assistant message with tool calls
            guard case .assistant(_, _, let toolCalls) = result[i],
                let toolCalls, !toolCalls.isEmpty
            else {
                i += 1
                continue
            }

            // Collect the contiguous tool result messages that follow
            let resultStart = i + 1
            var resultEnd = resultStart
            while resultEnd < result.count,
                case .toolResult = result[resultEnd]
            {
                resultEnd += 1
            }

            let toolResultSlice = Array(result[resultStart..<resultEnd])
            guard toolResultSlice.count > 1 else {
                // 0 or 1 results — nothing to reorder
                i = resultEnd
                continue
            }

            // Build a lookup from tool_call_id → tool result message
            var resultsByID: [String: LLMMessage] = [:]
            for msg in toolResultSlice {
                if case .toolResult(let id, _) = msg, !id.isEmpty {
                    resultsByID[id] = msg
                }
            }

            // Reorder to match the tool_calls order
            var reordered: [LLMMessage] = []
            for call in toolCalls {
                if let matched = resultsByID.removeValue(forKey: call.id) {
                    reordered.append(matched)
                }
            }
            // Append any results with unknown/missing IDs in their original order
            for msg in toolResultSlice {
                if case .toolResult(let id, _) = msg {
                    if resultsByID.removeValue(forKey: id) != nil || id.isEmpty {
                        reordered.append(msg)
                    }
                }
            }

            result.replaceSubrange(resultStart..<resultEnd, with: reordered)
            i = resultStart + reordered.count
        }
        return result
    }

    // MARK: - Tool Definitions

    /// Convert OpenAI tool definitions to raw `ToolSpec` dictionaries for prompt rendering.
    ///
    /// These are schema-only — no `execute` closures. The HTTP server passes them to
    /// `AgentEngine.generate(toolSpecs:)` so the chat template includes tool descriptions,
    /// but the server never executes them.
    static func convertToolDefinitions(_ tools: [OpenAI.ToolDefinition]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }

        return tools.map { tool in
            var functionDict: [String: any Sendable] = [
                "name": tool.function.name
            ]
            if let description = tool.function.description {
                functionDict["description"] = description
            }
            if let parameters = tool.function.parameters {
                functionDict["parameters"] = parameters.toSendable()
            }
            return [
                "type": tool.type,
                "function": functionDict,
            ] as [String: any Sendable]
        }
    }

    // MARK: - Image Content

    /// Decode a `data:` URI image content part into an `ImageAttachment`.
    ///
    /// Expects format: `data:<mimeType>;base64,<base64data>`
    static func convertImageContent(_ part: OpenAI.ContentPart) -> ImageAttachment? {
        guard part.type == .image_url,
            let urlString = part.image_url?.url,
            urlString.hasPrefix("data:")
        else {
            return nil
        }

        // Parse: data:<mimeType>;base64,<data>
        let afterData = urlString.dropFirst(5)  // drop "data:"
        guard let semicolonIndex = afterData.firstIndex(of: ";") else { return nil }

        let mimeType = String(afterData[afterData.startIndex..<semicolonIndex])
        let afterSemicolon = afterData[afterData.index(after: semicolonIndex)...]

        guard afterSemicolon.hasPrefix("base64,") else { return nil }
        let base64String = String(afterSemicolon.dropFirst(7))  // drop "base64,"

        guard let data = Data(base64Encoded: base64String) else { return nil }

        return ImageAttachment(data: data, mimeType: mimeType)
    }

    // MARK: - Private

    /// Split message content into joined text (nil-text parts skipped — the
    /// `textValue` rule, shared by every consumer so the engine prompt and
    /// the cache conversation can never disagree on the same message) and the
    /// raw image parts for the caller's per-policy image handling.
    private static func splitContent(
        _ content: OpenAI.MessageContent?
    ) -> (text: String, imageParts: [OpenAI.ContentPart]) {
        switch content {
        case nil:
            return ("", [])
        case .text(let string):
            return (string, [])
        case .parts(let parts):
            var texts: [String] = []
            var imageParts: [OpenAI.ContentPart] = []
            for part in parts {
                switch part.type {
                case .text:
                    if let text = part.text { texts.append(text) }
                case .image_url:
                    imageParts.append(part)
                }
            }
            return (texts.joined(separator: "\n"), imageParts)
        }
    }

    private static func convertAssistantToolCalls(
        _ toolCalls: [OpenAI.ToolCall]?
    ) -> [ToolCallInfo]? {
        toolCalls?.compactMap { call -> ToolCallInfo? in
            guard let name = call.function?.name else { return nil }
            return ToolCallInfo(
                id: call.id ?? UUID().uuidString,
                name: name,
                argumentsJSON: canonicalizeHTTPPrefixCacheToolArgumentsJSON(
                    call.function?.arguments ?? "{}"
                )
            )
        }
    }

    private static func reorderToolResultMessages(_ messages: [OpenAI.ChatMessage]) -> [OpenAI
        .ChatMessage]
    {
        var result = messages
        var index = 0

        while index < result.count {
            guard result[index].role == .assistant,
                let toolCalls = result[index].tool_calls,
                !toolCalls.isEmpty
            else {
                index += 1
                continue
            }

            let resultStart = index + 1
            var resultEnd = resultStart
            while resultEnd < result.count, result[resultEnd].role == .tool {
                resultEnd += 1
            }

            let toolResults = Array(result[resultStart..<resultEnd])
            guard toolResults.count > 1 else {
                index = resultEnd
                continue
            }

            var resultsByID: [String: OpenAI.ChatMessage] = [:]
            for message in toolResults {
                if let toolCallID = message.tool_call_id, !toolCallID.isEmpty {
                    resultsByID[toolCallID] = message
                }
            }

            var reordered: [OpenAI.ChatMessage] = []
            for toolCall in toolCalls {
                if let toolCallID = toolCall.id,
                    let matched = resultsByID.removeValue(forKey: toolCallID)
                {
                    reordered.append(matched)
                }
            }

            for message in toolResults {
                if let toolCallID = message.tool_call_id {
                    if resultsByID.removeValue(forKey: toolCallID) != nil {
                        reordered.append(message)
                    }
                } else {
                    reordered.append(message)
                }
            }

            result.replaceSubrange(resultStart..<resultEnd, with: reordered)
            index = resultStart + reordered.count
        }

        return result
    }

    private static func toolDefinitionDigest(_ tools: [OpenAI.ToolDefinition]?) -> String {
        guard let tools, !tools.isEmpty else {
            return HTTPPrefixCacheConversation.emptyToolDefinitionsDigest
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(tools) else {
            return HTTPPrefixCacheConversation.emptyToolDefinitionsDigest
        }
        return httpPrefixCacheDigest(for: data)
    }
}

// MARK: - AnyCodableValue → Sendable

extension AnyCodableValue {
    /// Convert to a type-erased `Sendable` value for `ToolSpec` dictionaries.
    func toSendable() -> any Sendable {
        switch self {
        case .null: return NSNull()
        case .bool(let v): return v
        case .int(let v): return v
        case .double(let v): return v
        case .string(let v): return v
        case .array(let v): return v.map { $0.toSendable() } as [any Sendable]
        case .object(let v): return v.mapValues { $0.toSendable() } as [String: any Sendable]
        }
    }
}
