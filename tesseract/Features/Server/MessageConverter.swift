//
//  MessageConverter.swift
//  tesseract
//

import Foundation
import MLXLMCommon
import Tokenizers

/// Converts between OpenAI-compatible API types and internal `LLMMessage` types.
enum MessageConverter {

    // MARK: - Messages

    /// Extract system prompt and convert OpenAI messages to internal `LLMMessage` values.
    ///
    /// Leading system messages (before any user/assistant/tool turn) are concatenated into
    /// the system prompt. System messages that appear mid-conversation are preserved in-place
    /// as `.system` entries in the returned messages array.
    static func convertMessages(_ messages: [OpenAI.ChatMessage]) -> (systemPrompt: String?, messages: [LLMMessage]) {
        var systemPrompt: String?
        var result: [LLMMessage] = []
        var hasNonSystemMessage = false

        for message in messages {
            switch message.role {
            case .system:
                let text = message.content?.textValue ?? ""
                if !hasNonSystemMessage {
                    // Leading system messages are concatenated into the system prompt
                    if let existing = systemPrompt {
                        systemPrompt = existing + "\n\n" + text
                    } else {
                        systemPrompt = text
                    }
                } else {
                    // Mid-conversation system messages stay in-place
                    result.append(.system(content: text))
                }

            case .user:
                hasNonSystemMessage = true
                let (text, images) = extractUserContent(message.content)
                result.append(.user(content: text, images: images))

            case .assistant:
                hasNonSystemMessage = true
                let content = message.content?.textValue ?? ""
                let infos = message.tool_calls?.compactMap { call -> ToolCallInfo? in
                    guard let name = call.function?.name else { return nil }
                    return ToolCallInfo(
                        id: call.id ?? UUID().uuidString,
                        name: name,
                        argumentsJSON: call.function?.arguments ?? "{}"
                    )
                }
                let toolCalls = infos?.isEmpty == false ? infos : nil
                result.append(.assistant(content: content, toolCalls: toolCalls))

            case .tool:
                hasNonSystemMessage = true
                let content = message.content?.textValue ?? ""
                let toolCallId = message.tool_call_id ?? ""
                result.append(.toolResult(toolCallId: toolCallId, content: content))
            }
        }

        return (systemPrompt, reorderToolResults(result))
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
            guard case .assistant(_, let toolCalls) = result[i],
                  let toolCalls, !toolCalls.isEmpty else {
                i += 1
                continue
            }

            // Collect the contiguous tool result messages that follow
            let resultStart = i + 1
            var resultEnd = resultStart
            while resultEnd < result.count,
                  case .toolResult = result[resultEnd] {
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
                "name": tool.function.name,
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
              urlString.hasPrefix("data:") else {
            return nil
        }

        // Parse: data:<mimeType>;base64,<data>
        let afterData = urlString.dropFirst(5) // drop "data:"
        guard let semicolonIndex = afterData.firstIndex(of: ";") else { return nil }

        let mimeType = String(afterData[afterData.startIndex..<semicolonIndex])
        let afterSemicolon = afterData[afterData.index(after: semicolonIndex)...]

        guard afterSemicolon.hasPrefix("base64,") else { return nil }
        let base64String = String(afterSemicolon.dropFirst(7)) // drop "base64,"

        guard let data = Data(base64Encoded: base64String) else { return nil }

        return ImageAttachment(data: data, mimeType: mimeType)
    }

    // MARK: - Private

    /// Extract text and images from user message content.
    private static func extractUserContent(_ content: OpenAI.MessageContent?) -> (text: String, images: [ImageAttachment]) {
        guard let content else { return ("", []) }

        switch content {
        case .text(let string):
            return (string, [])

        case .parts(let parts):
            var texts: [String] = []
            var images: [ImageAttachment] = []

            for part in parts {
                switch part.type {
                case .text:
                    if let text = part.text {
                        texts.append(text)
                    }
                case .image_url:
                    if let attachment = convertImageContent(part) {
                        images.append(attachment)
                    }
                }
            }

            return (texts.joined(separator: "\n"), images)
        }
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
