import Foundation
import MLXLMCommon

// MARK: - defaultConvertToLlm

/// Convert protocol messages to the low-level LLM representation, dropping any
/// that return `nil` from `toLLMMessage()` (e.g. UI-only or custom messages).
func defaultConvertToLlm(_ messages: [any AgentMessageProtocol]) -> [LLMMessage] {
    messages.compactMap { $0.toLLMMessage() }
}

// MARK: - toLLMCommonMessages

/// Bridge `LLMMessage` to `MLXLMCommon.Chat.Message` for the inference pipeline.
///
/// Since Tesseract uses local models with XML-based `<tool_call>` tags, assistant
/// messages reconstruct historical tool calls inline in the content string using
/// the model's expected wire format.
func toLLMCommonMessages(_ messages: [LLMMessage]) -> [Chat.Message] {
    messages.map { message in
        switch message {
        case .system(let content):
            .system(content)
        case .user(let content):
            .user(content)
        case .assistant(let content, let toolCalls):
            .assistant(reconstructAssistantContent(content, toolCalls: toolCalls))
        case .toolResult(_, let content):
            .tool(content)
        }
    }
}

// MARK: - Tool Call Reconstruction

/// Append `<tool_call>` XML tags to assistant content so the model sees what it called.
///
/// Produces Qwen 3.5's XML function format inside `<tool_call>` boundaries:
/// ```
/// <tool_call>
/// <function=tool_name>
/// <parameter=key>
/// value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
private func reconstructAssistantContent(
    _ content: String, toolCalls: [ToolCallInfo]?
) -> String {
    guard let toolCalls, !toolCalls.isEmpty else { return content }
    var result = content
    for call in toolCalls {
        result += "\n<tool_call>\n<function=\(call.name)>\n"

        if let arguments = ToolArgumentNormalizer.decode(call.argumentsJSON) {
            for key in arguments.keys.sorted() {
                guard let value = arguments[key] else { continue }
                result += "<parameter=\(key)>\n"
                result += formatToolCallParameterValue(value)
                result += "\n</parameter>\n"
            }
        }

        result += "</function>\n</tool_call>"
    }
    return result
}

private func formatToolCallParameterValue(_ value: JSONValue) -> String {
    switch value {
    case .string(let string):
        return string
    case .int(let int):
        return String(int)
    case .double(let double):
        return String(double)
    case .bool(let bool):
        return bool ? "True" : "False"
    case .null:
        return "None"
    case .array, .object:
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }
}

// MARK: - Message Factory Helpers

extension UserMessage {
    static func create(_ content: String) -> UserMessage {
        UserMessage(content: content)
    }
}

extension AssistantMessage {
    static func create(
        content: String,
        thinking: String? = nil,
        toolCalls: [ToolCallInfo] = []
    ) -> AssistantMessage {
        AssistantMessage(content: content, thinking: thinking, toolCalls: toolCalls)
    }

    static func fromStream(
        content: String,
        thinking: String? = nil,
        toolCalls: [ToolCallInfo] = []
    ) -> AssistantMessage {
        AssistantMessage(content: content, thinking: thinking, toolCalls: toolCalls)
    }
}

extension ToolResultMessage {
    static func create(
        toolCallId: String,
        toolName: String,
        result: AgentToolResult,
        isError: Bool
    ) -> ToolResultMessage {
        ToolResultMessage(
            toolCallId: toolCallId,
            toolName: toolName,
            content: result.content,
            isError: isError
        )
    }

    static func skipped(
        toolCallId: String,
        toolName: String,
        reason: String
    ) -> ToolResultMessage {
        ToolResultMessage(
            toolCallId: toolCallId,
            toolName: toolName,
            content: [.text(reason)],
            isError: true
        )
    }
}
