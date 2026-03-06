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
/// Since Tesseract uses local models with XML-based `<tool_call>` tags (not native
/// JSON tool calling), assistant messages reconstruct tool calls as inline XML in the
/// content string.
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
/// Produces the same format the `ToolCallParser` expects:
/// ```
/// <tool_call>
/// {"name":"tool_name","arguments":{...}}
/// </tool_call>
/// ```
private func reconstructAssistantContent(
    _ content: String, toolCalls: [ToolCallInfo]?
) -> String {
    guard let toolCalls, !toolCalls.isEmpty else { return content }
    var result = content
    for call in toolCalls {
        // Build the {"name":"...","arguments":{...}} JSON that the parser expects
        let argsFragment: String
        if call.argumentsJSON.isEmpty {
            argsFragment = "{}"
        } else if let normalized = ToolArgumentNormalizer.decode(call.argumentsJSON) {
            argsFragment = ToolArgumentNormalizer.encode(normalized)
        } else {
            argsFragment = call.argumentsJSON
        }
        result += "\n<tool_call>\n{\"name\":\"\(call.name)\",\"arguments\":\(argsFragment)}\n</tool_call>"
    }
    return result
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
