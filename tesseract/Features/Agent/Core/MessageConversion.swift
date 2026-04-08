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
        case .user(let content, let images):
            .user(
                content,
                images: images.compactMap { attachment in
                    attachment.ciImage.map { .ciImage($0) }
                }
            )
        case .assistant(let content, let reasoning, let toolCalls):
            .assistant(reconstructAssistantPromptContent(
                content,
                reasoning: reasoning,
                toolCalls: toolCalls?.map {
                    HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                } ?? []
            ))
        case .toolResult(_, let content):
            .tool(content)
        }
    }
}

// MARK: - Message Factory Helpers

extension UserMessage {
    static func create(_ content: String, images: [ImageAttachment] = []) -> UserMessage {
        UserMessage(content: content, images: images)
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
