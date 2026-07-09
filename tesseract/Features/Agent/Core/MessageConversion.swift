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
///
/// `visionActive` says whether the loaded container can ingest images (the VLM
/// variant). Tool-result images (browser screenshots) attach to their tool
/// message when it can — the Qwen3.5 template renders image pads for tool-role
/// content, and `UserInput(chat:)` collects images from every role — and
/// degrade to an explicit text note when it can't, so a text-only model is
/// told the pixels are absent instead of being invited to hallucinate them.
func toLLMCommonMessages(_ messages: [LLMMessage], visionActive: Bool = false) -> [Chat.Message] {
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
            .assistant(
                reconstructAssistantPromptContent(
                    content,
                    reasoning: reasoning,
                    toolCalls: toolCalls?.map {
                        HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                    } ?? []
                ))
        case .toolResult(_, let content, let images):
            if images.isEmpty {
                .tool(content)
            } else if visionActive {
                Chat.Message(
                    role: .tool,
                    content: content,
                    images: images.compactMap { attachment in
                        attachment.ciImage.map { .ciImage($0) }
                    }
                )
            } else {
                .tool(
                    content
                        + "\n[This tool call also returned \(images.count) image(s), "
                        + "but the current model session is text-only and cannot see them.]"
                )
            }
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
            isError: isError,
            details: result.details
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
