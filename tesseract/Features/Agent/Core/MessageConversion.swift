import Foundation
import MLX
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
///
/// `audioActive` is the audio sibling (**Audio-capable** loaded container):
/// user-message takes attach as 16 kHz sample arrays when it can hear, and
/// degrade to a text note when it can't — a spoken turn replayed against a
/// deaf model must say the audio is absent, not render an empty user turn.
func toLLMCommonMessages(
    _ messages: [LLMMessage], visionActive: Bool = false, audioActive: Bool = false
) -> [Chat.Message] {
    messages.map { message in
        switch message {
        case .system(let content):
            .system(content)
        case .user(let content, let images, let audios):
            if audioActive || audios.isEmpty {
                .user(
                    content,
                    images: images.compactMap { attachment in
                        attachment.ciImage.map { .ciImage($0) }
                    },
                    audios: audioActive
                        ? audios.compactMap { attachment in
                            attachment.samples.map { .array(MLXArray($0)) }
                        }
                        : []
                )
            } else {
                .user(
                    content
                        + "\n[This turn was spoken: \(audios.count) audio clip(s) attached, "
                        + "but the current model session cannot hear them.]",
                    images: images.compactMap { attachment in
                        attachment.ciImage.map { .ciImage($0) }
                    }
                )
            }
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
