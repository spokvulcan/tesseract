import Foundation
import MLXLMCommon
import Tokenizers

/// Converts ``AgentChatMessage`` arrays into ``UserInput`` for the MLX pipeline.
///
/// Stateless — all methods are static. The MLXLMCommon Jinja template pipeline
/// handles the actual ChatML `<|im_start|>role\ncontent<|im_end|>` formatting;
/// this enum just bridges our domain types into ``Chat.Message``.
enum AgentChatFormatter {

    /// Maps agent messages to a ``UserInput`` ready for ``ModelContainer/prepare(input:)``.
    ///
    /// - Parameters:
    ///   - messages: Conversation history in chronological order.
    ///   - tools: Optional tool schemas — the Jinja template renders these into
    ///     the system prompt's `<tools>` block automatically.
    static func makeUserInput(
        from messages: [AgentChatMessage],
        tools: [ToolSpec]? = nil
    ) -> UserInput {
        let chatMessages = messages.map { message -> Chat.Message in
            switch message.role {
            case .system: .system(message.content)
            case .user: .user(message.content)
            case .assistant:
                if message.toolCalls.isEmpty {
                    .assistant(message.content)
                } else {
                    // Reconstruct inline <tool_call> tags so the model sees what it called
                    .assistant(reconstructContent(message))
                }
            case .tool: .tool(message.content)
            }
        }
        return UserInput(chat: chatMessages, tools: tools)
    }

    /// Appends `<tool_call>` tags to assistant content for model context.
    private static func reconstructContent(_ message: AgentChatMessage) -> String {
        var content = message.content
        for call in message.toolCalls {
            if let data = try? JSONEncoder().encode(call.function),
               let json = String(data: data, encoding: .utf8)
            {
                content += "\n<tool_call>\n\(json)\n</tool_call>"
            }
        }
        return content
    }
}
