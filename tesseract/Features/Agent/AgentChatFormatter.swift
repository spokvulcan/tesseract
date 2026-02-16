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
            case .assistant: .assistant(message.content)
            case .tool: .tool(message.content)
            }
        }
        return UserInput(chat: chatMessages, tools: tools)
    }
}
