import Foundation
import MLXLMCommon

/// Converts between MLXLMCommon tool call types and OpenAI-compatible API types.
///
/// `ToolCallParser` produces `ToolCall` objects without IDs (Qwen3.5 XML format has no ID concept).
/// This converter assigns server-generated `call_<UUID>` IDs and JSON-stringifies arguments
/// for the OpenAI wire format.
nonisolated enum ToolCallConverter {

    /// Converts internal parsed tool calls into OpenAI-compatible format.
    ///
    /// Each tool call receives a unique `call_<UUID>` identifier and its arguments
    /// are serialized to a JSON string.
    static func convertToOpenAI(_ toolCalls: [ToolCall]) -> [OpenAI.ToolCall] {
        toolCalls.enumerated().map { index, toolCall in
            OpenAI.ToolCall(
                id: "call_\(UUID().uuidString)",
                type: "function",
                function: OpenAI.FunctionCall(
                    name: toolCall.function.name,
                    arguments: ToolArgumentNormalizer.encode(toolCall.function.arguments)
                ),
                index: index
            )
        }
    }

}
