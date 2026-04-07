import Foundation
import MLXLMCommon

/// Converts between MLXLMCommon tool call types and OpenAI-compatible API types.
///
/// `ToolCallParser` produces `ToolCall` objects without IDs (Qwen3.5 XML format has no ID concept).
/// This converter assigns server-generated `call_<UUID>` IDs and JSON-stringifies arguments
/// for the OpenAI wire format.
enum ToolCallConverter {

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

    /// Remaps tool call IDs in a message array using the provided mapping.
    ///
    /// Handles both `tool_call_id` on tool-result messages and `id` fields within
    /// `tool_calls` arrays on assistant messages. IDs not present in the map are
    /// left unchanged.
    static func mapToolCallIDs(
        _ messages: [OpenAI.ChatMessage],
        idMap: [String: String]
    ) -> [OpenAI.ChatMessage] {
        guard !idMap.isEmpty else { return messages }

        return messages.map { message in
            let toolCallIdReplacement = message.tool_call_id.flatMap { idMap[$0] }
            let hasToolCallsRemap = message.tool_calls?.contains { $0.id.flatMap({ idMap[$0] }) != nil } ?? false

            guard toolCallIdReplacement != nil || hasToolCallsRemap else { return message }

            var mapped = message
            if let replacement = toolCallIdReplacement {
                mapped.tool_call_id = replacement
            }
            if hasToolCallsRemap {
                mapped.tool_calls = message.tool_calls!.map { call in
                    guard let id = call.id, let replacement = idMap[id] else { return call }
                    var remapped = call
                    remapped.id = replacement
                    return remapped
                }
            }
            return mapped
        }
    }
}
