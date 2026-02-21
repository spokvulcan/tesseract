import Foundation
import MLXLMCommon

/// A "final answer" tool that forces the model to deliver its response via a tool call.
///
/// Like smolagents' `final_answer` pattern: the model must always emit `<tool_call>` tags,
/// either for data tools or for `respond`. This eliminates the "answer from memory" failure
/// mode where the model produces text without calling any tools.
///
/// In ``AgentRunner``, `respond` is intercepted before normal tool execution —
/// its `text` argument becomes the assistant's visible response, and the loop ends.
struct RespondTool: AgentTool {
    let name = "respond"
    let description = "Send your final response to the user. Call this after completing all tool calls, or when no tools are needed."
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The message to send to the user"),
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        // Should never be called — AgentRunner intercepts this tool.
        arguments.string(for: "text") ?? ""
    }
}
