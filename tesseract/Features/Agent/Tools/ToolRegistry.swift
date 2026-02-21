import MLXLMCommon
import Tokenizers

struct ToolRegistry: Sendable {
    private let tools: [String: any AgentTool]

    init(tools: [any AgentTool]) {
        var dict: [String: any AgentTool] = [:]
        for tool in tools {
            dict[tool.name] = tool
        }
        self.tools = dict
    }

    var toolNames: [String] {
        Array(tools.keys)
    }

    var toolSpecs: [ToolSpec] {
        tools.values.map(\.toolSpec)
    }

    func tool(named name: String) -> (any AgentTool)? {
        tools[name]
    }

    func hasNoRequiredParameters(_ name: String) -> Bool {
        guard let tool = tools[name] else { return false }
        return tool.parameters.allSatisfy { !$0.isRequired }
    }

    func execute(call: ToolCall) async throws -> String {
        guard let tool = tools[call.function.name] else {
            throw ToolRegistryError.unknownTool(call.function.name)
        }
        return try await tool.execute(arguments: call.function.arguments)
    }
}
