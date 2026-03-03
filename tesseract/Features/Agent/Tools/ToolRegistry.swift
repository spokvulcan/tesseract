import Foundation
import MLXLMCommon
import os
import Tokenizers

// MARK: - ToolRegistry (New — AgentToolDefinition-based)

/// Aggregates built-in tools and extension-contributed tools.
/// Built-in tools take precedence on name conflicts.
@MainActor
final class ToolRegistry {
    private var builtInTools: [AgentToolDefinition]
    private var extensionTools: [AgentToolDefinition]

    init(sandbox: PathSandbox, extensionHost: ExtensionHost) {
        self.builtInTools = BuiltInToolFactory.createAll(sandbox: sandbox)
        self.extensionTools = extensionHost.aggregatedTools()
    }

    /// All available tools (built-in first, then extension).
    var allTools: [AgentToolDefinition] {
        builtInTools + extensionTools
    }

    /// Lookup by name (built-in takes precedence).
    func tool(named name: String) -> AgentToolDefinition? {
        builtInTools.first(where: { $0.name == name })
            ?? extensionTools.first(where: { $0.name == name })
    }

    /// Tool specs for LLM (OpenAI function-calling format).
    var toolSpecs: [[String: any Sendable]] {
        allTools.map { $0.toolSpec }
    }

    /// Refresh extension tools (e.g., after extension registration changes).
    func refreshExtensionTools(from host: ExtensionHost) {
        self.extensionTools = host.aggregatedTools()
    }
}

// MARK: - LegacyToolRegistry (Old — AgentTool protocol-based)

/// Legacy registry used by AgentRunner and BenchmarkRunner.
/// Will be removed when those consumers are rewritten.
struct LegacyToolRegistry: Sendable {
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
