import Foundation
import MLXLMCommon

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

    /// Append a built-in tool after initialization (e.g., skill tool that
    /// depends on discovered data not available at init time). Replaces an
    /// existing tool of the same name — `AgentFactory.makeAgent` runs once per
    /// agent over the shared registry, and the second bootstrap (the
    /// Companion's headless agent) must not duplicate `use_skill`.
    func appendBuiltInTool(_ tool: AgentToolDefinition) {
        if let index = builtInTools.firstIndex(where: { $0.name == tool.name }) {
            builtInTools[index] = tool
        } else {
            builtInTools.append(tool)
        }
    }

    /// Refresh extension tools (e.g., after extension registration changes).
    func refreshExtensionTools(from host: ExtensionHost) {
        self.extensionTools = host.aggregatedTools()
    }
}
