import Foundation

// MARK: - WebToolsExtension

/// Extension providing web search and web fetch tools.
/// Registered in PackageBootstrap when web access is enabled.
final class WebToolsExtension: AgentExtension, @unchecked Sendable {
    let path = "web-tools"
    let commands: [String: RegisteredCommand] = [:]
    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [:]

    let tools: [String: AgentToolDefinition]

    init() {
        let searchTool = createWebSearchTool()
        let fetchTool = createWebFetchTool()
        tools = [
            searchTool.name: searchTool,
            fetchTool.name: fetchTool,
        ]
    }
}
