import Foundation

// MARK: - WebToolsExtension

/// Extension providing web search (and later web fetch) tools.
/// Registered in PackageBootstrap when web access is available.
final class WebToolsExtension: AgentExtension, @unchecked Sendable {
    let path = "web-tools"
    let commands: [String: RegisteredCommand] = [:]
    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [:]

    let tools: [String: AgentToolDefinition]

    init() {
        let searchTool = createWebSearchTool()
        tools = [
            searchTool.name: searchTool,
        ]
    }
}
