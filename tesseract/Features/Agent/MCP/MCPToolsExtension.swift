import Foundation
import os

// MARK: - MCPToolsExtension

/// The ``AgentExtension`` that surfaces every connected MCP server's tools to the
/// agent. Unlike a fixed-at-init extension, MCP tools materialize and vanish as
/// servers connect, disconnect, or change their list — so the tool set is a
/// snapshot the ``MCPClientManager`` pushes in via ``update(_:)``.
///
/// The snapshot lives behind a lock so `tools` is safe to read from any context
/// (the `ExtensionHost` aggregates on the MainActor; the lock keeps that honest
/// regardless of who calls).
final class MCPToolsExtension: AgentExtension, @unchecked Sendable {
    let path = "mcp-tools"
    let commands: [String: RegisteredCommand] = [:]
    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [:]

    private let snapshot = OSAllocatedUnfairLock<[String: AgentToolDefinition]>(initialState: [:])

    var tools: [String: AgentToolDefinition] { snapshot.withLock { $0 } }

    /// Replace the exposed tool set. First definition wins on a name collision
    /// (the same policy `ExtensionHost` applies across extensions).
    func update(_ definitions: [AgentToolDefinition]) {
        let resolved = Dictionary(
            definitions.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        snapshot.withLock { $0 = resolved }
    }
}
