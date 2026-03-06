import Foundation

/// Convenience factory that creates all four built-in tools.
nonisolated enum BuiltInToolFactory: Sendable {
    static func createAll(sandbox: PathSandbox) -> [AgentToolDefinition] {
        [
            createReadTool(sandbox: sandbox),
            createWriteTool(sandbox: sandbox),
            createEditTool(sandbox: sandbox),
            createLsTool(sandbox: sandbox),
        ]
    }
}
