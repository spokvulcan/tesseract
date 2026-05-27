import Foundation

/// Convenience factory that creates all built-in tools.
nonisolated enum BuiltInToolFactory: Sendable {
    /// Creates all built-in tools (read, write, edit, ls).
    static func createAll(sandbox: PathSandbox) -> [AgentToolDefinition] {
        createFileTools(sandbox: sandbox)
    }

    private static func createFileTools(sandbox: PathSandbox) -> [AgentToolDefinition] {
        let readTracker = FileReadTracker()
        return [
            createReadTool(sandbox: sandbox, readTracker: readTracker),
            createWriteTool(sandbox: sandbox, readTracker: readTracker),
            createEditTool(sandbox: sandbox, readTracker: readTracker),
            createLsTool(sandbox: sandbox),
        ]
    }
}
