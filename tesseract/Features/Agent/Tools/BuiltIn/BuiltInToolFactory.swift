import Foundation

/// Convenience factory that creates all four built-in tools.
nonisolated enum BuiltInToolFactory: Sendable {
    static func createAll(sandbox: PathSandbox) -> [AgentToolDefinition] {
        let readTracker = FileReadTracker()
        return [
            createReadTool(sandbox: sandbox, readTracker: readTracker),
            createWriteTool(sandbox: sandbox, readTracker: readTracker),
            createEditTool(sandbox: sandbox, readTracker: readTracker),
            createLsTool(sandbox: sandbox),
        ]
    }
}
