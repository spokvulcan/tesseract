import Foundation

/// Convenience factory that creates all built-in tools.
nonisolated enum BuiltInToolFactory: Sendable {
    /// Creates all built-in tools including scheduling tools.
    static func createAll(sandbox: PathSandbox, schedulingService: SchedulingService) -> [AgentToolDefinition] {
        var tools = createFileTools(sandbox: sandbox)
        tools.append(createCronCreateTool(schedulingService: schedulingService))
        tools.append(createCronListTool(schedulingService: schedulingService))
        tools.append(createCronDeleteTool(schedulingService: schedulingService))
        return tools
    }

    /// Creates only file-system tools (read, write, edit, ls). Used by benchmarks.
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
