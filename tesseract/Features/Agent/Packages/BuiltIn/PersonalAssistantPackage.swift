import Foundation

// MARK: - PersonalAssistantExtension

/// Built-in extension for the personal assistant package. The package is
/// prompt- and skill-driven; its tools are built in (the agenda tools, over
/// Reminders and Calendar), so the extension registers none of its own.
final class PersonalAssistantExtension: AgentExtension, @unchecked Sendable {
    let path = "personal-assistant"
    let commands: [String: RegisteredCommand] = [:]
    let tools: [String: AgentToolDefinition] = [:]

    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [
        .sessionStart: [
            ExtensionEventHandler { _, context in
                let cwd = await context.cwd
                Log.agent.info("[PersonalAssistant] Session started, cwd: \(cwd)")
                return nil
            }
        ]
    ]
}
