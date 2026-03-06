import Foundation

// MARK: - PersonalAssistantExtension

/// Built-in extension for the personal assistant package.
///
/// For MVP, the personal assistant is purely skill-driven — it relies on the
/// file tools (read, write, edit, ls) guided by skill instructions. No custom
/// tools are needed yet.
///
/// Future post-MVP additions:
/// - `create_reminder` tool using `UNUserNotificationCenter`
/// - `capture_current_app_context` tool using Accessibility APIs
/// - Custom memory indexing tools
final class PersonalAssistantExtension: AgentExtension, @unchecked Sendable {
    let path = "personal-assistant"
    let commands: [String: RegisteredCommand] = [:]
    let tools: [String: AgentToolDefinition] = [:]

    // TODO: Add .beforeAgentStart handler to inject memories into context
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
