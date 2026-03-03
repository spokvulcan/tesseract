import Foundation

#if DEBUG

// MARK: - TestExtension

/// Minimal no-op extension that validates the extension system compiles and dispatches events.
final class TestExtension: AgentExtension, @unchecked Sendable {
    let path = "test"
    let tools: [String: AgentToolDefinition] = [:]
    let commands: [String: RegisteredCommand] = [:]

    var handlers: [ExtensionEventType: [ExtensionEventHandler]] {
        [
            .sessionStart: [
                ExtensionEventHandler { _, _ in
                    Log.agent.debug("[TestExtension] Session started")
                    return nil
                }
            ]
        ]
    }
}

#endif
