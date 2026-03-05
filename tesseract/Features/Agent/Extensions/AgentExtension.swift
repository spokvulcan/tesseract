import Foundation

// MARK: - AgentExtension

/// Protocol for compiled Swift extensions registered at app startup.
/// Extensions provide custom tools, event handlers, and slash commands.
protocol AgentExtension: AnyObject, Sendable {
    /// Unique identifier path (e.g., "personal-assistant", "coding-tools").
    var path: String { get }

    /// Custom tools this extension provides.
    var tools: [String: AgentToolDefinition] { get }

    /// Event handlers keyed by event type.
    var handlers: [ExtensionEventType: [ExtensionEventHandler]] { get }

    /// Slash commands (future).
    var commands: [String: RegisteredCommand] { get }
}

// MARK: - ExtensionEventType

/// Events that extensions can subscribe to.
nonisolated enum ExtensionEventType: String, Sendable, Hashable {
    case sessionStart = "session_start"
    case sessionShutdown = "session_shutdown"
    case beforeAgentStart = "before_agent_start"
    case turnStart = "turn_start"
    case turnEnd = "turn_end"
    case toolCall = "tool_call"
    case toolResult = "tool_result"
    case context = "context"
    case input = "input"
    case resourcesDiscover = "resources_discover"
    case sessionBeforeCompact = "session_before_compact"
}

// MARK: - ExtensionEventHandler

/// Closure-based handler invoked when an extension event fires.
nonisolated struct ExtensionEventHandler: Sendable {
    let handle: @Sendable (ExtensionEventPayload, any ExtensionContext) async throws -> ExtensionEventResult?
}

// MARK: - ExtensionEventPayload

/// Payload delivered to extension handlers for each event type.
nonisolated enum ExtensionEventPayload: Sendable {
    case sessionStart
    case sessionShutdown
    case beforeAgentStart(systemPrompt: String, messages: [any AgentMessageProtocol & Sendable])
    case turnStart
    case turnEnd(assistantMessage: AssistantMessage, toolResults: [ToolResultMessage])
    case toolCall(toolCallId: String, toolName: String, argsJSON: String)
    case toolResult(toolCallId: String, toolName: String, result: AgentToolResult)
    case context(messages: [any AgentMessageProtocol & Sendable])
    case input(text: String)
    case resourcesDiscover
    case sessionBeforeCompact
}

// MARK: - ExtensionEventResult

/// What an extension handler can return to modify agent behavior.
nonisolated enum ExtensionEventResult: Sendable {
    /// No effect.
    case none

    /// Block a tool_call from executing.
    case block(reason: String)

    /// Override a tool_result with a modified result.
    case modifyToolResult(AgentToolResult)

    /// Override context messages.
    case modifyContext([any AgentMessageProtocol & Sendable])

    /// Transform user input text.
    case modifyInput(String)

    /// Input was fully handled by the extension.
    case handled

    /// Cancel pending compaction.
    case cancelCompaction

    /// Override system prompt and/or inject extra messages before agent start.
    case beforeAgentOverride(systemPrompt: String?, extraMessages: [any AgentMessageProtocol & Sendable]?)

    /// Report discovered resource paths from the extension's package.
    case discoveredResources(skillPaths: [URL], promptPaths: [URL], contextPaths: [URL])
}

// MARK: - RegisteredCommand

/// A slash command registered by an extension (future use).
nonisolated struct RegisteredCommand: Sendable {
    let name: String
    let description: String
    let execute: @Sendable (String, any ExtensionContext) async throws -> Void
}
