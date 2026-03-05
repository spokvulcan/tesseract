# Epic 4: Extension System

> Implement the extension protocol, host, and event dispatch system. Extensions are compiled Swift modules registered at app startup. All tasks are additive.

## Prerequisites

- Epic 0 (Foundation Types) — `AgentToolDefinition`, `AgentToolResult`, `AgentEvent`

## Directory

All new files go in `Features/Agent/Extensions/`.

---

## Task 4.1: Define AgentExtension Protocol

**File**: `Features/Agent/Extensions/AgentExtension.swift` (~80 lines)

**What to build**:

1. **Extension protocol**:
   ```swift
   protocol AgentExtension: AnyObject, Sendable {
       /// Unique identifier path (e.g., "personal-assistant", "coding-tools")
       var path: String { get }

       /// Custom tools this extension provides
       var tools: [String: AgentToolDefinition] { get }

       /// Event handlers keyed by event name
       var handlers: [ExtensionEventType: [ExtensionEventHandler]] { get }

       /// Slash commands (future)
       var commands: [String: RegisteredCommand] { get }
   }
   ```

2. **Extension event types** (MVP subset from spec E.2):
   ```swift
   enum ExtensionEventType: String, Sendable, Hashable {
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
   ```

3. **Event handler type** (closure-based for simplicity):
   ```swift
   struct ExtensionEventHandler: Sendable {
       let handle: @Sendable (ExtensionEventPayload, ExtensionContext) async throws -> ExtensionEventResult?
   }
   ```

4. **Event payloads** (union enum):
   ```swift
   enum ExtensionEventPayload: Sendable {
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
   ```

5. **Event results** (what extensions can return):
   ```swift
   enum ExtensionEventResult: Sendable {
       case none                                          // No effect
       case block(reason: String)                         // Block tool_call
       case modifyToolResult(AgentToolResult)              // Override tool_result
       case modifyContext([any AgentMessageProtocol & Sendable])  // Override context
       case modifyInput(String)                            // Transform input
       case handled                                        // Input fully handled
       case cancelCompaction                                // Cancel compaction
       case beforeAgentOverride(systemPrompt: String?, extraMessages: [any AgentMessageProtocol & Sendable]?)
       case discoveredResources(skillPaths: [URL], promptPaths: [URL], contextPaths: [URL])
   }
   ```

6. **Registered command** (for slash commands, future):
   ```swift
   struct RegisteredCommand: Sendable {
       let name: String
       let description: String
       let execute: @Sendable (String, ExtensionContext) async throws -> Void
   }
   ```

**Acceptance criteria**:
- Protocol compiles with Sendable constraints
- All event types and payloads defined
- Build succeeds

**Spec reference**: Sections E.1, E.2

---

## Task 4.2: Define ExtensionContext Protocol

**File**: `Features/Agent/Extensions/ExtensionContext.swift` (~40 lines)

**What to build**:

The context object passed to extension handlers when they fire:

```swift
protocol ExtensionContext: Sendable {
    /// Working directory / agent root
    var cwd: String { get }

    /// Current model (if loaded)
    var model: AgentModelRef? { get }

    /// Is the agent currently idle?
    func isIdle() -> Bool

    /// Abort current generation
    func abort()

    /// Get current system prompt
    func getSystemPrompt() -> String

    /// Get context usage stats
    func getContextUsage() -> ContextUsage?

    /// Trigger compaction
    func compact(options: CompactOptions?)
}

struct ContextUsage: Sendable {
    let estimatedTokens: Int
    let contextWindow: Int
    let percentUsed: Double
}

struct CompactOptions: Sendable {
    let force: Bool  // Compact even if under threshold
}
```

A concrete implementation will be created during integration (Epic 6). For now, this just defines the protocol.

**Acceptance criteria**:
- Protocol compiles
- Build succeeds

**Spec reference**: Section E.3

---

## Task 4.3: Implement ExtensionHost

**File**: `Features/Agent/Extensions/ExtensionHost.swift` (~120 lines)

**What to build**:

Central registry for extensions. Manages lifecycle and aggregates tools.

```swift
@MainActor
final class ExtensionHost {
    private var extensions: [any AgentExtension] = []

    /// Register an extension (called at app startup)
    func register(_ extension: any AgentExtension)

    /// Unregister an extension
    func unregister(path: String)

    /// Get all tools from all extensions (first registration wins on name conflicts)
    func aggregatedTools() -> [AgentToolDefinition]

    /// Get all extensions
    var registeredExtensions: [any AgentExtension] { get }

    /// Get a specific extension by path
    func getExtension(path: String) -> (any AgentExtension)?
}
```

**Tool aggregation rules**:
- Collect tools from all extensions in registration order
- First registration wins on name conflicts (log warning for duplicates)
- Built-in tools (from `BuiltInToolFactory`) are registered separately and always take precedence

**Extension registration order**:
- Extensions are registered in the order they are loaded (package order → explicit registrations)
- Order matters for event dispatch (first registered = first to handle)

**Acceptance criteria**:
- Register multiple extensions, tools aggregated correctly
- Duplicate tool names: first wins, warning logged
- Unregister removes extension and its tools
- Build succeeds

**Spec reference**: Section E.5 (partially)

---

## Task 4.4: Implement ExtensionRunner (Event Dispatch)

**File**: `Features/Agent/Extensions/ExtensionRunner.swift` (~150 lines)

**What to build**:

Central event dispatcher that fires events through all registered extensions.

```swift
@MainActor
final class ExtensionRunner {
    private let host: ExtensionHost
    private let contextFactory: () -> ExtensionContext

    init(host: ExtensionHost, contextFactory: @escaping () -> ExtensionContext)

    /// Fire an event through all extensions, collecting results
    func fire(
        event: ExtensionEventType,
        payload: ExtensionEventPayload
    ) async -> [ExtensionEventResult]

    /// Fire a tool_call event (before execution) — returns block result if any extension blocks
    func fireToolCall(
        toolCallId: String,
        toolName: String,
        argsJSON: String
    ) async -> ExtensionEventResult?

    /// Fire a tool_result event (after execution) — returns modified result if any extension overrides
    func fireToolResult(
        toolCallId: String,
        toolName: String,
        result: AgentToolResult
    ) async -> AgentToolResult
}
```

**Dispatch rules** (from spec E.5):
1. Iterate extensions in registration order
2. Create `ExtensionContext` for each event
3. Call each extension's handler for the event type
4. **Error handling**: Catch and log extension errors — never crash the agent
5. **Result aggregation**:
   - For `tool_call`: first `.block` result wins (tool is blocked)
   - For `tool_result`: last `.modifyToolResult` wins (last extension gets final say)
   - For `context`: apply modifications in sequence
   - For `resources_discover`: collect all discovered paths

**Tool wrapping integration** (from spec E.4):
The `ExtensionRunner` provides two hook points that the `AgentLoop` calls:
1. Before tool execution: `fireToolCall()` — if result is `.block`, skip the tool
2. After tool execution: `fireToolResult()` — modified result is used

These hooks are wired during Epic 6 integration.

**Acceptance criteria**:
- Events dispatched to all extensions in order
- Extension errors caught, logged, and don't crash
- `tool_call` blocking works (first block wins)
- `tool_result` modification works (last modify wins)
- Build succeeds

**Spec reference**: Sections E.4, E.5

---

## Task 4.5: Create a NoOp Test Extension

**File**: `Features/Agent/Extensions/TestExtension.swift` (~40 lines)

**What to build**:

A minimal no-op extension that validates the extension system works:

```swift
#if DEBUG
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
```

This validates:
- Extension protocol conformance compiles
- Handler closures compile with correct types
- Registration in `ExtensionHost` works
- Event dispatch reaches the extension

**Acceptance criteria**:
- Compiles in Debug mode
- Can be registered in ExtensionHost
- Event dispatch calls its handler
- Build succeeds

---

## Summary

After this epic, the project has a complete extension system. Extensions can register tools, handle events, and wrap tool calls. Nothing is wired to the live agent yet.

**New files created**: 5 (AgentExtension.swift, ExtensionContext.swift, ExtensionHost.swift, ExtensionRunner.swift, TestExtension.swift)
**Lines added**: ~430
**Existing files modified**: 0
