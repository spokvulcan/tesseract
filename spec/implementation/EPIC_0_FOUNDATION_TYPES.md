# Epic 0: Foundation Types

> Define the new type system that everything else builds on. All tasks are additive — no existing code is modified. The app builds and runs unchanged after this epic.

## Prerequisites

None. This is the first epic.

## Directory

All new files go in `Features/Agent/Core/`.

---

## Task 0.1: Create AgentMessage Protocol Hierarchy

**File**: `Features/Agent/Core/AgentMessage.swift` (~120 lines)

**What to build**:

1. `AgentMessageProtocol` — base protocol for all messages in the context array
   ```swift
   protocol AgentMessageProtocol: Sendable {
       func toLLMMessage() -> LLMMessage?
   }
   ```

2. `LLMMessage` — the type sent to the LLM (maps to MLXLMCommon's chat format)
   ```swift
   enum LLMMessage: Sendable {
       case system(content: String)
       case user(content: String)
       case assistant(content: String, toolCalls: [ToolCallInfo]?)
       case toolResult(toolCallId: String, content: String)
   }
   ```

3. `CoreMessage` — closed enum for the three message types the loop pattern-matches on
   ```swift
   enum CoreMessage: AgentMessageProtocol, Sendable {
       case user(UserMessage)
       case assistant(AssistantMessage)
       case toolResult(ToolResultMessage)
   }
   ```

4. `UserMessage` struct — `id: UUID`, `content: String`, `timestamp: Date`

5. `AssistantMessage` struct — `id: UUID`, `content: String`, `thinking: String?`, `toolCalls: [ToolCallInfo]`, `timestamp: Date`

6. `ToolResultMessage` struct — `id: UUID`, `toolCallId: String`, `toolName: String`, `content: [ContentBlock]`, `isError: Bool`, `timestamp: Date`

7. `ToolCallInfo` struct — `id: String`, `name: String`, `argumentsJSON: String`

8. `CustomAgentMessage` protocol (open, extensions add conformers)
   ```swift
   protocol CustomAgentMessage: AgentMessageProtocol {
       var customType: String { get }
   }
   ```

9. Type alias: `typealias AgentMessage = any AgentMessageProtocol`

**Acceptance criteria**:
- All types compile with `Sendable` conformance
- `CoreMessage.toLLMMessage()` returns correct `LLMMessage` for each case
- Build succeeds — no existing code references these types yet

**Spec reference**: Section B.1

---

## Task 0.2: Create ContentBlock and AgentToolResult

**File**: `Features/Agent/Core/AgentToolResult.swift` (~50 lines)

**What to build**:

1. `ContentBlock` enum
   ```swift
   enum ContentBlock: Sendable, Hashable {
       case text(String)
       case image(data: Data, mimeType: String)
   }
   ```

2. `AgentToolResult` struct
   ```swift
   struct AgentToolResult: Sendable {
       let content: [ContentBlock]
       let details: (any Sendable & Hashable)?

       // Convenience initializers
       static func text(_ string: String) -> AgentToolResult
       static func error(_ message: String) -> AgentToolResult
   }
   ```

3. `ToolProgressCallback` typealias
   ```swift
   typealias ToolProgressCallback = @Sendable (AgentToolResult) -> Void
   ```

**Acceptance criteria**:
- `AgentToolResult.text("hello")` creates a single text content block
- `AgentToolResult.error("fail")` creates a result (same structure, just a semantic convenience)
- Build succeeds

**Spec reference**: Section B.2

---

## Task 0.3: Create New AgentTool Definition (Concrete Struct)

**File**: `Features/Agent/Core/AgentToolDefinition.swift` (~80 lines)

**What to build**:

1. `AgentToolDefinition` — concrete struct with closures (NOT a protocol)
   ```swift
   struct AgentToolDefinition: Sendable {
       let name: String
       let label: String
       let description: String
       let parameterSchema: JSONSchema

       let execute: @Sendable (
           _ toolCallId: String,
           _ argsJSON: [String: JSONValue],
           _ signal: CancellationToken?,
           _ onUpdate: ToolProgressCallback?
       ) async throws -> AgentToolResult
   }
   ```

   Note: Named `AgentToolDefinition` (not `AgentTool`) to avoid collision with the existing `AgentTool` protocol. During Epic 6 integration, the old protocol is removed and this could be renamed if desired.

2. `JSONSchema` — simple JSON Schema representation for parameter validation
   ```swift
   struct JSONSchema: Sendable {
       let type: String  // "object"
       let properties: [String: PropertySchema]
       let required: [String]
   }

   struct PropertySchema: Sendable {
       let type: String  // "string", "integer", "boolean", "array"
       let description: String
       let enumValues: [String]?  // for enum types
       let items: PropertySchema?  // for array types
   }
   ```

3. `CancellationToken` — wraps Swift's `Task.isCancelled` + cooperative cancellation
   ```swift
   final class CancellationToken: Sendable {
       private let _isCancelled = ManagedAtomic<Bool>(false)

       var isCancelled: Bool { _isCancelled.load(ordering: .relaxed) }
       func cancel() { _isCancelled.store(true, ordering: .relaxed) }
   }
   ```

   Note: If `import Atomics` is not available, use `OSAllocatedUnfairLock<Bool>` or `NSLock` instead. Or simply use `actor`-based isolation. The implementation detail is flexible — the contract is `isCancelled: Bool` + `cancel()`.

4. Helper: `AgentToolDefinition.toolSpec` computed property that generates OpenAI-compatible function spec (same format as existing `AgentTool.toolSpec`, but using the new `JSONSchema` type).

**Acceptance criteria**:
- `[AgentToolDefinition]` is a valid Swift array type (no `any` needed)
- `AgentToolDefinition.toolSpec` generates JSON matching current tool spec format
- Build succeeds

**Spec reference**: Section B.2

---

## Task 0.4: Create AgentEvent Enum

**File**: `Features/Agent/Core/AgentEvent.swift` (~70 lines)

**What to build**:

1. `AgentEvent` — all events the loop can emit
   ```swift
   enum AgentEvent: Sendable {
       // Agent lifecycle
       case agentStart
       case agentEnd(messages: [any AgentMessageProtocol & Sendable])

       // Turn lifecycle
       case turnStart
       case turnEnd(message: AssistantMessage, toolResults: [ToolResultMessage])

       // Context transform lifecycle
       case contextTransformStart(reason: ContextTransformReason)
       case contextTransformEnd(reason: ContextTransformReason, didMutate: Bool)

       // Message lifecycle
       case messageStart(message: any AgentMessageProtocol & Sendable)
       case messageUpdate(message: AssistantMessage, streamDelta: AssistantStreamDelta)
       case messageEnd(message: any AgentMessageProtocol & Sendable)

       // Tool execution
       case toolExecutionStart(toolCallId: String, toolName: String, argsJSON: String)
       case toolExecutionUpdate(toolCallId: String, toolName: String, result: AgentToolResult)
       case toolExecutionEnd(toolCallId: String, toolName: String, result: AgentToolResult, isError: Bool)
   }
   ```

2. `AssistantStreamDelta`
   ```swift
   struct AssistantStreamDelta: Sendable {
       let textDelta: String?
       let thinkingDelta: String?
       let toolCallDelta: ToolCallDelta?
   }
   ```

3. `ToolCallDelta`
   ```swift
   struct ToolCallDelta: Sendable {
       let toolCallId: String
       let name: String?
       let argumentsDelta: String?
   }
   ```

4. `ContextTransformResult` and `ContextTransformReason`
   ```swift
   struct ContextTransformResult: Sendable {
       let messages: [any AgentMessageProtocol & Sendable]
       let didMutate: Bool
       let reason: ContextTransformReason
   }

   enum ContextTransformReason: Sendable {
       case compaction
       case extensionTransform(String)  // extension path identifier
   }
   ```

**Acceptance criteria**:
- All event cases compile with Sendable payloads
- `ContextTransformReason` is used by both event cases and the config closure
- No `Any` types anywhere
- Build succeeds

**Spec reference**: Section D.1

---

## Task 0.5: Create AgentLoopConfig

**File**: `Features/Agent/Core/AgentLoopConfig.swift` (~40 lines)

**What to build**:

```swift
struct AgentLoopConfig: Sendable {
    /// Model to use for generation
    let model: AgentModelRef

    /// REQUIRED: Converts [AgentMessage] → [LLMMessage] at the call boundary
    let convertToLlm: @Sendable ([any AgentMessageProtocol]) -> [LLMMessage]

    /// OPTIONAL: Transform context before convertToLlm
    /// Used for compaction, context pruning, injecting external context.
    /// Returns ContextTransformResult — if didMutate is true, the loop writes
    /// the result back into the authoritative context (not a temporary copy).
    /// The `reason` is declared upfront so the loop can emit contextTransformStart
    /// before awaiting the transform.
    let contextTransform: ContextTransformConfig?

    /// OPTIONAL: Returns steering messages (user interrupts mid-run)
    let getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])?

    /// OPTIONAL: Returns follow-up messages (queued for after agent finishes)
    let getFollowUpMessages: (@Sendable () async -> [any AgentMessageProtocol])?
}
```

`ContextTransformConfig` pairs a reason label with the transform closure:

```swift
struct ContextTransformConfig: Sendable {
    let reason: ContextTransformReason
    let transform: @Sendable ([any AgentMessageProtocol], CancellationToken?) async -> ContextTransformResult
}
```

This lets the loop emit `contextTransformStart(reason:)` *before* awaiting the transform, without depending on the result to know the reason. For compaction, the config is created as:
```swift
contextTransform: ContextTransformConfig(
    reason: .compaction,
    transform: makeTransformContext(contextManager: contextManager, contextWindow: 120_000)
)
```

Note: `AgentModelRef` is a placeholder type that wraps the model identifier. During Epic 2, this will be connected to `AgentEngine`/`LLMActor`. For now, it can be a simple struct:

```swift
struct AgentModelRef: Sendable {
    let id: String
}
```

**Acceptance criteria**:
- Config struct compiles with all closure types correctly typed
- Build succeeds

**Spec reference**: Section B.4

---

## Task 0.6: Create AgentState

**File**: `Features/Agent/Core/AgentState.swift` (~50 lines)

**What to build**:

```swift
/// Lightweight run phase — drives UI indicators for what the agent is doing right now
enum AgentPhase: Sendable, Equatable {
    case idle
    case transformingContext(ContextTransformReason)
    case streaming
    case executingTool(String)  // tool name
}

@MainActor
@Observable
final class AgentState {
    var systemPrompt: String = ""
    var model: AgentModelRef?
    var tools: [AgentToolDefinition] = []
    var messages: [any AgentMessageProtocol] = []
    var phase: AgentPhase = .idle
    var streamMessage: AssistantMessage?
    var pendingToolCalls: Set<String> = []
    var error: String?

    /// Convenience — true when not idle
    var isStreaming: Bool { phase != .idle }
}
```

Use `@Observable` (Observation framework, macOS 14+) instead of `@Published`/`ObservableObject` — this is the modern pattern and Tesseract targets macOS 26+.

`AgentPhase` replaces the simple `isStreaming: Bool`. The coordinator maps phase values to UI indicators (e.g., `.transformingContext(.compaction)` → "Compacting memories...").

**Acceptance criteria**:
- `AgentState` properties are observable from SwiftUI
- `AgentPhase` has cases for idle, transforming, streaming, executing tool
- `isStreaming` is a derived convenience (not stored)
- Build succeeds

**Spec reference**: Section B.5

---

## Task 0.7: Create AnyCodableValue Utility

**File**: `Features/Agent/Core/AnyCodableValue.swift` (~100 lines)

**What to build**:

A type-safe JSON value wrapper for the tagged message encoding system (needed by MessageCodecRegistry in Epic 2).

```swift
enum AnyCodableValue: Codable, Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableValue])
    case object([String: AnyCodableValue])
}
```

With:
- `Codable` conformance that encodes/decodes native JSON types
- `init(_ value: Any)` factory for converting from `JSONSerialization` output
- Convenience accessors: `stringValue`, `intValue`, `boolValue`, etc.
- `Hashable` conformance (needed by `AgentToolResult.details`)

**Acceptance criteria**:
- Round-trip: encode → decode produces identical values
- Handles nested objects and arrays
- Build succeeds

**Spec reference**: Section H.4

---

## Summary

After this epic, the project has 7 new files in `Features/Agent/Core/` defining the complete type system. No existing code is modified. The app builds and runs identically to before.

**New files created**: 7
**Lines added**: ~510
**Existing files modified**: 0
