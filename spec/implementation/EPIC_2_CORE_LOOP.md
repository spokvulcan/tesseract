# Epic 2: Core Loop

> Implement the Pi-style double-loop, the stateful Agent wrapper, message conversion, and message persistence. This is the heart of the rewrite. All code is additive — the old `AgentRunner` remains functional.

## Prerequisites

- Epic 0 (Foundation Types) — all types used heavily

## Directory

All new files go in `Features/Agent/Core/`.

---

## Task 2.1: Implement convertToLlm and Message Construction Utilities

**File**: `Features/Agent/Core/MessageConversion.swift` (~100 lines)

**What to build**:

1. **Default `convertToLlm` function**:
   ```swift
   func defaultConvertToLlm(_ messages: [any AgentMessageProtocol]) -> [LLMMessage] {
       messages.compactMap { $0.toLLMMessage() }
   }
   ```

2. **Bridge to MLXLMCommon chat format** (replaces current `AgentChatFormatter.swift`):
   ```swift
   func toLLMCommonMessages(_ messages: [LLMMessage]) -> [MLXLMCommon.Chat.Message]
   ```

   This converts our `LLMMessage` to the format that `AgentEngine`/`LLMActor` expects. Mapping:
   - `.system(content)` → `Chat.Message(role: "system", content: content)`
   - `.user(content)` → `Chat.Message(role: "user", content: content)`
   - `.assistant(content, toolCalls)` → `Chat.Message(role: "assistant", content: content)` with tool call tags reconstructed in content
   - `.toolResult(toolCallId, content)` → `Chat.Message(role: "tool", content: content)`

   Note: Since Tesseract uses local models with XML-based `<tool_call>` tags (not native JSON tool calling), assistant messages reconstruct tool calls as inline XML in the content string. This matches the current `AgentRunner.reconstructAssistantMessage()` behavior.

3. **Message factory helpers**:
   ```swift
   extension UserMessage {
       static func create(_ content: String) -> UserMessage
   }
   extension AssistantMessage {
       static func create(content: String, thinking: String?, toolCalls: [ToolCallInfo]) -> AssistantMessage
       static func fromStream(content: String, thinking: String?, toolCalls: [ToolCallInfo]) -> AssistantMessage
   }
   extension ToolResultMessage {
       static func create(toolCallId: String, toolName: String, result: AgentToolResult, isError: Bool) -> ToolResultMessage
       static func skipped(toolCallId: String, toolName: String, reason: String) -> ToolResultMessage
   }
   ```

**Acceptance criteria**:
- `defaultConvertToLlm` filters out custom messages that return `nil` from `toLLMMessage()`
- `toLLMCommonMessages` produces correct `Chat.Message` format
- Tool calls are reconstructed as `<tool_call>\n{JSON}\n</tool_call>` in assistant content
- Build succeeds

**Spec reference**: Section H.2

---

## Task 2.2: Implement the Core Agent Loop (agentLoop + runLoop)

**File**: `Features/Agent/Core/AgentLoop.swift` (~300 lines)

**What to build**:

The Pi double-loop from spec section C.1. This is the most critical implementation.

```swift
/// Entry point: starts a new agent loop with the given prompts
func agentLoop(
    prompts: [any AgentMessageProtocol],
    context: inout AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken?,
    emit: @Sendable (AgentEvent) -> Void
) async

/// Continue from existing context without new prompts (retry/resume)
func agentLoopContinue(
    context: inout AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken?,
    emit: @Sendable (AgentEvent) -> Void
) async
```

**`AgentContext`** (defined in `Features/Agent/Core/AgentContext.swift`, shared by the loop and the `Agent` class):
```swift
struct AgentContext: Sendable {
    var systemPrompt: String
    var messages: [any AgentMessageProtocol & Sendable]
    var tools: [AgentToolDefinition]?
}
```

**Double-Loop Implementation** (see spec C.1 pseudocode):

```
agentLoop(prompts, context, config, signal, emit):
    1. Push each prompt into context.messages
    2. Emit .agentStart
    3. For each prompt: emit .messageStart, .messageEnd
    4. Call runLoop(context, pendingMessages=[], config, signal, emit)

runLoop(context, pendingMessages, config, signal, emit):
    OUTER LOOP:
        INNER LOOP (while hasMoreToolCalls || pendingMessages.count > 0):
            1. Emit .turnStart (except first turn if no pending messages)

            2. If pendingMessages exist:
               - Push each into context.messages
               - Emit .messageStart/.messageEnd for each
               - Clear pendingMessages

            3. streamAssistantResponse():
               a. If config.contextTransform exists (let ct = config.contextTransform):
                  i.   Emit .contextTransformStart(reason: ct.reason)
                  ii.  Let result = await ct.transform(context.messages, signal)
                  iii. If result.didMutate:
                       context.messages = result.messages   // AUTHORITATIVE WRITE-BACK
                  iv.  Emit .contextTransformEnd(reason: ct.reason, didMutate: result.didMutate)
               b. Let llmMessages = config.convertToLlm(context.messages)
               c. Call LLM generation with streaming
               d. Emit .messageStart (assistant)
               e. For each chunk: emit .messageUpdate with AssistantStreamDelta
               f. Emit .messageEnd (final assistant message)
               g. Push assistant message into context.messages
               h. Return (assistantMessage, stopReason)

               **Critical rule**: Compaction is not ephemeral. When transformContext
               compacts, it replaces context.messages in place, inserting
               CompactionSummaryMessage and removing superseded history from the
               authoritative context. The old messages are gone — not kept in a
               shadow copy.

            4. If error or aborted: emit .turnEnd, .agentEnd, return

            5. Extract tool calls from assistant response

            6. If tool calls exist:
               executeToolCalls():
               a. For EACH tool call (SEQUENTIAL, not parallel):
                  i.   Find tool by name in context.tools
                  ii.  Validate args against schema
                  iii. Emit .toolExecutionStart(toolCallId, name, argsJSON)
                  iv.  Execute tool(toolCallId, args, signal, onUpdate)
                       - onUpdate emits .toolExecutionUpdate
                  v.   Emit .toolExecutionEnd(toolCallId, name, result, isError)
                  vi.  Create ToolResultMessage
                  vii. Push into context.messages
                  viii. Emit .messageStart/.messageEnd for tool result
                  ix.  Check config.getSteeringMessages()
                       - If steering arrived: skip remaining tools
                       - Skipped tools get "Skipped due to queued user message" error
               b. Return (toolResults, steeringMessages)

            7. Emit .turnEnd

            8. Set pendingMessages = steeringMessages (if any)
            9. Also poll config.getSteeringMessages() for new steering

        END INNER LOOP (no tool calls, no steering)

        10. Check config.getFollowUpMessages()
            - If follow-ups exist: set pendingMessages = followUps, continue outer loop

        11. No follow-ups: break outer loop

    END OUTER LOOP
    12. Emit .agentEnd(allNewMessages)
```

**Key rules (from spec C.2)**:
- No round cap
- Tools execute sequentially
- Steering skips remaining tools
- Follow-ups extend the session
- `contextTransform` runs before every LLM call
- Cancellation propagates via CancellationToken
- **Compaction is not ephemeral**: When `transformContext` returns `didMutate: true`, the loop writes the result back into `context.messages` permanently, replacing compacted history with the `CompactionSummaryMessage`

**LLM generation interface**: The loop needs a way to call the LLM. Define a protocol/closure:

```swift
typealias LLMGenerateFunction = @Sendable (
    String,          // system prompt
    [LLMMessage],    // messages
    [AgentToolDefinition]?,  // tools
    CancellationToken?
) -> AsyncThrowingStream<AgentGeneration, Error>
```

This will be connected to `AgentEngine.generate()` during Epic 6 integration. For now, store it in `AgentLoopConfig` or as a parameter.

**Acceptance criteria**:
- Double-loop structure matches spec exactly
- Steering interrupts skip remaining tools
- Follow-ups re-enter the outer loop
- No round cap
- Tools execute sequentially
- `contextTransform` called before every generation
- Cancellation propagates correctly
- Build succeeds (loop is standalone, not yet wired to existing engine)

**Spec reference**: Sections C.1, C.2, C.3

---

## Task 2.3: Implement Tool Execution Within the Loop

**Part of**: `Features/Agent/Core/AgentLoop.swift` (or separate helper file)

This task details the `executeToolCalls` function referenced in Task 2.2.

**What to build**:

```swift
private func executeToolCalls(
    toolCalls: [ToolCallInfo],
    tools: [AgentToolDefinition],
    signal: CancellationToken?,
    getSteeringMessages: (() async -> [any AgentMessageProtocol])?,
    emit: @Sendable (AgentEvent) -> Void
) async -> (results: [ToolResultMessage], steering: [any AgentMessageProtocol])
```

**Implementation**:
1. For each tool call (sequential):
   a. Look up tool by name in tools array
   b. If not found: create error ToolResultMessage `"Unknown tool: {name}"`
   c. Parse arguments from JSON string to `[String: JSONValue]`
   d. Validate against tool's `parameterSchema` (check required params present)
   e. Emit `.toolExecutionStart`
   f. Create `onUpdate` callback that emits `.toolExecutionUpdate`
   g. Call `tool.execute(toolCallId, args, signal, onUpdate)`
   h. Catch errors → create error ToolResultMessage
   i. Emit `.toolExecutionEnd`
   j. Emit `.messageStart`/`.messageEnd` for tool result
   k. **Check steering**: call `getSteeringMessages?.()`
      - If messages returned: skip remaining tools
      - For each skipped tool: create "Skipped due to queued user message" result

2. Return all results + any steering messages

**Schema validation** (basic):
- Check all `required` parameters are present in args
- Type checking is best-effort (the model is usually right, and the tool's closure handles conversion)

**Acceptance criteria**:
- Unknown tools produce error results, not crashes
- Missing required params produce error results
- Steering interrupts skip remaining tools with clear messages
- onUpdate callback emits events during execution
- Errors in tool execution are caught and returned as error results
- Build succeeds

**Spec reference**: Section C.1 (executeToolCalls block)

---

## Task 2.4: Implement Agent Class (Stateful Wrapper)

**File**: `Features/Agent/Core/Agent.swift` (~200 lines)

**What to build**:

The `Agent` class wraps `agentLoop` with state management, queues, and abort control. This is analogous to Pi's `Agent` class from `agent.ts`.

```swift
@MainActor
@Observable
final class Agent {
    // Observable state (UI reads this)
    private(set) var state: AgentState

    // Authoritative context (loop reads/writes this)
    // Uses the same AgentContext struct defined in Task 2.2
    private var context: AgentContext

    // Subscriptions
    private var subscribers: [(AgentEvent) -> Void] = []

    // Message queues
    private var steeringQueue: [any AgentMessageProtocol] = []
    private var followUpQueue: [any AgentMessageProtocol] = []

    // Abort control
    private var cancellationToken: CancellationToken?
    private var runTask: Task<Void, Never>?

    // Configuration
    private let config: AgentLoopConfig
    private let generate: LLMGenerateFunction

    init(config: AgentLoopConfig, systemPrompt: String, tools: [AgentToolDefinition], generate: LLMGenerateFunction)
    // Initializes context = AgentContext(systemPrompt: systemPrompt, messages: [], tools: tools)

    // Public API
    func prompt(_ message: any AgentMessageProtocol) async throws
    func `continue`() async throws
    func abort()
    func waitForIdle() async
    func subscribe(_ handler: @escaping (AgentEvent) -> Void) -> () -> Void

    // Queue management
    func pushSteering(_ message: any AgentMessageProtocol)
    func pushFollowUp(_ message: any AgentMessageProtocol)
}
```

**Implementation**:

`prompt(_ message)`:
1. Ensure not already streaming (throw if `state.phase != .idle`)
2. Create new `CancellationToken`
3. Create `Task` that calls `agentLoop(prompts: [message], ...)`
4. Update `state.phase = .streaming`
5. Subscribe to events to update `state`

`continue()`:
1. Ensure not already streaming (`state.phase != .idle`)
2. Verify last message is not assistant (precondition from spec C.3)
3. Call `agentLoopContinue(context:...)`

`abort()`:
1. Signal cancellation token
2. Cancel the run task

`waitForIdle()`:
1. If not streaming, return immediately
2. Otherwise, await the run task

`subscribe(_ handler)`:
1. Add handler to subscribers
2. Return unsubscribe closure that removes it

**Queue → config wiring**:
- `config.getSteeringMessages` drains `steeringQueue` (returns all + clears)
- `config.getFollowUpMessages` drains `followUpQueue` (returns all + clears)

**State updates** (via event subscription inside Agent):

The loop operates on `self.context` (the stored `AgentContext`). `state` is the `@Observable` projection for the UI. Sync rules:

- `.agentStart` → `state.phase = .streaming`
- `.contextTransformStart(let reason)` → `state.phase = .transformingContext(reason)`
- `.contextTransformEnd(_, let didMutate)` → if didMutate: `state.messages = self.context.messages` (authoritative sync); `state.phase = .streaming`
- `.messageUpdate` → `state.streamMessage = updated`
- `.toolExecutionStart(let id, let name, _)` → `state.pendingToolCalls.insert(id)`; `state.phase = .executingTool(name)`
- `.toolExecutionEnd(let id, _, _, _)` → `state.pendingToolCalls.remove(id)`; if pendingToolCalls.isEmpty: `state.phase = .streaming`
- `.agentEnd` → `state.messages = self.context.messages` (final sync); `state.phase = .idle`

**Acceptance criteria**:
- `prompt` starts the loop and updates state
- `abort` cancels in-progress generation
- `pushSteering` delivers messages that interrupt tool execution
- `pushFollowUp` delivers messages that extend the session after completion
- `subscribe` receives all events in order
- `waitForIdle` resolves when agent finishes
- `state.messages` reflects authoritative context after any mutating transform and at agentEnd
- `state.phase` tracks the current agent phase (idle, transformingContext, streaming, executingTool)
- Build succeeds

**Spec reference**: Section D.3

---

## Task 2.5: Implement Message Persistence (Tagged Encoding)

**File**: `Features/Agent/Core/MessagePersistence.swift` (~200 lines)

**What to build**:

1. **`PersistableMessage` protocol**:
   ```swift
   protocol PersistableMessage: AgentMessageProtocol, Codable {
       static var persistenceTag: String { get }
   }
   ```

2. **Conform core types**:
   - `UserMessage: PersistableMessage` → tag `"user"`
   - `AssistantMessage: PersistableMessage` → tag `"assistant"`
   - `ToolResultMessage: PersistableMessage` → tag `"tool_result"`

3. **`TaggedMessage`** (Codable envelope):
   ```swift
   struct TaggedMessage: Codable {
       let type: String
       let payload: [String: AnyCodableValue]
   }
   ```

4. **`MessageCodecRegistry`** (actor):
   ```swift
   actor MessageCodecRegistry {
       static let shared = MessageCodecRegistry()

       func register<M: PersistableMessage>(_ type: M.Type)
       func encode(_ message: any AgentMessageProtocol) throws -> TaggedMessage
       func decode(_ tagged: TaggedMessage) throws -> any AgentMessageProtocol
   }
   ```

5. **`OpaqueMessage`** (fallback for unknown types):
   ```swift
   struct OpaqueMessage: AgentMessageProtocol, Sendable {
       let tag: String
       let rawPayload: [String: AnyCodableValue]
       func toLLMMessage() -> LLMMessage? { nil }
   }
   ```

6. **Registration at startup**:
   ```swift
   func registerCoreMessageCodecs() async {
       let registry = MessageCodecRegistry.shared
       await registry.register(UserMessage.self)
       await registry.register(AssistantMessage.self)
       await registry.register(ToolResultMessage.self)
   }
   ```

**Acceptance criteria**:
- Encode a `UserMessage` → decode back to `UserMessage` with identical values
- Encode an `AssistantMessage` with tool calls → round-trips correctly
- Unknown tags decode as `OpaqueMessage`
- Re-encoding an `OpaqueMessage` preserves original JSON
- Build succeeds

**Spec reference**: Section H.4

---

## Task 2.6: Create CompactionSummaryMessage (First Custom Message Type)

**File**: `Features/Agent/Core/CompactionSummaryMessage.swift` (~30 lines)

**What to build**:

The first `CustomAgentMessage` conformer, demonstrating the open extension point:

```swift
struct CompactionSummaryMessage: CustomAgentMessage, PersistableMessage, Sendable {
    static let persistenceTag = "compaction_summary"
    let customType = "compaction_summary"

    let summary: String
    let tokensBefore: Int
    let timestamp: Date

    func toLLMMessage() -> LLMMessage? {
        .user(content: "<summary>\n\(summary)\n</summary>")
    }
}
```

Register it alongside core types:
```swift
await registry.register(CompactionSummaryMessage.self)
```

**Acceptance criteria**:
- Persists/loads correctly via MessageCodecRegistry
- `toLLMMessage()` returns a user message with XML-wrapped summary
- Build succeeds

**Spec reference**: Section H.1

---

## Task 2.7: Adapt AgentEngine Interface for New Loop

**File**: Modify `Features/Agent/AgentEngine.swift` (~30 lines changed)

**What to add**:

Add a new `generate` overload that accepts `[LLMMessage]` (the new type) alongside the existing `generate` that accepts `[AgentChatMessage]`:

```swift
/// New: Generate from LLMMessage array (for new AgentLoop)
func generate(
    systemPrompt: String,
    messages: [LLMMessage],
    tools: [AgentToolDefinition]?,
    parameters: AgentGenerateParameters
) throws -> AsyncThrowingStream<AgentGeneration, Error>
```

This method:
1. Converts `[LLMMessage]` → `[Chat.Message]` via `toLLMCommonMessages()` (from Task 2.1)
2. Converts `[AgentToolDefinition]` → `[ToolSpec]` via each tool's `.toolSpec` property
3. Delegates to existing `LLMActor.generate()`

The existing `generate(messages: [AgentChatMessage], tools: [ToolSpec]?, ...)` remains unchanged for backward compatibility with the old `AgentRunner`.

**Acceptance criteria**:
- New overload compiles and is callable from the `AgentLoop`
- Old overload still works for existing `AgentRunner`
- Build succeeds, existing behavior unchanged

---

## Summary

After this epic, the project has the complete new core loop alongside the old one. The `AgentLoop`, `Agent`, message conversion, and persistence are all implemented but not yet wired to the UI.

**New files created**: 5 (AgentLoop.swift, Agent.swift, MessageConversion.swift, MessagePersistence.swift, CompactionSummaryMessage.swift)
**Lines added**: ~830
**Existing files modified**: 1 (AgentEngine.swift — new overload added)
