# Epic 6: Session Integration

> This is the big switch. Wire the new core loop, tools, context, extensions, and packages together. Rewrite the coordinator and conversation store. Remove old domain tools. This epic transforms the running app from the old architecture to the new one.

## Prerequisites

- All previous epics (0-5) completed

## Strategy

This epic modifies existing files and deletes old ones. The changes are coordinated — partial completion may break the build. Plan to complete this epic in a single focused session.

**Order matters**: Tasks must be done sequentially (or in the specified groups).

---

## Task 6.1: Rewrite ToolRegistry for Built-In + Extension Tools

**File**: Modify `Features/Agent/Tools/ToolRegistry.swift`

**Current** (38 lines): Simple dictionary of `[any AgentTool]`

**New** (~80 lines): Aggregates built-in tools + extension tools using `AgentToolDefinition`

```swift
@MainActor
final class NewToolRegistry {
    private var builtInTools: [AgentToolDefinition] = []
    private var extensionTools: [AgentToolDefinition] = []

    init(sandbox: PathSandbox, extensionHost: ExtensionHost) {
        self.builtInTools = BuiltInToolFactory.createAll(sandbox: sandbox)
        self.extensionTools = extensionHost.aggregatedTools()
    }

    /// All available tools (built-in first, then extension)
    var allTools: [AgentToolDefinition] {
        builtInTools + extensionTools
    }

    /// Lookup by name (built-in takes precedence)
    func tool(named name: String) -> AgentToolDefinition? {
        builtInTools.first(where: { $0.name == name }) ??
        extensionTools.first(where: { $0.name == name })
    }

    /// Tool specs for LLM (OpenAI function-calling format)
    var toolSpecs: [ToolSpec] {
        allTools.map { $0.toolSpec }
    }

    /// Refresh extension tools (e.g., after extension registration changes)
    func refreshExtensionTools(from host: ExtensionHost) {
        self.extensionTools = host.aggregatedTools()
    }
}
```

**Migration note**: The old `ToolRegistry` (struct with `[any AgentTool]`) should be renamed or replaced. Since the old `AgentRunner` still references it during the transition, consider:
1. Rename old to `LegacyToolRegistry`
2. Create new `ToolRegistry` (or `NewToolRegistry`)
3. After all wiring is done, remove legacy

**Acceptance criteria**:
- Built-in tools (read, write, edit, list) registered
- Extension tools aggregated
- Tool lookup works (built-in precedence)
- Build succeeds

---

## Task 6.2: Rewrite AgentConversation + AgentConversationStore

**Files**: Modify `Features/Agent/AgentConversation.swift` AND `Features/Agent/AgentConversationStore.swift`

### 6.2a: Rewrite AgentConversation.swift

**Current** (38 lines): Uses `[AgentChatMessage]`, derives title from first message's `.role == .user` check

**New** (~50 lines): Uses `[any AgentMessageProtocol]`, derives title via protocol-aware logic

```swift
struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol]  // Was [AgentChatMessage]
    let createdAt: Date
    var updatedAt: Date

    /// Derive title from first user message content
    var title: String {
        for msg in messages {
            if let core = msg as? CoreMessage, case .user(let user) = core {
                let text = user.content.prefix(80)
                return text.isEmpty ? "New Conversation" : String(text)
            }
        }
        return "New Conversation"
    }

    var messageCount: Int { messages.count }
}

struct AgentConversationSummary: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int
}
```

**Why this must be a separate sub-task**: The current `title` computed property uses `AgentChatMessage.role`, which doesn't exist on `AgentMessageProtocol`. Without this fix, the build breaks as soon as the messages type changes.

### 6.2b: Rewrite AgentConversationStore.swift

**Current** (185 lines): Stores `[AgentChatMessage]` as JSON

**New** (~220 lines): Stores `[any AgentMessageProtocol]` using `MessageCodecRegistry`

**Changes**:

1. **Custom Codable for conversation file** (uses TaggedMessage):
   ```swift
   struct ConversationFile: Codable {
       let id: UUID
       let title: String
       let createdAt: Date
       let updatedAt: Date
       let messages: [TaggedMessage]
   }
   ```

2. **Save**:
   ```swift
   func save(_ conversation: AgentConversation) async {
       let taggedMessages = try await conversation.messages.asyncMap {
           try await MessageCodecRegistry.shared.encode($0)
       }
       let file = ConversationFile(
           id: conversation.id,
           title: conversation.title,
           createdAt: conversation.createdAt,
           updatedAt: conversation.updatedAt,
           messages: taggedMessages
       )
       // Write to disk as JSON
   }
   ```

3. **Load**:
   ```swift
   func load(id: UUID) async -> AgentConversation? {
       guard let file: ConversationFile = loadFromDisk(id) else { return nil }
       let messages = try await file.messages.asyncMap {
           try await MessageCodecRegistry.shared.decode($0)
       }
       return AgentConversation(id: file.id, messages: messages, ...)
   }
   ```

4. **No backward compatibility — clean break**: Old conversation files (using `AgentChatMessage` JSON format) are not migrated. The app is in active development with no external users. The store must handle the stale `index.json` at init time:

   ```swift
   func loadOrReset() {
       let indexURL = conversationsDir.appendingPathComponent("index.json")
       if FileManager.default.fileExists(atPath: indexURL.path) {
           // Try loading as new format. If it fails (old format), wipe and start clean.
           if let summaries = tryLoadNewFormatIndex(indexURL) {
               self.conversations = summaries
               return
           }
           // Old-format index — clear the entire conversations directory
           Log.agent.info("Clearing pre-redesign conversation data")
           try? FileManager.default.removeItem(at: conversationsDir)
           try? FileManager.default.createDirectory(at: conversationsDir, withIntermediateDirectories: true)
       }
       self.conversations = []
   }
   ```

   This prevents stale old-format summaries from appearing in the conversation list on the first launch after cutover. The wipe happens at store init, not deferred to a later epic.

**Acceptance criteria**:
- `AgentConversation` compiles with protocol-backed messages
- Title derivation works correctly (finds first `UserMessage` via pattern matching)
- New conversations save in tagged format
- Round-trip: save → load produces identical messages
- Unknown message types survive as `OpaqueMessage`
- Old-format `index.json` triggers a clean wipe of the conversations directory at init
- Build succeeds

**Spec reference**: Section H.4

---

## Task 6.3: Rewrite SystemPromptBuilder → SystemPromptAssembler Integration

**File**: Modify `Features/Agent/SystemPromptBuilder.swift`

**Strategy**: Replace the three-tier system with the Pi-style assembler. The old builder's content is partially preserved in the personal assistant package's `APPEND_SYSTEM.md`.

**Changes**:
1. Remove the three-tier prompt logic (minimal/condensed/default)
2. Replace `build(...)` with a call to `SystemPromptAssembler.assemble(...)`
3. The core prompt is now generic (from Task 3.3)
4. Assistant personality comes from package prompt appends
5. Memories are no longer injected into the system prompt — they're read via the `read` tool

**Backward compatibility for memories**: The old system injected memories directly into the prompt as "What I Know About You". The new system doesn't. Instead:
- The personal assistant's `APPEND_SYSTEM.md` instructs the model to read `memories.md`
- The `before_agent_start` extension event could inject a brief memory summary if desired

**Acceptance criteria**:
- System prompt uses Pi-style assembly
- No memory injection in core prompt
- Package APPEND_SYSTEM content is appended
- Skills are listed when `read` tool is available
- Context files loaded correctly
- Build succeeds

---

## Task 6.4: Rewrite AgentCoordinator (Thin UI Bridge)

**File**: Modify `Features/Agent/AgentCoordinator.swift`

**Current** (399 lines): Owns message history, context limits, memory loading, system prompt building

**New** (~250 lines): Thin UI bridge that delegates to `Agent` class

**What changes**:

1. **Remove**:
   - `contextLimit = 60` hard truncation
   - Observation masking (`withObservationMasking`)
   - Memory loading (`dataStore?.loadArray(AgentMemory.self, ...)`)
   - System prompt building
   - Direct AgentRunner interaction
   - **The coordinator's own `messages` array as source of truth**

2. **Add**:
   - Reference to `Agent` (from Epic 2)
   - Subscription to `AgentEvent` for UI updates
   - Delegation to `Agent.prompt()` for sending messages

3. **Keep**:
   - Voice I/O flow (startVoiceInput, stopVoiceInputAndSend, cancelVoiceInput)
   - Conversation management (new, load, delete)
   - @Published UI state (streamingText, isGenerating, voiceState, error)
   - SpeechCoordinator integration (auto-speak)
   - Notch controller updates

### Single source of truth for messages

**Critical design rule**: `agent.state.messages` is the **single source of truth**. The coordinator does NOT maintain its own message array. It derives UI messages from the agent's state.

The old plan had the coordinator appending user messages locally AND the agent loop pushing them into context — creating a double-append. The fix:

1. `sendMessage()` calls `agent.prompt(message)` — this is the ONLY place messages are added
2. The agent loop pushes the prompt into `context.messages` (Task 2.2, step 1)
3. `Agent.state.messages` is updated by the agent's internal event handling (Task 2.4)
4. The coordinator observes `agent.state.messages` and converts to `[AgentChatMessage]` for UI display

```swift
// The coordinator's @Published messages is a DERIVED view, not a primary store
@Published private(set) var displayMessages: [AgentChatMessage] = []

func sendMessage(_ text: String) async {
    // Do NOT append locally — let the agent own the message lifecycle
    let userMessage = UserMessage.create(text)
    streamingText = ""
    isGenerating = true

    do {
        try await agent.prompt(CoreMessage.user(userMessage))
    } catch {
        self.error = error.localizedDescription
    }
}
```

**Event subscription** (replaces the old stream iteration):
```swift
private func subscribeToAgentEvents() {
    _ = agent.subscribe { [weak self] event in
        Task { @MainActor in
            self?.handleAgentEvent(event)
        }
    }
}

private func handleAgentEvent(_ event: AgentEvent) {
    switch event {
    case .agentStart:
        isGenerating = true
        refreshDisplayMessages()
    case .contextTransformStart(let reason):
        // Show UI indicator (e.g., "Compacting memories..." for .compaction)
        updatePhaseIndicator(reason)
    case .contextTransformEnd(_, let didMutate):
        // Clear compaction indicator; refresh display if context was mutated
        clearPhaseIndicator()
        if didMutate { refreshDisplayMessages() }
    case .messageUpdate(_, let delta):
        if let text = delta.textDelta {
            streamingText += text
        }
    case .toolExecutionStart(_, let toolName, _):
        notchController?.updateToolCall(toolName)
    case .turnEnd(_, _):
        // Save after each turn for crash resilience
        refreshDisplayMessages()
        saveConversation()
    case .agentEnd(_):
        isGenerating = false
        streamingText = ""
        refreshDisplayMessages()
        saveConversation()  // Definitive save at session end
        autoSpeakIfEnabled()
    // ... other events
    }
}

/// Derive UI messages from the agent's authoritative state
private func refreshDisplayMessages() {
    self.displayMessages = agent.state.messages.map { AgentChatMessage(from: $0) }
}
```

**Conversation load/save**: When loading a conversation from disk, the messages are loaded into `Agent.state.messages` (the agent provides an API for this). When saving, the full `agent.state.messages` array is persisted — this includes `CompactionSummaryMessage` entries from compaction, so reloaded conversations start with pre-compacted context.

**Persistence timing**:
- **`turnEnd`**: Save after each complete turn (assistant message + tool results). Provides crash resilience — at most one turn of work is lost on a crash.
- **`agentEnd`**: Definitive save when the agent finishes. This is the minimum required save point.

**Critical**: No more `messages.suffix(60)` truncation. The full message history is maintained by the `Agent` and compaction handles context limits via `transformContext`.

**Acceptance criteria**:
- No context limit truncation
- No observation masking
- No memory loading/injection
- `agent.state.messages` is the single source of truth
- Coordinator's `displayMessages` is derived, never appended to directly
- No double-append of user or agent messages
- Delegates to Agent.prompt()
- Subscribes to events for UI updates
- `contextTransformStart` → UI indicator ("Compacting memories..." etc.)
- `contextTransformEnd` → clear indicator, refresh display if mutated
- Saves on `turnEnd` (crash resilience) and `agentEnd` (definitive)
- Full `agent.state.messages` (including `CompactionSummaryMessage`) is persisted
- Voice I/O still works
- Conversation save/load still works
- Build succeeds

---

## Task 6.5: Wire DependencyContainer for New Architecture

**File**: Modify `App/DependencyContainer.swift`

**Changes**:

1. **New dependencies to create**:
   ```swift
   // Path sandbox
   let sandbox = PathSandbox(root: agentRoot)

   // Extension system
   let extensionHost = ExtensionHost()
   let extensionRunner: ExtensionRunner  // created after context factory exists

   // Package system
   let packageRegistry = PackageRegistry()

   // New tool registry
   let toolRegistry = NewToolRegistry(sandbox: sandbox, extensionHost: extensionHost)

   // Context loading
   let contextLoader = ContextLoader(agentRoot: agentRoot)
   let skillRegistry: [SkillMetadata]  // discovered at startup

   // Compaction
   let contextManager = ContextManager(settings: .standard)

   // Agent (new core loop)
   let agent: Agent
   ```

2. **Bootstrap sequence** (in `setup()` or a new `setupAgent()` method):
   ```swift
   // 1. Bootstrap packages
   PackageBootstrap.bootstrap(
       packageRegistry: packageRegistry,
       extensionHost: extensionHost,
       agentRoot: agentRoot
   )

   // 2. Discover skills
   let skills = SkillRegistry.discover(
       locations: [agentRoot.appendingPathComponent("skills")],
       packageSkillPaths: packageRegistry.allSkillPaths
   )

   // 3. Load context
   let loadedContext = contextLoader.load(
       packagePaths: packageRegistry.allContextFilePaths,
       extensionPaths: []  // TODO: from resources_discover
   )

   // 4. Assemble system prompt
   let systemPrompt = SystemPromptAssembler.assemble(
       defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
       loadedContext: loadedContext,
       skills: skills,
       tools: toolRegistry.allTools,
       dateTime: Date(),
       agentRoot: agentRoot.path
   )

   // 5. Create Agent
   let config = AgentLoopConfig(
       model: AgentModelRef(id: selectedModelID),
       convertToLlm: defaultConvertToLlm,
       contextTransform: ContextTransformConfig(
           reason: .compaction,
           transform: makeTransformContext(contextManager: contextManager, contextWindow: 120_000)
       ),
       getSteeringMessages: nil,  // wired later for voice interrupts
       getFollowUpMessages: nil
   )

   let agent = Agent(config: config, systemPrompt: systemPrompt, tools: toolRegistry.allTools, generate: { systemPrompt, messages, tools, signal in
       try agentEngine.generate(
           systemPrompt: systemPrompt,
           messages: messages,
           tools: tools,
           parameters: .default
       )
   })

   // 6. Create coordinator (thin bridge)
   let agentCoordinator = AgentCoordinator(agent: agent, ...)
   ```

3. **Remove**:
   - Old `AgentDataStore` creation
   - Old `ToolRegistry` creation with domain tools
   - Old `AgentRunner` creation

**Acceptance criteria**:
- All new components wired correctly
- Package bootstrap runs at startup
- Skills discovered
- System prompt assembled
- Agent created with correct config
- Coordinator delegates to Agent
- Build succeeds

---

## Task 6.6: Remove Old Domain Tools

**Files to delete**:
- `Features/Agent/Tools/MemoryTools.swift`
- `Features/Agent/Tools/TaskTools.swift`
- `Features/Agent/Tools/GoalTools.swift`
- `Features/Agent/Tools/HabitTools.swift`
- `Features/Agent/Tools/MoodTools.swift`
- `Features/Agent/Tools/ReminderTools.swift`
- `Features/Agent/Tools/RespondTool.swift`
- `Features/Agent/Tools/AgentDataStore.swift`
- `Features/Agent/Tools/DateParsingUtility.swift`

**Also remove**:
- `Features/Agent/AgentChatFormatter.swift` (replaced by `MessageConversion.swift`)

**Total**: ~1,100 lines removed

**Before deleting**: Verify no remaining references to these files. Search for:
- `MemorySaveTool`, `MemoryUpdateTool`, `MemoryDeleteTool`
- `TaskCreateTool`, `TaskListTool`, `TaskCompleteTool`
- `GoalCreateTool`, `GoalListTool`, `GoalUpdateTool`
- `HabitCreateTool`, `HabitLogTool`, `HabitStatusTool`
- `MoodLogTool`, `MoodListTool`
- `ReminderSetTool`
- `RespondTool`
- `AgentDataStore`
- `DateParsingUtility`
- `AgentChatFormatter`

All references should have been removed in Tasks 6.1-6.5.

**Acceptance criteria**:
- All listed files deleted
- No remaining references in codebase
- Build succeeds

---

## Task 6.7: Update AgentChatMessage → Protocol-Based Messages

**File**: Modify `Features/Agent/AgentChatMessage.swift`

**Strategy**: Don't delete this file yet — views may still reference it for display. Instead:

1. Add conformance to `AgentMessageProtocol`:
   ```swift
   extension AgentChatMessage: AgentMessageProtocol {
       func toLLMMessage() -> LLMMessage? {
           switch role {
           case .system: return .system(content: content)
           case .user: return .user(content: content)
           case .assistant: return .assistant(content: content, toolCalls: nil)
           case .tool: return .toolResult(toolCallId: "", content: content)
           }
       }
   }
   ```

2. Add conversion to/from new types:
   ```swift
   extension AgentChatMessage {
       init(from message: any AgentMessageProtocol) {
           // Convert CoreMessage/CustomAgentMessage to AgentChatMessage for UI display
       }
   }
   ```

This is a transitional step. In Task 6.9, the views will be updated to use the new message types directly.

**Acceptance criteria**:
- `AgentChatMessage` conforms to `AgentMessageProtocol`
- Views still render messages correctly
- Build succeeds

---

## Task 6.8: Remove Old AgentRunner

**File**: `Features/Agent/AgentRunner.swift`

**Strategy**: The old `AgentRunner` is no longer referenced after Task 6.4 (coordinator rewrite). It can be deleted.

**Before deleting**: Verify `AgentRunner` has no remaining references. Check:
- `DependencyContainer.swift` — should use `Agent` now
- `AgentCoordinator.swift` — should use `Agent` now
- `BenchmarkRunner.swift` — will be updated in Epic 7

If `BenchmarkRunner` still references `AgentRunner`, keep the file but mark it deprecated. Update benchmarks in Epic 7.

**Acceptance criteria**:
- Old `AgentRunner` deleted (or deprecated if benchmarks still reference it)
- Build succeeds

---

## Task 6.9: Update Views for New Message Types

**Files**: Modify view files in `Features/Agent/Views/`

**Changes**:

The views currently render `AgentChatMessage`. They need to render the new protocol-based messages.

**Approach (MVP)**: Keep `AgentChatMessage` as a **UI display type only**. The coordinator's `refreshDisplayMessages()` (from Task 6.4) converts `agent.state.messages` to `[AgentChatMessage]` via the `init(from:)` added in Task 6.7.

Views are unchanged — they still bind to `coordinator.displayMessages: [AgentChatMessage]`.

**Minor view updates needed**:
1. `AgentConversationListView.swift` — bind to `displayMessages` instead of `messages`
2. `AgentContentView.swift` — same property name change
3. `AgentInputBarView.swift` — no changes (only uses `sendMessage()` action)

**No architectural change to views**. The conversion from protocol types → `AgentChatMessage` is the coordinator's responsibility (single place, easy to maintain).

**Acceptance criteria**:
- Chat UI renders messages correctly
- Streaming text displays during generation
- Thinking blocks display correctly
- Tool call/result display correctly
- Build succeeds

---

## Task 6.10: End-to-End Smoke Test

**No code changes**. Manually verify:

1. **Text chat**: Send a message → agent responds using file tools
2. **File operations**: Ask agent to "read memories.md" → agent uses read tool
3. **Memory skill**: Say "remember that I like coffee" → agent reads skill, uses edit/write on memories.md
4. **Task skill**: Say "add a task: buy groceries" → agent reads skill, edits tasks.md
5. **Voice input**: Hold Control+Space, speak, release → transcription → agent response
6. **Auto-speak**: Enable auto-speak → agent reads response aloud
7. **Conversation persistence**: Close and reopen app → conversation loads
8. **New conversation**: Create new → clean slate
9. **Compaction**: Send many messages → verify compaction triggers (check logs)

**Acceptance criteria**:
- All 9 scenarios work
- No crashes
- Logs show clean execution flow

---

## Summary

This is the most disruptive epic. After completion, the agent runs on the new Pi-aligned architecture.

**Files modified**: 7 (ToolRegistry, AgentConversation, ConversationStore, SystemPromptBuilder, AgentCoordinator, DependencyContainer, AgentChatMessage)
**Files deleted**: 10 (domain tools, formatter, data store)
**New files**: 0 (all new code was created in Epics 0-5)
**Lines removed**: ~1,100 (domain tools)
**Lines changed**: ~850 (rewrites of existing files)
