# Agent Proactive Scheduling — Design Spec

**Status**: Draft v3
**Date**: 2025-03-15
**Scope**: Heartbeat + Cron scheduling, background agent sessions, proactive user interactions

---

## 1. Vision

The agent becomes proactive — it can schedule its own work, run background sessions without user input, and surface results through notifications, TTS, and a dedicated scheduled-tasks view. The user configures high-level goals; the agent autonomously decides when and how to act.

This is NOT a coding agent feature. The agent is a general-purpose assistant running a small local model. It reviews tasks, plans the day, monitors things, runs commands, and talks to the user through TTS when it has something to say.

**Core principle**: Local inference is nearly free. The agent should be generous with background work — running for hours if needed, checking things frequently, and doing real thinking in the background.

---

## 2. Key Contracts

Before diving into architecture, two contracts that shape every design decision:

### 2.1 Persistence Contract

**"Persistent" means durable state on disk, restored on app launch.** It does NOT mean autonomous execution while the app is terminated.

What persists across app restarts:

- Task definitions (cron expression, prompt, enabled state, metadata)
- Heartbeat configuration and checklist
- Run history (per-run logs with timestamps, results, token counts)
- Background session transcripts (full agent conversation for each task)

What does NOT persist:

- In-flight task execution (if app quits mid-run, the run is marked `interrupted`)
- Timer state (rebuilt from task definitions on launch)

**Missed-run policy**: On app launch, the scheduler scans all enabled tasks using the **persisted** `nextRunAt` value (NOT recomputed). The persisted value is the source of truth for what was due while the app was closed:

1. Read each task's stored `nextRunAt` from `index.json`
2. Compare against `now` — if `nextRunAt >= now`, task is not overdue, skip
3. If `nextRunAt < now` (task was due while app was closed):
   - If missed by < 1 hour → enqueue for immediate catch-up
   - If missed by >= 1 hour → log a `TaskRunResult.missed(at: nextRunAt)` run entry
4. **After** evaluating the miss, advance `nextRunAt` to the next future occurrence
5. At most one catch-up per task per launch (no backlog accumulation)

**Critical ordering**: Step 4 must happen AFTER step 3. If you recompute `nextRunAt` first, you lose the evidence that a run was missed, because the new `nextRunAt` will be in the future.

### 2.2 Surfacing Contract

**Background results never merge into the user's chat conversation.** They are surfaced through:

- macOS notifications (clicking opens the background session)
- Menu bar badge (unread count)
- TTS speech (agent speaks the result proactively)
- Cron panel in sidebar (real-time status, run history)

The user views background work by opening the background session — a full-transparency read-only chat view showing all agent thinking, tool calls, and results. This is architecturally separate from `AgentConversationStore`.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Tesseract Agent App                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  User Chat   │  │  Cron Panel  │                        │
│  │  (existing)  │  │  (new view)  │                        │
│  └──────┬───────┘  └──────┬───────┘                        │
│         │                 │                                 │
│  ┌──────┴─────────────────┴──────────────────────────────┐  │
│  │         SchedulingService (MainActor — UI state)       │  │
│  │  @Observable: tasks, runHistory, heartbeatStatus       │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴────────────────────────────────┐  │
│  │         SchedulingActor (off-MainActor — execution)    │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │  Heartbeat   │  │  Cron Engine │  │  Task Store  │  │  │
│  │  │  (periodic)  │  │  (precision) │  │ (persistent) │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │  │
│  └─────────┼────────────────┼────────────────────────────┘  │
│            │                │                               │
│  ┌─────────┴────────────────┴────────────────────────────┐  │
│  │       BackgroundAgentFactory + Background Sessions     │  │
│  │  - Creates isolated Agent instances per task           │  │
│  │  - Separate conversation context from user chat        │  │
│  │  - Full tool access (read/write/edit/ls)               │  │
│  │  - Loads model on demand via AgentEngine               │  │
│  │  - One background agent at a time (sequential queue)   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              InferenceArbiter (GPU serialization)      │  │
│  │  LLM + TTS: independently lazy-loaded, co-resident    │  │
│  │  ImageGen: exclusive (evicts co-residents, prototype)  │  │
│  │  withExclusiveGPU(.slot) { } — scoped lease API       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Notification Layer                        │  │
│  │  macOS notifications · Menu bar badge · TTS speech    │  │
│  │  Cron panel updates                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Two Scheduling Mechanisms

### 4.1 Heartbeat (Awareness Layer)

Periodic, contextual check-ins. The agent wakes up, evaluates a checklist, and decides whether anything needs attention.

**How it works:**

1. Every N minutes (configurable, default 30m), the scheduler fires
2. A background agent instance is created (or the existing heartbeat session is resumed)
3. The agent reads `HEARTBEAT.md` — a markdown checklist of monitoring tasks
4. The agent evaluates each item, using tools as needed
5. If nothing needs attention → silent (`HEARTBEAT_OK`), no user notification
6. If something needs attention → agent acts, then notifies user

**HEARTBEAT.md** (user-editable, stored at `~/Library/Application Support/Tesseract Agent/agent/scheduled-tasks/heartbeat.md`):

```markdown
# Heartbeat Checklist

- Check if any scheduled tasks are overdue
- Review today's goals and assess progress
- Look for new files in ~/Downloads that need attention
- If it's after 5pm and no daily review has happened, start one
```

**Heartbeat configuration** (in settings or package config):

```json
{
  "heartbeat": {
    "enabled": true,
    "interval": "30m"
  }
}
```

**Design notes:**

- Heartbeat runs in a **dedicated background agent instance** (NOT the user's Agent)
- The heartbeat session accumulates context over time (agent remembers previous heartbeats)
- If heartbeat session gets too long, normal context compaction applies
- Heartbeat is the agent's "idle awareness" — it makes the agent feel alive

### 4.2 Cron (Precision Layer)

Exact-time scheduling for specific tasks. Supports both user-created and agent-created schedules.

**Cron expression format:** Standard 5-field (`minute hour day-of-month month day-of-week`)

**Semantic contract:**

- All times are interpreted in the **user's local timezone** (`TimeZone.current`)
- **DST transitions**: When clocks spring forward, a task scheduled during the skipped hour is run at the first valid minute after the transition. When clocks fall back, a task scheduled during the repeated hour runs once (on the first occurrence)
- **Day-of-month + day-of-week**: When both are constrained (not `*`), a date matches if **either** field matches (standard vixie-cron semantics)
- **Supported syntax**: `*`, single values (`5`), ranges (`1-5`), steps (`*/15`), lists (`1,15,30`), range-steps (`1-30/5`)
- **NOT supported**: `L` (last), `W` (nearest weekday), `?`, named days/months (`MON`, `JAN`)

**Examples:**

```
0 9 * * *       → Every day at 9am — "Plan my three tasks for today"
0 17 * * 1-5    → Weekdays at 5pm — "Review what I accomplished today"
*/15 * * * *    → Every 15 minutes — "Check build status"
0 9 * * 1       → Monday 9am — "Weekly review of goals"
```

**Task definition:**

```swift
struct ScheduledTask: Codable, Identifiable, Sendable {
    let id: UUID
    var name: String                    // Display name (not used as identifier)
    var description: String
    var cronExpression: String          // 5-field cron
    var prompt: String                  // What the agent should do
    var enabled: Bool
    var createdBy: TaskCreator          // .user or .agent(reason: String)
    var createdAt: Date
    var lastRunAt: Date?
    var lastRunResult: TaskRunResult?
    var nextRunAt: Date?                // Computed from cron expression
    var runCount: Int
    var maxRuns: Int?                   // nil = unlimited, 1 = one-shot
    var tags: [String]                  // For filtering/grouping
    var notifyUser: Bool                // Whether to send macOS notification
    var speakResult: Bool               // Whether to speak result via TTS
    var sessionId: UUID                 // Stable reference to background session
}

enum TaskCreator: Codable, Sendable {
    case user
    case agent(reason: String)
}

enum TaskRunResult: Codable, Sendable {
    case success(summary: String)
    case noActionNeeded
    case error(message: String)
    case interrupted                    // App quit mid-run
    case missed(at: Date)              // App was not running at scheduled time
}
```

---

## 5. Agent Tools

Three built-in tools for the agent to self-schedule (matching Claude Code's `CronCreate`/`CronList`/`CronDelete` pattern). No update tool — the agent deletes and recreates to modify a task.

### 5.1 `cron_create` — Create a new scheduled task

```json
{
  "name": "cron_create",
  "description": "Create a new scheduled task. Returns the task UUID for future reference.",
  "parameters": {
    "name": {
      "type": "string",
      "description": "Short display name for the task"
    },
    "description": {
      "type": "string",
      "description": "What this task does and why"
    },
    "cron": {
      "type": "string",
      "description": "5-field cron expression (e.g. '0 9 * * *' for daily at 9am)"
    },
    "prompt": {
      "type": "string",
      "description": "The instruction to execute when the task fires"
    },
    "notify": {
      "type": "boolean",
      "description": "Send macOS notification on completion",
      "default": true
    },
    "speak": {
      "type": "boolean",
      "description": "Speak the result via TTS",
      "default": false
    },
    "max_runs": {
      "type": "integer",
      "description": "Maximum number of runs (omit for unlimited, 1 for one-shot)"
    }
  },
  "required": ["name", "prompt", "cron"]
}
```

**Returns**: `{ "task_id": "550e8400-...", "name": "morning-planning", "next_run": "2025-03-16T09:00:00Z" }`

**Agent self-scheduling example:**

```
User: "Help me stay on track with my tasks today"
Agent: "I'll schedule check-ins throughout the day."
→ Agent calls cron_create tool:
  name: "morning-planning"
  cron: "0 9 * * *"
  prompt: "Review the user's goals file and suggest 3 tasks for today. Speak the plan."
  speak: true

→ Agent calls cron_create tool:
  name: "midday-check"
  cron: "0 13 * * *"
  prompt: "Check progress on today's tasks. If behind, suggest what to prioritize."
  notify: true

→ Agent calls cron_create tool:
  name: "evening-review"
  cron: "0 18 * * *"
  prompt: "Summarize what was accomplished today. Update the goals file."
  speak: true
```

### 5.2 `cron_list` — List scheduled tasks

```json
{
  "name": "cron_list",
  "description": "List all scheduled tasks with their UUID, status, next run time, and recent history.",
  "parameters": {
    "filter": {
      "type": "string",
      "enum": ["all", "active", "paused", "mine"],
      "default": "all"
    }
  }
}
```

**Returns**: Array of task summaries including `task_id`, `name`, `cron`, `enabled`, `next_run`, `last_run`, `created_by`.

### 5.3 `cron_delete` — Delete a scheduled task

```json
{
  "name": "cron_delete",
  "description": "Delete a scheduled task by UUID.",
  "parameters": {
    "task_id": {
      "type": "string",
      "description": "UUID of the task to delete"
    },
    "reason": {
      "type": "string",
      "description": "Why this task is being deleted"
    }
  },
  "required": ["task_id"]
}
```

All tools require UUID, never name-based lookup. Names are display-only and may be duplicated. To modify a task, delete it and recreate with new parameters.

---

## 6. Background Agent Sessions

When a scheduled task or heartbeat fires, it runs in a **background agent session** — a separate `Agent` instance with its own conversation context, fully isolated from the user's chat.

### 6.1 Session Factory

The current app has one stateful `Agent` with one mutable context (`AgentContext`). Background tasks must NOT reuse this instance — cross-contamination between user chat and background work would corrupt both.

```swift
/// Creates isolated Agent instances for background task execution.
/// Each background agent has its own AgentContext, conversation history,
/// and event subscribers — completely separate from the user's chat Agent.
///
/// IMPORTANT: Background agents must go through the same assembly pipeline
/// as the foreground agent in DependencyContainer.agent. This means:
/// - Package bootstrap (skills, context files, prompt appends)
/// - Skill discovery and registration
/// - Context file loading (AGENTS.md, CLAUDE.md, etc.)
/// - Full system prompt assembly (not just a task-specific prompt)
/// - Compaction wiring (background sessions accumulate and need compaction)
/// - Tool registry (same built-in + extension tools)
///
/// Without this parity, background agents silently behave like a different
/// agent — missing skills, context, and compaction.
@MainActor
final class BackgroundAgentFactory {
    private let agentEngine: AgentEngine    // Shared — one model, one GPU
    private let toolRegistry: ToolRegistry  // Shared — same tools available
    private let sessionStore: BackgroundSessionStore
    private let contextManager: ContextManager

    // --- Cached assembly inputs (computed once, reused across runs) ---
    // These are immutable between app launches. Rebuilt only on init or
    // when packages/extensions change (same cadence as foreground agent).
    // This avoids repeating the full discovery/load pipeline on MainActor
    // for every heartbeat/cron run, which would cause UI hitching.
    private let cachedSkills: [Skill]
    private let cachedLoadedContext: LoadedContext
    private let cachedTools: [AgentToolDefinition]
    private let cachedCompactionTransform: ContextTransformConfig
    private let cachedGenerateClosure: /* AgentEngine bridge closure */

    init(
        agentEngine: AgentEngine,
        toolRegistry: ToolRegistry,
        sessionStore: BackgroundSessionStore,
        contextManager: ContextManager,
        packageRegistry: PackageRegistry
    ) {
        self.agentEngine = agentEngine
        self.toolRegistry = toolRegistry
        self.sessionStore = sessionStore
        self.contextManager = contextManager

        // Cache immutable assembly inputs once (same work DependencyContainer does)
        let agentRoot = PathSandbox.defaultRoot
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let cachedSkillPaths = PackageBootstrap.cachedSkillPaths(
            from: packageRegistry, agentRoot: agentRoot
        )
        self.cachedSkills = SkillRegistry.discover(
            locations: [skillsDir], packageSkillFiles: cachedSkillPaths
        )
        let contextLoader = ContextLoader(agentRoot: agentRoot)
        self.cachedLoadedContext = contextLoader.load(
            packageContextFiles: packageRegistry.allContextFilePaths,
            packagePromptAppends: packageRegistry.allPromptAppendPaths,
            packageSystemOverrides: []
        )
        self.cachedTools = toolRegistry.allTools
        self.cachedCompactionTransform = makeCompactionTransform(
            contextManager: contextManager,
            contextWindow: 120_000,
            summarize: { /* same AgentEngine bridge as foreground */ }
        )
        self.cachedGenerateClosure = { /* same AgentEngine bridge as foreground */ }
    }

    /// Create a new background Agent for a scheduled task.
    /// Uses cached assembly inputs — only rebuilds the time-sensitive prompt
    /// suffix (date/time) and task-specific preamble per run.
    func createAgent(for task: ScheduledTask) async -> Agent {
        // 1. Load persisted session (or create new)
        let session = await sessionStore.loadOrCreate(sessionId: task.sessionId)

        // 2. Assemble system prompt using cached context + fresh date/time
        let agentRoot = PathSandbox.defaultRoot
        let basePrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: cachedLoadedContext,   // cached
            skills: cachedSkills,                  // cached
            tools: cachedTools,                    // cached
            dateTime: Date(),                      // fresh per run
            agentRoot: agentRoot.path
        )
        let systemPrompt = basePrompt + "\n\n" +
            SystemPromptAssembler.backgroundPreamble(for: task)  // task-specific

        // 3. Create agent config (compaction + generate closure are cached)
        let config = AgentLoopConfig(
            model: AgentModelRef(id: settingsManager.selectedAgentModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: cachedCompactionTransform,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        // 4. Create fresh Agent instance
        let agent = Agent(
            config: config,
            systemPrompt: systemPrompt,
            tools: cachedTools,
            generate: cachedGenerateClosure
        )

        // 5. Inject restored conversation as context
        agent.restoreMessages(session.messages)

        return agent
    }
}
```

**Key**: `AgentEngine` (the model/inference layer) is shared — there's only one GPU. But `Agent` instances (conversation state, tool execution context, event subscribers) are fully separate. The assembly pipeline inputs (packages, skills, context, compaction, tools) are **cached once at factory init** — only the date/time suffix and task preamble are rebuilt per run. This avoids repeating expensive discovery/loading on MainActor for every 30-minute heartbeat.

### 6.2 Session Identity and Storage

Sessions are identified by **UUID**, never by task name. Task names are display-only.

Each `ScheduledTask` has a stable `sessionId: UUID` assigned at creation. This UUID is used for:

- Persisting session transcripts on disk
- Linking run logs to their session
- Opening the session viewer in the UI

**BackgroundSessionStore** — separate from `AgentConversationStore`:

```swift
/// Stores background agent sessions separately from user conversations.
/// Unlike AgentConversationStore, these sessions:
/// - Do not require user messages to persist
/// - Have explicit metadata (sessionType, taskId, displayName, lastRunAt)
/// - Accumulate across multiple runs of the same task
actor BackgroundSessionStore {
    // Location: ~/Library/Application Support/Tesseract Agent/agent/background-sessions/

    /// Bump when on-disk session schema changes in incompatible ways.
    /// On version mismatch: wipe background-sessions directory (mirroring
    /// AgentConversationStore.migrateStorageVersionIfNeeded pattern).
    /// Must be set to 1 from day one — never ship without versioning.
    static let storageVersion = 1

    func loadOrCreate(sessionId: UUID) -> BackgroundSession
    func save(_ session: BackgroundSession)
    func delete(sessionId: UUID)
    func listAll() -> [BackgroundSessionSummary]

    /// Called on init. Checks .storage_version file against storageVersion.
    /// On mismatch: wipes directory, writes new version file.
    private func migrateStorageVersionIfNeeded() { ... }
}
```

```swift
struct BackgroundSession: Codable, Identifiable, Sendable {
    let id: UUID                        // Same as task.sessionId
    var sessionType: SessionType        // .heartbeat or .cron
    var displayName: String             // From task name (display-only)
    var taskId: UUID                    // Back-reference to ScheduledTask
    var messages: [TaggedMessage]       // Full conversation transcript
    var lastRunAt: Date?
    var createdAt: Date
}

enum SessionType: String, Codable, Sendable {
    case heartbeat
    case cron
}
```

**Storage layout:**

```
background-sessions/
├── index.json                    // Lightweight summaries for sidebar
├── {session-uuid}.json          // Full session transcript
└── ...
```

### 6.3 Inference Arbiter (Model Ownership)

The current app has fragmented model ownership: `DependencyContainer.prepareForInference` releases image gen models, `SpeechCoordinator.prepareForSpeech` is a separate callback, `SpeechEngine` loads TTS independently, and `AgentEngine` loads LLM independently. The scheduler needs coordinated access — a background task may need LLM inference followed immediately by TTS speech.

**Memory residency contract**: LLM (~3.8GB quantized) and TTS (~1.7GB, will be quantized further) are **independently lazy-loaded but allowed to co-reside** — each loads on first use and stays warm. Neither evicts the other. Both fit comfortably within the 20GB budget (~5.5GB combined). ImageGen (~8GB bf16) is the only **exclusive** slot — loading it evicts co-residents, and any co-resident load evicts it. ImageGen is a prototype (hidden from UI). Whisper STT runs on CoreML in a separate memory pool and is not managed by the arbiter.

**Solution**: Introduce `InferenceArbiter` — a single authority that manages model loading, tracks busy/idle state across consumers, and only swaps out ImageGen when needed.

````swift
/// Single authority for model ownership and GPU serialization.
/// Replaces ad-hoc prepareForInference/prepareForSpeech callbacks.
///
/// Memory residency model:
///   - LLM + TTS are co-resident (independently lazy-loaded, both allowed in
///     memory simultaneously). Neither evicts the other.
///   - ImageGen is exclusive (evicts co-resident models when loaded, and
///     co-resident loads evict ImageGen). Prototype-only, hidden from UI.
///   - STT (WhisperKit) is CoreML on a separate memory pool — not managed here.
///
/// GPU serialization:
///   Only one consumer generates at a time. Access is scoped via
///   withExclusiveGPU — a RAII-style API that acquires on entry, ensures the
///   required model is loaded, and releases on exit (including throws/cancellation).
///   This eliminates the race where two callers both observe idle before either
///   marks busy, and prevents leaked busy state from early returns.
@MainActor
final class InferenceArbiter {
    enum ModelSlot: Sendable {
        case llm            // AgentEngine — co-resident
        case tts            // SpeechEngine — co-resident
        case imageGen       // ImageGenEngine — exclusive
    }

    /// Which slots are currently loaded. LLM + TTS can coexist.
    private(set) var loadedSlots: Set<ModelSlot> = []

    /// The model ID currently loaded in the .llm slot (tracks SettingsManager selection).
    /// When the user changes agent model, the next acquire detects the mismatch and reloads.
    private var loadedLLMModelID: String?

    private let agentEngine: AgentEngine
    private let speechEngine: SpeechEngine
    private let imageGenEngine: ImageGenEngine
    private let zimageGenEngine: ZImageGenEngine
    private let settingsManager: SettingsManager

    /// FIFO waiter queue for GPU access. Each entry is identified by UUID
    /// for cancellation-safe removal. Continuations are throwing so that
    /// cancelled waiters can be resumed with CancellationError.
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []
    private var isLeased: Bool = false

    /// Scoped exclusive GPU access. Waits for any active lease to complete
    /// (FIFO order), ensures the required model is loaded, runs the closure,
    /// and releases the lease on exit — including on throw.
    ///
    /// Cancellation:
    ///   - While waiting in the queue: the waiter is removed and
    ///     CancellationError is thrown without ever acquiring the lease.
    ///   - After resumption but before ownership transfer: Task.checkCancellation()
    ///     runs before isLeased is set, so a cancelled waiter cannot inherit
    ///     the lease during handoff.
    ///   - Once the lease is acquired: cancellation propagates normally through
    ///     the body and the lease is released via defer.
    ///
    /// Usage:
    /// ```swift
    /// let result = try await arbiter.withExclusiveGPU(.llm) {
    ///     // GPU is yours, model is loaded
    ///     return try await engine.generate(...)
    /// }
    /// ```
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        body: () async throws -> T
    ) async throws -> T {
        // Block if lease is held OR waiters exist (prevents queue bypass).
        if isLeased || !waiters.isEmpty {
            // Use withTaskCancellationHandler to dequeue on cancellation.
            // Each waiter has a UUID so cancellation can remove it by identity.
            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    waiters.append((id: waiterID, continuation: continuation))
                }
            } onCancel: {
                // Runs concurrently — MainActor hop to safely mutate waiters.
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let idx = self.waiters.firstIndex(where: { $0.id == waiterID }) {
                        let removed = self.waiters.remove(at: idx)
                        removed.continuation.resume(throwing: CancellationError())
                    }
                    // If already resumed (race with handoff), the post-wait
                    // Task.checkCancellation() below prevents lease acquisition.
                }
            }
        }
        // A waiter may have been resumed just before cancellation won the race.
        // Re-check before claiming the lease so cancelled waiters never inherit it.
        try Task.checkCancellation()
        // At this point we own the lease — set flag before any await.
        isLeased = true
        defer {
            // Atomic handoff: if waiters exist, keep isLeased = true and
            // resume the next waiter directly. The resumed waiter inherits
            // the lease. Only set isLeased = false when the queue is drained.
            if !waiters.isEmpty {
                let next = waiters.removeFirst()
                next.continuation.resume()
            } else {
                isLeased = false
            }
        }

        // Ensure requested model is loaded (co-resident or exclusive)
        try await ensureLoaded(slot)

        return try await body()
    }

    /// Load a model slot. Co-resident slots coexist; ImageGen is exclusive.
    /// For .llm: also checks if the loaded model ID matches the current setting.
    private func ensureLoaded(_ slot: ModelSlot) async throws {
        switch slot {
        case .llm:
            let desiredModelID = settingsManager.selectedAgentModelID
            if loadedSlots.contains(.llm) && loadedLLMModelID == desiredModelID {
                return  // Already loaded with correct model
            }
            // Model changed or not loaded — (re)load
            if loadedSlots.contains(.imageGen) { unload(.imageGen) }
            if loadedSlots.contains(.llm) { unload(.llm) }
            try await load(slot, modelID: desiredModelID)
            loadedLLMModelID = desiredModelID

        case .tts:
            if loadedSlots.contains(slot) { return }
            if loadedSlots.contains(.imageGen) { unload(.imageGen) }
            try await load(slot)

        case .imageGen:
            if loadedSlots.contains(slot) { return }
            for loaded in loadedSlots { unload(loaded) }
            loadedLLMModelID = nil
            try await load(slot)
        }
    }

    private func load(_ slot: ModelSlot, modelID: String? = nil) async throws { ... }
    private func unload(_ slot: ModelSlot) { ... }
}
````

**Rules for background tasks:**

1. Background task fires → calls `arbiter.withExclusiveGPU(.llm) { ... }`
2. Arbiter enqueues behind any active lease (FIFO), then grants exclusive GPU
3. LLM is loaded if needed (no-op if already co-resident with correct model ID), ImageGen evicted if present
4. If user changed agent model in settings since last load, the old LLM is unloaded and the new one loaded
5. Body runs, lease auto-releases on completion/throw/cancellation via `defer`
6. For TTS: a second `withExclusiveGPU(.tts) { ... }` call — TTS loads alongside LLM (co-resident, no eviction)
7. Both models stay warm after lease release

**Migration path**: `InferenceArbiter.withExclusiveGPU` replaces the existing `prepareForInference` closure in `AgentCoordinator` and `prepareForSpeech` in `SpeechCoordinator`. Callers wrap their generation in the scoped block instead of manually tracking busy/idle.

### 6.4 Execution Queue

Only one background agent runs at a time (single GPU, serial generation). Tasks are queued sequentially:

```swift
// In SchedulingActor
private var executionQueue: [ScheduledTask] = []
private var isExecuting: Bool = false

func enqueue(_ task: ScheduledTask) {
    executionQueue.append(task)
    if !isExecuting {
        Task { await drainQueue() }
    }
}

private func drainQueue() async {
    isExecuting = true
    while let task = executionQueue.first {
        executionQueue.removeFirst()
        await waitForUserIdle()
        await ensureModelLoaded()
        await executeTask(task)
    }
    isExecuting = false
}
```

---

## 7. Safety Controls

Even though local inference is free, the scheduler needs guardrails to prevent runaway behavior.

### 7.1 Global Controls

- **Global pause**: One toggle to pause all scheduling (heartbeat + cron). Accessible from menu bar and settings
- **Max active tasks**: Hard limit of 50 active scheduled tasks (same as Claude Code). Agent tool returns error if exceeded
- **Max consecutive runs**: If a task fails 5 times in a row, it's auto-paused with a notification to the user

### 7.2 Provenance Tracking

Every task records who created it (`TaskCreator.user` or `TaskCreator.agent(reason:)`). The UI clearly labels agent-created tasks so users can audit and remove unwanted schedules.

### 7.3 Agent Scheduling Limits

To prevent runaway schedule creation by the agent:

- Agent cannot create more than 10 tasks in a single conversation turn
- Agent cannot create tasks with intervals shorter than 5 minutes
- All agent-created tasks start with `notifyUser: true` (user always knows)

---

## 8. Persistence

### 8.1 Task Store

**Location:** `~/Library/Application Support/Tesseract Agent/agent/scheduled-tasks/`

```
scheduled-tasks/
├── .storage_version              // Version check (same pattern as AgentConversationStore)
├── index.json                    // Task registry (lightweight)
├── tasks/
│   ├── {uuid}.json              // Full task definition
│   └── ...
├── runs/
│   ├── {task-uuid}/
│   │   ├── {run-uuid}.json     // Run log (prompt, result, duration)
│   │   └── ...
│   └── ...
└── heartbeat.md                  // User-editable heartbeat checklist
```

Both `ScheduledTaskStore` and `BackgroundSessionStore` use the same versioning pattern as `AgentConversationStore`: a `.storage_version` file checked on init, with wipe-and-recreate on mismatch. Both start at version 1.

**index.json** — compact, loaded at startup:

```json
{
  "version": 1,
  "tasks": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "morning-planning",
      "cronExpression": "0 9 * * *",
      "enabled": true,
      "nextRunAt": "2025-03-16T09:00:00Z",
      "createdBy": "agent",
      "sessionId": "660e8400-e29b-41d4-a716-446655440001"
    }
  ]
}
```

**Run log** — per-run record:

```json
{
  "id": "run-uuid",
  "taskId": "task-uuid",
  "sessionId": "session-uuid",
  "startedAt": "2025-03-15T09:00:03Z",
  "completedAt": "2025-03-15T09:00:47Z",
  "durationSeconds": 44,
  "result": "success",
  "summary": "Planned 3 tasks for today: finish scheduling spec, review PR #42, update dependencies.",
  "notifiedUser": true,
  "spokeResult": true,
  "tokensUsed": 2847
}
```

### 8.2 Startup Recovery

On app launch, `SchedulingService.start()`:

1. Check `.storage_version` — wipe and recreate on mismatch
2. Load `index.json` — restore all task definitions with their **persisted** `nextRunAt`
3. For each enabled task, evaluate missed-run policy against persisted `nextRunAt` (see section 2.1)
4. **After** evaluating misses, advance `nextRunAt` to next future occurrence for each task
5. Enqueue any catch-up tasks (missed by < 1 hour)
6. Rebuild polling timer
7. Load heartbeat config, start heartbeat timer if enabled

---

## 9. Notification Layer

When a background task produces a result worth surfacing:

### 9.1 macOS Notifications

```swift
// No additional entitlement needed for UNUserNotificationCenter in sandboxed apps.
// Just request authorization at runtime.
UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
```

Notification content:

- **Title**: Task name (e.g., "Morning Planning")
- **Body**: Summary (e.g., "3 tasks planned for today. Top priority: finish scheduling spec.")
- **Action**: Clicking opens the app to the relevant background session
- **Category**: `scheduledTask` — with "View Session" action button

### 9.2 Menu Bar Badge

- Unread background results show a badge count on the menu bar icon
- Badge clears when user views the Cron panel or the relevant session

### 9.3 TTS Speech

If `speakResult: true` on the task:

- Agent speaks the result summary through the TTS system (Qwen3TTS)
- Natural way for the agent to proactively talk to the user
- "Hey, it's 9am. I've planned your day — your top priority is finishing the scheduling spec, then reviewing PR 42, then updating dependencies."

### 9.4 Cron Panel Updates

- The Cron sidebar panel shows real-time status of running/completed tasks
- Live indicator while a background task is executing

---

## 10. UI Design

### 10.1 Sidebar Structure

```
┌─────────────────────┐
│  Tesseract Agent     │
├─────────────────────┤
│  Conversations       │  ← Existing
│    ├ Today's chat    │
│    ├ Yesterday       │
│    └ ...             │
├─────────────────────┤
│  Scheduled           │  ← NEW
│    ├ Heartbeat       │  (active, last run 12m ago)
│    ├ Morning Plan    │  (next: tomorrow 9:00am)
│    ├ Midday Check    │  (next: today 1:00pm)
│    └ Evening Review  │  (next: today 6:00pm)
├─────────────────────┤
│  Settings            │  ← Existing
└─────────────────────┘
```

### 10.2 Cron Panel (Main Content Area)

When user selects "Scheduled" in sidebar:

```
┌──────────────────────────────────────────────────┐
│  Scheduled Tasks                    [Pause All]  │
│                                          [+ New] │
├──────────────────────────────────────────────────┤
│                                                  │
│  Heartbeat                         every 30m     │
│  Last run: 12 minutes ago · OK (no action)       │
│  [View Session] [Configure] [Pause]              │
│                                                  │
│  ──────────────────────────────────────────────  │
│                                                  │
│  Morning Planning              0 9 * * *         │
│  Created by: Agent · Next: Mar 16, 9:00am        │
│  Last run: Today 9:00am · 3 tasks planned        │
│  [View Session] [Edit] [Delete]                  │
│                                                  │
│  Midday Check                  0 13 * * *        │
│  Created by: Agent · Next: Today, 1:00pm         │
│  Never run                                       │
│  [View Session] [Edit] [Delete]                  │
│                                                  │
│  Evening Review                0 18 * * *        │
│  Created by: User · Next: Today, 6:00pm          │
│  Last run: Yesterday 6:00pm · Summary done       │
│  [View Session] [Edit] [Delete]                  │
│                                                  │
├──────────────────────────────────────────────────┤
│  Run History                                     │
│  ┌────────────────────────────────────────────┐  │
│  │ Today                                      │  │
│  │  09:00  Morning Planning  done   44s 2.8K  │  │
│  │  08:30  Heartbeat         ok     12s 0.9K  │  │
│  │  08:00  Heartbeat         ok      8s 0.6K  │  │
│  │                                            │  │
│  │ Yesterday                                  │  │
│  │  18:00  Evening Review    done   67s 4.1K  │  │
│  │  ...                                       │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 10.3 Background Session View

When user clicks "View Session" on a task, they see the agent's background conversation — all the thinking, tool calls, and results from that task's runs. This is a read-only chat view showing full transparency into what the agent did.

Uses the same chat view component as user conversations, but:

- No input field (read-only in MVP)
- Session header shows task name, schedule, and run count
- Each run is visually separated (timestamp divider)

---

## 11. Scheduling Engine

### 11.1 Split Architecture

Following the app's existing pattern (MainActor facade + off-MainActor heavy work), the scheduler is split into two layers:

```swift
/// UI-facing state. Drives SwiftUI views.
/// Only holds observable state and delegates all work to SchedulingActor.
@MainActor
@Observable
final class SchedulingService {
    // Observable state for UI
    var tasks: [ScheduledTask] = []
    var runHistory: [TaskRun] = []
    var heartbeatStatus: HeartbeatStatus = .idle
    var currentlyRunningTaskId: UUID? = nil
    var unreadResultCount: Int = 0
    var isPaused: Bool = false          // Global pause

    // Delegate
    private let actor: SchedulingActor

    // Lifecycle (called by DependencyContainer)
    func start() async { ... }          // Restore state, start actor
    func stop() async { ... }           // Persist state, stop actor

    // Task CRUD (UI-driven, delegates to actor)
    func createTask(_ task: ScheduledTask) async { ... }
    func updateTask(_ task: ScheduledTask) async { ... }
    func deleteTask(id: UUID) async { ... }
    func pauseTask(id: UUID) async { ... }
    func resumeTask(id: UUID) async { ... }
    func pauseAll() { ... }
    func resumeAll() { ... }
}
```

```swift
/// Off-MainActor actor that owns polling, execution, persistence, and model loading.
/// Communicates state changes back to SchedulingService via async callbacks.
actor SchedulingActor {
    private let taskStore: ScheduledTaskStore
    private let sessionStore: BackgroundSessionStore
    private let agentFactory: BackgroundAgentFactory
    private let agentEngine: AgentEngine
    private let notificationService: NotificationService
    private var pollTask: Task<Void, Never>?

    // Polling loop — checks every 60 seconds
    func startPolling() {
        pollTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                await checkAndRunDueTasks()
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
    }

    func checkAndRunDueTasks() async {
        guard !isPaused else { return }
        let now = Date()
        let dueTasks = tasks.filter { $0.enabled && $0.isDue(at: now) }
        for task in dueTasks {
            await enqueue(task)
        }
    }

    private func executeTask(_ task: ScheduledTask) async -> TaskRun {
        // 1. Create isolated background agent (full assembly pipeline — uses cached inputs)
        let agent = await agentFactory.createAgent(for: task)

        // 2. Run LLM inference (scoped GPU lease — waits, loads, auto-releases)
        try await arbiter.withExclusiveGPU(.llm) {
            agent.prompt(UserMessage(content: task.prompt))
            await agent.waitForIdle()
        }

        // 3. Persist session transcript
        await sessionStore.save(agent.session)

        // 4. Build run log
        let run = TaskRun(/* ... */)
        await taskStore.saveRun(run)

        // 5. Notify (macOS notification, badge)
        if task.notifyUser && run.result != .noActionNeeded {
            await notificationService.notify(task: task, run: run)
        }

        // 6. TTS (scoped GPU lease — co-resident with LLM, no eviction)
        if task.speakResult && run.result != .noActionNeeded {
            try await arbiter.withExclusiveGPU(.tts) {
                await notificationService.speak(run.summary)
            }
        }

        // 7. Update task state (lastRunAt, nextRunAt, runCount)
        await taskStore.updateAfterRun(taskId: task.id, run: run)

        // 8. Report back to MainActor service
        await onTaskCompleted(task, run)

        return run
    }
}
```

### 11.2 Cron Expression Parser

Full parser with well-defined semantics. Must be thoroughly tested.

```swift
struct CronExpression: Codable, Sendable, Equatable {
    let minute: CronField       // 0-59
    let hour: CronField         // 0-23
    let dayOfMonth: CronField   // 1-31
    let month: CronField        // 1-12
    let dayOfWeek: CronField    // 0-7 (0 and 7 are Sunday)

    /// Parse a 5-field cron expression. Throws on invalid syntax.
    init(parsing expression: String) throws { ... }

    /// Next occurrence after the given date, in the given timezone.
    /// Handles DST transitions:
    /// - Spring forward: skipped times resolve to first valid minute after transition
    /// - Fall back: repeated times match on first occurrence only
    func nextOccurrence(after date: Date, in timeZone: TimeZone = .current) -> Date { ... }

    /// Whether the given date matches this expression (minute-level precision).
    func matches(_ date: Date, in timeZone: TimeZone = .current) -> Bool { ... }

    /// Human-readable description (e.g., "Every day at 9:00 AM")
    var humanReadable: String { ... }
}

enum CronField: Codable, Sendable, Equatable {
    case any                          // *
    case value(Int)                   // 5
    case range(Int, Int)              // 1-5
    case step(base: CronField, Int)   // */15 or 1-30/5
    case list([CronField])           // 1,15,30
}
```

**Test coverage requirements:**

- Every field type individually (value, range, step, list, any)
- Day-of-month + day-of-week OR semantics
- DST spring-forward and fall-back
- Month boundaries (28/29/30/31 day months)
- Leap year February 29
- Edge cases: `0 0 1 1 *` (midnight Jan 1), `59 23 31 12 *` (last minute of year)
- Invalid expressions (too few/many fields, out-of-range values)

---

## 12. Integration Points

### 12.1 DependencyContainer

```swift
// New lazy properties:
lazy var schedulingActor: SchedulingActor = { ... }()
lazy var schedulingService: SchedulingService = { ... }()
lazy var backgroundAgentFactory: BackgroundAgentFactory = { ... }()
lazy var backgroundSessionStore: BackgroundSessionStore = { ... }()
lazy var scheduledTaskStore: ScheduledTaskStore = { ... }()
lazy var notificationService: NotificationService = { ... }()

// Wire scheduling tools into agent's tool registry
// Wire schedulingService into view hierarchy via environment
```

### 12.2 Tool Registration

The three scheduling tools (`cron_create`, `cron_list`, `cron_delete`) are registered as built-in tools (not extension tools), since they're core to the agent's proactive capabilities.

### 12.3 Extension Events

New event types for the extension system:

```swift
case scheduledTaskCreated(task)
case scheduledTaskFired(task)
case scheduledTaskCompleted(task, result)
case heartbeatStart
case heartbeatEnd(result)
```

### 12.4 App Lifecycle

```swift
// In Info.plist — prevent automatic/sudden termination so we can persist state:
<key>NSSupportsAutomaticTermination</key>
<false/>
<key>NSSupportsSuddenTermination</key>
<false/>

// Note: These flags do NOT enable background execution.
// They prevent macOS from silently killing the app, giving us time to
// persist task state in applicationWillTerminate.
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Core Scheduling) — MVP

- [x] `CronExpression` parser with full semantic contract + tests
- [x] `ScheduledTask` model + `ScheduledTaskStore` persistence (with `.storage_version`)
- [x] `SchedulingActor` with 60-second polling loop
- [x] `SchedulingService` (MainActor facade)
- [x] Three agent tools: `cron_create`, `cron_list`, `cron_delete`
- [x] `BackgroundAgentFactory` — isolated agent instances with full assembly pipeline parity
- [x] `BackgroundSessionStore` — separate from user conversations (with `.storage_version`)
- [x] `InferenceArbiter` — unified model ownership across LLM/TTS/ImageGen
- [x] Wait-until-idle when user is active (via arbiter)
- [x] Sequential execution queue
- [x] Safety controls (global pause, max tasks, provenance, agent limits)
- [x] App lifecycle: persist on terminate, restore + catch-up on launch
- [x] Missed-run policy: evaluate persisted `nextRunAt` before advancing, < 1h catch-up, >= 1h mark missed

### Phase 2: Heartbeat — MVP

- [x] `HEARTBEAT.md` file + default template + seeding
- [x] Heartbeat evaluation loop (read checklist → evaluate → act or OK)
- [x] Heartbeat session persistence (accumulating context)
- [x] Heartbeat configuration in settings

### Phase 3: Notifications — MVP

- [x] `NotificationService` — macOS notification center integration
- [ ] Runtime permission request
- [ ] Menu bar badge (unread count)
- [ ] TTS speech for results (`speakResult` flag)

### Phase 4: UI — MVP

- [ ] Sidebar "Scheduled" section
- [ ] Cron panel (task list, create/edit/delete/pause)
- [ ] Global pause toggle
- [ ] Run history view
- [ ] Background session viewer (read-only chat)
- [ ] Task detail/edit sheet

### Phase 5: Multi-Agent Council — Post-MVP

- [ ] Executor + Reviewer agent pattern
- [ ] Inter-agent conversation protocol
- [ ] Council deliberation logs
- [ ] User message injection into background sessions
- [ ] Agent persona configuration

### Phase 6: Agent Autonomy — Post-MVP

- [ ] Agent "free time" concept (agent does what it wants during idle)
- [ ] Sub-agent invocation tool
- [ ] Agent-configurable personas
- [ ] Conversation context sharing (background agent reads user chat, opt-in)

---

## 14. Open Questions

1. **Interruption policy**: Currently "wait until idle" (like Claude Code). Should we add priority levels where urgent tasks can interrupt? Deferred — start with wait-until-idle.

2. **Agent freedom time**: The concept of "free time for the agent to do whatever it wants" needs more definition. What does the agent's system prompt look like for free-time sessions? Should it have access to the user's files? Deferred to post-MVP.

3. **Multi-agent protocol**: How do executor and reviewer agents communicate? Options:
   - (a) Sequential: executor runs, reviewer evaluates output
   - (b) Conversational: agents take turns in a shared session
   - (c) Structured: executor produces JSON output, reviewer grades it
     Current lean: (b) conversational, since we already have the conversation primitives. Deferred to post-MVP.

4. **Sub-agent invocation tool**: Future tool that lets an agent spawn another agent for a sub-task. Design TBD — depends on agent configuration system. Deferred to post-MVP.

5. **Task templates**: Should we ship default task templates (morning planning, evening review, etc.)? Or let the agent discover and create them organically? Lean: provide a few seed templates via the personal-assistant package, but let the agent evolve them.

6. **Resource limits**: One background agent at a time for now (sequential queue), since we have one model and one GPU. Future: parallel execution if multiple models or GPUs become available.

7. **Conversation context sharing**: Should background agents be able to read the user's chat history for context? This would help them understand what the user is working on. Privacy implication: the user should opt into this. Deferred to post-MVP.

---

## 15. Reference: Industry Patterns

| System                   | Mechanism                 | Persistence            | Multi-Agent                 | Proactive Notify               |
| ------------------------ | ------------------------- | ---------------------- | --------------------------- | ------------------------------ |
| Claude Code              | /loop + Desktop cron      | Session / OS-level     | No                          | Hooks only                     |
| OpenClaw                 | Heartbeat + Cron + Hooks  | Self-hosted persistent | No                          | Yes (messaging channels)       |
| Devin                    | Autonomous task execution | Cloud                  | Sub-agent dispatch          | GitHub comments                |
| Codex                    | Cloud sandbox jobs        | Cloud                  | Parallel agents             | Webhook                        |
| **Tesseract (MVP)**      | **Heartbeat + Cron**      | **Local persistent**   | **No (single agent)**       | **Notification + TTS + Badge** |
| **Tesseract (Post-MVP)** | **+ Agent Council**       | **+ Context sharing**  | **Yes (executor/reviewer)** | **+ Free time autonomy**       |

Tesseract's approach is unique: fully local, persistent background sessions with full transparency, TTS proactive speech, and (post-MVP) multi-agent council with agent "free time." No other system combines all of these.
