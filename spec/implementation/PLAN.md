# Agent Redesign: Implementation Plan

> Master plan for implementing the Pi-aligned agent redesign described in `spec/AGENT_REDESIGN.md`.

## Strategy

The rewrite touches ~3,500 lines of core agent code (AgentRunner, AgentCoordinator, SystemPromptBuilder, ToolRegistry, AgentTool, AgentChatMessage, AgentConversationStore) and removes ~1,100 lines of domain tools. The strategy is **additive-first**: build new components alongside old ones, then swap in a single integration epic.

This means the app continues to build and run after every epic. The old agent loop works until Epic 6 cuts over to the new one.

## Epic Overview

| Epic | Name | Tasks | New Files | Modified Files | Deleted Files |
|------|------|-------|-----------|----------------|---------------|
| 0 | [Foundation Types](EPIC_0_FOUNDATION_TYPES.md) | 7 | 6 | 0 | 0 |
| 1 | [Built-In Tools](EPIC_1_BUILT_IN_TOOLS.md) | 6 | 5 | 0 | 0 |
| 2 | [Core Loop](EPIC_2_CORE_LOOP.md) | 7 | 4 | 1 | 0 |
| 3 | [Context and Prompt](EPIC_3_CONTEXT_AND_PROMPT.md) | 6 | 4 | 0 | 0 |
| 4 | [Extension System](EPIC_4_EXTENSION_SYSTEM.md) | 5 | 4 | 0 | 0 |
| 5 | [Package System](EPIC_5_PACKAGE_SYSTEM.md) | 5 | ~10 | 0 | 0 |
| 6 | [Session Integration](EPIC_6_SESSION_INTEGRATION.md) | 10 | 0 | 7 | 10 |
| 7 | [Cleanup and Verification](EPIC_7_CLEANUP_AND_VERIFICATION.md) | 4 | 0 | 3 | 0 |
| 8 | [Bash Tool & Project Access](EPIC_8_BASH_AND_PROJECT_ACCESS.md) | 10 | 4 | 5+2 ent | 0 |

**Total: ~60 tasks, ~39 new files, ~17 modified files, ~10 deleted files**

## Dependency Graph

```
Epic 0: Foundation Types
   │
   ├──→ Epic 1: Built-In Tools
   │
   ├──→ Epic 2: Core Loop
   │       │
   │       └──→ Epic 3: Context and Prompt ──┐
   │                                          │
   ├──→ Epic 4: Extension System              │
   │       │                                  │
   │       └──→ Epic 5: Package System ←──────┘
   │
   └──────────→ Epic 6: Session Integration ←── ALL of 1-5
                   │
                   └──→ Epic 7: Cleanup and Verification
                   │
                   └──→ Epic 8: Bash Tool & Project Access ←── 1, 2, 3
```

**Parallelism**: Epics 1, 2, and 4 can be developed in parallel after Epic 0. Epic 3 depends on Epic 2 (uses `CompactionSummaryMessage` and `convertToLlm`). Epic 5 depends on both Epic 3 and Epic 4. Epic 6 integrates everything. Epic 7 is cleanup. Epic 8 (bash + project access) can start after Epics 1-3 and is independent of 6-7.

## File Map

### New Core Files (Features/Agent/Core/)

```
Features/Agent/Core/
├── AgentMessage.swift          ← E0: Protocol hierarchy + CoreMessage enum
├── AgentToolDefinition.swift   ← E0: Concrete tool struct (closures)
├── AgentToolResult.swift       ← E0: Structured result with content blocks
├── AgentEvent.swift            ← E0: Event enum for loop → UI
├── AgentLoopConfig.swift       ← E0: Loop configuration
├── AgentState.swift            ← E0: Observable agent state
├── AgentLoop.swift             ← E2: Pi double-loop implementation
├── Agent.swift                 ← E2: Stateful wrapper (queues, abort, subscriptions)
├── MessageConversion.swift     ← E2: convertToLlm + message construction
└── MessagePersistence.swift    ← E2: Tagged JSON encoding/decoding (MessageCodecRegistry)
```

### New Tool Files (Features/Agent/Tools/BuiltIn/)

```
Features/Agent/Tools/BuiltIn/
├── PathSandbox.swift           ← E1: Sandboxed path resolution (E8: multi-root)
├── ReadTool.swift              ← E1: File read with truncation
├── WriteTool.swift             ← E1: File write with mkdir
├── EditTool.swift              ← E1: Exact-match edit with fuzzy fallback
├── ListTool.swift              ← E1: Directory listing
├── BashTool.swift              ← E8: Bash command execution
└── ANSIStripper.swift          ← E8: ANSI escape code removal
```

### New Project Access Files (Features/Agent/ProjectAccess/)

```
Features/Agent/ProjectAccess/
├── ProjectAccessManager.swift  ← E8: Security-scoped bookmarks + NSOpenPanel
└── ProjectAccessView.swift     ← E8: UI for managing granted directories
```

### New Context Files (Features/Agent/Context/)

```
Features/Agent/Context/
├── ContextLoader.swift         ← E3: AGENTS.md, CLAUDE.md, SYSTEM.md loading
├── ContextManager.swift        ← E3: Token estimation + compaction
├── SkillRegistry.swift         ← E3: Skill discovery + XML formatting
└── SystemPromptAssembler.swift ← E3: Pi-style prompt assembly
```

### New Extension Files (Features/Agent/Extensions/)

```
Features/Agent/Extensions/
├── AgentExtension.swift        ← E4: Extension protocol
├── ExtensionHost.swift         ← E4: Registration + tool aggregation
├── ExtensionRunner.swift       ← E4: Event dispatch
└── ExtensionContext.swift      ← E4: Context passed to extension handlers
```

### New Package Files (Features/Agent/Packages/)

```
Features/Agent/Packages/
├── AgentPackage.swift          ← E5: Manifest model
├── PackageRegistry.swift       ← E5: Manifest loading + resource collection
└── BuiltIn/
    └── PersonalAssistantPackage.swift  ← E5: First-party package registration
```

### New Package Resources

```
Resources/AgentPackages/personal-assistant/
├── package.json                ← E5: Package manifest
├── skills/
│   ├── memory/SKILL.md         ← E5: Memory management skill
│   ├── tasks/SKILL.md          ← E5: Task management skill
│   └── notes/SKILL.md          ← E5: Note capture skill
├── prompts/
│   └── APPEND_SYSTEM.md        ← E5: Assistant personality prompt
└── data/
    ├── memories.md             ← E5: Seed data
    └── tasks.md                ← E5: Seed data
```

### Modified Files (Epic 6)

| File | Change |
|------|--------|
| `AgentRunner.swift` | Replace with new `AgentLoop.swift` integration |
| `AgentCoordinator.swift` | Thin down to UI bridge, remove domain logic |
| `SystemPromptBuilder.swift` | Replace with `SystemPromptAssembler.swift` |
| `ToolRegistry.swift` | Rewrite for built-in + extension tool aggregation |
| `AgentChatMessage.swift` | Keep as UI display type, add conversion from protocol types |
| `AgentConversation.swift` | Rewrite for protocol-backed messages + new title logic |
| `AgentConversationStore.swift` | Tagged JSON encoding via MessageCodecRegistry |

### Deleted Files (Epic 6)

| File | Reason |
|------|--------|
| `Tools/MemoryTools.swift` | → personal-assistant file workflow |
| `Tools/TaskTools.swift` | → personal-assistant skill |
| `Tools/GoalTools.swift` | → personal-assistant skill |
| `Tools/HabitTools.swift` | → personal-assistant extension tool |
| `Tools/MoodTools.swift` | → personal-assistant skill |
| `Tools/ReminderTools.swift` | → personal-assistant extension tool |
| `Tools/RespondTool.swift` | Not in Pi-shaped core |
| `Tools/AgentDataStore.swift` | → personal-assistant package |
| `Tools/DateParsingUtility.swift` | → personal-assistant package utility |
| `AgentChatFormatter.swift` | Replaced by MessageConversion.swift |

## Intentional Feature Scope Changes

This redesign intentionally removes several domain features from the shipped product. This is a **product decision**, not an oversight:

| Removed Feature | Reason | Recovery Path |
|----------------|--------|---------------|
| Goal tracking (goal_create, goal_list, goal_update) | Not part of Pi-aligned minimal core. Low usage during development. | Can be added as a skill or extension tool later. |
| Habit tracking (habit_create, habit_log, habit_status) | Complex streak logic doesn't fit file-only workflows. | Can return as an extension tool in the personal-assistant package. |
| Mood logging (mood_log, mood_list) | Niche feature, low priority. | Trivial to add as a skill (write to moods.md). |
| Reminder system (reminder_set) | Requires UNUserNotificationCenter side effect — a package extension tool, not a core tool. | Planned as a post-MVP extension tool in the personal-assistant package. |
| `respond` tool | Pi-shaped core doesn't use a final-answer interception tool. | Not needed — the model responds naturally when it has no tool calls. |

The redesigned app ships with: **memory management, task management, and note capture** via skills + file tools. This covers the primary personal assistant use cases.

## No Data Migration

The app is in active development with no external users. Existing JSON data files (memories.json, tasks.json, goals.json, habits.json, moods.json, reminders.json) will **not** be migrated. The cutover starts clean:

- Old JSON files are ignored (not deleted, but not read)
- New seed files (memories.md, tasks.md) are created fresh by the personal-assistant package
- Old conversation files are also not migrated — new format starts from scratch

This eliminates migration complexity and timing issues entirely.

## Testing Strategy

Each epic includes build verification. The app must build after every task.

- **Epics 0-5**: Additive only. No existing behavior changes. Build verification sufficient.
- **Epic 6**: Integration. This is where old → new swap happens. Smoke test all user flows.
- **Epic 7**: Cleanup. Update benchmarks, verify voice I/O, update docs.

## Risk Areas

1. **AgentEngine/LLMActor interface changes** (Epic 2) — The new loop needs streaming generation that returns structured events. The existing `AgentEngine.generate()` returns `AsyncThrowingStream<AgentGeneration>`, which is close but needs adaptation for the new `AgentEvent` model.

2. **Compaction with local models** (Epic 3) — Using the same model for summarization ties up the GPU. May need to run compaction between turns, not during generation.

3. **ToolCallParser compatibility** (Epic 2) — The existing parser handles `<tool_call>` XML tags. Pi uses server-side JSON tool calling. Since Tesseract uses local models with XML-based tool calling, the parser stays but the integration point changes.

4. **Benchmark suite** (Epic 7) — All 7 scenarios reference domain tools (memory, task, goal, etc.). After the rewrite, benchmarks need to test file-based workflows instead.

5. **Process in App Sandbox** (Epic 8) — Child processes inherit the sandbox. Some commands may fail unexpectedly (e.g., `ps`, `open`, Homebrew tools). The system prompt warns the LLM, but real-world testing will reveal edge cases.

6. **Security-scoped bookmark staleness** (Epic 8) — Bookmarks can become stale after macOS updates or disk changes. Implementation handles re-creation, but edge cases may exist.
