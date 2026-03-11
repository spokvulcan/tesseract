# Epic 7: Cleanup and Verification

> Clean up transitional code, update benchmarks, update documentation, and verify everything works end-to-end. No data migration — the app starts clean.

## Prerequisites

- Epic 6 (Session Integration) completed and smoke-tested

---

## Task 7.1: Update Benchmark Suite

**Files**: Modify files in `Features/Agent/Benchmark/`

**Current**: Benchmarks test 7 scenarios (S1-S7) using domain tools (memory_save, task_create, goal_create, etc.)

**New**: Benchmarks must test the file-based workflow instead.

**Changes to BenchmarkScenario.swift**:

Replace domain tool expectations with file tool expectations:

| Old Scenario | Old Expected Tool | New Expected Behavior |
|-------------|-------------------|----------------------|
| S1: Save memory | `memory_save` | `read` memories.md → `edit`/`write` to append |
| S2: Create task | `task_create` | `read` tasks.md → `edit`/`write` to append |
| S3: List tasks | `task_list` | `read` tasks.md → summarize |
| S4: Complete task | `task_complete` | `read` tasks.md → `edit` checkbox |
| S5: Create goal | `goal_create` | **Remove scenario** — goal tracking intentionally dropped (see PLAN.md scope decision) |
| S6: Multi-step | Various domain tools | Various file tools |
| S7: Conversation | `respond` or text | Text response (no respond tool) |

**Changes to BenchmarkEvaluator.swift**:
- Update tool call expectations
- Check file contents instead of tool return strings
- Verify the agent reads before writing (skill compliance)

**Changes to BenchmarkRunner.swift**:
- Use `Agent` instead of `AgentRunner`
- Set up sandbox, tools, skills for benchmark context
- Use benchmark-specific agent root (temporary directory)

**Acceptance criteria**:
- Scenarios S1-S4, S6, S7 updated for file-based workflow
- Scenario S5 (goals) removed — feature intentionally dropped
- BenchmarkRunner uses new Agent class
- Benchmarks run without crashes (pass rate may need tuning)
- Build succeeds

---

## Task 7.2: Remove Transitional Code

**Files**: Various cleanup

1. **Clean up `AgentChatMessage.swift`**:
   - Remove `AgentMessageProtocol` conformance (it's now purely a UI display type)
   - Keep `init(from: any AgentMessageProtocol)` conversion (used by coordinator)
   - Remove any dead code or unused properties

2. **Remove old `AgentRunner.swift`** if not already deleted in Task 6.8

3. **Remove `AgentChatFormatter.swift`** if not already deleted in Task 6.6

4. **Clean up `#if DEBUG` test extension** from Task 4.5 (or keep for development)

5. **Remove any `// TODO` comments referencing the rewrite**

6. **Verify no dead imports** — check for imports of removed modules

**Acceptance criteria**:
- No dead code
- No unused imports
- No TODO comments referencing the redesign
- Build succeeds with zero warnings (excluding pre-existing ones)

---

## Task 7.3: Update CLAUDE.md Documentation

**File**: Modify `CLAUDE.md`

**Update these sections**:

1. **Architecture section**: Update to reflect new structure
   ```
   Features/Agent/
   ├── Core/              # Foundation types, loop, agent, messages, persistence
   ├── Context/           # Context loading, skills, prompt assembly, compaction
   ├── Extensions/        # Extension protocol, host, runner
   ├── Packages/          # Package manifest, registry, built-in packages
   ├── Tools/
   │   ├── BuiltIn/       # read, write, edit, list, path sandbox
   │   └── ToolCallParser.swift
   ├── Views/             # SwiftUI chat views
   ├── Benchmark/         # Benchmark suite
   └── AgentNotch/        # Dynamic Island overlay
   ```

2. **Agent Architecture section**: Update description
   - Replace "16 tools" with "4 built-in tools + package/extension tools"
   - Replace "max 5 rounds" with "no round cap"
   - Replace "observation masking" with "compaction via transformContext"
   - Replace "contextLimit = 60" with "full history managed by compaction"
   - Add: extension system, skill system, package system descriptions

3. **Tools section**: Update tool list
   - Built-in: read, write, edit, list
   - Personal assistant (via skills): memory, tasks, notes

4. **Context management section**:
   - Remove observation masking description
   - Add compaction description
   - Add context file loading (AGENTS.md, CLAUDE.md, SYSTEM.md, APPEND_SYSTEM.md)

**Acceptance criteria**:
- Documentation accurately reflects new architecture
- No references to removed features (domain tools, observation masking, contextLimit, respond tool)

---

## Task 7.4: Final End-to-End Verification

**No code changes**. Comprehensive testing:

1. **Fresh start**:
   - Delete `~/Library/Application Support/Tesseract Agent/agent/`
   - Launch app → verify seed files created (memories.md, tasks.md)
   - Verify skills discovered (check logs for skill names)
   - Verify system prompt assembled correctly (check debug logs)

2. **Full conversation flow**:
   - New conversation
   - "Remember that my name is Alex"
   - Verify agent reads memories.md (or discovers it doesn't exist), uses write/edit
   - New conversation
   - "What's my name?"
   - Verify agent reads memories.md, answers correctly

3. **Long session**:
   - Send 20+ messages
   - Verify compaction triggers (check logs for "Compaction" entries)
   - Verify conversation continues normally after compaction

4. **Voice round-trip**:
   - Control+Space → speak → release
   - Verify transcription → agent response → auto-speak (if enabled)

5. **Benchmark suite**:
   - Run benchmarks via `--benchmark`
   - Verify at least 4/6 pass (some may need prompt tuning)

6. **Conversation persistence**:
   - Have a conversation
   - Force quit app
   - Relaunch → verify conversation loads correctly

**Acceptance criteria**:
- All scenarios pass
- No crashes or data loss
- Performance comparable to old architecture

---

## Summary

After this epic, the redesign is complete. The agent runs on the Pi-aligned architecture with:
- Minimal core (4 built-in tools, extensible loop)
- Personal assistant as a first-party package
- Skills for memory/task/note workflows
- Compaction instead of observation masking
- Extension system ready for future capabilities
- Clean codebase with no transitional code

**New files created**: 0
**Files modified**: ~3 (Benchmark files, CLAUDE.md)
**Files deleted**: 0-3 (transitional code cleanup)
