# Agent Redesign: Pi-Aligned Core, Personal Assistant as Package

> This revision replaces the earlier draft. The goal is no longer "Pi-inspired data tools for a built-in assistant." The goal is a Pi-shaped core agent, with personal assistant capabilities implemented above the core through extensions, skills, and packaged resources.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Pi Philosophy Applied to Tesseract](#2-pi-philosophy-applied-to-tesseract)
3. [Core Principles](#3-core-principles)
4. [MVP Scope](#4-mvp-scope)
5. [Architecture Overview](#5-architecture-overview)
6. [Core Runtime](#6-core-runtime)
7. [Built-In Tool Surface](#7-built-in-tool-surface)
8. [Extension, Skill, and Package Model](#8-extension-skill-and-package-model)
9. [Personal Assistant Package](#9-personal-assistant-package)
10. [Context and System Prompt](#10-context-and-system-prompt)
11. [Migration Path](#11-migration-path)
12. [File-by-File Breakdown](#12-file-by-file-breakdown)

---

## 1. Executive Summary

The redesigned Tesseract agent should copy Pi's philosophy, not just its three file tools.

That means:

- The **core agent stays minimal**.
- The **core agent is aggressively extensible**.
- The **core agent does not bake in personal assistant product decisions**.
- Product-specific behaviors like memories, tasks, notes, and reminders are implemented as **first-party extensions, skills, and packages**.

In concrete terms:

1. The core ships a Pi-like loop, prompt structure, context loading, and minimal built-in tools.
2. The core exposes a stable extension API so new capabilities can be added without rewriting core internals.
3. Tesseract's "personal assistant" becomes a bundled package that uses the extension API and skill files.

This keeps the harness generic and durable, while letting Tesseract ship a curated assistant experience on top.

---

## 2. Pi Philosophy Applied to Tesseract

Pi's philosophy is not "everything is markdown files."

Pi's philosophy is:

- Keep the core small.
- Do not dictate the user's workflow.
- Push specialized behavior out of the core.
- Make the harness easy to reshape with extensions, skills, prompt files, and packages.

That is the architecture we adopt.

### Pi source references

- `packages/coding-agent/README.md` - high-level philosophy, customization model, context files, skills, extensions, packages
- `packages/coding-agent/docs/extensions.md` - extension capabilities and extension-as-feature model
- `packages/coding-agent/docs/skills.md` - skill loading and progressive disclosure model
- `packages/coding-agent/docs/packages.md` - package bundling model for extensions, skills, prompts, and themes

### What This Means for Tesseract

The old draft still treated "assistant features" as the center of the architecture. It removed the old domain tools, but it still assumed the core itself was fundamentally a memory/task assistant.

This revision changes that:

- **Core Tesseract Agent** = a local, tool-calling agent harness.
- **Personal Assistant Package** = a first-party package layered on top of the harness.

So the architecture becomes:

| Layer | Responsibility |
|---|---|
| **Core harness** | LLM loop, streaming, context, built-in file tools, extension loading, skill loading |
| **First-party assistant package** | Memories, tasks, notes, reminders, prompt add-ons, assistant-specific tools |
| **Future packages** | Research pack, coding pack, journaling pack, CRM pack, etc. |

### Explicit Pi-Style Non-Goals for Core

The core should intentionally avoid product logic that Pi would push out:

- No built-in task system in the core
- No built-in memory schema in the core
- No built-in plan mode in the core
- No built-in sub-agent orchestration in the core
- No built-in permission flow in the core

If Tesseract wants those behaviors, they belong in packages or extensions.

### Pi source references

- `packages/coding-agent/README.md` - "No plan mode", "No built-in to-dos", and the general "push it out of core" stance
- `packages/coding-agent/examples/extensions/plan-mode/index.ts` - plan mode implemented as an extension, not core
- `packages/coding-agent/examples/extensions/todo.ts` - to-dos implemented as an extension tool/command package

---

## 3. Core Principles

1. **Minimal core, opinionated package.** The app can ship a polished assistant, but the assistant is not the core architecture.
2. **Aggressively extensible.** New tools, commands, prompts, and behaviors must be addable without editing the core loop.
3. **Files first, not files only.** File tools are the default substrate, but the real principle is that the harness stays generic.
4. **No stable IDs required for markdown workflows.** The personal assistant package may use exact-text edits and line-based manipulation. We do not introduce hidden IDs.
5. **No `maxToolRounds`.** The loop runs until the model stops, the user aborts, or cancellation propagates.
6. **Sandbox-native adaptation.** We mirror Pi's shape, but adapt where macOS App Sandbox changes the mechanism.
7. **First-class package seams.** "Assistant" features should be removable, replaceable, and swappable.

---

## 4. MVP Scope

### 4.1 Core Harness: In

| Feature | Why |
|---|---|
| Pi-like agent loop | The harness should behave like Pi at the core interaction level |
| Built-in file tools | Minimal useful default tool surface |
| Extension host | Required for Pi-style extensibility |
| Skill loader | Required for on-demand instruction packs |
| Package manifest loader | Lets us ship first-party capability packs cleanly |
| Context files (`AGENTS.md` / `CLAUDE.md`) | Matches Pi's project-context loading model |
| Replace/append system prompt files | Matches Pi's prompt override model |
| Automatic compaction | Needed for local small-context models |

### 4.2 Core Harness: Out

| Feature | Why deferred / excluded |
|---|---|
| Built-in memory/task/goal/habit/reminder logic | Belongs in packages, not core |
| Built-in plan mode | Pi pushes this out to extensions |
| Built-in sub-agents | Same reason |
| Permission popup framework | Can be added later as an extension hook |
| Background shell | App Sandbox constraint |

### 4.3 First-Party Package: In

The bundled **Personal Assistant Package** ships with the app and provides:

- `memories.md`
- `tasks.md`
- note capture workflows
- optional reminder integrations
- assistant-specific skills and prompt add-ons
- assistant-specific custom tools when file-only workflows are insufficient

That gives Tesseract the product experience you want, while preserving a Pi-like core.

---

## 5. Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Tesseract App Shell                     │
│         (UI, voice I/O, settings, lifecycle, overlays)     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Agent Harness                      │
│                                                             │
│  AgentCoordinator  -> UI/session state only                 │
│  AgentRunner       -> Pi-like generate -> tools -> repeat   │
│  AgentEngine       -> @MainActor wrapper                    │
│  LLMActor          -> MLX inference actor                   │
│  ToolRegistry      -> built-in tools + extension tools      │
│  ContextManager    -> token tracking + compaction           │
│  ContextLoader     -> AGENTS/CLAUDE + prompt files          │
│  SkillRegistry     -> discover skill metadata               │
│  ExtensionHost     -> register tools/hooks/providers        │
│  PackageRegistry   -> enable bundled/local packages         │
└────────────────────────────┬────────────────────────────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
     Built-in Tools      Skill Files      Extensions / Packages
     read/write/edit     Markdown         Swift modules + manifests
     list (sandboxed)    on-demand        first-party by default
```

### Layer Responsibilities

| Component | Responsibility |
|---|---|
| **AgentCoordinator** | UI state, streaming state, voice routing, cancellation |
| **AgentRunner** | Minimal Pi-like loop, no product logic |
| **ToolRegistry** | Aggregates built-ins and extension-contributed tools |
| **ExtensionHost** | Lifecycle for registering tools, hooks, and commands |
| **SkillRegistry** | Discovers skills and exposes them to prompt construction |
| **PackageRegistry** | Enables/disables first-party or local packages |
| **ContextLoader** | Loads `AGENTS.md`, `CLAUDE.md`, `SYSTEM.md`, `APPEND_SYSTEM.md` |

The key change is that product logic no longer lives in `AgentRunner`, `ToolRegistry`, or `SystemPromptBuilder`.

### Pi source references

- `packages/coding-agent/src/core/resource-loader.ts` - central loading pipeline for extensions, skills, prompts, themes, and context files
- `packages/coding-agent/src/core/extensions/runner.ts` - extension lifecycle and aggregation of extension-contributed tools/commands
- `packages/coding-agent/src/core/package-manager.ts` - package resource resolution and precedence rules

---

## 6. Core Runtime

### 6.1 AgentRunner

The loop should follow Pi's structure:

1. Start with current context.
2. Generate assistant output.
3. If there are tool calls, execute them and append tool results.
4. Re-generate.
5. If there are no tool calls, complete.
6. If queued steering or follow-up messages exist, continue.

No hard round cap is imposed.

### Core Loop Rules

- No `maxToolRounds`
- No built-in `respond` tool
- No domain-specific retry logic
- Cancellation must abort generation and pending tool execution
- Queued steering and follow-up messages remain supported
- Tool execution remains visible to the UI

This is intentionally closer to Pi's default stance: keep the loop simple, and solve special behavior with prompt quality, extensions, or package-specific hooks.

#### Pi source references

- `packages/agent/src/agent-loop.ts` - core generate -> tool -> continue loop
- `packages/coding-agent/README.md` - message queue semantics and interactive session behavior
- `packages/coding-agent/src/core/agent-session.ts` - higher-level session orchestration around the core agent loop

### 6.2 ContextManager

Unlike Pi (which uses cloud models with 200K+ context), Tesseract runs local models. The current target model supports a 120K context window, which is substantial but finite. Automatic compaction remains part of the core to handle long-running sessions gracefully.

This is an intentional platform adaptation, not a philosophical deviation.

Rules:

- Track estimated context usage continuously
- Compact proactively before overflow
- Preserve recent turns
- Insert a summary message for older turns
- Keep compaction pluggable so packages/extensions can customize the strategy later

#### Pi source references

- `packages/coding-agent/src/core/compaction/compaction.ts` - default compaction logic, token estimation, and extension hook points
- `packages/coding-agent/examples/extensions/custom-compaction.ts` - compaction customized via extension instead of hardcoding every policy in core

### 6.3 AgentCoordinator

`AgentCoordinator` becomes thinner:

- It should not know about memories/tasks/goals/etc.
- It should not own domain-specific tool orchestration
- It should only own UI state, conversation state, and bridge events to voice/UI
- **It must not truncate history before passing it to the runner.** The current `contextLimit = 60` hard clamp and observation masking (replacing tool results beyond the 20 most recent) must be removed. History management is the responsibility of `transformContext` and the compaction system — if the coordinator truncates first, compaction never sees the full history and cannot generate accurate summaries.

#### Pi source references

- `packages/coding-agent/src/core/agent-session.ts` - session/UI orchestration separated from the lower-level loop

---

## 7. Built-In Tool Surface

### 7.1 Default Built-Ins

The core should ship a minimal default tool set:

1. `read`
2. `write`
3. `edit`
4. `list`

### Why `list` Exists

Pi uses `bash` for discovery (`ls`, `find`, `grep`). Tesseract cannot safely ship unrestricted `bash` in the App Sandbox.

So the closest sandbox-safe equivalent is a read-only discovery tool.

Without a discovery primitive, the model can create files but cannot reliably rediscover them later.

`list` is therefore a platform adaptation of Pi's discovery affordance, not a philosophical departure.

#### Pi source references

- `packages/coding-agent/src/core/tools/read.ts` - built-in read tool
- `packages/coding-agent/src/core/tools/write.ts` - built-in write tool
- `packages/coding-agent/src/core/tools/edit.ts` - built-in edit tool
- `packages/coding-agent/src/core/tools/ls.ts` - built-in directory discovery tool
- `packages/coding-agent/src/core/tools/find.ts` - built-in file discovery tool
- `packages/coding-agent/src/core/tools/grep.ts` - built-in content discovery tool
- `packages/coding-agent/src/core/tools/path-utils.ts` - path safety and normalization utilities used by file tools

### 7.2 Built-In Tool Definitions

### `read`

```text
read(path: string, offset?: int, limit?: int)
```

- Reads file contents from a sandboxed base directory
- Returns raw text with line numbers

### `write`

```text
write(path: string, content: string)
```

- Creates or overwrites a file
- Parent directories are created automatically

### `edit`

```text
edit(path: string, oldText: string, newText: string)
```

- Exact text replacement
- Fails on zero matches or multiple matches

### `list`

```text
list(path?: string, recursive?: bool, limit?: int)
```

- Lists files and directories relative to the sandbox root
- Optional recursive mode with a hard cap
- Read-only

### 7.3 Extension-Contributed Tools

The important rule is not "these four tools forever."

The important rule is:

- The built-in tool surface stays small.
- Additional tools come from extensions or packages.

That means reminders, calendar integration, notifications, contacts, or app intents do **not** belong in core. They are extension tools.

#### Pi source references

- `packages/coding-agent/src/core/tools/index.ts` - built-in tool registry and default tool surface
- `packages/coding-agent/examples/extensions/built-in-tool-renderer.ts` - extensions can replace or wrap built-in tool behavior

---

## 8. Extension, Skill, and Package Model

This is the center of the redesign.

### 8.1 Skills

Skills are markdown instruction packs, loaded on demand.

They are the Tesseract equivalent of Pi's skills:

- stored as files
- listed in the system prompt
- not fully inlined unless needed
- readable by the model using `read`

### Skill Locations

```text
~/Library/Application Support/Tesseract Agent/agent/skills/
{project}/.agents/skills/
```

### Skill Shape

```markdown
# Personal Assistant Memory Skill

Use this skill when the user wants you to remember, update, or forget personal facts.

## Workflow
1. Read memories.md first.
2. Prefer edit for precise changes.
3. Use write only for full rewrites.
```

#### Pi source references

- `packages/coding-agent/docs/skills.md` - skill structure, locations, and on-demand loading behavior
- `packages/coding-agent/src/core/skills.ts` - skill discovery, dedupe, and prompt formatting
- `packages/coding-agent/src/core/resource-loader.ts` - integration of resolved skill paths into the runtime load graph

### 8.2 Extensions

Extensions add behavior to the harness:

- custom tools
- tool hooks
- prompt additions
- slash commands
- custom compaction strategies
- permission gates
- OS integrations

### macOS Adaptation

Pi implements extensions as TypeScript modules.

Tesseract should keep the same architectural role, but use a macOS-native mechanism:

- **First-party extensions**: compiled Swift modules registered at app startup
- **Local package manifests**: enable/disable shipped extensions and attach skills/prompts/resources
- **Future signed extension bundles**: possible later, if code-signing/distribution story is worth it

The philosophy matches Pi. The implementation respects macOS app signing and sandbox constraints.

#### Pi source references

- `packages/coding-agent/docs/extensions.md` - extension API and supported behaviors
- `packages/coding-agent/src/core/extensions/types.ts` - extension contracts, tool definitions, events, and hook return types
- `packages/coding-agent/src/core/extensions/loader.ts` - extension discovery rules and manifest-driven entry points
- `packages/coding-agent/src/core/extensions/runner.ts` - extension lifecycle and event dispatch
- `packages/coding-agent/src/core/extensions/wrapper.ts` - interception of tool calls and tool results
- `packages/coding-agent/src/core/resource-loader.ts` - extension loading, merging, conflict handling, and runtime assembly

### 8.3 Packages

Packages are capability bundles.

A package can contain:

- skill files
- prompt templates
- prompt appenders
- configuration
- references to bundled extension modules
- starter data files

### Package Examples

- `personal-assistant`
- `journaling`
- `research`
- `coding`

Tesseract should ship `personal-assistant` as the default first-party package.

### 8.4 Package Manifest

Example:

```json
{
  "name": "personal-assistant",
  "enabled": true,
  "skills": [
    "skills/memory/SKILL.md",
    "skills/tasks/SKILL.md"
  ],
  "promptAppendFiles": [
    "prompts/assistant/APPEND_SYSTEM.md"
  ],
  "extensions": [
    "PersonalAssistantExtension"
  ],
  "seedFiles": [
    "data/memories.md",
    "data/tasks.md"
  ]
}
```

The package owns the assistant-specific resources. The core only knows how to load packages.

#### Pi source references

- `packages/coding-agent/docs/packages.md` - package manifest and conventional directory model
- `packages/coding-agent/src/core/package-manager.ts` - package manifest parsing, resource collection, and scope precedence
- `packages/coding-agent/src/core/resource-loader.ts` - resolved package resources flowing into extension/skill/prompt/theme loading
- `packages/coding-agent/examples/extensions/with-deps/package.json` - concrete example of package manifest declaring extension entries

---

## 9. Personal Assistant Package

The personal assistant experience is moved out of core and into a bundled first-party package.

### 9.1 Responsibilities

The `personal-assistant` package provides:

- assistant prompt additions
- memory/task/note skills
- seeded data files
- optional custom tools for reminders or system integrations
- assistant-specific formatting conventions

### 9.2 Default Data Directory

```text
~/Library/Application Support/Tesseract Agent/agent/
├── conversations/
├── skills/
├── packages/
│   └── personal-assistant/
│       ├── package.json
│       ├── skills/
│       ├── prompts/
│       └── data/
├── memories.md        # seeded by personal-assistant package
├── tasks.md           # seeded by personal-assistant package
└── notes/             # created by agent/package workflows
```

### 9.3 No Stable IDs

This package intentionally uses markdown as the source of truth and exact-text editing for updates.

We accept:

- duplicate handling by model judgment
- line-based reasoning when needed
- occasional rewrite of the full file when structure drifts

We do **not** add hidden IDs.

### 9.4 Assistant-Specific Tools

The package should prefer file workflows first.

If a feature requires true side effects, it may register custom tools such as:

- `create_reminder`
- `schedule_notification`
- `capture_current_app_context`

Those tools are package tools, not core tools.

### Pi source references

- `packages/coding-agent/docs/packages.md` - package structure and resource ownership
- `packages/coding-agent/examples/extensions/todo.ts` - stateful/product-specific behavior packaged above the core
- `packages/coding-agent/examples/extensions/question.ts` - interactive custom tool defined outside the core tool surface

---

## 10. Context and System Prompt

### 10.1 Prompt Structure

The default system prompt should follow Pi's structure more closely:

1. Role
2. Available tools
3. "You may have additional custom tools"
4. Guidelines
5. Documentation / local resource paths
6. Project context files
7. Skills listing
8. Current date/time
9. Current working directory (or agent working root)

### 10.2 Core Prompt

```text
You are an expert local assistant operating inside Tesseract, a tool-calling agent harness.
You help users by reading files, editing files, writing files, and using other tools provided by the current package or project.

Available tools:
- read: Read file contents
- write: Create or overwrite files
- edit: Make surgical edits to files (find exact text and replace)
- list: List files and directories

In addition to the tools above, you may have access to other custom tools depending on the current package or project.

Guidelines:
- Use list to discover files and directories
- Use read to examine files before editing
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- Be concise in your responses
- Show file paths clearly when working with files

Tesseract resources:
- Project context files are loaded from AGENTS.md or CLAUDE.md
- The system prompt may be replaced by SYSTEM.md or extended by APPEND_SYSTEM.md
- Skills may be available and should be used when relevant

Current date and time: {dateTime}
Current working directory: {agentRoot}
```

This keeps the core generic. The personal tone and assistant-specific guidance belong in the package append prompt.

#### Pi source references

- `packages/coding-agent/src/core/system-prompt.ts` - prompt composition order and tool/guideline sections
- `packages/coding-agent/README.md` - documented context file and prompt override behavior

### 10.3 Context Files

**Authoritative rule (resolves Pi vs. sandbox tension):**

Pi walks ancestor directories from filesystem root to cwd, collecting `AGENTS.md`/`CLAUDE.md` at each level. Tesseract runs inside the macOS App Sandbox, which prevents arbitrary filesystem walking. Tesseract therefore loads context files from a **fixed, enumerable set of locations** — not by walking the filesystem.

Context file sources (in load order):
1. **Global agent directory**: `~/Library/Application Support/Tesseract Agent/agent/AGENTS.md` (or `CLAUDE.md`)
2. **Package-provided context**: Context files bundled by enabled packages
3. **Extension-provided context**: Context files registered via `resources_discover` event

There is **no ancestor directory walking**. This is a deliberate sandbox adaptation of Pi's model.

Prompt override files:
- `SYSTEM.md` → replaces the entire default core prompt (loaded from global agent dir, or package-provided)
- `APPEND_SYSTEM.md` → appended to the prompt (loaded from global agent dir, or package-provided)

#### Pi source references

- `packages/coding-agent/README.md` - context file and prompt override behavior
- `packages/coding-agent/src/core/system-prompt.ts` - `loadProjectContextFiles()` and prompt assembly
- `packages/coding-agent/src/core/resource-loader.ts` - runtime lookup for `SYSTEM.md` and `APPEND_SYSTEM.md`

### 10.4 Skills in Prompt Construction

Skills should be listed in the prompt, but not fully injected by default.

The prompt should make it clear that the agent can read skill files when relevant.

This matches Pi better than hard-coding all assistant behavior into the base prompt.

#### Pi source references

- `packages/coding-agent/docs/skills.md` - "descriptions in context, full skill on demand" model
- `packages/coding-agent/src/core/skills.ts` - XML prompt formatting for available skills
- `packages/coding-agent/src/core/system-prompt.ts` - skill list appended into the system prompt only when `read` is available

---

## 11. Migration Path

### Phase 1: Establish the Core Harness

1. Simplify `AgentRunner` to the Pi-shaped loop
2. Remove built-in product assumptions from `SystemPromptBuilder`
3. Add `ContextLoader` for `AGENTS.md`, `CLAUDE.md`, `SYSTEM.md`, `APPEND_SYSTEM.md`
4. Add `SkillRegistry`
5. Add `ExtensionHost`
6. Add `PackageRegistry`
7. Replace domain-tool-first `ToolRegistry` with built-ins + extension registration

### Phase 2: Replace Core Domain Features with a First-Party Package

8. Remove memory/task/goal/habit/mood/reminder logic from the core tool registry
9. Create a bundled `personal-assistant` package
10. Move memory/task behavior into package skills and prompt appenders
11. Keep `memories.md` and `tasks.md` as package-owned seed files
12. Add assistant-specific extension tools only where file tools are insufficient

### Phase 3: Cleanup and Compatibility

13. Migrate existing JSON data into package-owned markdown files
14. Keep old JSON files as backups during migration
15. Update benchmarks to target the new core + package composition
16. Add package enable/disable controls in settings if needed

---

## 12. File-by-File Breakdown

### 12.1 New Core Files

| File | Purpose |
|---|---|
| `Features/Agent/ContextLoader.swift` | Loads AGENTS/CLAUDE, SYSTEM, APPEND_SYSTEM files |
| `Features/Agent/ContextManager.swift` | Token estimation + compaction |
| `Features/Agent/Extensions/ExtensionHost.swift` | Registers extension tools, hooks, commands |
| `Features/Agent/Extensions/AgentExtension.swift` | Protocol for native extensions |
| `Features/Agent/Skills/SkillRegistry.swift` | Discovers skill metadata |
| `Features/Agent/Packages/PackageRegistry.swift` | Loads package manifests and resources |
| `Features/Agent/Packages/AgentPackage.swift` | Manifest model |
| `Features/Agent/Tools/ReadTool.swift` | Built-in file tool |
| `Features/Agent/Tools/WriteTool.swift` | Built-in file tool |
| `Features/Agent/Tools/EditTool.swift` | Built-in file tool |
| `Features/Agent/Tools/ListTool.swift` | Built-in discovery tool |
| `Features/Agent/Tools/PathSandbox.swift` | Sandboxed path resolution |

### 12.2 New First-Party Package Files

| File | Purpose |
|---|---|
| `Features/Agent/Packages/BuiltIn/PersonalAssistantPackage.swift` | Registers the bundled personal assistant package |
| `Resources/AgentPackages/personal-assistant/package.json` | Package manifest |
| `Resources/AgentPackages/personal-assistant/skills/...` | Memory/task/note skills |
| `Resources/AgentPackages/personal-assistant/prompts/APPEND_SYSTEM.md` | Assistant-specific prompt addition |
| `Resources/AgentPackages/personal-assistant/data/memories.md` | Seed data |
| `Resources/AgentPackages/personal-assistant/data/tasks.md` | Seed data |

### 12.3 Modified Core Files

| File | Changes |
|---|---|
| `AgentRunner.swift` | Simplify to Pi-shaped loop, no max round cap |
| `AgentCoordinator.swift` | Keep UI/session/voice concerns only |
| `SystemPromptBuilder.swift` | Rebuild around Pi-style core prompt + loaded resources |
| `ToolRegistry.swift` | Built-ins + extension registration, not domain tools |
| `AgentTool.swift` | Support extension-contributed tool metadata cleanly |
| `AgentConversationStore.swift` | Keep persistence, but do not bake in assistant domain assumptions |

### 12.4 Removed from Core

| File | Why |
|---|---|
| `Tools/MemoryTools.swift` | Moved to personal-assistant package behavior |
| `Tools/TaskTools.swift` | Moved to personal-assistant package behavior |
| `Tools/GoalTools.swift` | Not core |
| `Tools/HabitTools.swift` | Not core |
| `Tools/MoodTools.swift` | Not core |
| `Tools/ReminderTools.swift` | If needed, becomes package extension |
| `Tools/RespondTool.swift` | Not needed in Pi-shaped core |
| `Tools/AgentDataStore.swift` | No longer the center of the architecture |

---

## Appendix: Final Architectural Stance

The correct Pi-aligned design for Tesseract is:

- **Core agent = generic harness**
- **Assistant experience = packaged capability layer**

The core should feel like Pi:

- small built-in surface
- file-first tools
- explicit prompt structure
- context files
- skills
- extensions
- packages
- minimal built-in assumptions

The product should still feel like Tesseract:

- local
- private
- voice-first
- polished
- useful on day one

Those are compatible as long as the assistant is treated as a first-party package, not the definition of the core.

---

# Part II: Technical Specification (Pi Source-Level Detail)

> The sections above describe *what* we build and *why*. The appendices below describe *exactly how*, derived line-by-line from the Pi source code at `/Users/owl/projects/pi-mono`. Every type, algorithm, and contract is documented at the level of detail needed to implement the Swift equivalent without referencing the Pi TypeScript source.

---

## A. Two-Layer Architecture

Pi separates the agent into two packages. Tesseract must preserve this separation.

### A.1 Layer 1: Core Agent Loop (`packages/agent/`)

Five files, ~1,000 LOC. Contains:

- `types.ts` — `AgentMessage`, `AgentTool`, `AgentContext`, `AgentLoopConfig`, `AgentEvent`, `AgentState`
- `agent-loop.ts` — `agentLoop()`, `agentLoopContinue()`, `runLoop()`, `streamAssistantResponse()`, `executeToolCalls()`
- `agent.ts` — `Agent` class: stateful wrapper with queues, subscriptions, abort control

This layer knows **nothing** about files, skills, extensions, compaction, sessions, or UI. It is a pure generate→tools→repeat machine.

### A.2 Layer 2: Coding Agent Session (`packages/coding-agent/src/core/`)

~30 files, ~13,500 LOC. Contains:

- `agent-session.ts` — `AgentSession`: lifecycle, compaction, retry, extension dispatch
- `system-prompt.ts` — prompt assembly
- `resource-loader.ts` — discovers extensions, skills, prompts, themes, context files
- `extensions/` — extension types, loader, runner, wrapper
- `compaction/` — token estimation, cut-point detection, summarization
- `skills.ts` — skill discovery and prompt formatting
- `tools/` — built-in tool implementations
- `messages.ts` — custom message types and `convertToLlm()`
- `session-manager.ts` — JSONL tree persistence
- `package-manager.ts` — package resolution

### A.3 Tesseract Mapping

| Pi Layer | Tesseract Equivalent |
|---|---|
| `packages/agent/` | `Features/Agent/Core/` — pure loop, types, Agent class |
| `packages/coding-agent/src/core/` | `Features/Agent/` — session, extensions, skills, tools, compaction |
| `packages/ai/` | `AgentEngine` + `LLMActor` (MLX inference, already exists) |

---

## B. Core Type System

### B.1 AgentMessage

Pi uses an extensible union type via TypeScript declaration merging — extensions add new cases to `CustomAgentMessages` and the union grows at compile time. Swift enums are **closed** — no case can be added across module boundaries. A single `enum AgentMessage` with fixed cases contradicts the stated extensibility goal.

**Solution**: Protocol-backed message model with a closed core enum + open custom protocol.

```swift
// ── Core message protocol ──────────────────────────────────────────
// Every message in the context conforms to this.
protocol AgentMessageProtocol: Sendable {
    /// How convertToLlm should handle this message.
    func toLLMMessage() -> LLMMessage?
}

// ── Core messages (closed, exhaustive match in the loop) ───────────
// These three are the only types the core loop pattern-matches on.
enum CoreMessage: AgentMessageProtocol, Sendable {
    case user(UserMessage)
    case assistant(AssistantMessage)
    case toolResult(ToolResultMessage)

    func toLLMMessage() -> LLMMessage? {
        switch self {
        case .user(let m):       return m.toLLMMessage()
        case .assistant(let m):  return m.toLLMMessage()
        case .toolResult(let m): return m.toLLMMessage()
        }
    }
}

// ── Custom message protocol (open, extensions add conformers) ──────
// Extensions and the session layer define new types that conform here.
// The core loop does NOT pattern-match on these — it only calls toLLMMessage().
protocol CustomAgentMessage: AgentMessageProtocol {
    /// Display label for UI rendering (e.g., "compaction_summary", "branch_summary")
    var customType: String { get }
}

// ── The context array uses the protocol ────────────────────────────
// AgentMessage is a type alias for the existential.
typealias AgentMessage = any AgentMessageProtocol

// ── Session-layer custom types (first-party, but outside core) ─────
struct CompactionSummaryMessage: CustomAgentMessage, Sendable {
    let customType = "compaction_summary"
    let summary: String
    let tokensBefore: Int
    let timestamp: Date

    func toLLMMessage() -> LLMMessage? {
        .user(content: "<summary>\n\(summary)\n</summary>")
    }
}

// Extensions add more conformers — no core changes needed.
```

**Why this works**:
- The core loop calls `toLLMMessage()` on every message — it doesn't need to know every concrete type.
- `CoreMessage` is a closed enum: the loop can `switch` exhaustively over user/assistant/toolResult.
- Extensions add new `CustomAgentMessage` conformers at compile time (compiled Swift modules).
- Pattern matching on custom types uses `if let msg = message as? CompactionSummaryMessage` where needed (UI rendering, conversation display).
- Persistence uses tagged JSON encoding with a type discriminator — see H.4 for the complete encoding/decoding design.
- `Sendable` is enforced by the protocol, not by `Any` casts.

**Critical design point**: The core loop (`agentLoop`) works with `[AgentMessage]` internally, but converts to LLM-compatible `[LLMMessage]` only at the LLM call boundary via `convertToLlm`. Custom message types exist in the context without confusing the LLM because each type's `toLLMMessage()` handles the conversion.

### B.2 AgentTool

Pi's `AgentTool<TParameters, TDetails>` is a plain TypeScript object — generics erase at runtime, so heterogeneous tool arrays just work. Swift associated types do not erase this way. Storing `[any AgentTool]` with associated types compiles, but you cannot invoke `execute` or aggregate tools without an explicit type-erasure wrapper.

**Solution**: Use a concrete struct with closures, not a protocol with associated types. This matches Pi's runtime shape: tools are registered as values with a JSON schema and an execute function. The generic types exist only at tool construction time, not at the storage/dispatch boundary.

```swift
// Concrete, non-generic tool type — storable in arrays, no type erasure needed
struct AgentTool: Sendable {
    let name: String
    let label: String                    // human-readable, for UI
    let description: String              // for LLM
    let parameterSchema: JSONSchema      // JSON Schema for parameter validation

    // Execute receives raw JSON args (decoded from LLM output).
    // onUpdate streams partial results to the UI during long-running tools.
    // Returns final structured result with content (for LLM) and optional details (for UI).
    let execute: @Sendable (
        _ toolCallId: String,
        _ argsJSON: [String: JSONValue],
        _ signal: CancellationToken?,
        _ onUpdate: ToolProgressCallback?
    ) async throws -> AgentToolResult
}

// Progress callback for streaming tool updates (e.g., bash output lines, download progress)
typealias ToolProgressCallback = @Sendable (AgentToolResult) -> Void

// Non-generic result type — details is type-erased but Sendable
struct AgentToolResult: Sendable {
    let content: [ContentBlock]              // text or image, sent to LLM
    let details: (any Sendable & Hashable)?  // tool-specific UI data, nil if none
}

enum ContentBlock: Sendable {
    case text(String)
    case image(data: Data, mimeType: String)
}
```

**Why closures over protocols**: Pi's `AgentTool` is already a plain object — `{ name, label, description, parameters, execute }`. The Swift struct mirrors this exactly. Each tool factory (e.g., `createReadTool(cwd:)`) returns a configured `AgentTool` with its typed parameters decoded inside the closure. The outer system only sees `AgentTool` — no generics leak.

**onUpdate flow**: The loop creates a `ToolProgressCallback` for each tool call that emits `toolExecutionUpdate` events. The tool calls `onUpdate` with partial `AgentToolResult` values during execution. The final return value is the complete result. Tools that don't stream (most file tools) simply ignore the callback.

**Key differences from current Tesseract**: Pi tools return structured `AgentToolResult` with separate `content` (for LLM) and `details` (for UI), not just a String. Pi tools also receive `toolCallId`, `signal`, and `onUpdate`. The concrete struct approach means `[AgentTool]` works directly — no `[any AgentTool]` or `AnyAgentTool` wrapper needed.

### B.3 AgentContext

```swift
struct AgentContext {
    var systemPrompt: String
    var messages: [AgentMessage]
    var tools: [AgentTool]?  // concrete struct, no type erasure needed
}
```

### B.4 AgentLoopConfig

```swift
struct AgentLoopConfig {
    // Model to use for generation
    let model: LLMModel

    // REQUIRED: Converts AgentMessage[] → LLM Message[] at the call boundary
    // Filters out custom types, converts bash executions to user text, etc.
    let convertToLlm: ([AgentMessage]) -> [LLMMessage]

    // OPTIONAL: Transform context before convertToLlm
    // Used for compaction, context pruning, injecting external context
    let transformContext: (([AgentMessage], CancellationToken?) async -> [AgentMessage])?

    // OPTIONAL: Returns steering messages (user interrupts mid-run)
    // Called after each tool execution
    let getSteeringMessages: (() async -> [AgentMessage])?

    // OPTIONAL: Returns follow-up messages (queued for after agent finishes)
    // Called when agent has no more tool calls
    let getFollowUpMessages: (() async -> [AgentMessage])?
}
```

### B.5 AgentState

```swift
struct AgentState {
    var systemPrompt: String
    var model: LLMModel
    var tools: [AgentTool]  // concrete struct array
    var messages: [AgentMessage]
    var isStreaming: Bool
    var streamMessage: AgentMessage?
    var pendingToolCalls: Set<String>
    var error: String?
}
```

---

## C. Agent Loop — Exact Behavior

This is the most critical section. The **loop structure** from `agent-loop.ts` lines 104–198 must be transferred exactly — the double-loop, steering/follow-up semantics, event emission order, and `transformContext`/`convertToLlm` hook points. **Tool behaviors** (section I) are adapted case-by-case where Pi assumes bash or other capabilities Tesseract does not have.

### C.1 Double-Loop Structure

```
agentLoop(prompts, context, config, signal)
  │
  ├─ Push prompts into context
  ├─ Emit agent_start, turn_start, message events for prompts
  │
  └─ runLoop(context, newMessages, config, signal, stream)
       │
       ├─ OUTER LOOP (while true):
       │   │
       │   ├─ INNER LOOP (while hasMoreToolCalls || pendingMessages.length > 0):
       │   │   │
       │   │   ├─ Emit turn_start (except first turn)
       │   │   │
       │   │   ├─ If pendingMessages (steering):
       │   │   │   └─ Push each into context, emit message events
       │   │   │
       │   │   ├─ streamAssistantResponse():
       │   │   │   ├─ Apply transformContext (if configured)
       │   │   │   ├─ Apply convertToLlm
       │   │   │   ├─ Call LLM with streaming
       │   │   │   ├─ Emit message_start/update/end events
       │   │   │   └─ Return final AssistantMessage
       │   │   │
       │   │   ├─ If error/aborted → emit turn_end, agent_end, return
       │   │   │
       │   │   ├─ Extract tool calls from response
       │   │   │
       │   │   ├─ If tool calls exist:
       │   │   │   ├─ executeToolCalls():
       │   │   │   │   ├─ For each tool call (sequential):
       │   │   │   │   │   ├─ Find tool by name
       │   │   │   │   │   ├─ Validate arguments against schema
       │   │   │   │   │   ├─ Emit toolExecutionStart(toolCallId, name, argsJSON)
       │   │   │   │   │   ├─ Execute tool(toolCallId, args, signal, onUpdate)
       │   │   │   │   │   │   onUpdate callback emits toolExecutionUpdate(result)
       │   │   │   │   │   ├─ Emit toolExecutionEnd(toolCallId, name, result, isError)
       │   │   │   │   │   ├─ Create ToolResultMessage
       │   │   │   │   │   ├─ Emit message_start/end for result
       │   │   │   │   │   │
       │   │   │   │   │   └─ Check getSteeringMessages():
       │   │   │   │   │       If steering arrived → skip remaining tools
       │   │   │   │   │       (skipped tools get "Skipped due to queued user message")
       │   │   │   │   │
       │   │   │   │   └─ Return { toolResults, steeringMessages }
       │   │   │   │
       │   │   │   └─ Push tool results into context
       │   │   │
       │   │   ├─ Emit turn_end
       │   │   │
       │   │   └─ Collect pendingMessages from steering (or poll getSteeringMessages)
       │   │
       │   ├─ Inner loop done (no tool calls, no steering)
       │   │
       │   ├─ Check getFollowUpMessages():
       │   │   If follow-ups → set as pendingMessages, continue outer loop
       │   │
       │   └─ No follow-ups → break outer loop
       │
       └─ Emit agent_end with all new messages
```

### C.2 Critical Loop Rules (from source)

1. **No round cap.** The outer loop runs until: no tool calls AND no steering AND no follow-ups.
2. **Steering skips remaining tools.** When `getSteeringMessages()` returns messages after a tool execution, remaining tool calls in the same assistant response are skipped with error messages.
3. **Follow-ups extend the session.** After the inner loop completes (no tool calls, no steering), `getFollowUpMessages()` is checked. If messages exist, they become the next `pendingMessages` and the outer loop continues.
4. **`transformContext` runs every turn.** Before each LLM call, context is passed through `transformContext` (for compaction), then `convertToLlm` (for message type conversion).
5. **Tools execute sequentially.** Pi does NOT parallelize tool calls. Each tool runs to completion before the next starts.
6. **Cancellation propagates via AbortSignal.** Every tool execution receives the signal. The loop checks `stopReason === "aborted"` after generation.

### C.3 agentLoopContinue

A separate entry point for retries — continues from existing context without adding new messages. Used when:
- Retrying after an error
- Resuming from queued steering/follow-up messages

Precondition: last message must NOT be role=assistant (would confuse the LLM).

### C.4 Differences from Current Tesseract AgentRunner

| Current Tesseract | Pi | Change Required |
|---|---|---|
| `maxToolRounds = 5` | No round cap | Remove cap |
| `respond` tool intercept | No respond tool | Remove respond tool |
| Within-turn dedup | Not in Pi core | Can keep as extension hook |
| Stall recovery (nudge + synthetic) | Not in Pi core | Move to extension or remove |
| Single-loop | Double-loop (steering + follow-ups) | Rewrite loop |
| Returns `AsyncThrowingStream<Event>` | Returns `EventStream<AgentEvent>` | Adapt to EventStream pattern |
| No `transformContext` | `transformContext` hook | Add hook |
| No `convertToLlm` | `convertToLlm` hook | Add hook |

---

## D. Event Model

### D.1 AgentEvent Types

All payloads must be `Sendable` under Swift 6. Tool args are raw JSON strings (always Sendable). Tool results use `[ContentBlock]` (the LLM-facing content). No `Any` anywhere.

```swift
enum AgentEvent: Sendable {
    // Agent lifecycle
    case agentStart
    case agentEnd(messages: [any AgentMessageProtocol & Sendable])

    // Turn lifecycle (a turn = one assistant response + tool calls/results)
    case turnStart
    case turnEnd(message: AssistantMessage, toolResults: [ToolResultMessage])

    // Message lifecycle
    case messageStart(message: any AgentMessageProtocol & Sendable)
    case messageUpdate(message: AssistantMessage, streamDelta: AssistantStreamDelta)
    case messageEnd(message: any AgentMessageProtocol & Sendable)

    // Tool execution lifecycle
    case toolExecutionStart(toolCallId: String, toolName: String, argsJSON: String)
    case toolExecutionUpdate(toolCallId: String, toolName: String, result: AgentToolResult)
    case toolExecutionEnd(toolCallId: String, toolName: String, result: AgentToolResult, isError: Bool)
}

// Stream delta — concrete, Sendable
struct AssistantStreamDelta: Sendable {
    let textDelta: String?
    let thinkingDelta: String?
    let toolCallDelta: ToolCallDelta?
}

struct ToolCallDelta: Sendable {
    let toolCallId: String
    let name: String?
    let argumentsDelta: String?
}
```

**Note**: `any AgentMessageProtocol & Sendable` is used for event payloads that carry arbitrary message types (e.g., `agentEnd` includes all new messages). The protocol already requires `Sendable`, but the explicit `& Sendable` satisfies the compiler for existential storage in a `Sendable` enum.

### D.2 Event Flow (exact sequence)

```
agentStart
├─ turnStart
│  ├─ messageStart (user prompt)
│  ├─ messageEnd (user prompt)
│  ├─ messageStart (assistant, streaming begins)
│  ├─ messageUpdate × N (text deltas, thinking deltas, tool call deltas)
│  ├─ messageEnd (assistant, final)
│  ├─ toolExecutionStart (tool 1, argsJSON)
│  ├─ toolExecutionUpdate × N (partial AgentToolResult with content + details)
│  ├─ toolExecutionEnd (tool 1, final AgentToolResult with content + details)
│  ├─ messageStart (toolResult 1)
│  ├─ messageEnd (toolResult 1)
│  ├─ toolExecutionStart (tool 2) ... etc
│  └─ turnEnd
├─ turnStart (next turn, if more tool calls)
│  └─ ... (repeat)
└─ agentEnd(allNewMessages)
```

### D.3 Agent Class (Stateful Wrapper)

The `Agent` class wraps the loop with:

1. **State management** — `AgentState` with observable properties
2. **Event subscription** — `subscribe(fn) → unsubscribe`
3. **Message queues** — steering queue (interrupt) + follow-up queue (wait)
4. **Queue modes** — `"all"` (drain entire queue) or `"one-at-a-time"` (one per turn)
5. **Abort control** — `abort()` cancels current generation + tool execution
6. **Wait for idle** — `waitForIdle()` returns when agent finishes
7. **Prompt** — `prompt(message)` starts a new loop (throws if already streaming)
8. **Continue** — `continue()` retries from current context

---

## E. Extension System — Exact API

### E.1 Extension Protocol

```swift
protocol AgentExtension {
    /// Unique identifier path
    var path: String { get }

    /// Custom tools this extension provides
    var tools: [String: RegisteredTool] { get }

    /// Event handlers keyed by event type
    var handlers: [String: [ExtensionEventHandler]] { get }

    /// Slash commands
    var commands: [String: RegisteredCommand] { get }

    /// Keyboard shortcuts
    var shortcuts: [String: ExtensionShortcut] { get }
}
```

### E.2 Extension Event Types (Pi has 30+)

For MVP, Tesseract needs these event hooks:

| Event | When Fired | Return Type |
|---|---|---|
| `session_start` | Initial session load | void |
| `session_shutdown` | Session closing | void |
| `before_agent_start` | Before prompt is sent | Optional system prompt override, extra messages |
| `turn_start` | Each turn begins | void |
| `turn_end` | Each turn ends | void |
| `message_start` | Message begins | void |
| `message_update` | Streaming delta | void |
| `message_end` | Message complete | void |
| `tool_call` | Before tool executes | Optional block (prevent execution) |
| `tool_result` | After tool executes | Optional content/details override |
| `context` | Before LLM call, after transformContext | Optional modified messages |
| `input` | User input received | Optional transform or "handled" |
| `resources_discover` | On startup/reload | Additional skill/prompt paths |
| `session_before_compact` | Before compaction | Optional cancel |

### E.3 Extension Context

Extensions receive a context object when handling events:

```swift
protocol ExtensionContext {
    var cwd: String { get }
    var model: LLMModel? { get }
    func isIdle() -> Bool
    func abort()
    func getSystemPrompt() -> String
    func getContextUsage() -> ContextUsage?
    func compact(options: CompactOptions?)
}
```

### E.4 Tool Wrapping (Extension Hooks Around Tools)

Pi wraps every tool call through extensions (`wrapper.ts`):

1. Before execution: fire `tool_call` event → extensions can block
2. Execute the tool
3. After execution: fire `tool_result` event → extensions can modify result

This is how extensions intercept and transform tool behavior without replacing the tool itself.

### E.5 ExtensionRunner

Central dispatcher that:
- Iterates extensions in load order
- Creates `ExtensionContext` per event
- Catches and reports extension errors (never crashes the agent)
- Aggregates registered tools from all extensions (first registration wins)

---

## F. Skill System — Exact Spec

### F.1 Skill File Format

Skills are markdown files with YAML frontmatter:

```markdown
---
name: memory-management
description: Use this skill when the user wants to remember, update, or forget personal facts.
---

# Memory Management

## Workflow
1. Read memories.md first using the read tool
2. Use edit for precise changes (old text must match exactly)
3. Use write only for full rewrites when structure has drifted
```

**Required fields:**
- `description`: Max 1024 chars. **Required** — skills without descriptions are skipped entirely.

**Optional fields:**
- `name`: lowercase a-z, 0-9, hyphens only. Max 64 chars. If omitted, Pi falls back to the parent directory name. If present but mismatched with the parent directory, Pi warns but still loads the skill. Tesseract adopts the same lenient behavior.
- `disable-model-invocation`: If `true`, skill is NOT listed in prompt (only invocable via explicit command).

**Pi behavior notes (for accuracy)**:
- Pi does NOT require `name` — it falls back to the parent directory name if missing.
- Pi warns on name/directory mismatch but does not reject the skill.
- The only hard requirement is `description` — without it, the skill is skipped.

### F.2 Skill Discovery Rules

1. **Root .md files**: Direct `.md` children of a skills directory (not just `SKILL.md` — any `.md` file with valid frontmatter)
2. **SKILL.md in subdirectories**: Recursive scan for `SKILL.md` files under subdirectories
3. **Deduplication**: By name (first loaded wins). Collisions are reported as diagnostics.
4. **Ignore patterns**: Respects `.gitignore`, `.ignore`, `.fdignore` files

**Note**: Pi supports both patterns — a single `memory.md` at the skills root, or `memory/SKILL.md` in a subdirectory. Tesseract supports both.

### F.3 Skill Locations (load order = precedence order)

```
1. ~/Library/Application Support/Tesseract Agent/agent/skills/     (user-global)
2. {project}/.agents/skills/                                   (project-local)
3. Package-provided skill paths                                (from packages)
4. Extension-provided skill paths                              (from resources_discover)
```

### F.4 Skills in System Prompt (XML format)

```xml
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory.

<available_skills>
  <skill>
    <name>memory-management</name>
    <description>Use this skill when the user wants to remember, update, or forget personal facts.</description>
    <location>/Users/owl/Library/Application Support/Tesseract Agent/agent/skills/memory-management/SKILL.md</location>
  </skill>
</available_skills>
```

Skills are listed but NOT inlined. The model uses `read` to load the full skill when needed. This is Pi's progressive disclosure model.

---

## G. Compaction — Exact Algorithm

### G.1 When to Compact

```swift
func shouldCompact(contextTokens: Int, contextWindow: Int, settings: CompactionSettings) -> Bool {
    guard settings.enabled else { return false }
    return contextTokens > contextWindow - settings.reserveTokens
}
```

Default settings (for Tesseract's 120K context window):
- `reserveTokens`: 16,384 — headroom for the next assistant response + tool calls
- `keepRecentTokens`: 20,000 — recent turns preserved verbatim after compaction
- `enabled`: true

These match Pi's defaults and are valid at Tesseract's 120K scale: compaction triggers at ~104K tokens, summarizes older history, and preserves the 20K most recent tokens verbatim. If Tesseract targets smaller models with 4–8K context windows in the future, these values must scale down proportionally (e.g., `reserveTokens: 2048`, `keepRecentTokens: 2048`).

### G.2 Token Estimation

```swift
func estimateTokens(_ message: AgentMessage) -> Int {
    var chars = 0
    // Sum all text content (text blocks, thinking blocks, tool call names + JSON args)
    // For images: estimate 4,800 chars (≈1,200 tokens)
    return (chars + 3) / 4  // ceil(chars / 4) heuristic
}
```

Pi uses usage data from the last assistant response when available (`usage.totalTokens`), falling back to the heuristic for trailing messages after that response.

### G.3 Cut Point Detection

Rules:
1. Walk backwards from newest message, accumulating estimated token counts
2. Stop when accumulated ≥ `keepRecentTokens`
3. Find the closest valid cut point at or after that position
4. **Valid cut points**: user, assistant, custom, branchSummary, compactionSummary messages
5. **Never cut at tool results** (they must follow their tool call)
6. **Turn-aware splitting**: If cut point is mid-turn (not at a user message), also generate a "turn prefix summary" for context

### G.4 Summarization

Two prompts:
1. **Initial summary** (no previous compaction): Generate structured checkpoint
2. **Update summary** (has previous compaction): Merge new information into existing summary

Summary format:
```
## Goal
## Constraints & Preferences
## Progress
### Done / In Progress / Blocked
## Key Decisions
## Next Steps
## Critical Context
```

After summarization, file operation lists (read files, modified files) are appended to maintain file awareness across compaction boundaries.

### G.5 Tesseract Adaptation

Pi's compaction uses the same LLM model to generate summaries. For Tesseract:
- The local model generates the summary (using same inference pipeline)
- Compaction is triggered proactively (before overflow), not reactively
- `transformContext` is the hook point — compaction runs inside it, before each LLM call
- Compaction settings are configurable (extensions can override)

---

## H. Message Types and Conversion

### H.1 Custom Message Types

Pi extends `AgentMessage` via TypeScript declaration merging. Tesseract uses the open `CustomAgentMessage` protocol (see B.1). Session-layer and extension types conform independently:

```swift
// Session-layer types (first-party, outside core)
struct CompactionSummaryMessage: CustomAgentMessage, Sendable {
    let customType = "compaction_summary"
    let summary: String
    let tokensBefore: Int
    let timestamp: Date
    func toLLMMessage() -> LLMMessage? {
        .user(content: "<summary>\n\(summary)\n</summary>")
    }
}

// Extension-provided types (compiled modules add more)
struct BranchSummaryMessage: CustomAgentMessage, Sendable {
    let customType = "branch_summary"
    let branchName: String
    let summary: String
    func toLLMMessage() -> LLMMessage? {
        .user(content: "<branch_summary>\n\(summary)\n</branch_summary>")
    }
}

// Generic fallback for extensions that don't need a dedicated struct
struct GenericCustomMessage: CustomAgentMessage, Sendable {
    let customType: String
    let content: [ContentBlock]
    let display: Bool
    let timestamp: Date
    func toLLMMessage() -> LLMMessage? {
        let text = content.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined()
        return text.isEmpty ? nil : .user(content: text)
    }
}
```

### H.2 convertToLlm

This is the critical boundary function. Called before every LLM call. Because every message conforms to `AgentMessageProtocol`, conversion is a single `compactMap` — no exhaustive switch needed, no changes required when new message types are added.

```swift
func convertToLlm(_ messages: [AgentMessage]) -> [LLMMessage] {
    messages.compactMap { $0.toLLMMessage() }
}
```

Each message type owns its own conversion logic via `toLLMMessage()`. The core never needs to know about extension-defined message types.

### H.3 Why This Matters

Current Tesseract uses `AgentChatMessage` with a simple `Role` enum (system/user/assistant/tool). The new design needs the richer message type system to support:
- Compaction summaries that appear as context, not as tool results
- Extension-injected messages that the LLM sees but the UI renders differently
- Future message types (branch summaries, etc.) without changing the core loop or the `convertToLlm` function

### H.4 Message Persistence — Tagged Encoding

The protocol-backed message model requires a concrete encoding/decoding strategy for persisting heterogeneous `[any AgentMessageProtocol]` to JSON. Without this, `AgentConversationStore` cannot serialize conversations.

**Design**: Tagged JSON encoding with a `type` discriminator field and an inline `payload` object. Each message type registers its tag. Encoding and decoding both go through a central registry so that unknown types survive round-trips.

```swift
// ── Encoding protocol ─────────────────────────────────────────────
protocol PersistableMessage: AgentMessageProtocol, Codable {
    /// Unique tag for JSON discriminator (e.g., "user", "assistant", "compaction_summary")
    static var persistenceTag: String { get }
}

// ── Tagged envelope for JSON ──────────────────────────────────────
// The payload is stored as a JSON dictionary (not Data) so it serializes
// as an inline object in the conversation file, not a base64 string.
struct TaggedMessage: Codable {
    let type: String                           // discriminator tag
    let payload: [String: AnyCodableValue]     // inline JSON object

    init<M: PersistableMessage>(_ message: M) throws {
        self.type = M.persistenceTag
        // Encode to JSON dict via round-trip: M → Data → [String: Any] → [String: AnyCodableValue]
        let data = try JSONEncoder().encode(message)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EncodingError.invalidValue(message, .init(codingPath: [], debugDescription: "Expected JSON object"))
        }
        self.payload = dict.mapValues { AnyCodableValue($0) }
    }

    /// Encode from an opaque message (preserves original JSON without re-interpretation)
    init(opaque: OpaqueMessage) {
        self.type = opaque.tag
        self.payload = opaque.rawPayload
    }
}

// ── Registry (actor for concurrency safety) ───────────────────────
// Core types are registered at startup. Extensions register theirs on load.
// Actor isolation guarantees safe concurrent access — no @unchecked Sendable needed.
actor MessageCodecRegistry {
    static let shared = MessageCodecRegistry()

    private var codecs: [String: MessageCodec] = [:]

    struct MessageCodec {
        let encode: (any AgentMessageProtocol) throws -> TaggedMessage
        let decode: ([String: AnyCodableValue]) throws -> any AgentMessageProtocol
    }

    func register<M: PersistableMessage>(_ type: M.Type) {
        codecs[M.persistenceTag] = MessageCodec(
            encode: { message in
                guard let typed = message as? M else {
                    throw EncodingError.invalidValue(message, .init(codingPath: [], debugDescription: "Type mismatch"))
                }
                return try TaggedMessage(typed)
            },
            decode: { dict in
                // Round-trip: [String: AnyCodableValue] → Data → M
                let data = try JSONEncoder().encode(dict)
                return try JSONDecoder().decode(M.self, from: data)
            }
        )
    }

    func encode(_ message: any AgentMessageProtocol) throws -> TaggedMessage {
        // OpaqueMessage has its own encode path — preserves original JSON
        if let opaque = message as? OpaqueMessage {
            return TaggedMessage(opaque: opaque)
        }
        // Look up codec by concrete type's tag
        if let persistable = message as? any PersistableMessage {
            let tag = type(of: persistable).persistenceTag
            guard let codec = codecs[tag] else {
                throw EncodingError.invalidValue(message, .init(codingPath: [], debugDescription: "No codec for tag: \(tag)"))
            }
            return try codec.encode(message)
        }
        throw EncodingError.invalidValue(message, .init(codingPath: [], debugDescription: "Message is not PersistableMessage"))
    }

    func decode(_ tagged: TaggedMessage) throws -> any AgentMessageProtocol {
        guard let codec = codecs[tagged.type] else {
            // Unknown type — preserve as opaque. Survives future saves via TaggedMessage(opaque:).
            return OpaqueMessage(tag: tagged.type, rawPayload: tagged.payload)
        }
        return try codec.decode(tagged.payload)
    }
}

// ── Opaque fallback for unknown types ─────────────────────────────
// If a conversation was saved with an extension-defined message type
// and that extension is later removed, the message survives as opaque.
// On the next save, TaggedMessage(opaque:) re-serializes the original
// payload unchanged — no data loss across load/save cycles.
struct OpaqueMessage: AgentMessageProtocol, Sendable {
    let tag: String
    let rawPayload: [String: AnyCodableValue]  // original JSON dict, not Data
    func toLLMMessage() -> LLMMessage? { nil }  // invisible to LLM
}
```

**AnyCodableValue**: A simple enum wrapping JSON primitives (string, number, bool, null, array, object) that conforms to `Codable` and `Sendable`. Standard pattern — many open-source implementations exist. This avoids `Any` in Codable contexts.

**Conversation file format**: Keep JSON (not JSONL trees). Payloads are inline JSON objects, not base64-encoded data:

```json
{
    "id": "uuid",
    "title": "...",
    "createdAt": "...",
    "updatedAt": "...",
    "messages": [
        { "type": "user", "payload": { "content": "Hello", "timestamp": "..." } },
        { "type": "assistant", "payload": { "content": "Hi!", "toolCalls": [] } },
        { "type": "compaction_summary", "payload": { "summary": "...", "tokensBefore": 45000 } }
    ]
}
```

**Why JSON over JSONL trees**: JSONL trees enable branching, but Tesseract has no branching UI. JSON files are simpler, already used by the current `AgentConversationStore`, and directly `Codable`. If branching is needed later, it can be added as a migration.

**Extension registration**: Extensions that define `PersistableMessage` types call `await MessageCodecRegistry.shared.register(MyMessage.self)` during `session_start`. The registry must be populated before any conversation is loaded. Since `MessageCodecRegistry` is an actor, registration is inherently serialized and safe.

---

## I. Tool Implementation Details

> **Adaptation principle**: Tool *interfaces* (schema, result structure, abort handling) are copied from Pi exactly. Tool *behaviors* are adapted where Pi assumes capabilities Tesseract does not have (e.g., bash, unrestricted filesystem access). Each adaptation is called out explicitly below.

### I.1 read Tool

**Schema**: `read(path: String, offset?: Int, limit?: Int)`

Implementation details (adapted from Pi):
- Resolves path relative to sandbox root
- Detects images by MIME type → returns as image content block
- Text files: splits into lines, applies offset (1-indexed to 0-indexed)
- **Truncation**: Max 2,000 lines OR 30KB (whichever hit first)
- Truncated output includes actionable notice: `[Showing lines 1-2000 of 5432. Use offset=2001 to continue.]`
- Returns structured result with `content` (for LLM) and `details` (truncation info for UI)

**Sandbox adaptation**: Pi's read tool suggests `bash: sed -n '...' | head -c ...` for oversized single lines. Tesseract has no bash tool. Instead, when a single line exceeds the byte limit, return a truncated version of the line with a notice: `[Line N is X KB, truncated to 30KB. Content may be incomplete.]` This is a behavioral adaptation — the model cannot use an alternative tool, so the read tool must handle it gracefully.

### I.2 write Tool

**Schema**: `write(path: String, content: String)`

- Creates parent directories automatically (`mkdir -p` equivalent)
- Returns success message with byte count
- Abort-aware (checks signal before and after write)

### I.3 edit Tool

**Schema**: `edit(path: String, oldText: String, newText: String)`

Implementation details from Pi:
- Reads file, detects line ending style (LF vs CRLF)
- Normalizes to LF for matching, restores original endings on write
- Strips BOM if present
- **Exact match required** — oldText must appear exactly once
- If zero matches: **fuzzy match fallback** — attempts normalized comparison (collapse whitespace, trim)
- If fuzzy match found: returns error with suggestion showing what was found
- If multiple matches: returns error with count
- Generates unified diff for UI display
- Reports first changed line number (for editor navigation)

### I.4 list Tool (Tesseract-specific)

**Schema**: `list(path?: String, recursive?: Bool, limit?: Int)`

Tesseract's sandbox-safe replacement for Pi's `bash ls`, `find`, `grep`:
- Lists files and directories relative to sandbox root
- Recursive mode with hard cap (e.g., 500 entries)
- Returns formatted listing with file sizes and types
- Read-only — no modifications

### I.5 Path Sandboxing

All tools must resolve paths through a sandbox:
- Relative paths resolve against the agent's working root
- Absolute paths are allowed only within the sandbox
- Path traversal (`../`) that escapes the sandbox is rejected
- Symlinks that point outside the sandbox are rejected

The sandbox root for Tesseract: `~/Library/Application Support/Tesseract Agent/agent/`

---

## J. Resource Loading Pipeline

### J.1 Load Order

Pi's `DefaultResourceLoader.reload()` loads resources in this order:

1. **Packages** → resolve enabled packages → collect extension/skill/prompt/theme paths
2. **Extensions** → load from package paths + CLI paths → register tools/commands/hooks
3. **Skills** → load from package paths + default dirs + CLI paths → deduplicate by name
4. **Prompt templates** → load from package paths + default dirs
5. **Themes** → load from package paths + default dirs
6. **Context files** → `AGENTS.md`/`CLAUDE.md` from global dir + ancestor dirs + cwd
7. **System prompt override** → `SYSTEM.md` from project dir → global dir
8. **Append system prompt** → `APPEND_SYSTEM.md` from project dir → global dir

### J.2 Context File Discovery

**Pi's algorithm** (from `loadProjectContextFiles()` — for reference only):

```
1. Check global agent dir for AGENTS.md or CLAUDE.md
2. Walk from filesystem root to cwd, collecting AGENTS.md/CLAUDE.md at each level
3. Deduplicate by path (each file loaded at most once)
4. Order: global first, then ancestors (root → cwd)
```

**Tesseract's adapted algorithm** (no filesystem walking — see section 10.3):

```
1. Check global agent dir for AGENTS.md or CLAUDE.md
2. Collect context files from enabled packages (in package load order)
3. Collect context files from extension resources_discover events
4. Deduplicate by path (each file loaded at most once)
5. Order: global first, then packages, then extensions
```

### J.3 System Prompt Assembly

From `buildSystemPrompt()`:

```
1. If SYSTEM.md exists → use it as base (replaces default prompt)
   Else → use default prompt template
2. Append APPEND_SYSTEM.md content (if exists)
3. Append context files as "# Project Context" sections
4. Append skills listing (XML format, only if read tool available)
5. Append "Current date and time: {dateTime}"
6. Append "Current working directory: {agentRoot}"
```

### J.4 Tesseract Adaptation

For the sandboxed macOS app (see also section 10.3 for the authoritative rule):
- **No filesystem walking** — no ancestor directory traversal. Context files load from a fixed set of locations: global agent dir, package-provided paths, and extension-provided paths.
- Context file discovery order: global agent dir → package-provided → extension-provided
- Extensions are compiled Swift modules, not runtime-loaded TypeScript
- Package paths resolve within the app bundle + Application Support
- `SYSTEM.md` and `APPEND_SYSTEM.md` are checked in the global agent dir first, then in package-provided paths

---

## K. Current Tesseract Delta Analysis

### K.1 What Exists and Can Be Kept

| Component | Lines | Status |
|---|---|---|
| `AgentEngine.swift` | ~150 | Keep — @MainActor MLX wrapper |
| `LLMActor.swift` | ~150 | Keep — actor-isolated model container |
| `AgentTool.swift` | 125 | **Rewrite** — concrete struct with closures (see B.2), structured `AgentToolResult`, `onUpdate` callback |
| `ToolRegistry.swift` | 38 | **Rewrite** — needs extension tool aggregation |
| `ToolCallParser.swift` | 215+ | Keep — streaming `<tool_call>` parser (Pi uses server-side parsing) |
| `AgentChatMessage.swift` | 91 | **Rewrite** — replace with `AgentMessageProtocol` hierarchy (see B.1), `CoreMessage` enum, `CustomAgentMessage` protocol, tagged persistence (see H.4) |
| `AgentConversationStore.swift` | 185 | **Rewrite** — tagged JSON encoding via `MessageDecoderRegistry` (see H.4), adapt to protocol-backed message types |
| `AgentDebugLogger.swift` | ~100 | Keep |
| Voice/Notch/Views | ~2000+ | Keep — UI layer unchanged |
| Benchmark suite | ~1500+ | **Update** — adapt to new tool surface |

### K.2 What Must Be Rewritten

| Component | Current Lines | Reason |
|---|---|---|
| `AgentRunner.swift` | 364 | Needs Pi double-loop, no round cap, event model, transformContext/convertToLlm hooks |
| `AgentCoordinator.swift` | 399 | Remove memory loading and pre-prompt injection. **Remove `contextLimit = 60` hard truncation** — full history must flow through `transformContext`/compaction instead of being clamped before the runner sees it. Remove observation masking (20 recent). |
| `SystemPromptBuilder.swift` | 175 | Replace 3-tier system with Pi-style assembly (tools + guidelines + context + skills) |

### K.3 What Must Be Created

| Component | Estimated Lines | Priority |
|---|---|---|
| `AgentMessage.swift` (protocol hierarchy + core types) | ~200 | P0 — Core |
| `AgentLoop.swift` (Pi double-loop) | ~300 | P0 — Core |
| `AgentState.swift` (observable state) | ~80 | P0 — Core |
| `AgentSession.swift` (lifecycle, compaction, extensions) | ~400 | P0 — Core |
| `ContextManager.swift` (token estimation + compaction) | ~300 | P0 — Core |
| `ContextLoader.swift` (AGENTS.md, SYSTEM.md) | ~100 | P0 — Core |
| `ExtensionHost.swift` (protocol + runner) | ~250 | P1 — Extension system |
| `AgentExtension.swift` (protocol) | ~100 | P1 — Extension system |
| `ExtensionRunner.swift` (event dispatch) | ~200 | P1 — Extension system |
| `SkillRegistry.swift` (discovery + formatting) | ~200 | P1 — Skills |
| `PackageRegistry.swift` (manifest loading) | ~150 | P1 — Packages |
| `AgentPackage.swift` (manifest model) | ~50 | P1 — Packages |
| `ReadTool.swift` | ~100 | P0 — Tools |
| `WriteTool.swift` | ~60 | P0 — Tools |
| `EditTool.swift` | ~150 | P0 — Tools |
| `ListTool.swift` | ~80 | P0 — Tools |
| `PathSandbox.swift` | ~60 | P0 — Tools |
| `PersonalAssistantPackage.swift` | ~100 | P2 — Package |
| Skill files (memory, tasks, notes) | ~200 | P2 — Package |
| `APPEND_SYSTEM.md` (assistant personality) | ~50 | P2 — Package |

### K.4 What Must Be Removed from Core

| File | Lines | Destination |
|---|---|---|
| `MemoryTools.swift` | ~140 | → personal-assistant extension tool (or file workflow) |
| `TaskTools.swift` | 111 | → personal-assistant skill |
| `GoalTools.swift` | 136 | → personal-assistant skill |
| `HabitTools.swift` | 348+ | → personal-assistant extension tool |
| `MoodTools.swift` | 97 | → personal-assistant skill |
| `ReminderTools.swift` | 83 | → personal-assistant extension tool |
| `RespondTool.swift` | 23 | Delete (not in Pi) |
| `AgentDataStore.swift` | 65 | → personal-assistant package (tools use file tools instead) |
| `DateParsingUtility.swift` | 232+ | → personal-assistant package utility |

---

## L. Implementation Order

### Phase 0: Foundation Types (do first, touches everything)

1. Define `AgentMessageProtocol`, `CoreMessage`, `CustomAgentMessage` protocol hierarchy
2. Define `AgentToolResult` with content blocks
3. Define `AgentTool` concrete struct (closures, no associated types)
4. Define `AgentEvent` enum
5. Define `AgentLoopConfig` struct
6. Define `AgentState` struct

### Phase 1: Core Loop (the heart of the rewrite)

7. Implement `agentLoop()` — Pi's double-loop with steering + follow-ups
8. Implement `agentLoopContinue()` — retry entry point
9. Implement `Agent` class (Swift: @MainActor ObservableObject) with queues + subscriptions
10. Implement `convertToLlm()` for all message types
11. Wire to existing `AgentEngine`/`LLMActor` for generation

### Phase 2: Built-In Tools

12. Implement `PathSandbox` — sandboxed path resolution
13. Implement `ReadTool` — with truncation, offset/limit
14. Implement `WriteTool` — with mkdir, abort handling
15. Implement `EditTool` — with exact match, fuzzy fallback, diff
16. Implement `ListTool` — sandbox-safe directory listing

### Phase 3: Context and Prompt

17. Implement `ContextLoader` — AGENTS.md, CLAUDE.md, SYSTEM.md, APPEND_SYSTEM.md
18. Implement `SkillRegistry` — discovery, dedup, XML formatting
19. Rewrite `SystemPromptBuilder` — Pi-style assembly
20. Implement `ContextManager` — token estimation + compaction

### Phase 4: Extension System

21. Define `AgentExtension` protocol
22. Implement `ExtensionHost` — registration, tool aggregation
23. Implement `ExtensionRunner` — event dispatch, error handling
24. Wire tool wrapping (before/after hooks)

### Phase 5: Package System

25. Define `AgentPackage` manifest model
26. Implement `PackageRegistry` — manifest loading, resource collection
27. Create `personal-assistant` package (skills, prompt, seed files)
28. Migrate domain tools to package extension tools where needed

### Phase 6: Session Layer

29. Rewrite `AgentCoordinator` — thin UI bridge, no domain logic, **remove contextLimit=60 hard truncation and observation masking** (compaction via `transformContext` replaces both)
30. Rewrite `AgentConversationStore` — tagged JSON encoding via `MessageDecoderRegistry` (see H.4)
31. Wire voice I/O through extension events (or keep direct for MVP)
32. Update benchmarks

---

## M. Open Questions

1. ~~**Conversation persistence format**~~: **Resolved** — Keep JSON with tagged message encoding (see H.4). JSONL trees can be added later if branching is needed.

2. **Extension loading mechanism**: First-party compiled modules (simple) vs. runtime-loadable bundles (flexible). Recommend: compiled modules for MVP, runtime bundles later.

3. **Compaction model**: Use the same local model for summarization, or a dedicated smaller model? Same model is simpler but ties up the GPU.

4. **Voice I/O integration point**: Keep voice directly in AgentCoordinator (simpler) or move to an extension (cleaner architecture, more work)?

5. **Backward compatibility**: How to handle existing JSON data (memories.json, tasks.json, etc.)? Recommend: migration tool that converts to markdown files, keep JSON as backup.

### M.1 Resolved Decisions

These were previously open questions, now closed:

- **maxToolRounds**: No cap. This is a core principle (section 3.5) and a loop rule (section C.2). If a model loops pathologically, the user aborts. There is no soft cap, no hard cap, and no extension-overridable cap. The loop runs until: no tool calls AND no steering AND no follow-ups, or the user aborts.
- **Conversation persistence format**: JSON with tagged encoding. See H.4.
