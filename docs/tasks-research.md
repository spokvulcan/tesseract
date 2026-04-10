# Tasks Feature Research: From Primitive Checkboxes to Proactive Accountability

## Problem Statement

The current task system is too primitive for a proactive agent to be effective. Tasks are plain markdown checkboxes with zero metadata:

```markdown
- [ ] Build the landing page
- [x] Fix audio latency bug
```

The agent cannot answer basic questions like:
- "What should I work on today?" (no dates, no priorities)
- "You're behind on X" (no deadlines to compare against)
- "This is urgent" (no priority information)
- "You created this 3 days ago and haven't started" (no timestamps)
- "This blocks Y, do it first" (no dependencies)

The scheduled tasks system (cron-based `ScheduledTask`) is powerful but serves a different purpose -- it runs agent prompts on a schedule. What's missing is **structured user tasks** that give the proactive agent the context it needs to push, remind, and hold the user accountable.

---

## Current State

### User Tasks (personal-assistant package)

**Skill**: `AgentPackages/personal-assistant/skills/tasks/SKILL.md`
**Storage**: `tasks.md` in agent working directory
**Format**: `- [ ] text` / `- [x] text`
**Operations**: Create, list, complete, delete via file read/write/edit
**Metadata**: None

### Scheduled Tasks (ScheduledTask model)

**Storage**: JSON files in `~/Library/Application Support/Tesseract Agent/agent/scheduled-tasks/`
**Model**: Full metadata -- UUID, cron, prompt, tags, run history, notifications, TTS
**Purpose**: Background agent execution on cron schedules (heartbeat, recurring checks)

### Gap

User tasks have **no structure**. Scheduled tasks have **rich structure but wrong purpose**. The proactive agent needs structured user tasks to reason about what to push the user on.

---

## External Research

### Existing Task Management Approaches for AI Agents

#### MCP Servers & Plugins (ranked by relevance)

| Project | Approach | Key Insight |
|---------|----------|-------------|
| **Scopecraft/MDTM** | TOML frontmatter in .md files, one file per task | Phase-based organization, parent/subtask hierarchy |
| **TaskMD** | YAML frontmatter + markdown body | Designed specifically for AI agents, has Claude Code skills |
| **Overseer** | SQLite, 3-level hierarchy (Milestone/Task/Subtask) | Priority 1-5, progressive context inheritance |
| **atlas-mcp-server** | Neo4j graph (Projects > Tasks > Knowledge) | Knowledge graph linking, dependency tracking |
| **agentic-tools-mcp** | Unlimited hierarchy, priority/complexity 1-10 | Estimated vs actual hours, agent memories |
| **mcp-tasks** | Multi-format (MD/JSON/YAML), minimal metadata | Auto-WIP management, batch operations |

#### Plain-Text Task Formats

**todo.txt** -- One line per task, strict field ordering:
```
(A) 2025-03-26 Call plumber @home +HouseRepair due:2025-03-28
```
- Priority: `(A)`-`(Z)`, contexts: `@where`, projects: `+name`, extensions: `key:value`
- Proven at scale, simple parsing, but lacks multi-line descriptions

**Obsidian Tasks** -- Inline emoji signifiers on checkbox lines:
```
- [ ] Build landing page 📅 2025-03-28 🔼 ➕ 2025-03-26 ⏳ 2025-03-27
```
- Emoji fields: `📅` due, `⏳` scheduled, `🛫` start, `➕` created, `✅` done
- Priority via `⏫`/`🔼`/`🔽`/`⏬`
- Advantage: stays on one line, human-readable. Disadvantage: emoji parsing, no multi-line context

**Taskwarrior Urgency Model** -- Computed urgency from 13 weighted factors:
- Due date proximity (12.0), blocking others (8.0), high priority (6.0), active (4.0), age (2.0)
- Waiting (-3.0), blocked (-5.0)
- Key insight: urgency is **computed, not assigned** -- it changes over time

**MDTM (Roo Commander)** -- TOML frontmatter + markdown body, one file per task:
```toml
+++
id = "TASK-001"
title = "Build landing page"
status = "in_progress"
priority = "high"
due_date = "2025-03-28"
+++
## Description
...
## Acceptance Criteria
- [ ] ...
```

### Proactive/Accountability Agent Patterns

**thunlp ProactiveAgent**: Senses user environment (app focus, AFK status), generates task suggestions. Reward model calibrates intervention frequency based on accept/reject/ignore feedback.

**ClawHub Proactive Agent Workspace**: Structured files -- `AGENTS.md` (operating rules), `SOUL.md` (identity), `USER.md` (human context), `SESSION-STATE.md` (working memory), `HEARTBEAT.md` (periodic self-check). WAL protocol captures corrections before responding.

**Macaron**: Recognizes procrastination as an emotional regulation issue. Shifts from "coach" to "supportive friend" based on user state. Provides the dopamine hit needed to initiate action.

**ChatGPT Tasks**: Scheduled tasks generating dynamic content (not static reminders). Push notifications. Limit of 10 active tasks.

**Key behavioral insight**: The most effective proactive agents:
1. Compute urgency dynamically (not just static priority)
2. Escalate gradually (gentle reminder -> direct push -> accountability confrontation)
3. Distinguish importance from urgency (Eisenhower matrix)
4. Track user response patterns (accept/ignore/reject) to calibrate intervention frequency
5. Treat procrastination as emotional, not organizational

---

## Design Options

### Option A: Single-File Markdown with Inline Metadata

Keep everything in `tasks.md`, add structured metadata inline using `key:value` pairs (todo.txt style).

```markdown
# Tasks

- [ ] Build the landing page priority:high due:2025-03-28 created:2025-03-26 scheduled:2025-03-27
- [ ] Fix audio latency bug priority:critical due:2025-03-26 created:2025-03-25 project:tesseract
- [x] Design system prompt priority:medium created:2025-03-20 done:2025-03-24
- [ ] Write tests for cron parser priority:low created:2025-03-22 project:tesseract
```

**Pros**:
- Minimal change from current format
- Single file, easy to read/write
- Human-editable
- Stays on one line per task -- good for LLM context windows

**Cons**:
- No room for descriptions, subtasks, or notes
- Parsing `key:value` pairs requires careful skill instructions
- Gets unwieldy with many metadata fields
- No natural grouping by project/date

### Option B: Single-File Markdown with YAML Frontmatter Sections

One `tasks.md` file but with a YAML header for metadata and structured sections.

```markdown
---
version: 1
last_updated: 2025-03-26T10:00:00
---

## Today (2025-03-26)

### Build the landing page
- **priority**: high
- **due**: 2025-03-28
- **created**: 2025-03-26
- **status**: in_progress
- **project**: tesseract-web
- **notes**: Start with hero section. Check Figma for designs.
- [ ] Hero section
- [ ] Features grid
- [ ] CTA section

### Fix audio latency bug
- **priority**: critical
- **due**: 2025-03-26
- **created**: 2025-03-25
- **status**: pending
- **project**: tesseract
- **blocked_by**: waiting for test device

## Tomorrow (2025-03-27)

### Write tests for cron parser
- **priority**: low
- **created**: 2025-03-22
- **status**: pending
- **project**: tesseract

## Completed

### Design system prompt
- **priority**: medium
- **created**: 2025-03-20
- **completed**: 2025-03-24
- **project**: tesseract
```

**Pros**:
- Rich metadata per task
- Supports subtasks, notes, descriptions
- Natural date-based grouping
- Human-readable and editable
- Agent can reason about daily plans

**Cons**:
- More complex parsing for the agent
- File grows over time (needs archival strategy)
- Harder to do simple operations (complete a task = move between sections)
- Multi-line format uses more tokens in agent context

### Option C: YAML Frontmatter Per-Task Files (MDTM Style)

One markdown file per task in a `tasks/` directory.

```
tasks/
  TASK-001_build-landing-page.md
  TASK-002_fix-audio-latency.md
  TASK-003_write-cron-tests.md
  archive/
    TASK-000_design-system-prompt.md
```

Each file:
```markdown
---
id: TASK-001
title: Build the landing page
status: in_progress
priority: high
due: 2025-03-28
scheduled: 2025-03-27
created: 2025-03-26
project: tesseract-web
tags: [frontend, web]
blocked_by: []
---

## Description
Create the landing page for the Tesseract website.

## Subtasks
- [ ] Hero section
- [ ] Features grid
- [ ] CTA section

## Notes
- Check Figma for designs
- Use the new design system colors
```

**Pros**:
- Clean separation, each task is self-contained
- Rich metadata via YAML frontmatter (reliable parsing)
- Unlimited space for descriptions, notes, subtasks
- Easy to archive (move to `archive/`)
- Git-friendly (each task change is a separate diff)

**Cons**:
- Agent must `ls` + `read` multiple files (more tool calls)
- Overhead for simple tasks ("buy milk" doesn't need its own file)
- Needs an index or listing operation
- More complex skill instructions

### Option D: Single-File Structured Markdown (Recommended)

A hybrid: single `tasks.md` file with lightweight YAML-like metadata per task, using a consistent block format. Balances richness with simplicity.

```markdown
# Tasks

## Build the landing page
- status: in_progress
- priority: high
- due: 2025-03-28
- created: 2025-03-26
- project: tesseract-web
- [ ] Hero section
- [ ] Features grid
- [ ] CTA section

## Fix audio latency bug
- status: pending
- priority: critical
- due: 2025-03-26
- created: 2025-03-25
- project: tesseract

## Write tests for cron parser
- status: pending
- priority: low
- created: 2025-03-22
- project: tesseract

## ~~Design system prompt~~
- status: done
- priority: medium
- created: 2025-03-20
- completed: 2025-03-24
- project: tesseract
```

**Pros**:
- Single file (one read to get everything)
- Each task is an H2 section -- easy to find, edit, delete
- Metadata as `- key: value` list items -- natural markdown, no special parsing
- Supports subtasks as checkboxes within the section
- Completed tasks use strikethrough on heading (visually distinct)
- Agent can add notes as plain text within a section
- Human-readable and editable
- Moderate token usage (compact but not cryptic)

**Cons**:
- File grows (needs periodic archival of completed tasks)
- H2 headings as task identifiers means task titles must be unique
- Less structured than YAML frontmatter (agent must be careful with format)

---

## Recommended Approach: Option D (Single-File Structured Markdown)

### Why Option D

1. **Minimal friction**: One file, one read. The agent already works with `tasks.md` -- this is an evolution, not a rewrite.

2. **Right amount of metadata**: Priority, dates, status, project -- enough for the proactive agent to reason about urgency and scheduling without being overwhelming.

3. **Human-editable**: Users can open the file and understand/edit it. No special tools needed.

4. **Agent-friendly**: H2 headings as task boundaries make edit operations reliable. `- key: value` is natural for LLMs to parse and generate.

5. **Subtask support**: Checkboxes within a task section give progress tracking for free.

6. **Single-file simplicity**: No directory management, no index files, no multi-file reads.

### Proposed Metadata Fields

| Field | Required | Format | Purpose |
|-------|----------|--------|---------|
| `status` | Yes | `pending` / `in_progress` / `blocked` / `done` / `cancelled` | Lifecycle state |
| `priority` | Yes | `critical` / `high` / `medium` / `low` | Importance level |
| `created` | Yes | `YYYY-MM-DD` | When task was created (enables age-based urgency) |
| `due` | No | `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` | Hard deadline |
| `scheduled` | No | `YYYY-MM-DD` | When user plans to work on it |
| `completed` | No | `YYYY-MM-DD` | When task was finished |
| `project` | No | Free text | Grouping / filtering |
| `blocked_by` | No | Free text | What's blocking this task |
| `notes` | No | Free text (inline) | Additional context below metadata |

### Fields Deliberately Excluded

- **ID**: H2 heading serves as identifier. UUIDs add complexity without value for a personal system.
- **Tags**: `project` covers the main grouping need. Tags add parsing complexity.
- **Estimated effort**: Useful but adds friction to task creation. Can be added later.
- **Recurrence**: Handled by the existing ScheduledTask/cron system.
- **Energy level**: Interesting concept but too speculative for v1.
- **Dependencies (task-to-task)**: `blocked_by` as free text is simpler than formal dependency graphs.

### Urgency Computation (for proactive agent)

The agent should compute urgency dynamically when deciding what to push:

```
Urgency factors (inspired by Taskwarrior):
- Due date within 24h:  +10
- Due date within 3 days: +5
- Priority critical:      +8
- Priority high:          +5
- Status blocked:         -5
- Age > 3 days (pending): +2
- Age > 7 days (pending): +4
- Has subtasks, none done: +1
```

This doesn't need to be code -- it should be guidance in the skill file so the agent reasons about it naturally.

### Archival Strategy

When tasks accumulate, completed/cancelled tasks should be moved to `tasks-archive.md` (same format). The skill should instruct the agent to archive completed tasks older than 7 days.

---

## Implementation Plan

### Phase 1: Enhanced Task Skill

Update `AgentPackages/personal-assistant/skills/tasks/SKILL.md` with:

1. **New format specification** (Option D format)
2. **Migration instructions** (how to handle existing plain checkboxes)
3. **Required metadata** on task creation (status, priority, created date)
4. **Urgency reasoning** guidance for proactive behavior
5. **Archival workflow** for completed tasks

### Phase 2: Proactive Behavior Skill

Create a new skill `AgentPackages/personal-assistant/skills/accountability/SKILL.md`:

1. **Morning check-in**: "What are you working on today?" -- show today's scheduled tasks, overdue items, high-priority items
2. **Nudge protocol**: Graduated escalation based on task urgency and time since last interaction
3. **End-of-day review**: What got done, what didn't, re-prioritize for tomorrow
4. **Weekly review prompt**: Review all tasks, archive completed, re-evaluate priorities

### Phase 3: Heartbeat Integration

Update the heartbeat system to read `tasks.md` and:

1. Check for overdue tasks
2. Check for tasks scheduled for today that haven't been started
3. Generate proactive nudges based on urgency computation
4. Report via notification/TTS

### Phase 4 (Future): Scheduled Task Bridge

Consider a bridge where high-urgency user tasks automatically generate scheduled task check-ins:
- "Task X is due tomorrow" -> create a cron check that reminds at 9 AM
- "Task Y has been pending for 5 days" -> create a one-time nudge

---

## Key Design Decisions to Make

1. **Single file vs directory**: Option D (single file) is recommended, but if tasks frequently have long descriptions/subtask lists, Option C (per-file) scales better. Start with D, migrate to C if needed.

2. **How strict should the format be?**: The skill needs to balance flexibility (users say "add a task to buy milk" and shouldn't need to specify every field) with consistency (agent must reliably parse). Recommendation: require `status`, `priority`, `created`; default the rest.

3. **Completed task handling**: Strikethrough heading + `status: done` + `completed: date` vs. moving to a separate section vs. archiving to separate file. Recommendation: mark done in-place, archive periodically.

4. **Proactive timing**: How aggressive should the agent be? This is a user preference that should be configurable (in `memories.md` or settings). Start gentle, let the user calibrate.

5. **Token budget**: A full `tasks.md` with 20 tasks and metadata will use ~2K-3K tokens in context. This is fine for the 120K window but should be monitored.
