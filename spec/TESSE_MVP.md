# Tesse: Local AI Life Agent — MVP Specification

> *"The last human invention"* — A privacy-first, voice-first personal agent that helps humans stay on track, achieve more, and live healthier lives. Runs entirely on-device.

**Status**: Business Analysis & MVP Plan
**Date**: 2026-02-15
**Version**: 0.1

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Market Analysis](#2-market-analysis)
3. [Target User](#3-target-user)
4. [Core Value Propositions](#4-core-value-propositions)
5. [Agent Philosophy](#5-agent-philosophy)
6. [MVP Feature Scope](#6-mvp-feature-scope)
7. [LLM Selection: Nanbeige4.1-3B](#7-llm-selection-nanbeige41-3b)
8. [Agent Architecture Overview](#8-agent-architecture-overview)
9. [Tool System Design](#9-tool-system-design)
10. [Memory & Personalization](#10-memory--personalization)
11. [Voice Interaction Design](#11-voice-interaction-design)
12. [Behavioral Science Framework](#12-behavioral-science-framework)
13. [Integration with Tesseract](#13-integration-with-tesseract)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [MVP Success Criteria](#15-mvp-success-criteria)
16. [Post-MVP Roadmap](#16-post-mvp-roadmap)
17. [References](#17-references)

---

## 1. Vision & Problem Statement

### The Problem

Humans are bad at:
- **Staying on track** — We set goals and forget them. We plan our week and drift by Tuesday.
- **Doing hard things consistently** — Discipline is finite. Motivation fades. Accountability partners aren't always available.
- **Resting properly** — We either overwork or procrastinate. Few people rest deliberately and effectively.
- **Seeing the big picture** — Day-to-day noise drowns out what actually matters for long-term wellbeing.

Existing solutions fail because:
- **Cloud AI assistants** (ChatGPT, Gemini) know nothing about you between sessions, require internet, and your most personal data flows through corporate servers
- **Coaching apps** (Aris, Rocky.ai, Jules) are cloud-based, subscription-locked, and use generic models — they can't truly personalize
- **Productivity tools** (Todoist, Notion) are passive — they track but don't coach, don't adapt, don't speak to you
- **Apple Intelligence / Siri** is general-purpose and not coaching-focused — it answers questions but doesn't guide your life

### The Vision

**Tesse** is a local AI agent that lives on your Mac, knows your goals, understands your patterns, and actively helps you become who you want to be. It speaks to you. It listens. It remembers everything. And it never phones home.

Tesse is not a chatbot. It's a **life operating system** — an always-available personal coach that combines:
- The consistency of software (never forgets, never judges, always available)
- The warmth of a good coach (empathetic, encouraging, honest)
- The intelligence of modern AI (understands context, reasons about your situation)
- The privacy of a personal journal (everything stays on your machine)

### Why "The Last Human Invention"

The ambition is not to build another AI chat app. The ambition is to build the agent that helps every human unlock their potential — one that improves itself alongside the human it serves. This starts with an MVP on macOS, but the vision is universal: **every person deserves a personal coach, and AI makes that possible for the first time in history**.

---

## 2. Market Analysis

### Market Size

| Metric | Value | Source |
|--------|-------|--------|
| Global AI assistant market (2025) | $3.35B | MarketsAndMarkets |
| Projected AI assistant market (2030) | $21.11B | MarketsAndMarkets |
| CAGR | 17.5% | Grand View Research |
| On-device AI market growth | Rapid, privacy-driven | CoherentMI |

### Competitive Landscape

| Product | Approach | Local? | Coaching? | Voice? | Price |
|---------|----------|--------|-----------|--------|-------|
| **Aris** | AI life coach, well-being focus | No (cloud) | Yes | No | $199/yr |
| **Rocky.ai** | Personal development coach | No (cloud) | Yes | No | Subscription |
| **Jules** | Free AI life coach | No (cloud) | Yes | No | Free |
| **Hapday** | Guided sessions, daily plans | No (cloud) | Yes | No | Subscription |
| **Tolan** | Voice-first AI companion | No (cloud) | Partial | Yes | Subscription |
| **Second Me** | AI digital twin, local | Yes | No | No | Open source |
| **Apple Intelligence** | General assistant | Hybrid | No | Yes | Built-in |
| **Tesse (ours)** | **Life coach + agent** | **Yes** | **Yes** | **Yes** | **Built-in** |

### The Gap

**No existing product combines all four**:
1. Fully local/private execution
2. Voice-first interaction
3. Structured coaching methodology
4. Agent capabilities (tools, memory, proactive behavior)

This is the gap Tesse fills.

### Validation Signals

- **Tolan** (voice-first AI app): 200,000+ MAU, 4.8 stars since Feb 2025 — proves voice-first AI has strong PMF
- **Second Me** (local AI twin): 14,300+ GitHub stars — proves demand for private, personalized AI
- **75% of coaching businesses** use AI co-pilots in 2026 — proves AI coaching works
- **AI coaching users are more honest with AI** than human coaches — the privacy advantage compounds

---

## 3. Target User

### Primary Persona: "The Ambitious Professional"

- **Age**: 25-45
- **Device**: Mac (Apple Silicon)
- **Profile**: Knowledge worker, founder, creative, or student pushing themselves
- **Pain points**:
  - Sets ambitious goals but struggles with follow-through
  - Knows what they should do but can't always make themselves do it
  - Feels guilty about resting, or doesn't rest enough
  - Wants accountability but doesn't want to pay $500/month for a human coach
  - Privacy-conscious — doesn't want their inner thoughts on OpenAI's servers
- **What they want**: A tireless, non-judgmental partner who keeps them honest with themselves

### Secondary Persona: "The Health-Conscious Optimizer"

- Tracks habits, sleep, exercise, nutrition
- Wants an intelligent system that connects the dots between their data points
- Values privacy — health data is deeply personal

### Secondary Persona: "The Overwhelmed Achiever"

- Too many commitments, struggling to prioritize
- Needs help saying no, delegating, and focusing
- Wants someone to help them see the forest, not just the trees

---

## 4. Core Value Propositions

### 1. "Your coach never sleeps"
Available 24/7, voice or text. No scheduling, no fees, no awkward conversations.

### 2. "Your data never leaves"
Every conversation, every goal, every vulnerable moment stays on your machine. Zero cloud dependency for core functionality.

### 3. "It actually knows you"
Unlike cloud AI that forgets between sessions, Tesse builds a persistent model of who you are — your goals, patterns, strengths, struggles. Every conversation makes it better at helping you.

### 4. "It does things, not just says things"
Tesse is an agent, not a chatbot. It sets reminders, tracks habits, manages your todo list, and follows up. It closes the loop between intention and action.

### 5. "Built on science, not vibes"
Grounded in Cognitive Behavioral Therapy, Motivational Interviewing, and habit formation science. Not generic platitudes — structured interventions that actually work.

---

## 5. Agent Philosophy

### Design Principles

1. **Agent, not oracle** — Tesse orchestrates tools and templates. The 3B model dispatches to deterministic systems rather than trying to "know" everything. It doesn't calculate your calories — it reads from a structured tracker and formats the response.

2. **Structure over improvisation** — 3B models excel at template filling and tool dispatch. Lean into structured frameworks (CBT worksheets, SMART goals, habit trackers) rather than free-form therapy simulation.

3. **Short and warm, not long and generic** — Responses should be 1-3 sentences in voice mode, 1-2 paragraphs max in text mode. Quality of connection over quantity of words.

4. **Proactive, not passive** — Tesse should initiate conversations: morning check-ins, goal reminders, streak celebrations, end-of-day reflections. The user shouldn't always have to come to Tesse.

5. **Honest, not sycophantic** — When the user is avoiding something, Tesse names it gently. When a goal is unrealistic, Tesse says so. Trust requires honesty.

6. **Celebrate small wins** — The habit science is clear: positive reinforcement of small steps beats pressure for big leaps. Tesse notices and celebrates progress.

7. **Know when to shut up** — Not every moment needs coaching. Sometimes the user just wants to vent. Tesse should detect this and just listen.

### Personality

Tesse is:
- **Warm but not saccharine** — Like a good friend who happens to be great at keeping you accountable
- **Direct but not harsh** — Says what needs to be said, with care
- **Curious** — Asks good follow-up questions rather than assuming
- **Encouraging without being patronizing** — Celebrates genuinely, not performatively
- **Grounded** — References your actual data, goals, and history rather than generating generic advice

---

## 6. MVP Feature Scope

### In Scope (MVP)

#### 6.1 Conversational Agent Core
- Text-based chat interface with streaming responses
- Voice input (via existing WhisperKit STT)
- Voice output (via existing Qwen3-TTS)
- Conversation history persistence (local JSON)
- System prompt establishing Tesse's personality and role

#### 6.2 Goal & Task Management
- **Tool: `create_goal`** — Set goals with name, description, target date
- **Tool: `list_goals`** — View active goals and progress
- **Tool: `update_goal`** — Mark progress, add notes
- **Tool: `create_task`** — Break goals into actionable tasks
- **Tool: `complete_task`** — Check off completed tasks
- **Tool: `list_tasks`** — View today's tasks, upcoming, overdue

#### 6.3 Habit Tracking
- **Tool: `create_habit`** — Define habits with frequency (daily, weekdays, weekly)
- **Tool: `log_habit`** — Record habit completion
- **Tool: `habit_status`** — View streaks, completion rates, trends

#### 6.4 Daily Check-ins
- **Tool: `morning_checkin`** — Review today's goals, tasks, habits; set intentions
- **Tool: `evening_reflection`** — Review day, celebrate wins, note learnings
- **Tool: `mood_log`** — Quick mood capture (1-10 + optional note)

#### 6.5 Basic Memory
- **Tool: `remember`** — Store a fact about the user (preference, context, important info)
- **Tool: `recall`** — Retrieve relevant memories for current conversation
- Key-value fact store (structured, searchable)
- Conversation summarization for long-term context

#### 6.6 Reminders & Nudges
- **Tool: `set_reminder`** — Schedule a reminder for a specific time
- **Tool: `set_recurring_reminder`** — Daily/weekly recurring nudges
- macOS notification integration for delivery

### Out of Scope (Post-MVP)

- Calendar integration (Apple Calendar API)
- Health data integration (Apple HealthKit)
- Financial tracking
- Multi-device sync
- Fine-tuned model (use base Nanbeige4.1-3B first)
- Proactive conversation initiation (scheduled check-ins that open the app)
- Web search / external information retrieval
- Document/file analysis
- Integration with other apps (Notes, Reminders, etc.)
- Multi-modal input (screenshots, images)

---

## 7. LLM Selection: Nanbeige4.1-3B

### Why This Model

| Criteria | Nanbeige4.1-3B | Nearest Alternative (Qwen3-4B) |
|----------|---------------|-------------------------------|
| **Tool calling (BFCL-v4)** | **56.50%** | 44.87% |
| **Agent benchmarks** | 500+ round tool invocations | Limited |
| **Architecture** | LlamaForCausalLM (universal support) | Qwen3 (requires custom code) |
| **Context window** | 256K tokens | 128K tokens |
| **License** | Apache 2.0 | Apache 2.0 |
| **MLX weights** | Available (bf16, 8-bit) | Available |
| **Arena-Hard-v2** | **73.2** | 34.9 |
| **GPQA (reasoning)** | **83.8** | 65.8 |

Nanbeige4.1-3B is the clear choice because:
1. **Best-in-class tool calling** for a 3B model — critical for an agent
2. **Native `<tool_call>` tags** in the chat template — no prompt engineering hacks needed
3. **LlamaForCausalLM architecture** — MLX supports this natively, no custom model code
4. **256K context** — room for conversation history + tool definitions + memory retrieval
5. **Chain-of-thought** via `<think>` tags — reasoning before acting

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM (decoder-only, GQA) |
| Parameters | ~3.8B (non-embedding: ~3B) |
| Hidden size | 2560 |
| Layers | 32 |
| Attention heads | 20 query / 4 KV (GQA 5:1) |
| Head dim | 128 |
| Intermediate size | 10496 |
| Vocab size | 166,144 (bilingual EN/ZH) |
| Context length | 262,144 tokens (256K) |
| RoPE theta | 70,000,000 |
| Disk (bf16) | ~7.87 GB |
| Disk (8-bit) | ~4.18 GB |
| Disk (4-bit, est.) | ~2.1 GB |

### Chat Format

ChatML-style with `<|im_start|>` / `<|im_end|>` delimiters:

```
<|im_start|>system
You are Tesse, a personal life coach and agent...<|im_end|>
<|im_start|>user
I want to start exercising more<|im_end|>
<|im_start|>assistant
<think>
The user wants to build an exercise habit. I should ask about their current activity level and help them set a SMART goal. Let me also check if they have any existing fitness-related goals.
</think>

<tool_call>
{"name": "list_goals", "arguments": {"category": "health"}}
</tool_call><|im_end|>
```

### Tool Definition Format

Tools are defined in `<tools>` tags in the system prompt:

```json
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "create_habit",
      "description": "Create a new habit to track",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "Name of the habit"},
          "frequency": {"type": "string", "enum": ["daily", "weekdays", "weekly"]}
        },
        "required": ["name", "frequency"]
      }
    }
  }
]
</tools>
```

### Inference Performance Expectations (Apple Silicon)

| Metric | 8-bit (~4.2 GB) | 4-bit (~2.1 GB) |
|--------|-----------------|-----------------|
| Tokens/sec | ~30-50 tok/s | ~50-80 tok/s |
| Time to first token | 80-150ms | 50-100ms |
| Memory usage | ~5-6 GB | ~3-4 GB |
| Concurrent with STT+TTS | Yes (M3 Pro 18GB+) | Yes (M2 16GB+) |

### Recommended Inference Parameters

```
temperature: 0.6
top_p: 0.95
repetition_penalty: 1.0
max_tokens: 2048 (short responses preferred)
```

### Risks

1. **Benchmark skepticism** — Nanbeige4.1-3B's numbers are almost too good. The authors show learning curves on evaluation benchmarks, raising checkpoint-selection concerns. **Mitigation**: Our use case (tool calling + short responses) is well-suited for 3B models regardless. If Nanbeige disappoints in practice, Qwen3-4B is a drop-in fallback (same ChatML format, similar tool calling support).

2. **Chinese-default system prompt** — The model's default persona is "Nanbeige" with Chinese instructions. **Mitigation**: We override the system prompt completely with Tesse's English persona and tool definitions.

3. **Large vocabulary** — 166K tokens means larger embedding tables and slightly slower first-token inference. **Mitigation**: Acceptable tradeoff for the quality gains. 4-bit quantization brings it to ~2.1 GB.

---

## 8. Agent Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                   Voice │ Text │ Hotkey                       │
└──────────┬──────────────┼───────────────┬───────────────────┘
           │              │               │
     ┌─────▼─────┐  ┌────▼────┐   ┌─────▼──────┐
     │ WhisperKit│  │  Text   │   │  Hotkey    │
     │   STT     │  │  Input  │   │  Manager   │
     └─────┬─────┘  └────┬────┘   └─────┬──────┘
           │              │               │
           └──────────────┼───────────────┘
                          │
                ┌─────────▼──────────┐
                │  AgentCoordinator  │
                │  (State Machine)   │
                └─────────┬──────────┘
                          │
              ┌───────────▼────────────┐
              │    Context Builder     │
              │  ┌──────────────────┐  │
              │  │ System Prompt    │  │
              │  │ Tool Definitions │  │
              │  │ Retrieved Memory │  │
              │  │ Conversation Ctx │  │
              │  └──────────────────┘  │
              └───────────┬────────────┘
                          │
               ┌──────────▼──────────┐
               │   Nanbeige4.1-3B    │
               │   (MLX Inference)   │
               └──────────┬──────────┘
                          │
                   ┌──────▼──────┐
                   │  Response   │
                   │   Router    │
                   └──┬─────┬───┘
                      │     │
            ┌─────────▼┐  ┌▼──────────┐
            │ Tool Call │  │   Text    │
            │ Executor  │  │ Response  │
            └─────┬─────┘  └─────┬────┘
                  │              │
            ┌─────▼─────┐       │
            │  Tool      │       │
            │  Results   │       │
            └─────┬─────┘       │
                  │              │
                  │   (loop if   │
                  │   more tool  │
                  │   calls)     │
                  └──────┬───────┘
                         │
              ┌──────────▼──────────┐
              │   Output Router     │
              │  Text UI │ TTS      │
              └──────────┬──────────┘
                         │
                    ┌────▼─────┐
                    │  Memory  │
                    │  Extract │
                    │  & Store │
                    └──────────┘
```

### Agent Loop (ReAct-style with native tool calling)

```
1. User sends message (text or voice→STT)
2. Context Builder assembles:
   - System prompt (Tesse persona + coaching framework)
   - Tool definitions (JSON schema in <tools> tags)
   - Retrieved memories (relevant facts about user)
   - Recent conversation history (last N turns)
3. Send to Nanbeige4.1-3B
4. Model responds with either:
   a. <tool_call>{...}</tool_call> → Execute tool → Feed result back as <tool_response> → goto 3
   b. Text response → Display/speak to user
5. Extract & store any new facts from the conversation
6. Update conversation history
```

### Key Design Decisions

**Why not ReAct text parsing?** — Nanbeige4.1-3B has native `<tool_call>` tags trained into the model. Using these is more reliable than parsing free-text "Action:" / "Observation:" patterns, especially at 3B scale.

**Why not constrained decoding for MVP?** — The model's native tool-calling format is well-trained. Start without grammar constraints; add them if we see malformed tool calls in practice. Simpler is better for MVP.

**Why single-step tool calls?** — 3B models struggle with multi-step planning. Let the model call one tool at a time, see the result, and decide what's next. The loop handles multi-step workflows emergently.

**Max tool call depth: 5** — Safety limit. If the model hasn't resolved the user's request in 5 tool calls, it should summarize what it found and ask for clarification.

---

## 9. Tool System Design

### Tool Registry

Tools are defined as Swift structs conforming to a `AgentTool` protocol:

```swift
protocol AgentTool {
    static var name: String { get }
    static var description: String { get }
    static var parameters: JSONSchema { get }
    func execute(arguments: [String: Any]) async throws -> String
}
```

### MVP Tool Set (14 tools)

#### Goals & Tasks

| Tool | Description | Arguments |
|------|-------------|-----------|
| `create_goal` | Create a new goal | `name`, `description`, `target_date?`, `category?` |
| `list_goals` | List active goals | `category?`, `status?` |
| `update_goal` | Update goal progress | `goal_id`, `progress_note?`, `status?` |
| `create_task` | Create a task (optionally linked to a goal) | `title`, `due_date?`, `goal_id?`, `priority?` |
| `list_tasks` | List tasks | `filter?` (today/upcoming/overdue/all) |
| `complete_task` | Mark task as done | `task_id` |

#### Habits

| Tool | Description | Arguments |
|------|-------------|-----------|
| `create_habit` | Define a new habit | `name`, `frequency`, `time_of_day?` |
| `log_habit` | Record habit completion | `habit_name`, `date?`, `note?` |
| `habit_status` | View habit streaks and stats | `habit_name?` (all if omitted) |

#### Memory & Self

| Tool | Description | Arguments |
|------|-------------|-----------|
| `remember` | Store a fact about the user | `fact`, `category?` |
| `recall` | Retrieve relevant memories | `query` |

#### Daily Flow

| Tool | Description | Arguments |
|------|-------------|-----------|
| `mood_log` | Log current mood | `score` (1-10), `note?` |
| `set_reminder` | Set a one-time reminder | `message`, `time` |
| `get_current_time` | Get current date and time | (none) |

### Tool Result Format

Tool results are returned to the model as `<tool_response>` in the user role:

```
<|im_start|>user
<tool_response>
{"status": "success", "data": {"habit": "Exercise", "streak": 5, "last_logged": "2026-02-14"}}
</tool_response><|im_end|>
```

### Storage Backend

MVP uses simple JSON files in `~/Library/Application Support/Tesseract/agent/`:

```
agent/
├── goals.json          # Goal entries
├── tasks.json          # Task entries
├── habits.json         # Habit definitions + log entries
├── memories.json       # Key-value fact store
├── moods.json          # Mood log entries
├── reminders.json      # Scheduled reminders
└── conversations/      # Conversation history by date
    ├── 2026-02-15.json
    └── ...
```

No database for MVP. JSON files are simple, inspectable, and sufficient for single-user local storage. Migration to SQLite in post-MVP if needed.

---

## 10. Memory & Personalization

### Three-Layer Memory Architecture

Inspired by Second Me's Hierarchical Memory Modeling and Mem0:

#### Layer 1: Working Memory (Context Window)
- Current conversation turns (last 10-20 messages)
- Injected into the model's context directly
- Ephemeral — cleared on conversation end

#### Layer 2: Fact Memory (Persistent Key-Value Store)
- Structured facts about the user: `{"category": "preference", "fact": "prefers morning workouts", "added": "2026-02-15"}`
- Written via `remember` tool or extracted automatically
- Retrieved via keyword matching on `recall` tool (upgrade to embedding search post-MVP)
- Budget: inject top 10 most relevant facts into system prompt (~500 tokens)

#### Layer 3: Narrative Memory (Conversation Summaries)
- End-of-conversation summaries stored as dated entries
- Model generates a 2-3 sentence summary of each conversation
- Last 5 conversation summaries included in system prompt
- Provides continuity across sessions without replaying full history

### Memory Injection Template

```
## What I Know About You
- You prefer morning workouts (Feb 15)
- Your main goal is to launch your startup by Q3 (Feb 14)
- You've been struggling with sleep lately (Feb 13)
- You enjoy running but get bored after 30 minutes (Feb 12)

## Recent Conversations
- Feb 14: We discussed your pitch deck progress. You were feeling anxious about the investor meeting. We broke the preparation into 3 tasks.
- Feb 13: You mentioned poor sleep quality. We set up a "no screens after 10pm" habit.
```

### Privacy Guarantee

All memory is:
- Stored in plain-text JSON files the user can inspect, edit, and delete
- Never transmitted over the network
- Never used for model training
- Deletable via `forget` command (post-MVP) or manual file deletion

---

## 11. Voice Interaction Design

### Voice Pipeline

Tesse leverages Tesseract's existing STT and TTS infrastructure:

```
User speaks
    → WhisperKit STT (50-200ms)
    → Nanbeige4.1-3B inference (100-400ms)
    → Qwen3-TTS synthesis (50-150ms)
    → Audio playback

Total estimated latency: 200-750ms (under the 1-second conversational threshold)
```

### Interaction Modes

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| **Voice-to-Voice** | Microphone (push-to-talk) | TTS speaker | Hands-free coaching, morning check-in |
| **Text-to-Text** | Keyboard | Screen | Deep planning, reviewing data |
| **Text-to-Voice** | Keyboard | TTS speaker | User types, Tesse speaks (accessibility) |
| **Voice-to-Text** | Microphone | Screen | User speaks, Tesse responds in text |

### Voice Design Principles

1. **Concise** — Voice responses should be 1-3 sentences. Nobody wants to listen to a paragraph.
2. **Natural pauses** — Insert sentence breaks for natural TTS pacing.
3. **Confirm actions** — "I've added 'Go for a run' to your morning habits. Your streak starts today."
4. **No data dumps in voice** — If the user asks to "show my goals", display in UI and give a brief voice summary: "You have 3 active goals. Your startup launch goal is 60% complete."

### Hotkey Integration

| Hotkey | Action |
|--------|--------|
| Configurable (e.g., Ctrl+Space) | Push-to-talk for Tesse voice input |
| Configurable (e.g., Ctrl+T) | Open Tesse text chat panel |

---

## 12. Behavioral Science Framework

### Methodology

Tesse's coaching is grounded in three evidence-based frameworks:

#### 12.1 Cognitive Behavioral Therapy (CBT) — For Mindset

**When to use**: User expresses negative self-talk, catastrophizing, or feeling stuck.

**Techniques encoded as conversation templates**:

| Technique | Template Pattern |
|-----------|-----------------|
| **Thought Record** | "What happened? → What did you think? → What did you feel? → What's the evidence for and against?" |
| **Cognitive Distortion ID** | Detect patterns: all-or-nothing, catastrophizing, mind-reading, should-statements |
| **Behavioral Activation** | When mood is low, suggest small achievable activities from user's habit list |
| **Reframing** | "Another way to look at this is..." — gentle perspective shifts |

**Implementation**: These are NOT free-form therapy. Tesse follows structured conversation flows. The model fills in template slots rather than improvising psychological interventions.

#### 12.2 Motivational Interviewing (MI) — For Behavior Change

**When to use**: User wants to change but feels ambivalent. User sets a goal but doesn't start.

**OARS Technique**:
- **Open questions**: "What would it look like if you achieved this?"
- **Affirmations**: "You've already shown you can do this — you kept your exercise streak for 12 days."
- **Reflective listening**: "It sounds like you want to exercise but mornings feel too rushed."
- **Summarizing**: "So on one hand, you want to get healthier, and on the other hand, your schedule feels packed. Let's find where it fits."

**Implementation**: System prompt includes MI principles. Model is instructed to ask before telling, affirm before challenging.

#### 12.3 Fogg Behavior Model — For Habit Formation

**B = MAP** (Behavior = Motivation × Ability × Prompt)

Tesse's role in each:
- **Motivation**: Track and celebrate streaks, remind of "why", reference user's stated goals
- **Ability**: Break tasks into tiny steps ("Instead of 'exercise for 60 minutes', start with 'put on your running shoes'")
- **Prompt**: Reminders, check-ins, contextual nudges

**Specific techniques**:
- **Habit stacking**: "After I [existing habit], I will [new habit]"
- **Implementation intentions**: "When [situation], I will [behavior]"
- **2-day rule**: Missing one day is fine. Missing two breaks the pattern.

### Conversation Routing

The model doesn't need to "know" psychology — it needs to recognize which framework to apply:

```
User says something negative about themselves
  → Route to CBT thought record template

User expresses wanting to change but not starting
  → Route to MI open questions

User asks to create a new habit
  → Route to Fogg model (tiny habits, habit stacking)

User reports completing a goal/habit
  → Celebrate! Affirm. Ask what enabled success.

User just wants to vent
  → Active listening. Reflect back. Don't problem-solve unless asked.
```

---

## 13. Integration with Tesseract

### Architecture Fit

Tesse follows the same patterns as existing features:

| Pattern | Existing Example | Tesse Equivalent |
|---------|-----------------|------------------|
| Coordinator | `DictationCoordinator`, `SpeechCoordinator` | `AgentCoordinator` |
| Engine (Actor wrapper) | `TranscriptionEngine`, `SpeechEngine` | `AgentEngine` |
| Dedicated Actor | `WhisperActor`, `TTSActor` | `LLMActor` |
| Model Management | `ModelDownloadManager` | Same — add Nanbeige4.1-3B to registry |
| Navigation | `NavigationItem.dictation`, `.speech` | `NavigationItem.agent` |
| History | `TranscriptionHistory` | `AgentConversationHistory` |
| Settings | `SettingsManager` (AppStorage) | Add agent-related settings |

### File Organization

```
tesseract/
├── Features/
│   └── Agent/
│       ├── AgentCoordinator.swift      # State machine (idle → thinking → responding)
│       ├── AgentEngine.swift           # Model wrapper, inference, tool dispatch
│       ├── AgentConversationHistory.swift # Persistence
│       ├── Tools/
│       │   ├── AgentTool.swift         # Protocol
│       │   ├── GoalTools.swift         # create/list/update goal
│       │   ├── TaskTools.swift         # create/list/complete task
│       │   ├── HabitTools.swift        # create/log/status habit
│       │   ├── MemoryTools.swift       # remember/recall
│       │   ├── DailyTools.swift        # mood_log, get_current_time
│       │   └── ReminderTools.swift     # set_reminder
│       ├── Memory/
│       │   ├── FactMemory.swift        # Key-value fact store
│       │   ├── NarrativeMemory.swift   # Conversation summaries
│       │   └── ContextBuilder.swift    # Assembles full context for model
│       └── Views/
│           ├── AgentContentView.swift  # Main chat interface
│           ├── AgentMessageView.swift  # Individual message bubble
│           └── AgentInputView.swift    # Text input + voice button

Vendor/
└── mlx-llm-swift/
    ├── Package.swift
    └── Sources/
        └── MLXLLM/
            ├── LLMModel.swift          # Model loading, generation
            ├── LLMConfig.swift         # Model configuration
            ├── Tokenizer.swift         # Tokenizer wrapper
            ├── KVCache.swift           # KV cache management
            └── Sampling.swift          # Temperature, top-p, repetition penalty
```

### Vendor Package: mlx-llm-swift

A new SPM package following the `mlx-audio-swift` and `mlx-image-swift` patterns:

- **Dependencies**: mlx-swift, swift-transformers (tokenizer), swift-huggingface (model download)
- **Core**: `LlamaModel` implementation — Nanbeige4.1-3B is standard Llama architecture
- **Note**: `mlx-swift-lm` (already a dependency) already has Llama support. Evaluate whether to wrap it or build custom. Custom gives us full control over:
  - Streaming token generation
  - KV cache management
  - Tool call tag detection during generation
  - Memory-efficient inference alongside STT/TTS models

### DependencyContainer Additions

```swift
// In DependencyContainer.swift
lazy var agentEngine = AgentEngine()
lazy var agentCoordinator = AgentCoordinator(
    agentEngine: agentEngine,
    speechEngine: speechEngine,         // For voice output
    transcriptionEngine: transcriptionEngine  // For voice input
)
```

### Memory Budget

Running Tesse alongside existing models:

| Component | Memory (est.) |
|-----------|--------------|
| Nanbeige4.1-3B (4-bit) | ~3 GB |
| WhisperKit (Large V3 Turbo) | ~3 GB |
| Qwen3-TTS (1.7B bf16) | ~4 GB |
| App + framework overhead | ~1 GB |
| **Total** | **~11 GB** |

Fits within 16GB Apple Silicon. For 18GB+ (M3 Pro), comfortable headroom. For 32GB+ (M3 Max), all models can coexist simultaneously.

**Strategy**: Lazy load/unload. Only one of {Whisper, Nanbeige, Qwen3-TTS} needs to be in memory at a time for voice interaction flow. Load Nanbeige when agent tab is active; unload when switching to dictation.

---

## 14. Risks & Mitigations

### Technical Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Nanbeige4.1-3B underperforms benchmarks in practice | Medium | Medium | Qwen3-4B as drop-in fallback (same ChatML format). Our tool set is simple (14 tools) — well within 3B capability |
| Tool call parsing failures (malformed JSON) | Medium | Low-Medium | Native `<tool_call>` tags reduce this. Add JSON validation + retry (ask model to fix). Add constrained decoding if needed |
| Memory exceeds device capacity when all models loaded | High | Low | Lazy load/unload strategy. Only one heavy model at a time. 4-bit quantization for Nanbeige |
| Slow inference blocks UI | Medium | Low | Run on dedicated `LLMActor`. Stream tokens. Never block MainActor |
| Model generates harmful coaching advice | High | Low | Structured templates limit free-form advice. Disclaimer in onboarding. System prompt safety rails |

### Product Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| User expects ChatGPT-level conversation quality | High | High | Set expectations in onboarding: "Tesse is a focused life coach, not a general knowledge AI." Lean into structure (tools, templates) over open-ended chat |
| User shares deeply sensitive information | High | Medium | Clear privacy guarantee. All data local. But: add disclaimer that Tesse is not a therapist or medical professional |
| Habit/goal tracking feels like yet another app | Medium | Medium | Differentiate through conversation. Tesse isn't an app you fill out — it's a coach you talk to, who happens to track things for you |
| Small model confabulates (makes up facts) | Medium | Medium | Tool-grounded responses. Model reads from storage, doesn't recall from parameters. When unsure, model says "I don't have that information" |

### Ethical Considerations

1. **Not a therapist** — Tesse uses CBT and MI techniques but is NOT therapy. Clear disclaimer required.
2. **Not medical advice** — Health tracking and suggestions are informational, not prescriptive.
3. **User agency** — Tesse suggests, never demands. The user is always in control.
4. **Transparency** — The user can see all stored memories, goals, and data. No hidden models or profiles.
5. **Crisis detection** — If user expresses suicidal ideation or severe distress, Tesse should provide crisis hotline information and recommend professional help. This is a hard-coded response, not model-generated.

---

## 15. MVP Success Criteria

### Functional Criteria (Must Have)

- [ ] User can have a text conversation with Tesse in the app
- [ ] Tesse can call tools and execute them (create/list goals, tasks, habits)
- [ ] Conversation history persists across app restarts
- [ ] User can speak to Tesse via push-to-talk and hear voice responses
- [ ] Tesse remembers facts about the user across conversations
- [ ] Tesse can set reminders that deliver macOS notifications
- [ ] Model loads in <10 seconds, generates at >30 tok/s (4-bit)
- [ ] Total memory usage stays under 12GB with agent active

### Quality Criteria (Should Have)

- [ ] Tool call success rate >90% over 100 test conversations
- [ ] Voice-to-voice response latency <2 seconds end-to-end
- [ ] Tesse's personality is consistent (warm, direct, coaching-oriented)
- [ ] Responses are concise (avg <100 tokens in voice mode)
- [ ] Model handles 5+ turn conversations without losing context

### Validation Method

1. **Dogfooding** — Use Tesse daily for 2 weeks as personal coach
2. **Tool reliability** — Log all tool calls, measure success rate
3. **Conversation quality** — Review 50 conversation transcripts for coherence, personality consistency, and useful coaching behavior
4. **Performance** — Measure inference speed, memory usage, voice latency across M2/M3/M4 devices

---

## 16. Post-MVP Roadmap

### Phase 2: Proactive Agent
- Scheduled check-ins (morning/evening notifications that open Tesse)
- Smart nudges based on habit data (e.g., "You usually run at 7am — heading out soon?")
- Daily/weekly summary reports (generated locally)

### Phase 3: Deep Integration
- Apple Calendar read access (schedule-aware coaching)
- Apple HealthKit integration (sleep, steps, heart rate → coaching insights)
- Apple Reminders/Notes sync
- Screen time awareness

### Phase 4: Advanced Memory
- Embedding-based memory retrieval (nomic-embed-text, 137M params)
- Long-term personality modeling (detect patterns in mood, productivity, habits over months)
- Conversation topic clustering

### Phase 5: Model Refinement
- Fine-tune Nanbeige4.1-3B on coaching conversation data
- Custom LoRA for Tesse-specific tool calling patterns
- Distill from larger models (run GPT-4 conversations → train Tesse on them)
- Evaluate newer 3B-class models as they release

### Phase 6: Multi-Modal & Multi-Platform
- Image input (meal photos for nutrition tracking)
- iOS companion app (same model, synced data)
- Watch complications (quick mood log, habit check)
- Always-on ambient listening mode (with explicit consent)

---

## 17. References

### Model
- [Nanbeige4.1-3B — HuggingFace](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
- [Nanbeige4.1-3B MLX bf16](https://huggingface.co/mlx-community/Nanbeige4.1-3B-bf16)
- [Nanbeige4.1-3B MLX 8-bit](https://huggingface.co/mlx-community/Nanbeige4.1-3B-8bit)
- [Nanbeige4-3B Technical Report (arXiv:2512.06266)](https://arxiv.org/abs/2512.06266)
- [Kaitchup: Nanbeige4.1 Analysis](https://kaitchup.substack.com/p/nanbeige41-only-3b-parameters-but)

### Agent & Tool Calling
- [Small LMs for Efficient Agentic Tool Calling (arXiv:2512.15943)](https://arxiv.org/abs/2512.15943)
- [SLMs for Agentic Systems Survey (arXiv:2510.03847)](https://arxiv.org/abs/2510.03847)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Tool Calling Best Practices](https://medium.com/@laurentkubaski/tool-or-function-calling-best-practices-a5165a33d5f1)
- [Constrained Decoding for Structured LLM Output](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)

### Memory & Personalization
- [Mem0: Memory Layer for AI](https://mem0.ai/)
- [Second Me: AI Digital Twin](https://www.secondme.io/)
- [Personalized Long-term LLM Interactions (arXiv:2510.07925)](https://arxiv.org/abs/2510.07925)

### Coaching & Behavioral Science
- [AI-Powered CBT Chatbots Review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11904749/)
- [Motivational Interviewing Guide (ANHCO)](https://anhco.org/blog/motivational-interviewing-the-ultimate-2025-guide-for-coaches)
- [AI Agent Behavioral Science (arXiv:2506.06366)](https://arxiv.org/html/2506.06366v2)
- [AI Coaching Trends 2026 (Delenta)](https://www.delenta.com/blog/ai-coaching-trends-tools-2026)

### Voice AI
- [Voice Agent Latency Guide (Twilio)](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents)
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/)
- [Natural Conversation with Voice Agents (CHI 2025)](https://dl.acm.org/doi/10.1145/3706598.3714228)

### Market
- [AI Assistant Market — MarketsAndMarkets](https://www.marketsandmarkets.com/blog/ICT/ai-assistant)
- [On-Device AI 2026](https://www.aitechboss.com/on-device-ai-2026/)
- [Tolan: 200K MAU Voice AI (OpenAI)](https://openai.com/index/tolan/)

### Apple/MLX
- [MLX on Apple Silicon — WWDC25](https://developer.apple.com/videos/play/wwdc2025/298/)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [Benchmarking ML on Apple Silicon (arXiv:2510.18921)](https://arxiv.org/abs/2510.18921)
