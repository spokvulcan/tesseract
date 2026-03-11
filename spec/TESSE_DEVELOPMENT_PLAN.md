# Tesse — Development Plan

> A generic local AI agent framework with voice and text interfaces.
> Built simple. Not baked-in to any specific use case. The system prompt makes it a coach, a coder, a writer — whatever the user needs.

**Guiding Principle**: Genius things are simple. Build a solid agent that can hear, think, speak, use tools, and remember. Everything else is instructions.

---

## Architecture at a Glance

```
User speaks/types
    → [WhisperKit STT] or [Text Input]
    → [Agent Loop: LLM + Tools]
    → [Text UI] or [Qwen3-TTS]
User reads/listens
```

That's it. The entire system in 4 lines.

---

## Epic 1: LLM Inference Engine

> Get Nanbeige4.1-3B running on MLX Swift. Load it, feed it tokens, get tokens back. Streaming.

### 1.1 — Scaffold `mlx-llm-swift` vendor package

Create `Vendor/mlx-llm-swift/` with Package.swift. Single library target `MLXLLM`. Dependencies: `mlx-swift`, `mlx-swift-lm`, `swift-transformers`, `swift-huggingface`. Add as local package dependency in tesseract.xcodeproj.

**Acceptance**: Package resolves and builds (empty library).

### 1.2 — Model downloading and weight loading

Implement `LLMModelLoader` that downloads Nanbeige4.1-3B weights from HuggingFace (use `mlx-community/Nanbeige4.1-3B-8bit`) and loads them into MLX. Reuse `mlx-swift-lm`'s existing Llama support — Nanbeige is standard `LlamaForCausalLM`. Cache to `~/Library/Caches/mlx-llm/`. Expose load progress for UI.

**Acceptance**: Model downloads, loads into memory, reports ready status. Weights verified by running a single forward pass.

### 1.3 — Tokenizer integration

Wrap the HuggingFace tokenizer (166K vocab, ChatML special tokens). Encode text → token IDs, decode token IDs → text. Handle special tokens: `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<think>`, `</think>`.

**Acceptance**: Round-trip encode/decode preserves text. Special tokens correctly identified.

### 1.4 — Streaming text generation

Implement `generate(prompt: [Int], maxTokens: Int) -> AsyncThrowingStream<Int, Error>` that yields one token at a time. Support parameters: temperature, topP, repetitionPenalty. Handle EOS token (`<|im_end|>`, ID 166101). Implement KV cache for efficient autoregressive generation.

**Acceptance**: Given a tokenized prompt, streams coherent text token-by-token. Measurable tok/s matches expectations (30-80 on Apple Silicon).

### 1.5 — ChatML message formatting

Build a `ChatMLFormatter` that converts `[ChatMessage]` → tokenized prompt. Message roles: `system`, `user`, `assistant`. Insert `<|im_start|>role\n...content...<|im_end|>\n` delimiters. Handle tool calls and tool responses in the conversation.

```swift
struct ChatMessage {
    let role: Role  // .system, .user, .assistant, .tool
    let content: String
}
```

**Acceptance**: Formatted prompts match Nanbeige4.1-3B's expected chat template exactly.

### 1.6 — `LLMActor` isolation wrapper

Create a `nonisolated` actor `LLMActor` that owns the model and runs inference off the main thread. Expose: `loadModel()`, `generate(messages:tools:) -> AsyncThrowingStream<String, Error>`, `unloadModel()`. Report `isLoaded`, `isLoading` status.

**Acceptance**: Generation runs without blocking MainActor. UI stays responsive during inference.

---

## Epic 2: Agent Core

> The agent loop: take a message, maybe call tools, return a response. Generic — knows nothing about coaching or any specific domain.

### 2.1 — Tool protocol and registry

Define the tool contract:

```swift
protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parametersSchema: String { get }  // JSON Schema string
    func execute(arguments: String) async throws -> String
}
```

`ToolRegistry` holds registered tools. Can generate `<tools>[...]</tools>` block for the system prompt from all registered tools. Tools are just a protocol — anyone can implement one.

**Acceptance**: Tools can be registered and their JSON schemas rendered for the system prompt.

### 2.2 — Tool call parser

Parse model output to detect `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` tags. Extract tool name and arguments JSON. Handle edge cases: partial tags in streaming, malformed JSON (return error to model for retry), multiple tool calls in one response.

**Acceptance**: Correctly parses valid tool calls. Gracefully handles malformed output. Works with streaming (detects tags as they appear).

### 2.3 — Agent loop

The core loop that makes this an agent, not just an LLM wrapper:

```
1. Build context: system prompt + tools + memory + conversation history
2. Call LLM (streaming)
3. If response contains <tool_call>:
   a. Execute tool
   b. Append tool result as tool message
   c. Goto 2 (max depth: 5)
4. If response is text:
   a. Yield to UI/TTS
   b. Done
```

Implement as `AgentRunner` class. Not tied to any UI — pure logic. Takes messages in, yields text chunks out, calls tools in between. Configurable `maxToolRounds`.

**Acceptance**: Given a user message and registered tools, agent can chain tool calls and produce a final text response. Stops at max depth.

### 2.4 — Think tag handling

During streaming, detect `<think>...</think>` blocks. Strip them from the displayed/spoken output but optionally log them. The user sees only the final response after `</think>`. Option to show thinking in debug mode.

**Acceptance**: Thinking tokens are generated (model reasons) but not shown to user. Thinking can be viewed in debug/logs.

### 2.5 — System prompt builder

Build the full system prompt by composing parts:

```
[User-provided system instructions]  ← The "personality" — coaching, coding, whatever
[Tool definitions block]             ← Auto-generated from ToolRegistry
[Memory/context block]               ← Optional, injected by memory system
[Current date/time]                  ← Always useful for an agent
```

Each part is optional. The builder doesn't know what the instructions say — it just assembles them.

**Acceptance**: System prompt is correctly assembled with all components. Tools section matches model's expected format.

---

## Epic 3: Text Chat UI

> A simple chat interface. Messages go in, responses stream out. Nothing fancy — functional and clean.

### 3.1 — Navigation and feature scaffold

Add `NavigationItem.agent` case. Create `tesseract/Features/Agent/` directory. Add `AgentContentView` placeholder. Wire into `ContentView` navigation and `DependencyContainer`.

**Acceptance**: Agent tab appears in sidebar. Navigating to it shows the placeholder view.

### 3.2 — Chat message model

Define the conversation data model:

```swift
struct AgentMessage: Identifiable, Codable {
    let id: UUID
    let role: MessageRole  // .user, .assistant, .tool, .system
    let content: String
    let timestamp: Date
    var toolCall: ToolCallInfo?    // If this is a tool call message
    var toolResult: ToolResultInfo? // If this is a tool result
}
```

Keep it simple. A message is text with a role and a timestamp.

**Acceptance**: Messages can be created, encoded to JSON, decoded back.

### 3.3 — `AgentCoordinator` state machine

`@MainActor` class with states: `idle`, `loading` (model loading), `thinking` (LLM generating), `error`. Holds `@Published messages: [AgentMessage]`. Exposes `send(text: String)` that feeds into the agent loop and appends streaming response.

**Acceptance**: User sends text → state transitions to thinking → tokens stream in → response appended → state returns to idle.

### 3.4 — Chat view with streaming display

SwiftUI chat interface:
- Scrollable message list (user bubbles right, assistant bubbles left)
- Text input field at bottom with send button
- Streaming text appears word-by-word as model generates
- Auto-scroll to bottom on new messages
- Loading indicator during model load / thinking

No fancy styling for MVP. Functional first.

**Acceptance**: User can type a message, see it appear, watch the response stream in, and continue the conversation.

### 3.5 — `AgentEngine` — connecting UI to LLM

`@MainActor` wrapper that owns `LLMActor` and `ToolRegistry`. Handles model lifecycle (load/unload). Exposes `generate(messages:) -> AsyncThrowingStream<String, Error>` that runs the agent loop internally.

Integrates with `ModelDownloadManager` for download progress. Adds Nanbeige4.1-3B to the model registry.

**Acceptance**: AgentCoordinator calls AgentEngine, which calls LLMActor, which generates tokens. Full chain works end-to-end.

---

## Epic 4: Conversation Persistence

> Save and load conversations so nothing is lost between app restarts.

### 4.1 — Conversation storage

Save conversations as JSON to `~/Library/Application Support/Tesseract/agent/conversations/`. One file per conversation session. Auto-save after each assistant response completes.

```
conversations/
├── 2026-02-16_14-30.json   # Timestamped session
├── 2026-02-16_19-45.json
└── ...
```

**Acceptance**: Close app, reopen — last conversation restored.

### 4.2 — Conversation history UI

Show list of past conversations in a sidebar or dropdown. Each entry shows date + first user message as preview. Tap to load. "New conversation" button to start fresh.

**Acceptance**: User can browse past conversations and resume any of them.

### 4.3 — Context window management

As conversation grows, keep only the last N messages in the LLM context (configurable, default ~20 messages). Older messages are persisted but not sent to the model. Simple truncation — no summarization for MVP.

**Acceptance**: Long conversations don't exhaust the context window. Model only sees recent messages.

---

## Epic 5: Tool Implementation

> The actual tools that make Tesse useful. Each tool is a self-contained struct — add or remove without touching the core.

### 5.1 — `time_get` tool

The simplest possible tool. Returns the current date, time, and day of week. Proves the tool system works end-to-end.

**Acceptance**: User asks "what time is it?" → model calls `time_get` → returns correct time → model formats response.

### 5.2 — `memory_save` / `memory_search` tools

`memory_save`: Save a fact, preference, or important information to `~/Library/Application Support/Tesseract Agent/agent/memories.json`. Accepts an optional `category` parameter (e.g. preference, health, work, personal). Deduplicates on exact fact text.
`memory_search`: Search stored memories by keyword overlap scoring. Returns top 10 matches with dates and categories.

Simple JSON array of `{ "id": "...", "fact": "...", "category": "...", "createdAt": "..." }` entries.

**Acceptance**: "Remember that I prefer morning workouts" → stored. Next session: "What do you know about my exercise preferences?" → searches and returns the fact.

### 5.3 — Goal management tools

`goal_create`, `goal_list`, `goal_update`. Stored in `goals.json`. Each goal has: id, name, description, status (active/completed/archived), created date, target date (optional), progress notes.

**Acceptance**: User creates a goal through conversation, lists goals, marks progress — all via natural language that triggers tool calls.

### 5.4 — Task management tools

`task_create`, `task_list`, `task_complete`. Stored in `tasks.json`. Each task has: id, title, status (pending/done), due date (optional), goal_id (optional link to a goal), priority (optional).

**Acceptance**: User asks to break a goal into steps → model calls `task_create` for each step. User asks "what should I do today?" → model calls `task_list` with today filter.

### 5.5 — Habit tracking tools

`habit_create`, `habit_log`, `habit_status`. Stored in `habits.json`. Each habit has: name, frequency (daily/weekdays/weekly), log entries (dates completed), created date. `habit_status` calculates current streak and completion rate.

**Acceptance**: User creates a habit, logs it daily, asks for status → sees streak count and completion percentage.

### 5.6 — Mood logging tool

`mood_log`: Record mood score (1-10) with optional note. Stored in `moods.json` with timestamp. `mood_list`: Show recent mood entries.

**Acceptance**: "I'm feeling about a 7 today, pretty good" → logged. "How has my mood been this week?" → shows recent entries.

### 5.7 — Reminder tool

`reminder_set`: Schedule a macOS notification for a future time. Uses `UNUserNotificationCenter` for delivery. Parse relative times ("in 30 minutes") and absolute times ("at 3pm").

**Acceptance**: "Remind me to stretch in 20 minutes" → notification fires 20 minutes later with "Time to stretch" message.

---

## Epic 6: Voice Integration

> User speaks → STT → Agent → TTS → User hears. Connects existing Tesseract infrastructure to the agent.

### 6.1 — Voice input (STT → Agent)

Add a push-to-talk button in the agent chat view. On press: capture audio via `AudioCaptureEngine` → transcribe via `TranscriptionEngine` → feed transcribed text to `AgentCoordinator.send()` as if the user typed it.

Reuse existing dictation infrastructure — no new audio code needed.

**Acceptance**: User holds voice button, speaks, releases → transcribed text appears as user message → agent responds.

### 6.2 — Voice output (Agent → TTS)

After agent generates a text response, optionally speak it via `SpeechEngine` / `SpeechCoordinator`. Add a toggle: "Voice responses on/off". When on, every assistant response is spoken aloud.

Strip tool call details from spoken output — only speak the final human-readable response.

**Acceptance**: Agent responds in text AND speaks the response. Toggle disables voice.

### 6.3 — Agent hotkey

Register a configurable hotkey (via existing `HotkeyManager`) for "Talk to Tesse". When pressed: open agent view if not visible, activate push-to-talk. When released: send to agent.

Add to `SettingsManager` for user configuration.

**Acceptance**: User presses hotkey from anywhere → agent activates → voice input captured → response generated.

### 6.4 — Voice-optimized response mode

When voice output is enabled, add `[Voice mode: keep responses to 1-3 sentences]` instruction to the system prompt. This makes the model respond concisely for spoken delivery without hardcoding any behavior.

**Acceptance**: Voice mode responses are noticeably shorter than text mode. Same model, different instructions.

---

## Epic 7: Memory System

> Tesse remembers things across conversations. Not just chat history — structured facts and conversation summaries.

### 7.1 — Fact memory store

A persistent JSON store for user facts. Tools write to it (via `memory_save`), context builder reads from it. Each fact: `{ id, fact, category, createdAt }`. Keyword-scored search for retrieval (via `memory_search`).

**Acceptance**: Facts persist across conversations and app restarts. Retrieved facts are injected into context.

### 7.2 — Context builder

Assembles the full context for each LLM call:
1. System prompt (user-configurable instructions)
2. Tool definitions (auto-generated from registry)
3. Retrieved facts (from fact memory, top 10 most relevant)
4. Recent conversation summaries (last 3-5 sessions, if available)
5. Current date/time
6. Current conversation messages (last N turns)

Each section is modular and optional.

**Acceptance**: Context is correctly assembled. Model receives relevant memory alongside the conversation.

### 7.3 — Conversation summarization

At end of conversation (or periodically), ask the model to generate a 2-3 sentence summary. Store in `summaries.json`. These summaries are injected into future conversations for continuity.

**Acceptance**: Conversation ends → summary generated and stored → next conversation includes summary as context.

---

## Epic 8: Settings & Configuration

> Make it configurable. The user decides what Tesse is.

### 8.1 — Agent settings page

Add agent section to Settings:
- **System prompt**: Editable text area — the user's instructions to Tesse (default provided but fully editable)
- **Model selection**: Which model to use (initially just Nanbeige4.1-3B-8bit)
- **Voice responses**: Toggle on/off
- **Max tool rounds**: Slider (1-10, default 5)
- **Temperature**: Slider (0.0-1.0, default 0.6)

Stored via `@AppStorage`.

**Acceptance**: All settings persist and take effect immediately.

### 8.2 — Default system prompt

Ship a thoughtful default system prompt that makes Tesse a warm, helpful general-purpose assistant. NOT hardcoded coaching — just a good starting personality. The user can change it to anything.

```
You are Tesse, a personal AI assistant. You are warm, direct, and helpful.
You remember what the user tells you and use tools to help them stay organized.
Keep responses concise. Ask clarifying questions when needed.
```

**Acceptance**: Works well out of the box. Users can override completely.

### 8.3 — Model download management

Integrate Nanbeige4.1-3B into the existing `ModelDownloadManager`. Show download progress on the Models page. Allow choosing between 8-bit (~4.2 GB) and 4-bit (~2.1 GB) variants.

**Acceptance**: User can download the LLM model from the Models page with progress indication.

---

## Post-MVP Epics

> What comes after the foundation is solid. Roughly ordered by value.

### Post-MVP A: Proactive Notifications
- Scheduled check-ins (morning/evening reminders that open Tesse)
- Habit reminders at configured times
- Follow-up on incomplete tasks
- Daily/weekly auto-generated summaries

### Post-MVP B: Advanced Memory
- Embedding-based recall (nomic-embed-text, 137M params) — replaces keyword search
- Automatic fact extraction (model identifies facts to remember without explicit `remember` calls)
- Memory management UI (view, edit, delete stored facts)
- Conversation topic tagging and search

### Post-MVP C: Safety Layer
- Crisis keyword detection with hardcoded safety response (hotline numbers, professional help recommendation)
- Content filtering for harmful advice
- Disclaimer system for medical/financial/legal topics
- "Not a therapist" onboarding disclosure

### Post-MVP D: Apple Integrations
- Apple Calendar read access (schedule-aware context)
- Apple Reminders sync (bidirectional)
- Apple HealthKit (sleep, steps, heart rate data)
- Screen time data access

### Post-MVP E: Model Flexibility
- Support multiple models (switch between Nanbeige, Qwen3, Llama, etc.)
- 4-bit quantization conversion pipeline
- Model comparison tools (benchmark on same prompts)
- Fine-tuning pipeline for custom LoRA adapters

### Post-MVP F: Multi-Platform
- iOS companion app (shared data format, same model)
- Encrypted cross-device sync
- watchOS complications (quick mood log, habit check)
- Menu bar quick-access mode

---

## Implementation Order

The epics are designed to be built in sequence, each building on the previous:

```
Epic 1 (LLM Engine)     → Can generate text
Epic 2 (Agent Core)      → Can call tools
Epic 3 (Text Chat UI)    → User can talk to it
Epic 4 (Persistence)     → Conversations survive restarts
Epic 5 (Tools)           → Agent becomes useful
Epic 6 (Voice)           → Agent can hear and speak
Epic 7 (Memory)          → Agent remembers across sessions
Epic 8 (Settings)        → User configures everything
```

Each epic's tasks can be implemented one at a time. Each task has clear acceptance criteria. After each task, the app builds and the new capability works.

**Total tasks**: 30 (MVP) + 6 post-MVP epics

**Estimated effort**: The system is intentionally simple. No databases, no complex abstractions, no overengineering. JSON files, Swift protocols, and the patterns already proven in Tesseract's STT/TTS/ImageGen features.
