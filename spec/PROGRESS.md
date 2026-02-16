# Tesse — Implementation Progress

> Track every task as it's built. Update status after each session.

**Legend**: `[ ]` Not started · `[~]` In progress · `[x]` Done · `[-]` Skipped

---

## Epic 1: LLM Inference Engine

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 1.1 | Scaffold `mlx-llm-swift` vendor package | [x] | 2026-02-16 | Wired MLXLLM + MLXLMCommon as SPM dependencies (reused existing mlx-swift-lm, no separate vendor package needed) |
| 1.2 | Model downloading and weight loading | [x] | 2026-02-16 | AgentEngine loads from local dir via `loadModelContainer(directory:)`. Model registered on Models page with ModelDownloadManager handling download/verify/delete. |
| 1.3 | Tokenizer integration | [x] | 2026-02-16 | AgentTokenizer wraps HF tokenizer with all 7 ChatML special tokens resolved at init. Encode/decode via ModelContainer. |
| 1.4 | Streaming text generation | [ ] | | |
| 1.5 | ChatML message formatting | [ ] | | |
| 1.6 | `LLMActor` isolation wrapper | [ ] | | |

## Epic 2: Agent Core

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 2.1 | Tool protocol and registry | [ ] | | |
| 2.2 | Tool call parser | [ ] | | |
| 2.3 | Agent loop | [ ] | | |
| 2.4 | Think tag handling | [ ] | | |
| 2.5 | System prompt builder | [ ] | | |

## Epic 3: Text Chat UI

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 3.1 | Navigation and feature scaffold | [ ] | | |
| 3.2 | Chat message model | [ ] | | |
| 3.3 | `AgentCoordinator` state machine | [ ] | | |
| 3.4 | Chat view with streaming display | [ ] | | |
| 3.5 | `AgentEngine` — connecting UI to LLM | [ ] | | |

## Epic 4: Conversation Persistence

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 4.1 | Conversation storage | [ ] | | |
| 4.2 | Conversation history UI | [ ] | | |
| 4.3 | Context window management | [ ] | | |

## Epic 5: Tool Implementation

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 5.1 | `get_current_time` tool | [ ] | | |
| 5.2 | `remember` / `recall` tools | [ ] | | |
| 5.3 | Goal management tools | [ ] | | |
| 5.4 | Task management tools | [ ] | | |
| 5.5 | Habit tracking tools | [ ] | | |
| 5.6 | Mood logging tool | [ ] | | |
| 5.7 | Reminder tool | [ ] | | |

## Epic 6: Voice Integration

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 6.1 | Voice input (STT → Agent) | [ ] | | |
| 6.2 | Voice output (Agent → TTS) | [ ] | | |
| 6.3 | Agent hotkey | [ ] | | |
| 6.4 | Voice-optimized response mode | [ ] | | |

## Epic 7: Memory System

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 7.1 | Fact memory store | [ ] | | |
| 7.2 | Context builder | [ ] | | |
| 7.3 | Conversation summarization | [ ] | | |

## Epic 8: Settings & Configuration

| # | Task | Status | Date | Notes |
|---|------|--------|------|-------|
| 8.1 | Agent settings page | [ ] | | |
| 8.2 | Default system prompt | [ ] | | |
| 8.3 | Model download management | [x] | 2026-02-16 | Nanbeige4.1-3B registered in ModelDefinition.all with Agent category. Download/verify/delete via existing ModelDownloadManager. |

---

## Post-MVP

| Epic | Description | Status | Notes |
|------|-------------|--------|-------|
| A | Proactive Notifications | [ ] | |
| B | Advanced Memory (embeddings) | [ ] | |
| C | Safety Layer | [ ] | |
| D | Apple Integrations | [ ] | |
| E | Model Flexibility | [ ] | |
| F | Multi-Platform | [ ] | |

---

## Session Log

<!-- Add entries here as work progresses -->

| Date | Session | What was done |
|------|---------|---------------|
| 2026-02-16 | Research & Planning | Created TESSE_MVP.md (business analysis), TESSE_DEVELOPMENT_PLAN.md (epics & tasks), PROGRESS.md (this file) |
| 2026-02-16 | Epic 1 + 8.3 | **1.1**: Wired MLXLLM/MLXLMCommon dependencies. **1.2**: AgentEngine with model loading from local dir, 1-token verification, integrated into Models page via ModelDownloadManager. **1.3**: AgentTokenizer — resolved all 7 ChatML special tokens (im_start/end, endoftext, think, tool_call), encode/decode, isEndOfGeneration. **8.3**: Added ModelCategory.agent, registered Nanbeige4.1-3B in ModelDefinition.all, wired AgentEngine in DependencyContainer + ModelsPageView. |
