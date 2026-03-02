# tesse-ract Development Plan

> Last updated: 2026-02-28

This document tracks the current state of tesse-ract, what has been built, what is in progress, and what comes next. It replaces the original research blueprint (`reasearch/local Wispr Flow alternative.md`) which envisioned a Tauri/Rust app — the project pivoted to native Swift/SwiftUI + MLX for Apple Silicon.

---

## Vision

A privacy-first, offline AI assistant for macOS that runs entirely on-device. Three core capabilities:

1. **Voice Dictation** — push-to-talk transcription injected into any app
2. **Text-to-Speech** — high-quality voice synthesis with streaming and long-form support
3. **AI Agent** — on-device LLM with tool calling for personal productivity (memory, goals, tasks, habits, moods, reminders)

All inference happens locally. No accounts, no telemetry, no cloud dependency.

---

## Architecture Stack

| Layer | Technology |
|-------|-----------|
| Platform | macOS 26+, Apple Silicon only |
| Language | Swift 6.2 |
| UI | SwiftUI |
| ML Runtime | MLX (Apple's ML framework for Apple Silicon) |
| ASR | WhisperKit (CoreML, Whisper Large V3 Turbo) |
| TTS | Qwen3-TTS 1.7B via mlx-audio-swift (vendored) |
| LLM | Nanbeige4.1-3B / Qwen3-4B variants via mlx-swift-lm |
| Image Gen | FLUX.2-klein-4B / Z-Image 6B via mlx-image-swift (vendored) |
| Distribution | Mac App Store (sandboxed) |

---

## What Has Been Built

### Core Platform (on `main`)

| Feature | Status | Key Files |
|---------|--------|-----------|
| Push-to-talk dictation | **Done** | `DictationCoordinator`, `AudioCaptureEngine`, `HotkeyManager` |
| WhisperKit transcription | **Done** | `TranscriptionEngine`, `ModelManager` |
| Text injection (clipboard + Cmd+V) | **Done** | `TextInjector` |
| Transcription post-processing | **Done** | `TranscriptionPostProcessor` |
| Transcription history | **Done** | `TranscriptionHistory` (JSON, 100 entries) |
| Audio device selection | **Done** | `AudioDeviceManager` |
| Global hotkey system | **Done** | `HotkeyManager` (CGEventTap + NSEvent fallback) |
| Menu bar integration | **Done** | `MenuBarManager` |
| Settings (hotkey, language, model, recording) | **Done** | `SettingsManager` (@AppStorage) |
| Recording overlay (pill + full-screen border) | **Done** | `RecordingPanelController` |
| Permissions management | **Done** | `PermissionsManager` (mic, accessibility) |
| Dependency injection | **Done** | `DependencyContainer` |
| Unified logging | **Done** | `Log` enum via `PublicLogger` |
| App Store metadata & privacy manifest | **Done** | `APP_STORE_METADATA.md`, `PrivacyInfo.xcprivacy` |
| Split entitlements (debug/release) | **Done** | `tesseract.entitlements`, `tesseractRelease.entitlements` |

### Text-to-Speech (on `main`)

| Feature | Status | Key Files |
|---------|--------|-----------|
| Qwen3-TTS voice synthesis | **Done** | `SpeechEngine`, `SpeechCoordinator` |
| Streaming generation + playback | **Done** | `AudioPlaybackManager` |
| Long-form mode (auto-segment) | **Done** | `TextSegmenter` |
| Voice anchor (consistent voice across segments) | **Done** | Voice prefix + anchor KV cache |
| Forced aligner (word-level timestamps) | **Done** | `ForcedAligner` in mlx-audio-swift |
| TTS notch overlay with word highlighting | **Done** | `TTSNotchPanelController` |
| Voice design (custom voice description) | **Done** | Via TTS parameters |
| Performance: 18-21 tok/s | **Done** | MLXFast.RoPE, code0 embed reuse |

### Image Generation (on `main`, hidden from UI)

| Feature | Status | Key Files |
|---------|--------|-----------|
| FLUX.2-klein-4B port | **Builds, untested** | `ImageGenEngine`, mlx-image-swift |
| Z-Image 6B port | **Builds, untested** | `ZImageGenEngine`, mlx-image-swift |

### AI Agent (on `feat/agent-llm`, 32 commits ahead of main)

| Feature | Status | Key Files |
|---------|--------|-----------|
| LLM inference (MLX) | **Done** | `AgentEngine`, `LLMActor` |
| Streaming text generation | **Done** | `AgentGeneration` |
| Tool call parsing (ChatML) | **Done** | `ToolCallParser`, `AgentTokenizer` |
| Multi-round tool execution loop | **Done** | `AgentRunner` (max 5 rounds) |
| 15 productivity tools | **Done** | `Tools/` (memory, goals, tasks, habits, mood, reminders) |
| Tool deduplication (within-turn + cross-turn) | **Done** | `AgentRunner` |
| Stall recovery heuristics | **Done** | Synthetic tool injection on empty rounds |
| System prompt builder (3 tiers) | **Done** | `SystemPromptBuilder` (minimal/condensed/default) |
| Chat UI with streaming display | **Done** | `AgentContentView`, `AgentConversationListView` |
| Collapsible thinking blocks | **Done** | `<think>` parsing + disclosure UI |
| Conversation persistence | **Done** | `AgentConversationStore` (JSON, per-UUID) |
| Data persistence | **Done** | `AgentDataStore` (memories, goals, tasks, habits, moods, reminders) |
| Push-to-talk voice input | **Done** | Mic capture → WhisperKit → send as message |
| TTS voice output (auto-speak) | **Done** | Routes to `SpeechCoordinator` |
| Dynamic Island notch overlay | **Done** | `AgentNotch/` (8-phase state machine) |
| Global agent hotkey (Control+Space) | **Done** | Triggers voice interaction from anywhere |
| Benchmark framework | **Done** | 7 scenarios, 100% pass rate (7/7) |
| 4 agent models supported | **Done** | Nanbeige4.1-3B, Qwen3-4B ×3 variants |
| UI refactor (@EnvironmentObject) | **Done** | Replaced 12-param dependency passing |

---

## What's In Progress

### `feat/agent-llm` → `main` merge

The agent branch is mature: 32 commits, ~10K lines, 7/7 benchmark pass rate, full voice I/O. Ready to merge.

**Pre-merge checklist:**
- [ ] Final build verification on clean checkout
- [ ] Verify no regressions in dictation and TTS features
- [ ] Resolve any merge conflicts with `main`
- [ ] Squash or preserve commit history (team preference)

---

## Roadmap

### Phase 1: Stabilize & Ship (next)

Goal: Merge agent, clean up, prepare for initial release.

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Merge `feat/agent-llm` to `main` | **P0** | Small | 32 commits, well-tested |
| Decide image gen: ship or defer | **P0** | Decision | Currently hidden; untested at runtime |
| Set up support URL | **P1** | Small | Required for App Store |
| Set up privacy policy URL | **P1** | Small | Required for App Store |
| Set up marketing URL | **P1** | Small | Required for App Store |
| USPTO trademark check for "tesse-ract" | **P1** | Small | Noted as "needs manual check" |
| End-to-end smoke test (all 3 features) | **P1** | Medium | Dictation + TTS + Agent on clean install |
| App Store submission | **P1** | Medium | Metadata ready, entitlements split |

### Phase 2: LLM-Powered Dictation (differentiator)

Goal: Wire the on-device LLM into the dictation pipeline for intelligent text cleanup. This is the original killer feature from the research doc — what users pay $15/month for with Wispr Flow.

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| LLM post-processing for dictation | **P0** | Medium | Remove fillers, fix punctuation, handle self-corrections |
| Formatting modes (raw, clean, professional) | **P1** | Medium | User selects output style |
| Context-aware formatting | **P2** | Large | Adapt to active app (Slack vs email vs code) |
| Custom vocabulary / domain hints | **P2** | Medium | Whisper prompt parameter for technical terms |

### Phase 3: Agent Polish

Goal: Improve agent reliability, UX, and capabilities.

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Tesseract animation for Dynamic Island | **P1** | Medium | Design brief exists (`docs/AGENT_TESSERACT_ANIMATION.md`) |
| Make context window limit configurable | **P1** | Small | Currently hardcoded at 60 messages |
| Parameter sweep (`scripts/bench.sh full`) | **P2** | Small | Find optimal temp/topP/repPenalty |
| Calendar integration tool | **P2** | Medium | Read/create calendar events |
| Weather tool | **P2** | Small | Local weather queries |
| Agent fine-tuning (LoRA/QLoRA) | **P3** | Large | Train on tool-calling data for higher accuracy |
| Multi-modal input (images) | **P3** | Large | Send screenshots/photos to agent |

### Phase 4: Voice & Audio

Goal: Improve the voice interaction experience.

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Silero VAD (voice activity detection) | **P1** | Medium | Auto start/stop without push-to-talk |
| Streaming transcription | **P2** | Large | Real-time text as you speak |
| Voice commands ("delete that", "new paragraph") | **P2** | Medium | Post-transcription command parsing |
| Improved audio level metering | **P3** | Small | Better visual feedback |
| Multi-language auto-detect | **P3** | Medium | WhisperKit language detection |

### Phase 5: Image Generation (if shipping)

Goal: Make image gen production-ready if the decision is to ship it.

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Runtime validation of FLUX.2 | **P0** | Medium | Currently builds but untested |
| Runtime validation of Z-Image | **P0** | Medium | Currently builds but untested |
| Memory profiling (8GB+ models) | **P1** | Medium | Must not OOM on 16GB machines |
| Quality assessment vs Python baseline | **P1** | Medium | Compare output images |
| Prompt templates / gallery UI | **P2** | Medium | Better UX than raw text input |
| Unhide from navigation | **P2** | Small | Remove UI hide flag |

### Phase 6: Platform Expansion (future)

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| iOS companion app | **P3** | Large | SwiftUI shared, MLX on iOS |
| Widget / Shortcut integration | **P3** | Medium | macOS Shortcuts for dictation |
| Spotlight integration | **P3** | Medium | Search agent memories/notes |
| Multiple Whisper model sizes | **P3** | Small | Small/Medium/Large user selection |

---

## Models Inventory

| Category | Model | Size | Status |
|----------|-------|------|--------|
| **ASR** | Whisper Large V3 Turbo | ~1.5 GB | Shipping |
| **TTS** | Qwen3-TTS VoiceDesign 1.7B | ~3.6 GB | Shipping |
| **Agent** | Nanbeige4.1-3B (8-bit) | ~4.2 GB | Shipping |
| **Agent** | Qwen3-4B Instruct 2507 | ~4.5 GB | Shipping (default) |
| **Agent** | Qwen3-4B Thinking 2507 | ~4.5 GB | Shipping |
| **Agent** | Qwen3-4B Opus Distill | ~3.8 GB | Shipping |
| **Image** | FLUX.2-klein-4B | ~8 GB | Hidden, untested |
| **Image** | Z-Image 6B | ~12 GB | Hidden, untested |

All models downloaded on-demand from Hugging Face. No models bundled in the app binary.

---

## Hotkey Map

| Hotkey | Feature | Configurable |
|--------|---------|-------------|
| Option+Space | Dictation (push-to-talk) | Yes |
| fn+Space | Speech / TTS | Yes |
| Control+Space | Agent (voice interaction) | Yes |

---

## Memory Budget (concurrent usage)

| Component | RAM | Notes |
|-----------|-----|-------|
| macOS + app | ~4 GB | Base overhead |
| WhisperKit (Large V3 Turbo) | ~1.5 GB | Loaded on first dictation |
| Qwen3-TTS 1.7B | ~3.6 GB | Loaded on first TTS |
| Agent LLM (Qwen3-4B) | ~4.5 GB | Loaded on first agent use |
| **Total peak** | **~13.6 GB** | All three loaded simultaneously |

16GB machines can run all three features. 8GB machines need model unloading between features (handled by `prepareForInference()` in `DependencyContainer`).

---

## Open Decisions

| Decision | Options | Impact |
|----------|---------|--------|
| Ship image generation? | Ship hidden / Ship visible / Cut entirely | Scope, binary size, memory |
| Merge strategy for agent branch? | Merge commit / Squash / Rebase | Git history cleanliness |
| Free vs paid? | Free / Freemium / One-time purchase | Business model |
| Agent model default? | Qwen3-4B Instruct (current) / Opus Distill | Quality vs speed tradeoff |

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Development guidelines, build commands, architecture overview |
| `ARCHITECTURE.md` | System architecture, data flow, component details |
| `APP_STORE_METADATA.md` | App Store submission metadata and review notes |
| `docs/TTS_STREAMING_AND_VOICE_CONSISTENCY.md` | TTS architecture, voice anchor, streaming pipeline |
| `docs/TTS_PERFORMANCE_INVESTIGATION.md` | TTS optimization history, benchmark data |
| `docs/IMAGE_GENERATION.md` | FLUX.2 port architecture, API gotchas |
| `docs/AGENT_FINETUNING.md` | Agent benchmark system, failure analysis, tuning history |
| `docs/AGENT_DYNAMIC_ISLAND.md` | Dynamic Island overlay design spec |
| `docs/AGENT_TESSERACT_ANIMATION.md` | Tesseract animation design brief |

---

## Branch Status

| Branch | Base | Status | Action |
|--------|------|--------|--------|
| `main` | — | Stable | Dictation + TTS + Image (hidden) |
| `feat/agent-llm` | main | **Ready to merge** | 32 commits, 7/7 benchmarks |
| `feat/image-generation` | — | Merged to main | PR #5 closed |
| `fix/tts-notch-multisegment-sync` | — | Merged to main | PR #4 closed |
| `feature/tts-with-vendored-mlx-audio` | — | Merged to main | PR #3 closed |
| `codex/overlay-recovery-fix` | — | Stale? | Investigate or delete |
