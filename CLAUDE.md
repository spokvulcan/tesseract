# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tesseract Agent is a privacy-focused, fully offline AI assistant for macOS. All inference runs locally on Apple Silicon using MLX.

- **Platform**: macOS 26+ (Apple Silicon only), Swift 6.2 / SwiftUI
- **ASR**: WhisperKit (CoreML, 16kHz mono float32)
- **TTS**: Qwen3TTS via MLX (voice anchoring)
- **LLM Agent**: On-device MLXLM (tool-calling agent loop)
- **Image Gen**: FLUX.2-klein-4B via MLX (hidden from UI, untested)

## Build & Dev

```bash
scripts/dev.sh dev       # Build (Release, incremental) + kill + relaunch
scripts/dev.sh build     # Build only
scripts/dev.sh run       # Kill + relaunch (skip build)
scripts/dev.sh log       # Tail app logs
scripts/dev.sh clean     # Clean build artifacts
scripts/dev.sh archive   # Create .xcarchive for App Store
```

Tests use Swift `Testing` framework (not XCTest), in `tesseractTests/`. Not required during MVP — focus on build verification.

## Architecture

### Key Patterns

- **DependencyContainer** (`App/`): `@MainActor`, lazy-initialized. Wires all services. 14 objects injected via `.injectDependencies(from:)` in `Core/ViewModifiers.swift`.
- **Actor Isolation**: `WhisperActor` (WhisperKit), `LLMActor` (MLXLM), `ContextManager` (compaction). All `ObservableObject` classes are `@MainActor`. `Agent` uses `@Observable`.
- **State Machines**: `DictationCoordinator`, `SpeechCoordinator`, `AgentCoordinator`, `AgentNotchState`.
- **Settings**: `@AppStorage` via `SettingsManager`. Constants use `private enum Defaults` pattern.
- **Hotkeys**: Option+Space (dictation), fn+Space (TTS), Control+Space (agent). All configurable.
- **Memory Budget**: One large model at a time on 8GB. `prepareForInference` releases others first.

### Agent Architecture (`Features/Agent/`)

**Inference stack**: `LLMActor` → `AgentEngine` → `Agent` (double-loop orchestrator).

**Double-loop** (`Core/AgentLoop.swift`): Outer loop handles follow-ups, inner loop handles tool calls + steering (user interrupts). No fixed round limit — runs until no more tool calls. Tools execute sequentially; steering can interrupt remaining tools mid-turn.

**4 built-in tools** (`Tools/BuiltIn/`): `read` (2K lines/50KB, supports images), `write` (atomic, auto-creates dirs), `edit` (exact match + fuzzy, unified diff), `ls` (500 entries/50KB). All sandboxed via `PathSandbox`. Extension tools aggregated by `ToolRegistry`.

**Extensibility**:
- **Packages** (`Packages/`): Discoverable via `package.json`. Provide skills, context files, system prompt overrides, seed data, extensions.
- **Extensions** (`Extensions/`): Plugins contributing tools. First registration wins on name conflicts.
- **Skills** (`Context/SkillRegistry.swift`): Markdown files with YAML frontmatter. Formatted as XML in system prompt.

**System prompt assembly** (`Context/SystemPromptAssembler.swift`): SYSTEM.md override → APPEND_SYSTEM.md → context files (AGENTS.md/CLAUDE.md) → skills → date/time + working dir.

**Context compaction** (`Context/ContextManager.swift`): Token-based — 120K window, 16K reserve, 20K recent tokens kept. Summarizes old messages via LLM.

**Events**: 13 types (lifecycle, turns, transforms, streaming, tool execution). Buffered thread-safely, drained on MainActor.

**Benchmarks**: `--benchmark` flag runs 14 scenarios (S1–S4, S6–S15). Reports to sandbox tmp dir.

**Conversations**: `AgentConversationStore` — index.json + `{uuid}.json` at `~/Library/Application Support/Tesseract Agent/agent/conversations/`.

## Entitlements & Privacy

Separate files for Debug (`tesseract/tesseract.entitlements`) and Release (`tesseract/tesseractRelease.entitlements`) — currently identical. Update **both** when adding entitlements.

Entitlements: app-sandbox, audio-input, network-client. Privacy manifest (`PrivacyInfo.xcprivacy`): no tracking, no data collection, UserDefaults access only. Info.plist declares `NSMicrophoneUsageDescription` and `NSAccessibilityUsageDescription`.

## Logging

**Never use `print()`** — not captured when launched via `open`. Use unified logging:

```swift
// App code: use Log enum (Core/Logging.swift)
Log.agent.debug("Tool call: \(toolName)")  // categories: .audio, .transcription, .speech, .general, .image, .agent

// Vendor code: use os.Logger directly
private let logger = Logger(subsystem: "app.tesseract.agent", category: "mylib")
```

`Log` uses `PublicLogger` — marks all interpolations `.public` for `log stream` visibility.

## Conventions

**Commits**: [Conventional Commits](https://www.conventionalcommits.org/) — `<type>(<scope>): <summary>`. Types: `feat`, `fix`, `refactor`, `style`, `perf`, `docs`, `test`, `build`, `ci`, `chore`.

**Skills**: Invoke `/macos-development` before writing or reviewing macOS/Swift/SwiftUI code.

**App Store**: See `APP_STORE_METADATA.md` for metadata, review notes, and export compliance.

## Documentation

- `TESSE_DEVELOPMENT_PLAN.md` — Roadmap
- `ARCHITECTURE.md` — System architecture (dictation focus)
- `docs/` — TTS streaming/performance, image gen, agent prompt engineering research, tool call XML reconstruction
