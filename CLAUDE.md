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
scripts/dev.sh dev         # Build (Debug) + kill + relaunch — fast iteration (~10s)
scripts/dev.sh dev-release # Build (Release, incremental) + kill + relaunch — perf testing
scripts/dev.sh build       # Build only
scripts/dev.sh run         # Kill + relaunch (skip build)
scripts/dev.sh log         # Tail app logs
scripts/dev.sh clean       # Clean build artifacts
scripts/dev.sh archive     # Create .xcarchive for App Store
```

Tests use Swift `Testing` framework (not XCTest), in `tesseractTests/`. Not required during MVP — focus on build verification.

## Architecture

### Observation Model

The app uses Swift's `@Observable` (Observation framework) as the primary state model. Key types:

- **`@Observable`**: `SettingsManager`, `DictationCoordinator`, `SpeechCoordinator`, `SpeechEngine`, `TranscriptionEngine`, `TranscriptionHistory`, `AudioCaptureEngine`, `AgentCoordinator`, `AgentEngine`, `Agent`, `AgentState`, `OverlayState`
- **Still `ObservableObject`** (lower-priority, not yet migrated): `DependencyContainer`, `PermissionsManager`, `ModelDownloadManager`, `AgentConversationStore`, `ImageGenEngine`, `ZImageGenEngine`, `AudioPlaybackManager`, `MenuBarManager`, `AudioDeviceManager`, `HotkeyManager`, `TextInjector`, `AudioLevelBuffer`
- **Inject `@Observable` types** with `.environment(instance)`, consume with `@Environment(Type.self)`. Use `@Bindable var x = x` in body for `$x` bindings.
- **Inject `ObservableObject` types** with `.environmentObject(instance)`, consume with `@EnvironmentObject`.
- **Do NOT use `@AppStorage` inside `@Observable`** — it fails to compile (backing-storage collision). Use manual `UserDefaults` with `didSet` persistence.

### Key Patterns

- **Window Scene**: Uses `Window("Tesseract", id: "main")` (single-instance, not `WindowGroup`). Settings live in the sidebar, not a separate `Settings` scene.
- **DependencyContainer** (`App/`): `@MainActor`, lazy-initialized composition root. Wires all services. Scoped injection via `Core/ViewModifiers.swift` — 5 groups: core, dictation, speech, agent, model.
- **AgentFactory** (`Features/Agent/AgentFactory.swift`): Extracts the multi-step agent bootstrap (package discovery, tool/skill registration, prompt assembly, compaction wiring) out of the container.
- **Actor Isolation**: `WhisperActor` (WhisperKit), `LLMActor` (MLXLM), `TTSActor` (Qwen3TTS), `ContextManager` (compaction). All `@Observable`/`ObservableObject` classes are `@MainActor`.
- **State Machines**: `DictationCoordinator`, `SpeechCoordinator`, `AgentCoordinator`.
- **Settings**: `SettingsManager` is `@Observable @MainActor` with manual `UserDefaults` persistence (`didSet` + `register(defaults:)` in init). No singleton — injected via `DependencyContainer`.
- **Platform Adapters** (`Platform/`): All AppKit bridge code — `HotkeyManager`, `TextInjector`, `MenuBarManager`, overlay/notch panel controllers, `OverlayScreenLocator`.
- **Non-view Observation**: Uses Swift 6.2 `Observations` async sequences for observing `@Observable` state in non-SwiftUI code (DependencyContainer, AppDelegate, MenuBarManager).
- **Hotkeys**: Option+Space (dictation), fn+Space (TTS), Control+Space (agent). All configurable.
- **Memory Budget**: Target 20GB unified memory. LLM + TTS are co-resident (both loaded simultaneously).

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
- `ARCHITECTURE.md` — System architecture
- `docs/macos26-swiftui-architecture-review.md` — Architecture review and modernization plan (Observation migration, Window scene, modularity decisions)
- `docs/` — TTS streaming/performance, image gen, agent prompt engineering research, tool call XML reconstruction
