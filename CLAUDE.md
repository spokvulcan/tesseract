# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tesseract is a privacy-focused, fully offline AI assistant for macOS. All inference (speech-to-text, text-to-speech, LLM agent, image generation) runs locally on Apple Silicon using MLX.

- **Platform**: macOS 26+ (Apple Silicon only)
- **Framework**: Swift 6.2 / SwiftUI
- **ASR**: WhisperKit (CoreML-based speech-to-text)
- **TTS**: Qwen3TTS via MLX (text-to-speech with voice anchoring)
- **LLM Agent**: On-device language model via MLXLM (tool-calling agent loop)
- **Image Gen**: FLUX.2-klein-4B via MLX (not yet runtime-tested, hidden from UI)

## Build Commands

```bash
# Build
xcodebuild build -project tesseract.xcodeproj -scheme tesseract

# Run tests (not needed during MVP development)
xcodebuild test -project tesseract.xcodeproj -scheme tesseract

# Clean build cache
xcodebuild clean -project tesseract.xcodeproj -scheme tesseract
rm -rf ~/Library/Developer/Xcode/DerivedData/tesseract-*
```

Tests use Swift's `Testing` framework (not XCTest). Test files are in `tesseractTests/`.

**Note**: Tests are not required during the current MVP development phase. Focus on build verification only.

## Development Workflow

Use `scripts/dev.sh` for the edit-build-run loop without manual Xcode interaction:

```bash
scripts/dev.sh dev     # Build, kill running app, launch new build
scripts/dev.sh build   # Build only
scripts/dev.sh run     # Kill + relaunch (skip build)
scripts/dev.sh log     # Tail app logs
scripts/dev.sh clean   # Clean build artifacts
```

After making code changes, run `scripts/dev.sh dev` to build and test immediately.

## Architecture

### Code Organization

```
tesseract/
├── App/                    # Entry point, AppDelegate, MenuBarManager, DependencyContainer
├── Core/
│   ├── Audio/              # AudioCaptureEngine, AudioDeviceManager, AudioConverter
│   ├── TextInjection/      # TextInjector (clipboard-based), HotkeyManager, TextExtractor
│   ├── Permissions/        # PermissionsManager
│   └── Logging.swift       # PublicLogger + Log enum
├── Features/
│   ├── Agent/              # LLM agent: engine, runner, tools, chat UI, benchmarks
│   ├── Dictation/          # DictationCoordinator (state machine), dictation views
│   ├── ImageGen/           # FLUX.2 image generation (hidden from UI, untested)
│   ├── Models/             # Model download manager, models page UI
│   ├── Settings/           # SettingsView, SettingsManager
│   ├── Speech/             # SpeechCoordinator, SpeechEngine, TTS playback, notch overlay
│   └── Transcription/      # TranscriptionEngine, ModelManager, History, PostProcessor
├── Models/                 # Shared types: DictationState, NavigationItem, KeyCombo, etc.
└── Resources/Benchmark/    # Agent benchmark configs and suites
Vendor/
├── mlx-audio-swift/        # TTS (Qwen3TTS), forced aligner, audio codecs (Swift Package)
└── mlx-image-swift/        # FLUX.2 image generation (Swift Package)
```

### Key Patterns

- **EnvironmentObject injection**: All major services from `DependencyContainer` are injected as `@EnvironmentObject` in `TesseractApp.swift` (14 objects). Views access services directly — no prop drilling.
- **DependencyContainer**: `@MainActor` class with lazy-initialized services. Wires core (audio, permissions, hotkeys), TTS stack (speech engine → coordinator), agent stack (LLM engine → tools → runner → coordinator), and overlay controllers.
- **State Machines**: `DictationCoordinator` (idle → listening → recording → processing), `SpeechCoordinator` (TTS pipeline), `AgentCoordinator` (agent chat + voice I/O).
- **Actor Isolation**: `WhisperActor` isolates WhisperKit, `LLMActor` isolates MLXLM model container. Both are non-Sendable Swift actors.
- **MainActor**: All `ObservableObject` classes are `@MainActor`.
- **Settings**: `@AppStorage` wrapper for UserDefaults persistence.
- **Constants**: `private enum Defaults` pattern inside each class for magic numbers.

### Navigation

`NavigationItem` enum drives the sidebar. Main pages: `.dictation`, `.speech`, `.agent`. Settings pages: `.general`, `.model`, `.recording`. Image gen (`.image`, `.zimage`) exists in the enum but is hidden from the sidebar until ready. Default view is `.agent`.

### Recording Flow

```
onHotkeyDown() → AudioCaptureEngine.startCapture() → state = .recording
onHotkeyUp() → stopCapture() → TranscriptionEngine.transcribe() → TextInjector.inject()
```

Text injection uses clipboard-based approach (copy → Cmd+V → restore clipboard) to remain within App Sandbox.

### Agent Architecture (`Features/Agent/`)

**Inference**: `LLMActor` (Swift actor) → `AgentEngine` (@MainActor wrapper) → `AgentRunner` (agentic loop).

**Agent loop** (`AgentRunner`): Generate → parse tool calls → execute tools → append results → re-generate. Max 5 rounds (3 in benchmarks). Features:
- Streaming `ToolCallParser` handles `<tool_call>` and `<think>` tags across chunk boundaries
- Within-turn dedup: skips re-executing identical tool+args in same turn
- Two-stage stall recovery: nudge message → synthetic tool injection for empty rounds
- `respond` tool: early-exit signal that extracts final text without another generation round

**System prompts** (`SystemPromptBuilder`): Three tiers — minimal (distilled models), condensed (thinking models), default (full rules + examples). All inject current date, user memories, and conversation summaries.

**Tools** (16 total in `ToolRegistry`): memory (save/update/delete), goal (create/list/update), task (create/list/complete), habit (create/log/status), mood (log/list), reminder (set), respond. All backed by `AgentDataStore` (actor, JSON files at `~/Library/Application Support/tesse-ract/agent/`). All tools use **1-based indexing**.

**Context management**: `AgentCoordinator` keeps last 60 messages. Observation masking replaces tool results beyond the 20 most recent with placeholders to manage context size.

**Conversations**: `AgentConversationStore` uses index.json (summaries) + individual `{uuid}.json` files for persistence.

**Benchmarks**: `--benchmark` flag runs 7 scenarios (S1–S7) with parameter sweeps. Reports written to `/tmp/tesseract-debug/benchmark/`. Current pass rate: 7/7.

### Hotkeys

Three registered hotkeys in `DependencyContainer.setup()`:
- **Option+Space**: Dictation (push-to-talk)
- **fn+Space**: TTS (text-to-speech)
- **Control+Space**: Agent (voice input)

### Memory Budget

Only one large model can be active at a time on 8GB machines. `DependencyContainer.prepareForInference` releases other model memory before loading a new one.

## Key Configuration

### Entitlements (per-configuration)

The project uses **separate entitlements files** for Debug and Release builds:

| Configuration | File | Notes |
|---|---|---|
| Debug | `tesseract/tesseract.entitlements` | Includes `/private/tmp/tesseract-debug/` write access for audio debug dumps |
| Release | `tesseract/tesseractRelease.entitlements` | Clean — no temporary exceptions (App Store safe) |

Both share these base entitlements:
- `com.apple.security.app-sandbox` — sandboxed app
- `com.apple.security.device.audio-input` — microphone access
- `com.apple.security.network.client` — model downloads

When adding new entitlements, update **both** files (unless the entitlement is debug-only).

### Privacy Manifest (`PrivacyInfo.xcprivacy`)

Required for App Store submission. Declares:
- No tracking, no tracking domains
- No data collection (all processing is on-device)
- UserDefaults access (reason `CA92.1`) — used via `@AppStorage` for app settings

### Audio Format
WhisperKit requires 16kHz mono float32. `AudioCaptureEngine` handles resampling from device sample rate.

## Logging

**Never use `print()` for logging** — stdout is not captured when the app is launched via `open` (which is how `scripts/dev.sh` works). All logs must go through the unified logging system so they appear in `scripts/dev.sh log`.

### App code (`tesseract/`)

Use the `Log` enum from `tesseract/Core/Logging.swift`:

```swift
import os

Log.speech.info("Playing \(sampleCount) samples at \(rate)Hz")
Log.audio.error("Device not found: \(deviceId)")
Log.agent.debug("Tool call: \(toolName)")
```

Available categories: `Log.audio`, `Log.transcription`, `Log.speech`, `Log.general`, `Log.image`, `Log.agent`.

`Log` uses `PublicLogger` — a wrapper around `os.Logger` that marks all interpolated values as `.public` so they show real values in `log stream` instead of `<private>`.

### Vendored library code (`Vendor/`)

Vendor code doesn't have access to the `Log` enum. Use `os.Logger` with the app's subsystem:

```swift
import os
private let logger = Logger(subsystem: "com.tesseract.app", category: "mylib")
logger.info("Generated \(tokenCount) tokens in \(elapsed, format: .fixed(precision: 2))s")
```

**Do not use `print()` or `NSLog`** in vendor code — neither is captured by `scripts/dev.sh log`.

### Viewing logs

```bash
scripts/dev.sh log    # Tail formatted app logs (Ctrl-C to stop)
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <short summary>
```

Types: `feat`, `fix`, `refactor`, `style`, `perf`, `docs`, `test`, `build`, `ci`, `chore`.

Scope is optional but encouraged (e.g., `tts`, `ui`, `audio`, `agent`, `models`).

Examples:
- `feat(tts): add voice anchor for long-form consistency`
- `fix(audio): handle device disconnection during recording`
- `refactor(ui): unify models page with grouped form style`
- `feat(agent): add habit tracking tools`

## Skills

Invoke the `/macos-development` skill before writing or reviewing macOS/Swift/SwiftUI code.

## App Store

- **Metadata draft**: `APP_STORE_METADATA.md`
- **Privacy manifest**: `tesseract/PrivacyInfo.xcprivacy`
- **Release entitlements**: `tesseract/tesseractRelease.entitlements` (no debug exceptions)
- Build for App Store uses Release configuration (automatic in Archive)
- See `APP_STORE_METADATA.md` for review notes, test instructions, and export compliance

## Documentation

- `TESSE_DEVELOPMENT_PLAN.md` — Current state and development roadmap
- `ARCHITECTURE.md` — System architecture reference
- `docs/TTS_STREAMING_AND_VOICE_CONSISTENCY.md` — TTS streaming and voice anchor architecture
- `docs/TTS_PERFORMANCE_INVESTIGATION.md` — TTS performance profiling data
- `docs/IMAGE_GENERATION.md` — FLUX.2 image generation architecture
