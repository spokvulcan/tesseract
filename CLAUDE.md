# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tesseract is a privacy-focused, offline voice-to-text dictation app for macOS. It captures audio via push-to-talk, transcribes locally using WhisperKit, and injects text into any focused application.

- **Platform**: macOS 26+ (Apple Silicon only)
- **Framework**: Swift 6.2 / SwiftUI
- **ASR Engine**: WhisperKit (CoreML-based)
- **VAD**: Silero VAD (planned)

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
│   ├── TextInjection/      # TextInjector (clipboard-based), HotkeyManager
│   └── Permissions/        # PermissionsManager
├── Features/
│   ├── Dictation/          # DictationCoordinator (state machine), views
│   ├── Transcription/      # TranscriptionEngine, ModelManager, History
│   └── Settings/           # SettingsView, SettingsManager
└── Models/                 # DictationState, TranscriptionResult, KeyCombo, etc.
```

### Key Patterns

- **Dependency Injection**: `DependencyContainer` manages all service instantiation
- **State Machine**: `DictationCoordinator` manages: idle → listening → recording → processing
- **Actor Isolation**: `WhisperActor` isolates WhisperKit access (not Sendable)
- **MainActor**: All `ObservableObject` classes are `@MainActor`
- **Settings**: `@AppStorage` wrapper for UserDefaults persistence

### Recording Flow

```
onHotkeyDown() → AudioCaptureEngine.startCapture() → state = .recording
onHotkeyUp() → stopCapture() → TranscriptionEngine.transcribe() → TextInjector.inject()
```

### Text Injection Architecture

Uses clipboard-based injection (copy → simulate Cmd+V → restore) instead of Accessibility APIs to remain within App Sandbox constraints.

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
Log.general.debug("State changed to \(newState)")
```

Available categories: `Log.audio`, `Log.transcription`, `Log.speech`, `Log.general`.

`Log` uses `PublicLogger` — a wrapper around `os.Logger` that marks all interpolated values as `.public` so they show real values in `log stream` instead of `<private>`.

### Vendored library code (`Vendor/`)

Vendor code doesn't have access to the `Log` enum. Use `NSLog` instead:

```swift
NSLog("[MyLib] Generated %d tokens in %.2fs", tokenCount, elapsed)
```

`NSLog` output is captured by `scripts/dev.sh log` (it filters `process == "tesseract" AND subsystem == ""`). It shows without a category tag but is still visible.

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

Scope is optional but encouraged (e.g., `tts`, `ui`, `audio`, `models`).

Examples:
- `feat(tts): add voice anchor for long-form consistency`
- `fix(audio): handle device disconnection during recording`
- `refactor(ui): unify models page with grouped form style`
- `perf(tts): reuse code0 embeddings across predictor steps`

## Skills

Invoke the `/macos-development` skill before writing or reviewing macOS/Swift/SwiftUI code.

## App Store

- **Metadata draft**: `APP_STORE_METADATA.md`
- **Privacy manifest**: `tesseract/PrivacyInfo.xcprivacy`
- **Release entitlements**: `tesseract/tesseractRelease.entitlements` (no debug exceptions)
- Build for App Store uses Release configuration (automatic in Archive)
- See `APP_STORE_METADATA.md` for review notes, test instructions, and export compliance

## Documentation

- `PLAN.md` - 27-task implementation plan across 5 phases
- `IMPLEMENTATION_LOG.md` - Progress tracking and architecture decisions
