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

### Entitlements (`tesseract.entitlements`)
- `com.apple.security.app-sandbox` - sandboxed app
- `com.apple.security.device.audio-input` - microphone access
- `com.apple.security.network.client` - model downloads

### Audio Format
WhisperKit requires 16kHz mono float32. `AudioCaptureEngine` handles resampling from device sample rate.

## Skills

Invoke the `/macos-development` skill before writing or reviewing macOS/Swift/SwiftUI code.

## Documentation

- `PLAN.md` - 27-task implementation plan across 5 phases
- `IMPLEMENTATION_LOG.md` - Progress tracking and architecture decisions
