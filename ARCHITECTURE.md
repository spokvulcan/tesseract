# Tesseract Architecture

This document describes the architecture of Tesseract, a privacy-focused, offline voice-to-text dictation app for macOS.

For development guidelines and build commands, see [CLAUDE.md](./CLAUDE.md).

---

## Overview

Tesseract captures audio via push-to-talk, transcribes locally using WhisperKit (CoreML-based), and injects text into any focused application. All processing happens on-device with no network requests for transcription.

**Key Principles:**
- Privacy-first: No audio data leaves the device
- Offline: Bundled Whisper model, no downloads required
- Sandboxed: Uses clipboard-based text injection to stay within App Sandbox
- Responsive: Real-time audio feedback during recording

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Tesseract                              │
├─────────────────────────────────────────────────────────────────┤
│  App Layer                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ AppDelegate  │ │ MenuBar      │ │ DependencyContainer      │ │
│  │              │ │ Manager      │ │ (Service Orchestration)  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Coordinator                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DictationCoordinator                        │   │
│  │         (State Machine: idle → recording → processing)   │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Core Services                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ AudioCapture │ │ Transcription│ │ TextInjector             │ │
│  │ Engine       │ │ Engine       │ │ (Clipboard + Cmd+V)      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ HotkeyManager│ │ Permissions  │ │ SettingsManager          │ │
│  │ (CGEventTap) │ │ Manager      │ │ (@AppStorage)            │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  External                                                       │
│  ┌──────────────┐ ┌──────────────┐                              │
│  │ WhisperKit   │ │ AVFoundation │                              │
│  │ (CoreML ASR) │ │ (Audio I/O)  │                              │
│  └──────────────┘ └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
tesseract/
├── App/                         # Application lifecycle
│   ├── TesseractApp.swift       # SwiftUI App entry point
│   ├── AppDelegate.swift        # macOS lifecycle, single instance
│   ├── DependencyContainer.swift# Service instantiation & wiring
│   ├── MenuBarManager.swift     # Status bar menu
│   └── *PanelController.swift   # Recording overlay windows
│
├── Core/                        # Platform services
│   ├── Audio/
│   │   ├── AudioCaptureEngine.swift   # AVAudioEngine recording
│   │   ├── AudioDeviceManager.swift   # Input device enumeration
│   │   └── AudioConverter.swift       # Format conversion
│   ├── TextInjection/
│   │   ├── TextInjector.swift         # Clipboard-based paste
│   │   └── HotkeyManager.swift        # Global hotkey listener
│   └── Permissions/
│       └── PermissionsManager.swift   # Mic & Accessibility checks
│
├── Features/                    # Feature modules
│   ├── Dictation/
│   │   ├── DictationCoordinator.swift # Main state machine
│   │   └── Views/                     # Recording UI components
│   ├── Transcription/
│   │   ├── TranscriptionEngine.swift  # WhisperKit wrapper
│   │   ├── ModelManager.swift         # Bundled model location
│   │   ├── TranscriptionHistory.swift # Persistence
│   │   └── TranscriptionPostProcessor.swift # Text cleanup
│   └── Settings/
│       ├── SettingsManager.swift      # @AppStorage wrapper
│       └── SettingsView.swift         # Settings UI
│
└── Models/                      # Data types
    ├── DictationState.swift     # State enum
    ├── DictationError.swift     # Error types
    ├── AudioData.swift          # Audio sample container
    ├── TranscriptionResult.swift# Transcription output
    ├── KeyCombo.swift           # Hotkey configuration
    └── ...
```

---

## Core Concepts

### 1. State Machine (DictationCoordinator)

The application's main flow is controlled by a state machine:

```
         onHotkeyDown()              onHotkeyUp()
              │                           │
              ▼                           ▼
┌──────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────┐
│ idle │──│recording│──│processing│──│  inject   │──│ idle │
└──────┘  └─────────┘  └──────────┘  └───────────┘  └──────┘
    ▲                                                   │
    └──────────────────── error ◄──────────────────────┘
                      (auto-reset 3s)
```

**States:**
- `idle` - Ready for input
- `recording` - Capturing audio (hotkey held)
- `processing` - Running WhisperKit transcription
- `error(String)` - Error with auto-reset

### 2. Dependency Injection (DependencyContainer)

All services are instantiated lazily in `DependencyContainer`:

```swift
@MainActor
final class DependencyContainer: ObservableObject {
    lazy var settingsManager = SettingsManager.shared
    lazy var audioCaptureEngine = AudioCaptureEngine()
    lazy var transcriptionEngine = TranscriptionEngine()
    lazy var textInjector = TextInjector()
    lazy var hotkeyManager = HotkeyManager()
    // ...

    lazy var dictationCoordinator: DictationCoordinator = {
        DictationCoordinator(
            audioCapture: audioCaptureEngine,
            transcriptionEngine: transcriptionEngine,
            textInjector: textInjector,
            // ...
        )
    }()
}
```

### 3. Actor Isolation

Thread safety is ensured through Swift concurrency:

- **@MainActor**: All UI-related classes (coordinators, managers, engines)
- **WhisperActor**: Isolates non-Sendable WhisperKit library
- **@unchecked Sendable**: Audio buffer classes with manual NSLock

```swift
// Audio tap runs on real-time thread, needs manual synchronization
nonisolated final class SampleBuffer: @unchecked Sendable {
    private var samples: [Float] = []
    private let lock = NSLock()
    // ...
}

// WhisperKit isolated in dedicated actor
actor WhisperActor {
    private var whisperKit: WhisperKit?
    func transcribe(_ samples: [Float], language: String) async throws -> TranscriptionResult
}
```

### 4. Reactive State (Combine)

UI updates flow through Combine publishers:

```swift
// DictationCoordinator publishes state
@Published private(set) var state: DictationState = .idle
@Published private(set) var lastTranscription: String = ""

// Views observe state
struct DictationContentView: View {
    @ObservedObject var coordinator: DictationCoordinator

    var body: some View {
        switch coordinator.state {
        case .idle: IdleView()
        case .recording: RecordingView()
        // ...
        }
    }
}
```

---

## Data Flow

### Recording to Text Injection

```
1. User presses hotkey (e.g., Option+Space)
   └─► HotkeyManager.onHotkeyDown()
       └─► DictationCoordinator.startRecording()
           └─► AudioCaptureEngine.startCapture()
               └─► AVAudioEngine tap collects samples

2. User releases hotkey
   └─► HotkeyManager.onHotkeyUp()
       └─► DictationCoordinator.stopRecordingAndProcess()
           ├─► AudioCaptureEngine.stopCapture() → AudioData
           └─► TranscriptionEngine.transcribe(audioData)
               └─► WhisperActor → WhisperKit inference
                   └─► TranscriptionResult

3. Post-processing
   └─► TranscriptionPostProcessor.process(text)
       └─► Remove artifacts, fix capitalization
           └─► TextInjector.inject(cleanedText)
               ├─► Copy to clipboard
               └─► Simulate Cmd+V (if not focused on own window)
```

### Audio Format Pipeline

```
Microphone (48kHz stereo)
    │
    ▼
AVAudioEngine input tap (device sample rate, mono float32)
    │
    ▼
SampleBuffer (thread-safe accumulation)
    │
    ▼
Resample to 16kHz (linear interpolation)
    │
    ▼
WhisperKit (16kHz mono float32 required)
```

---

## Key Components

### AudioCaptureEngine

Handles microphone access and audio capture:

- Creates AVAudioEngine per recording session
- Installs tap on input node for sample collection
- Real-time RMS audio level metering
- Resamples to 16kHz for WhisperKit

### TranscriptionEngine

Wraps WhisperKit for speech recognition:

- Loads bundled CoreML model (Whisper Large V3 Turbo)
- Configures compute: GPU encoder, Neural Engine decoder
- Greedy decoding with language-specific prefill
- Returns text, segments, and timing info

### TextInjector

Clipboard-based text injection:

- Saves original clipboard contents
- Copies transcribed text to clipboard
- Simulates Cmd+V via CGEvent (when external app focused)
- Restores original clipboard (configurable)

### HotkeyManager

Global hotkey detection with graceful degradation:

- **Primary**: CGEventTap (requires Accessibility permission)
  - Can suppress events from reaching other apps
- **Fallback**: NSEvent global monitors
  - Observation only, no suppression
- Supports modifier-only hotkeys (e.g., Option alone)

---

## Settings & Persistence

### SettingsManager

Uses `@AppStorage` for UserDefaults persistence:

```swift
@AppStorage("hotkey") var hotkeyData: Data?
@AppStorage("language") var language: String = "auto"
@AppStorage("maxRecordingDuration") var maxRecordingDuration: Double = 30
@AppStorage("playSounds") var playSounds: Bool = true
// ...
```

### TranscriptionHistory

JSON file persistence:

- Location: `~/Library/Application Support/Tesseract/transcription_history.json`
- Stores up to 100 entries (configurable)
- Auto-prunes oldest entries

---

## Platform Integration

### Entitlements

```xml
com.apple.security.app-sandbox          <!-- Sandboxed app -->
com.apple.security.device.audio-input   <!-- Microphone access -->
com.apple.security.network.client       <!-- Model downloads (future) -->
```

### Permissions

| Permission | Purpose | Fallback |
|------------|---------|----------|
| Microphone | Audio capture | Required, no fallback |
| Accessibility | Hotkey suppression | NSEvent monitors (no suppression) |

### Menu Bar

- Status item with waveform icon
- Quick actions: Toggle recording, Settings, Quit
- Updates state based on DictationCoordinator

---

## Error Handling

Errors are typed and provide recovery suggestions:

```swift
enum DictationError: LocalizedError {
    case microphonePermissionDenied
    case modelNotLoaded
    case audioCaptureFailed(String)
    case transcriptionFailed(String)
    case noSpeechDetected
    case recordingTooShort
    // ...

    var recoverySuggestion: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Grant microphone access in System Settings > Privacy & Security"
        // ...
        }
    }
}
```

Error states auto-reset after 3 seconds to return to idle.

---

## Future Considerations

- **Silero VAD**: Voice Activity Detection for automatic start/stop
- **Multiple Models**: Support for different Whisper model sizes
- **Streaming**: Real-time transcription during recording
- **Custom Vocabulary**: User-defined word corrections
