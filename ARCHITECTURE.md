# Tesseract Architecture

This document describes the architecture of Tesseract Agent, a privacy-focused, fully offline AI assistant for macOS.

For development guidelines and build commands, see [CLAUDE.md](./CLAUDE.md).
For the detailed architecture review and modernization rationale, see [docs/macos26-swiftui-architecture-review.md](./docs/macos26-swiftui-architecture-review.md).

---

## Overview

Tesseract Agent runs entirely on-device on Apple Silicon. It provides dictation (speech-to-text), text-to-speech, and an LLM-powered agent with tool-calling capabilities. All inference uses local models: WhisperKit (CoreML) for ASR, MLX for LLM and TTS.

**Key Principles:**
- Privacy-first: No audio or text data leaves the device
- Offline: All models run locally on Apple Silicon
- Sandboxed: App Sandbox with clipboard-based text injection
- Responsive: Real-time audio feedback, streaming inference

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Tesseract Agent                             │
├──────────────────────────────────────────────────────────────────────┤
│  App Layer                                                           │
│  ┌──────────────┐ ┌──────────────────────────┐ ┌──────────────────┐  │
│  │ TesseractApp │ │ DependencyContainer      │ │ AppDelegate      │  │
│  │ Window scene │ │ (Composition Root)       │ │ (AppKit bridge)  │  │
│  └──────────────┘ └──────────────────────────┘ └──────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│  Coordinators (@Observable, @MainActor)                              │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────────────────┐  │
│  │  Dictation     │ │  Speech        │ │  Agent                   │  │
│  │  Coordinator   │ │  Coordinator   │ │  Coordinator             │  │
│  └────────────────┘ └────────────────┘ └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│  Engines (@Observable, @MainActor)                                   │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────────────────┐  │
│  │ Transcription  │ │ Speech         │ │  Agent                   │  │
│  │ Engine         │ │ Engine         │ │  Engine                  │  │
│  └────────────────┘ └────────────────┘ └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│  Actors (inference isolation)                                        │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────┐             │
│  │ WhisperActor   │ │ TTSActor       │ │ LLMActor     │             │
│  │ (CoreML ASR)   │ │ (MLX TTS)      │ │ (MLX LLM)    │             │
│  └────────────────┘ └────────────────┘ └──────────────┘             │
├──────────────────────────────────────────────────────────────────────┤
│  Platform Adapters (AppKit)                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │ HotkeyMgr    │ │ TextInjector │ │ MenuBarMgr   │ │ Panel      │  │
│  │ (CGEventTap) │ │ (Clipboard)  │ │ (NSStatusBar)│ │ Controllers│  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
tesseract/
├── App/                         # Application lifecycle
│   ├── TesseractApp.swift       # SwiftUI App entry (Window scene)
│   ├── AppDelegate.swift        # macOS lifecycle, single instance
│   └── DependencyContainer.swift# Composition root, service wiring
│
├── Core/                        # Shared services
│   ├── Audio/
│   │   ├── AudioCaptureEngine.swift   # @Observable, AVAudioEngine recording
│   │   ├── AudioDeviceManager.swift   # Input device enumeration
│   │   └── AudioConverter.swift       # Format conversion
│   ├── Permissions/
│   │   └── PermissionsManager.swift   # Mic & Accessibility checks
│   ├── ViewModifiers.swift      # Scoped dependency injection
│   └── Logging.swift            # Unified logging (Log enum)
│
├── Platform/                    # AppKit bridge code
│   ├── HotkeyManager.swift           # Global hotkey listener (CGEventTap)
│   ├── TextInjector.swift             # Clipboard-based paste injection
│   ├── TextExtractor.swift            # Selected text extraction
│   ├── MenuBarManager.swift           # Status bar menu (NSStatusItem)
│   ├── OverlayPanelController.swift   # Recording pill overlay (NSPanel)
│   ├── FullScreenBorderPanelController.swift  # Full-screen glow overlay
│   ├── OverlayScreenLocator.swift     # Screen detection for overlays
│   ├── AgentNotchPanelController.swift# Agent notch overlay
│   └── TTSNotchPanelController.swift  # TTS notch overlay
│
├── Features/                    # Feature modules
│   ├── Dictation/
│   │   ├── DictationCoordinator.swift # @Observable state machine
│   │   └── Views/                     # Recording UI components
│   ├── Speech/
│   │   ├── SpeechCoordinator.swift    # @Observable TTS orchestrator
│   │   ├── SpeechEngine.swift         # @Observable, wraps TTSActor
│   │   └── Views/ + NotchOverlay/     # TTS UI
│   ├── Transcription/
│   │   ├── TranscriptionEngine.swift  # @Observable, wraps WhisperActor
│   │   ├── TranscriptionHistory.swift # @Observable, JSON persistence
│   │   └── TranscriptionPostProcessor.swift
│   ├── Agent/
│   │   ├── AgentCoordinator.swift     # @Observable UI bridge
│   │   ├── AgentEngine.swift          # @Observable, wraps LLMActor
│   │   ├── AgentFactory.swift         # Bootstrap: packages, tools, prompt
│   │   ├── Core/                      # Agent loop, state, config
│   │   ├── Tools/                     # Built-in + extension tools
│   │   ├── Context/                   # System prompt, skills, compaction
│   │   └── Views/                     # Chat UI
│   ├── Settings/
│   │   ├── SettingsManager.swift      # @Observable, manual UserDefaults
│   │   └── SettingsView.swift         # Settings UI sections
│   └── Models/                        # Model download management
│
└── Models/                      # Shared data types
    ├── DictationState.swift
    ├── NavigationItem.swift     # Sidebar routing enum
    ├── KeyCombo.swift
    └── ...
```

---

## Observation and Data Flow

### State Model

The app uses Swift's Observation framework (`@Observable`) for all primary state types. This replaced the older `ObservableObject` + `@Published` + Combine model.

**SwiftUI views** consume `@Observable` types via `@Environment(Type.self)`:

```swift
struct DictationContentView: View {
    @Environment(DictationCoordinator.self) private var coordinator
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        // For bindings, use @Bindable:
        @Bindable var settings = settings
        Toggle("Play sounds", isOn: $settings.playSounds)
    }
}
```

**Non-view code** (DependencyContainer, AppDelegate, MenuBarManager) observes `@Observable` state using Swift 6.2's `Observations` async sequence:

```swift
Task { [weak self] in
    guard let self else { return }
    for await state in Observations { self.dictationCoordinator.state } {
        self.overlayPanelController.handleStateChange(state)
    }
}
```

### Settings Persistence

`SettingsManager` is the `@Observable @MainActor` **Settings Facade**: it keeps one
bindable stored property per setting (so SwiftUI `$settings.foo` bindings and
per-property Observation work), but persistence lives behind a **Settings Store**
seam injected *below* the facade. Each `didSet` forwards to the store via the
property's `Setting` in the **Settings Catalogue**; the catalogue is the single
source of truth for every default (no more `register(defaults:)`). See ADR-0002 and
`CONTEXT.md` → Language → Settings persistence.

```swift
protocol SettingsStore {                       // typed, default-on-read; no register(defaults:)
    func bool(for key: String, default: Bool) -> Bool
    func set<V>(_ value: V, for key: String)
    func setOptional(_ value: String?, for key: String)   // nil ⇒ remove the key
    // … int/double/string/optionalString …
}

enum SettingsCatalogue {                       // one Setting per persisted primitive; the only home for a default
    static let playSounds = Setting.bool("playSounds", default: true)
    // … ~37 settings …
}

@Observable @MainActor
final class SettingsManager {
    private let store: any SettingsStore
    var playSounds: Bool {                      // declared WITHOUT a default (see below)
        didSet { SettingsCatalogue.playSounds.write(playSounds, to: store) }
    }
    init(store: any SettingsStore = UserDefaultsSettingsStore()) {
        self.store = store
        self.playSounds = SettingsCatalogue.playSounds.load(from: store)   // direct first assignment skips didSet
        // … one per property … then normalizePersistedSelectionsIfNeeded()
    }
}
```

Two adapters make the seam real: `UserDefaultsSettingsStore` (the only production
Swift that calls `UserDefaults`; owns default-on-read via `object(forKey:) == nil`)
and `InMemorySettingsStore` (tests — hermetic, parallel-safe). The two genuine side
effects (launch-at-login via `SMAppService`, dock visibility via `NSApp`) stay in
the facade's `didSet`, above the store.

**`@Observable` + `didSet` in `init`:** under `@Observable` a property re-assignment
in `init` *fires* `didSet`; only a *direct, property-named first* assignment skips it
(via the storage-restrictions init accessor), and only when the property has no
declaration default. So properties are declared `var foo: Bool` (not `= false`) and
hydrated by `self.foo = Catalogue.foo.load(...)` — construction performs zero store
writes and runs no side effects. The lone exception is stale-value migration, which
runs after hydration and so persists through the store.

`@AppStorage` is NOT compatible with `@Observable` (compiler error), which is why
the facade keeps explicit stored properties rather than property wrappers.

### Dependency Injection

`DependencyContainer` creates all services lazily and injects them into the SwiftUI hierarchy via scoped modifiers:

```swift
.injectDependencies(from: container)
// Expands to:
//   .injectCoreDependencies(...)       — settings, permissions, container
//   .injectDictationDependencies(...)  — coordinator, engine, history, audio
//   .injectSpeechDependencies(...)     — coordinator, engine
//   .injectAgentDependencies(...)      — coordinator, engine, conversation store
//   .injectModelDependencies(...)      — download manager, inference arbiter
```

AppKit consumers (MenuBarManager, panel controllers) receive dependencies via constructor injection — they cannot use `@Environment`.

---

## Core Concepts

### 1. Window Scene

The app uses `Window("Tesseract", id: "main")` — a single-instance window. This avoids the multi-window workarounds needed with `WindowGroup`. Settings live in the sidebar (`NavigationSplitView`), not a separate `Settings` scene.

### 2. State Machines

Coordinators manage user-facing flows as state machines:

- **DictationCoordinator**: idle → recording → processing → inject → idle
- **SpeechCoordinator**: idle → capturingText → generating → streaming/playing → idle
- **AgentCoordinator**: bridges the Agent double-loop to SwiftUI via cached `ChatRow` arrays

### 3. Actor Isolation

Thread safety uses Swift concurrency:

- **@MainActor**: All coordinators, engines, managers, views
- **Actors**: `WhisperActor` (CoreML), `TTSActor` (MLX TTS), `LLMActor` (MLX LLM), `ContextManager` (compaction)
- **@unchecked Sendable**: `SampleBuffer`, `AudioLevelRelay` (manual NSLock for real-time audio thread)

### 4. Agent Architecture

**Inference stack**: `LLMActor` → `AgentEngine` → `Agent` (double-loop orchestrator).

**Agent bootstrap** (`AgentFactory.makeAgent()`): Discovers packages → registers extensions → discovers skills → loads context files → assembles system prompt → wires compaction → creates Agent instance.

**Double-loop** (`Core/AgentLoop.swift`): Outer loop handles follow-ups, inner loop handles tool calls + steering. No fixed round limit.

**4 built-in tools**: `read`, `write`, `edit`, `ls` — all sandboxed via `PathSandbox`.

**Extensibility**: Packages, Extensions (tool plugins), Skills (markdown with YAML frontmatter).

### 5. Platform Adapters

All AppKit bridging lives in `Platform/`. These are the features that SwiftUI cannot cover:

- Global hotkeys (CGEventTap)
- Clipboard text injection (CGEvent Cmd+V simulation)
- Always-on-top overlay panels (NSPanel)
- Menu bar status item (NSStatusItem)
- Notch overlays for TTS/Agent

Panel controllers receive state via push methods (`handleStateChange`, `handleAudioLevelChange`) — they are publisher-agnostic. The DependencyContainer owns the `Observations` subscriptions and pushes values.

---

## Data Flow

### Recording to Text Injection

```
1. User presses hotkey (Option+Space)
   └─► HotkeyManager.onHotkeyDown()
       └─► DictationCoordinator.startRecording()
           └─► AudioCaptureEngine.startCapture()

2. User releases hotkey
   └─► DictationCoordinator.stopRecordingAndProcess()
       ├─► AudioCaptureEngine.stopCapture() → AudioData
       └─► TranscriptionEngine.transcribe(audioData)
           └─► WhisperActor → WhisperKit inference → TranscriptionResult

3. Post-processing
   └─► TranscriptionPostProcessor → TextInjector.inject()
       ├─► Copy to clipboard
       └─► Simulate Cmd+V
```

### Audio Format Pipeline

```
Microphone (48kHz stereo) → AVAudioEngine tap (device rate, mono float32)
  → SampleBuffer (thread-safe) → Resample to 16kHz → WhisperKit
```

---

## Decisions and Rationale

Key architectural decisions documented in `docs/macos26-swiftui-architecture-review.md`:

- **`Window` not `WindowGroup`**: Product intent is a single main window. `Window` eliminates 5 workarounds for multi-window suppression.
- **`@Observable` not `ObservableObject`**: Observation framework tracks property access precisely (no coarse object-wide invalidation). Better SwiftUI performance.
- **No `@AppStorage` in `@Observable`**: Compiler incompatibility. All settings use manual `UserDefaults` with `didSet`.
- **No `SettingsManager.shared` singleton**: Injected via `DependencyContainer`. AppKit consumers get it via constructor injection.
- **`Observations` async sequence for non-view code**: Replaces Combine `$property.sink` for observing `@Observable` types outside SwiftUI views.
- **`AgentFactory` separate from container**: Container wires dependencies; factory orchestrates multi-step bootstrap.
- **Panel controllers are publisher-agnostic**: Accept values via `handleStateChange`/`handleAudioLevelChange` methods. The subscription mechanism lives in DependencyContainer and can change independently.
- **Defer Agent package extraction**: Don't extract `Features/Agent` into a separate Swift package until dependency boundaries are clearer.
- **Defer separate Settings scene**: Keep settings in the main window sidebar.
- **Defer UI automation**: Invest in coordinator unit tests first.
