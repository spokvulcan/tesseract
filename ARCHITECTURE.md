# Tesseract Architecture

This document describes the architecture of Tesseract Agent, a privacy-focused, fully offline AI assistant for macOS.

For development guidelines and build commands, see [CLAUDE.md](./CLAUDE.md).
For domain vocabulary, see [CONTEXT.md](./CONTEXT.md); for decision records, see `docs/adr/`.

---

## Overview

Tesseract Agent runs entirely on-device on Apple Silicon. It provides dictation (speech-to-text), text-to-speech, an LLM-powered agent with tool-calling capabilities, and a local OpenAI-compatible HTTP server accelerated by a tiered KV prefix cache. All inference uses local models: WhisperKit (CoreML) for ASR, MLX for LLM and TTS.

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
│  │ Transcription  │ │ SpeechEngine   │ │  Agent                   │  │
│  │ Engine         │ │ Presenter      │ │  Engine                  │  │
│  └────────────────┘ └────────────────┘ └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│  Model adapters behind ports (actor-isolated inference)              │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────┐  │
│  │ SpeechRecognizer     │ │ SpeechEngine (pkg)   │ │ LLMActor     │  │
│  │ WhisperKit (ASR)     │ │ Qwen3 TTS (MLX)      │ │ MLX LLM      │  │
│  └──────────────────────┘ └──────────────────────┘ └──────────────┘  │
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

Representative, not exhaustive — trust the file system over this listing.

```
tesseract/
├── App/                         # Application lifecycle
│   ├── TesseractApp.swift       # SwiftUI App entry (Window scene)
│   ├── AppDelegate.swift        # macOS lifecycle, single instance, window management
│   ├── AppBindings.swift        # App Bindings: launch sequence + subscription rules
│   ├── AppTerminationCoordinator.swift # Teardown ordering (closure-struct steps)
│   └── DependencyContainer.swift# Composition root, pure wiring
│
├── Core/                        # Shared services
│   ├── Audio/
│   │   ├── AudioCaptureEngine.swift   # @Observable, AVAudioEngine recording
│   │   ├── AudioMeter.swift           # MeterFrame + real-time FFT meter tap
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
│   ├── OverlayPanel.swift             # Dumb fixed-frame overlay host (NSPanel)
│   ├── OverlayPlacement.swift         # Overlay canvas frame math (pure value, unit-tested)
│   ├── PillMetrics.swift              # Pill canvas + per-phase sizes (placement + variants)
│   ├── OverlayScreenLocator.swift     # Screen detection for overlays
│   └── TTSNotchPanelController.swift  # TTS notch overlay (separate; not unified)
│
├── Features/                    # Feature modules
│   ├── Dictation/
│   │   ├── DictationCoordinator.swift # Thin composer over Voice Capture Session
│   │   ├── DictationFeed.swift        # Overlay Feed: phases, beats, meter
│   │   ├── OverlayVariants.swift      # Overlay Variant registry (exploration scaffolding)
│   │   ├── Proofread/                 # Proofread Pass (ADR-0034): policy, verdicts, MLX adapter
│   │   ├── Corrections/               # Correction Pair flywheel (#289): value + bounded store
│   │   └── Views/                     # Recording UI components + overlay variants
│   ├── Speech/                        # engine v2 lives in Vendor/tesseract-speech
│   │   ├── SpeechCoordinator.swift    # @Observable orchestrator; drains engine events
│   │   ├── SpeechEnginePresenter.swift# @Observable residency mirror of the pkg engine
│   │   ├── ArbiterGPULease.swift      # GPULeasing adapter over InferenceArbiter
│   │   ├── AudioPlayback.swift        # @MainActor playback port (seam)
│   │   ├── AudioPlaybackManager.swift # AVFoundation adapter (real pause/resume)
│   │   ├── WordHighlightSurface.swift # Spoken-word highlight port (ADR-0004)
│   │   └── Views/ + NotchOverlay/     # TTS UI; WordTimeline + TTSWordTracker
│   ├── Transcription/
│   │   ├── TranscriptionEngine.swift  # @Observable facade over SpeechRecognizer
│   │   ├── SpeechRecognizer.swift     # Model port (seam) for ASR
│   │   ├── WhisperKitSpeechRecognizer.swift  # CoreML adapter
│   │   ├── TranscriptionHistory.swift # @Observable, JSON persistence
│   │   └── TranscriptionPostProcessor.swift
│   ├── Agent/
│   │   ├── ChatSession.swift          # @Observable spine; folds agent events into ChatItems (ADR-0024)
│   │   ├── AgentRunController.swift   # Foreground run: lease + isGenerating + cancel
│   │   ├── LivePart.swift             # Throttled observable box for the one streaming part
│   │   ├── AgentVoiceInputController.swift  # Composer push-to-talk (leaf)
│   │   ├── ComposerDraftController.swift # Composer draft: text + image queue/drop/Quick Look (leaf)
│   │   ├── AgentEngine.swift          # @Observable, wraps LLMActor (chat path)
│   │   ├── AgentFactory.swift         # Bootstrap: packages, tools, prompt
│   │   ├── LLMActor.swift             # MLX LLM inference actor
│   │   ├── GPULeaseQueue.swift        # FIFO GPU mutual-exclusion lease
│   │   ├── InferenceArbiter.swift     # Lease + model ownership facade
│   │   ├── Core/                      # Agent loop, state reducer, accumulator
│   │   ├── Tools/                     # Built-in + extension tools
│   │   ├── Commands/                  # Slash command registry + parser
│   │   ├── Context/                   # System prompt, skills, compaction
│   │   ├── ParoQuant/                 # PARO-quantized weight loading
│   │   └── Views/                     # Chat UI
│   ├── Server/                        # Local OpenAI-compatible HTTP server
│   │   ├── HTTPServer.swift           # HTTP/1.1 server
│   │   ├── CompletionHandler.swift    # Streaming + non-streaming completions
│   │   ├── ServerInferenceService.swift   # Dispatcher: Completion Route → two arms
│   │   ├── CompletionRoute.swift      # Pure cache-aware vs standard decision
│   │   ├── ServerCompletion.swift     # Actor-confined cache-aware execution module
│   │   ├── PrefixCacheManager.swift   # Radix-tree KV snapshot cache (RAM tier)
│   │   ├── SSDSnapshotStore.swift     # SSD tier: writer queue + body I/O
│   │   ├── SnapshotLedger.swift       # SSD tier: manifest/budget/LRU authority
│   │   ├── PrefillPlanner.swift       # Tokenizer-affine pre-prefill decisions
│   │   ├── LeafAdmissionBuilder.swift # GPU-free leaf-snapshot routing
│   │   ├── EvictionPolicy.swift       # Pure eviction scoring + AlphaTuner
│   │   └── Telemetry/                 # Prompt-cache telemetry store
│   ├── Settings/
│   │   ├── SettingsManager.swift      # @Observable Settings Facade
│   │   ├── SettingsStore.swift        # Persistence seam + Setting declarations
│   │   ├── SettingsCatalogue.swift    # Single home for every default
│   │   ├── SettingsWindowView.swift   # Native Settings scene: TabView of panes
│   │   └── Panes/                     # One file per Settings pane
│   └── Models/                        # Model download management
│
└── Models/                      # Shared data types
    ├── DictationError.swift
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

**Non-view code** (App Bindings, MenuBarManager) observes `@Observable` state using Swift 6.2's `Observations` async sequence:

```swift
Task { [weak self] in
    guard let self else { return }
    for await state in Observations { self.inputs.dictationState() } {
        self.effects.pushDictationStateToMenuBar(state)
    }
}
```

The app's long-lived runtime subscriptions *with a rule* — selected
speech-to-text model auto-load and hot-swap, the lazy LLM reload guard, the
server enable/port reactions, the Overlay Variant switch, hotkey re-binding,
the dictation-phase rule (menu bar mirror + overlay z-order re-assert) — live
in **App Bindings** (`App/AppBindings.swift`), which also owns the launch
ordering: set up the panel, install every subscription, then run the initial
dictation-model load as an owned child task so the HTTP server never waits on
a model load. Effects leave
through a closure-struct the composition root wires —
the launch mirror of `AppTerminationCoordinator`'s teardown steps — which makes
every rule hermetically testable (`AppBindingsTests`). See `CONTEXT.md` → App
composition.

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

### Speech Seams (model ports + playback)

The speech features use the same **facade-above / port-below** shape as the Settings
Store, so the engines' and coordinator's orchestration is testable without models, a
microphone, or `AVAudioEngine`. Three seams sit *below* the `@Observable @MainActor`
facades (ADR-0003; vocabulary in `CONTEXT.md` → **Language → Speech model ports and
playback**):

- **`SpeechRecognizer`** — the ASR model port below `TranscriptionEngine`. The engine
  keeps the timeout race, lazy load, `.mlmodelc` verification, lifecycle state, and
  `DictationError` mapping *above* the port; the port is model-only.
- **TTS (engine v2)** — the TTS engine is the `SpeechEngine` **actor** in the
  `Vendor/tesseract-speech` package (ADR-0038/0039), consumed through its
  session/utterance API. Its own ports live in the package: `SpeechSynthesizing`
  (model port; production adapter `Qwen3Synthesizer` → re-vendored MLXAudioTTS)
  and `GPULeasing` (app adapter `ArbiterGPULease` over `InferenceArbiter`). The
  app-side `SpeechEnginePresenter` is the `@Observable @MainActor` residency
  mirror for views and the arbiter — a presenter, not a facade: orchestration
  lives in the package engine.
- **`AudioPlayback`** — a `@MainActor` *sibling* seam (not a model port) below
  `SpeechCoordinator`, turning generated samples into sound. It is
  `@MainActor protocol AudioPlayback: AnyObject` (the coordinator calls it
  *synchronously* while draining engine events), unlike the model ports which are
  `Sendable nonisolated protocol` actor-backed ports `await`-ed off-main.

```
DictationCoordinator ─(Transcribing)→ TranscriptionEngine ─(SpeechRecognizer)→ adapter
SpeechCoordinator   ───────────────→  SpeechEngine (pkg)  ─(SpeechSynthesizing)→ adapter
SpeechCoordinator   ──(AudioPlayback)────────────────────────────────────────→ adapter
                       engine/coordinator-facing        actor engine          model-facing port
```

Each seam has two adapters — a framework-backed one
(`WhisperKitSpeechRecognizer` in the app; `Qwen3Synthesizer` in the package;
`AudioPlaybackManager` in the app — the only production code touching
WhisperKit / MLX / AVFoundation for these features) and a scripted/in-memory
peer in the tests (`InMemorySpeechRecognizer`, `ScriptedSpeechSynthesizer`,
`InMemoryAudioPlayback`; the package's contract tests use their own
`ScriptedSynthesizer`). Coordinator tests run the **real package engine** over
the scripted synthesizer — replace-don't-layer. The model ports are **actors**
(so `Sendable` is free); the playback adapters are `@MainActor final class`es.
The in-memory playback adapter exposes a **non-wall-clock virtual clock**
(`advance(by:)`) so pacing waits are deterministic. One execution-convention
trap rides these seams: the app target builds with
`NonisolatedNonsendingByDefault`, the package does not, so app-side witnesses
of package protocols spell `@concurrent` explicitly on `async` closure
parameters (`ArbiterGPULease`, test leases).

### Dependency Injection

`DependencyContainer` creates all services lazily and injects them into the SwiftUI hierarchy via scoped modifiers:

```swift
.injectDependencies(from: container)
// Expands to:
//   .injectCoreDependencies(...)       — settings, permissions, container
//   .injectDictationDependencies(...)  — coordinator, engine, history, audio
//   .injectSpeechDependencies(...)     — coordinator, engine presenter
//   .injectAgentDependencies(...)      — coordinator, engine, conversation store
//   .injectModelDependencies(...)      — download manager, inference arbiter
//   .injectServerDependencies(...)     — HTTP server, generation log, cache telemetry
```

AppKit consumers (MenuBarManager, panel controllers) receive dependencies via constructor injection — they cannot use `@Environment`.

---

## Core Concepts

### 1. Window Scene

The app uses `Window("Tesseract", id: "main")` — a single-instance window. This avoids the multi-window workarounds needed with `WindowGroup`. Settings live in the sidebar (`NavigationSplitView`), not a separate `Settings` scene.

### 2. State Machines

Coordinators manage user-facing flows as state machines:

- **DictationCoordinator**: idle → recording → processing → idle (text injection happens during processing)
- **SpeechCoordinator**: idle → capturingText → generating → streaming/playing (⇄ paused) → idle
- **ChatSession**: folds the Agent double-loop's events into committed `ChatItem` values plus one streaming `LivePart` (ADR-0024)

### 3. Actor Isolation

Thread safety uses Swift concurrency. The app target builds with
`SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so every type is implicitly
`@MainActor` unless it opts out (`actor`, `nonisolated`).

- **@MainActor** (the implicit default): all coordinators, engines, managers, views
- **Actors**: `WhisperKitSpeechRecognizer` (CoreML ASR adapter), `SpeechEngine` + `Qwen3Synthesizer` (TTS engine + MLX adapter, TesseractSpeech package), `LLMActor` (MLX LLM), `ContextManager` (compaction)
- **@unchecked Sendable**: `SampleBuffer`, `AudioLevelRelay` (manual NSLock for real-time audio thread)

Trap: a protocol that an actor adapter satisfies must be declared
`nonisolated protocol` — otherwise the protocol inherits the MainActor default
and drags the actor's conformance (including its `init`) onto the main actor.
The speech model ports (ADR-0003) are the worked example.

### 4. Agent Architecture

**Inference stack**: `LLMActor` → `AgentEngine` → `Agent` (double-loop orchestrator).

**Agent bootstrap** (`AgentFactory.makeAgent()`): Discovers packages → registers extensions → discovers skills → loads context files → assembles system prompt → wires compaction → creates Agent instance.

**Double-loop** (`Features/Agent/Core/AgentLoop.swift`): Outer loop handles follow-ups, inner loop handles tool calls + steering. No fixed round limit.

**4 built-in tools**: `read`, `write`, `edit`, `ls` — all sandboxed via `PathSandbox`.

**Extensibility**: Packages, Extensions (tool plugins), Skills (markdown with YAML frontmatter), slash commands (built-in + skills + extensions).

**Image input**: The chat composer shows image affordances only when the selected
agent model is vision-capable and the global "Use vision models when available"
setting is on. File picker, paste, and window-level drag/drop all flow through
`ImageIngest`: supported raster types only, 10 MB per image, typed rejections,
and an eight-image pending queue. Committed and pending images materialize into a
conversation-wide Quick Look preview set, while the server-side cache keys images
by **Image Digest** rather than UI attachment identity. Vocabulary: `CONTEXT.md`
→ Vision capability and mode, Image-aware prefix caching.

### 5. GPU Lease Arbitration

GPU inference is serialized behind a lease. `GPULeaseQueue` is the pure FIFO
mutual-exclusion mechanism (atomic handoff, cancellation-safe); `InferenceArbiter`
composes it with model ownership (`.llm`/`.tts` slots, load/unload,
reload-on-mismatch), so model identity cannot change under a running consumer.
Lease-acquiring consumers depend on the single-member `InferenceArbitrating` seam;
tests inject `InMemoryInferenceArbiter`. Vocabulary: CONTEXT.md → GPU lease
arbitration.

### 6. HTTP Server and Prefix Cache

`Features/Server/` hosts a local OpenAI-compatible HTTP server (`HTTPServer`,
`CompletionHandler`) that drives the same `LLMActor` through the GPU lease. The
public surface is `/health`, `/v1/models`, `/v1/chat/completions`, plus
integration endpoints under `/integrations/opencode/`. `/v1/models` lists
downloaded agent models only; `/v1/chat/completions` honors `request.model` for
downloaded in-catalog models, falls back to the selected agent model when the
field is omitted, and returns OpenAI-shaped `model_not_found` for unknown or
undownloaded IDs.
`ServerInferenceService` is the dispatcher: it owns the **Completion Route**
(`CompletionRoute`, the pure request-shape decision) and composes two arms —
the cache-aware **Server Completion** module (`ServerCompletion`, an
actor-confined module stored in `LLMActor`; ADR-0015) and the agent engine's
managed fallback.
Repeated prompts are accelerated by a tiered KV prefix cache
(`PrefixCacheManager`): a radix tree of KV-cache snapshots in RAM, spilled to SSD
(`SSDSnapshotStore` + `SnapshotLedger`), with flop-aware LRU eviction
(`EvictionPolicy`, `AlphaTuner`). Vocabulary: CONTEXT.md → Prefix cache snapshot
lifecycle, SSD snapshot ledger, Prefill orchestration, Eviction tuning.
Verification gates: docs/testing.md → Loaded-model verification.
`Features/Server/Integrations/` configures external clients against the live
server: the server itself serves a setup script whose one-liner runs the
**Config Merge** (`OpenCodeConfigMerge`, a pure function over an
`IntegrationSnapshot` of port + downloaded models + capabilities) — OpenCode is
the first adapter. HTTP requests load the vision variant for vision-capable
models unconditionally (ADR-0008), so a generated config never advertises what
the server won't serve. Vocabulary: CONTEXT.md → Client integrations.

### 7. Platform Adapters

All AppKit bridging lives in `Platform/`. These are the features that SwiftUI cannot cover:

- Global hotkeys (CGEventTap)
- Clipboard text injection (CGEvent Cmd+V simulation)
- Always-on-top overlay panels (NSPanel)
- Menu bar status item (NSStatusItem)
- Notch overlay for TTS

The Overlay Panel is a dumb, fixed-frame host: created once at launch, permanently ordered front, never resizing or fading. The hosted Overlay Variant view observes the Overlay Feed directly and owns all visibility and motion in SwiftUI; the panel's only runtime inputs are `setContent` (variant switch), `setPlacement`, and `reassertFront` (z-order hygiene, driven by an App Bindings rule on non-idle phases).

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
           └─► SpeechRecognizer port → WhisperKit inference → TranscriptionResult

3. Post-processing
   └─► TranscriptionPostProcessor → TextInjector.inject()
       ├─► Copy to clipboard
       └─► Simulate Cmd+V
```

### Audio Format Pipeline

```
Microphone (48kHz stereo) → [Voice Processing: AEC+AGC+NS, optional toggle]
  → AVAudioEngine tap (device rate, mono float32) → SampleBuffer (thread-safe)
  → Resample to 16kHz (anti-aliased, AudioConverter) → WhisperKit
  └─► RawCapture (native rate, pre-resample) → Capture Dump (bounded WAV ring)
```

---

## Decisions and Rationale

Key architectural decisions (durable records live in `docs/adr/`):

- **`Window` not `WindowGroup`**: Product intent is a single main window. `Window` eliminates 5 workarounds for multi-window suppression.
- **`@Observable` not `ObservableObject`**: Observation framework tracks property access precisely (no coarse object-wide invalidation). Better SwiftUI performance.
- **No `@AppStorage` in `@Observable`**: Compiler incompatibility. All settings use manual `UserDefaults` with `didSet`.
- **No `SettingsManager.shared` singleton**: Injected via `DependencyContainer`. AppKit consumers get it via constructor injection.
- **Speech model ports below the engines/coordinator**: `SpeechRecognizer`, the TesseractSpeech package's `SpeechSynthesizing`/`GPULeasing`, and the `@MainActor` `AudioPlayback` sibling seam make the speech engines' and coordinator's orchestration testable without models, a mic, or `AVAudioEngine` — same facade-above / port-below shape as the Settings Store. See ADR-0003/0038 and `CONTEXT.md` → Speech model ports and playback.
- **`Observations` async sequence for non-view code**: Replaces Combine `$property.sink` for observing `@Observable` types outside SwiftUI views.
- **`AgentFactory` separate from container**: Container wires dependencies; factory orchestrates multi-step bootstrap.
- **Overlay Panel is a dumb host; the Overlay Feed is the one signal surface**: The panel never animates its own frame or visibility — SwiftUI owns all motion, which removes the two-animation-system jank (map #283). Overlay Variants render from the shared `DictationFeed` (typed phases/errors, outcome beats, level + spectrum); the dictation pipeline never learns which variant is live.
- **App Bindings owns the launch sequence and subscription rules**: Carved out of the composition root behind a closure-struct interface — the launch mirror of `AppTerminationCoordinator`. One dictation-state subscription feeds the overlays and the menu bar (no second path, no race), and the initial selected speech-to-text model load runs as an owned child task so the HTTP server is reachable immediately at launch. It also heals a missing dictation-model selection onto a downloaded variant and hot-swaps when the user changes the selection. The container stays pure wiring and passes the deletion test. See `CONTEXT.md` → App composition.
- **Defer Agent package extraction**: Don't extract `Features/Agent` into a separate Swift package until dependency boundaries are clearer.
- **Defer separate Settings scene**: Keep settings in the main window sidebar.
- **Defer UI automation**: Invest in coordinator unit tests first.
