# Tesseract Architecture

This document describes the architecture of Tesseract Agent, a privacy-focused, fully offline AI assistant for macOS.

For development guidelines and build commands, see [CLAUDE.md](./CLAUDE.md).
For domain vocabulary, see [CONTEXT.md](./CONTEXT.md); for decision records, see `docs/adr/`.

---

## Overview

Tesseract Agent runs entirely on-device on Apple Silicon. It provides dictation (speech-to-text), text-to-speech, an LLM-powered agent with tool-calling capabilities, and a local OpenAI-compatible HTTP server accelerated by a tiered KV prefix cache. All inference uses local models: WhisperKit (CoreML) for ASR, MLX for LLM and TTS, with the TTS codec's conv stack on the Neural Engine through Core ML where the chip allows (ADR-0075).

**Key Principles:**
- Privacy-first: No audio or text data leaves the device
- Offline: All models run locally on Apple Silicon
- Non-sandboxed: the agent left the App Sandbox for cross-process Accessibility perception (ADR-0047); Hardened Runtime and Developer-ID notarization unchanged
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
│  │ WhisperKit (ASR)     │ │ Qwen3 TTS (MLX, ANE) │ │ MLX LLM      │  │
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
│   ├── SettledWidth.swift       # Settled Width policy + modifier (pure value, unit-tested)
│   ├── StorageEnvironment.swift # Storage roots: the owner's, or per-process scratch under tests (ADR-0073)
│   ├── ViewModifiers.swift      # Scoped dependency injection
│   └── Logging.swift            # Unified logging (Log enum)
│
├── Platform/                    # AppKit bridge code
│   ├── HotkeyManager.swift           # Global hotkey listener (CGEventTap)
│   ├── ObjCExceptions.swift           # NSException → thrown error seam (wraps AVFAudio graph steps)
│   ├── ObjCExceptionCatcher.h/.m      # The one Objective-C file: the @try behind ObjCExceptions
│   ├── TextInjector.swift             # Clipboard-based paste injection
│   ├── InAppReplacer.swift            # Puts a Lens fix back in the app (ADR-0085)
│   ├── TextExtractor.swift            # Selected text extraction
│   ├── MenuBarManager.swift           # Status bar menu (NSStatusItem)
│   ├── OverlayPlacement.swift         # Overlay canvas frame math (pure value, unit-tested)
│   ├── OverlayScreenLocator.swift     # Screen detection for overlays
│   └── SpeechOverlayPanel.swift       # Speech Overlay: read-along words over every app
│
├── Features/                    # Feature modules
│   ├── Dictation/
│   │   ├── DictationCoordinator.swift # Thin composer over Voice Capture Session
│   │   ├── DictationFeed.swift        # Overlay Feed: phases, beats, Live Preview, hold, meter
│   │   ├── Proofread/                 # Proofread Pass (ADR-0034): policy, verdicts, MLX adapter
│   │   ├── Corrections/               # Correction Pair flywheel (#289): value + bounded store
│   │   ├── LearnedWords/              # Learned Words (ADR-0085): store, matcher, sounds-alike key
│   │   ├── Lens/                      # Lens (ADR-0085/0086): the one dictation overlay, Live Preview, the fix
│   │   ├── CatchRecord/               # Catch Record (PRD #612): the Dictation page's model, chart, tiles, takes
│   │   └── Views/                     # Main window shell, the Dictation page, the history sheet
│   ├── Speech/                        # engine v2 lives in Vendor/tesseract-speech
│   │   ├── SpeechCoordinator.swift    # @Observable orchestrator; drains engine events
│   │   ├── SpeechEnginePresenter.swift# @Observable residency mirror of the pkg engine
│   │   ├── AudioPlayback.swift        # @MainActor playback port (seam)
│   │   ├── AudioPlaybackManager.swift # AVFoundation adapter (real pause/resume)
│   │   ├── WordHighlightSurface.swift # Spoken-word highlight port (ADR-0004)
│   │   ├── ReadAlong/                 # Read-Along: the one heard-word clock (ADR-0076/0077)
│   │   ├── Reader/                    # Reader: TextKit 2 text view, bookmark, document store
│   │   ├── Overlay/                   # Speech Overlay: one scrolling feed, island or captions
│   │   ├── Voices/                    # Voice library and designer (VoiceDesign checkpoints)
│   │   └── Views/                     # Speech page: control bar, display settings
│   ├── Transcription/
│   │   ├── TranscriptionEngine.swift  # @Observable facade over SpeechRecognizer
│   │   ├── SpeechRecognizer.swift     # Model port (seam) for ASR
│   │   ├── WhisperKitSpeechRecognizer.swift  # CoreML adapter
│   │   ├── TranscriptionHistory.swift # @Observable, JSON persistence; entries keep catches and app
│   │   └── TranscriptionPostProcessor.swift
│   ├── Agent/
│   │   ├── ChatSession.swift          # @Observable spine; folds agent events into ChatItems (ADR-0024)
│   │   ├── AgentRunController.swift   # Foreground run: LLM gate + isGenerating + cancel
│   │   ├── LivePart.swift             # Throttled observable box for the one streaming part
│   │   ├── AgentVoiceInputController.swift  # Composer push-to-talk (leaf)
│   │   ├── ComposerDraftController.swift # Composer draft: text + image queue/drop/Quick Look (leaf)
│   │   ├── AgentEngine.swift          # @Observable, wraps LLMActor (chat path)
│   │   ├── AgentFactory.swift         # Bootstrap: packages, tools, prompt
│   │   ├── LLMActor.swift             # MLX LLM inference actor
│   │   ├── Speculation.swift          # Speculation: resident drafters + the per-request Speculation Plan (ADR-0079)
│   │   ├── LLMGate.swift              # One LLM generation at a time, FIFO (ADR-0081)
│   │   ├── InferenceArbiter.swift     # LLM gate + model identity + residency facade
│   │   ├── Core/                      # Agent loop, state reducer, accumulator
│   │   ├── Tools/                     # Built-in + extension tools
│   │   ├── Commands/                  # Slash command registry + parser
│   │   ├── Context/                   # System prompt, skills, compaction
│   │   ├── ParoQuant/                 # PARO-quantized weight loading
│   │   └── Views/                     # Chat UI
│   ├── Companion/                     # Jarvis: the owner's day (ADR-0080)
│   │   ├── Agenda/                    # Reminders + Calendar port (EventKit, in-memory), facade, tools, Areas
│   │   ├── Capture/                   # Capture parser, capture door, hotkey panel
│   │   ├── Engine/                    # Day Engine (pure decider), Delivery Ladder, governor, nudges
│   │   ├── Moments/                   # Moments, cards, prompts, card parsing, fallbacks
│   │   ├── Thread/                    # Day Thread: one conversation per day on its own agent
│   │   ├── Loop/                      # Companion runtime: gather → decide → perform, day state
│   │   ├── Today/                     # The Today page (Timeline layout)
│   │   ├── Delivery/                  # Jarvis panel, banners and nudges, glyph state
│   │   ├── Perception/                # Notification Center watcher, seen ledger, owner rules
│   │   ├── CodingAgents/              # Agent signals, Claude Code hooks (Config Merge)
│   │   ├── Profile/ Recall/           # Profile + proposals, remember/forget/recall, FTS index, embedder
│   │   ├── Presence/                  # Idle monitor, frontmost app, power
│   │   ├── Trace/                     # Companion Trace (closed vocabulary, JSONL per day)
│   │   └── Voice/ VoiceOverlay/       # Voice session (the voice rung)
│   ├── Server/                        # Local OpenAI-compatible HTTP server
│   │   ├── HTTPServer.swift           # HTTP/1.1 server
│   │   ├── CompletionHandler.swift    # HTTP framing edge: LLM gate, validation, start
│   │   ├── CompletionDelivery.swift   # One delivery script; Delivery Sink seam (JSON body, SSE)
│   │   ├── ServerInferenceService.swift   # Dispatcher: Completion Route → two arms
│   │   ├── CompletionRoute.swift      # Pure cache-aware vs standard decision
│   │   ├── ServerCompletion.swift     # Actor-confined cache-aware execution module
│   │   ├── CacheClaim.swift          # Cache Claim: a request's lane, pins and Leaf Lease; check-out, check-in, rewind; one conclusion (ADR-0069)
│   │   ├── LeafLease.swift           # Scalar lease identity, shared tree/writer exclusion, exact recurrent rewind
│   │   ├── PrefixCacheManager.swift   # Radix-tree KV snapshot cache (RAM tier)
│   │   ├── SSDSnapshotStore.swift     # SSD tier: writer queue + body I/O
│   │   ├── SnapshotLedger.swift       # SSD tier: manifest/budget/LRU authority
│   │   ├── SnapshotPayload.swift      # Snapshot Payload: a snapshot's SSD byte form, built by Deferred Payload Extraction
│   │   ├── PrefillPlanner.swift       # Tokenizer-affine pre-prefill decisions
│   │   ├── RequestKeyingPhase.swift   # Request Keying: a request's keys and facts, derived once (Keyed Request, ADR-0070)
│   │   ├── ConversationRender.swift   # Conversation Render module: the one chat-template application, Emitted Path Resolve, Generation Prompt probe
│   │   ├── LeafStorePhase.swift       # Leaf Store phase: fast path vs boundary path (+Report, +Executors)
│   │   ├── LeafAdmission.swift        # Leaf Admission: the one way a leaf enters the prefix cache (ADR-0078)
│   │   ├── LeafStoreCounters.swift    # Boundary-turn tally by reason, logged at unload
│   │   ├── LeafAdmissionBuilder.swift # GPU-free leaf-snapshot routing
│   │   ├── LiveLeafCapture.swift      # Pure fast-path eligibility (live vs boundary)
│   │   ├── EmittedPathIndex.swift     # Emitted Path Index: rendered-bytes hash → fed ids per fingerprint (ADR-0063)
│   │   ├── EmittedPathResolve.swift   # Resolve composition + per-request Emitted Path telemetry
│   │   ├── EmittedPathRegistration.swift # Register a finished turn's path behind the fidelity gate
│   │   ├── EvictionPolicy.swift       # Pure eviction scoring (production alpha = 0)
│   │   ├── AlphaTuner.swift           # Retained replay implementation; disabled in production (#504)
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
    ├── CheckBeforePasting.swift # Which takes wait in the Lens (ADR-0086)
    ├── NavigationItem.swift     # Sidebar routing enum
    ├── KeyCombo.swift
    └── ...

tesseract-ios/                   # The iPhone app's own files (ADR-0066, ADR-0084)
├── TesseractPhoneApp.swift      # Entry: the Library in a NavigationStack
├── PhoneContainer.swift         # Composition root (pure wiring)
├── PhoneReading.swift           # The Library's open texts: a Reader per text
├── PhoneAudioSession.swift      # Spoken-audio session around the Mac's playback adapter
├── PhoneIntake.swift            # Files opened in the app, and the Library Inbox taken in
├── PhonePocket.swift            # Interruptions, routes, heat, lock-screen commands, Now Playing
├── PhoneVoice.swift             # The neural voice: download, Voice Preparation, Speed Check, which voice reads
├── PhoneModelFetching.swift     # Model Fetching over a background URLSession
└── Views/                       # Library, Reader (UITextView on TextKit 2), transport, voices, settings

tesseract-share/                 # "Read in Tesseract": the share extension
├── ShareViewController.swift    # Reads what was shared, drops its text in the Library Inbox
└── ExtensionPreprocessing.js    # Runs in Safari: hands over the page's HTML
```

### The iPhone app

`tesseract-ios` is a second app target in the same project. Release 1 reads
text aloud (ADR-0084), so the target takes only the read-aloud code:

- **Its own files** live in `tesseract-ios/`: views, adapters, the composition
  root, and later its Info.plist and entitlements.
- **Shared files** come from five folders of `tesseract/`, added to the target
  as their own synchronized groups (the project's "Shared with tesseract-ios"
  group): `Core`, `Features/Models`, `Features/Settings`, `Features/Speech` and
  `Models`. A new file in one of them builds in both apps. A Mac-only file
  there is excluded from `tesseract-ios` by name, in that group's exceptions
  (File Inspector → Target Membership). Xcode can't exclude a whole folder, so
  Mac-only code belongs in the Mac-only folders when it can.
- **Everything else** in `tesseract/` (the app shell, `Platform/`, the agent,
  server, companion and dictation) is Mac-only and never reaches the phone.
- **No `#if os`.** Where shared code needs something only the Mac has, the
  dependency moves behind a port the Mac adapts: the speech code reads its
  settings through `SpeechSettings`, which the Mac's `SettingsManager`
  conforms to, and the Settings Catalogue keeps its Mac-only entries in
  `SettingsCatalogue+Mac.swift`.
- **The speech package** (`Vendor/tesseract-speech`) builds for iOS. Its
  second production synthesizer, `SystemVoiceSynthesizer`, puts the
  **System Voice** (`AVSpeechSynthesizer`) behind the same port as the neural
  voice, at its 24 kHz frames, so the coordinator, the Read-Along and the
  Reader can't tell them apart.
- **Shared logic for the phone lives in the shared folders**, so the Mac test
  target covers it: the **Library** (`ReaderLibrary`), the phone's Settings
  Facade (`PhoneSettings`), the **Preset Voices**, a text's language
  (`TTSLanguage.detected`), and getting text in (`Features/Speech/Intake`:
  a page's article through swift-readability, a PDF's text through PDFKit,
  Markdown without its marks, the **Library Inbox**), and reading in the pocket
  (`Features/Speech/Pocket`: `PocketControls`, which turns a call, lost
  headphones, the lock screen's buttons or the app leaving the screen into
  what the Reader does, and the **Thermal Policy**). Whatever stops the audio
  from outside the app stops the reading at the heard sentence, and playing
  again reads from it: a stream held paused in the background could not come
  back once the system stopped its audio engine.
- **The neural voice** (`PhoneVoice`): Qwen3-TTS 0.6B on the Neural Engine
  (ADR-0084, ADR-0085). Its catalog entry (`ModelDefinition.phoneVoice`)
  downloads through the Mac's download manager, over a second **Model
  Fetching** adapter: a background URLSession that goes on with the screen
  locked, resumes mid-file and rejoins a transfer after a relaunch (the app
  delegate hands it the session's events), Wi-Fi only unless the owner
  allows cellular. A trimming adapter in front of it keeps only the codec's
  decoder from the file that also holds its encoder. **Voice Preparation**
  then builds the graphs into Application Support (out of backups), the first
  time in minutes and afterwards in seconds, and the **Speed Check** times a
  render. The engine's synthesizer is a `VoiceHandover`: the neural voice for
  each segment while it is ready, keeps up and the **Thermal Policy** allows
  it, the **System Voice** otherwise. A segment the neural voice fails before
  any of its audio is read by the System Voice, which reads on until the app
  is next in front and the voice prepares again. The speed menu offers the
  rates the Speed Check allows.
- **The share extension** (`tesseract-share`, embedded in the app) builds the
  `Intake` folder and nothing else of the app. In Safari its script hands over
  the page's HTML, so the app never fetches a page. It never runs the voice:
  it drops the text in the app group's **Library Inbox**, and the app takes
  it in when it comes to the front.

CI's `build-ios` job builds the target for any iOS device, unsigned; the
release pipeline waits for it. The shared code's tests run in the Mac test
target.

---

## Observation and Data Flow

### State Model

The app uses Swift's Observation framework (`@Observable`) for all primary state types. This replaced the older `ObservableObject` + `@Published` + Combine model.

**SwiftUI views** consume `@Observable` types via `@Environment(Type.self)`:

```swift
struct GeneralSettingsPane: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        // For bindings, use @Bindable:
        @Bindable var settings = settings
        Toggle("Play Sounds", isOn: $settings.playSounds)
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

The app's long-lived runtime subscriptions *with a rule* (selected
speech-to-text model auto-load and hot-swap, the lazy LLM reload guard, the
server enable/port reactions, hotkey re-binding, the dictation-phase rule that
mirrors the menu bar) live in **App Bindings** (`App/AppBindings.swift`), which
also owns the launch ordering: start the speech surfaces, install every
subscription, then run the initial dictation-model load as an owned child
task so the HTTP server never waits on a model load. Effects leave
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
and `InMemorySettingsStore` (tests — hermetic, parallel-safe; also the test host's
own container, per ADR-0073). The two genuine side
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
  `DictationError` mapping *above* the port; the port is model-only. The engine
  also runs the Live Preview lane, which skips rather than queues behind the final
  pass and is cancelled by it. The WhisperKit adapter runs Whisper's audio encoder
  on the GPU and its text decoder on the Neural Engine (ADR-0086).
- **TTS (engine v2)** — the TTS engine is the `SpeechEngine` **actor** in the
  `Vendor/tesseract-speech` package (ADR-0038/0039), consumed through its
  session/utterance API. Its own ports live in the package: `SpeechSynthesizing`
  (model port; production adapter `Qwen3Synthesizer` over the package's own
  `Qwen3TTS` target, first-party since ADR-0071). The engine takes no turn at
  the LLM gate: speech runs beside the LLM (ADR-0081).
  `Qwen3Synthesizer` loads only the folder the app hands it (the Model Catalog's
  Voice Engine folder) and never downloads; the engine checks that folder before
  any load and throws `modelUnavailable` when it's incomplete.
  After warm-up it moves the codec's conv stack to the Neural Engine: a Core ML
  model the package builds from the checkpoint and keeps in the cache directory
  the app passes (under `StorageEnvironment.caches`). It stays on MLX where the
  Neural Engine can't run it all (ADR-0074/0075). For the iPhone, the model
  can move the talker and the code predictor there too
  (`Qwen3TTSModel.prepareNeuralVoice`, ADR-0084/0085): two more Core ML
  graphs the package writes (the talker one position per call with its KV
  cache in Core ML state, the code predictor a whole frame per call with its
  sampling inside), after which generation runs MLX only on the CPU, for the
  prompt's embeddings and the codec's front end. A **Preset Voice**
  (`Voice.preset`) is a CustomVoice checkpoint's own speaker: the engine
  refuses one the checkpoint doesn't have and never takes a Reference Take
  for it. Each segment's words are
  timed by the talker's own attention (ADR-0077): the model streams its
  Alignment Head's row for every frame, and the adapter's `WordTimer` turns the
  rows and the frames' loudness into word starts, which the stream carries as
  `SpeechEvent.words`. The
  app-side `SpeechEnginePresenter` is the `@Observable @MainActor` residency
  mirror for views and the arbiter — a presenter, not a facade: orchestration
  lives in the package engine. It follows the Readiness the engine publishes,
  so a reload the engine starts by itself, for an utterance on a session that
  outlived Offload Model, shows as loaded too.
- **`AudioPlayback`** — a `@MainActor` *sibling* seam (not a model port) below
  `SpeechCoordinator`, turning generated samples into sound. It is
  `@MainActor protocol AudioPlayback: AnyObject` (the coordinator calls it
  *synchronously* while draining engine events), unlike the model ports which are
  `Sendable nonisolated protocol` actor-backed ports `await`-ed off-main. Its
  `heardPlaybackTime()` (the head less the output's latency) is the Read-Along's
  clock; pacing reads the head itself.

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
parameters.

### Dependency Injection

`DependencyContainer` creates all services lazily and injects them into the SwiftUI hierarchy via scoped modifiers:

```swift
.injectDependencies(from: container)
// Expands to:
//   .injectCoreDependencies(...)       — settings, permissions, container
//   .injectDictationDependencies(...)  — coordinator, engine, history, pairs, Learned Words, audio, fixInLens
//   .injectSpeechDependencies(...)     — coordinator, engine presenter
//   .injectAgentDependencies(...)      — coordinator, engine, conversation store
//   .injectModelDependencies(...)      — download manager, inference arbiter
//   .injectServerDependencies(...)     — HTTP server, generation log, cache telemetry
```

The main window's `ContentView` applies the core scope at its root and each page's scopes at that page, so `MainWindowPageTests` renders every page with the same wiring the app ships. The Settings window applies `injectDependencies` whole.

AppKit consumers (MenuBarManager, panel controllers) receive dependencies via constructor injection — they cannot use `@Environment`.

Under a test runner (the app is the unit-test host) the container keeps everything it stores in a scratch directory per test process, through `StorageEnvironment`, and its settings in memory, and the host opens no windows (ADR-0073). The model folder is the one location left in place.

---

## Core Concepts

### 1. Window Scene

The app uses `Window("Tesseract", id: "main")` — a single-instance window. This avoids the multi-window workarounds needed with `WindowGroup`. Settings live in the sidebar (`NavigationSplitView`), not a separate `Settings` scene.

### 2. State Machines

Coordinators manage user-facing flows as state machines:

- **DictationCoordinator**: idle → recording → processing → idle (text injection happens during processing; a held take skips it and waits in the Lens)
- **LensModel**: hidden → listening → finishing → landed, or fixing (a held take, a take reopened to fix, or a take opened from the Dictation page) → done → hidden
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

**Streaming detokenization**: `AppTokenizerLoader` constructs the tokenizer and
recognizes its effective decoder configuration for both ordinary and PARO loads.
The app-owned `LinearStreamingDetokenizer` is shared by the live token loop and
Emitted Path fidelity replay. Live delivery emits on the naive decoder's token
steps for recognized ByteLevel configurations and uses naive streaming otherwise;
replay retains verified segment delivery and its window/fallback paths (ADR-0065).

**Speculative decoding**: `Speculation` owns the drafters (the MTP head inside a
Qwen3.5-family checkpoint, the DFlash2 draft beside Qwen3.8-27B): `LLMActor`'s
load attaches them, the Model Session carries them, and each request asks for a
**Speculation Plan**. The Server Completion and the Raw Generation Start read the
same plan, and the session builds its iterator from it (ADR-0079).
`DFlash2Support` and `MTPDrafterSupport` keep the per-family detection, pairing
and loading.

**Agent bootstrap** (`AgentFactory.makeAgent()`): Discovers packages → registers extensions → discovers skills → loads context files → assembles system prompt → wires compaction → creates Agent instance.

**Double-loop** (`Features/Agent/Core/AgentLoop.swift`): Outer loop handles follow-ups, inner loop handles tool calls + steering. No fixed round limit.

**Built-in tools**: `read`, `write`, `edit`, `ls` (all sandboxed via `PathSandbox`), and the
Companion's tools, registered in every conversation: the agenda tools (`agenda`,
`add_reminder`, `update_reminder`, `add_event`, `move_event`), `notification_rule`, and
`remember` / `forget` / `recall`.

**Extensibility**: Packages, Extensions (tool plugins), Skills (markdown with YAML frontmatter), slash commands (built-in + skills + extensions).

**Image input**: The chat composer shows image affordances only when the selected
agent model is vision-capable and the global "Use vision models when available"
setting is on. File picker, paste, and window-level drag/drop all flow through
`ImageIngest`: supported raster types only, 10 MB per image, typed rejections,
and an eight-image pending queue. Committed and pending images materialize into a
conversation-wide Quick Look preview set, while the server-side cache keys images
by **Image Digest** rather than UI attachment identity. Vocabulary: `CONTEXT.md`
→ Vision capability and mode, Image-aware prefix caching.

### 5. The LLM Gate

Models run side by side; memory, not the GPU, is what they share (ADR-0081).
Only the language model is serialized: it is one container with one prefix
cache, so chat turns, the Companion's moments, HTTP requests, `/compact`,
reloads and offload take turns at the **LLM Gate**. `LLMGate` is the pure FIFO
mechanism (atomic handoff, cancellation-safe); `InferenceArbiter` composes it with
the LLM's identity (load, reload-on-mismatch) in `withLLM`, so the model cannot
change under a running generation, and mirrors residency (`.llm`/`.tts`) for
Offload Model. Speech, dictation, the proofreader and the embedder never take the
gate: MLX's process-wide evaluation lock keeps concurrent use safe, so a voice
reply interleaves with a generation instead of waiting for it. Whisper's audio
encoder runs on the GPU too, and a generation slows while it decodes (ADR-0086).
The proofreader still *skips* while the LLM generates (ADR-0034), so a dictation
never waits.
LLM consumers depend on the single-member `InferenceArbitrating` seam; tests
inject `InMemoryInferenceArbiter`. The menu bar's Models section shows what is
loaded and what each model is doing, with the app's memory. Vocabulary:
CONTEXT.md → LLM gate.

### 6. HTTP Server and Prefix Cache

`Features/Server/` hosts a local OpenAI-compatible HTTP server (`HTTPServer`,
`CompletionHandler`) that drives the same `LLMActor` through the LLM gate. The
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
Both paths use `ManagedGenerationDriver` and `GenerationStreamLoop` to parse
raw model output, forward reasoning and tool calls, honor cancellation, and
publish terminal metrics. The parser starts where the request's **Generation
Prompt** left the model, inside a think block or outside it: Conversation
Render measures what the template appends to open the assistant turn, and
each request checks that measurement against the tokens it fed (ADR-0070).
On the cache-aware arm, Request Keying derives each request's facts once,
into a Keyed Request (or an unkeyed one) that every later phase reads. Reasoning stays as emitted by the model: native
`reasoning_effort` shapes the prompt, while the ordinary generation-token limit
bounds output. There is no thinking-length or repetition intervention, forced
think closure, or continuation restart (ADR-0060 amendment).
Repeated prompts are accelerated by a tiered KV prefix cache
(`PrefixCacheManager`): a radix tree of KV-cache snapshots in RAM, spilled to SSD
(`SSDSnapshotStore` + `SnapshotLedger`), with eviction scoring
(`EvictionPolicy`) fixed at `alpha = 0` (recency within the eligible set).
`AlphaTuner` remains in source but is not attached to production caches: its
replay allocated multi-gigabyte synthetic arrays and blocked MainActor
([#504](https://github.com/spokvulcan/tesseract/issues/504)). The Budget Floor,
pressure response and SSD demotion remain active. Each keyed request holds one
Cache Claim (`CacheClaim`, ADR-0069): its reserve lane, its Restore Pins and,
after a Leaf Handoff, its Leaf Lease, concluded exactly once inside the
request's LLM gate turn. Vocabulary: CONTEXT.md → Prefix cache snapshot
lifecycle, SSD snapshot ledger, Prefill orchestration, Eviction tuning.
Verification gates: docs/testing.md → Loaded-model verification.
`Features/Server/Integrations/` configures external clients against the live
server: the server itself serves a setup script whose one-liner runs the
**Config Merge** (`OpenCodeConfigMerge`, a pure function over an
`IntegrationSnapshot` of port + downloaded models + capabilities) — OpenCode is
the first adapter. HTTP requests load the vision variant for vision-capable
models unconditionally (ADR-0008), so a generated config never advertises what
the server won't serve. Vocabulary: CONTEXT.md → Client integrations.

### 7. The Companion

Jarvis thinks at moments; code keeps the promises (ADR-0080, `CONTEXT.md` →
Companion: the day). **Today** is the main window's first page. Apple
Reminders and Calendar are the single source of truth, behind one **Agenda**
port (`EventKitAgendaStore`; `InMemoryAgendaStore` for the test host and every
test) whose six tools ride every conversation. The capture hotkey is one key
(Right ⌥ alone: tap to type, hold to speak), detected from modifier flags by
`ModifierKeyDetector` beside the key-combo matcher.

The **Day Engine** is a pure decider: `CompanionRuntime` gathers a
`DaySnapshot` (agenda, presence, the app in front, power), feeds one
`DaySignal` at a time, persists the returned `DayState`, and performs the
`DayEffect`s — scheduling OS nudges, running a moment, putting a card on a
rung, changing Reminders, writing the **Companion Trace**. A **Moment** is one
generation over the day's **Day Thread** (`DayThread`, its own agent with the
same system prompt and tools as every chat, so all share one cached prefix);
its JSON card is validated, retried once, or replaced by a deterministic card.
Cards reach the owner through the **Delivery Ladder**: the glyph, the **Jarvis
Panel** (`Platform/GlassPanel.swift`, a borderless non-activating panel over
`NSGlassEffectView`), a banner, or voice. The system prompt carries no time;
every user message carries a stored **Now Tag**. Other apps' banners are sorted
by code on arrival (`NotificationSources`: a person, an app's news, or noise —
the system's banners and games, recognised by `AppIdentityResolver`), so only
people reach a model; moments take their turn at the LLM Gate like any chat.

Tests never touch EventKit or a model: the engine is covered by decision
tables, the tools run against the in-memory Agenda, and cards parse canned
replies (`docs/testing.md` → Companion).

### 8. Platform Adapters

All AppKit bridging lives in `Platform/`. These are the features that SwiftUI cannot cover:

- Global hotkeys (CGEventTap)
- Clipboard text injection (CGEvent Cmd+V simulation)
- Putting a Lens fix back in the app: the changed text selected back through
  Accessibility and pasted over, or erased with backspaces and the fix pasted,
  never in a password field (`InAppReplacer`)
- Always-on-top overlay panels (NSPanel)
- Menu bar status item (NSStatusItem)
- The Speech Overlay: the words being read, over every app (a separate panel)

The one dictation overlay is the **Lens** (PRD #612, ADR-0085, ADR-0086), in a `GlassPanel` at the bottom center of the screen, built at launch so the first press shows it at once. It follows the Overlay Feed itself (phases, beats, the Live Preview, the take's app, the hold), so no App Bindings rule pushes to it. The panel never becomes key while the owner talks; it takes the keyboard only while a held take waits or a take is being fixed, and gives it back before anything pastes. The Overlay Variant registry, the classic pill and its fixed-frame panel are gone; `OverlayPlacement` and `OverlayScreenLocator` remain for the Companion's voice overlay.

### 9. The Dictation Page (the Catch Record)

The Dictation page is the **Catch Record** (PRD #612): what the Learned Words
caught this week, what they are, and today's takes, each one click from a fix
in the Lens. `DictationContentView` reads the Learned Word, Correction Pair and
transcription history stores from the environment and builds a `CatchRecord`
from their values (`tesseract/Features/Dictation/CatchRecord/`). `CatchRecord`
is a pure value: seven days of catches and fixes, one tile per Learned Word
still known, today's takes, the page's sentence and the line under it, and the
runs that mark a caught or fixed word. The page and `CatchRecordTests` read the
same thing. On it:

- `CatchWeekChart`: the week in Swift Charts, Caught and You fixed stacked per
  day with a legend (design language §5).
- `LearnedWordTile`: one Learned Word (heard forms, the fix that taught it as a
  before-and-after strip, the apps it is left alone in), with Forget in its
  context menu. Forget returns the store's receipt, and the page's Undo line
  hands it back.
- `TodayTakeRow`: one of today's takes, its catches marked by `CaughtText`; a
  click opens it in the Lens.

The toolbar holds the record button, History, and Export Corrections (the
Correction Pairs as JSONL). History presents `TranscriptionHistorySheet`: the
full history by day, its catches marked the same way, where an entry's context
menu opens it in the Lens, copies it or deletes it. The page and the sheet
reach the Lens through `FixInLensAction`, an environment value that
`injectDictationDependencies` wires to `LensController.openFromPage`; its
default does nothing, so a preview or a test needs no Lens. Each
`TranscriptionEntry` keeps its catches and the app the take went to, so a take
opened from the page shows what was caught and a fix can leave a word alone in
that app; entries written before that decode with neither.

Gone with the old page: its custom-glass recording button
(`RecordingButtonView`), the history's side-by-side Correction Pair editor and
its Flag as Wrong button, and the focus request that opened that editor from
the retired pill (`requestFocus(pairID:)` and `focusEntryID` on
`TranscriptionHistory`). Fixing a take happens only in the Lens.

---

## Data Flow

### Recording to Text Injection

```
1. User presses hotkey (Option+Space)
   └─► HotkeyManager.onHotkeyDown()
       └─► DictationCoordinator.startRecording()
           ├─► AudioCaptureEngine.startCapture()
           └─► Live Preview pump (ADR-0086), 300 ms after each decode lands
               ├─► LivePreviewAssembler: the audio since the last confirmed segment
               ├─► TranscriptionEngine.transcribePartial → TranscriptionResult
               ├─► all but the last two segments confirmed; the rest is the tail
               └─► regex cleanup + Learned Words → DictationFeed → Lens (listening)
   Shift tapped while the key is held
   └─► HotkeyManager onShiftTap → DictationCoordinator.shiftTapped(): the take will wait

2. User releases hotkey
   └─► DictationCoordinator.stopRecordingAndProcess()
       ├─► the preview pump and its decode in flight are cancelled
       ├─► AudioCaptureEngine.stopCapture() → AudioData
       └─► TranscriptionEngine.transcribe(audioData): the full pass over the whole take
           └─► SpeechRecognizer port → WhisperKit inference → TranscriptionResult

3. Post-processing
   └─► TranscriptionPostProcessor → Learned Words → Proofread Pass (opt-in)
       ├─► TranscriptionHistory.add: the text, its catches and the app
       ├─► TextInjector.inject() (not for a held take)
       │   ├─► Copy to clipboard
       │   └─► Simulate Cmd+V
       └─► Lens: the take lands and the words the preview had wrong settle
           (LensSettle), or a held take waits: Return pastes it, Esc keeps it

4. Fixing a word after the paste (Control+Option+Space, ADR-0085)
   └─► Lens opens on the last take; the owner types the word meant
       ├─► LensFix picks the words it replaces and decides what the fix teaches
       ├─► LearnedWordStore learns it (or leaves a word alone in that app)
       ├─► CorrectionPairStore marks the take gold with the fix
       ├─► TranscriptionHistory.replaceText: the entry's new text and catches
       └─► InAppReplacer puts the fix in the app, only while the pasted
           text is still the last thing typed there

5. Fixing a word from the Dictation page (the Catch Record)
   └─► a click on one of today's takes (or Fix a Word in the history sheet)
       └─► FixInLensAction (the fixInLens environment value)
           └─► LensController.openFromPage(take): refused while a take is
               recorded or a fix is put back; a take open in the Lens is kept
               (a held one waits for Control+Option+Space, with its fixes)
               ├─► the Lens opens the take as the history keeps it
               ├─► the same fix as step 4, recorded as made from the page
               └─► nothing is pasted back; if it is the take Control+Option+Space
                   reopens, the reopen starts from the fixed text
```

A silent capture stops before step 2's transcription. Learned Words apply inside
the Voice Capture Session, so Voice Input and the Companion's captures gain them
too.

### Audio Format Pipeline

```
Microphone (48kHz stereo) → [Voice Processing: AEC+AGC+NS, armed for the app's lifetime]
  → AVAudioEngine tap, one per take (device rate, mono float32) → SampleBuffer (thread-safe)
  │   └─► heartbeat per buffer → live-input check (every 0.5 s while the take is open)
  → Resample to 16kHz (anti-aliased, AudioConverter) → WhisperKit
  └─► RawCapture (native rate, pre-resample) → Capture Dump (bounded WAV ring)

TTS samples → SpeechCoordinator → AudioPlayback (AudioPlaybackManager, its own AVAudioEngine)
```

Every capture — dictation, Voice Input, a voice-session take — installs its
own tap on the kept engine at the device rate and removes it at the stop. A
dead input is caught while the take is open: the **live-input check**
restarts an input that never delivered a buffer (once per take), and marks
one that went quiet dead for the capture's owner. Speech plays on its own
engine and never overlaps a voice-session capture: the session is
half-duplex, so the mic closes before a reply speaks and opens after it
stops (ADR-0082).

---

## Decisions and Rationale

Key architectural decisions (durable records live in `docs/adr/`):

- **`Window` not `WindowGroup`**: Product intent is a single main window. `Window` eliminates 5 workarounds for multi-window suppression.
- **`@Observable` not `ObservableObject`**: Observation framework tracks property access precisely (no coarse object-wide invalidation). Better SwiftUI performance.
- **No `@AppStorage` in `@Observable`**: Compiler incompatibility. All settings use manual `UserDefaults` with `didSet`.
- **No `SettingsManager.shared` singleton**: Injected via `DependencyContainer`. AppKit consumers get it via constructor injection.
- **Speech model ports below the engines/coordinator**: `SpeechRecognizer`, the TesseractSpeech package's `SpeechSynthesizing`, and the `@MainActor` `AudioPlayback` sibling seam make the speech engines' and coordinator's orchestration testable without models, a mic, or `AVAudioEngine` — same facade-above / port-below shape as the Settings Store. See ADR-0003/0038 and `CONTEXT.md` → Speech model ports and playback.
- **`Observations` async sequence for non-view code**: Replaces Combine `$property.sink` for observing `@Observable` types outside SwiftUI views.
- **`AgentFactory` separate from container**: Container wires dependencies; factory orchestrates multi-step bootstrap.
- **The Overlay Feed is the one signal surface**: The Lens renders from the shared `DictationFeed` (typed phases/errors, outcome beats, the Live Preview, the hold, level + spectrum); the dictation pipeline never learns what draws it. The Lens replaced the Overlay Variant registry and its Setting (PRD #612).
- **The Lens streams, the paste stays a full pass** (ADR-0086): the Live Preview decodes the take with the same Whisper model while it is recorded, only for show; release cancels it, and what pastes is Whisper's full pass over the whole take. Whisper's audio encoder runs on the GPU, which halves the wait after release.
- **App Bindings owns the launch sequence and subscription rules**: Carved out of the composition root behind a closure-struct interface — the launch mirror of `AppTerminationCoordinator`. One dictation-state subscription feeds the menu bar (no second path, no race), and the initial selected speech-to-text model load runs as an owned child task so the HTTP server is reachable immediately at launch. It also heals a missing dictation-model selection onto a downloaded variant and hot-swaps when the user changes the selection. The container stays pure wiring and passes the deletion test. See `CONTEXT.md` → App composition.
- **Defer Agent package extraction**: Don't extract `Features/Agent` into a separate Swift package until dependency boundaries are clearer.
- **Defer separate Settings scene**: Keep settings in the main window sidebar.
- **Defer UI automation**: Invest in coordinator unit tests first.
