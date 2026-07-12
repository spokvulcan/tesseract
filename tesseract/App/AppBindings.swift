//
//  AppBindings.swift
//  tesseract
//

import Combine
import Foundation
import Observation

/// App Bindings: the deep module owning the app's launch sequence and every
/// long-lived runtime subscription *with a rule* — carved out of the composition
/// root, which stays pure wiring. The launch mirror of
/// `AppTerminationCoordinator`: that module owns teardown ordering; this one
/// owns launch ordering. The Settings Facade comes in concrete; effects leave
/// through a flat closure-struct the container wires.
@MainActor
final class AppBindings {

    struct Inputs {
        /// Tracked read of the dictation feed's phase. Read inside an
        /// `Observations` closure, so the rule re-fires on every phase change.
        let dictationState: @MainActor () -> DictationFeed.Phase
        /// Tracked read of the feed's terminal beat — drives the overlay
        /// affordance window (the panel is clickable only while a beat's
        /// affordances linger, ticket #289).
        let dictationBeat: @MainActor () -> DictationFeed.Beat?
        /// Tracked read of the speech coordinator's state — feeds the menu
        /// bar's status glyph (the speaking animation).
        let speechState: @MainActor () -> SpeechState
        /// Gate read of the hotkey manager's currently bound dictation combo —
        /// read in the rule body (not tracked), to skip no-op re-binds.
        let currentDictationHotkey: @MainActor () -> KeyCombo
        /// Gate read of the inference arbiter's LLM slot — read in the rule
        /// body (not tracked), so the initial model-id emission never forces a
        /// model load and lazy loading is preserved.
        let isLLMSlotLoaded: @MainActor () -> Bool
        /// Gate read of the selected speech-to-text model's on-disk location;
        /// nil while it is not downloaded.
        let whisperModelPath: @MainActor () -> URL?
        /// Gate read of the transcription engine's load state, so a download
        /// completion never reloads an engine that is already serving.
        let isTranscriptionModelLoaded: @MainActor () -> Bool
        /// The model download manager's status stream, watched for the Whisper
        /// model's download completing.
        let modelDownloadStatuses: AnyPublisher<[String: ModelStatus], Never>

        init(
            dictationState: @escaping @MainActor () -> DictationFeed.Phase,
            dictationBeat: @escaping @MainActor () -> DictationFeed.Beat? = { nil },
            speechState: @escaping @MainActor () -> SpeechState,
            currentDictationHotkey: @escaping @MainActor () -> KeyCombo,
            isLLMSlotLoaded: @escaping @MainActor () -> Bool,
            whisperModelPath: @escaping @MainActor () -> URL?,
            isTranscriptionModelLoaded: @escaping @MainActor () -> Bool,
            modelDownloadStatuses: AnyPublisher<[String: ModelStatus], Never>
        ) {
            self.dictationState = dictationState
            self.dictationBeat = dictationBeat
            self.speechState = speechState
            self.currentDictationHotkey = currentDictationHotkey
            self.isLLMSlotLoaded = isLLMSlotLoaded
            self.whisperModelPath = whisperModelPath
            self.isTranscriptionModelLoaded = isTranscriptionModelLoaded
            self.modelDownloadStatuses = modelDownloadStatuses
        }
    }

    struct Effects {
        /// Creates the overlay panel (contentless; the variant rule's initial
        /// emission installs the hosted view).
        let setUpOverlayPanel: @MainActor () -> Void
        /// Installs the Overlay Variant selected by the setting (view +
        /// placement) into the panel.
        let setOverlayVariant: @MainActor (String) -> Void
        /// Z-order hygiene: re-asserts the always-front panel when dictation
        /// becomes active (something may have ordered above it since launch).
        let reassertOverlayFront: @MainActor () -> Void
        /// Flips the overlay panel between click-through (resting) and
        /// interactive (while a beat's affordances linger, ticket #289).
        let setOverlayInteractive: @MainActor (Bool) -> Void
        let pushDictationStateToMenuBar: @MainActor (DictationFeed.Phase) -> Void
        let pushSpeechStateToMenuBar: @MainActor (SpeechState) -> Void
        /// Builds the capture engine (incl. its Voice Processing configuration)
        /// ahead of the first press, so no capture pays the VPIO setup cost
        /// interactively.
        let prewarmAudioCapture: @MainActor () -> Void
        /// Loads the Proofread Pass's model ahead of the first dictation (a
        /// no-op while the pass is disabled or its model isn't downloaded),
        /// so no pass pays the model load interactively.
        let prewarmProofreader: @MainActor () async -> Void
        /// Brings the living memory up (ADR-0035): loads the embedder, then
        /// seeds the store from the owner's existing corpus on first run. A
        /// no-op once seeded, and while memory is off.
        let startMemory: @MainActor () async -> Void
        let updateDictationHotkey: @MainActor (KeyCombo) -> Void
        let updateTTSHotkey: @MainActor (KeyCombo) -> Void
        let updateAgentHotkey: @MainActor (KeyCombo) -> Void
        let updateAppshotHotkey: @MainActor (KeyCombo) -> Void
        let startHTTPServer: @MainActor () async -> Void
        let stopHTTPServer: @MainActor () -> Void
        let updateHTTPServerPort: @MainActor (UInt16) async -> Void
        let reloadLLMIfNeeded: @MainActor () async throws -> Void
        /// Loads the Whisper model from its on-disk path into the
        /// transcription engine.
        let loadWhisperModel: @MainActor (URL) async -> Void

        init(
            setUpOverlayPanel: @escaping @MainActor () -> Void,
            setOverlayVariant: @escaping @MainActor (String) -> Void,
            reassertOverlayFront: @escaping @MainActor () -> Void,
            setOverlayInteractive: @escaping @MainActor (Bool) -> Void = { _ in },
            pushDictationStateToMenuBar: @escaping @MainActor (DictationFeed.Phase) -> Void,
            pushSpeechStateToMenuBar: @escaping @MainActor (SpeechState) -> Void,
            prewarmAudioCapture: @escaping @MainActor () -> Void = {},
            prewarmProofreader: @escaping @MainActor () async -> Void = {},
            startMemory: @escaping @MainActor () async -> Void = {},
            updateDictationHotkey: @escaping @MainActor (KeyCombo) -> Void,
            updateTTSHotkey: @escaping @MainActor (KeyCombo) -> Void,
            updateAgentHotkey: @escaping @MainActor (KeyCombo) -> Void,
            updateAppshotHotkey: @escaping @MainActor (KeyCombo) -> Void,
            startHTTPServer: @escaping @MainActor () async -> Void,
            stopHTTPServer: @escaping @MainActor () -> Void,
            updateHTTPServerPort: @escaping @MainActor (UInt16) async -> Void,
            reloadLLMIfNeeded: @escaping @MainActor () async throws -> Void,
            loadWhisperModel: @escaping @MainActor (URL) async -> Void
        ) {
            self.setUpOverlayPanel = setUpOverlayPanel
            self.setOverlayVariant = setOverlayVariant
            self.reassertOverlayFront = reassertOverlayFront
            self.setOverlayInteractive = setOverlayInteractive
            self.pushDictationStateToMenuBar = pushDictationStateToMenuBar
            self.pushSpeechStateToMenuBar = pushSpeechStateToMenuBar
            self.prewarmAudioCapture = prewarmAudioCapture
            self.prewarmProofreader = prewarmProofreader
            self.startMemory = startMemory
            self.updateDictationHotkey = updateDictationHotkey
            self.updateTTSHotkey = updateTTSHotkey
            self.updateAgentHotkey = updateAgentHotkey
            self.updateAppshotHotkey = updateAppshotHotkey
            self.startHTTPServer = startHTTPServer
            self.stopHTTPServer = stopHTTPServer
            self.updateHTTPServerPort = updateHTTPServerPort
            self.reloadLLMIfNeeded = reloadLLMIfNeeded
            self.loadWhisperModel = loadWhisperModel
        }
    }

    private let settings: SettingsManager
    private let inputs: Inputs
    private let effects: Effects
    private var observationTasks: [Task<Void, Never>] = []
    private var cancellables = Set<AnyCancellable>()
    /// Reverts the overlay to click-through after the affordance window.
    private var overlayInteractivityRevert: Task<Void, Never>?
    private var whisperLoadTask: Task<Void, Never>?
    private var whisperLoadInFlightPath: URL?
    private var lastSelectedSpeechModelStatus: ModelStatus?
    private var hasStarted = false

    init(settings: SettingsManager, inputs: Inputs, effects: Effects) {
        self.settings = settings
        self.inputs = inputs
        self.effects = effects
    }

    func start() {
        guard !hasStarted else { return }
        hasStarted = true

        effects.setUpOverlayPanel()

        installSubscriptions()

        // Build the capture engine (and arm Voice Processing) once at launch —
        // the arm is the expensive step (PRD #188) and paying it here keeps
        // every press at engine-start cost. As a task rather than inline: the
        // arm takes hundreds of ms and must not block the launch turn. Launch
        // is also when no other in-process audio can be running yet, which is
        // exactly when arming is safe (arming starves concurrent in-process
        // capture — measured, ADR-0025). No-ops without microphone permission;
        // the first press arms instead in that case.
        Task { [weak self] in
            self?.effects.prewarmAudioCapture()
            // After the capture arm (same background task, launch-only): the
            // proofread model load is MLX weight I/O, harmless to sequence.
            await self?.effects.prewarmProofreader()
            // And last, behind both: the memory embedder, and — on a machine
            // that has never run this before — the backfill of the owner's
            // existing conversations. Last because it is the one launch task
            // nothing else waits on, and on first run it is the long one.
            await self?.effects.startMemory()
        }

        // Load an already-downloaded Whisper model as an owned child task,
        // *after* every subscription is installed — the HTTP server (and every
        // other rule) must never wait on a speech-model load.
        whisperLoadTask = Task { [weak self] in
            guard let self else { return }
            await self.loadWhisperModelIfAvailable()
        }
    }

    func stop() {
        for task in observationTasks {
            task.cancel()
        }
        observationTasks = []
        cancellables = []
        overlayInteractivityRevert?.cancel()
        overlayInteractivityRevert = nil
        whisperLoadTask?.cancel()
        whisperLoadTask = nil
    }

    private func loadWhisperModelIfAvailable() async {
        guard let path = inputs.whisperModelPath() else {
            Log.general.warning("Whisper model not downloaded — download it from the Models page")
            return
        }
        // Collapse concurrent triggers of the same load (the launch task, a
        // download completion, and a selection change healed onto the same
        // model can all fire together) into one engine load per path.
        guard whisperLoadInFlightPath != path else { return }
        whisperLoadInFlightPath = path
        defer { whisperLoadInFlightPath = nil }
        await effects.loadWhisperModel(path)
    }

    /// Selection follows availability only when the selected model is missing:
    /// if the selected speech-to-text model is not on disk but another variant
    /// is, flip the selection to the downloaded one and return true. Covers a
    /// fresh install that downloads only the compact variant and deletion of
    /// the selected variant — dictation is never silently dead while a speech
    /// model exists on disk. An available selection is never overridden.
    private func healSpeechToTextSelectionIfNeeded(statuses: [String: ModelStatus]) -> Bool {
        let selectedID = settings.selectedSpeechToTextModelID
        if ModelCatalog.isDownloaded(selectedID, statuses: statuses) { return false }
        let downloadedVariant = ModelCatalog.downloaded(
            in: .speechToText, definitions: ModelDefinition.all, statuses: statuses
        ).first
        guard let downloadedVariant else { return false }
        Log.transcription.info(
            "Selected dictation model \(selectedID) is not on disk — switching to \(downloadedVariant.id)"
        )
        settings.selectedSpeechToTextModelID = downloadedVariant.id
        return true
    }

    // MARK: - Subscriptions

    // Every subscription relies on `Observations` emitting the current value at
    // subscription time to apply each rule's initial seed.
    private func installSubscriptions() {
        // One rule per status emission, two duties in order: heal the
        // dictation model selection onto a model that exists on disk, then
        // auto-load the selected model the moment its own download completes
        // (so dictation works without restarting the app) — unless the engine
        // is already serving a model. A heal hands the load to the
        // selection-change rule below — reacting to the status too would load
        // the same model twice. The manual last-status dedupe replaces
        // `removeDuplicates()` because the key (the selected id) is itself
        // mutable: unrelated download progress must never thrash the engine.
        inputs.modelDownloadStatuses
            .sink { [weak self] statuses in
                guard let self else { return }
                let healed = self.healSpeechToTextSelectionIfNeeded(statuses: statuses)
                let status = statuses[self.settings.selectedSpeechToTextModelID]
                let statusChanged = status != self.lastSelectedSpeechModelStatus
                self.lastSelectedSpeechModelStatus = status
                guard !healed, statusChanged, case .downloaded = status else { return }
                guard !self.inputs.isTranscriptionModelLoaded() else { return }
                Task {
                    await self.loadWhisperModelIfAvailable()
                }
            }
            .store(in: &cancellables)

        // Hot-swap the dictation model when the selection changes — loading
        // the newly selected model replaces the previous one, freeing its
        // memory. The initial emission is dropped: the launch load is
        // `whisperLoadTask`'s job, and it must not be raced by a duplicate.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await _ in Observations({ self.settings.selectedSpeechToTextModelID })
                    .dropFirst()
                {
                    await self.loadWhisperModelIfAvailable()
                }
            })

        // Reload the agent model when the selection changes — but only while an
        // LLM slot is already loaded. The gate drops the initial emission, so a
        // selected-but-unused model never forces a multi-gigabyte load at launch.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await _ in Observations({ self.settings.selectedAgentModelID }) {
                    guard self.inputs.isLLMSlotLoaded() else { continue }
                    do {
                        try await self.effects.reloadLLMIfNeeded()
                    } catch {
                        Log.agent.error("Agent model reload failed: \(error.localizedDescription)")
                    }
                }
            })

        // Start/stop the HTTP server with the setting. The initial emission is
        // load-bearing: enabled-at-launch produces the launch start.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await enabled in Observations({ self.settings.isServerEnabled }) {
                    if enabled {
                        await self.effects.startHTTPServer()
                    } else {
                        self.effects.stopHTTPServer()
                    }
                }
            })

        // Apply port changes live, clamped by the one shared rule the server
        // was also constructed with.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await port in Observations({ self.settings.serverPort }) {
                    await self.effects.updateHTTPServerPort(HTTPServer.clampedPort(port))
                }
            })

        // The single dictation-phase subscription: the menu bar mirrors every
        // emission, and any non-idle phase re-asserts the overlay panel's
        // z-order (the variant view renders the phase by observing the feed
        // directly — no push). A live phase also ends any affordance window:
        // an active pill must never intercept clicks.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await state in Observations({ self.inputs.dictationState() }) {
                    self.effects.pushDictationStateToMenuBar(state)
                    if state != .idle {
                        self.effects.reassertOverlayFront()
                        self.endOverlayAffordanceWindow()
                    }
                }
            })

        // The overlay affordance window (ticket #289): a committed/rejected
        // beat makes the panel clickable while the variant's affordances
        // linger, then it reverts to click-through — the resting state of an
        // invisible always-front panel. The grace past the shared linger
        // covers the pill's own fade-out.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await beat in Observations({ self.inputs.dictationBeat() }) {
                    guard let beat else { continue }
                    switch beat.outcome {
                    case .committed, .rejected:
                        self.effects.setOverlayInteractive(true)
                        self.overlayInteractivityRevert?.cancel()
                        self.overlayInteractivityRevert = Task { [weak self] in
                            try? await Task.sleep(
                                for: DictationFeed.affordanceLinger + .milliseconds(300))
                            guard !Task.isCancelled else { return }
                            self?.effects.setOverlayInteractive(false)
                        }
                    case .empty, .cancelled, .superseded:
                        break
                    }
                }
            })

        // Speech state feeds the menu bar's glyph the same way — one
        // subscription, one surface.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await state in Observations({ self.inputs.speechState() }) {
                    self.effects.pushSpeechStateToMenuBar(state)
                }
            })

        // Keep the live Overlay Variant matched to the setting. The initial
        // emission installs the launch variant's view into the panel.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await variantID in Observations({ self.settings.overlayVariantRaw }) {
                    self.effects.setOverlayVariant(variantID)
                }
            })

        // Re-bind the dictation hotkey on change. The initial emission matches
        // the combo the wiring already bound, so the unchanged-combo guard makes
        // launch a no-op instead of a redundant re-bind.
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await hotkey in Observations({ self.settings.hotkey })
                where hotkey != self.inputs.currentDictationHotkey() {
                    self.effects.updateDictationHotkey(hotkey)
                }
            })

        // Re-bind the TTS and agent hotkeys on change; re-applying the current
        // combo at subscription time is a harmless re-register.
        installAuxiliaryHotkeySubscriptions()
    }

    private func endOverlayAffordanceWindow() {
        overlayInteractivityRevert?.cancel()
        overlayInteractivityRevert = nil
        effects.setOverlayInteractive(false)
    }

    /// The TTS / agent / appshot hotkey re-binds — same shape as the dictation
    /// hotkey subscription, minus its unchanged-combo launch guard.
    private func installAuxiliaryHotkeySubscriptions() {
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await hotkey in Observations({ self.settings.ttsHotkey }) {
                    self.effects.updateTTSHotkey(hotkey)
                }
            })
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await hotkey in Observations({ self.settings.agentHotkey }) {
                    self.effects.updateAgentHotkey(hotkey)
                }
            })
        observationTasks.append(
            Task { [weak self] in
                guard let self else { return }
                for await hotkey in Observations({ self.settings.appshotHotkey }) {
                    self.effects.updateAppshotHotkey(hotkey)
                }
            })
    }

}
