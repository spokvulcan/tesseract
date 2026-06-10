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
        /// Tracked read of the dictation coordinator's state. Read inside an
        /// `Observations` closure, so the rule re-fires on every state change.
        let dictationState: @MainActor () -> DictationState
        /// Tracked read of the audio capture engine's level, observed the same way.
        let audioLevel: @MainActor () -> Float
        /// Gate read of the hotkey manager's currently bound dictation combo —
        /// read in the rule body (not tracked), to skip no-op re-binds.
        let currentDictationHotkey: @MainActor () -> KeyCombo
        /// Gate read of the inference arbiter's LLM slot — read in the rule
        /// body (not tracked), so the initial model-id emission never forces a
        /// model load and lazy loading is preserved.
        let isLLMSlotLoaded: @MainActor () -> Bool
        /// Gate read of the Whisper model's on-disk location; nil while it is
        /// not downloaded.
        let whisperModelPath: @MainActor () -> URL?
        /// Gate read of the transcription engine's load state, so a download
        /// completion never reloads an engine that is already serving.
        let isTranscriptionModelLoaded: @MainActor () -> Bool
        /// The model download manager's status stream, watched for the Whisper
        /// model's download completing.
        let modelDownloadStatuses: AnyPublisher<[String: ModelStatus], Never>

        init(
            dictationState: @escaping @MainActor () -> DictationState,
            audioLevel: @escaping @MainActor () -> Float,
            currentDictationHotkey: @escaping @MainActor () -> KeyCombo,
            isLLMSlotLoaded: @escaping @MainActor () -> Bool,
            whisperModelPath: @escaping @MainActor () -> URL?,
            isTranscriptionModelLoaded: @escaping @MainActor () -> Bool,
            modelDownloadStatuses: AnyPublisher<[String: ModelStatus], Never>
        ) {
            self.dictationState = dictationState
            self.audioLevel = audioLevel
            self.currentDictationHotkey = currentDictationHotkey
            self.isLLMSlotLoaded = isLLMSlotLoaded
            self.whisperModelPath = whisperModelPath
            self.isTranscriptionModelLoaded = isTranscriptionModelLoaded
            self.modelDownloadStatuses = modelDownloadStatuses
        }
    }

    struct Effects {
        /// Sets the border overlay's glow theme. Called synchronously *before*
        /// panel setup so a user's non-default theme shows on the very first frame.
        let setBorderGlowTheme: @MainActor (GlowTheme) -> Void
        /// Creates both overlay panels' views.
        let setUpOverlayPanels: @MainActor () -> Void
        let setPillOverlayEnabled: @MainActor (Bool) -> Void
        let setBorderOverlayEnabled: @MainActor (Bool) -> Void
        let pushDictationStateToPill: @MainActor (DictationState) -> Void
        let pushDictationStateToBorder: @MainActor (DictationState) -> Void
        let pushDictationStateToMenuBar: @MainActor (DictationState) -> Void
        let pushAudioLevelToPill: @MainActor (Float) -> Void
        let pushAudioLevelToBorder: @MainActor (Float) -> Void
        let updateDictationHotkey: @MainActor (KeyCombo) -> Void
        let updateTTSHotkey: @MainActor (KeyCombo) -> Void
        let updateAgentHotkey: @MainActor (KeyCombo) -> Void
        let startHTTPServer: @MainActor () async -> Void
        let stopHTTPServer: @MainActor () -> Void
        let updateHTTPServerPort: @MainActor (UInt16) async -> Void
        let reloadLLMIfNeeded: @MainActor () async throws -> Void
        /// Loads the Whisper model from its on-disk path into the
        /// transcription engine.
        let loadWhisperModel: @MainActor (URL) async -> Void

        init(
            setBorderGlowTheme: @escaping @MainActor (GlowTheme) -> Void,
            setUpOverlayPanels: @escaping @MainActor () -> Void,
            setPillOverlayEnabled: @escaping @MainActor (Bool) -> Void,
            setBorderOverlayEnabled: @escaping @MainActor (Bool) -> Void,
            pushDictationStateToPill: @escaping @MainActor (DictationState) -> Void,
            pushDictationStateToBorder: @escaping @MainActor (DictationState) -> Void,
            pushDictationStateToMenuBar: @escaping @MainActor (DictationState) -> Void,
            pushAudioLevelToPill: @escaping @MainActor (Float) -> Void,
            pushAudioLevelToBorder: @escaping @MainActor (Float) -> Void,
            updateDictationHotkey: @escaping @MainActor (KeyCombo) -> Void,
            updateTTSHotkey: @escaping @MainActor (KeyCombo) -> Void,
            updateAgentHotkey: @escaping @MainActor (KeyCombo) -> Void,
            startHTTPServer: @escaping @MainActor () async -> Void,
            stopHTTPServer: @escaping @MainActor () -> Void,
            updateHTTPServerPort: @escaping @MainActor (UInt16) async -> Void,
            reloadLLMIfNeeded: @escaping @MainActor () async throws -> Void,
            loadWhisperModel: @escaping @MainActor (URL) async -> Void
        ) {
            self.setBorderGlowTheme = setBorderGlowTheme
            self.setUpOverlayPanels = setUpOverlayPanels
            self.setPillOverlayEnabled = setPillOverlayEnabled
            self.setBorderOverlayEnabled = setBorderOverlayEnabled
            self.pushDictationStateToPill = pushDictationStateToPill
            self.pushDictationStateToBorder = pushDictationStateToBorder
            self.pushDictationStateToMenuBar = pushDictationStateToMenuBar
            self.pushAudioLevelToPill = pushAudioLevelToPill
            self.pushAudioLevelToBorder = pushAudioLevelToBorder
            self.updateDictationHotkey = updateDictationHotkey
            self.updateTTSHotkey = updateTTSHotkey
            self.updateAgentHotkey = updateAgentHotkey
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
    private var whisperLoadTask: Task<Void, Never>?
    private var hasStarted = false

    init(settings: SettingsManager, inputs: Inputs, effects: Effects) {
        self.settings = settings
        self.inputs = inputs
        self.effects = effects
    }

    func start() {
        guard !hasStarted else { return }
        hasStarted = true

        // Seed the border's glow theme synchronously *before* its view is
        // created, so a user's non-default theme shows on the very first frame
        // instead of flashing the default while the async settings observation
        // catches up.
        effects.setBorderGlowTheme(settings.glowTheme)
        effects.setUpOverlayPanels()

        installSubscriptions()

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
        whisperLoadTask?.cancel()
        whisperLoadTask = nil
    }

    private func loadWhisperModelIfAvailable() async {
        guard let path = inputs.whisperModelPath() else {
            Log.general.warning("Whisper model not downloaded — download it from the Models page")
            return
        }
        await effects.loadWhisperModel(path)
    }

    // MARK: - Subscriptions

    /// Every subscription relies on `Observations` emitting the current value at
    /// subscription time to apply each rule's initial seed.
    private func installSubscriptions() {
        // Auto-load the Whisper model the moment its download completes, so
        // dictation works without restarting the app — unless the engine is
        // already serving a model.
        inputs.modelDownloadStatuses
            .compactMap { $0[WhisperModel.modelID] }
            .removeDuplicates()
            .sink { [weak self] status in
                guard case .downloaded = status else { return }
                guard let self, !self.inputs.isTranscriptionModelLoaded() else { return }
                Task {
                    await self.loadWhisperModelIfAvailable()
                }
            }
            .store(in: &cancellables)

        // Reload the agent model when the selection changes — but only while an
        // LLM slot is already loaded. The gate drops the initial emission, so a
        // selected-but-unused model never forces a multi-gigabyte load at launch.
        observationTasks.append(Task { [weak self] in
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
        observationTasks.append(Task { [weak self] in
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
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await port in Observations({ self.settings.serverPort }) {
                await self.effects.updateHTTPServerPort(HTTPServer.clampedPort(port))
            }
        })

        // The single dictation-state subscription: raw state fans out to both
        // overlay panels (the disabled one stays hidden) and the menu bar, so
        // every surface always sees the same emission — no second path, no race.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await state in Observations({ self.inputs.dictationState() }) {
                self.effects.pushDictationStateToPill(state)
                self.effects.pushDictationStateToBorder(state)
                self.effects.pushDictationStateToMenuBar(state)
            }
        })

        // Keep the border's glow theme live — pure view data, re-applied on change.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await glowTheme in Observations({ self.settings.glowTheme }) {
                self.effects.setBorderGlowTheme(glowTheme)
            }
        })

        // Forward audio level to both overlays; each drops it while disabled, so
        // the hidden overlay does no SwiftUI work at audio frame-rate. The
        // enabled gate inside the panel is the single source of truth for which
        // overlay is live — no separate active-overlay pointer to keep in sync.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await level in Observations({ self.inputs.audioLevel() }) {
                self.effects.pushAudioLevelToPill(level)
                self.effects.pushAudioLevelToBorder(level)
            }
        })

        // Re-bind the dictation hotkey on change. The initial emission matches
        // the combo the wiring already bound, so the unchanged-combo guard makes
        // launch a no-op instead of a redundant re-bind.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settings.hotkey }) {
                if hotkey != self.inputs.currentDictationHotkey() {
                    self.effects.updateDictationHotkey(hotkey)
                }
            }
        })

        // Re-bind the TTS and agent hotkeys on change; re-applying the current
        // combo at subscription time is a harmless re-register.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settings.ttsHotkey }) {
                self.effects.updateTTSHotkey(hotkey)
            }
        })
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settings.agentHotkey }) {
                self.effects.updateAgentHotkey(hotkey)
            }
        })

        // Enable the overlay matching the current style and disable the other.
        // Each panel's enabled flag gates both its visibility and its
        // audio-frame handling, so the inactive overlay stays hidden and idle.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await style in Observations({ self.settings.overlayStyle }) {
                switch style {
                case .pill:
                    self.effects.setPillOverlayEnabled(true)
                    self.effects.setBorderOverlayEnabled(false)
                case .fullScreenBorder:
                    self.effects.setPillOverlayEnabled(false)
                    self.effects.setBorderOverlayEnabled(true)
                }
            }
        })
    }
}
