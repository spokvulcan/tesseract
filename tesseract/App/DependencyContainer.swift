//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
import SwiftUI
import os

@MainActor
final class DependencyContainer: ObservableObject {
    // Core Services
    lazy var settingsManager = SettingsManager.shared
    lazy var permissionsManager = PermissionsManager()
    lazy var audioDeviceManager = AudioDeviceManager()

    // Audio
    lazy var audioCaptureEngine = AudioCaptureEngine()

    // Transcription
    lazy var modelManager = ModelManager(modelDownloadManager: modelDownloadManager)
    lazy var transcriptionEngine = TranscriptionEngine()
    lazy var transcriptionHistory = TranscriptionHistory()

    // Text Injection
    lazy var textInjector = TextInjector()
    lazy var hotkeyManager = HotkeyManager()

    // Model Downloads
    lazy var modelDownloadManager = ModelDownloadManager()

    // Speech (TTS)
    lazy var textExtractor = TextExtractor()
    lazy var speechEngine = SpeechEngine()
    lazy var audioPlaybackManager = AudioPlaybackManager()
    lazy var ttsNotchPanelController = TTSNotchPanelController()
    lazy var speechCoordinator: SpeechCoordinator = {
        SpeechCoordinator(
            textExtractor: textExtractor,
            speechEngine: speechEngine,
            playbackManager: audioPlaybackManager,
            settings: settingsManager,
            notchOverlay: ttsNotchPanelController,
            prepareForSpeech: { [weak self] in
                self?.transcriptionEngine.releaseModelForTTSIfIdle()
            }
        )
    }()

    // Overlays
    lazy var overlayPanelController = OverlayPanelController()
    lazy var fullScreenBorderController = FullScreenBorderPanelController()
    private var settingsCancellables = Set<AnyCancellable>()

    // Coordinator
    lazy var dictationCoordinator: DictationCoordinator = {
        DictationCoordinator(
            audioCapture: audioCaptureEngine,
            transcriptionEngine: transcriptionEngine,
            textInjector: textInjector,
            history: transcriptionHistory,
            settings: settingsManager
        )
    }()

    private var hasSetup = false

    init() {}

    func setup() async {
        // Prevent duplicate setup from multiple window instances
        guard !hasSetup else { return }
        hasSetup = true
        // Setup hotkey callbacks
        hotkeyManager.currentHotkey = settingsManager.hotkey
        hotkeyManager.onHotkeyDown = { [weak self] in
            self?.dictationCoordinator.onHotkeyDown()
        }
        hotkeyManager.onHotkeyUp = { [weak self] in
            self?.dictationCoordinator.onHotkeyUp()
        }
        // Register TTS hotkey
        hotkeyManager.registerHotkey(
            id: "tts",
            combo: settingsManager.ttsHotkey,
            onDown: { [weak self] in
                self?.speechCoordinator.onHotkeyPressed()
            }
        )

        hotkeyManager.startListening()
        startSettingsObservation()

        // Setup overlay panels
        overlayPanelController.setup(
            statePublisher: dictationCoordinator.$state,
            audioLevelPublisher: audioCaptureEngine.$audioLevel
        )

        fullScreenBorderController.setup(
            statePublisher: dictationCoordinator.$state,
            audioLevelPublisher: audioCaptureEngine.$audioLevel
        )

        // Set initial active overlay based on settings
        updateActiveOverlay()

        // Observe settings changes to switch overlays dynamically
        // Use NotificationCenter to observe UserDefaults changes for overlayStyle
        NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.updateActiveOverlay()
            }
            .store(in: &settingsCancellables)

        // Load Whisper model from cache if already downloaded
        await loadWhisperModelIfAvailable()

        // Auto-load Whisper model when download completes
        modelDownloadManager.$statuses
            .compactMap { $0[WhisperModel.modelID] }
            .removeDuplicates()
            .sink { [weak self] status in
                guard case .downloaded = status else { return }
                guard let self, !self.transcriptionEngine.isModelLoaded else { return }
                Task {
                    await self.loadWhisperModelIfAvailable()
                }
            }
            .store(in: &settingsCancellables)
    }

    private func loadWhisperModelIfAvailable() async {
        if modelManager.isModelAvailable(), let modelPath = modelManager.getModelPath() {
            do {
                try await transcriptionEngine.loadModel(from: modelPath)
                Log.general.info("Loaded Whisper model from: \(modelPath.path)")
            } catch {
                Log.general.error("Failed to load Whisper model: \(error)")
            }
        } else {
            Log.general.warning("Whisper model not downloaded — download it from the Models page")
        }
    }

    /// Updates which overlay controller is active based on current settings
    private func updateActiveOverlay() {
        switch settingsManager.overlayStyle {
        case .pill:
            overlayPanelController.setEnabled(true)
            fullScreenBorderController.setEnabled(false)
        case .fullScreenBorder:
            overlayPanelController.setEnabled(false)
            fullScreenBorderController.setEnabled(true)
        }
    }

    private var lastTTSHotkey: KeyCombo?

    private func startSettingsObservation() {
        lastTTSHotkey = settingsManager.ttsHotkey

        NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                guard let self else { return }
                let newHotkey = settingsManager.hotkey
                if newHotkey != hotkeyManager.currentHotkey {
                    hotkeyManager.updateHotkey(newHotkey)
                }

                let newTTSHotkey = settingsManager.ttsHotkey
                if newTTSHotkey != lastTTSHotkey {
                    lastTTSHotkey = newTTSHotkey
                    hotkeyManager.updateRegisteredHotkey(id: "tts", combo: newTTSHotkey)
                }
            }
            .store(in: &settingsCancellables)
    }
}
