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
    lazy var modelManager = ModelManager()
    lazy var transcriptionEngine = TranscriptionEngine()
    lazy var transcriptionHistory = TranscriptionHistory()

    // Text Injection
    lazy var textInjector = TextInjector()
    lazy var hotkeyManager = HotkeyManager()

    // Speech (TTS)
    lazy var textExtractor = TextExtractor()
    lazy var speechEngine = SpeechEngine()
    lazy var audioPlaybackManager = AudioPlaybackManager()
    lazy var speechCoordinator: SpeechCoordinator = {
        SpeechCoordinator(
            textExtractor: textExtractor,
            speechEngine: speechEngine,
            playbackManager: audioPlaybackManager,
            settings: settingsManager,
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

        // Load bundled model
        if let modelPath = modelManager.getBundledModelPath() {
            do {
                try await transcriptionEngine.loadModel(from: modelPath)
                Log.general.info("Loaded bundled model from: \(modelPath.path)")
            } catch {
                Log.general.error("Failed to load bundled model: \(error)")
            }
        } else {
            Log.general.warning("Bundled model not found")
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
