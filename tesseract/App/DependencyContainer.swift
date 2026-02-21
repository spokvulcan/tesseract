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

    // Agent (LLM)
    lazy var agentEngine = AgentEngine()
    lazy var agentDataStore = AgentDataStore()
    lazy var toolRegistry: ToolRegistry = {
        let store = agentDataStore
        return ToolRegistry(tools: [
            MemorySaveTool(store: store),
            MemoryUpdateTool(store: store),
            MemoryDeleteTool(store: store),
            GoalCreateTool(store: store),
            GoalListTool(store: store),
            GoalUpdateTool(store: store),
            TaskCreateTool(store: store),
            TaskListTool(store: store),
            TaskCompleteTool(store: store),
            HabitCreateTool(store: store),
            HabitLogTool(store: store),
            HabitStatusTool(store: store),
            MoodLogTool(store: store),
            MoodListTool(store: store),
            ReminderSetTool(store: store),
            RespondTool(),
        ])
    }()
    lazy var agentRunner: AgentRunner = {
        AgentRunner(engine: agentEngine, toolRegistry: toolRegistry)
    }()
    lazy var agentConversationStore = AgentConversationStore()
    lazy var agentNotchController = AgentNotchPanelController()
    lazy var agentCoordinator: AgentCoordinator = {
        AgentCoordinator(
            agentRunner: agentRunner,
            conversationStore: agentConversationStore,
            dataStore: agentDataStore,
            audioCapture: audioCaptureEngine,
            transcriptionEngine: transcriptionEngine,
            settings: settingsManager,
            prepareForInference: { [weak self] in
                guard let self else { return }
                if imageGenEngine.isModelLoaded {
                    imageGenEngine.releaseModel()
                    Log.general.info("Released image gen model for agent inference")
                }
                if zimageGenEngine.isModelLoaded {
                    zimageGenEngine.releaseModel()
                    Log.general.info("Released Z-image gen model for agent inference")
                }
            },
            loadAgentModel: { [weak self] in
                guard let self else { return }
                let modelID = self.settingsManager.selectedAgentModelID
                guard case .downloaded = self.modelDownloadManager.statuses[modelID],
                      !self.agentEngine.isLoading,
                      let path = self.modelDownloadManager.modelPath(for: modelID)
                else { return }
                if self.agentEngine.isModelLoaded {
                    self.agentEngine.unloadModel()
                }
                try await self.agentEngine.loadModel(from: path)
            },
            speechCoordinator: speechCoordinator,
            notchController: agentNotchController
        )
    }()

    // Image Generation
    lazy var imageGenEngine = ImageGenEngine()
    lazy var zimageGenEngine = ZImageGenEngine()

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
            notchOverlay: ttsNotchPanelController
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
        // Register Agent hotkey
        hotkeyManager.registerHotkey(
            id: "agent",
            combo: settingsManager.agentHotkey,
            onDown: { [weak self] in self?.onAgentHotkeyDown() },
            onUp: { [weak self] in self?.onAgentHotkeyUp() }
        )

        // Wire agent notch tap to navigate to agent window
        agentNotchController.onTap = {
            guard let appDelegate = NSApp.delegate as? AppDelegate else { return }
            appDelegate.navigateToAgent()
        }

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

    private func onAgentHotkeyDown() {
        agentCoordinator.startVoiceInput()
        startAgentAudioLevelForwarding()
    }

    private func onAgentHotkeyUp() {
        stopAgentAudioLevelForwarding()
        agentCoordinator.stopVoiceInputAndSend()
    }

    private var agentAudioLevelCancellable: AnyCancellable?

    private func startAgentAudioLevelForwarding() {
        agentAudioLevelCancellable = audioCaptureEngine.$audioLevel
            .receive(on: RunLoop.main)
            .sink { [weak self] level in
                guard let self else { return }
                self.agentNotchController.state.phase = .listening(audioLevel: level)
            }
    }

    private func stopAgentAudioLevelForwarding() {
        agentAudioLevelCancellable?.cancel()
        agentAudioLevelCancellable = nil
    }

    private var lastTTSHotkey: KeyCombo?
    private var lastAgentHotkey: KeyCombo?

    private func startSettingsObservation() {
        lastTTSHotkey = settingsManager.ttsHotkey
        lastAgentHotkey = settingsManager.agentHotkey

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

                let newAgentHotkey = settingsManager.agentHotkey
                if newAgentHotkey != lastAgentHotkey {
                    lastAgentHotkey = newAgentHotkey
                    hotkeyManager.updateRegisteredHotkey(id: "agent", combo: newAgentHotkey)
                }
            }
            .store(in: &settingsCancellables)
    }
}
