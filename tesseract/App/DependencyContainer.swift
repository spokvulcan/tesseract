//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
import Observation
import SwiftUI
import os

@MainActor
final class DependencyContainer: ObservableObject {
    // Core Services
    let settingsManager = SettingsManager()
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

    // New architecture (Epics 0-5)
    lazy var agentSandbox: PathSandbox = {
        PathSandbox(root: PathSandbox.defaultRoot)
    }()
    lazy var extensionHost = ExtensionHost()
    lazy var packageRegistry = PackageRegistry()
    lazy var contextManager = ContextManager(settings: .standard)
    lazy var newToolRegistry: ToolRegistry = {
        ToolRegistry(sandbox: agentSandbox, extensionHost: extensionHost, schedulingService: schedulingService)
    }()
    lazy var agent: Agent = AgentFactory.makeAgent(
        engine: agentEngine,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        selectedModelID: settingsManager.selectedAgentModelID
    )
    lazy var agentConversationStore = AgentConversationStore()
    lazy var scheduledTaskStore = ScheduledTaskStore()
    lazy var schedulingActor: SchedulingActor = {
        SchedulingActor(
            taskStore: scheduledTaskStore,
            executeTask: { task in
                // Phase 1 stub — BackgroundAgentFactory replaces this
                Log.agent.info("SchedulingActor stub: would execute '\(task.name)'")
                return .noActionNeeded
            }
        )
    }()
    lazy var schedulingService: SchedulingService = {
        SchedulingService(actor: schedulingActor, store: scheduledTaskStore)
    }()
    lazy var agentCoordinator: AgentCoordinator = {
        AgentCoordinator(
            agent: agent,
            conversationStore: agentConversationStore,
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
            formatRawPrompt: { [weak self] systemPrompt, tools in
                guard let self else { throw AgentEngineError.modelNotLoaded }
                return try await self.agentEngine.formatRawPrompt(systemPrompt: systemPrompt, tools: tools)
            },
            speechCoordinator: speechCoordinator
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
    lazy var overlayPanelController = OverlayPanelController(settings: settingsManager)
    lazy var fullScreenBorderController = FullScreenBorderPanelController(settings: settingsManager)
    private var cancellables = Set<AnyCancellable>()
    private var observationTasks: [Task<Void, Never>] = []

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
            onDown: { [weak self] in self?.agentCoordinator.startVoiceInput() },
            onUp: { [weak self] in self?.agentCoordinator.stopVoiceInputAndSend() }
        )

        // Register message codecs for the new persistence layer (Epic 2)
        await registerCoreMessageCodecs()

        hotkeyManager.startListening()
        startSettingsObservation()

        // Setup overlay panels
        overlayPanelController.setup()
        fullScreenBorderController.setup()

        // Forward dictation state to overlay panels
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await state in Observations({ self.dictationCoordinator.state }) {
                self.overlayPanelController.handleStateChange(state)
                self.fullScreenBorderController.handleStateChange(state)
            }
        })

        // Forward audio level to overlay panels
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await level in Observations({ self.audioCaptureEngine.audioLevel }) {
                self.overlayPanelController.handleAudioLevelChange(level)
                self.fullScreenBorderController.handleAudioLevelChange(level)
            }
        })

        // Set initial active overlay based on settings
        updateActiveOverlay()

        // Observe overlay style changes to switch overlays dynamically
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await _ in Observations({ self.settingsManager.overlayStyle }) {
                self.updateActiveOverlay()
            }
        })

        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await glowTheme in Observations({ self.settingsManager.glowTheme }) {
                self.fullScreenBorderController.handleGlowThemeChange(glowTheme)
            }
        })

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
            .store(in: &cancellables)

        // Start scheduling service (includes polling loop)
        await schedulingService.start()
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

    private func startSettingsObservation() {
        // Observe dictation hotkey changes
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settingsManager.hotkey }) {
                if hotkey != self.hotkeyManager.currentHotkey {
                    self.hotkeyManager.updateHotkey(hotkey)
                }
            }
        })

        // Observe TTS hotkey changes
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settingsManager.ttsHotkey }) {
                self.hotkeyManager.updateRegisteredHotkey(id: "tts", combo: hotkey)
            }
        })

        // Observe agent hotkey changes
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await hotkey in Observations({ self.settingsManager.agentHotkey }) {
                self.hotkeyManager.updateRegisteredHotkey(id: "agent", combo: hotkey)
            }
        })
    }
}
