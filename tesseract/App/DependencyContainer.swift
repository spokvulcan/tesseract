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

    // Agent (LLM). Plumb `settingsManager` so the SSD prefix-cache config
    // is snapshotted from live user settings at each model load. Benchmark
    // and test call sites use `AgentEngine()` to stay SSD-disabled for
    // reproducibility.
    lazy var agentEngine = AgentEngine(settingsManager: settingsManager)

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
    lazy var agentConversationStore = AgentConversationStore()
    lazy var scheduledTaskStore = ScheduledTaskStore()
    lazy var backgroundSessionStore = BackgroundSessionStore()
    lazy var inferenceArbiter: InferenceArbiter = {
        InferenceArbiter(
            agentEngine: agentEngine,
            speechEngine: speechEngine,
            imageGenEngine: imageGenEngine,
            zimageGenEngine: zimageGenEngine,
            settingsManager: settingsManager,
            modelDownloadManager: modelDownloadManager
        )
    }()
    lazy var serverInferenceService = ServerInferenceService(
        engine: agentEngine,
        arbiter: inferenceArbiter
    )
    lazy var agent: Agent = AgentFactory.makeAgent(
        inferenceService: serverInferenceService,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        settingsManager: settingsManager
    )
    lazy var backgroundAgentFactory: BackgroundAgentFactory = {
        BackgroundAgentFactory(
            inferenceService: serverInferenceService,
            toolRegistry: newToolRegistry,
            extensionHost: extensionHost,
            sessionStore: backgroundSessionStore,
            packageRegistry: packageRegistry,
            settingsManager: settingsManager,
            arbiter: inferenceArbiter
        )
    }()
    lazy var schedulingActor: SchedulingActor = {
        SchedulingActor(
            taskStore: scheduledTaskStore,
            executeTask: { [weak self] task in
                guard let factory = await MainActor.run(body: {
                    self?.schedulingService.resetAgentTurnCounter()
                    return self?.backgroundAgentFactory
                }) else {
                    return .error(message: "DependencyContainer deallocated")
                }
                return await factory.executeAndPersist(task: task)
            },
            executeHeartbeat: { [weak self] task in
                guard let factory = await MainActor.run(body: {
                    self?.schedulingService.resetAgentTurnCounter()
                    return self?.backgroundAgentFactory
                }) else {
                    return .error(message: "DependencyContainer deallocated")
                }
                return await factory.executeAndPersist(task: task, sessionType: .heartbeat)
            },
            persistInFlightSession: { [weak self] in
                guard let factory = await MainActor.run(body: { self?.backgroundAgentFactory }) else { return }
                await factory.persistInFlightSession()
            }
        )
    }()
    // HTTP Server
    lazy var httpServer = HTTPServer(port: UInt16(clamping: max(1, settingsManager.serverPort)))

    lazy var notificationService = NotificationService(settings: settingsManager)
    lazy var schedulingService: SchedulingService = {
        SchedulingService(actor: schedulingActor, store: scheduledTaskStore, settings: settingsManager, notificationService: notificationService, speechCoordinator: speechCoordinator)
    }()
    lazy var agentCoordinator: AgentCoordinator = {
        AgentCoordinator(
            agent: agent,
            conversationStore: agentConversationStore,
            audioCapture: audioCaptureEngine,
            transcriptionEngine: transcriptionEngine,
            settings: settingsManager,
            arbiter: inferenceArbiter,
            formatRawPrompt: { [weak self] systemPrompt, tools in
                guard let self else { throw AgentEngineError.modelNotLoaded }
                return try await self.agentEngine.formatRawPrompt(systemPrompt: systemPrompt, tools: tools)
            },
            speechCoordinator: speechCoordinator,
            toolRegistry: newToolRegistry,
            extensionHost: extensionHost,
            packageRegistry: packageRegistry,
            contextManager: contextManager,
            contextWindow: 262_144,
            summarize: makeSummarizeClosure(
                inferenceService: serverInferenceService,
                parametersProvider: { [settingsManager] in
                    var parameters = AgentGenerateParameters.forModel(settingsManager.selectedAgentModelID)
                    parameters.triAttention = settingsManager.makeTriAttentionConfig()
                    return parameters
                }
            )
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
            notchOverlay: ttsNotchPanelController,
            arbiter: inferenceArbiter
        )
    }()

    // Overlays
    lazy var overlayPanelController = OverlayPanelController(settings: settingsManager)
    lazy var fullScreenBorderController = FullScreenBorderPanelController(settings: settingsManager)
    private var cancellables = Set<AnyCancellable>()
    private var agentUnsubscribe: (@MainActor () -> Void)?
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

        // Wire background session loader for notification deep-links
        agentCoordinator.loadBackgroundSessionById = { [backgroundSessionStore] sessionId in
            guard let session = await backgroundSessionStore.load(sessionId: sessionId) else {
                throw CocoaError(.fileNoSuchFile)
            }
            let messages = try SyncMessageCodec.decodeAll(session.messages)
            return (messages, session.displayName)
        }

        // Reset agent turn counter when a new agent run starts
        agentUnsubscribe = agent.subscribe { [weak schedulingService] event in
            if case .agentStart = event {
                schedulingService?.resetAgentTurnCounter()
            }
        }

        // Sync notification authorization changes from PermissionsManager → NotificationService
        permissionsManager.onNotificationAuthorizationChanged = { [weak self] authorized in
            self?.notificationService.syncAuthorization(authorized)
        }

        // Request notification authorization before starting the scheduling service so that
        // missed-run catch-ups on launch can post notifications immediately.
        await notificationService.requestAuthorization()
        let authorized = await permissionsManager.checkNotificationPermission()
        notificationService.syncAuthorization(authorized)

        // Start scheduling service (includes polling loop + heartbeat from persisted config)
        await schedulingService.start()

        // Register HTTP server routes
        registerHTTPRoutes()

        // Observe server toggle/port changes (Observations emits current value immediately)
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await enabled in Observations({ self.settingsManager.isServerEnabled }) {
                if enabled {
                    await self.httpServer.start()
                } else {
                    self.httpServer.stop()
                }
            }
        })
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await port in Observations({ self.settingsManager.serverPort }) {
                await self.httpServer.updatePort(UInt16(clamping: max(1, port)))
            }
        })
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

    private func registerHTTPRoutes() {
        httpServer.route(.GET, "/health") { _, writer in
            try await writer.send(.json(["status": "ok"] as [String: String]))
        }

        let engine = agentEngine
        let arbiter = inferenceArbiter
        let downloads = modelDownloadManager
        httpServer.route(.GET, "/v1/models") { _, writer in
            // Encodable conformance requires MainActor context (Swift 6.2 isolation inference)
            let data: Data = await MainActor.run {
                // List all agent-category models that are downloaded. The
                // currently-loaded one is reported with `state: "loaded"`;
                // the rest are `"available"`. Undownloaded models are omitted
                // because `CompletionHandler` ignores `request.model` for
                // routing (see docs/HTTP_SERVER_SPEC.md §4.2) — it always
                // serves whatever `selectedAgentModelID` resolves to — so
                // advertising an undownloaded id would be misleading.
                let loadedID: String? = engine.isModelLoaded ? arbiter.loadedLLMModelID : nil
                let models: [OpenAI.ModelObject] = ModelDefinition.all
                    .filter { $0.category == .agent }
                    .compactMap { definition -> OpenAI.ModelObject? in
                        guard case .downloaded = downloads.statuses[definition.id] else {
                            return nil
                        }
                        let isLoaded = definition.id == loadedID
                        return OpenAI.ModelObject(
                            id: definition.id,
                            type: "llm",
                            owned_by: "tesseract",
                            max_context_length: 262_144,
                            loaded_context_length: isLoaded ? 262_144 : nil,
                            state: isLoaded ? "loaded" : "available"
                        )
                    }
                return (try? JSONEncoder().encode(OpenAI.ModelListResponse(data: models)))
                    ?? Data("{}".utf8)
            }
            try await writer.send(.jsonBody(data))
        }

        let completionHandler = CompletionHandler(
            arbiter: inferenceArbiter,
            inferenceService: serverInferenceService,
            downloads: modelDownloadManager
        )
        httpServer.route(.POST, "/v1/chat/completions") { request, writer in
            try await completionHandler.handle(request: request, writer: writer)
        }
    }
}
