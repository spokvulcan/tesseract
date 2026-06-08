//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
import Observation
import SwiftUI
import MLX
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
        ToolRegistry(sandbox: agentSandbox, extensionHost: extensionHost)
    }()
    lazy var agentConversationStore = AgentConversationStore()
    lazy var inferenceArbiter: InferenceArbiter = {
        InferenceArbiter(
            agentEngine: agentEngine,
            speechEngine: speechEngine,
            settingsManager: settingsManager,
            modelDownloadManager: modelDownloadManager
        )
    }()
    lazy var serverInferenceService = ServerInferenceService(
        engine: agentEngine,
        arbiter: inferenceArbiter
    )
    lazy var serverGenerationLog = ServerGenerationLog()
    lazy var promptCacheTelemetryStore = PromptCacheTelemetryStore()
    lazy var agent: Agent = AgentFactory.makeAgent(
        inferenceService: serverInferenceService,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        settingsManager: settingsManager
    )
    // HTTP Server
    lazy var httpServer = HTTPServer(port: UInt16(clamping: max(1, settingsManager.serverPort)))

    lazy var terminationCoordinator = makeTerminationCoordinator()
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
                    settingsManager.makeAgentGenerateParameters()
                }
            )
        )
    }()

    // Speech (TTS)
    lazy var textExtractor = TextExtractor()
    lazy var speechEngine = SpeechEngine()
    lazy var ttsNotchPanelController = TTSNotchPanelController()
    lazy var speechCoordinator: SpeechCoordinator = {
        // `playback` is left to the coordinator's production default
        // (`AudioPlaybackManager()`) — the AVFoundation adapter is needed by
        // nothing else in the graph, so there is no shared handle to wire here.
        // Mirrors `SettingsManager()` above, which relies on its
        // `UserDefaultsSettingsStore()` default. Tests inject `InMemoryAudioPlayback`.
        SpeechCoordinator(
            textExtractor: textExtractor,
            speechEngine: speechEngine,
            settings: settingsManager,
            notchOverlay: ttsNotchPanelController,
            arbiter: inferenceArbiter
        )
    }()

    // Overlays — two configured instances of the one OverlayPanel module.
    lazy var pillOverlay = OverlayPanel(
        state: OverlayState(),
        placement: .pill,
        content: { GlobalOverlayHUD(overlayState: $0) }
    )
    lazy var borderOverlay = OverlayPanel(
        state: OverlayState(),
        placement: .fullScreenBorder,
        content: { FullScreenBorderOverlayView(overlayState: $0) }
    )
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

    func prepareForTermination() async {
        await terminationCoordinator.prepareForTermination()
    }

    private func makeTerminationCoordinator() -> AppTerminationCoordinator {
        AppTerminationCoordinator(steps: .init(
            stopHotkeys: { [hotkeyManager] in
                hotkeyManager.stopListening()
            },
            cancelForegroundGenerationAndWait: { [agentCoordinator] in
                await agentCoordinator.cancelGenerationAndWait()
            },
            stopHTTPServerAndDrain: { [httpServer] in
                await httpServer.stopAndDrain()
            },
            cancelLLMGenerationAndWait: { [agentEngine] in
                await agentEngine.cancelGenerationAndWait()
            },
            stopSpeech: { [speechCoordinator] in
                speechCoordinator.stop()
            },
            cancelSpeechGeneration: { [speechEngine] in
                await speechEngine.cancelGeneration()
            },
            clearSpeechVoiceAnchor: { [speechEngine] in
                await speechEngine.clearVoiceAnchor()
            },
            unloadLLM: { [agentEngine] in
                agentEngine.unloadModel()
            },
            awaitLLMUnload: { [agentEngine] in
                await agentEngine.awaitPendingUnload()
            },
            unloadSpeech: { [speechEngine] in
                speechEngine.unloadModel()
            },
            synchronizeGPU: {
                Stream.gpu.synchronize()
            }
        ))
    }

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
            onDown: { [weak self] in self?.agentCoordinator.voiceInput.start() },
            onUp: { [weak self] in self?.agentCoordinator.voiceInput.finishCapture() }
        )

        // Register message codecs for the new persistence layer (Epic 2)
        await registerCoreMessageCodecs()

        hotkeyManager.startListening()
        startSettingsObservation()

        // Setup overlay panels. Seed the border's glow theme synchronously
        // *before* its view is created, so a user's non-default theme shows on the
        // very first frame instead of flashing the default while the settings
        // observation below catches up (it emits at subscription, but async).
        borderOverlay.state.glowTheme = settingsManager.glowTheme
        pillOverlay.setup()
        borderOverlay.setup()

        // Forward dictation state to both overlays (the disabled one stays hidden).
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await state in Observations({ self.dictationCoordinator.state }) {
                self.pillOverlay.handleStateChange(state)
                self.borderOverlay.handleStateChange(state)
            }
        })

        // Forward audio level to both overlays; each drops it while disabled, so the
        // hidden overlay does no SwiftUI work at audio frame-rate. The `isEnabled`
        // gate inside the panel is the single source of truth for which overlay is
        // live — no separate active-overlay pointer to keep in sync.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await level in Observations({ self.audioCaptureEngine.audioLevel }) {
                self.pillOverlay.handleAudioLevelChange(level)
                self.borderOverlay.handleAudioLevelChange(level)
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

        // Keep the border's glow theme live — pure view data, set on state directly.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await glowTheme in Observations({ self.settingsManager.glowTheme }) {
                self.borderOverlay.state.glowTheme = glowTheme
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

        // Guarded on an already-loaded `.llm` slot so the initial `Observations`
        // emit at subscription time does not force a model load at app launch —
        // lazy loading is preserved.
        observationTasks.append(Task { [weak self] in
            guard let self else { return }
            for await _ in Observations({ self.settingsManager.selectedAgentModelID }) {
                guard self.inferenceArbiter.loadedSlots.contains(.llm) else { continue }
                do {
                    try await self.inferenceArbiter.reloadLLMIfNeeded()
                } catch {
                    Log.agent.error("Agent model reload failed: \(error.localizedDescription)")
                }
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

    /// Enables the overlay matching the current style and disables the other. Each
    /// panel's `isEnabled` then gates both its visibility and its audio-frame
    /// handling, so the disabled overlay stays hidden and idle.
    private func updateActiveOverlay() {
        switch settingsManager.overlayStyle {
        case .pill:
            pillOverlay.setEnabled(true)
            borderOverlay.setEnabled(false)
        case .fullScreenBorder:
            pillOverlay.setEnabled(false)
            borderOverlay.setEnabled(true)
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
            downloads: modelDownloadManager,
            activityLog: serverGenerationLog,
            settings: settingsManager
        )
        httpServer.route(.POST, "/v1/chat/completions") { request, writer in
            try await completionHandler.handle(request: request, writer: writer)
        }
    }
}
