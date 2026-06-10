//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
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

    // Agent (LLM). The inference actor is created here and injected into the
    // agent engine; the server dispatcher reaches the same actor for the
    // cache-aware completion route (ADR-0006). Plumb `settingsManager` so the
    // SSD prefix-cache config is snapshotted from live user settings at each
    // model load. Benchmark and test call sites use `AgentEngine()` to stay
    // SSD-disabled for reproducibility.
    lazy var llmActor = LLMActor()
    lazy var agentEngine = AgentEngine(
        settingsManager: settingsManager,
        llmActor: llmActor
    )

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
        completionStarter: llmActor,
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
    lazy var httpServer = HTTPServer(port: HTTPServer.clampedPort(settingsManager.serverPort))

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

    // Menu bar — constructed here so App Bindings can wire its dictation-state
    // effect before the app delegate attaches the window-management callbacks.
    lazy var menuBarManager: MenuBarManager = {
        let manager = MenuBarManager(settings: settingsManager)
        manager.coordinator = dictationCoordinator
        manager.history = transcriptionHistory
        manager.speechCoordinator = speechCoordinator
        return manager
    }()

    // App Bindings owns the launch sequence and every runtime subscription
    // with a rule; this container only wires its inputs and effects.
    lazy var appBindings = makeAppBindings()

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

        // Register HTTP server routes before App Bindings starts the server.
        registerHTTPRoutes()

        // Hand off to App Bindings: the launch ordering and every runtime
        // subscription with a rule live (and are tested) there.
        appBindings.start()
    }

    private func makeAppBindings() -> AppBindings {
        AppBindings(
            settings: settingsManager,
            inputs: .init(
                dictationState: { [dictationCoordinator] in
                    dictationCoordinator.state
                },
                audioLevel: { [audioCaptureEngine] in
                    audioCaptureEngine.audioLevel
                },
                currentDictationHotkey: { [hotkeyManager] in
                    hotkeyManager.currentHotkey
                },
                isLLMSlotLoaded: { [inferenceArbiter] in
                    inferenceArbiter.loadedSlots.contains(.llm)
                },
                whisperModelPath: { [modelManager] in
                    modelManager.isModelAvailable() ? modelManager.getModelPath() : nil
                },
                isTranscriptionModelLoaded: { [transcriptionEngine] in
                    transcriptionEngine.isModelLoaded
                },
                modelDownloadStatuses: modelDownloadManager.$statuses.eraseToAnyPublisher()
            ),
            effects: .init(
                setBorderGlowTheme: { [borderOverlay] in
                    borderOverlay.state.glowTheme = $0
                },
                setUpOverlayPanels: { [pillOverlay, borderOverlay] in
                    pillOverlay.setup()
                    borderOverlay.setup()
                },
                setPillOverlayEnabled: { [pillOverlay] in
                    pillOverlay.setEnabled($0)
                },
                setBorderOverlayEnabled: { [borderOverlay] in
                    borderOverlay.setEnabled($0)
                },
                pushDictationStateToPill: { [pillOverlay] in
                    pillOverlay.handleStateChange($0)
                },
                pushDictationStateToBorder: { [borderOverlay] in
                    borderOverlay.handleStateChange($0)
                },
                pushDictationStateToMenuBar: { [menuBarManager] in
                    menuBarManager.updateState(from: $0)
                },
                pushAudioLevelToPill: { [pillOverlay] in
                    pillOverlay.handleAudioLevelChange($0)
                },
                pushAudioLevelToBorder: { [borderOverlay] in
                    borderOverlay.handleAudioLevelChange($0)
                },
                updateDictationHotkey: { [hotkeyManager] in
                    hotkeyManager.updateHotkey($0)
                },
                updateTTSHotkey: { [hotkeyManager] in
                    hotkeyManager.updateRegisteredHotkey(id: "tts", combo: $0)
                },
                updateAgentHotkey: { [hotkeyManager] in
                    hotkeyManager.updateRegisteredHotkey(id: "agent", combo: $0)
                },
                startHTTPServer: { [httpServer] in
                    await httpServer.start()
                },
                stopHTTPServer: { [httpServer] in
                    httpServer.stop()
                },
                updateHTTPServerPort: { [httpServer] in
                    await httpServer.updatePort($0)
                },
                reloadLLMIfNeeded: { [inferenceArbiter] in
                    try await inferenceArbiter.reloadLLMIfNeeded()
                },
                loadWhisperModel: { [transcriptionEngine] modelPath in
                    do {
                        try await transcriptionEngine.loadModel(from: modelPath)
                        Log.general.info("Loaded Whisper model from: \(modelPath.path)")
                    } catch {
                        Log.general.error("Failed to load Whisper model: \(error)")
                    }
                }
            )
        )
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
