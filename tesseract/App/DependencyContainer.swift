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

    /// The **Capture Dump** (PRD #175) — one shared ring buffer for every
    /// capture surface, so bounds are enforced across the whole app.
    lazy var captureDumpStore: CaptureDumpStore = {
        let base =
            FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first ?? FileManager.default.temporaryDirectory
        return CaptureDumpStore(
            directory:
                base
                .appendingPathComponent("Tesseract Agent", isDirectory: true)
                .appendingPathComponent("CaptureDump", isDirectory: true)
        )
    }()

    // Transcription
    lazy var transcriptionEngine = TranscriptionEngine()
    lazy var transcriptionHistory = TranscriptionHistory()

    // Text Injection
    lazy var textInjector = TextInjector()
    lazy var hotkeyManager = HotkeyManager()

    // Model Downloads
    lazy var modelDownloadManager = ModelDownloadManager()

    // Agent (LLM). The inference actor is created here and injected into the
    // agent engine; the server dispatcher reaches the same actor for the
    // cache-aware completion route (ADR-0015). Plumb `settingsManager` so the
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
    lazy var promptCacheTelemetryStore = PromptCacheTelemetryStore(
        enduranceAccumulator: ssdEnduranceAccumulator
    )
    /// Eager (non-lazy) so the JSONL diagnostics file is written from the
    /// first request on, whether or not the telemetry UI ever opens.
    let promptCacheDiagnosticsFileSink = PromptCacheDiagnosticsFileSink()
    /// Eager for the same reason: the endurance ledger (PRD #150) must
    /// count every SSD write/delete from launch — "persist from day
    /// one" is the ADR-0019 decision that replaces a write throttle.
    let ssdEnduranceAccumulator = SSDEnduranceAccumulator()
    lazy var agent: Agent = AgentFactory.makeAgent(
        inferenceService: serverInferenceService,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        settingsManager: settingsManager,
        mcpToolsExtension: mcpClientManager.toolsExtension
    )
    // HTTP Server
    lazy var httpServer = HTTPServer(port: HTTPServer.clampedPort(settingsManager.serverPort))

    // Agent Browser + Browser MCP Server (PRD #189). The browser owns the
    // Agent Profile and hands each MCP client a private Browser Session; the
    // MCP server rides the one loopback HTTP listener alongside the OpenAI
    // routes. Production shows real windows (ADR-0026: always-visible browsing).
    lazy var agentBrowser = AgentBrowser(presenter: AgentBrowserWindowPresenter())
    lazy var browserToolExecutor = BrowserToolExecutor(browser: agentBrowser)
    // Local-only tool-usage telemetry (ADR-0031): durable JSONL under
    // Application Support, covering both the in-app agent and external
    // HTTP clients through the one server choke point.
    lazy var browserMCPTelemetry = BrowserMCPTelemetryRecorder(
        isEnabled: { [settingsManager] in settingsManager.browserMCPTelemetryEnabled }
    )
    lazy var mcpBrowserServer = MCPBrowserServer(
        browser: agentBrowser,
        executor: browserToolExecutor,
        isEnabled: { [settingsManager] in settingsManager.browserMCPServerEnabled },
        telemetry: browserMCPTelemetry
    )

    // MCP client (PRD #190): the in-app agent connects to configured HTTP MCP
    // servers, and to its own Browser MCP server in-process (ADR-0027 dogfood).
    // The built-in Browser server is always connected in-process; the *Web
    // Access* switch (`webAccessEnabled`) governs whether its tools reach the
    // agent, applied per-turn by AgentRunController. The separate *HTTP exposure*
    // switch (`browserMCPServerEnabled`) gates only the loopback `/mcp` listener,
    // not this in-process path (ADR-0028). Reaching the browser server in-process
    // — not over the loopback socket — decouples browser-use in chat from the
    // inference HTTP listener (which only starts with `isServerEnabled`). Tools
    // land in `newToolRegistry` via the manager's `MCPToolsExtension`, refreshed
    // whenever a connection's tool set changes.
    lazy var mcpClientManager = MCPClientManager(
        configsProvider: { [settingsManager] in
            [MCPServerConfig.builtInBrowser(enabled: true)] + settingsManager.mcpServers
        },
        makeTransport: { [mcpBrowserServer] config in
            switch config.transport {
            case .inProcessBrowser:
                return InProcessMCPTransport(handle: { request in
                    await mcpBrowserServer.handle(request: request, origin: .inProcess)
                })
            case .http:
                // An unparseable persisted URL yields a nil endpoint; the
                // transport then fails the connection cleanly (US #6) rather
                // than pointing at a fabricated host.
                return HTTPMCPTransport(
                    endpoint: URL(string: config.url), headers: config.headers)
            }
        },
        refreshRegistry: { [newToolRegistry, extensionHost] in
            newToolRegistry.refreshExtensionTools(from: extensionHost)
        }
    )

    lazy var terminationCoordinator = makeTerminationCoordinator()

    // Chat leaves (ADR-0024): standalone controllers for everything not
    // derived from agent events. Constructed here — not in the views — because
    // cross-cutting surfaces need them too (Appshot stages into the Composer
    // Draft; the push-to-talk hotkey drives voice input).
    lazy var agentVoiceInput = AgentVoiceInputController(
        audioCapture: audioCaptureEngine,
        transcriptionEngine: transcriptionEngine,
        settings: settingsManager,
        captureDump: captureDumpStore
    )
    lazy var composerDraft = ComposerDraftController(conversationImages: { [agent] in
        agent.state.messages.flatMap { message -> [ImageAttachment] in
            if let user = message.asUser { return user.images }
            if let tool = message.asToolResult {
                return tool.content.imageAttachments(namespace: tool.id)
            }
            return []
        }
    })
    lazy var visionAvailability = VisionAvailabilityController(
        settings: settingsManager,
        draft: composerDraft,
        isVisionCapable: { [modelDownloadManager] in modelDownloadManager.isVisionCapable($0) },
        downloadedAgentModels: { [modelDownloadManager] in
            modelDownloadManager.downloadedModels(in: .agent)
        }
    )
    lazy var agentSystemPromptInspector: AgentSystemPromptInspector = {
        AgentSystemPromptInspector(
            promptSource: { [agent] in (agent.state.systemPrompt, agent.state.tools) },
            formatRawPrompt: { [weak self] systemPrompt, tools in
                guard let self else { throw AgentEngineError.modelNotLoaded }
                return try await self.agentEngine.formatRawPrompt(
                    systemPrompt: systemPrompt, tools: tools)
            }
        )
    }()
    lazy var commandPalette = SlashCommandPaletteController(
        extensionHost: extensionHost, packageRegistry: packageRegistry
    )
    lazy var skillPills = SkillPillController(
        discoverSkills: { [packageRegistry] in
            PackageBootstrap.discoverAgentSkills(packageRegistry: packageRegistry)
        },
        settings: settingsManager
    )

    // The Chat Session (ADR-0024): the single agent-event subscriber and the
    // store every chat view reads. Leaf behavior it needs (slash registry,
    // skill argument assembly, draft clearing) arrives as closures so the
    // session never holds the controllers.
    lazy var chatSession: ChatSession = {
        ChatSession(
            agent: agent,
            conversationStore: agentConversationStore,
            arbiter: inferenceArbiter,
            toolRegistry: newToolRegistry,
            settings: settingsManager,
            speechCoordinator: speechCoordinator,
            contextManager: contextManager,
            contextWindow: 262_144,
            summarize: makeSummarizeClosure(
                inferenceService: serverInferenceService,
                parametersProvider: { [settingsManager] in
                    settingsManager.makeAgentGenerateParameters()
                }
            ),
            commandRegistry: { [commandPalette] in commandPalette.commandRegistry },
            assembleSkillArguments: { [skillPills] name, text in
                skillPills.assembleArguments(skillName: name, userText: text)
            },
            recordSkillInvocation: { [skillPills] name in
                skillPills.recordUserInvocation(skillName: name)
            },
            clearComposerDraft: { [composerDraft] in composerDraft.clearDraft() },
            restoreComposerDraft: { [composerDraft] text, images in
                composerDraft.restore(text: text, images: images)
            },
            onConversationSwitch: { [composerDraft, agentSystemPromptInspector, skillPills] in
                composerDraft.resetEphemeral()
                agentSystemPromptInspector.reset()
                skillPills.refreshPills()
            }
        )
    }()

    // Appshot — the double-Command frontmost-window capture (PRD #170). Stages
    // through the Composer Draft; the app delegate attaches the window-summon
    // callback it owns.
    lazy var appshotController = AppshotController(
        capturer: ScreenCaptureKitAppshotCapturer(),
        composerDraft: composerDraft
    )

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
    // The pill follows the system appearance (owner-selected). The
    // `contentAppearance` seam on OverlayPanel remains the lever if a forced
    // light `.clear` glass is ever wanted — glass reads the AppKit
    // appearance, not the SwiftUI color scheme.
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
        manager.onTakeAppshot = { [appshotController] in
            Task { await appshotController.takeAppshot() }
        }
        // `AgentEngine.unloadModel` flushes pending SSD writes before the
        // teardown, so a plain offload never costs the disk tier its fresh
        // snapshots.
        manager.onOffloadModel = { [inferenceArbiter] in
            Task { await inferenceArbiter.offloadAllModels() }
        }
        manager.onClearMemoryCache = { [agentEngine] in
            agentEngine.llmActor.prefixCacheAdmin.clearRAMTier()
        }
        // Disk clear must not race the detached unload: a live ledger's
        // manifest persist after the wipe would resurrect the store.
        manager.onClearDiskCache = { [inferenceArbiter, agentEngine, settingsManager] in
            let root = settingsManager.ssdPrefixCacheRootURL
            Task {
                await inferenceArbiter.offloadAllModels()
                await agentEngine.awaitPendingUnload()
                SSDSnapshotStore.wipeArtifacts(at: root)
            }
        }
        manager.serverStatus = { [httpServer, settingsManager] in
            (
                isRunning: httpServer.isRunning,
                port: Int(HTTPServer.clampedPort(settingsManager.serverPort))
            )
        }
        manager.isModelLoaded = { [inferenceArbiter] in
            !inferenceArbiter.loadedSlots.isEmpty
        }
        manager.residentCacheBytes = { [agentEngine] in
            agentEngine.llmActor.prefixCacheAdmin.residentRAMBytes
        }
        manager.diskCacheBytes = { [settingsManager] in
            SSDSnapshotStore.artifactBytes(at: settingsManager.ssdPrefixCacheRootURL)
        }
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
            settings: settingsManager,
            captureDump: captureDumpStore
        )
    }()

    private var hasSetup = false

    init() {}

    func prepareForTermination() async {
        await terminationCoordinator.prepareForTermination()
        // Close Agent Browser windows/sessions on the way out.
        mcpBrowserServer.closeAllSessions()
    }

    private func makeTerminationCoordinator() -> AppTerminationCoordinator {
        AppTerminationCoordinator(
            steps: .init(
                stopHotkeys: { [hotkeyManager] in
                    hotkeyManager.stopListening()
                },
                cancelForegroundGenerationAndWait: { [chatSession] in
                    await chatSession.cancelGenerationAndWait()
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
            onDown: { [weak self] in self?.agentVoiceInput.start() },
            onUp: { [weak self] in self?.agentVoiceInput.finishCapture() }
        )
        // Register Appshot hotkey (one-shot tap, no held state)
        hotkeyManager.registerHotkey(
            id: "appshot",
            combo: settingsManager.appshotHotkey,
            onDown: { [weak self] in
                Task { await self?.appshotController.takeAppshot() }
            }
        )

        // Register message codecs for the new persistence layer (Epic 2)
        await registerCoreMessageCodecs()

        hotkeyManager.startListening()

        // Register HTTP server routes before App Bindings starts the server.
        registerHTTPRoutes()

        // Hand off to App Bindings: the launch ordering and every runtime
        // subscription with a rule live (and are tested) there.
        appBindings.start()

        // Wire the MCP client (PRD #190). Materialize the agent first so its
        // MCP tools extension is registered with the ExtensionHost before the
        // manager refreshes the registry; then connect the configured servers
        // (the built-in Browser server plus any user-added ones) and keep them
        // reconciled with settings.
        _ = agent
        mcpClientManager.start()
    }

    private func makeAppBindings() -> AppBindings {
        AppBindings(
            settings: settingsManager,
            inputs: .init(
                dictationState: { [dictationCoordinator] in
                    dictationCoordinator.state
                },
                speechState: { [speechCoordinator] in
                    speechCoordinator.state
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
                whisperModelPath: { [modelDownloadManager, settingsManager] in
                    guard
                        let path = modelDownloadManager.modelPath(
                            for: settingsManager.selectedSpeechToTextModelID),
                        WhisperModelContract.isComplete(at: path)
                    else { return nil }
                    return path
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
                pushSpeechStateToMenuBar: { [menuBarManager] in
                    menuBarManager.updateState(fromSpeech: $0)
                },
                pushAudioLevelToPill: { [pillOverlay] in
                    pillOverlay.handleAudioLevelChange($0)
                },
                pushAudioLevelToBorder: { [borderOverlay] in
                    borderOverlay.handleAudioLevelChange($0)
                },
                prewarmAudioCapture: { [audioCaptureEngine] in
                    audioCaptureEngine.prewarm()
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
                updateAppshotHotkey: { [hotkeyManager] in
                    hotkeyManager.updateRegisteredHotkey(id: "appshot", combo: $0)
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
                // because `CompletionHandler` validates `request.model` before
                // entering the lease queue; advertising an undownloaded id would
                // promise a model that immediately returns `model_not_found`.
                let loadedID: String? = engine.isModelLoaded ? arbiter.loadedLLMModelID : nil
                let models: [OpenAI.ModelObject] =
                    downloads
                    .downloadedModels(in: .agent)
                    .map { definition -> OpenAI.ModelObject in
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

        // OpenCode Integration (PRD #74): the Setup One-liner fetches the
        // script, which POSTs the user's existing config to the Config Merge.
        // Snapshots are taken per request so re-runs reflect live state.
        let settings = settingsManager
        httpServer.route(.GET, IntegrationRoutes.openCodeSetupScript) { _, writer in
            let response: HTTPResponse = await MainActor.run {
                OpenCodeIntegrationEndpoint.setupScriptResponse(
                    snapshot: IntegrationSnapshotBuilder.current(
                        downloads: downloads,
                        settings: settings
                    )
                )
            }
            try await writer.send(response)
        }
        httpServer.route(.POST, IntegrationRoutes.openCodeMerge) { request, writer in
            let response: HTTPResponse = await MainActor.run {
                OpenCodeIntegrationEndpoint.mergeResponse(
                    existingConfig: request.body,
                    snapshot: IntegrationSnapshotBuilder.current(
                        downloads: downloads,
                        settings: settings
                    )
                )
            }
            try await writer.send(response)
        }

        // Browser MCP Server (PRD #189): the `/mcp` endpoint. Registered
        // unconditionally; the handler refuses (503) when the setting is off.
        mcpBrowserServer.attach(to: httpServer)
    }
}
