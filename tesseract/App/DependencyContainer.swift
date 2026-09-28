//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
import SwiftUI
import Carbon.HIToolbox
import MLX
import TesseractSpeech
import os

@MainActor
final class DependencyContainer: ObservableObject {
    // Core Services
    /// Under a test runner the settings live in memory, so a test run never
    /// reads or changes the owner's (ADR-0073).
    let settingsManager = SettingsManager(
        store: ProcessEnvironment.isRunningTests
            ? InMemorySettingsStore() : UserDefaultsSettingsStore())
    lazy var permissionsManager = PermissionsManager()
    lazy var audioDeviceManager = AudioDeviceManager()

    // Audio
    lazy var audioCaptureEngine = AudioCaptureEngine()

    /// The **Capture Dump** (PRD #175) — one shared ring buffer for every
    /// capture surface, so bounds are enforced across the whole app.
    lazy var captureDumpStore: CaptureDumpStore = {
        CaptureDumpStore(
            directory:
                StorageEnvironment.applicationSupport
                .appendingPathComponent("Tesseract Agent", isDirectory: true)
                .appendingPathComponent("CaptureDump", isDirectory: true),
            protectedFileNames: { [weak self] in
                // Gold Correction Pairs keep their audio (ticket #289).
                self?.correctionPairStore.protectedAudioFileNames ?? []
            }
        )
    }()

    // The Correction Pair flywheel (ticket #289): every dictation take is
    // recorded as a training-pair candidate; overlay flags and history edits
    // turn candidates gold.
    lazy var correctionPairStore = CorrectionPairStore()

    // Transcription
    lazy var transcriptionEngine = TranscriptionEngine()
    lazy var transcriptionHistory = TranscriptionHistory()

    // Proofread Pass (ADR-0034): the second, small co-resident MLX model that
    // polishes transcriptions. The pass is pure policy over injected closures;
    // the model actor never touches the arbiter — skip-when-busy reads
    // whether the LLM is generating, and never waits for it.
    lazy var proofreadModel = ProofreadModel()
    // Built in a method, not inline: a lazy initializer is checked as a
    // default-argument context, which must have a *single* isolation — and
    // this wiring necessarily references both the main actor and the
    // ProofreadModel actor.
    lazy var proofreadPass: ProofreadPass = makeProofreadPass()

    private func makeProofreadPass() -> ProofreadPass {
        let model = proofreadModel
        let settings = settingsManager
        let arbiter = inferenceArbiter
        let downloads = modelDownloadManager
        return ProofreadPass(
            isEnabled: { settings.proofreadDictation },
            isLLMBusy: { arbiter.isLLMBusy },
            modelDirectory: {
                downloads.isDownloaded(ModelDefinition.defaultProofreadModelID)
                    ? downloads.modelPath(for: ModelDefinition.defaultProofreadModelID)
                    : nil
            },
            loadModel: { try await model.load(from: $0) },
            runModel: { try await model.run(system: $0, text: $1) },
            unloadModel: { await model.unload() }
        )
    }

    /// The only thing in the app that watches for the *absence* of a person:
    /// presence for the Companion's moments.
    lazy var idleMonitor = IdleMonitor()

    /// The internal completion path — the agent's own model, streamed through
    /// the shared inference service and folded to plain text. Compaction runs
    /// on it.
    lazy var internalCompletion: @Sendable (String) async throws -> String =
        makeSummarizeClosure(
            inferenceService: serverInferenceService,
            parametersProvider: { [settingsManager] in
                settingsManager.makeAgentGenerateParameters()
            }
        )

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
        let registry = ToolRegistry(sandbox: agentSandbox, extensionHost: extensionHost)
        // The agenda tools ride every conversation, Companion on or off:
        // "remind me" always becomes a real reminder in Reminders.
        for tool in createAgendaTools(agenda: agenda) {
            registry.appendBuiltInTool(tool)
        }
        // "Never tell me about CI passing": the owner's notification rules.
        registry.appendBuiltInTool(createNotificationRuleTool(store: triageRuleStore))
        // Memory the owner controls: remember, forget, recall — pull only.
        for tool in createProfileTools(
            profile: profileStore, index: recallIndex, embed: recallEmbedding)
        {
            registry.appendBuiltInTool(tool)
        }
        return registry
    }()

    // MARK: - The Companion's day

    /// Apple Reminders and Calendar, the single source of truth for the
    /// owner's tasks and plans. The test host gets the in-memory store, so a
    /// test run never asks for or touches the owner's real data (ADR-0073).
    lazy var agendaStore: any AgendaStore =
        ProcessEnvironment.isRunningTests ? InMemoryAgendaStore() : EventKitAgendaStore()
    lazy var agenda = Agenda(
        store: agendaStore,
        areaMapJSON: { [settingsManager] in settingsManager.companionAreasJSON },
        defaultCalendarID: { [settingsManager] in settingsManager.companionDefaultCalendarID },
        trace: companionTrace)
    lazy var captureService = CaptureService(agenda: agenda)
    /// The capture panel's own push-to-talk, separate from the composer's.
    lazy var captureVoiceInput = AgentVoiceInputController(
        audioCapture: audioCaptureEngine,
        transcriptionEngine: transcriptionEngine,
        settings: settingsManager,
        proofreadPass: proofreadPass,
        captureDump: captureDumpStore
    )
    lazy var capturePanel = CapturePanelController(
        capture: captureService, voice: captureVoiceInput)
    lazy var companionNotifier: CompanionNotifier = {
        let notifier = CompanionNotifier()
        notifier.onOpen = { (NSApp.delegate as? AppDelegate)?.navigateToToday() }
        return notifier
    }()
    /// The Day Thread's own agent: the same system prompt and tools as every
    /// chat (one cached prefix), compacting only past the thread's ceiling.
    lazy var dayAgent: Agent = AgentFactory.makeAgent(
        inferenceService: serverInferenceService,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        settingsManager: settingsManager,
        gating: ToolGating(webAccessEnabled: settingsManager.webAccessEnabled),
        mcpToolsExtension: mcpClientManager.toolsExtension,
        compactionWindow: DayThread.compactionWindow(
            ceiling: settingsManager.companionThreadCeilingTokens)
    )
    lazy var dayThread = DayThread(
        agent: dayAgent,
        store: DayThreadStore(backing: agentConversationStore, day: DayKey(for: Date())),
        arbiter: inferenceArbiter,
        inferenceService: serverInferenceService,
        toolRegistry: newToolRegistry,
        settings: settingsManager,
        speechCoordinator: speechCoordinator,
        contextManager: contextManager,
        summarize: internalCompletion,
        trace: companionTrace)
    lazy var triageRuleStore = TriageRuleStore(settings: settingsManager)
    /// The owner's Profile: facts they approved, and Jarvis's proposals.
    lazy var profileStore = ProfileStore(
        url: ProcessEnvironment.isRunningTests ? nil : ProfileStore.productionURL,
        trace: companionTrace)
    /// Full-text search over saved conversations, for `recall`.
    lazy var recallIndex = RecallIndex(
        conversationsDirectory: StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent/agent/conversations", isDirectory: true),
        databaseURL: StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent/companion/recall.sqlite"))
    /// The embedder that reranks recall by meaning, loaded on first use when
    /// its model is downloaded.
    lazy var memoryEmbedder = MemoryEmbedder()
    /// Whether the recall embedder holds its model, for the Models page.
    @Published private(set) var isEmbedderLoaded = false
    private var recallEmbedding: RecallEmbedding {
        let embedder = memoryEmbedder
        let downloads = modelDownloadManager
        let id = ModelDefinition.defaultEmbeddingModelID
        return { [weak self] texts in
            let directory: URL? = await MainActor.run {
                downloads.isDownloaded(id) ? downloads.modelPath(for: id) : nil
            }
            guard let directory else { return nil }
            do { try await embedder.load(from: directory) } catch { return nil }
            await MainActor.run { self?.isEmbedderLoaded = true }
            let vectors = await embedder.embed(texts)
            return vectors.count == texts.count ? vectors : nil
        }
    }
    lazy var frontmostApp = FrontmostAppTracker()
    lazy var powerMonitor = PowerMonitor()
    /// The Jarvis panel's own push-to-talk.
    lazy var panelVoiceInput = AgentVoiceInputController(
        audioCapture: audioCaptureEngine,
        transcriptionEngine: transcriptionEngine,
        settings: settingsManager,
        proofreadPass: proofreadPass,
        captureDump: captureDumpStore
    )
    /// The floating card rung: the Breakpoint card in a Siri-style glass panel.
    lazy var jarvisPanel: JarvisPanelController = JarvisPanelController(
        thread: dayThread, voice: panelVoiceInput,
        onAction: { [weak self] action in self?.companionRuntime.act(action) },
        onExpand: { (NSApp.delegate as? AppDelegate)?.navigateToToday() },
        onCapture: { [weak self] text in
            Task { await self?.captureService.capture(text, source: "panel") }
        })
    lazy var companionRuntime: CompanionRuntime = CompanionRuntime(
        settings: settingsManager, agenda: agenda, notifier: companionNotifier,
        trace: companionTrace, idleMonitor: idleMonitor, presence: companionPresence,
        thread: dayThread,
        stateStore: ProcessEnvironment.isRunningTests ? DayStateStore(url: nil) : .production,
        frontmost: frontmostApp, power: powerMonitor,
        delivery: CompanionDelivery(
            showPanel: { [weak self] card in self?.jarvisPanel.show(card) },
            retractPanel: { [weak self] cardID in self?.jarvisPanel.retract(cardID: cardID) },
            closePanel: { [weak self] in self?.jarvisPanel.close() },
            speak: { [weak self] line in self?.speechCoordinator.speakText(line) },
            openApp: { name in AppOpener.open(named: name) }),
        profile: profileStore)

    /// The Companion Trace: every Jarvis decision, card, reaction and agenda
    /// change, one JSONL file per day under Application Support.
    lazy var companionTrace = CompanionTrace()
    lazy var agentConversationStore = AgentConversationStore()
    lazy var inferenceArbiter: InferenceArbiter = {
        // TTS residency arrives as closures (evaluated lazily), so the
        // arbiter never holds the speech engine.
        InferenceArbiter(
            agentEngine: agentEngine,
            settingsManager: settingsManager,
            modelDownloadManager: modelDownloadManager,
            isTTSLoaded: { [weak self] in self?.speechEnginePresenter.isModelLoaded ?? false },
            unloadTTS: { [weak self] in await self?.speechEnginePresenter.unload() }
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
        gating: ToolGating(webAccessEnabled: settingsManager.webAccessEnabled),
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
        proofreadPass: proofreadPass,
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
            summarize: internalCompletion,
            commandRegistry: { [commandPalette] in commandPalette.commandRegistry },
            skillExecution: SkillExecution(
                assembleArguments: { [skillPills] name, text in
                    skillPills.assembleArguments(skillName: name, userText: text)
                },
                recordInvocation: { [skillPills] name in
                    skillPills.recordUserInvocation(skillName: name)
                }
            ),
            clearComposerDraft: { [composerDraft] in composerDraft.clearDraft() },
            restoreComposerDraft: { [composerDraft] text, images in
                composerDraft.restore(text: text, images: images)
            },
            onConversationSwitch: {
                [composerDraft, agentSystemPromptInspector, skillPills] in
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

    /// Jarvis's ambient presence on the menu-bar glyph.
    lazy var companionPresence = CompanionPresence()

    // PROTOTYPE — the Companion voice-overlay concepts (map #301, ticket
    // #328): scripted demo scenes on throwaway overlay surfaces, driven from
    // Settings. Since #310 also the live voice session's surface.
    lazy var companionVoicePrototype = CompanionVoicePrototype(
        settings: settingsManager,
        openChat: { (NSApp.delegate as? AppDelegate)?.navigateToAgent() }
    )

    // The voice session (#310): voice as a mode of the one conversation —
    // binds the #328 overlay, the speech engine, and the auto-listen loop to
    // the interactive chat. Spoken and typed turns are the same persisted
    // message stream. Half-duplex (ADR-0082): the mic is closed while he
    // speaks; a key press or a click interrupts him.
    lazy var companionVoiceSession: CompanionVoiceSessionController = {
        let controller = CompanionVoiceSessionController(
            capture: VoiceCaptureSession(
                audioCapture: audioCaptureEngine,
                transcriptionEngine: transcriptionEngine,
                captureDump: captureDumpStore,
                isCaptureDumpEnabled: { [settingsManager] in
                    settingsManager.captureDumpEnabled
                }
            ),
            meterLevel: { [dictationFeed] in dictationFeed.level },
            meterSpectrum: { [dictationFeed] in dictationFeed.spectrum },
            inputDead: { [audioCaptureEngine] in audioCaptureEngine.isInputDead },
            sendMessage: { [weak self] text in
                self?.chatSession.sendMessage(text, bypassCommandParsing: true)
            },
            stageToComposer: { [composerDraft] text in
                composerDraft.restore(text: text, images: [])
            },
            speak: { [weak self] text, onDone in
                self?.speechCoordinator.speakText(text, showsOverlay: false, onSuccess: onDone)
            },
            stopSpeaking: { [weak self] in self?.speechCoordinator.stop() },
            speechState: { [weak self] in self?.speechCoordinator.state ?? .idle },
            currentConversationID: { [weak self] in
                self?.agentConversationStore.currentConversation?.id
            },
            overlay: companionVoicePrototype,
            recorder: companionTrace,
            settings: settingsManager,
            proofreadPass: proofreadPass
        )
        // The reply hook: while a session is live it owns the spoken reply
        // and the auto-listen loop; autoSpeak stays the chat-only path.
        chatSession.voiceReplyHandler = { [weak controller] text in
            controller?.replyCompleted(text) ?? false
        }
        return controller
    }()

    // Speech (TTS) — engine v2 (ADR-0038/0039): the engine actor lives in the
    // TesseractSpeech package behind its ports; the presenter mirrors
    // residency for views and the arbiter; the coordinator drives sessions.
    lazy var textExtractor = TextExtractor()
    lazy var speechEnginePresenter: SpeechEnginePresenter = {
        // The catalog's Voice Engine entry downloads this same spec (the
        // ADR-0037 precision gate is recorded there), and the engine loads it
        // from that entry's folder: the store root plus the repo's
        // subdirectory, as `modelPath(for:)` builds it. The engine never
        // downloads; without the files it throws `modelUnavailable`.
        let store = ModelDownloadManager.modelStorageURL
        return SpeechEnginePresenter(
            engine: SpeechEngine(
                model: ModelDefinition.textToSpeechModelSpec,
                // The codec's Neural Engine model is built from the
                // checkpoint once and kept in Caches (ADR-0075).
                synthesizer: Qwen3Synthesizer(
                    checkpointDirectory: { spec in
                        store.appendingPathComponent(
                            ModelDefinition.storageSubdirectory(forRepo: spec.repo))
                    },
                    neuralEngineCache: Qwen3Synthesizer.neuralEngineCache(
                        in: StorageEnvironment.caches))
            )
        )
    }()
    /// The **Read-Along** (ADR-0076): the one clock of which word is being
    /// heard. It is the coordinator's Word Highlight Surface; the Reader and
    /// the Speech Overlay follow it.
    lazy var speechReadAlong = SpeechReadAlong()
    /// Each designed voice's Reference Take (ADR-0072), shared by the
    /// coordinator, which speaks with it, and the voice library, which lists
    /// and forgets voices.
    lazy var pinnedVoiceStore = PinnedVoiceStore()
    lazy var speechReader = SpeechReader(
        coordinator: speechCoordinator, readAlong: speechReadAlong, settings: settingsManager)
    lazy var voiceLibrary = VoiceLibrary(settings: settingsManager, pinnedVoices: pinnedVoiceStore)
    lazy var speechOverlayPanel = SpeechOverlayPanel(
        readAlong: speechReadAlong, settings: settingsManager, coordinator: speechCoordinator,
        isSpeechPageInFront: { [unowned self] in self.speechReader.isInFront },
        openSpeechPage: { (NSApp.delegate as? AppDelegate)?.navigateToSpeech() })
    lazy var speechCoordinator: SpeechCoordinator = {
        // `playback` is left to the coordinator's production default
        // (`AudioPlaybackManager()`) — the AVFoundation adapter is needed by
        // nothing else in the graph, so there is no shared handle to wire here.
        // Tests inject `InMemoryAudioPlayback`.
        let coordinator = SpeechCoordinator(
            textExtractor: textExtractor,
            engine: speechEnginePresenter,
            voiceEngineStatus: { [modelDownloadManager] in
                // Re-read from disk: the folder can change behind the
                // catalog's back (deleted in Finder, fetched by v2-listen).
                let id = ModelDefinition.defaultTextToSpeechModelID
                modelDownloadManager.refreshStatus(for: id)
                return modelDownloadManager.status(for: id)
            },
            settings: settingsManager,
            notchOverlay: speechReadAlong,
            pinnedVoices: pinnedVoiceStore
        )
        coordinator.onVoiceEngineMissing = {
            (NSApp.delegate as? AppDelegate)?.navigateToModels()
        }
        return coordinator
    }()

    // Overlay — the Overlay Feed every variant renders from, and the dumb
    // panel that hosts whichever Overlay Variant the setting selects (the
    // App Bindings variant rule installs the view; the panel itself is
    // contentless). The pill follows the system appearance (owner-selected);
    // the `contentAppearance` seam on OverlayPanel remains the lever if a
    // forced light `.clear` glass is ever wanted — glass reads the AppKit
    // appearance, not the SwiftUI color scheme.
    lazy var dictationFeed = DictationFeed()
    lazy var pillOverlay = OverlayPanel(placement: .pill)

    // Menu bar — constructed here so App Bindings can wire its dictation-state
    // effect before the app delegate attaches the window-management callbacks.
    lazy var menuBarManager: MenuBarManager = {
        let manager = MenuBarManager(settings: settingsManager)
        manager.coordinator = dictationCoordinator
        manager.history = transcriptionHistory
        manager.speechCoordinator = speechCoordinator
        // Jarvis's presence on the quietest rung (#327 §3).
        companionPresence.onChange = { [weak manager] state in
            manager?.updateState(fromCompanion: state)
        }
        manager.onTakeAppshot = { [appshotController] in
            Task { await appshotController.takeAppshot() }
        }
        // `AgentEngine.unloadModel` flushes pending SSD writes before the
        // teardown, so a plain offload never costs the disk tier its fresh
        // snapshots.
        manager.onOffloadModel = { [inferenceArbiter, proofreadPass] in
            Task {
                await inferenceArbiter.offloadAllModels()
                // The proofread model lives outside the arbiter's slots —
                // "Offload Model" frees it explicitly.
                await proofreadPass.unload()
            }
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
        manager.modelActivity = { [weak self] in self?.modelActivity() ?? .empty }
        return manager
    }()

    // App Bindings owns the launch sequence and every runtime subscription
    // with a rule; this container only wires its inputs and effects.
    lazy var appBindings = makeAppBindings()

    // Coordinator
    lazy var dictationCoordinator: DictationCoordinator = {
        let coordinator = DictationCoordinator(
            audioCapture: audioCaptureEngine,
            transcriptionEngine: transcriptionEngine,
            textInjector: textInjector,
            history: transcriptionHistory,
            settings: settingsManager,
            feed: dictationFeed,
            proofreadPass: proofreadPass,
            captureDump: captureDumpStore,
            pairs: correctionPairStore
        )
        // The Live Partial pump (ticket #291) runs only while the selected
        // variant consumes the signal — the coordinator reads a policy
        // closure; which variant is live never crosses into the pipeline.
        coordinator.isLivePartialsEnabled = { [weak self] in
            guard let self else { return false }
            return OverlayVariants.variant(for: self.settingsManager.overlayVariantRaw)
                .usesLivePartials
        }
        // PROTOTYPE (Dictation page redesign, never merge): in a development
        // build the dictation lab's learned replacements take the proofread
        // slot; the old pass runs only when the lab's switch asks for it.
        if PrototypeGate.isDevelopmentBuild {
            let lab = DictationLab.shared
            coordinator.labRefine = { lab.refine($0) }
            coordinator.labUsesProofreadPass = { lab.usesProofreadPass }
        }
        return coordinator
    }()

    /// The variant-agnostic overlay action surface (ticket #289): variants
    /// render the feed and call these — they never see the coordinator.
    lazy var overlayActions = OverlayActions(
        flagLastTakeWrong: { [weak self] in self?.dictationCoordinator.flagLastTakeWrong() },
        editLastTake: { [weak self] in self?.dictationCoordinator.editLastTake() },
        insertRawAnyway: { [weak self] in self?.dictationCoordinator.insertRawAnyway() }
    )

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
                unloadLLM: { [agentEngine] in
                    agentEngine.unloadModel()
                },
                awaitLLMUnload: { [agentEngine] in
                    await agentEngine.awaitPendingUnload()
                },
                unloadSpeech: { [speechEnginePresenter] in
                    await speechEnginePresenter.unload()
                },
                synchronizeGPU: {
                    Stream.gpu.synchronize()
                }
            ))
    }

    func setup() async {
        // The app is also the unit-test host. Never start model prewarms,
        // inference reloads, or the owner's background services from tests.
        guard !ProcessEnvironment.isRunningTests else { return }
        // The SSD read experiment owns its single target model and scratch tier.
        guard !CommandLine.arguments.contains("--ssd-read-bench") else { return }
        // The TurboQuant bench reads the process-wide MLX memory counters; the
        // Companion and the prewarms must not load a model beside it.
        guard !CommandLine.arguments.contains("--turboquant-bench") else { return }
        // Prevent duplicate setup from multiple window instances
        guard !hasSetup else { return }
        hasSetup = true
        // The launch steps and their cross-step ordering invariants are declared,
        // validated, and executed as a BootstrapSequence — the runner checks the
        // declared order against the invariants (routes-before-bindings,
        // perception-callbacks-before-start, agent-before-MCP) before running any
        // step. See `bootstrapSequence()` / `BootstrapStep` below.
        await bootstrapSequence().run()
    }

    // MARK: - Bootstrap declaration

    /// The production launch steps, in launch order. This enum is the single
    /// source of step identity: `bootstrapSequence()` builds exactly one closure
    /// per case through an exhaustive switch, so the executed order is
    /// `allCases` order and no step can be declared without a closure — the
    /// declaration (names + invariants, below) and the execution cannot diverge.
    nonisolated enum BootstrapStep: String, CaseIterable {
        case registerHotkeys
        case attachDictationMeters
        case registerMessageCodecs
        case startHotkeyListening
        case registerHTTPRoutes
        case startAppBindings
        case startCompanion
        case materializeAgent
        case startMCPClient
    }

    /// Declared step names in launch order — reachable without a container so the
    /// declaration is testable (the container isn't constructible in tests).
    nonisolated static var bootstrapStepNames: [String] {
        BootstrapStep.allCases.map(\.rawValue)
    }

    /// The order-critical cross-step invariants, each with the reason a reorder
    /// would break. Reachable without a container, like the step names.
    nonisolated static var bootstrapInvariants: [BootstrapSequence.Invariant] {
        [
            .init(
                before: BootstrapStep.registerHTTPRoutes.rawValue,
                after: BootstrapStep.startAppBindings.rawValue,
                why: "App Bindings starts the HTTP server; its routes must be "
                    + "registered before the server that serves them starts."
            ),
            .init(
                before: BootstrapStep.materializeAgent.rawValue,
                after: BootstrapStep.startMCPClient.rawValue,
                why: "Materializing the agent registers its MCP tools extension "
                    + "with the ExtensionHost before the MCP manager refreshes "
                    + "the registry (ADR-0048 late-connect hazard)."
            ),
        ]
    }

    /// Build the production sequence. Step order is `BootstrapStep.allCases`;
    /// each case maps to exactly one closure via the exhaustive switch in
    /// `bootstrapAction(for:)`, so declaration and execution stay in lockstep.
    func bootstrapSequence() -> BootstrapSequence {
        BootstrapSequence(
            steps: BootstrapStep.allCases.map { step in
                BootstrapSequence.Step(name: step.rawValue, run: bootstrapAction(for: step))
            },
            invariants: Self.bootstrapInvariants
        )
    }

    private func bootstrapAction(for step: BootstrapStep) -> @MainActor () async -> Void {
        switch step {
        case .registerHotkeys:
            return { [self] in
                // Register dictation push-to-talk — through the same registration
                // API as every other hotkey (audit #285 item 7).
                hotkeyManager.registerHotkey(
                    id: HotkeyManager.dictationHotkeyID,
                    combo: settingsManager.hotkey,
                    onDown: { [weak self] in self?.dictationCoordinator.onHotkeyDown() },
                    onUp: { [weak self] in self?.dictationCoordinator.onHotkeyUp() }
                )
                // Register TTS hotkey. In a voice session it and the Agent
                // hotkey are the interrupt key: Jarvis stops and listens
                // (ADR-0082).
                hotkeyManager.registerHotkey(
                    id: "tts",
                    combo: settingsManager.ttsHotkey,
                    onDown: { [weak self] in
                        guard let self else { return }
                        if companionVoiceSession.isActive {
                            companionVoiceSession.bargeIn(source: "key")
                        } else {
                            speechCoordinator.onHotkeyPressed()
                        }
                    }
                )
                // Register Agent hotkey
                hotkeyManager.registerHotkey(
                    id: "agent",
                    combo: settingsManager.agentHotkey,
                    onDown: { [weak self] in
                        guard let self else { return }
                        if companionVoiceSession.isActive {
                            companionVoiceSession.bargeIn(source: "key")
                        } else {
                            agentVoiceInput.start()
                        }
                    },
                    onUp: { [weak self] in self?.agentVoiceInput.finishCapture() }
                )
                // Register the capture hotkey: tap to type, hold to speak.
                // As one key (Right ⌥ by default), another key pressed with
                // it cancels: the owner is typing.
                hotkeyManager.registerHotkey(
                    id: "capture",
                    combo: settingsManager.captureHotkey,
                    onDown: { [weak self] in self?.capturePanel.hotkeyDown() },
                    onUp: { [weak self] in self?.capturePanel.hotkeyUp() },
                    onCancel: { [weak self] in self?.capturePanel.hotkeyCancelled() }
                )
                // PROTOTYPE (Dictation page redesign, never merge): the
                // dictation lab and its fix shortcut, ⌃⌥Space, in development
                // builds only.
                if PrototypeGate.isDevelopmentBuild {
                    let lab = DictationLab.shared
                    lab.attach(
                        feed: dictationFeed, coordinator: dictationCoordinator,
                        injector: textInjector, settings: settingsManager,
                        keyDownCount: { [hotkeyManager] in hotkeyManager.keyDownCount },
                        voice: LabVoiceFix(
                            audioCapture: audioCaptureEngine,
                            transcriptionEngine: transcriptionEngine, settings: settingsManager))
                    hotkeyManager.registerHotkey(
                        id: "dictationLabFix",
                        combo: KeyCombo(
                            keyCode: UInt16(kVK_Space), modifiers: [.control, .option]),
                        onDown: { lab.shortcutDown() },
                        onUp: { lab.shortcutUp() }
                    )
                }
                // Register Appshot hotkey (one-shot tap, no held state)
                hotkeyManager.registerHotkey(
                    id: "appshot",
                    combo: settingsManager.appshotHotkey,
                    onDown: { [weak self] in
                        Task { await self?.appshotController.takeAppshot() }
                    }
                )
            }
        case .attachDictationMeters:
            return { [self] in
                // Pump the capture engine's meter frames into the Overlay Feed —
                // the one attachment point (the feed owns the consuming task).
                dictationFeed.attachMeters(audioCaptureEngine.meters)
            }
        case .registerMessageCodecs:
            return {
                // Register message codecs for the new persistence layer (Epic 2)
                await registerCoreMessageCodecs()
            }
        case .startHotkeyListening:
            return { [self] in hotkeyManager.startListening() }
        case .registerHTTPRoutes:
            return { [self] in registerHTTPRoutes() }
        case .startAppBindings:
            return { [self] in
                // Hand off to App Bindings: the launch ordering and every runtime
                // subscription with a rule live (and are tested) there.
                appBindings.start()
            }
        case .startCompanion:
            return { [self] in
                // The Companion's loop follows its switch from here on.
                companionRuntime.start()
            }
        case .materializeAgent:
            return { [self] in
                // Wire the MCP client (PRD #190). Materialize the agent first so its
                // MCP tools extension is registered with the ExtensionHost before the
                // manager refreshes the registry.
                _ = agent
            }
        case .startMCPClient:
            return { [self] in
                // Connect the configured servers (the built-in Browser server plus
                // any user-added ones) and keep them reconciled with settings.
                mcpClientManager.start()
            }
        }
    }

    private func makeAppBindings() -> AppBindings {
        AppBindings(
            settings: settingsManager,
            inputs: .init(
                dictationState: { [dictationFeed] in
                    dictationFeed.phase
                },
                dictationBeat: { [dictationFeed] in
                    dictationFeed.beat
                },
                speechState: { [speechCoordinator] in
                    speechCoordinator.state
                },
                currentDictationHotkey: { [hotkeyManager] in
                    hotkeyManager.currentDictationHotkey
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
                setUpOverlayPanel: { [pillOverlay] in
                    pillOverlay.setup()
                },
                setOverlayVariant: { [pillOverlay, dictationFeed, overlayActions] variantID in
                    let variant = OverlayVariants.variant(for: variantID)
                    pillOverlay.setPlacement(variant.placement)
                    pillOverlay.setContent(variant.makeView(dictationFeed, overlayActions))
                },
                reassertOverlayFront: { [pillOverlay] in
                    pillOverlay.reassertFront()
                },
                setOverlayInteractive: { [pillOverlay] in
                    pillOverlay.setInteractive($0)
                },
                pushDictationStateToMenuBar: { [menuBarManager] in
                    menuBarManager.updateState(from: $0)
                },
                pushSpeechStateToMenuBar: { [menuBarManager] in
                    menuBarManager.updateState(fromSpeech: $0)
                },
                prewarmAudioCapture: { [audioCaptureEngine] in
                    audioCaptureEngine.prewarm()
                },
                prewarmProofreader: { [proofreadPass] in
                    await proofreadPass.prewarm()
                },
                updateDictationHotkey: { [hotkeyManager] in
                    hotkeyManager.updateRegisteredHotkey(
                        id: HotkeyManager.dictationHotkeyID, combo: $0)
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
                updateCaptureHotkey: { [hotkeyManager] in
                    hotkeyManager.updateRegisteredHotkey(id: "capture", combo: $0)
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
                },
                startSpeechSurfaces: { [unowned self] in
                    self.speechReader.start()
                    self.speechOverlayPanel.start()
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
                // queueing at the LLM gate; advertising an undownloaded id would
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

        // Coding agents (Claude Code hooks): localhost-only by the server's
        // bind; browsers are refused, since they send an Origin and curl
        // never does.
        let runtime = companionRuntime
        for (_, kind) in ClaudeCodeHooks.events {
            httpServer.route(.POST, ClaudeCodeHooks.routePrefix + kind.rawValue) {
                request, writer in
                let fromBrowser: Bool = await MainActor.run { request.header("Origin") != nil }
                guard !fromBrowser else {
                    try await writer.send(.error(status: 403, message: "Forbidden"))
                    return
                }
                let signal = AgentSignal.parse(hookBody: request.body, kind: kind, at: Date())
                await MainActor.run { runtime.receive(signal) }
                try await writer.send(.json(["ok": true]))
            }
        }
        httpServer.route(.GET, ClaudeCodeHooks.setupScriptPath) { _, writer in
            let port: Int = await MainActor.run { Int(HTTPServer.clampedPort(settings.serverPort)) }
            try await writer.send(
                HTTPResponse(
                    statusCode: 200, statusText: "OK",
                    headers: [("Content-Type", "text/x-shellscript; charset=utf-8")],
                    body: Data(ClaudeCodeHooks.setupScript(port: port).utf8)))
        }
        httpServer.route(.POST, ClaudeCodeHooks.mergePath) { request, writer in
            let port: Int = await MainActor.run { Int(HTTPServer.clampedPort(settings.serverPort)) }
            do {
                let merged = try ClaudeCodeHooks.merge(existing: request.body, port: port)
                try await writer.send(.jsonBody(merged))
            } catch {
                try await writer.send(
                    .error(
                        status: 422, message: "The settings file isn't valid JSON; left unchanged.")
                )
            }
        }

        // Browser MCP Server (PRD #189): the `/mcp` endpoint. Registered
        // unconditionally; the handler refuses (503) when the setting is off.
        mcpBrowserServer.attach(to: httpServer)
    }
}
