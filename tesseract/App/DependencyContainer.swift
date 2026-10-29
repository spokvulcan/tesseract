//
//  DependencyContainer.swift
//  tesseract
//

import Foundation
import Combine
import SwiftUI
import MLX
import TesseractSpeech
import os

@MainActor
final class DependencyContainer: ObservableObject {
    // Core Services
    /// Where the Launch Mode keeps off the owner's data (a test host, a
    /// scratch launch), the settings live in memory (ADR-0073).
    let settingsManager = SettingsManager(
        store: LaunchMode.current.keepsOffOwnersData
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
    // recorded as a training-pair candidate; a word fixed in the Lens or
    // "Insert anyway" on a rejected take turns it gold.
    lazy var correctionPairStore = CorrectionPairStore()

    // The Learned Words (PRD #612): "heard → meant" replacements the owner's
    // fixes in the Lens taught, applied by every Voice Capture Session.
    lazy var learnedWordStore = LearnedWordStore()

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
                downloads.usableDirectory(for: ModelDefinition.defaultProofreadModelID)
            },
            loadModel: { try await model.load(from: $0) },
            runModel: { try await model.run(system: $0, text: $1) },
            unloadModel: { await model.unload() }
        )
    }

    /// **Voice Capture** (ADR-0096): every push-to-talk surface's takes, with
    /// every stage — the shared mic and Whisper, the Capture Dump, the
    /// Learned Words, the Proofread Pass and the dictation language — and
    /// whether any take is open, for the Models section.
    lazy var voiceCapture: VoiceCapture = makeVoiceCapture()

    private func makeVoiceCapture() -> VoiceCapture {
        let settings = settingsManager
        let pass = proofreadPass
        return VoiceCapture(
            microphone: audioCaptureEngine,
            recognizer: transcriptionEngine,
            captureDump: captureDumpStore,
            isCaptureDumpEnabled: { settings.captureDumpEnabled },
            learnedWords: learnedWordStore,
            proofread: { await pass.proofread($0) },
            language: { settings.language })
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
    lazy var modelDownloadManager = ModelDownloadManager(fetching: HuggingFaceModelFetching())

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
    /// The room the agent chat compacts in: the selected model's window, read
    /// at every check so a model picked between turns compacts in its own.
    lazy var chatCompactionWindow: @MainActor @Sendable () -> CompactionWindow = {
        [settingsManager] in
        CompactionWindow(
            contextLength: ModelDefinition.contextLength(of: settingsManager.selectedAgentModelID))
    }
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
            profile: profileStore, index: recallIndex, embed: recallEmbedder.recallEmbedding)
        {
            registry.appendBuiltInTool(tool)
        }
        return registry
    }()

    // MARK: - The Companion's day

    /// Apple Reminders and Calendar, the single source of truth for the
    /// owner's tasks and plans. Where the Launch Mode keeps off the owner's
    /// data (a test host, a scratch launch), the in-memory store, so neither
    /// asks for or touches the owner's real data (ADR-0073).
    lazy var agendaStore: any AgendaStore =
        LaunchMode.current.keepsOffOwnersData ? InMemoryAgendaStore() : EventKitAgendaStore()
    lazy var agenda = Agenda(
        store: agendaStore,
        areaMapJSON: { [settingsManager] in settingsManager.companionAreasJSON },
        defaultCalendarID: { [settingsManager] in settingsManager.companionDefaultCalendarID },
        trace: companionTrace)
    lazy var captureService = CaptureService(agenda: agenda)
    /// The capture panel's own push-to-talk, separate from the composer's.
    lazy var captureVoiceInput = AgentVoiceInputController(session: voiceCapture.makeSession())
    lazy var capturePanel = CapturePanelController(
        capture: captureService, voice: captureVoiceInput)
    lazy var companionNotifier: CompanionNotifier = {
        let notifier = CompanionNotifier(usesOS: LaunchMode.current.postsNotifications)
        notifier.onOpen = { (NSApp.delegate as? AppDelegate)?.navigateToToday() }
        return notifier
    }()
    /// The Day Thread's own agent: the same system prompt and tools as every
    /// chat (one cached prefix), compacting only past the thread's ceiling,
    /// or sooner when the selected model's window is smaller. It sees a
    /// picture in the turn it's shown, a line in its place after (ADR-0090).
    lazy var dayAgent: Agent = AgentFactory.makeAgent(
        inferenceService: serverInferenceService,
        packageRegistry: packageRegistry,
        extensionHost: extensionHost,
        toolRegistry: newToolRegistry,
        contextManager: contextManager,
        settingsManager: settingsManager,
        gating: ToolGating(webAccessEnabled: settingsManager.webAccessEnabled),
        mcpToolsExtension: mcpClientManager.toolsExtension,
        compactionWindow: { [settingsManager] in
            DayThread.compactionWindow(settings: settingsManager)
        },
        convertToLlm: DayThreadPictures.llmMessages
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
        trace: companionTrace,
        restoreDraft: { [todayDraft] text, images in
            todayDraft.restore(text: text, images: images)
        })
    /// Today's composer draft, its own: the words and pictures not yet sent,
    /// and Quick Look over the Day Thread's pictures.
    lazy var todayDraft = ComposerDraftController(conversationImages: { [dayAgent] in
        dayAgent.state.messages.flatMap { $0.transcriptImages }
    })
    /// Whether Jarvis can see pictures: the selected model and the vision
    /// setting, as for the agent chat. The Jarvis panel asks it too.
    lazy var todayVisionAvailability = VisionAvailabilityController(
        settings: settingsManager,
        draft: todayDraft,
        isVisionCapable: { [modelDownloadManager] in modelDownloadManager.isVisionCapable($0) },
        downloadedAgentModels: { [modelDownloadManager] in
            modelDownloadManager.downloadedModels(in: .agent)
        }
    )
    lazy var triageRuleStore = TriageRuleStore(settings: settingsManager)
    /// The owner's Profile: facts they approved, and Jarvis's proposals.
    lazy var profileStore = ProfileStore(
        url: LaunchMode.current.keepsOffOwnersData ? nil : ProfileStore.productionURL,
        trace: companionTrace)
    /// Full-text search over saved conversations, for `recall`.
    lazy var recallIndex = RecallIndex(
        conversationsDirectory: StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent/agent/conversations", isDirectory: true),
        databaseURL: StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent/companion/recall.sqlite"))
    /// The embedder that reranks recall by meaning, loaded on first use when
    /// its model is downloaded.
    lazy var recallEmbedder = RecallEmbedder(
        embedder: MemoryEmbedder(),
        modelDirectory: { [modelDownloadManager] in
            modelDownloadManager.usableDirectory(for: ModelDefinition.defaultEmbeddingModelID)
        })
    lazy var frontmostApp = FrontmostAppTracker()
    lazy var powerMonitor = PowerMonitor()
    /// The Jarvis panel's own push-to-talk.
    lazy var panelVoiceInput = AgentVoiceInputController(session: voiceCapture.makeSession())
    /// Today's composer's own push-to-talk.
    lazy var todayVoiceInput = AgentVoiceInputController(session: voiceCapture.makeSession())
    /// The floating card rung: the Breakpoint card in a Siri-style glass panel.
    lazy var jarvisPanel: JarvisPanelController = JarvisPanelController(
        thread: dayThread, voice: panelVoiceInput,
        day: { [weak self] in self?.companionRuntime.timeline() },
        now: { [weak self] in self?.companionRuntime.now() ?? Date() },
        liveCard: { [weak self] id in self?.companionRuntime.state.cards.first { $0.id == id } },
        onAction: { [weak self] action in self?.companionRuntime.act(action) },
        onExpand: { (NSApp.delegate as? AppDelegate)?.navigateToToday() },
        capture: captureService,
        // The panel asks Today's verdict, fresh: it can come up before Today
        // was ever opened.
        pictureRemedy: { [todayVisionAvailability] in
            todayVisionAvailability.refresh()
            return todayVisionAvailability.imageInputAvailable
                ? nil : todayVisionAvailability.remedy.message
        })
    lazy var companionRuntime: CompanionRuntime = CompanionRuntime(
        settings: settingsManager, agenda: agenda, notifier: companionNotifier,
        trace: companionTrace, idleMonitor: idleMonitor, presence: companionPresence,
        thread: dayThread,
        stateStore: LaunchMode.current.keepsOffOwnersData ? DayStateStore(url: nil) : .production,
        frontmost: frontmostApp, power: powerMonitor,
        delivery: CompanionDelivery(
            showPanel: { [weak self] card in self?.jarvisPanel.show(card) },
            showStep: { [weak self] cue in
                self?.jarvisPanel.show(cue)
                // A step's start or end is a moment to notice: a soft chime,
                // like a timer, with the app's sounds on.
                if self?.settingsManager.playSounds == true { NSSound(named: "Glass")?.play() }
            },
            showBreak: { [weak self] cue in self?.jarvisPanel.show(cue) },
            retractBreak: { [weak self] in self?.jarvisPanel.retractBreak() },
            retractPanel: { [weak self] cardID in self?.jarvisPanel.retract(cardID: cardID) },
            closePanel: { [weak self] in self?.jarvisPanel.close() },
            isPanelUp: { [weak self] in self?.jarvisPanel.isShowing ?? false },
            speak: { [weak self] line in self?.speechCoordinator.speakText(line) },
            openApp: { name in AppOpener.open(named: name) }),
        profile: profileStore)

    /// The Companion Trace: every Jarvis decision, card, reaction and agenda
    /// change, one JSONL file per day under Application Support.
    lazy var companionTrace = CompanionTrace()
    lazy var agentConversationStore = AgentConversationStore()
    lazy var inferenceArbiter = InferenceArbiter(
        agentEngine: agentEngine,
        settingsManager: settingsManager,
        modelDownloadManager: modelDownloadManager
    )

    /// **Model Residency**: which models are in memory, what each is doing,
    /// and Offload Model — one row per model's holder
    /// (`DependencyContainer+ModelResidency.swift`).
    lazy var modelResidency: ModelResidency = makeModelResidency()
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
        mcpToolsExtension: mcpClientManager.toolsExtension,
        compactionWindow: chatCompactionWindow
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
    lazy var agentVoiceInput = AgentVoiceInputController(session: voiceCapture.makeSession())
    lazy var composerDraft = ComposerDraftController(conversationImages: { [agent] in
        agent.state.messages.flatMap { $0.transcriptImages }
    })
    lazy var visionAvailability = VisionAvailabilityController(
        settings: settingsManager,
        draft: composerDraft,
        isVisionCapable: { [modelDownloadManager] in modelDownloadManager.isVisionCapable($0) },
        downloadedAgentModels: { [modelDownloadManager] in
            modelDownloadManager.downloadedModels(in: .agent)
        }
    )
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
            compactionWindow: chatCompactionWindow,
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
            onConversationSwitch: { [composerDraft, skillPills] in
                composerDraft.resetEphemeral()
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
            capture: voiceCapture.makeSession(),
            meterLevel: { [dictationFeed] in dictationFeed.level },
            meterSpectrum: { [dictationFeed] in dictationFeed.spectrum },
            inputDead: { [audioCaptureEngine] in audioCaptureEngine.isInputDead },
            sendMessage: { [weak self] text in
                self?.chatSession.sendMessage(text, bypassCommandParsing: true)
            },
            stageToComposer: { [composerDraft] text in
                composerDraft.restore(text: text, images: [])
            },
            // The reply speaks as it is written (ADR-0097).
            speak: { [weak self] text, onDone in
                self?.speechCoordinator.startStreamedSpeech(text, onSuccess: onDone)
            },
            speakMore: { [weak self] text in
                self?.speechCoordinator.appendStreamedSpeech(text)
            },
            finishSpeaking: { [weak self] in self?.speechCoordinator.finishStreamedSpeech() },
            stopSpeaking: { [weak self] in self?.speechCoordinator.stop() },
            cancelReply: { [weak self] in self?.chatSession.cancelGeneration() },
            speechState: { [weak self] in self?.speechCoordinator.state ?? .idle },
            currentConversationID: { [weak self] in
                self?.agentConversationStore.currentConversation?.id
            },
            overlay: companionVoicePrototype,
            recorder: companionTrace,
            settings: settingsManager
        )
        // The reply hook: while a session is live it owns the spoken reply
        // and the auto-listen loop; autoSpeak stays the chat-only path.
        chatSession.voiceReplyHandler = { [weak controller] text in
            controller?.replyCompleted(text) ?? false
        }
        chatSession.voiceReplyTextHandler = { [weak controller] piece in
            controller?.replyText(piece)
        }
        builtVoiceSession = controller
        return controller
    }()
    /// The voice session once something built it: a hotkey or the Models
    /// section asks whether a session is active without building one (and the
    /// chat session with it).
    private(set) var builtVoiceSession: CompanionVoiceSessionController?

    // LFM3 Voice: a voice session in the chat where LFM3-Audio hears the
    // owner, answers with the agent's tools and speaks the answer as it
    // writes it. The loop and the overlay are the voice session's; the ear,
    // the mind and the voice are one LFM3-Audio load of its own.
    lazy var lfm3Voice: LFM3VoiceMode = {
        let mode = makeLFM3Voice()
        builtLFM3Voice = mode
        return mode
    }()
    /// LFM3 Voice once something built it, for the hotkeys.
    private(set) var builtLFM3Voice: LFM3VoiceMode?

    // Speech (TTS) — engine v2 (ADR-0038/0039): the engine actor lives in the
    // TesseractSpeech package behind its ports; the presenter mirrors
    // residency for views and Model Residency; the coordinator drives sessions.
    lazy var textExtractor = TextExtractor()
    lazy var speechEnginePresenter: SpeechEnginePresenter = {
        // The catalog's Voice Engine entry downloads this same spec (the
        // ADR-0037 precision gate is recorded there), and the engine loads it
        // from that entry's folder, by the one folder rule `modelPath(for:)`
        // applies (pure, so the engine's off-main closure applies it too).
        // The engine never downloads; without the files it throws
        // `modelUnavailable`.
        let store = ModelDownloadManager.modelStorageURL
        return SpeechEnginePresenter(
            engine: SpeechEngine(
                model: ModelDefinition.textToSpeechModelSpec,
                // The codec's Neural Engine model is built from the
                // checkpoint once and kept in Caches (ADR-0075).
                synthesizer: Qwen3Synthesizer(
                    checkpointDirectory: { spec in
                        ModelDefinition.directory(forRepo: spec.repo, in: store)
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

    // The Overlay Feed: the dictation signals the Lens renders from.
    lazy var dictationFeed = DictationFeed()

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
        // The started step's time left, or the time until what comes next,
        // beside the glyph.
        companionPresence.onClockChange = { [weak manager] clock in
            manager?.updateClock(clock)
        }
        manager.companionOn = { [settingsManager] in settingsManager.companionHeartbeatEnabled }
        manager.onCaptureThought = { [weak self] in self?.capturePanel.openForTyping() }
        manager.onTakeAppshot = { [appshotController] in
            Task { await appshotController.takeAppshot() }
        }
        // `AgentEngine.unloadModel` flushes pending SSD writes before the
        // teardown, so a plain offload never costs the disk tier its fresh
        // snapshots.
        manager.onOffloadModel = { [weak self] in
            Task { await self?.modelResidency.offload() }
        }
        manager.onClearMemoryCache = { [agentEngine] in
            agentEngine.llmActor.prefixCacheAdmin.clearRAMTier()
        }
        // Disk clear must not race the detached unload: a live ledger's
        // manifest persist after the wipe would resurrect the store. It frees
        // the voice and the language model, as it always has.
        manager.onClearDiskCache = { [weak self, inferenceArbiter, agentEngine, settingsManager] in
            let root = settingsManager.ssdPrefixCacheRootURL
            Task {
                await self?.offloadVoice()
                await inferenceArbiter.offloadLLM()
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
        manager.canOffload = { [weak self] in self?.modelResidency.canOffload ?? false }
        manager.residentCacheBytes = { [agentEngine] in
            agentEngine.llmActor.prefixCacheAdmin.residentRAMBytes
        }
        manager.diskCacheBytes = { [settingsManager] in
            SSDSnapshotStore.artifactBytes(at: settingsManager.ssdPrefixCacheRootURL)
        }
        manager.modelActivity = { [weak self] in self?.modelResidency.activity ?? .empty }
        return manager
    }()

    // App Bindings owns the launch sequence and every runtime subscription
    // with a rule; this container only wires its inputs and effects.
    lazy var appBindings = makeAppBindings()

    /// The **Hotkey Table**: every global hotkey, registered at launch and
    /// re-bound when its setting changes. The container supplies what each
    /// action is; the table says which key does it.
    lazy var hotkeyTable = HotkeyTable(
        settings: settingsManager, registrar: hotkeyManager, actions: makeHotkeyActions())

    private func makeHotkeyActions() -> HotkeyActions {
        HotkeyActions(
            startDictation: { [weak self] in self?.dictationCoordinator.onHotkeyDown() },
            finishDictation: { [weak self] in self?.dictationCoordinator.onHotkeyUp() },
            cancelDictation: { [weak self] in self?.dictationCoordinator.cancel() },
            holdTake: { [weak self] in self?.dictationCoordinator.shiftTapped() },
            fixLastTake: { [weak self] in self?.dictationLens.fixHotkeyPressed() },
            speakSelectedText: { [weak self] in self?.speechCoordinator.onHotkeyPressed() },
            startTalking: { [weak self] in self?.agentVoiceInput.start() },
            finishTalking: { [weak self] in self?.agentVoiceInput.finishCapture() },
            captureKeyDown: { [weak self] in self?.capturePanel.hotkeyDown() },
            captureKeyUp: { [weak self] in self?.capturePanel.hotkeyUp() },
            captureKeyCancelled: { [weak self] in self?.capturePanel.hotkeyCancelled() },
            takeAppshot: { [weak self] in Task { await self?.appshotController.takeAppshot() } },
            isVoiceSessionActive: { [weak self] in self?.liveVoiceSession != nil },
            interruptVoiceSession: { [weak self] in
                self?.liveVoiceSession?.bargeIn(source: "key")
            })
    }

    // Coordinator
    lazy var dictationCoordinator: DictationCoordinator = {
        let coordinator = DictationCoordinator(
            session: voiceCapture.makeSession(),
            textInjector: textInjector,
            history: transcriptionHistory,
            settings: settingsManager,
            feed: dictationFeed,
            pairs: correctionPairStore
        )
        // Every committed take is the one ⌃⌥Space reopens in the Lens.
        coordinator.onTakeCommitted = { [weak self] take in
            self?.dictationLens.takeCommitted(take)
        }
        return coordinator
    }()

    /// The **Lens** (PRD #612, ADR-0085, ADR-0086): the one dictation
    /// overlay. It streams the take while the key is held, holds it on ⇧,
    /// and ⌃⌥Space reopens the last take to fix a word by typing the one
    /// meant.
    lazy var dictationLens: LensController = makeDictationLens()

    private func makeDictationLens() -> LensController {
        let replacer = InAppReplacer(
            injector: textInjector,
            inputCount: { [hotkeyManager] in hotkeyManager.inputCount })
        let controller = LensController(
            model: LensModel(
                learnedWords: learnedWordStore, pairs: correctionPairStore,
                history: transcriptionHistory),
            pasteBack: LensController.PasteBack(
                replacer: replacer, injector: textInjector,
                restoreClipboard: { [settingsManager] in settingsManager.restoreClipboard }),
            vocabulary: { [learnedWordStore, correctionPairStore] in
                LensVocabulary.build(
                    learned: learnedWordStore.words,
                    recentTexts: correctionPairStore.pairs.prefix(200).compactMap {
                        $0.correction ?? $0.committed
                    })
            },
            isDictating: { [dictationFeed] in dictationFeed.phase.isActive },
            checkBeforePasting: { [settingsManager] in settingsManager.checkBeforePasting },
            fixHotkeyLabel: { [settingsManager] in settingsManager.fixHotkey.displayString },
            pastes: { [settingsManager] in settingsManager.autoInsertText },
            canTapShift: { [settingsManager] in !settingsManager.hotkey.isSingleModifier })
        controller.presenter = LensPanelPresenter(controller: controller, feed: dictationFeed)
        controller.onInsertRawAnyway = { [weak self] in
            self?.dictationCoordinator.insertRawAnyway()
        }
        controller.watch(dictationFeed)
        return controller
    }

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
                },
                persistTelemetry: { [ssdEnduranceAccumulator] in
                    ssdEnduranceAccumulator.persistNow()
                }
            ))
    }

    func setup() async {
        // The owner's background work — prewarms, model loads, the Companion,
        // the server, the hotkeys — runs only where the Launch Mode runs it
        // (ADR-0095): never in the unit tests' host, and never beside a
        // harness, which owns its model (a Companion moment that loaded a
        // second copy beside a DFlash2 ruler run deadlocked both in MLX's
        // compile).
        guard LaunchMode.current.runsOwnersBackgroundWork else { return }
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
                // Every global hotkey, bound now and re-bound when its
                // setting changes.
                hotkeyTable.start()
                // The Lens is the dictation overlay: build its panel now so
                // the first press shows it at once.
                dictationLens.prepare()
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
                speechState: { [speechCoordinator] in
                    speechCoordinator.state
                },
                isLLMLoaded: { [inferenceArbiter] in inferenceArbiter.isLLMLoaded },
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
                            max_context_length: definition.contextLength,
                            loaded_context_length: isLoaded ? definition.contextLength : nil,
                            state: isLoaded ? "loaded" : "available"
                        )
                    }
                let version =
                    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString")
                    as? String
                let response = OpenAI.ModelListResponse(server_version: version, data: models)
                return (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
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
