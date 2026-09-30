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
    // the model actor never touches the arbiter — skip-when-busy reads the
    // lease instead of queueing on it.
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
            isGPUBusy: { arbiter.isGPULeaseHeld },
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
        ToolRegistry(sandbox: agentSandbox, extensionHost: extensionHost)
    }()

    /// The Companion Trace: every Jarvis decision, card, reaction and agenda
    /// change, one JSONL file per day under Application Support.
    lazy var companionTrace = CompanionTrace()
    lazy var agentConversationStore = AgentConversationStore()
    lazy var inferenceArbiter: InferenceArbiter = {
        // TTS residency arrives as closures (evaluated lazily) because the
        // speech engine's GPU lease adapter needs the arbiter — stored
        // references in both directions would recurse at construction.
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
    // message stream; barge-in is app-observed at the engine seam (#326).
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
            sendMessage: { [weak self] text in
                self?.chatSession.sendMessage(text, bypassCommandParsing: true)
            },
            stageToComposer: { [composerDraft] text in
                composerDraft.restore(text: text, images: [])
            },
            speak: { [weak self] text, onDone in
                self?.speechCoordinator.speakText(
                    text, showsOverlay: false, route: .voiceSession, onSuccess: onDone)
            },
            stopSpeaking: { [weak self] in self?.speechCoordinator.stop() },
            pauseSpeaking: { [weak self] in self?.speechCoordinator.pause() },
            resumeSpeaking: { [weak self] in self?.speechCoordinator.resume() },
            speechState: { [weak self] in self?.speechCoordinator.state ?? .idle },
            currentConversationID: { [weak self] in
                self?.agentConversationStore.currentConversation?.id
            },
            overlay: companionVoicePrototype,
            recorder: companionTrace,
            settings: settingsManager,
            proofreadPass: proofreadPass,
            // The Echo Floor's far-end signal and the Soft Barge duck
            // (ADR-0041) — both live on the coordinator's active sink.
            playbackLevel: { [weak self] in
                self?.speechCoordinator.playbackLevelNow() ?? 0
            },
            fadeSpeech: { [weak self] target, duration in
                self?.speechCoordinator.fadePlayback(to: target, over: duration)
            },
            // ADR-0041: the capture engine is held (and hosts the reply's
            // playback) for the session's lifetime.
            beginVoiceHold: { [weak self] in self?.audioCaptureEngine.beginVoiceHold() },
            endVoiceHold: { [weak self] in self?.audioCaptureEngine.endVoiceHold() }
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
                        in: StorageEnvironment.caches)),
                gpu: ArbiterGPULease(arbiter: inferenceArbiter)
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
        // Dual-Path Playback (ADR-0041): the voice-session sink renders
        // session replies through the VPIO capture engine under its voice
        // hold; every other TTS surface keeps the dedicated engine.
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
        coordinator.voiceSessionPlayback = VoiceSessionPlayback(host: audioCaptureEngine)
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
