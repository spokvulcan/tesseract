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

    // Memory (ADR-0035, map #314): the two-layer living store. The embedder is
    // a third co-resident MLX model, outside the arbiter like the proofreader
    // — embedding is a tiny forward pass with no decode, so it does not need
    // to contend for the lease at all. Built in methods for the same isolation
    // reason as the Proofread Pass above.
    lazy var memoryStore: MemoryStore = makeMemoryStore()
    lazy var memoryEmbedder = MemoryEmbedder()
    lazy var memoryEngine: MemoryEngine = makeMemoryEngine()
    /// The only thing in the app that watches for the *absence* of a person.
    lazy var idleMonitor = IdleMonitor()
    /// Consolidation (ADR-0035 §7). Runs on the agent's own model — the owner's
    /// call: the thinking that turns a day into beliefs is the last place to
    /// economise. It takes the `.llm` lease per generation and drops it between,
    /// so a foreground turn never waits behind more than one call.
    lazy var memorySleep: MemorySleep = makeMemorySleep()

    private func makeMemoryStore() -> MemoryStore {
        // `TESSERACT_MEMORY_DIR` runs the app against a scratch store — the seam
        // for driving the Memory window against seeded fixtures, and for any
        // future experiment that must not touch what the owner actually said.
        var home = PathSandbox.defaultRoot.appendingPathComponent("memory", isDirectory: true)
        if let override = ProcessInfo.processInfo.environment["TESSERACT_MEMORY_DIR"] {
            home = URL(fileURLWithPath: override, isDirectory: true)
            Log.memory.info("Memory store overridden to \(home.path)")
        }
        do {
            return try MemoryStore(directory: home)
        } catch {
            // A memory store that cannot open must not take the app down: the
            // assistant is usable without memory, and it is not usable crashed.
            // Fall back to a scratch store so every call site still has
            // somewhere to go — this launch simply forgets.
            Log.memory.error(
                "Memory store failed to open at \(home.path): \(error.localizedDescription)")
        }
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("tesseract-memory-\(UUID().uuidString)", isDirectory: true)
        do {
            return try MemoryStore(directory: scratch)
        } catch {
            // A fresh temp directory is unwritable: the filesystem is gone, and
            // nothing else in this app is going to work either.
            Log.memory.fault("Memory store cannot open anywhere: \(error.localizedDescription)")
            preconditionFailure("Memory store could not open a scratch database")
        }
    }

    private func makeMemoryEngine() -> MemoryEngine {
        let store = memoryStore
        let embedder = memoryEmbedder
        let settings = settingsManager
        let downloads = modelDownloadManager
        return MemoryEngine(
            store: store,
            embedder: embedder,
            isEnabled: { settings.memoryEnabled },
            isDictationCaptureEnabled: { settings.memoryCaptureDictation },
            embedderDirectory: {
                downloads.isDownloaded(ModelDefinition.defaultEmbeddingModelID)
                    ? downloads.modelPath(for: ModelDefinition.defaultEmbeddingModelID)
                    : nil
            }
        )
    }

    /// The internal completion path — the agent's own model, streamed through
    /// the shared inference service and folded to plain text. One closure
    /// serves compaction, sleep, and the Companion's memory callback; they run
    /// on the same model *on purpose* (the owner's call: the thinking that
    /// turns a day into beliefs is the last place to economise).
    lazy var internalCompletion: @Sendable (String) async throws -> String =
        makeSummarizeClosure(
            inferenceService: serverInferenceService,
            parametersProvider: { [settingsManager] in
                settingsManager.makeAgentGenerateParameters()
            }
        )

    private func makeMemorySleep() -> MemorySleep {
        let settings = settingsManager
        return MemorySleep(
            engine: memoryEngine,
            store: memoryStore,
            arbiter: inferenceArbiter,
            complete: internalCompletion,
            isEnabled: { settings.memoryEnabled && settings.memorySleepEnabled }
        )
    }

    /// The Companion's zero-dialog sensing tier (#308): presence spans, app
    /// sessions, power transitions → the observation stream. Writes are gated
    /// on the Companion toggle inside the recorder.
    lazy var sensedObservations = SensedObservationRecorder(
        store: memoryStore,
        isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled }
    )

    /// Wire idleness to consolidation (ADR-0035 §7). The owner walking away is
    /// the *only* thing that starts a sleep, and him coming back is the only
    /// thing that stops one — instantly, by cancelling it mid-generation.
    /// The sensed-observation recorder shares the same two transitions: idle
    /// closes a presence span, return opens one.
    func startMemoryConsolidationLoop() {
        idleMonitor.onIdle = { [memorySleep, sensedObservations] in
            memorySleep.start()
            sensedObservations.ownerWentIdle()
        }
        idleMonitor.onReturn = { [memorySleep, sensedObservations] in
            memorySleep.yield()
            sensedObservations.ownerReturned()
        }
        idleMonitor.start()
        sensedObservations.start()
    }

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
        // The living memory's three hands (ADR-0035). Appended rather than built
        // into the factory because they need the engine, which the factory —
        // a `nonisolated` free function over a sandbox — has no way to reach.
        registry.appendBuiltInTool(createRememberTool(memory: memoryEngine))
        registry.appendBuiltInTool(createRecallTool(memory: memoryEngine))
        registry.appendBuiltInTool(createContestTool(memory: memoryEngine))
        // The tracking model's five typed talk-time tools (#308) — registered
        // in every conversation, same as memory's: the check-in IS the
        // measuring instrument, whichever conversation it happens in.
        registry.appendBuiltInTool(createPlanDayTool(store: memoryStore))
        registry.appendBuiltInTool(createLogStepTool(store: memoryStore))
        registry.appendBuiltInTool(createLogSampleTool(store: memoryStore))
        registry.appendBuiltInTool(createLogTaskTool(store: memoryStore))
        registry.appendBuiltInTool(createCloseDayTool(store: memoryStore))
        return registry
    }()
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
            },
            memory: memoryEngine
        )
    }()

    // Appshot — the double-Command frontmost-window capture (PRD #170). Stages
    // through the Composer Draft; the app delegate attaches the window-summon
    // callback it owns.
    lazy var appshotController = AppshotController(
        capturer: ScreenCaptureKitAppshotCapturer(),
        composerDraft: composerDraft
    )

    // The Companion walking skeleton (map #301, ticket #303): a deliberately
    // crude lived-with heartbeat — an instrument the owner wears while the
    // Companion grillings run. Absorbed or retired by the map's exit PRDs.
    lazy var companionHeartbeat = CompanionHeartbeat(
        isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled },
        speaks: { [settingsManager] in settingsManager.companionHeartbeatSpeaks },
        // Audio-only while the voice overlay is the summons surface — one
        // beat must never raise two visual surfaces (TTS notch + overlay).
        speak: { [weak self] text in
            guard let self else { return }
            self.speechCoordinator.speakText(
                text, showsOverlay: !self.settingsManager.companionBeatsUseOverlay)
        },
        onEngage: { (NSApp.delegate as? AppDelegate)?.navigateToAgent() },
        // #328 wearing instrument: beats summon the picked overlay concept
        // when the toggle is on; unanswered falls back to the banner.
        overlaySummonsEnabled: { [settingsManager] in settingsManager.companionBeatsUseOverlay },
        summonOverlay: { [weak self] title, body in
            await self?.companionVoicePrototype.summonBeat(title: title, line: body)
                ?? .unanswered
        },
        // The beat asks the question; memory supplies the *particular* one. A
        // nil here — memory empty, model unsure, output unusable — falls back to
        // the beat's own hardcoded line, which is the whole safety story: generic
        // is a disappointment, invented would be a betrayal.
        composeBody: { [weak self] beat in
            guard let self else { return beat.prompt }
            let line = await MemoryCallback.compose(
                cue: beat.prompt,
                engine: self.memoryEngine,
                arbiter: self.inferenceArbiter,
                complete: self.internalCompletion
            )
            return line ?? beat.prompt
        }
    )

    // PROTOTYPE — the Companion voice-overlay concepts (map #301, ticket
    // #328): scripted demo scenes on throwaway overlay surfaces, driven from
    // Settings. Deleted when the concepts prune to a winner.
    lazy var companionVoicePrototype = CompanionVoicePrototype(
        settings: settingsManager,
        openChat: { (NSApp.delegate as? AppDelegate)?.navigateToAgent() }
    )

    // Speech (TTS) — engine v2 (ADR-0038/0039): the engine actor lives in the
    // TesseractSpeech package behind its ports; the presenter mirrors
    // residency for views and the arbiter; the coordinator drives sessions.
    lazy var textExtractor = TextExtractor()
    lazy var speechEnginePresenter = SpeechEnginePresenter(
        engine: SpeechEngine(
            // ADR-0037 precision gate, measured 2026-07-13 (v2-listen longform,
            // 480-word article, release): q8 peaks at 3.29 GB — over the ≤3 GB
            // envelope — q6 at 2.88 GB. q6 is the shipped default.
            model: .voiceDesign17B(.q6),
            synthesizer: Qwen3Synthesizer(),
            gpu: ArbiterGPULease(arbiter: inferenceArbiter)
        )
    )
    lazy var ttsNotchPanelController = TTSNotchPanelController()
    lazy var speechCoordinator: SpeechCoordinator = {
        // `playback` is left to the coordinator's production default
        // (`AudioPlaybackManager()`) — the AVFoundation adapter is needed by
        // nothing else in the graph, so there is no shared handle to wire here.
        // Mirrors `SettingsManager()` above, which relies on its
        // `UserDefaultsSettingsStore()` default. Tests inject `InMemoryAudioPlayback`.
        SpeechCoordinator(
            textExtractor: textExtractor,
            engine: speechEnginePresenter,
            settings: settingsManager,
            notchOverlay: ttsNotchPanelController
        )
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
            pairs: correctionPairStore,
            memory: memoryEngine
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
        // Prevent duplicate setup from multiple window instances
        guard !hasSetup else { return }
        hasSetup = true
        // Register dictation push-to-talk — through the same registration API
        // as every other hotkey (audit #285 item 7).
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

        // Pump the capture engine's meter frames into the Overlay Feed — the
        // one attachment point (the feed owns the consuming task).
        dictationFeed.attachMeters(audioCaptureEngine.meters)

        // Register message codecs for the new persistence layer (Epic 2)
        await registerCoreMessageCodecs()

        hotkeyManager.startListening()

        // Register HTTP server routes before App Bindings starts the server.
        registerHTTPRoutes()

        // Hand off to App Bindings: the launch ordering and every runtime
        // subscription with a rule live (and are tested) there.
        appBindings.start()

        // Arm the Companion walking skeleton (#303) — a sleeping tick task and
        // one date comparison per 30 s unless the experimental toggle is on.
        companionHeartbeat.start()

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
                startMemory: { [self] in
                    await memoryEngine.prewarm()
                    await MemoryBackfill.run(engine: memoryEngine)
                    startMemoryConsolidationLoop()
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
