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
            isEnabled: { settings.memoryEnabled && settings.memorySleepEnabled },
            companionNightly: { [weak self] in await self?.companionSleep.nightly() }
        )
    }

    /// The entity's practice at the tail of the sleep pass (ADR-0046, #370):
    /// the standing-instructions review; #373 adds the Digest.
    lazy var companionSleep = CompanionSleep(
        store: memoryStore,
        recorder: companionFlightRecorder,
        arbiter: inferenceArbiter,
        complete: internalCompletion,
        isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled }
    )

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
        // The one generic tracking door (ADR-0046, #369) — registered in
        // every conversation, same as memory's: the check-in IS the measuring
        // instrument, whichever conversation it happens in.
        registry.appendBuiltInTool(createTrackTool(store: memoryStore))
        // The flight recorder's one write door (#326; the read path died with
        // ADR-0046 — the standing conversation is the record).
        registry.appendBuiltInTool(
            createLogFeedbackTool(
                recorder: companionFlightRecorder,
                currentConversationID: { [weak self] in
                    self?.agentConversationStore.currentConversation?.id
                }
            ))
        // The wake palette — book, revise, cancel (ADR-0040, #369). Registered
        // in every conversation like memory's tools: "remind me tomorrow" said
        // in chat books a wake through the same one door the Companion's own
        // turns use.
        registry.appendBuiltInTool(
            createBookWakeTool(
                store: memoryStore,
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        registry.appendBuiltInTool(
            createReviseWakeTool(
                store: memoryStore,
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        registry.appendBuiltInTool(
            createCancelWakeTool(
                store: memoryStore,
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        registry.appendBuiltInTool(
            createReviseInstructionsTool(
                store: memoryStore,
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        // The delivery palette (ADR-0040 §10) — one typed tool per rung. The
        // shared registry carries them for the Companion's headless agent;
        // each is declared `audience: .companionOnly`, so the interactive
        // chat's tool sync (`AgentRunController`) drops them.
        registry.appendBuiltInTool(
            createSetGlyphTool(
                presence: companionPresence,
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        registry.appendBuiltInTool(
            createNotifyTool(deliver: { [weak self] title, body in
                await self?.companionLoop.deliverNotification(title: title, body: body)
            }))
        registry.appendBuiltInTool(
            createSpeakTool(deliver: { [weak self] text in
                self?.companionLoop.deliverSpoken(text)
            }))
        registry.appendBuiltInTool(
            createSummonOverlayTool(
                summon: { [weak self] line in
                    self?.companionSummons.summon(line: line)
                },
                recorder: companionFlightRecorder,
                context: companionTurnContext
            ))
        // The deposit door (ADR-0046 #372) — `.dialogueOnly`: surfaces only
        // while a summoned dialogue is the current chat, and the headless
        // agent's tool set drops it.
        registry.appendBuiltInTool(
            createReportBackTool(
                store: memoryStore,
                recorder: companionFlightRecorder,
                currentConversationID: { [weak self] in
                    self?.agentConversationStore.currentConversation?.id
                },
                depositLanded: { [weak self] id in
                    self?.companionDialogue.depositLanded(in: id)
                }
            ))
        return registry
    }()

    /// The one "put a conversation on his screen" action — the loop's
    /// reaction routing and the summons engagement reach the UI through it.
    private func presentConversation(_ id: UUID) {
        chatSession.loadConversation(id)
        (NSApp.delegate as? AppDelegate)?.navigateToAgent()
    }

    /// The Companion's interaction-fact log (#326): app-owned, App Support,
    /// retention forever. Only app code writes; the model reads via
    /// its own standing conversation and testifies via `log_feedback`.
    lazy var companionFlightRecorder = CompanionFlightRecorder()
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
            conversationMemory: ConversationMemory(memory: memoryEngine),
            // One Jarvis everywhere (#370): the IDENTITY section rides the
            // interactive chat — and the voice session, which sends through it.
            companionIdentity: CompanionIdentity(
                store: memoryStore,
                isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled }
            ),
            // The dialogue ledger's activity signal (#372) — lazy through the
            // container so the session and the ledger can reference each other.
            onDialogueActivity: { [weak self] id in
                self?.companionDialogue.activity(in: id)
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

    // The Companion (ADR-0040): the entity's harness. The turn context is the
    // correlation box the tools and the runner share; the runner is the
    // headless turn envelope; the loop is the ticking evaluator that grants
    // turns. Replaces the walking skeleton (#303).
    lazy var companionTurnContext = CompanionTurnContext()
    /// Jarvis's ambient presence (#327 §3): the glyph and the chat strip
    /// render it; the runner and the summons path drive it.
    lazy var companionPresence = CompanionPresence(recorder: companionFlightRecorder)
    lazy var companionTurnRunner = CompanionTurnRunner(
        // Deferred bootstrap (`unowned` is safe: the container outlives every
        // consumer) — a second full agent whose context never collides with
        // the chat session's, over the same shared tool registry.
        makeAgent: { [unowned self] in
            let headless = AgentFactory.makeAgent(
                inferenceService: self.serverInferenceService,
                packageRegistry: self.packageRegistry,
                extensionHost: self.extensionHost,
                toolRegistry: self.newToolRegistry,
                contextManager: self.contextManager,
                settingsManager: self.settingsManager,
                mcpToolsExtension: self.mcpClientManager.toolsExtension
            )
            // A Mission Control turn has no dialogue to report back from
            // (#372): the headless agent drops the `.dialogueOnly` tools.
            headless.updateTools(
                self.newToolRegistry.allTools.filter { $0.audience != .dialogueOnly })
            return headless
        },
        arbiter: inferenceArbiter,
        conversationStore: agentConversationStore,
        memory: memoryEngine,
        recorder: companionFlightRecorder,
        settings: settingsManager,
        context: companionTurnContext,
        presence: companionPresence,
        isModelDownloaded: { [modelDownloadManager] in
            modelDownloadManager.isDownloaded($0)
        }
    )
    // Explicitly typed: the loop's `speak` closure reaches the summons, and
    // the summons's banner fallback reaches the loop — inference across the
    // two lazy initializers would be circular.
    lazy var companionLoop: CompanionLoop = CompanionLoop(
        store: memoryStore,
        recorder: companionFlightRecorder,
        runner: companionTurnRunner,
        notifier: CompanionNotifier(),
        idleMonitor: idleMonitor,
        sensed: sensedObservations,
        calendar: CompanionCalendarReader(),
        isGPUBusy: { [inferenceArbiter] in inferenceArbiter.isGPULeaseHeld },
        // Briefing evidence, not a gate (#371): live owner activity — voice
        // session, interactive generation, dictation capture, TTS he is
        // listening to, or the app frontmost with input in the last two
        // minutes. The loop samples this each tick for "he last used the
        // app…"; owner-attention protection is the arbiter's FIFO now.
        isOwnerEngaged: { [unowned self] in
            if self.companionVoiceSession.isActive { return true }
            if self.chatSession.isGenerating { return true }
            if self.agentVoiceInput.voiceState == .recording
                || self.agentVoiceInput.voiceState == .transcribing
            {
                return true
            }
            if self.speechCoordinator.state.isActive { return true }
            return NSApp.isActive && IdleMonitor.hidIdleSeconds() < 120
        },
        isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled },
        // The spoken rung — choreography lives in `CompanionSummons`.
        speak: { [weak self] text in
            self?.companionSummons.deliver(line: text)
        },
        openConversation: { [weak self] id in self?.presentConversation(id) },
        perceiveDayStart: { [weak self] now in self?.companionPerception.dayStarted(at: now) }
    )
    /// The fold's perception substrate (ADR-0046, #368): the v1 Event
    /// producers. Power and app-session verdicts arrive through the sensed-
    /// observation pipeline's doors (wired in `bootstrap`, beside the loop's
    /// arming); day-start rides the loop's own detection.
    lazy var companionPerception = CompanionPerception(
        store: memoryStore,
        recorder: companionFlightRecorder,
        isEnabled: { [settingsManager] in settingsManager.companionHeartbeatEnabled }
    )

    /// The summoned dialogue's ledger (ADR-0046 #372): mints the dialogue
    /// chat on engagement, tracks the Report-Back debt, and delivers the one
    /// harness nudge when a dialogue ends or goes quiet without depositing.
    lazy var companionDialogue: CompanionDialogue = CompanionDialogue(
        recorder: companionFlightRecorder,
        openDialogue: { [weak self] line in
            guard let self else { return nil }
            let id = self.chatSession.beginDialogue(line: line)
            (NSApp.delegate as? AppDelegate)?.navigateToAgent()
            return id
        },
        isAgentBusy: { [weak self] in self?.chatSession.isGenerating ?? false },
        currentConversationID: { [weak self] in
            self?.agentConversationStore.currentConversation?.id
        },
        sendNudge: { [weak self] text in
            self?.chatSession.sendMessage(text, images: [], bypassCommandParsing: true)
        }
    )

    // The summons conductor (ADR-0040 §10/§11, #328): speak → overlay →
    // reaction routing → banner fallback for the unanswered. Conduct lives in
    // the type; this container hands it doors.
    lazy var companionSummons: CompanionSummons = CompanionSummons(
        settings: settingsManager,
        presence: companionPresence,
        recorder: companionFlightRecorder,
        context: companionTurnContext,
        speakPlain: { [weak self] text in
            self?.speechCoordinator.speakText(text)
        },
        speakUnderOverlay: { [weak self] text in
            self?.speechCoordinator.speakText(text, showsOverlay: false)
        },
        summonOverlay: { [weak self] title, line in
            await self?.companionVoicePrototype.summonBeat(title: title, line: line)
                ?? .unanswered
        },
        beginDialogue: { [weak self] line in
            self?.companionDialogue.begin(line: line, via: "summons-engage")
        },
        enterVoiceSession: { [weak self] via in
            self?.companionVoiceSession.enter(via: via)
        },
        postFallbackBanner: { [weak self] line, wakeID, conversationID in
            await self?.companionLoop.deliverUnansweredFallback(
                line: line, wakeID: wakeID, conversationID: conversationID)
        }
    )

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
            recorder: companionFlightRecorder,
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
        // The dialogue ledger listens on the session's end (#372): a summoned
        // dialogue that concluded without a deposit gets its one nudge here.
        controller.onSessionEnded = { [weak self] in
            self?.companionDialogue.voiceSessionEnded()
        }
        return controller
    }()

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
        // Dual-Path Playback (ADR-0041): the voice-session sink renders
        // session replies through the VPIO capture engine under its voice
        // hold; every other TTS surface keeps the dedicated engine.
        let coordinator = SpeechCoordinator(
            textExtractor: textExtractor,
            engine: speechEnginePresenter,
            settings: settingsManager,
            notchOverlay: ttsNotchPanelController
        )
        coordinator.voiceSessionPlayback = VoiceSessionPlayback(host: audioCaptureEngine)
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

        // Arm the Companion loop (ADR-0040) — a sleeping tick task and one
        // due-ness evaluation per 30 s unless the Companion toggle is on.
        companionLoop.start()

        // Arm the perception substrate (ADR-0046, #368): Events accumulate on
        // the record; nothing consumes them until the purist clock (#371).
        sensedObservations.onPowerTransition = { [weak self] onAC in
            self?.companionPerception.powerChanged(onACPower: onAC)
        }
        sensedObservations.onSustainedAppSession = { [weak self] app, start, end in
            self?.companionPerception.sustainedAppSession(app: app, start: start, end: end)
        }
        companionPerception.start()

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
                modelDownloadStatuses: modelDownloadManager.$statuses.eraseToAnyPublisher(),
                isAgentModelDownloaded: { [modelDownloadManager] in
                    modelDownloadManager.isDownloaded($0)
                },
                isMemorySleepRunning: { [memorySleep] in memorySleep.isRunning }
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
                },
                pushCompanionAsleep: { [companionPresence] in
                    companionPresence.setAsleep($0)
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
