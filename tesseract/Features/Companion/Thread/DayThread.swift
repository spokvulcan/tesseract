//
//  DayThread.swift
//  tesseract
//
//  The Day Thread: one append-only conversation per day, which the Today
//  page's chat shows. It opens with the Day Opening (the owner's Profile,
//  Areas, today's agenda and last night's carry-over); each moment appends
//  its request and Jarvis's card; the owner's own Today messages go into the
//  same thread. A new day starts a new thread — the normal compaction point —
//  and within a day compaction runs only past the thread's ceiling.
//
//  It runs on its own agent with the same system prompt and tools as every
//  chat, so every moment and every Today turn reuses one cached prefix, and
//  the thread's own history is prefilled only as far as it grew.
//

import Foundation
import Observation

// MARK: - Store

/// The Day Thread's view of the shared conversation store: exactly one
/// conversation, the day's, saved through the one store instance so the
/// conversation index never has two writers.
@MainActor
final class DayThreadStore: AgentConversationStoring {

    private let backing: AgentConversationStore
    private(set) var currentConversation: AgentConversation?
    private(set) var day: DayKey
    private let calendar: Calendar

    init(backing: AgentConversationStore, day: DayKey, calendar: Calendar = .current) {
        self.backing = backing
        self.day = day
        self.calendar = calendar
    }

    func loadMostRecent() { show(day: day) }

    /// Make `day`'s thread current: its saved conversation, or a fresh one.
    func show(day: DayKey) {
        saveCurrent()
        self.day = day
        let id = AgentConversation.dayThreadID(for: day)
        currentConversation =
            backing.conversation(id: id)
            ?? AgentConversation(
                id: id, createdAt: day.start(calendar: calendar) ?? Date(), origin: .dayThread)
    }

    /// A day has one thread; "new" keeps it.
    @discardableResult
    func createNew() -> AgentConversation {
        if currentConversation == nil { show(day: day) }
        return currentConversation!
    }

    func load(id: UUID) {
        guard id != currentConversation?.id, let conversation = backing.conversation(id: id),
            conversation.isDayThread
        else { return }
        saveCurrent()
        currentConversation = conversation
    }

    func delete(id: UUID) {
        backing.delete(id: id)
        if currentConversation?.id == id {
            currentConversation = AgentConversation(
                id: id, createdAt: day.start(calendar: calendar) ?? Date(), origin: .dayThread)
        }
    }

    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        currentConversation?.messages = messages
    }

    func saveCurrent() {
        guard let current = currentConversation, !current.messages.isEmpty else { return }
        backing.save(current)
    }
}

// MARK: - Thread

@Observable @MainActor
final class DayThread {

    /// The Today page's chat over the thread.
    @ObservationIgnored let chat: ChatSession
    @ObservationIgnored private let agent: Agent
    @ObservationIgnored private let store: DayThreadStore
    @ObservationIgnored private let arbiter: any InferenceArbitrating
    @ObservationIgnored private let inferenceService: ServerInferenceService
    @ObservationIgnored private let settings: SettingsManager
    @ObservationIgnored private let trace: CompanionTrace
    @ObservationIgnored private let contextManager: ContextManager
    @ObservationIgnored private let summarize: @Sendable (String) async throws -> String
    @ObservationIgnored private let now: @MainActor () -> Date
    @ObservationIgnored private let toolRegistry: ToolRegistry

    /// Builds the Day Opening when the thread starts.
    @ObservationIgnored var openingProvider: (@MainActor () -> String)?

    /// A moment is generating in the thread right now.
    private(set) var momentRunning: MomentKind?

    var day: DayKey { store.day }

    init(
        agent: Agent, store: DayThreadStore, arbiter: any InferenceArbitrating,
        inferenceService: ServerInferenceService, toolRegistry: ToolRegistry,
        settings: SettingsManager, speechCoordinator: SpeechCoordinator?,
        contextManager: ContextManager,
        summarize: @escaping @Sendable (String) async throws -> String,
        trace: CompanionTrace, now: @escaping @MainActor () -> Date = Date.init
    ) {
        self.agent = agent
        self.store = store
        self.arbiter = arbiter
        self.inferenceService = inferenceService
        self.settings = settings
        self.trace = trace
        self.contextManager = contextManager
        self.summarize = summarize
        self.now = now
        self.toolRegistry = toolRegistry
        self.chat = ChatSession(
            agent: agent, conversationStore: store, arbiter: arbiter, toolRegistry: toolRegistry,
            settings: settings, speechCoordinator: speechCoordinator,
            contextManager: contextManager,
            contextWindow: Self.compactionWindow(ceiling: settings.companionThreadCeilingTokens),
            summarize: summarize,
            momentSummary: { request, reply in
                MomentTranscript.summary(request: request, reply: reply)
            })
    }

    /// Compaction triggers when the thread passes its ceiling: the window is
    /// the ceiling plus the compactor's reserve.
    static func compactionWindow(ceiling: Int) -> Int {
        max(ceiling, 8_000) + CompactionSettings.standard.reserveTokens
    }

    var isChatBusy: Bool { chat.isGenerating }

    /// Show `day`'s thread (after the 04:00 rollover, the new day's).
    func show(day: DayKey) {
        guard day != store.day || store.currentConversation == nil else { return }
        chat.showDayThread { store.show(day: day) }
    }

    /// Whether the owner can send now: not while a turn or a moment is
    /// generating, so every turn lands in the thread in order.
    var canSend: Bool { !chat.isGenerating && momentRunning == nil }

    /// The owner's own message in Today.
    func send(_ text: String) {
        guard canSend else { return }
        openIfNeeded()
        chat.sendMessage(text)
    }

    /// Start the thread with its Day Opening, once.
    func openIfNeeded() {
        guard agent.state.messages.isEmpty, let opening = openingProvider?() else { return }
        let message = UserMessage(content: opening, timestamp: now(), turnOrigin: .moment)
        guard chat.appendCommitted([message]) else { return }
        trace.record(
            .threadOpened, conversationID: store.currentConversation?.id,
            fields: [
                "day": .string(day.rawValue),
                "openingTokens": .int(TokenEstimator.estimate(opening)),
            ])
    }

    // MARK: Moments

    /// One moment: the request appended to the thread, one generation over
    /// the thread with no tool run, the reply appended after it.
    func runMoment(_ request: MomentRequest) async -> MomentOutcome {
        guard !chat.isGenerating else { return .failed("the Today chat is busy", nil) }
        momentRunning = request.kind
        defer { momentRunning = nil }
        openIfNeeded()
        await compactIfPastCeiling()
        syncActiveTools()

        let started = now()
        let message = UserMessage(content: request.text, timestamp: started, turnOrigin: .moment)
        var parameters = settings.makeAgentGenerateParameters()
        parameters.maxTokens = request.kind.maxTokens
        let cached = CachedTokenBox()
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: inferenceService, parametersProvider: { [parameters] in parameters },
            onStart: { count in cached.value = count })
        let systemPrompt = agent.state.systemPrompt
        let tools = agent.state.tools
        let history = agent.state.messages + [message]
        let llmMessages = history.compactMap { $0.toLLMMessage() }
        let modelID = settings.selectedAgentModelID
        let vision: LLMVisionRequirement =
            settings.useVisionWhenAvailable ? .visionIfCapable : .fromSettings

        do {
            let result = try await arbiter.withExclusiveGPU(
                .llm, llmModelIDOverride: nil, llmVision: vision
            ) { () async throws -> MomentGenerationResult in
                var accumulator = GenerationAccumulator()
                var builder = AssistantPartsBuilder()
                builder.model = modelID
                var info: AgentGeneration.Info?
                for try await generation in generate(systemPrompt, llmMessages, tools, nil) {
                    if case .info(let measured) = generation { info = measured }
                    accumulator.ingest(generation)
                    _ = builder.ingest(generation)
                }
                _ = builder.closeForTerminal()
                let reply = builder.finalize(stopReason: builder.terminalStopReason)
                return MomentGenerationResult(
                    text: accumulator.text, reply: reply, info: info,
                    hitCap: builder.hitLengthLimit)
            }
            let measure = MomentMeasure(
                promptTokens: result.info?.promptTokenCount ?? 0,
                cachedTokens: cached.value,
                outputTokens: result.info?.generationTokenCount ?? 0,
                prefillSeconds: result.info?.promptTime ?? 0,
                generateSeconds: result.info?.generateTime ?? 0,
                latencySeconds: now().timeIntervalSince(started),
                hitCap: result.hitCap, modelID: modelID)
            // Request and reply join the thread whatever the reply holds: the
            // thread records what was asked and said, append-only. A moment
            // runs no tools, so a tool call it emitted is dropped rather than
            // left in the thread without a result.
            var reply = result.reply
            reply.content.removeAll { if case .toolCall = $0 { true } else { false } }
            if reply.content.isEmpty { reply.content = [.text(TextPart(text: "(no card)"))] }
            chat.appendCommitted([message, reply])
            return .reply(result.text, measure)
        } catch {
            return .failed(error.localizedDescription, nil)
        }
    }

    /// The same resolve every chat runs before a turn, so a moment carries
    /// exactly the chat's tools and system prompt: one cached prefix.
    private func syncActiveTools() {
        let tools = ActiveToolSet.resolve(
            from: toolRegistry.allTools,
            gating: ToolGating(webAccessEnabled: settings.webAccessEnabled))
        agent.updateTools(tools)
        agent.syncSystemPrompt(facts: ActiveToolSet.promptFacts(for: tools))
    }

    private func compactIfPastCeiling() async {
        let ceiling = settings.companionThreadCeilingTokens
        let before = TokenEstimator.estimateTotal(agent.state.messages)
        guard before > ceiling else { return }
        await agent.forceCompact(
            contextManager: contextManager,
            contextWindow: Self.compactionWindow(ceiling: ceiling), summarize: summarize)
        chat.adoptAgentMessages()
        let after = TokenEstimator.estimateTotal(agent.state.messages)
        trace.record(
            .threadCompacted, conversationID: store.currentConversation?.id,
            fields: ["beforeTokens": .int(before), "afterTokens": .int(after)])
    }
}

/// Where the inference start reports the cache's share of the prompt.
@MainActor
private final class CachedTokenBox {
    var value = 0
}

private nonisolated struct MomentGenerationResult: Sendable {
    let text: String
    let reply: AssistantMessage
    let info: AgentGeneration.Info?
    let hitCap: Bool
}

// MARK: - Transcript

/// How a moment turn reads in the Today chat: one quiet line instead of the
/// request and the JSON card (the card itself shows at the top of Today).
nonisolated enum MomentTranscript {
    static func summary(request: UserMessage, reply: AssistantMessage?) -> String {
        let header = request.content.split(separator: "\n").first.map(String.init) ?? ""
        let name =
            header.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .components(separatedBy: " — ").first ?? "Jarvis"
        guard let reply else { return name }
        let text = reply.content.compactMap { part -> String? in
            if case .text(let text) = part { return text.text }
            return nil
        }.joined()
        if let data = CardParser.jsonObject(in: text),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let line = object["line"] as? String ?? object["carry_over"] as? String
        {
            return "\(name) · \(line)"
        }
        return name
    }
}
