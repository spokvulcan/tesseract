//
//  CompanionTurnRunner.swift
//  tesseract
//
//  One turn = one full agent run granted to the entity (ADR-0040). Headless
//  sibling of the Chat Session's send path: same arbiter lease discipline,
//  same memory injection, same tool loop — but every turn folds into Mission
//  Control, the one standing conversation that is the entity's whole cognitive
//  state (ADR-0046): the turn's context is the conversation so far, its
//  messages append origin-tagged on the opening, and per-turn minting is
//  retired. Full observability is the point: every tool call the entity makes
//  is in the one transcript the owner can open.
//
//  The runner never decides anything. It hands the entity the opening message
//  (instructions + situation briefing) and records what happened.
//

import Foundation

@MainActor
final class CompanionTurnRunner {

    struct Outcome {
        let turnID: UUID
        let conversationID: UUID
    }

    /// The delivery tools and `book_wake` read the in-flight correlation ids.
    /// Injected: the tool registry closes over the same box, and the container
    /// must be able to build the registry without forcing this runner.
    let context: CompanionTurnContext

    private(set) var isRunning = false

    private let makeAgent: () -> Agent
    private let arbiter: any InferenceArbitrating
    private let conversationStore: AgentConversationStore
    /// Mission Control's memory decoration (ADR-0045): the same enrich verb the
    /// chat rides, with the injection-dedupe set spanning the runner's lifetime
    /// — the standing conversation is one session, so what turn 1 injected is
    /// still in turn 7's context and must not be told again. Never reset.
    private let conversationMemory: ConversationMemory
    private let recorder: CompanionFlightRecorder
    private let settings: SettingsManager
    private let presence: CompanionPresence
    /// The download check behind `effectiveModelID` — an undownloaded
    /// Companion model must degrade to the owner's model, never to a turn
    /// that cannot run.
    private let isModelDownloaded: (String) -> Bool

    /// Built on first use — the second `AgentFactory.makeAgent` bootstrap is
    /// not free, and the Companion may be disabled for this whole launch.
    private var agent: Agent?

    init(
        makeAgent: @escaping () -> Agent,
        arbiter: any InferenceArbitrating,
        conversationStore: AgentConversationStore,
        memory: MemoryEngine,
        recorder: CompanionFlightRecorder,
        settings: SettingsManager,
        context: CompanionTurnContext,
        presence: CompanionPresence,
        isModelDownloaded: @escaping (String) -> Bool
    ) {
        self.makeAgent = makeAgent
        self.arbiter = arbiter
        self.conversationStore = conversationStore
        self.conversationMemory = ConversationMemory(memory: memory)
        self.recorder = recorder
        self.settings = settings
        self.context = context
        self.presence = presence
        self.isModelDownloaded = isModelDownloaded
    }

    /// Run one turn. Returns nil on failure — the caller (the loop) owns
    /// retries and the generic fallback; the wake stays unconsumed either way
    /// until the caller says otherwise.
    func run(origin: TurnOrigin, opening: String, wakeIDs: [UUID] = []) async -> Outcome? {
        guard !isRunning else { return nil }
        isRunning = true
        presence.beginThinking()
        defer {
            isRunning = false
            context.end()
            presence.endThinking()
        }

        let turnID = UUID()
        // The fold's state so far (ADR-0046): the turn's context is Mission
        // Control as it stands, and the turn appends to it — no minting.
        let missionControl = conversationStore.missionControl()
        let conversationID = missionControl.id
        context.begin(
            turnID: turnID, wakeIDs: wakeIDs, conversationID: conversationID, origin: origin)

        let modelID = effectiveModelID()
        recorder.record(
            "turn.started",
            turnID: turnID,
            conversationID: conversationID,
            modelID: modelID,
            snapshot: ["origin": origin.rawValue]
        )

        let agent = ensureAgent()

        // The same enrichment the chat path applies (ADR-0035 §5, ADR-0045):
        // retrieval against the opening, ridden as injectedContext so the
        // persisted conversation records exactly what the turn saw. The
        // opening carries the turn's origin — the per-turn tag the retired
        // per-turn conversations used to carry (ADR-0046).
        let openingMessage = UserMessage(content: opening, turnOrigin: origin)
        let user = (await conversationMemory.enrich(openingMessage)).asUser ?? openingMessage

        do {
            // The owner always wins the slot: this waits in the arbiter's FIFO
            // behind any interactive generation, never cancels one. The model
            // override swaps to the Companion's model even if the owner
            // temp-switched the interactive default (ADR-0040 §9).
            try await arbiter.withExclusiveGPU(
                .llm, llmModelIDOverride: modelID, llmVision: .fromSettings
            ) {
                agent.loadMessages(missionControl.messages)
                agent.prompt(CoreMessage.user(user))
                await agent.waitForIdle()
            }
        } catch is CancellationError {
            recorder.record(
                "turn.failed", turnID: turnID, conversationID: conversationID,
                note: "cancelled")
            return nil
        } catch {
            recorder.record(
                "turn.failed", turnID: turnID, conversationID: conversationID,
                note: error.localizedDescription)
            return nil
        }

        // Append the turn to the fold: the agent context now holds Mission
        // Control as loaded plus this turn's messages, and the save replaces
        // the standing conversation wholesale — without touching the UI's
        // current conversation. A crashed turn never reaches this line — its
        // wake re-presents (the correctness invariant).
        let messages = agent.context.messages
        var updated = missionControl
        updated.messages = messages
        conversationStore.save(updated)

        let summary = Self.lastAssistantText(in: messages) ?? "(silent turn)"
        recorder.record(
            "turn.completed",
            turnID: turnID,
            conversationID: conversationID,
            modelID: modelID,
            snapshot: ["origin": origin.rawValue, "messages": String(messages.count)],
            note: String(summary.prefix(300))
        )
        return Outcome(turnID: turnID, conversationID: conversationID)
    }

    // MARK: - Private

    private func ensureAgent() -> Agent {
        if let agent { return agent }
        let built = makeAgent()
        agent = built
        return built
    }

    /// The Companion model when it's actually on disk; otherwise nil (normal
    /// selected-model semantics) — a missing download must degrade to "runs on
    /// whatever the owner runs", never to a turn that cannot run.
    private func effectiveModelID() -> String? {
        let companion = settings.companionModelID
        guard !companion.isEmpty, companion != settings.selectedAgentModelID,
            isModelDownloaded(companion)
        else { return nil }
        return companion
    }

    /// Last displayable assistant text — used only for recorder notes.
    private static func lastAssistantText(in messages: [any AgentMessageProtocol & Sendable])
        -> String?
    {
        for message in messages.reversed() {
            if let text = message.asAssistant?.text, !text.isEmpty { return text }
        }
        return nil
    }
}
