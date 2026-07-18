import Foundation
import Observation
import os

// MARK: - EventBuffer

/// Thread-safe event buffer used by the Agent emitter.
/// Written from any isolation domain (the loop runs off-MainActor),
/// drained on MainActor before state transitions.
private nonisolated final class EventBuffer: Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: [AgentEvent]())

    func append(_ event: AgentEvent) {
        lock.withLock { $0.append(event) }
    }

    func drain() -> [AgentEvent] {
        lock.withLock { buffer in
            let copy = buffer
            buffer.removeAll()
            return copy
        }
    }
}

// MARK: - Agent

/// Stateful wrapper around `agentLoop`. Manages queues, abort control,
/// observable state, and event subscriptions.
///
/// Analogous to Pi's `Agent` class. The UI observes `state` (an `@Observable`
/// `AgentState`); the loop operates on the authoritative `context`.
@MainActor
@Observable
final class Agent {

    // MARK: - Observable State

    /// UI-facing projection. Updated by event handlers and `finishRun`.
    private(set) var state = AgentState()

    // MARK: - Authoritative Context

    /// The loop reads/writes this. Copied out before a run, copied back after.
    private(set) var context: AgentContext

    // MARK: - Subscriptions

    private var subscribers: [UUID: (AgentEvent) -> Void] = [:]

    // MARK: - Message Queues

    private var steeringQueue: [any AgentMessageProtocol & Sendable] = []
    private var followUpQueue: [any AgentMessageProtocol & Sendable] = []

    // MARK: - Run Control

    private var cancellationToken: CancellationToken?
    private var runTask: Task<Void, Never>?

    // MARK: - Event Buffering

    /// Events from the loop are buffered here and drained on MainActor.
    /// This guarantees all events are processed before `finishRun` sets idle.
    private nonisolated let eventBuffer = EventBuffer()

    // MARK: - Configuration

    private let baseConfig: AgentLoopConfig
    private let generate: LLMGenerateFunction

    /// The prompt facts the current system prompt was assembled from, and the
    /// factory-captured reassembly closure `syncSystemPrompt` runs when they
    /// change (ADR-0048). Ignored by Observation: views observe
    /// `state.systemPrompt`, not the bookkeeping.
    @ObservationIgnored private var promptFacts: PromptToolFacts?
    @ObservationIgnored private var reassembleSystemPrompt:
        (@MainActor (PromptToolFacts) -> String)?

    // MARK: - Init

    init(
        config: AgentLoopConfig,
        systemPrompt: String,
        tools: [AgentToolDefinition],
        generate: @escaping LLMGenerateFunction
    ) {
        self.baseConfig = config
        self.generate = generate
        self.context = AgentContext(
            systemPrompt: systemPrompt,
            messages: [],
            tools: tools
        )
        state.systemPrompt = systemPrompt
        state.tools = tools
    }

    // MARK: - Tool Management

    /// Update the active tool set. Called before each prompt to reflect dynamic settings.
    func updateTools(_ tools: [AgentToolDefinition]) {
        guard !state.isBusy else { return }
        context.tools = tools
        state.tools = tools
    }

    // MARK: - System Prompt Management

    /// Wire the system-prompt reassembler (ADR-0048). `AgentFactory` captures
    /// the assembly inputs (loaded context, skills, agent root) in `reassemble`
    /// and records the facts the launch prompt was built from, so a later
    /// facts change can rebuild the prompt without the factory's collaborators.
    func setSystemPromptReassembler(
        initialFacts: PromptToolFacts,
        _ reassemble: @escaping @MainActor (PromptToolFacts) -> String
    ) {
        promptFacts = initialFacts
        reassembleSystemPrompt = reassemble
    }

    /// Re-derive the system prompt when the resolved tool set's prompt facts
    /// change — a Web Access flip, or browser tools materializing after a late
    /// MCP connect. Guarded like `updateTools`; a no-op when the facts are
    /// unchanged, so the prompt (and the prefix cache riding it) is only ever
    /// invalidated by a real orientation change.
    func syncSystemPrompt(facts: PromptToolFacts) {
        guard !state.isBusy, facts != promptFacts,
            let reassemble = reassembleSystemPrompt
        else { return }
        promptFacts = facts
        let prompt = reassemble(facts)
        context.systemPrompt = prompt
        state.systemPrompt = prompt
    }

    // MARK: - Message State Management

    /// Replace the agent's message log — restoring a conversation from disk,
    /// switching conversations, or truncating for edit-and-resend.
    func loadMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        guard !state.isBusy else { return }
        context.messages = messages
        state.messages = messages.map { $0 as any AgentMessageProtocol }
    }

    // MARK: - Public API

    /// Start a new agent loop with the given message.
    func prompt(_ message: any AgentMessageProtocol & Sendable) {
        guard !state.isBusy else { return }
        beginRun { [gen = generate, cfg = makeLoopConfig(), emit = makeEmitter()] ctx, token in
            await agentLoop(
                prompts: [message],
                context: &ctx,
                config: cfg,
                generate: gen,
                signal: token,
                emit: emit
            )
            return ctx
        }
    }

    /// Continue from existing context without new prompts (retry/resume).
    func `continue`() {
        guard !state.isBusy else { return }
        // Precondition: last message must not be an unfinished assistant turn.
        if context.messages.last is AssistantMessage { return }
        beginRun { [gen = generate, cfg = makeLoopConfig(), emit = makeEmitter()] ctx, token in
            await agentLoopContinue(
                context: &ctx,
                config: cfg,
                generate: gen,
                signal: token,
                emit: emit
            )
            return ctx
        }
    }

    /// Force context compaction outside of the normal agent loop (the `/compact`
    /// slash command). **Awaited** under the **Agent Run** lease, so `isGenerating`
    /// clears on lease completion exactly like a `send` turn — there is no
    /// event-derived busy-flag gate. Cancellation propagates from the lease task.
    func forceCompact(
        contextManager: ContextManager,
        contextWindow: Int,
        summarize: @escaping @Sendable (String) async throws -> String
    ) async {
        guard !state.isBusy else { return }
        guard !context.messages.isEmpty else { return }

        state.isBusy = true

        let messages = context.messages
        let emit = makeEmitter()
        emit(.contextTransformStart(reason: .compaction))

        do {
            let compacted = try await contextManager.compact(
                messages: messages,
                contextWindow: contextWindow,
                summarize: summarize
            )
            context.messages = compacted
            state.messages = compacted.map { $0 as any AgentMessageProtocol }
            emit(
                .contextTransformEnd(
                    reason: .compaction,
                    didMutate: true,
                    messages: compacted
                ))
        } catch {
            emit(
                .contextTransformEnd(
                    reason: .compaction,
                    didMutate: false,
                    messages: nil
                ))
            Log.agent.error("Forced compaction failed: \(error.localizedDescription)")
        }

        // Standalone run-lifecycle envelope: drain the transform events *then*
        // settle the busy bit, mirroring `finishRun` — subscribers see every
        // event before the agent reads as idle.
        drainPendingEvents()
        state.isBusy = false
    }

    /// Cancel in-progress generation.
    func abort() {
        cancellationToken?.cancel()
        runTask?.cancel()
    }

    /// Resolves when the agent becomes idle (returns immediately if already idle).
    func waitForIdle() async {
        guard let task = runTask else { return }
        await task.value
    }

    /// Subscribe to agent events. Returns an unsubscribe closure.
    func subscribe(_ handler: @escaping (AgentEvent) -> Void) -> @MainActor () -> Void {
        let id = UUID()
        subscribers[id] = handler
        return { [weak self] in
            self?.subscribers.removeValue(forKey: id)
        }
    }

    // MARK: - Queue Management

    /// Push a steering message that interrupts the current tool-execution batch.
    func pushSteering(_ message: any AgentMessageProtocol & Sendable) {
        steeringQueue.append(message)
    }

    /// Push a follow-up message that extends the session after the inner loop settles.
    func pushFollowUp(_ message: any AgentMessageProtocol & Sendable) {
        followUpQueue.append(message)
    }

    // MARK: - Private — Run Lifecycle

    /// Shared setup for `prompt` and `continue`.
    private func beginRun(
        _ body: @escaping @Sendable (inout AgentContext, CancellationToken?) async -> AgentContext
    ) {
        let token = CancellationToken()
        cancellationToken = token
        state.isBusy = true

        // Copy context out (value type — the loop mutates its own copy).
        let snapshot = context

        runTask = Task { [weak self] in
            var ctx = snapshot
            let result = await body(&ctx, token)
            // `body` is @Sendable and runs off-MainActor, so this continuation
            // resumes off-MainActor too. `finishRun` is @MainActor (it mutates
            // observable `state`), so hop back explicitly — matching the
            // steering-queue closures below. Calling it directly here traps
            // under Swift's dynamic isolation check.
            await MainActor.run { self?.finishRun(result) }
        }
    }

    /// Called on MainActor after the loop returns. Copies the authoritative context
    /// back, drains any remaining buffered events, then sets idle.
    private func finishRun(_ finalContext: AgentContext) {
        // Copy context back first so state.messages is current when events drain.
        context = finalContext
        state.messages = finalContext.messages.map { $0 as any AgentMessageProtocol }

        // Process any events that haven't been drained yet. This guarantees
        // subscribers see all events before the idle transition.
        drainPendingEvents()

        state.isBusy = false
        cancellationToken = nil
        runTask = nil
    }

    // MARK: - Private — Config & Emitter

    /// Build a loop config that wires the steering/follow-up queues to the base config.
    private func makeLoopConfig() -> AgentLoopConfig {
        AgentLoopConfig(
            model: baseConfig.model,
            convertToLlm: baseConfig.convertToLlm,
            contextTransform: baseConfig.contextTransform,
            getSteeringMessages: { [weak self] in
                await MainActor.run {
                    guard let self else { return [] }
                    defer { self.steeringQueue.removeAll() }
                    return self.steeringQueue
                }
            },
            getFollowUpMessages: { [weak self] in
                await MainActor.run {
                    guard let self else { return [] }
                    defer { self.followUpQueue.removeAll() }
                    return self.followUpQueue
                }
            }
        )
    }

    /// Build a `@Sendable` emit closure that buffers events and kicks MainActor drain.
    ///
    /// Events are written to a lock-protected buffer (safe from any isolation domain)
    /// and a MainActor task is enqueued to drain them. `finishRun` does a final drain
    /// before setting idle, so late events never overwrite the idle state.
    private func makeEmitter() -> @Sendable (AgentEvent) -> Void {
        let buffer = eventBuffer
        return { [weak self] event in
            buffer.append(event)
            Task { @MainActor [weak self] in
                self?.drainPendingEvents()
            }
        }
    }

    /// Drain all buffered events, processing each on MainActor in order.
    private func drainPendingEvents() {
        let events = eventBuffer.drain()
        for event in events {
            handleEvent(event)
        }
    }

    /// Process a single agent event: fold it into observable state through the
    /// **Agent State Reducer**, then notify subscribers — pi-mono's
    /// `processEvents` shape (reduce all state, *then* notify). Because the
    /// reduce fully settles before any listener runs, subscribers always read
    /// current `AgentState`.
    private func handleEvent(_ event: AgentEvent) {
        AgentStateReducer.reduce(event, into: state)

        for handler in subscribers.values {
            handler(event)
        }
    }
}
