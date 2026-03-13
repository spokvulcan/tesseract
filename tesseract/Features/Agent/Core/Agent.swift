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

    // MARK: - Message State Management

    /// Load persisted messages into the agent's context and state.
    /// Used when restoring a conversation from disk.
    func loadMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        guard state.phase == .idle else { return }
        context.messages = messages
        state.messages = messages.map { $0 as any AgentMessageProtocol }
    }

    /// Reset agent messages (e.g. when creating a new conversation).
    func resetMessages(_ messages: [any AgentMessageProtocol & Sendable] = []) {
        guard state.phase == .idle else { return }
        context.messages = messages
        state.messages = messages.map { $0 as any AgentMessageProtocol }
        state.streamMessage = nil
        state.pendingToolCalls.removeAll()
        state.error = nil
    }

    // MARK: - Public API

    /// Start a new agent loop with the given message.
    func prompt(_ message: any AgentMessageProtocol & Sendable) {
        guard state.phase == .idle else { return }
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
        guard state.phase == .idle else { return }
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
        state.phase = .streaming
        state.error = nil

        // Copy context out (value type — the loop mutates its own copy).
        let snapshot = context

        runTask = Task { [weak self] in
            var ctx = snapshot
            let result = await body(&ctx, token)
            self?.finishRun(result)
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

        state.phase = .idle
        state.streamMessage = nil
        state.pendingToolCalls.removeAll()
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

    /// Process a single agent event: update observable state and notify subscribers.
    private func handleEvent(_ event: AgentEvent) {
        // Sync messages BEFORE notifying subscribers so they see current state.
        // The loop mutates its own context copy; these events carry snapshots.
        switch event {
        case .messageEnd(let message):
            // pi-mono: clear progressive stream on commit (this._state.streamMessage = null)
            state.streamMessage = nil
            // Commit to observable state on message_end (pi-mono pattern).
            // Guard: skip empty assistant messages from cancel/error paths.
            if let assistant = message as? AssistantMessage {
                let hasContent = !assistant.content.isEmpty
                    || (assistant.thinking?.isEmpty == false)
                    || !assistant.toolCalls.isEmpty
                guard hasContent else { break }
            }
            state.messages.append(message)
        case .turnEnd(_, _, let contextMessages):
            // Authoritative sync — full replace from loop context snapshot.
            state.messages = contextMessages.map { $0 as any AgentMessageProtocol }
        case .agentEnd:
            // finishRun sets state.messages from finalContext; no need to duplicate here.
            break
        default:
            break
        }

        // Notify all subscribers
        for handler in subscribers.values {
            handler(event)
        }

        // Update other observable state
        switch event {
        case .agentStart:
            state.phase = .streaming

        case .contextTransformStart(let reason):
            state.phase = .transformingContext(reason)

        case .contextTransformEnd(_, let didMutate, let messages):
            if didMutate, let messages {
                state.messages = messages.map { $0 as any AgentMessageProtocol }
            }
            state.phase = .streaming

        case .messageUpdate(let message, _):
            state.streamMessage = message

        case .toolExecutionStart(let id, let name, _):
            state.pendingToolCalls.insert(id)
            state.phase = .executingTool(name)

        case .toolExecutionEnd(let id, _, _, _):
            state.pendingToolCalls.remove(id)
            if state.pendingToolCalls.isEmpty {
                state.phase = .streaming
            }

        case .turnStart, .turnEnd, .agentEnd, .messageStart, .messageEnd,
             .toolExecutionUpdate, .malformedToolCall:
            break
        }
    }
}
