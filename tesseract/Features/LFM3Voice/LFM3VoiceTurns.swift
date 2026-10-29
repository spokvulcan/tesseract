//
//  LFM3VoiceTurns.swift
//  tesseract
//
//  **LFM3 Voice**'s turns. The owner's words go to LFM3-Audio with the chat
//  so far; it answers in speech and text at once, or calls one of the agent's
//  tools and answers after the result. Every turn lands in the chat as regular
//  messages (the owner's words, the calls and their results, the answer), so
//  typed and spoken turns are one conversation.
//

import Foundation
import MLXLMCommon

@MainActor
final class LFM3VoiceTurns {

    /// The agent's tools a voice turn may call: the day, the owner's memory
    /// and the web. Fewer tools keep the prompt short (about 1,000 tokens).
    static let toolNames = [
        "agenda", "add_reminder", "add_event", "remember", "recall", "browser.search",
        "browser.fetch",
    ]
    /// Tool rounds before a turn has to answer.
    static let maxToolRounds = 3
    /// The chat's share of the 32K window.
    static let historyTokens = 15_000
    /// The longest tool result the model reads; the chat keeps the whole one.
    static let toolResultCharacters = 6_000

    private let engine: LFM3VoiceEngine
    private let playback: LFM3VoicePlayback
    private let tools: @MainActor () -> [AgentToolDefinition]
    private let chatMessages: @MainActor () -> [any AgentMessageProtocol]
    private let commit: @MainActor ([any AgentMessageProtocol & Sendable]) -> Bool
    private let onError: @MainActor (String) -> Void

    /// The voice session's reply hooks, wired once the session exists.
    var replyText: @MainActor (ChatSession.ReplyText) -> Void = { _ in }
    var replyCompleted: @MainActor (String) -> Void = { _ in }

    private var turn: Task<Void, Never>?
    private var toolSignal: CancellationToken?
    /// What the turn wrote and the chat doesn't have yet. The prompt reads it
    /// too, so a commit the chat refuses (a typed run is live) is retried.
    private var uncommitted: [any AgentMessageProtocol & Sendable] = []

    init(
        engine: LFM3VoiceEngine,
        playback: LFM3VoicePlayback,
        tools: @escaping @MainActor () -> [AgentToolDefinition],
        chatMessages: @escaping @MainActor () -> [any AgentMessageProtocol],
        commit: @escaping @MainActor ([any AgentMessageProtocol & Sendable]) -> Bool,
        onError: @escaping @MainActor (String) -> Void
    ) {
        self.engine = engine
        self.playback = playback
        self.tools = tools
        self.chatMessages = chatMessages
        self.commit = commit
        self.onError = onError
    }

    // MARK: - Interface

    /// The owner's turn. A turn still running is cancelled first, and its end
    /// reaches the session before this one starts.
    func send(_ text: String) {
        let previous = turn
        cancel()
        let signal = CancellationToken()
        toolSignal = signal
        turn = Task { [weak self] in
            await previous?.value
            guard !Task.isCancelled else { return }
            await self?.run(text, signal: signal)
        }
    }

    /// Stops the turn's answer and its tools; its end still reaches the session.
    func cancel() {
        turn?.cancel()
        toolSignal?.cancel()
    }

    /// Waits for the turn in flight, after `cancel`.
    func settle() async {
        await turn?.value
    }

    // MARK: - The turn

    private func run(_ words: String, signal: CancellationToken) async {
        playback.beginTurn()
        record(UserMessage(content: words))
        let tools = tools()
        var shown = ""
        var called = Set<String>()
        var mustAnswer = false
        do {
            for round in 0... {
                let mayCallTools = !mustAnswer && round < Self.maxToolRounds
                shown = ""
                let written = try await write(tools: tools, mayCallTools: mayCallTools) {
                    shown += $0
                }
                guard mayCallTools, !written.toolCalls.isEmpty else {
                    if !written.spoke, !shown.isEmpty {
                        try await engine.read(shown) { playback.play($0) }
                    }
                    if !shown.isEmpty { record(AssistantMessage(content: shown)) }
                    break
                }
                // What it said before the call is complete as it stands.
                if !shown.isEmpty { replyText(.partEnded) }
                let calls = written.toolCalls.map {
                    ToolCallInfo(
                        id: UUID().uuidString, name: $0.function.name,
                        argumentsJSON: ToolArgumentNormalizer.encode($0.function.arguments))
                }
                record(AssistantMessage(content: shown, toolCalls: calls))
                for call in calls {
                    record(await execute(call, tools: tools, signal: signal))
                }
                // The same call twice: what it needs is in the results already.
                let keys = calls.map {
                    "\($0.name) \(canonicalizeHTTPPrefixCacheToolArgumentsJSON($0.argumentsJSON))"
                }
                if keys.contains(where: called.contains) { mustAnswer = true }
                called.formUnion(keys)
            }
        } catch is CancellationError {
            // A barge-in or the owner's next turn: what was shown stays.
            if !shown.isEmpty {
                record(AssistantMessage(content: shown, stopReason: .aborted))
            }
        } catch {
            Log.agent.error("LFM3 Voice turn failed: \(error.localizedDescription)")
            onError("LFM3 Voice: \(error.localizedDescription)")
            shown = ""
        }
        commitPending()
        playback.endTurn()
        replyCompleted(shown)
    }

    /// One answer from the model, its text and speech handled as they come.
    /// The chat saves what the turn wrote while the model reads the prompt.
    private func write(
        tools: [AgentToolDefinition], mayCallTools: Bool, shown: (String) -> Void
    ) async throws -> LFM3VoiceAnswer {
        let prompt = makePrompt(tools: tools)
        let (outputs, continuation) = AsyncStream.makeStream(of: LFM3VoiceOutput.self)
        async let written = engine.answer(
            prompt, mayCallTools: mayCallTools, outputs: continuation)
        commitPending()
        for await output in outputs {
            switch output {
            case .text(let piece):
                shown(piece)
                replyText(.delta(piece))
            case .speech(let samples):
                playback.play(samples)
            }
        }
        return try await written
    }

    // MARK: - Prompt

    /// The newest messages that fit, from one of the owner's turns on.
    private func makePrompt(tools: [AgentToolDefinition]) -> LFM3VoicePrompt {
        var kept = [any AgentMessageProtocol]()
        var tokens = 0
        for message in (chatMessages() + uncommitted).reversed() {
            tokens += TokenEstimator.estimate(message)
            guard tokens <= Self.historyTokens else { break }
            kept.append(message)
        }
        var history = defaultConvertToLlm(Array(kept.reversed()))
        if let first = history.firstIndex(where: {
            if case .user = $0 { true } else { false }
        }) {
            history.removeFirst(first)
        }
        return LFM3VoicePrompt(
            input: AgentEngine.buildUserInput(
                systemPrompt: LFM3VoiceEngine.systemPrompt,
                messages: history.map(Self.forVoice), toolSpecs: tools.map(\.toolSpec)))
    }

    /// The voice's copy of a message: no pictures, which LFM3-Audio can't
    /// see, and long tool results cut.
    private static func forVoice(_ message: LLMMessage) -> LLMMessage {
        switch message {
        case .user(let content, let images) where !images.isEmpty:
            .user(content: content)
        case .toolResult(let id, let content, _) where content.utf8.count > toolResultCharacters:
            .toolResult(
                toolCallId: id, content: String(content.prefix(toolResultCharacters)) + "\n…")
        default:
            message
        }
    }

    // MARK: - Tools

    private func execute(
        _ call: ToolCallInfo, tools: [AgentToolDefinition], signal: CancellationToken
    ) async -> ToolResultMessage {
        switch CheckedToolCall(call, tools: tools) {
        case .refused(let result):
            return result
        case .ready(let tool, let arguments):
            Log.agent.debug("LFM3 Voice tool call \(call.name) args: \(call.argumentsJSON)")
            let (result, isError) = await tool.run(
                call, arguments: arguments, signal: signal, onUpdate: nil)
            return .create(
                toolCallId: call.id, toolName: call.name, result: result, isError: isError)
        }
    }

    // MARK: - The chat

    private func record(_ message: any AgentMessageProtocol & Sendable) {
        uncommitted.append(message)
    }

    private func commitPending() {
        guard !uncommitted.isEmpty, commit(uncommitted) else { return }
        uncommitted = []
    }
}
