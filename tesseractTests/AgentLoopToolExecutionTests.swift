//
//  AgentLoopToolExecutionTests.swift
//  tesseractTests
//
//  Loop-level integration coverage for the agent double-loop's tool execution
//  (PRD #138). Each test drives `agentLoop` strictly through its existing
//  interface — a scripted generation stream in, the emitted event sequence and
//  final conversation state out — so the suite pins the orchestration (branch
//  handling, commit ordering, steering skips, cancellation, follow-ups)
//  without reaching into loop internals.
//
//  One branch is deliberately covered by proxy: the "Failed to parse tool
//  arguments as JSON" error result cannot be scripted through the loop,
//  because tool-call identity is minted by `AssistantMessageProjection` from a
//  structured `ToolCall` — `ToolArgumentNormalizer.encode` always produces a
//  valid JSON object, or falls back to "{}" (which the loop short-circuits to
//  empty args). The unencodable-arguments test pins that fallback end-to-end;
//  the parse-error branch itself is defensive and shares the single commit
//  step with every other branch.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentLoopToolExecutionTests {

    // MARK: - Scripting helpers

    /// Thread-safe sink for every `AgentEvent` the loop emits.
    private nonisolated final class EventRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [AgentEvent] = []

        func record(_ event: AgentEvent) {
            lock.lock(); events.append(event); lock.unlock()
        }

        var all: [AgentEvent] {
            lock.lock(); defer { lock.unlock() }
            return events
        }
    }

    /// A scripted LLM: each `generate` call pops the next turn's events and
    /// records the `LLMMessage` array it was handed, so tests can assert what
    /// the follow-up generation actually consumed.
    private nonisolated final class ScriptedLLM: @unchecked Sendable {
        private let lock = NSLock()
        private var turns: [[AgentGeneration]]
        private var received: [[LLMMessage]] = []

        init(turns: [[AgentGeneration]]) {
            self.turns = turns
        }

        var callCount: Int {
            lock.lock(); defer { lock.unlock() }
            return received.count
        }

        func messages(forCall index: Int) -> [LLMMessage] {
            lock.lock(); defer { lock.unlock() }
            return index < received.count ? received[index] : []
        }

        var generate: LLMGenerateFunction {
            { [self] _, messages, _, _ in
                lock.lock()
                received.append(messages)
                let turn = turns.isEmpty ? [] : turns.removeFirst()
                lock.unlock()
                return AsyncThrowingStream { continuation in
                    for event in turn { continuation.yield(event) }
                    continuation.finish()
                }
            }
        }
    }

    /// Consume-on-read message queue, mirroring how the coordinator hands
    /// steering / follow-up messages to the loop exactly once.
    private nonisolated final class OneShotMessageQueue: @unchecked Sendable {
        private let lock = NSLock()
        private var queued: [any AgentMessageProtocol & Sendable]

        init(_ messages: [any AgentMessageProtocol & Sendable]) {
            queued = messages
        }

        func drain() -> [any AgentMessageProtocol] {
            lock.lock(); defer { lock.unlock() }
            let out = queued
            queued = []
            return out
        }
    }

    /// Minimal thread-safe mutable box for observing tool executions.
    private nonisolated final class Locked<Value: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Value

        init(_ value: Value) { stored = value }

        var value: Value {
            get { lock.lock(); defer { lock.unlock() }; return stored }
            set { lock.lock(); stored = newValue; lock.unlock() }
        }
    }

    private func makeConfig(
        getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])? = nil,
        getFollowUpMessages: (@Sendable () async -> [any AgentMessageProtocol])? = nil
    ) -> AgentLoopConfig {
        AgentLoopConfig(
            model: AgentModelRef(id: "loop-tool-execution-test"),
            convertToLlm: defaultConvertToLlm,
            contextTransform: nil,
            getSteeringMessages: getSteeringMessages,
            getFollowUpMessages: getFollowUpMessages
        )
    }

    private func makeTool(
        name: String,
        required: [String] = [],
        execute:
            @escaping @Sendable (
                String, [String: JSONValue], CancellationToken?, ToolProgressCallback?
            ) async throws -> AgentToolResult = { _, _, _, _ in .text("ok") }
    ) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name,
            label: name,
            description: "test tool",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: required),
            execute: execute
        )
    }

    private func toolCall(_ name: String, arguments: [String: JSONValue] = [:]) -> AgentGeneration {
        .toolCall(GenerationFixtures.toolCall(name: name, arguments: arguments))
    }

    /// Runs the loop over one user prompt with everything recorded.
    private func run(
        turns: [[AgentGeneration]],
        tools: [AgentToolDefinition],
        signal: CancellationToken? = nil,
        getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])? = nil,
        getFollowUpMessages: (@Sendable () async -> [any AgentMessageProtocol])? = nil
    ) async -> (events: [AgentEvent], context: AgentContext, llm: ScriptedLLM) {
        let llm = ScriptedLLM(turns: turns)
        let recorder = EventRecorder()
        var context = AgentContext(systemPrompt: "sys", messages: [], tools: tools)
        await agentLoop(
            prompts: [UserMessage(content: "hi")],
            context: &context,
            config: makeConfig(
                getSteeringMessages: getSteeringMessages,
                getFollowUpMessages: getFollowUpMessages
            ),
            generate: llm.generate,
            signal: signal,
            emit: { recorder.record($0) }
        )
        return (recorder.all, context, llm)
    }

    // MARK: - Event projections

    /// Compact, order-preserving projection of an event for golden sequences.
    private nonisolated func kind(_ event: AgentEvent) -> String {
        switch event {
        case .agentStart: return "agentStart"
        case .agentEnd: return "agentEnd"
        case .generationError: return "generationError"
        case .turnStart: return "turnStart"
        case .turnEnd: return "turnEnd"
        case .contextTransformStart: return "contextTransformStart"
        case .contextTransformEnd: return "contextTransformEnd"
        case .messageStart(let message): return "messageStart(\(messageKind(message)))"
        case .messageUpdate: return "messageUpdate"
        case .messageEnd(let message): return "messageEnd(\(messageKind(message)))"
        case .malformedToolCall: return "malformedToolCall"
        case .toolExecutionStart(_, let name, _): return "toolExecutionStart(\(name))"
        case .toolExecutionUpdate: return "toolExecutionUpdate"
        case .toolExecutionEnd(_, let name, _, _): return "toolExecutionEnd(\(name))"
        }
    }

    private nonisolated func messageKind(_ message: any AgentMessageProtocol) -> String {
        switch message {
        case is UserMessage: return "user"
        case is AssistantMessage: return "assistant"
        case is ToolResultMessage: return "toolResult"
        default: return "other"
        }
    }

    private nonisolated func kinds(_ events: [AgentEvent]) -> [String] {
        events.map { kind($0) }
    }

    /// Committed messages of one concrete type, in `messageEnd` order.
    private nonisolated func committed<Message>(
        _ events: [AgentEvent], as _: Message.Type
    ) -> [Message] {
        events.compactMap { event in
            guard case .messageEnd(let message) = event else { return nil }
            return message as? Message
        }
    }

    // MARK: - Error branches

    @Test func unknownToolCommitsErrorResultThroughTheLoop() async throws {
        let executed = Locked(false)
        let echo = makeTool(name: "echo") { _, _, _, _ in
            executed.value = true
            return .text("ok")
        }

        let (events, context, llm) = await run(
            turns: [
                [toolCall("missing_tool")],
                [.text("done")],
            ],
            tools: [echo]
        )

        // Exact event sequence: no toolExecutionStart/End — the error result is
        // committed directly, then the follow-up turn runs.
        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                "turnStart",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                "agentEnd",
            ])
        #expect(executed.value == false)

        // The committed result carries the branch's content and error flag,
        // bound to the assistant's tool-call id.
        let result = try #require(committed(events, as: ToolResultMessage.self).first)
        let assistant = try #require(committed(events, as: AssistantMessage.self).first)
        #expect(result.toolCallId == assistant.toolCalls[0].id)
        #expect(result.toolName == "missing_tool")
        #expect(result.isError == true)
        #expect(result.content.textContent == "Unknown tool: missing_tool")

        // Final conversation state: user, assistant(toolCall), error result,
        // follow-up assistant — and the follow-up generation consumed the result.
        #expect(context.messages.count == 4)
        #expect((context.messages[2] as? ToolResultMessage) == result)
        #expect(llm.callCount == 2)
        #expect(
            llm.messages(forCall: 1).contains(
                .toolResult(toolCallId: result.toolCallId, content: "Unknown tool: missing_tool")
            ))
    }

    @Test func missingRequiredParametersCommitsErrorResultThroughTheLoop() async throws {
        let executed = Locked(false)
        let search = makeTool(name: "search", required: ["query", "limit"]) { _, _, _, _ in
            executed.value = true
            return .text("ok")
        }

        let (events, context, llm) = await run(
            turns: [
                [toolCall("search")],  // no arguments at all
                [.text("done")],
            ],
            tools: [search]
        )

        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                "turnStart",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                "agentEnd",
            ])
        #expect(executed.value == false)

        let result = try #require(committed(events, as: ToolResultMessage.self).first)
        let assistant = try #require(committed(events, as: AssistantMessage.self).first)
        #expect(result.toolCallId == assistant.toolCalls[0].id)
        #expect(result.toolName == "search")
        #expect(result.isError == true)
        #expect(result.content.textContent == "Missing required parameters: query, limit")

        #expect(context.messages.count == 4)
        #expect((context.messages[2] as? ToolResultMessage) == result)
        #expect(llm.callCount == 2)
    }

    /// The closest scriptable relative of the "unparseable arguments" branch
    /// (see the header note): arguments `JSONEncoder` cannot encode collapse to
    /// "{}" in the projection, and the loop executes the tool with empty args
    /// rather than committing a parse-error result.
    @Test func unencodableArgumentsExecuteToolWithEmptyArgs() async throws {
        let seenArgs = Locked<[String: JSONValue]?>(nil)
        let echo = makeTool(name: "echo") { _, args, _, _ in
            seenArgs.value = args
            return .text("ok")
        }

        let (events, _, _) = await run(
            turns: [
                [toolCall("echo", arguments: ["x": .double(.infinity)])],
                [.text("done")],
            ],
            tools: [echo]
        )

        // The tool genuinely executed — with empty args — and its result was
        // committed as a success, not routed to any error branch.
        #expect(kinds(events).contains("toolExecutionStart(echo)"))
        #expect(seenArgs.value == [:])
        let result = try #require(committed(events, as: ToolResultMessage.self).first)
        #expect(result.isError == false)
        #expect(result.content.textContent == "ok")
    }

    // MARK: - Happy round-trip

    @Test func happyToolRoundTripCommitsResultAndFeedsFollowUpGeneration() async throws {
        let seenArgs = Locked<[String: JSONValue]?>(nil)
        let echo = makeTool(name: "echo", required: ["value"]) { _, args, _, _ in
            seenArgs.value = args
            return .text("tool says 7")
        }

        let (events, context, llm) = await run(
            turns: [
                [.text("checking"), toolCall("echo", arguments: ["value": .int(7)])],
                [.text("done")],
            ],
            tools: [echo]
        )

        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageUpdate",
                "messageEnd(assistant)",
                "toolExecutionStart(echo)", "toolExecutionEnd(echo)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                "turnStart",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                "agentEnd",
            ])

        // The tool received the round-tripped arguments.
        #expect(seenArgs.value == ["value": .int(7)])

        // The committed result is a success bound to the assistant's call id.
        let result = try #require(committed(events, as: ToolResultMessage.self).first)
        let assistant = try #require(committed(events, as: AssistantMessage.self).first)
        #expect(result.toolCallId == assistant.toolCalls[0].id)
        #expect(result.isError == false)
        #expect(result.content.textContent == "tool says 7")

        // Final context: user, assistant(toolCall), result, follow-up assistant.
        #expect(context.messages.count == 4)
        #expect((context.messages[1] as? AssistantMessage)?.toolCalls.count == 1)
        #expect((context.messages[2] as? ToolResultMessage) == result)
        #expect((context.messages[3] as? AssistantMessage)?.content == "done")

        // The follow-up generation consumed the committed result.
        #expect(llm.callCount == 2)
        #expect(
            llm.messages(forCall: 1).contains(
                .toolResult(toolCallId: result.toolCallId, content: "tool says 7")
            ))
    }

    // MARK: - Steering skip

    @Test func steeringSkipsRemainingToolCallsAndEntersSteeringTurn() async throws {
        let firstExecuted = Locked(false)
        let secondExecuted = Locked(false)
        let first = makeTool(name: "first") { _, _, _, _ in
            firstExecuted.value = true
            return .text("first done")
        }
        let second = makeTool(name: "second") { _, _, _, _ in
            secondExecuted.value = true
            return .text("second done")
        }
        let steering = OneShotMessageQueue([UserMessage(content: "wait")])

        let (events, context, llm) = await run(
            turns: [
                [toolCall("first"), toolCall("second")],
                [.text("ok")],
            ],
            tools: [first, second],
            getSteeringMessages: { steering.drain() }
        )

        // First executes; second is committed as skipped with no execution
        // events; the steering message then opens the next turn.
        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageUpdate",
                "messageEnd(assistant)",
                "toolExecutionStart(first)", "toolExecutionEnd(first)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                "turnStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                "agentEnd",
            ])
        #expect(firstExecuted.value == true)
        #expect(secondExecuted.value == false)

        let results = committed(events, as: ToolResultMessage.self)
        try #require(results.count == 2)
        #expect(results[0].toolName == "first")
        #expect(results[0].isError == false)
        #expect(results[0].content.textContent == "first done")
        #expect(results[1].toolName == "second")
        #expect(results[1].isError == true)
        #expect(results[1].content.textContent == "Skipped due to queued user message")

        // Final context: user, assistant, result, skipped result, steering
        // user message, steering-turn assistant.
        #expect(context.messages.count == 6)
        #expect((context.messages[3] as? ToolResultMessage) == results[1])
        #expect((context.messages[4] as? UserMessage)?.content == "wait")

        // The steering turn's generation saw both committed results and the
        // interjection.
        #expect(llm.callCount == 2)
        let followUpMessages = llm.messages(forCall: 1)
        #expect(followUpMessages.contains(.user(content: "wait", images: [])))
        #expect(
            followUpMessages.contains(
                .toolResult(
                    toolCallId: results[1].toolCallId,
                    content: "Skipped due to queued user message")
            ))
    }

    // MARK: - Cancellation

    @Test func cancellationMidToolSequenceCommitsFinishedResultAndEndsRun() async throws {
        let signal = CancellationToken()
        let secondExecuted = Locked(false)
        let first = makeTool(name: "first") { _, _, token, _ in
            token?.cancel()
            return .text("first done")
        }
        let second = makeTool(name: "second") { _, _, _, _ in
            secondExecuted.value = true
            return .text("second done")
        }

        let (events, context, llm) = await run(
            turns: [
                [toolCall("first"), toolCall("second")]
            ],
            tools: [first, second],
            signal: signal
        )

        // The in-flight tool's result is still committed; the remaining call is
        // dropped (not executed, not committed as skipped) and the loop ends
        // with its terminal events, without a follow-up generation.
        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageUpdate",
                "messageEnd(assistant)",
                "toolExecutionStart(first)", "toolExecutionEnd(first)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                "agentEnd",
            ])
        #expect(secondExecuted.value == false)

        let results = committed(events, as: ToolResultMessage.self)
        try #require(results.count == 1)
        #expect(results[0].toolName == "first")
        #expect(results[0].content.textContent == "first done")

        // Final context: user, assistant, the one finished result.
        #expect(context.messages.count == 3)
        #expect((context.messages[2] as? ToolResultMessage) == results[0])
        #expect(llm.callCount == 1)
    }

    // MARK: - Golden event order

    /// One golden sequence for a scripted multi-turn run — tool turn, text
    /// turn, then a queued follow-up turn — so any future reordering of the
    /// loop's emits fails loudly.
    @Test func goldenEventOrderForScriptedMultiTurnRun() async {
        let echo = makeTool(name: "echo")
        let followUps = OneShotMessageQueue([UserMessage(content: "follow up")])

        let (events, context, llm) = await run(
            turns: [
                [.text("t"), toolCall("echo")],
                [.text("answer")],
                [.text("follow-up answer")],
            ],
            tools: [echo],
            getFollowUpMessages: { followUps.drain() }
        )

        #expect(
            kinds(events) == [
                "agentStart",
                "messageStart(user)", "messageEnd(user)",
                // Turn 1 (first turn: no turnStart): tool call + commit.
                "messageStart(assistant)", "messageUpdate", "messageUpdate",
                "messageEnd(assistant)",
                "toolExecutionStart(echo)", "toolExecutionEnd(echo)",
                "messageStart(toolResult)", "messageEnd(toolResult)",
                "turnEnd",
                // Turn 2: the follow-up generation over the tool result.
                "turnStart",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                // Turn 3: the queued follow-up message opens an outer-loop round.
                "turnStart",
                "messageStart(user)", "messageEnd(user)",
                "messageStart(assistant)", "messageUpdate", "messageEnd(assistant)",
                "turnEnd",
                "agentEnd",
            ])

        // Final context: user, assistant, result, assistant, follow-up user,
        // assistant — three generations total.
        #expect(context.messages.count == 6)
        #expect(llm.callCount == 3)
    }
}
