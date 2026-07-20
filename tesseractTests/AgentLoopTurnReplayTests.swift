//
//  AgentLoopTurnReplayTests.swift
//  tesseractTests
//
//  Loop-level coverage for the turn replay breaker: an identical consecutive
//  assistant turn gets its tool calls refused (committed as corrective error
//  results, never re-executed), and a second replay stops the run. Everything
//  drives `agentLoop` through its existing interface — scripted generation in,
//  events and final context out — mirroring `AgentLoopToolExecutionTests`.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentLoopTurnReplayTests {

    // MARK: - Scripting helpers (mirrors AgentLoopToolExecutionTests)

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

    private func makeTool(
        name: String,
        onExecute: @escaping @Sendable () -> Void = {}
    ) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name,
            label: name,
            description: "test tool",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            execute: { _, _, _, _ in
                onExecute()
                return .text("ok")
            }
        )
    }

    private func toolCall(_ name: String, arguments: [String: JSONValue] = [:]) -> AgentGeneration {
        .toolCall(GenerationFixtures.toolCall(name: name, arguments: arguments))
    }

    /// The looping shape from the live incident: answer text plus a
    /// bookkeeping tool call.
    private func replayedTurn(arguments: [String: JSONValue] = ["v": .int(1)]) -> [AgentGeneration]
    {
        [.text("Good — logging it now."), toolCall("track", arguments: arguments)]
    }

    private func run(
        turns: [[AgentGeneration]],
        tools: [AgentToolDefinition],
        getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])? = nil
    ) async -> (events: [AgentEvent], context: AgentContext, llm: ScriptedLLM) {
        let llm = ScriptedLLM(turns: turns)
        let recorder = EventRecorder()
        var context = AgentContext(systemPrompt: "sys", messages: [], tools: tools)
        await agentLoop(
            prompts: [UserMessage(content: "hi")],
            context: &context,
            config: AgentLoopConfig(
                model: AgentModelRef(id: "loop-replay-test"),
                convertToLlm: defaultConvertToLlm,
                contextTransform: nil,
                getSteeringMessages: getSteeringMessages,
                getFollowUpMessages: nil
            ),
            generate: llm.generate,
            signal: nil,
            emit: { recorder.record($0) }
        )
        return (recorder.all, context, llm)
    }

    private nonisolated func committed<Message>(
        _ events: [AgentEvent], as _: Message.Type
    ) -> [Message] {
        events.compactMap { event in
            guard case .messageEnd(let message) = event else { return nil }
            return message as? Message
        }
    }

    private nonisolated func generationErrors(_ events: [AgentEvent]) -> [String] {
        events.compactMap { event in
            guard case .generationError(let message) = event else { return nil }
            return message
        }
    }

    // MARK: - First replay: refuse, correct, continue

    @Test func firstReplayRefusesExecutionAndCommitsCorrectiveResults() async throws {
        let executions = Locked(0)
        let track = makeTool(name: "track") { executions.value += 1 }

        let (events, context, llm) = await run(
            turns: [
                replayedTurn(),
                replayedTurn(),
                [.text("All set.")],
            ],
            tools: [track]
        )

        // The tool ran exactly once — the replay's call was refused.
        #expect(executions.value == 1)

        // Both turns committed a result; the replay's is the corrective error.
        let results = committed(events, as: ToolResultMessage.self)
        try #require(results.count == 2)
        #expect(results[0].isError == false)
        #expect(results[0].content.textContent == "ok")
        #expect(results[1].isError == true)
        #expect(results[1].content.textContent == TurnReplayGuard.replayRefusal)

        // The refusal is bound to the replay's own call id and reached the
        // follow-up generation.
        let assistants = committed(events, as: AssistantMessage.self)
        try #require(assistants.count == 3)
        #expect(results[1].toolCallId == assistants[1].toolCalls[0].id)
        #expect(llm.callCount == 3)
        #expect(
            llm.messages(forCall: 2).contains(
                .toolResult(
                    toolCallId: results[1].toolCallId,
                    content: TurnReplayGuard.replayRefusal)
            ))

        // The run recovered: no generationError, normal agentEnd.
        #expect(generationErrors(events).isEmpty)
        #expect((context.messages.last as? AssistantMessage)?.text == "All set.")
    }

    @Test func replayRefusalStillEmitsNoToolExecutionEvents() async throws {
        let track = makeTool(name: "track")

        let (events, _, _) = await run(
            turns: [
                replayedTurn(),
                replayedTurn(),
                [.text("done")],
            ],
            tools: [track]
        )

        let executionStarts = events.filter {
            if case .toolExecutionStart = $0 { return true } else { return false }
        }
        #expect(executionStarts.count == 1)
    }

    // MARK: - Second replay: stop the run

    @Test func secondReplayTerminatesTheRunWithAVisibleError() async throws {
        let executions = Locked(0)
        let track = makeTool(name: "track") { executions.value += 1 }

        let (events, _, llm) = await run(
            turns: [
                replayedTurn(),
                replayedTurn(),
                replayedTurn(),
                [.text("never generated")],
            ],
            tools: [track]
        )

        // Three generations, one execution, then the run stops — the scripted
        // fourth turn is never consumed.
        #expect(llm.callCount == 3)
        #expect(executions.value == 1)
        #expect(generationErrors(events) == [TurnReplayGuard.terminationNotice])

        // Terminal protocol: the error precedes turnEnd and agentEnd.
        let kinds = events.map { event -> String in
            switch event {
            case .generationError: return "generationError"
            case .turnEnd: return "turnEnd"
            case .agentEnd: return "agentEnd"
            default: return "other"
            }
        }.filter { $0 != "other" }
        #expect(kinds.suffix(3) == ["generationError", "turnEnd", "agentEnd"])
    }

    // MARK: - What is NOT a replay

    @Test func sameToolWithDifferentArgumentsExecutesEveryTurn() async throws {
        let executions = Locked(0)
        let track = makeTool(name: "track") { executions.value += 1 }

        let (events, _, _) = await run(
            turns: [
                replayedTurn(arguments: ["v": .int(1)]),
                replayedTurn(arguments: ["v": .int(2)]),
                replayedTurn(arguments: ["v": .int(3)]),
                [.text("done")],
            ],
            tools: [track]
        )

        #expect(executions.value == 3)
        #expect(generationErrors(events).isEmpty)
        #expect(committed(events, as: ToolResultMessage.self).allSatisfy { !$0.isError })
    }

    /// A steering interjection resets the chain: repeating the turn after the
    /// user spoke is a fresh request ("do it again"), not a replay.
    @Test func steeringMessageResetsTheReplayChain() async throws {
        let executions = Locked(0)
        let track = makeTool(name: "track") { executions.value += 1 }
        let steering = OneShotMessageQueue([UserMessage(content: "again please")])

        let (events, _, _) = await run(
            turns: [
                replayedTurn(),
                replayedTurn(),
                [.text("done")],
            ],
            tools: [track],
            getSteeringMessages: { steering.drain() }
        )

        // The steering message lands after the first turn's execution, so the
        // second (identical) turn follows a user interjection — it executes.
        #expect(executions.value == 2)
        #expect(generationErrors(events).isEmpty)
        #expect(committed(events, as: ToolResultMessage.self).allSatisfy { !$0.isError })
    }
}
