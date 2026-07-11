//
//  AgentStateReducerTests.swift
//  tesseractTests
//
//  Tests the **Agent State Reducer** at its own seam: construct an `AgentState`,
//  feed it a scripted `[AgentEvent]`, assert the resulting message log. No
//  `Agent`, loop, arbiter, or loaded model — the fold rules are verifiable in
//  milliseconds. The reducer owns the message fold only; run-presentation
//  detail belongs to the Chat Session's fold (ADR-0024) and the busy bit to
//  the run envelope.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentStateReducerTests {

    private func reduce(_ events: [AgentEvent], into state: AgentState) {
        for event in events {
            AgentStateReducer.reduce(event, into: state)
        }
    }

    // MARK: - Message commit

    /// `.messageEnd` appends the committed assistant message.
    @Test func messageEndAppendsAssistantMessage() {
        let state = AgentState()
        reduce([.messageEnd(message: AssistantMessage(content: "final answer"))], into: state)

        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.text == "final answer")
    }

    /// The `hasContent` guard: an empty assistant message (no text, thinking, or
    /// tool calls — the shape produced by cancel/error paths) is dropped.
    @Test func messageEndDropsEmptyAssistantTurn() {
        let state = AgentState()
        reduce([.messageEnd(message: AssistantMessage(content: ""))], into: state)

        #expect(state.messages.isEmpty)
    }

    /// An assistant message that carries only tool calls (no text) still has
    /// content and is committed.
    @Test func messageEndKeepsAssistantTurnWithOnlyToolCalls() {
        let state = AgentState()
        let toolCall = ToolCallInfo(id: "call-1", name: "read_file", argumentsJSON: "{}")
        reduce(
            [.messageEnd(message: AssistantMessage(content: "", toolCalls: [toolCall]))],
            into: state)

        #expect(state.messages.count == 1)
    }

    // MARK: - Turn commit

    /// `.turnEnd` authoritatively replaces `messages` from the loop's context
    /// snapshot — which carries tool results the streaming path never emitted.
    @Test func turnEndFullyReplacesMessagesFromContextSnapshot() {
        let state = AgentState()
        state.messages = [UserMessage(content: "stale")]

        let snapshot: [any AgentMessageProtocol & Sendable] = [
            UserMessage(content: "hi"),
            AssistantMessage(
                content: "calling tool",
                toolCalls: [ToolCallInfo(id: "c1", name: "read_file", argumentsJSON: "{}")]),
            ToolResultMessage(
                toolCallId: "c1", toolName: "read_file", content: [.text("file body")]),
        ]
        reduce(
            [
                .turnEnd(
                    message: AssistantMessage(content: "calling tool"),
                    toolResults: [],
                    contextMessages: snapshot)
            ], into: state)

        #expect(state.messages.count == 3)
        #expect(state.messages.last is ToolResultMessage)
    }

    // MARK: - Context transform

    /// A mutating `.contextTransformEnd` replaces the log with the compacted
    /// messages.
    @Test func contextTransformEndAppliesMutatedMessages() {
        let state = AgentState()
        state.messages = [UserMessage(content: "long history")]

        let compacted: [any AgentMessageProtocol & Sendable] = [
            AssistantMessage(content: "summary")
        ]
        reduce(
            [.contextTransformEnd(reason: .compaction, didMutate: true, messages: compacted)],
            into: state)

        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.text == "summary")
    }

    /// A non-mutating `.contextTransformEnd` leaves `messages` untouched.
    @Test func contextTransformEndWithoutMutationKeepsMessages() {
        let state = AgentState()
        state.messages = [UserMessage(content: "keep me")]

        reduce(
            [.contextTransformEnd(reason: .compaction, didMutate: false, messages: nil)],
            into: state)

        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asUser?.content == "keep me")
    }

    // MARK: - Presentation events

    /// Run-presentation events fold to nothing here — the busy bit belongs to
    /// the run envelope and the live-stream detail to the Chat Session.
    @Test func presentationEventsLeaveTheMessageLogAndBusyBitAlone() {
        let state = AgentState()
        state.messages = [UserMessage(content: "hi")]
        let partial = AssistantMessage(content: "stream…")

        reduce(
            [
                .agentStart,
                .messageStart(message: partial),
                .messageUpdate(
                    message: partial,
                    event: .textDelta(contentIndex: 0, delta: "stream…", partial: partial)),
                .toolExecutionStart(toolCallId: "c1", toolName: "read_file", argsJSON: "{}"),
                .toolExecutionEnd(
                    toolCallId: "c1", toolName: "read_file",
                    result: AgentToolResult(content: []), isError: false),
                .agentEnd(messages: []),
            ], into: state)

        #expect(state.messages.count == 1)
        #expect(state.isBusy == false)
    }

    // MARK: - Scripted sequence

    /// A full scripted turn folds to the expected settled log: start →
    /// stream → commit, ending with the message committed.
    @Test func scriptedStreamingTurnSettlesToCommittedState() {
        let state = AgentState()
        let partial = AssistantMessage(content: "hel")
        let final = AssistantMessage(content: "hello")

        reduce(
            [
                .agentStart,
                .messageStart(message: final),
                .messageUpdate(
                    message: partial,
                    event: .textDelta(contentIndex: 0, delta: "hel", partial: partial)),
                .messageEnd(message: final),
            ], into: state)

        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.text == "hello")
    }
}
