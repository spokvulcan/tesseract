//
//  AgentStateReducerTests.swift
//  tesseractTests
//
//  Tests the **Agent State Reducer** at its own seam: construct an `AgentState`,
//  feed it a scripted `[AgentEvent]`, assert the resulting state. No `Agent`,
//  loop, arbiter, or loaded model — the fold rules are verifiable in
//  milliseconds. Asserts external behavior (resulting `messages`/`streamMessage`/
//  `phase`/`pendingToolCalls`), never private switch structure.
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

    // MARK: - Tool execution

    /// A `.toolExecutionStart` records the call id in `pendingToolCalls` and
    /// drives the phase to `.executingTool(name)`.
    @Test func toolExecutionStartAddsPendingCallAndExecutingPhase() {
        let state = AgentState()
        reduce([.toolExecutionStart(toolCallId: "call-1", toolName: "read_file", argsJSON: "{}")], into: state)

        #expect(state.pendingToolCalls == ["call-1"])
        #expect(state.phase == .executingTool("read_file"))
    }

    /// `.toolExecutionEnd` drops its id from `pendingToolCalls`; the phase only
    /// returns to `.streaming` once the *last* outstanding call finishes.
    @Test func toolExecutionEndResumesStreamingOnlyWhenAllCallsDone() {
        let state = AgentState()
        reduce([
            .toolExecutionStart(toolCallId: "call-1", toolName: "read_file", argsJSON: "{}"),
            .toolExecutionStart(toolCallId: "call-2", toolName: "write_file", argsJSON: "{}"),
            .toolExecutionEnd(toolCallId: "call-1", toolName: "read_file",
                              result: AgentToolResult(content: []), isError: false),
        ], into: state)

        // One still outstanding — stays in the executing phase.
        #expect(state.pendingToolCalls == ["call-2"])
        #expect(state.phase == .executingTool("write_file"))

        reduce([
            .toolExecutionEnd(toolCallId: "call-2", toolName: "write_file",
                              result: AgentToolResult(content: []), isError: false),
        ], into: state)

        #expect(state.pendingToolCalls.isEmpty)
        #expect(state.phase == .streaming)
    }

    // MARK: - Message commit

    /// `.messageEnd` clears the progressive `streamMessage` and appends the
    /// committed assistant message.
    @Test func messageEndClearsStreamAndAppendsAssistantMessage() {
        let state = AgentState()
        state.streamMessage = AssistantMessage(content: "partial…")

        let committed = AssistantMessage(content: "final answer")
        reduce([.messageEnd(message: committed)], into: state)

        #expect(state.streamMessage == nil)
        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.content == "final answer")
    }

    /// The `hasContent` guard: an empty assistant message (no text, thinking, or
    /// tool calls — the shape produced by cancel/error paths) is dropped, but
    /// `streamMessage` is still cleared.
    @Test func messageEndDropsEmptyAssistantTurn() {
        let state = AgentState()
        state.streamMessage = AssistantMessage(content: "partial…")

        reduce([.messageEnd(message: AssistantMessage(content: ""))], into: state)

        #expect(state.streamMessage == nil)
        #expect(state.messages.isEmpty)
    }

    /// An assistant message that carries only tool calls (no text) still has
    /// content and is committed.
    @Test func messageEndKeepsAssistantTurnWithOnlyToolCalls() {
        let state = AgentState()
        let toolCall = ToolCallInfo(id: "call-1", name: "read_file", argumentsJSON: "{}")
        reduce([.messageEnd(message: AssistantMessage(content: "", toolCalls: [toolCall]))], into: state)

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
            AssistantMessage(content: "calling tool", toolCalls: [ToolCallInfo(id: "c1", name: "read_file", argumentsJSON: "{}")]),
            ToolResultMessage(toolCallId: "c1", toolName: "read_file", content: [.text("file body")]),
        ]
        reduce([.turnEnd(message: AssistantMessage(content: "calling tool"),
                         toolResults: [],
                         contextMessages: snapshot)], into: state)

        #expect(state.messages.count == 3)
        #expect(state.messages.last is ToolResultMessage)
    }

    // MARK: - Streaming

    /// `.messageUpdate` publishes the in-flight assistant message as
    /// `streamMessage`.
    @Test func messageUpdateSetsStreamMessage() {
        let state = AgentState()
        let partial = AssistantMessage(content: "stream…")
        reduce([.messageUpdate(message: partial,
                               streamDelta: AssistantStreamDelta(textDelta: "stream…", thinkingDelta: nil, toolCallDelta: nil))],
               into: state)

        #expect(state.streamMessage?.content == "stream…")
    }

    // MARK: - Lifecycle phase

    /// `.agentStart` moves the run into `.streaming`.
    @Test func agentStartEntersStreamingPhase() {
        let state = AgentState()
        reduce([.agentStart], into: state)
        #expect(state.phase == .streaming)
    }

    // MARK: - Context transform phase

    /// `.contextTransformStart` shows the transform phase; `.contextTransformEnd`
    /// resumes `.streaming` and applies the mutated messages.
    @Test func contextTransformStartThenEndDrivesPhaseAndAppliesMessages() {
        let state = AgentState()

        reduce([.contextTransformStart(reason: .compaction)], into: state)
        #expect(state.phase == .transformingContext(.compaction))

        let compacted: [any AgentMessageProtocol & Sendable] = [AssistantMessage(content: "summary")]
        reduce([.contextTransformEnd(reason: .compaction, didMutate: true, messages: compacted)], into: state)

        #expect(state.phase == .streaming)
        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.content == "summary")
    }

    /// A non-mutating `.contextTransformEnd` leaves `messages` untouched but
    /// still resumes `.streaming`.
    @Test func contextTransformEndWithoutMutationKeepsMessages() {
        let state = AgentState()
        state.messages = [UserMessage(content: "keep me")]

        reduce([.contextTransformEnd(reason: .compaction, didMutate: false, messages: nil)], into: state)

        #expect(state.phase == .streaming)
        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asUser?.content == "keep me")
    }

    // MARK: - Scripted sequence

    /// A full scripted turn folds to the expected settled state: start →
    /// stream → commit, ending streaming with the message committed and no
    /// residual stream.
    @Test func scriptedStreamingTurnSettlesToCommittedState() {
        let state = AgentState()
        let partial = AssistantMessage(content: "hel")
        let final = AssistantMessage(content: "hello")

        reduce([
            .agentStart,
            .messageStart(message: final),
            .messageUpdate(message: partial,
                           streamDelta: AssistantStreamDelta(textDelta: "hel", thinkingDelta: nil, toolCallDelta: nil)),
            .messageEnd(message: final),
        ], into: state)

        #expect(state.phase == .streaming)   // finishRun owns the .idle transition, not the fold
        #expect(state.streamMessage == nil)
        #expect(state.messages.count == 1)
        #expect(state.messages.first?.asAssistant?.content == "hello")
    }
}
