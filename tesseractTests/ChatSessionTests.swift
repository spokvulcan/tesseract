//
//  ChatSessionTests.swift
//  tesseractTests
//
//  Seam 1 of the chat rewrite (ADR-0024): scripted agent-event sequences drive
//  the Chat Session's fold; the tests assert the folded items, the Live Part
//  lifecycle (created at part start, committed at part end), and the run-phase
//  transitions. This one seam replaces the old reducer / transcript-projection /
//  dispatch-ordering test families.
//
//  Streams are scripted two ways: hand-built `AssistantMessageEvent`s for
//  precise lifecycle checks, and `AssistantPartsBuilder`-generated sequences so
//  the session is exercised by exactly what the agent loop emits.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ChatSessionTests {

    // MARK: - Fixtures

    private func makeSession(
        agent: Agent? = nil,
        store: InMemoryAgentConversationStore = InMemoryAgentConversationStore()
    ) -> ChatSession {
        ChatSession(
            agent: agent ?? makeNoOpAgent(modelID: "test-model"),
            conversationStore: store,
            arbiter: InMemoryInferenceArbiter(),
            liveMarkdownThrottle: .zero
        )
    }

    /// Feed one raw generation event through a builder and hand every produced
    /// stream event to the session — the exact shape the agent loop emits.
    private func drive(
        _ session: ChatSession, _ builder: inout AssistantPartsBuilder, _ event: AgentGeneration
    ) {
        if case .events(let events) = builder.ingest(event) {
            for streamEvent in events {
                session.handle(.messageUpdate(message: streamEvent.partial, event: streamEvent))
            }
        }
    }

    // MARK: - Live Part lifecycle

    @Test func textStreamCreatesLivePartAndCommitsOnEnd() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        #expect(session.runPhase == .streaming)

        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))
        #expect(session.liveMessage?.id == builder.messageID)
        #expect(session.livePart == nil)

        drive(session, &builder, .text("Hello"))
        let live = session.livePart
        #expect(live != nil)
        #expect(live?.kind == .text)
        #expect(live?.partIndex == 0)
        #expect(live?.messageID == builder.messageID)
        #expect(live?.displayText == "Hello")

        drive(session, &builder, .text(" world"))
        // Same box, appended — zero throttle publishes immediately.
        #expect(session.livePart === live)
        #expect(live?.displayText == "Hello world")
        // Committed items untouched during deltas.
        #expect(session.items.isEmpty)

        let final = builder.finalize(stopReason: .stop)
        session.handle(.messageUpdate(message: final, event: .done(reason: .stop, message: final)))
        session.handle(.messageEnd(message: final))

        #expect(session.livePart == nil)
        #expect(session.liveMessage == nil)
        #expect(session.items.count == 1)
        guard case .assistant(let committed) = session.items[0] else {
            Issue.record("expected committed assistant item")
            return
        }
        #expect(committed.id == builder.messageID)
        #expect(committed.text == "Hello world")
        #expect(committed.stopReason == .stop)
    }

    @Test func thinkingThenTextThenToolCallFoldsIntoOrderedParts() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        drive(session, &builder, .thinkStart)
        #expect(session.livePart?.kind == .thinking)
        drive(session, &builder, .thinking("pondering"))
        #expect(session.livePart?.displayText == "pondering")
        // No duration while the part is still streaming.
        #expect(session.thinkingDuration(messageID: builder.messageID, partIndex: 0) == nil)

        drive(session, &builder, .thinkEnd)
        // Part committed into the live message; box dropped.
        #expect(session.livePart == nil)
        #expect(session.liveMessage?.thinking == "pondering")
        // Duration measured between thinkingStart and thinkingEnd, keyed by
        // message id + part index.
        #expect(session.thinkingDuration(messageID: builder.messageID, partIndex: 0) != nil)

        drive(session, &builder, .text("Answer"))
        #expect(session.livePart?.kind == .text)
        #expect(session.livePart?.partIndex == 1)

        drive(session, &builder, .toolCall(GenerationFixtures.toolCall(name: "read_file")))
        // Text part closed by the tool call; tool calls never stream.
        #expect(session.livePart == nil)
        #expect(session.liveMessage?.content.count == 3)

        let final = builder.finalize(stopReason: builder.terminalStopReason)
        session.handle(
            .messageUpdate(message: final, event: .done(reason: .toolUse, message: final)))
        session.handle(.messageEnd(message: final))

        guard case .assistant(let committed) = session.items.first else {
            Issue.record("expected committed assistant item")
            return
        }
        #expect(committed.content.count == 3)
        guard case .thinking(let think) = committed.content[0],
            case .text(let text) = committed.content[1],
            case .toolCall(let call) = committed.content[2]
        else {
            Issue.record("expected ordered thinking/text/toolCall parts")
            return
        }
        #expect(think.thinking == "pondering")
        #expect(text.text == "Answer")
        #expect(call.name == "read_file")
        #expect(committed.stopReason == .toolUse)
    }

    // MARK: - Tool execution and results

    @Test func toolExecutionDrivesRunPhaseAndResultLinksByCallID() {
        let session = makeSession()
        session.handle(.agentStart)

        let call = ToolCallInfo(id: "call-1", name: "read_file", argumentsJSON: "{}")
        let assistant = AssistantMessage(content: "", toolCalls: [call], stopReason: .toolUse)
        session.handle(.messageEnd(message: assistant))

        session.handle(
            .toolExecutionStart(toolCallId: "call-1", toolName: "read_file", argsJSON: "{}"))
        #expect(session.runPhase == .executingTool("read_file"))

        let result = ToolResultMessage(
            toolCallId: "call-1", toolName: "read_file", content: [.text("file body")])
        session.handle(.messageStart(message: result))
        session.handle(.messageEnd(message: result))
        #expect(session.toolResult(for: "call-1")?.content.textContent == "file body")

        session.handle(
            .toolExecutionEnd(
                toolCallId: "call-1", toolName: "read_file",
                result: .text("file body"), isError: false))
        #expect(session.runPhase == .streaming)
    }

    // MARK: - Cancellation and errors

    @Test func cancellationPreservesPartialContent() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        drive(session, &builder, .text("partial ans"))

        // The loop's abort path: error(aborted) then messageEnd with the snapshot.
        let aborted = builder.snapshot(stopReason: .aborted)
        session.handle(
            .messageUpdate(message: aborted, event: .error(reason: .aborted, error: aborted)))
        session.handle(.messageEnd(message: aborted))
        session.handle(.agentEnd(messages: []))

        #expect(session.runPhase == .idle)
        #expect(session.livePart == nil)
        guard case .assistant(let committed) = session.items.first else {
            Issue.record("expected preserved partial assistant item")
            return
        }
        #expect(committed.text == "partial ans")
        #expect(committed.stopReason == .aborted)
    }

    @Test func emptyCancelledTurnIsDropped() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        session.handle(.messageStart(message: builder.snapshot()))
        let aborted = builder.snapshot(stopReason: .aborted)
        session.handle(.messageEnd(message: aborted))
        session.handle(.agentEnd(messages: []))

        #expect(session.items.isEmpty)
        #expect(session.liveMessage == nil)
    }

    @Test func generationErrorFeedsTheBannerSlot() {
        let session = makeSession()
        session.handle(.agentStart)
        session.handle(.generationError(message: "vision tower rejected the image"))
        #expect(session.error == "vision tower rejected the image")
    }

    // MARK: - Authoritative resyncs

    @Test func turnEndResyncsItemsFromContextSnapshot() {
        let session = makeSession()
        session.handle(.agentStart)

        let user = UserMessage(content: "question")
        let call = ToolCallInfo(id: "c1", name: "list", argumentsJSON: "{}")
        let assistant = AssistantMessage(content: "checking", toolCalls: [call])
        let result = ToolResultMessage(toolCallId: "c1", toolName: "list", content: [.text("ok")])
        let context: [any AgentMessageProtocol & Sendable] = [
            CoreMessage.user(user), assistant, result,
        ]

        session.handle(
            .turnEnd(message: assistant, toolResults: [result], contextMessages: context))

        #expect(session.items.count == 3)
        guard case .user(let u) = session.items[0] else {
            Issue.record("expected user item first")
            return
        }
        #expect(u.content == "question")
        #expect(session.toolResult(for: "c1")?.content.textContent == "ok")
    }

    @Test func compactionTransformResyncsAndRestoresPhase() {
        let session = makeSession()
        session.handle(.agentStart)

        session.handle(.contextTransformStart(reason: .compaction))
        #expect(session.runPhase == .transformingContext(.compaction))

        let summary = CompactionSummaryMessage(summary: "the gist", tokensBefore: 500)
        session.handle(
            .contextTransformEnd(reason: .compaction, didMutate: true, messages: [summary]))
        #expect(session.runPhase == .streaming)
        guard case .system(_, let text) = session.items.first else {
            Issue.record("expected system item for the compaction marker")
            return
        }
        #expect(text.contains("500 tokens"))
    }

    // MARK: - Full-script phases

    @Test func agentEndReturnsToIdleAndClearsLiveState() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        session.handle(.messageStart(message: builder.snapshot()))
        drive(session, &builder, .text("hi"))
        #expect(session.livePart != nil)

        let final = builder.finalize(stopReason: .stop)
        session.handle(.messageEnd(message: final))
        session.handle(.agentEnd(messages: [final]))

        #expect(session.runPhase == .idle)
        #expect(session.livePart == nil)
        #expect(session.liveMessage == nil)
        #expect(session.items.count == 1)
    }

    @Test func userMessageCommitsOnMessageEnd() {
        let session = makeSession()
        let user = CoreMessage.user(UserMessage(content: "hello there"))
        session.handle(.agentStart)
        session.handle(.messageStart(message: user))
        #expect(session.items.isEmpty)
        session.handle(.messageEnd(message: user))
        #expect(session.items.count == 1)
        guard case .user(let committed) = session.items[0] else {
            Issue.record("expected user item")
            return
        }
        #expect(committed.content == "hello there")
    }

    // MARK: - Edit & resend

    @Test func beginEditingTruncatesAndReturnsDraft() {
        let agent = makeNoOpAgent(modelID: "test-model")
        let session = makeSession(agent: agent)

        let firstUser = UserMessage(content: "first question")
        let answer = AssistantMessage(content: "first answer")
        let secondUser = UserMessage(content: "second question", images: [])
        agent.loadMessages([
            CoreMessage.user(firstUser), answer, CoreMessage.user(secondUser),
        ])
        session.handle(.agentStart)
        session.handle(
            .turnEnd(
                message: answer, toolResults: [],
                contextMessages: [
                    CoreMessage.user(firstUser), answer, CoreMessage.user(secondUser),
                ]))
        session.handle(.agentEnd(messages: []))

        let draft = session.beginEditingMessage(secondUser.id)
        #expect(draft?.text == "second question")
        #expect(session.items.count == 2)
        #expect(agent.context.messages.count == 2)
    }

    // MARK: - Live Part throttle

    @Test func livePartThrottleBoundsRepublishes() async throws {
        let live = LivePart(
            messageID: UUID(), partIndex: 0, kind: .text, throttle: .milliseconds(50))
        live.append("a")  // immediate: window open on creation? first append publishes when elapsed
        live.append("b")
        live.append("c")
        // Raw always has everything.
        #expect(live.raw == "abc")
        // displayText catches up once the trailing flush fires. Poll instead of
        // a single fixed sleep — the parallel test processes can starve the
        // flush task well past its deadline.
        let deadline = ContinuousClock.now + .seconds(3)
        while live.displayText != "abc", ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(live.displayText == "abc")
    }
}
