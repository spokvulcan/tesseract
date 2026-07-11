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
        store: InMemoryAgentConversationStore = InMemoryAgentConversationStore(),
        arbiter: InMemoryInferenceArbiter = InMemoryInferenceArbiter(),
        restoreComposerDraft: @MainActor @escaping (String, [ImageAttachment]) -> Void = { _, _ in }
    ) -> ChatSession {
        ChatSession(
            agent: agent ?? makeNoOpAgent(modelID: "test-model"),
            conversationStore: store,
            arbiter: arbiter,
            restoreComposerDraft: restoreComposerDraft,
            liveMarkdownThrottle: .zero
        )
    }

    private func waitUntilIdle(
        _ session: ChatSession, timeout: Duration = .seconds(3)
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while session.isGenerating {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now >= deadline {
                Issue.record("ChatSession did not become idle within timeout")
                return
            }
        }
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
        // Text part closed by the tool call; no Live Part exists for tool
        // calls (this parse had no preceding deltas — the atomic pair).
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

    // MARK: - Streaming tool calls (Open Tool Call + Tool Clock)

    @Test func writingToolPhaseAndToolClockSpanWritingToExecution() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        // Pre-name fragment: still plain streaming, no row, no clock.
        drive(session, &builder, .toolCallDelta(name: nil, argumentsDelta: #"{"na"#))
        #expect(session.runPhase == .streaming)
        #expect(session.liveMessage?.content.isEmpty != false)

        // Name-lock: the Open Tool Call is born — row in the live message,
        // writing phase entered, Tool Clock anchored.
        drive(
            session, &builder,
            .toolCallDelta(name: "read_file", argumentsDelta: #"me": "read_file", "arg"#))
        #expect(session.runPhase == .writingTool("read_file"))
        guard case .toolCall(let open) = session.liveMessage?.content.first else {
            Issue.record("expected the Open Tool Call in the live message")
            return
        }
        #expect(open.name == "read_file")
        let anchor = session.toolStartInstant(for: open.id)
        #expect(anchor != nil)
        // No Live Part for tool calls; deltas are a rendering no-op.
        #expect(session.livePart == nil)

        drive(
            session, &builder,
            .toolCallDelta(name: "read_file", argumentsDelta: #"uments": {}}"#))
        #expect(session.runPhase == .writingTool("read_file"))

        // Parse commits in place: same id, back to streaming.
        drive(session, &builder, .toolCall(GenerationFixtures.toolCall(name: "read_file")))
        #expect(session.runPhase == .streaming)
        guard case .toolCall(let committed) = session.liveMessage?.content.first else {
            Issue.record("expected the committed tool-call part")
            return
        }
        #expect(committed.id == open.id)

        // Execution start must not reset the clock; execution end freezes it.
        session.handle(
            .toolExecutionStart(toolCallId: committed.id, toolName: "read_file", argsJSON: "{}"))
        #expect(session.runPhase == .executingTool("read_file"))
        #expect(session.toolStartInstant(for: committed.id) == anchor)

        session.handle(
            .toolExecutionEnd(
                toolCallId: committed.id, toolName: "read_file",
                result: .text("ok"), isError: false))
        #expect(session.toolDuration(for: committed.id) != nil)
        #expect(session.toolStartInstant(for: committed.id) == nil)
    }

    @Test func abandonedToolCallVanishesAndPhaseRecovers() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        drive(
            session, &builder,
            .toolCallDelta(name: "read_file", argumentsDelta: #"{"name": "read_file""#))
        #expect(session.runPhase == .writingTool("read_file"))

        // Parser finalize reclassified the unclosed block as raw text: the
        // retraction rides the next part's snapshot — row gone, phase back.
        drive(session, &builder, .text(#"<tool_call>{"name": "read_file""#))
        #expect(session.runPhase == .streaming)
        let hasToolCall = session.liveMessage?.content.contains {
            if case .toolCall = $0 { return true } else { return false }
        }
        #expect(hasToolCall == false)
    }

    @Test func malformedCloseLeavesTheWritingPhasePromptly() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        drive(
            session, &builder,
            .toolCallDelta(name: "read_file", argumentsDelta: #"{"name": "read_file", bad"#))
        #expect(session.runPhase == .writingTool("read_file"))

        // The block closed unparseable: the builder retracts with no stream
        // event of its own; the loop's distinct malformed event resets the
        // phase immediately, not at the next part boundary.
        guard
            case .malformed(let raw) = builder.ingest(
                .malformedToolCall(#"{"name": "read_file", bad"#))
        else {
            Issue.record("expected malformed step")
            return
        }
        session.handle(.malformedToolCall(raw: raw))
        #expect(session.runPhase == .streaming)
    }

    @Test func unclosedToolCallAtTerminalLeavesNoTraceInTheCommit() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        drive(session, &builder, .text("Checking."))
        drive(
            session, &builder,
            .toolCallDelta(name: "read_file", argumentsDelta: #"{"name": "read_file""#))
        #expect(session.runPhase == .writingTool("read_file"))

        // Stream ends with the call unclosed (e.g. token limit): terminal
        // close retracts it; done resets the phase; the commit carries text only.
        for event in builder.closeForTerminal() {
            session.handle(.messageUpdate(message: event.partial, event: event))
        }
        let final = builder.finalize(stopReason: builder.terminalStopReason)
        session.handle(.messageUpdate(message: final, event: .done(reason: .stop, message: final)))
        session.handle(.messageEnd(message: final))
        session.handle(.agentEnd(messages: []))

        #expect(session.runPhase == .idle)
        guard case .assistant(let committed) = session.items.first else {
            Issue.record("expected committed assistant item")
            return
        }
        #expect(committed.text == "Checking.")
        #expect(committed.toolCalls.isEmpty)
        #expect(committed.stopReason == .stop)
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

    // MARK: - Pending Row

    /// `sendMessage` raises the Pending Row synchronously — the transcript
    /// shows the user message while the run is still queued behind the lease —
    /// and the event-spine commit of the same message lowers it.
    @Test func sendMessageRaisesPendingRowUntilTheUserCommit() async throws {
        var restored: [(String, Int)] = []
        let session = makeSession(restoreComposerDraft: { text, images in
            restored.append((text, images.count))
        })

        session.sendMessage("hello cold start")
        #expect(session.pendingUserMessage?.content == "hello cold start")

        try await waitUntilIdle(session)
        // The commit (or the turn-end resync) replaced the Pending Row with
        // the committed item; nothing was restored to the composer.
        #expect(session.pendingUserMessage == nil)
        #expect(restored.isEmpty)
        #expect(
            session.items.contains { item in
                if case .user(let msg) = item { return msg.content == "hello cold start" }
                return false
            })
    }

    /// A run that dies before the user message ever enters the agent context
    /// (cancel while queued, load failure) lowers the Pending Row and restores
    /// its content to the composer — the transcript never shows a message the
    /// agent doesn't have.
    @Test func pendingRowRestoresComposerDraftWhenRunDiesBeforeCommit() async throws {
        let arbiter = InMemoryInferenceArbiter()
        arbiter.ensureLoadedError = AgentEngineError.modelNotDownloaded(
            modelID: "chat-session-nonexistent-model")
        var restored: [(String, Int)] = []
        let session = makeSession(
            arbiter: arbiter,
            restoreComposerDraft: { text, images in restored.append((text, images.count)) })

        session.sendMessage("doomed message")
        #expect(session.pendingUserMessage != nil)

        try await waitUntilIdle(session)
        #expect(session.pendingUserMessage == nil)
        #expect(restored.count == 1)
        #expect(restored.first?.0 == "doomed message")
        #expect(session.items.isEmpty)
    }

    /// Cancel while the run sits queued behind the lease — the queued-wait
    /// analogue of the load-failure path: same restore (text *and* images),
    /// no ghost message.
    @Test func pendingRowRestoresComposerDraftOnCancelWhileQueued() async throws {
        let arbiter = InMemoryInferenceArbiter()
        arbiter.leaseDelay = .seconds(10)
        var restored: [(String, [ImageAttachment])] = []
        let session = makeSession(
            arbiter: arbiter,
            restoreComposerDraft: { text, images in restored.append((text, images)) })

        let image = ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
        session.sendMessage("cancelled while queued", images: [image])
        #expect(session.pendingUserMessage != nil)
        session.cancelGeneration()

        try await waitUntilIdle(session)
        #expect(session.pendingUserMessage == nil)
        #expect(restored.count == 1)
        #expect(restored.first?.0 == "cancelled while queued")
        #expect(restored.first?.1 == [image])
        #expect(session.items.isEmpty)
    }

    /// The user commit lowers the Pending Row through the event fold, so a
    /// later settle has nothing to restore.
    @Test func userCommitLowersPendingRowThroughTheFold() {
        let arbiter = InMemoryInferenceArbiter()
        arbiter.leaseDelay = .seconds(10)
        let session = makeSession(arbiter: arbiter)

        session.sendMessage("hello")
        #expect(session.pendingUserMessage != nil)

        session.handle(.messageEnd(message: CoreMessage.user(UserMessage(content: "hello"))))
        #expect(session.pendingUserMessage == nil)
        session.cancelGeneration()
    }

    /// Slash commands never raise a Pending Row — an unknown command (name +
    /// arguments, so the parser can't read it as partial typing) is an error,
    /// not a message.
    @Test func slashCommandDoesNotRaisePendingRow() {
        let session = makeSession()
        session.sendMessage("/definitely-not-a-command with args")
        #expect(session.pendingUserMessage == nil)
        #expect(session.error != nil)
    }

    // MARK: - Waiting Row

    /// The Waiting Row shows from send (queued, cold start) through turn
    /// prefill, and hides the moment the first delta opens a Live Part.
    @Test func waitingRowSpansQueueAndPrefillUntilFirstDelta() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        #expect(session.showsWaitingRow == false)

        // Queued behind the lease (cold start): isGenerating is up eagerly,
        // no agent events yet.
        session.agentRun.markStarted()
        #expect(session.showsWaitingRow == true)

        session.handle(.agentStart)
        #expect(session.showsWaitingRow == true)

        // Message opened, still no content — prefill continues.
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))
        #expect(session.showsWaitingRow == true)

        // First delta: the Live Part takes over.
        drive(session, &builder, .thinkStart)
        drive(session, &builder, .thinking("pondering"))
        #expect(session.showsWaitingRow == false)
    }

    /// Between parts and during tool execution the Waiting Row stays hidden
    /// (the live message has content; tool rows own their spinners) — it
    /// returns only for the next turn's prefill, after the tool batch ends.
    @Test func waitingRowHiddenBetweenPartsAndDuringToolsShownForNextTurnPrefill() {
        let session = makeSession()
        var builder = AssistantPartsBuilder()

        session.agentRun.markStarted()
        session.handle(.agentStart)
        let start = builder.snapshot()
        session.handle(.messageStart(message: start))
        session.handle(.messageUpdate(message: start, event: .start(partial: start)))

        // Thinking part streams and commits — the between-parts gap.
        drive(session, &builder, .thinkStart)
        drive(session, &builder, .thinking("pondering"))
        drive(session, &builder, .thinkEnd)
        #expect(session.livePart == nil)
        #expect(session.showsWaitingRow == false)

        // Turn ends in a tool call; execution runs under the tool row's spinner.
        let call = ToolCallInfo(id: "call-1", name: "read_file", argumentsJSON: "{}")
        let assistant = AssistantMessage(
            content: "", toolCalls: [call], stopReason: .toolUse)
        session.handle(.messageEnd(message: assistant))
        session.handle(
            .toolExecutionStart(toolCallId: "call-1", toolName: "read_file", argsJSON: "{}"))
        #expect(session.showsWaitingRow == false)

        // Tool batch done → the next turn's prefill: the Waiting Row returns.
        session.handle(
            .toolExecutionEnd(
                toolCallId: "call-1", toolName: "read_file",
                result: .text("ok"), isError: false))
        #expect(session.showsWaitingRow == true)

        // Run over → hidden.
        session.agentRun.finish()
        session.handle(.agentEnd(messages: []))
        #expect(session.showsWaitingRow == false)
    }

    // MARK: - Conversation switch

    @Test func loadConversationAdoptsTheStoredTranscript() {
        let user = UserMessage(content: "stored question")
        let answer = AssistantMessage(content: "stored answer")
        let conversation = AgentConversation(messages: [CoreMessage.user(user), answer])
        let store = InMemoryAgentConversationStore(seed: [conversation])
        let agent = makeNoOpAgent(modelID: "test-model")
        let session = makeSession(agent: agent, store: store)

        session.loadConversation(conversation.id)

        #expect(session.items.count == 2)
        #expect(agent.context.messages.count == 2)
        #expect(store.currentConversation?.id == conversation.id)
    }

    @Test func newConversationPersistsTheOutgoingTranscript() throws {
        let store = InMemoryAgentConversationStore()
        let agent = makeNoOpAgent(modelID: "test-model")
        let session = makeSession(agent: agent, store: store)
        store.createNew()
        let outgoingID = try #require(store.currentConversation).id
        agent.loadMessages([CoreMessage.user(UserMessage(content: "keep me"))])

        session.newConversation()

        #expect(store.currentConversation?.id != outgoingID)
        #expect(session.items.isEmpty)
        #expect(agent.context.messages.isEmpty)

        // The outgoing transcript survived the switch: it loads back whole.
        session.loadConversation(outgoingID)
        #expect(session.items.count == 1)
    }

    /// Deleting the *current* conversation is a switch like any other: the
    /// session lands on the store's fresh conversation with the outgoing
    /// one's error banner down (it belonged to the deleted transcript).
    @Test func deleteCurrentConversationSwitchesCleanAndDropsTheBanner() throws {
        let store = InMemoryAgentConversationStore()
        let agent = makeNoOpAgent(modelID: "test-model")
        let session = makeSession(agent: agent, store: store)
        store.createNew()
        let id = try #require(store.currentConversation).id
        let user = UserMessage(content: "doomed")
        agent.loadMessages([CoreMessage.user(user)])
        session.handle(.agentStart)
        session.handle(
            .turnEnd(
                message: AssistantMessage(content: "reply"), toolResults: [],
                contextMessages: [CoreMessage.user(user)]))
        session.handle(.agentEnd(messages: []))
        session.error = "stale banner"

        session.deleteConversation(id)

        #expect(session.error == nil)
        #expect(session.items.isEmpty)
        #expect(agent.context.messages.isEmpty)
        #expect(store.currentConversation?.id != id)
    }

    /// Deleting a conversation that is not current is not a switch — the
    /// live transcript, banner, and agent context stay untouched.
    @Test func deleteOtherConversationLeavesTheSessionUntouched() {
        let other = AgentConversation(
            messages: [CoreMessage.user(UserMessage(content: "other"))])
        let store = InMemoryAgentConversationStore(seed: [other])
        let agent = makeNoOpAgent(modelID: "test-model")
        let session = makeSession(agent: agent, store: store)
        store.createNew()
        agent.loadMessages([CoreMessage.user(UserMessage(content: "mine"))])
        session.error = "still mine"

        session.deleteConversation(other.id)

        #expect(session.error == "still mine")
        #expect(agent.context.messages.count == 1)
        #expect(store.currentConversation != nil)
    }

    /// A switch while the run sits queued behind the lease settles the
    /// Pending Row to the composer — restore, never silently drop.
    @Test func conversationSwitchRestoresAQueuedPendingRowToTheComposer() {
        let arbiter = InMemoryInferenceArbiter()
        arbiter.leaseDelay = .seconds(10)
        var restored: [String] = []
        let session = makeSession(
            arbiter: arbiter,
            restoreComposerDraft: { text, _ in restored.append(text) })
        session.sendMessage("queued behind the lease")
        #expect(session.pendingUserMessage != nil)

        session.newConversation()

        #expect(session.pendingUserMessage == nil)
        #expect(restored == ["queued behind the lease"])
        #expect(session.items.isEmpty)
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
