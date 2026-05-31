//
//  AgentCoordinatorChatTranscriptCharacterizationTests.swift
//  tesseractTests
//
//  Golden-master tests pinning the CURRENT [ChatRow] output of AgentCoordinator
//  before the Chat Transcript extraction (#28). Committed-state row building is
//  driven through the public `loadConversation` path against a hermetic
//  InMemoryAgentConversationStore — no SwiftUI, no live agent, no disk. These
//  lock the subtle grouping / per-Turn / answer-discrimination rules so the
//  extraction is provably behaviour-preserving (except the documented
//  image-only-user-message convergence, which gets its own before/after
//  assertion once the module lands).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentCoordinatorChatTranscriptCharacterizationTests {

    // MARK: - Harness

    private func makeAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "chat-transcript-characterization-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in AsyncThrowingStream { $0.finish() } }
        )
    }

    /// Seeds the conversation store with `messages` and loads it through the
    /// public `loadConversation` path (which calls the private `rebuildRows()`),
    /// returning the coordinator so a test can read `rows` or toggle expansion.
    private func loadedCoordinator(
        for messages: [any AgentMessageProtocol & Sendable]
    ) -> AgentCoordinator {
        let conversation = AgentConversation(messages: messages)
        let store = InMemoryAgentConversationStore(seed: [conversation])
        let coordinator = AgentCoordinator(
            agent: makeAgent(),
            conversationStore: store,
            settings: SettingsManager(store: InMemorySettingsStore())
        )
        coordinator.loadConversation(conversation.id)
        return coordinator
    }

    private func coordinatorRows(
        for messages: [any AgentMessageProtocol & Sendable]
    ) -> [ChatRow] {
        loadedCoordinator(for: messages).rows
    }

    // MARK: - Grouping

    /// A lone user message followed by the assistant's final answer renders as
    /// exactly two rows: the user row (id = turn uuid) and the answer row
    /// (id = "<assistantID>-answer"). A single answer is not a multi-step turn,
    /// so there is no turn header and `hasStepsAbove` is false.
    @Test func userMessageThenAnswerRendersUserRowAndAnswerRow() {
        let user = UserMessage(content: "Hello")
        let assistant = AssistantMessage(content: "Hi there")
        let rows = coordinatorRows(for: [user, assistant])

        #expect(rows.count == 2)

        #expect(rows.first?.id == user.id.uuidString)
        guard case .user(let userRow) = rows.first?.kind else {
            Issue.record("row 0 not a user row: \(String(describing: rows.first?.kind))"); return
        }
        #expect(userRow.content == "Hello")
        #expect(userRow.images.isEmpty)
        #expect(userRow.messageID == user.id)

        #expect(rows.last?.id == "\(assistant.id.uuidString)-answer")
        guard case .assistantText(let answerRow) = rows.last?.kind else {
            Issue.record("row 1 not an answer row: \(String(describing: rows.last?.kind))"); return
        }
        #expect(answerRow.content == "Hi there")
        #expect(answerRow.messageID == assistant.id)
        #expect(answerRow.hasStepsAbove == false)
    }

    /// A lone user message with no assistant response renders as a single user row.
    @Test func userMessageWithNoResponseRendersOnlyUserRow() {
        let user = UserMessage(content: "Anyone there?")
        let rows = coordinatorRows(for: [user])
        #expect(rows.count == 1)
        #expect(rows.first?.id == user.id.uuidString)
        if case .user = rows.first?.kind {} else {
            Issue.record("expected a user row, got \(String(describing: rows.first?.kind))")
        }
    }

    /// Two user prompts each answered render as four rows in turn order, with no
    /// cross-turn bleed: user1, answer1, user2, answer2.
    @Test func multipleTurnsRenderInOrder() {
        let user1 = UserMessage(content: "First")
        let asst1 = AssistantMessage(content: "Answer one")
        let user2 = UserMessage(content: "Second")
        let asst2 = AssistantMessage(content: "Answer two")
        let rows = coordinatorRows(for: [user1, asst1, user2, asst2])

        #expect(rows.map(\.id) == [
            user1.id.uuidString,
            "\(asst1.id.uuidString)-answer",
            user2.id.uuidString,
            "\(asst2.id.uuidString)-answer",
        ])
    }

    /// A compaction marker starts its own turn and renders a system row whose id
    /// is "<markerID>-system" and whose text reports the pre-compaction token count.
    @Test func compactionMarkerRendersSystemRow() {
        let user1 = UserMessage(content: "Before")
        let asst1 = AssistantMessage(content: "Reply before")
        let compaction = CompactionSummaryMessage(summary: "summary text", tokensBefore: 4096)
        let user2 = UserMessage(content: "After")
        let asst2 = AssistantMessage(content: "Reply after")
        let rows = coordinatorRows(for: [user1, asst1, compaction, user2, asst2])

        #expect(rows.count == 5)
        #expect(rows[2].id == "\(compaction.id.uuidString)-system")
        guard case .system(let systemRow) = rows[2].kind else {
            Issue.record("row 2 not a system row: \(rows[2].kind)"); return
        }
        #expect(systemRow.content == "[Context compacted — 4096 tokens summarized]")
    }

    // MARK: - Per-Turn body

    /// An expanded tool-calling turn renders user, a turn header (1 step), the
    /// tool-call row (matched to its result by id, last step stamped, file path
    /// and display title derived from the tool), and the final answer with
    /// `hasStepsAbove`.
    @Test func expandedToolCallTurnRendersToolRowMatchedToResult() {
        let user = UserMessage(content: "Read the file")
        let call = ToolCallInfo(id: "tc1", name: "read_file", argumentsJSON: #"{"path":"/tmp/foo.txt"}"#)
        let asst1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file",
            content: [.text("file contents")], isError: false
        )
        let asst2 = AssistantMessage(content: "Here is the file")

        let coordinator = loadedCoordinator(for: [user, asst1, result, asst2])
        coordinator.toggleTurnExpanded(user.id)   // turn id == first user message id
        let rows = coordinator.rows

        #expect(rows.count == 4)

        guard case .turnHeader(let header) = rows[1].kind else {
            Issue.record("row 1 not a turn header: \(rows[1].kind)"); return
        }
        #expect(header.stepCount == 1)
        #expect(header.isExpanded == true)
        #expect(header.isGenerating == false)
        #expect(header.turnID == user.id)

        #expect(rows[2].id == "\(asst1.id.uuidString)-tool-0")
        guard case .toolCall(let tool) = rows[2].kind else {
            Issue.record("row 2 not a tool call: \(rows[2].kind)"); return
        }
        #expect(tool.displayTitle == "Reading foo.txt")
        #expect(tool.iconName == "doc.text")
        #expect(tool.resultContent == "file contents")
        #expect(tool.isError == false)
        #expect(tool.filePath == "/tmp/foo.txt")
        #expect(tool.isLast == true)
        #expect(tool.isDetailExpanded == false)

        guard case .assistantText(let answer) = rows[3].kind else {
            Issue.record("row 3 not an answer: \(rows[3].kind)"); return
        }
        #expect(answer.content == "Here is the file")
        #expect(answer.hasStepsAbove == true)
    }

    /// Intermediate assistant text (content alongside tool calls, not the final
    /// message) renders as a toolText step — distinct from the final answer row —
    /// and is not stamped `isLast` when a tool call follows it.
    @Test func intermediateTextIsDistinctFromFinalAnswer() {
        let user = UserMessage(content: "Do it")
        let call = ToolCallInfo(id: "tc1", name: "ls", argumentsJSON: #"{"path":"/tmp"}"#)
        let asst1 = AssistantMessage(content: "Let me check", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "ls",
            content: [.text("foo.txt")], isError: false
        )
        let asst2 = AssistantMessage(content: "Done")

        let coordinator = loadedCoordinator(for: [user, asst1, result, asst2])
        coordinator.toggleTurnExpanded(user.id)
        let rows = coordinator.rows

        // user, header(2 steps), intermediate toolText, tool call, answer
        #expect(rows.count == 5)
        #expect(rows[2].id == "\(asst1.id.uuidString)-text")
        guard case .toolText(let interim) = rows[2].kind else {
            Issue.record("row 2 not toolText: \(rows[2].kind)"); return
        }
        #expect(interim.content == "Let me check")
        #expect(interim.isLast == false)

        #expect(rows[3].id == "\(asst1.id.uuidString)-tool-0")
        if case .toolCall = rows[3].kind {} else {
            Issue.record("row 3 not a tool call: \(rows[3].kind)")
        }

        guard case .assistantText(let answer) = rows[4].kind else {
            Issue.record("row 4 not an answer: \(rows[4].kind)"); return
        }
        #expect(answer.content == "Done")
    }

    // MARK: - Image-only user message (current full-rebuild behaviour)

    /// An image-only user message (no caption) renders its user row on the full
    /// rebuild — images present, empty content. This is the behaviour the
    /// extraction converges *both* paths onto (today the streaming tail-patch
    /// drops it); pinned here as the pre-extraction baseline.
    @Test func imageOnlyUserMessageRendersUserRowOnFullRebuild() {
        let image = ImageAttachment(data: Data([0x01, 0x02]), mimeType: "image/png", filename: "pic.png")
        let user = UserMessage(content: "", images: [image])
        let asst = AssistantMessage(content: "Nice picture")
        let rows = coordinatorRows(for: [user, asst])

        #expect(rows.count == 2)
        #expect(rows.first?.id == user.id.uuidString)
        guard case .user(let userRow) = rows.first?.kind else {
            Issue.record("row 0 not a user row: \(String(describing: rows.first?.kind))"); return
        }
        #expect(userRow.content == "")
        #expect(userRow.images.count == 1)
        #expect(userRow.images.first?.id == image.id)
    }
}
