//
//  ChatTranscriptControllerTests.swift
//  tesseractTests
//
//  Tests the **Chat Transcript Controller** at its own public interface — fed a
//  scripted `(messages, stream, isGenerating)` trio. No `Agent`, no conversation
//  store. Migrated down from `AgentCoordinatorChatTranscriptCharacterizationTests`,
//  which had to route through `loadConversation` to reach the private
//  `rebuildRows`. Adds the throttled streaming tail-patch, which the
//  committed-only characterization could not surface.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ChatTranscriptControllerTests {

    // MARK: - Harness

    /// Full-rebuild rows for a committed (non-generating) transcript.
    private func committedRows(_ messages: [any AgentMessageProtocol]) -> [ChatRow] {
        let controller = ChatTranscriptController()
        controller.rebuild(messages: messages, stream: nil, isGenerating: false)
        return controller.rows
    }

    // MARK: - Grouping

    @Test func userMessageThenAnswerRendersUserRowAndAnswerRow() {
        let user = UserMessage(content: "Hello")
        let assistant = AssistantMessage(content: "Hi there")
        let rows = committedRows([user, assistant])

        #expect(rows.count == 2)
        #expect(rows.first?.id == user.id.uuidString)
        guard case .user(let userRow) = rows.first?.kind else {
            Issue.record("row 0 not a user row: \(String(describing: rows.first?.kind))"); return
        }
        #expect(userRow.content == "Hello")
        #expect(userRow.messageID == user.id)

        #expect(rows.last?.id == "\(assistant.id.uuidString)-answer")
        guard case .assistantText(let answerRow) = rows.last?.kind else {
            Issue.record("row 1 not an answer row: \(String(describing: rows.last?.kind))"); return
        }
        #expect(answerRow.content == "Hi there")
        #expect(answerRow.hasStepsAbove == false)
    }

    @Test func multipleTurnsRenderInOrder() {
        let user1 = UserMessage(content: "First")
        let asst1 = AssistantMessage(content: "Answer one")
        let user2 = UserMessage(content: "Second")
        let asst2 = AssistantMessage(content: "Answer two")
        let rows = committedRows([user1, asst1, user2, asst2])

        #expect(
            rows.map(\.id) == [
                user1.id.uuidString,
                "\(asst1.id.uuidString)-answer",
                user2.id.uuidString,
                "\(asst2.id.uuidString)-answer",
            ])
    }

    @Test func compactionMarkerRendersSystemRow() {
        let user1 = UserMessage(content: "Before")
        let asst1 = AssistantMessage(content: "Reply before")
        let compaction = CompactionSummaryMessage(summary: "summary text", tokensBefore: 4096)
        let user2 = UserMessage(content: "After")
        let asst2 = AssistantMessage(content: "Reply after")
        let rows = committedRows([user1, asst1, compaction, user2, asst2])

        #expect(rows.count == 5)
        #expect(rows[2].id == "\(compaction.id.uuidString)-system")
        guard case .system(let systemRow) = rows[2].kind else {
            Issue.record("row 2 not a system row: \(rows[2].kind)"); return
        }
        #expect(systemRow.content == "[Context compacted — 4096 tokens summarized]")
    }

    @Test func imageOnlyUserMessageRendersUserRow() {
        let image = ImageAttachment(
            data: Data([0x01, 0x02]), mimeType: "image/png", filename: "pic.png")
        let user = UserMessage(content: "", images: [image])
        let asst = AssistantMessage(content: "Nice picture")
        let rows = committedRows([user, asst])

        #expect(rows.count == 2)
        guard case .user(let userRow) = rows.first?.kind else {
            Issue.record("row 0 not a user row: \(String(describing: rows.first?.kind))"); return
        }
        #expect(userRow.content.isEmpty)
        #expect(userRow.images.count == 1)
        #expect(userRow.images.first?.id == image.id)
    }

    // MARK: - Per-Turn body (via toggleTurnExpanded)

    @Test func expandedToolCallTurnRendersToolRowMatchedToResult() {
        let user = UserMessage(content: "Read the file")
        let call = ToolCallInfo(
            id: "tc1", name: "read_file", argumentsJSON: #"{"path":"/tmp/foo.txt"}"#)
        let asst1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file",
            content: [.text("file contents")], isError: false
        )
        let asst2 = AssistantMessage(content: "Here is the file")
        let messages: [any AgentMessageProtocol] = [user, asst1, result, asst2]

        let controller = ChatTranscriptController()
        controller.rebuild(messages: messages, stream: nil, isGenerating: false)
        controller.toggleTurnExpanded(user.id, messages: messages, stream: nil, isGenerating: false)
        let rows = controller.rows

        #expect(rows.count == 4)
        guard case .turnHeader(let header) = rows[1].kind else {
            Issue.record("row 1 not a turn header: \(rows[1].kind)"); return
        }
        #expect(header.stepCount == 1)
        #expect(header.isExpanded == true)
        #expect(header.turnID == user.id)

        #expect(rows[2].id == "\(asst1.id.uuidString)-tool-0")
        guard case .toolCall(let tool) = rows[2].kind else {
            Issue.record("row 2 not a tool call: \(rows[2].kind)"); return
        }
        #expect(tool.displayTitle == "Reading foo.txt")
        #expect(tool.resultContent == "file contents")
        #expect(tool.filePath == "/tmp/foo.txt")
        #expect(tool.isLast == true)

        guard case .assistantText(let answer) = rows[3].kind else {
            Issue.record("row 3 not an answer: \(rows[3].kind)"); return
        }
        #expect(answer.content == "Here is the file")
        #expect(answer.hasStepsAbove == true)
    }

    /// A tool result carrying an `.image` block surfaces it as `resultImages` on
    /// the projected tool row (slice #116), alongside its text — previously the
    /// image was dropped by `projectTurn`.
    @Test func toolResultImageBlockSurvivesProjectionIntoResultImages() {
        let user = UserMessage(content: "Screenshot the page")
        let call = ToolCallInfo(id: "tc1", name: "screenshot", argumentsJSON: "{}")
        let asst1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "screenshot",
            content: [
                .text("captured"),
                .image(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png"),
            ],
            isError: false
        )
        let asst2 = AssistantMessage(content: "Done")
        let messages: [any AgentMessageProtocol] = [user, asst1, result, asst2]

        let controller = ChatTranscriptController()
        controller.rebuild(messages: messages, stream: nil, isGenerating: false)
        controller.toggleTurnExpanded(user.id, messages: messages, stream: nil, isGenerating: false)
        let rows = controller.rows

        guard case .toolCall(let tool) = rows[2].kind else {
            Issue.record("row 2 not a tool call: \(rows[2].kind)"); return
        }
        #expect(tool.resultContent == "captured")
        #expect(tool.resultImages.count == 1)
        #expect(tool.resultImages.first?.mimeType == "image/png")
        #expect(tool.resultImages.first?.data == ImageTestFixtures.tinyPNGData)
    }

    /// Regression: tool-result image attachment ids identify the *occurrence* —
    /// (tool-result, position) — not the content. The transcript builds the
    /// row's attachments once and `conversationImages()` re-derives them again;
    /// Quick Look matches the clicked id against that re-derived set. So the id
    /// must be (a) stable across calls under the same tool-result id — a random
    /// `UUID()` per call left tool-result images un-clickable — and (b) unique
    /// per occurrence, including byte-identical images in *different* results, so
    /// clicking the second of two identical screenshots opens that one.
    @Test func toolResultImageAttachmentIDsIdentifyOccurrenceNotContent() {
        let pngA = ImageTestFixtures.tinyPNGData
        let pngB = pngA + Data([0xFF])
        let blocks: [ContentBlock] = [
            .image(data: pngA, mimeType: "image/png"),  // index 0
            .text("between"),
            .image(data: pngA, mimeType: "image/png"),  // index 2 — same bytes
            .image(data: pngB, mimeType: "image/png"),  // index 3
        ]
        let ns1 = UUID()
        let ns2 = UUID()

        // Stable across independent projections under the same tool-result id.
        #expect(
            blocks.imageAttachments(namespace: ns1).map(\.id)
                == blocks.imageAttachments(namespace: ns1).map(\.id))

        // Every position distinct within a result, even byte-identical images.
        let ids1 = blocks.imageAttachments(namespace: ns1).map(\.id)
        #expect(ids1.count == 3)
        #expect(Set(ids1).count == 3)

        // Different tool result ⇒ disjoint ids, so byte-identical images across
        // results never collide (the bug this fix closes).
        let ids2 = blocks.imageAttachments(namespace: ns2).map(\.id)
        #expect(Set(ids1).isDisjoint(with: Set(ids2)))
    }

    @Test func intermediateTextIsDistinctFromFinalAnswer() {
        let user = UserMessage(content: "Do it")
        let call = ToolCallInfo(id: "tc1", name: "ls", argumentsJSON: #"{"path":"/tmp"}"#)
        let asst1 = AssistantMessage(content: "Let me check", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "ls",
            content: [.text("foo.txt")], isError: false
        )
        let asst2 = AssistantMessage(content: "Done")
        let messages: [any AgentMessageProtocol] = [user, asst1, result, asst2]

        let controller = ChatTranscriptController()
        controller.rebuild(messages: messages, stream: nil, isGenerating: false)
        controller.toggleTurnExpanded(user.id, messages: messages, stream: nil, isGenerating: false)
        let rows = controller.rows

        // user, header(2 steps), intermediate toolText, tool call, answer
        #expect(rows.count == 5)
        #expect(rows[2].id == "\(asst1.id.uuidString)-text")
        guard case .toolText(let interim) = rows[2].kind else {
            Issue.record("row 2 not toolText: \(rows[2].kind)"); return
        }
        #expect(interim.content == "Let me check")
        #expect(interim.isLast == false)

        guard case .assistantText(let answer) = rows[4].kind else {
            Issue.record("row 4 not an answer: \(rows[4].kind)"); return
        }
        #expect(answer.content == "Done")
    }

    // MARK: - Streaming tail-patch

    /// With throttling disabled, `patchStreamingTail` re-projects the active turn
    /// against the latest stream and splices it onto the stable prefix, bumping
    /// `streamingRowVersion` each call.
    @Test func tailPatchSplicesLatestStreamOntoStablePrefix() {
        let user = UserMessage(content: "Question")
        let messages: [any AgentMessageProtocol] = [user]

        let controller = ChatTranscriptController(streamingThrottle: .zero)
        controller.rebuild(
            messages: messages, stream: AssistantMessage(content: "par"), isGenerating: true)
        #expect(controller.streamingRowVersion == 0)

        controller.patchStreamingTail(
            messages: messages,
            stream: AssistantMessage(content: "partial answer"),
            isGenerating: true
        )

        #expect(controller.streamingRowVersion == 1)
        #expect(controller.rows.first?.id == user.id.uuidString)  // stable prefix
        guard case .streamingText(let streamed) = controller.rows.last?.kind else {
            Issue.record(
                "last row not streamingText: \(String(describing: controller.rows.last?.kind))")
            return
        }
        #expect(streamed.content == "partial answer")
    }

    /// `.zero` throttle applies every patch; a large throttle gates rapid patches
    /// within the window. Together they pin the throttle as real and disable-able.
    @Test func throttleGatesRapidPatchesAndZeroDisablesIt() {
        let user = UserMessage(content: "Question")
        let messages: [any AgentMessageProtocol] = [user]

        let unthrottled = ChatTranscriptController(streamingThrottle: .zero)
        unthrottled.rebuild(
            messages: messages, stream: AssistantMessage(content: "a"), isGenerating: true)
        for _ in 0..<3 {
            unthrottled.patchStreamingTail(
                messages: messages, stream: AssistantMessage(content: "a"), isGenerating: true)
        }
        #expect(unthrottled.streamingRowVersion == 3)

        let throttled = ChatTranscriptController(streamingThrottle: .seconds(60))
        throttled.rebuild(
            messages: messages, stream: AssistantMessage(content: "a"), isGenerating: true)
        for _ in 0..<3 {
            throttled.patchStreamingTail(
                messages: messages, stream: AssistantMessage(content: "a"), isGenerating: true)
        }
        #expect(throttled.streamingRowVersion == 0)
    }

    // MARK: - Reset

    @Test func resetClearsRowsAndExpansion() {
        let user = UserMessage(content: "Hello")
        let asst = AssistantMessage(content: "Hi")
        let controller = ChatTranscriptController()
        controller.rebuild(messages: [user, asst], stream: nil, isGenerating: false)
        #expect(controller.rows.isEmpty == false)

        controller.reset()
        #expect(controller.rows.isEmpty)
    }
}
