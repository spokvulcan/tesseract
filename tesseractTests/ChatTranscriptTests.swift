//
//  ChatTranscriptTests.swift
//  tesseractTests
//
//  Exhaustive pure-value tests for the Chat Transcript projection (#28). The
//  module is pure, so every test is an input -> output row assertion with no
//  SwiftUI, no agent, no socket, no clock — the timestamp formatter is injected.
//  Exercises grouping, the per-Turn body, the streaming interleave, the
//  zero/empty-Turn fallbacks, the image-only convergence, and purity. The
//  interface is the test surface; nothing reaches into private storage.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ChatTranscriptTests {

    // MARK: - Fixtures

    private func ctx(
        generating: Bool = false,
        expandedTurns: Set<UUID> = [],
        expandedDetails: Set<String> = [],
        stream: AssistantMessage? = nil
    ) -> ChatTranscript.Context {
        ChatTranscript.Context(
            isGenerating: generating,
            expandedTurns: expandedTurns,
            expandedDetails: expandedDetails,
            stream: stream,
            formatTimestamp: { _ in "TS" }
        )
    }

    private func toolCall(id: String, name: String, path: String) -> ToolCallInfo {
        ToolCallInfo(id: id, name: name, argumentsJSON: #"{"path":"\#(path)"}"#)
    }

    // MARK: - Grouping

    @Test func groupsSingleTurn() {
        let user = UserMessage(content: "hi")
        let asst = AssistantMessage(content: "hello")
        let turns = ChatTranscript.turns(from: [user, asst])
        #expect(turns.count == 1)
        #expect(turns[0].id == user.id)
        #expect(turns[0].messages.count == 2)
    }

    @Test func groupsMultipleTurnsAtUserBoundaries() {
        let u1 = UserMessage(content: "a"); let a1 = AssistantMessage(content: "A")
        let u2 = UserMessage(content: "b"); let a2 = AssistantMessage(content: "B")
        let turns = ChatTranscript.turns(from: [u1, a1, u2, a2])
        #expect(turns.map(\.id) == [u1.id, u2.id])
    }

    @Test func compactionMarkerStartsNewTurn() {
        let u1 = UserMessage(content: "a"); let a1 = AssistantMessage(content: "A")
        let compaction = CompactionSummaryMessage(summary: "s", tokensBefore: 10)
        let u2 = UserMessage(content: "b")
        let turns = ChatTranscript.turns(from: [u1, a1, compaction, u2])
        #expect(turns.map(\.id) == [u1.id, compaction.id, u2.id])
        #expect(turns[1].messages.count == 1)  // compaction marker alone
    }

    @Test func emptyLogYieldsNoTurns() {
        #expect(ChatTranscript.turns(from: []).isEmpty)
    }

    /// Malformed input: leading non-user messages (no opening user/compaction).
    /// Pinned behaviour — grouped into one Turn whose id is the first message's
    /// uuid (the old backward scan returned an empty tail here).
    @Test func leadingNonUserMessagesGroupIntoOneTurn() {
        let a1 = AssistantMessage(content: "orphan")
        let a2 = AssistantMessage(content: "second")
        let turns = ChatTranscript.turns(from: [a1, a2])
        #expect(turns.count == 1)
        #expect(turns[0].id == a1.id)
        #expect(turns[0].messages.count == 2)
    }

    /// The streaming tail-patch uses `activeTurn(from:)` (a bounded backward
    /// scan) instead of grouping the whole log; it must return exactly what
    /// `turns(from:).last` would — same id, same message count — across every
    /// case of the grouping rule, or the fast path and the full rebuild disagree.
    @Test func activeTurnMatchesLastOfTurns() {
        let u1 = UserMessage(content: "a"); let a1 = AssistantMessage(content: "A")
        let u2 = UserMessage(content: "b"); let a2 = AssistantMessage(content: "B")
        let compaction = CompactionSummaryMessage(summary: "s", tokensBefore: 8)
        let orphan = AssistantMessage(content: "orphan")

        let cases: [[any AgentMessageProtocol]] = [
            [],  // empty
            [u1],  // single user, no response
            [u1, a1],  // one full turn
            [u1, a1, u2, a2],  // multi-turn -> last
            [compaction, u2],  // compaction boundary then user
            [orphan, a1],  // leading non-boundary (malformed)
        ]

        for messages in cases {
            let expected = ChatTranscript.turns(from: messages).last
            let active = ChatTranscript.activeTurn(from: messages)
            #expect(active?.id == expected?.id)
            #expect(active?.messages.count == expected?.messages.count)
        }
    }

    /// A leading `OpaqueMessage` (the unknown-persistence-tag fallback, the one
    /// message type whose identity comes from a stored id rather than a core
    /// `Identifiable`) must yield the *same* Turn id from the full rebuild
    /// (`turns(from:)`) and the streaming tail-patch (`activeTurn(from:)`), and a
    /// *stable* one across calls. Otherwise `messageUUID`'s random fallback would
    /// re-randomize the active Turn's id every streaming tick, churning its
    /// `ForEach` identity and desyncing its expansion toggle across the splice.
    @Test func opaqueMessageTurnIDIsStableAndAgreesAcrossPaths() {
        let opaque = OpaqueMessage(tag: "future_unknown_tag", rawPayload: [:])
        let messages: [any AgentMessageProtocol] = [opaque]

        #expect(opaque.messageUUID == opaque.messageUUID)  // stable, not re-randomized

        let fromTurns = ChatTranscript.turns(from: messages).last?.id
        let fromActive = ChatTranscript.activeTurn(from: messages)?.id
        #expect(fromTurns == fromActive)
        #expect(fromTurns == opaque.messageUUID)
    }

    // MARK: - Committed projection

    @Test func userThenAnswerProjectsTwoRows() {
        let user = UserMessage(content: "Hello")
        let asst = AssistantMessage(content: "Hi there")
        let projection = ChatTranscript.rows(from: [user, asst], ctx())
        #expect(projection.rows.count == 2)
        #expect(projection.rows[0].id == user.id.uuidString)
        #expect(projection.rows[1].id == "\(asst.id.uuidString)-answer")
        #expect(projection.activeTurnStart == 0)
        guard case .assistantText(let answer) = projection.rows[1].kind else {
            Issue.record("row 1 not an answer"); return
        }
        #expect(answer.hasStepsAbove == false)
        #expect(answer.timestamp == "TS")
    }

    @Test func userOnlyTurnProjectsSingleRow() {
        let user = UserMessage(content: "anyone?")
        let rows = ChatTranscript.rows(from: [user], ctx()).rows
        #expect(rows.count == 1)
        #expect(rows[0].id == user.id.uuidString)
    }

    @Test func compactionRendersSystemRow() {
        let compaction = CompactionSummaryMessage(summary: "s", tokensBefore: 4096)
        let rows = ChatTranscript.rows(from: [compaction], ctx()).rows
        #expect(rows.count == 1)
        #expect(rows[0].id == "\(compaction.id.uuidString)-system")
        guard case .system(let system) = rows[0].kind else {
            Issue.record("not system row"); return
        }
        #expect(system.content == "[Context compacted — 4096 tokens summarized]")
    }

    @Test func collapsedToolTurnHidesStepsShowsHeaderAndAnswer() {
        let user = UserMessage(content: "read")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("data")])
        let a2 = AssistantMessage(content: "done")
        let rows = ChatTranscript.rows(from: [user, a1, result, a2], ctx()).rows

        #expect(rows.count == 3)  // user, header, answer (steps hidden when collapsed)
        guard case .turnHeader(let header) = rows[1].kind else { Issue.record("no header"); return }
        #expect(header.stepCount == 1)
        #expect(header.isExpanded == false)
        #expect(header.isGenerating == false)
    }

    @Test func expandedToolTurnShowsToolRowMatchedToResult() {
        let user = UserMessage(content: "read")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("data")], isError: false)
        let a2 = AssistantMessage(content: "done")
        let rows = ChatTranscript.rows(from: [user, a1, result, a2], ctx(expandedTurns: [user.id]))
            .rows

        #expect(rows.count == 4)  // user, header, tool, answer
        #expect(rows[2].id == "\(a1.id.uuidString)-tool-0")
        guard case .toolCall(let tool) = rows[2].kind else { Issue.record("no tool row"); return }
        #expect(tool.displayTitle == "Reading foo.txt")
        #expect(tool.iconName == "doc.text")
        #expect(tool.resultContent == "data")
        #expect(tool.isError == false)
        #expect(tool.filePath == "/tmp/foo.txt")
        #expect(tool.isLast == true)

        guard case .assistantText(let answer) = rows[3].kind else {
            Issue.record("no answer"); return
        }
        #expect(answer.hasStepsAbove == true)
    }

    @Test func errorToolResultMarksToolRow() {
        let user = UserMessage(content: "read")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("boom")], isError: true)
        let a2 = AssistantMessage(content: "failed")
        let rows = ChatTranscript.rows(from: [user, a1, result, a2], ctx(expandedTurns: [user.id]))
            .rows
        guard let row = rows.first(where: { $0.id == "\(a1.id.uuidString)-tool-0" }),
            case .toolCall(let tool) = row.kind
        else { Issue.record("no tool row"); return }
        #expect(tool.isError == true)
        #expect(tool.resultContent == "boom")
    }

    @Test func intermediateTextIsAStepNotTheAnswer() {
        let user = UserMessage(content: "do")
        let call = toolCall(id: "tc1", name: "ls", path: "/tmp")
        let a1 = AssistantMessage(content: "checking", toolCalls: [call])
        let result = ToolResultMessage(toolCallId: "tc1", toolName: "ls", content: [.text("x")])
        let a2 = AssistantMessage(content: "done")
        let rows = ChatTranscript.rows(from: [user, a1, result, a2], ctx(expandedTurns: [user.id]))
            .rows

        #expect(rows.count == 5)  // user, header(2), toolText, toolCall, answer
        #expect(rows[2].id == "\(a1.id.uuidString)-text")
        guard case .toolText(let interim) = rows[2].kind else {
            Issue.record("no interim text"); return
        }
        #expect(interim.content == "checking")
        #expect(interim.isLast == false)  // a tool call follows it
        guard case .assistantText(let answer) = rows[4].kind else {
            Issue.record("no answer"); return
        }
        #expect(answer.content == "done")
    }

    @Test func thinkingEmittedOnlyWhenNonEmptyAfterTrim() {
        let user = UserMessage(content: "q")
        // Whitespace-only thinking on a final answer -> no thinking step, no header.
        let blank = AssistantMessage(content: "answer", thinking: "   ")
        let blankRows = ChatTranscript.rows(from: [user, blank], ctx(expandedTurns: [user.id])).rows
        #expect(blankRows.count == 2)

        // Real thinking on an intermediate step -> a thinking step row.
        let thinker = AssistantMessage(
            content: "", thinking: "pondering",
            toolCalls: [toolCall(id: "t", name: "ls", path: "/x")]
        )
        let result = ToolResultMessage(toolCallId: "t", toolName: "ls", content: [.text("r")])
        let final = AssistantMessage(content: "final")
        let rows = ChatTranscript.rows(
            from: [user, thinker, result, final], ctx(expandedTurns: [user.id])
        ).rows
        #expect(rows.count == 5)  // user, header(2), thinking, tool, answer
        #expect(rows[2].id == "\(thinker.id.uuidString)-thinking")
        guard case .thinking(let think) = rows[2].kind else {
            Issue.record("no thinking row"); return
        }
        #expect(think.content == "pondering")
    }

    // MARK: - Streaming (active turn / tail patch)

    @Test func activeTurnNoAssistantNoStreamShowsIndicator() {
        let user = UserMessage(content: "hi")
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let rows = ChatTranscript.rows(for: turn, ctx(generating: true, stream: nil))
        #expect(rows.count == 2)
        #expect(rows[0].id == user.id.uuidString)
        #expect(rows[1].id == "streaming-indicator")
        if case .streamingIndicator = rows[1].kind {
        } else {
            Issue.record("not a streaming indicator")
        }
    }

    @Test func activeTurnStreamingThinkingAndTextExpanded() {
        let user = UserMessage(content: "hi")
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let stream = AssistantMessage(content: "answer so far", thinking: "thinking...")
        let rows = ChatTranscript.rows(
            for: turn,
            ctx(generating: true, expandedTurns: [ChatTranscript.streamingTurnID], stream: stream)
        )
        #expect(
            rows.map(\.id) == [
                user.id.uuidString, "streaming-header", "streaming-thinking", "streaming-answer",
            ])
        guard case .turnHeader(let header) = rows[1].kind else { Issue.record("no header"); return }
        #expect(header.stepCount == 1)  // thinking is the one live step
        #expect(header.isGenerating == true)
        #expect(header.turnID == ChatTranscript.streamingTurnID)
    }

    @Test func headerStepCountIsCommittedPlusLiveAndCommittedStepNotLast() {
        let user = UserMessage(content: "do")
        let committedCall = toolCall(id: "tc1", name: "ls", path: "/tmp")
        let a1 = AssistantMessage(content: "", toolCalls: [committedCall])
        let result = ToolResultMessage(toolCallId: "tc1", toolName: "ls", content: [.text("r")])
        let turn = ChatTranscript.Turn(id: user.id, messages: [user, a1, result])
        let stream = AssistantMessage(
            content: "", toolCalls: [toolCall(id: "tc2", name: "read_file", path: "/y")])

        let rows = ChatTranscript.rows(
            for: turn,
            ctx(generating: true, expandedTurns: [ChatTranscript.streamingTurnID], stream: stream)
        )
        guard case .turnHeader(let header) = rows[1].kind else { Issue.record("no header"); return }
        #expect(header.stepCount == 2)  // 1 committed + 1 live

        guard case .toolCall(let committed) = rows[2].kind else {
            Issue.record("no committed tool"); return
        }
        #expect(committed.isLast == false)  // live steps follow, so not last
        #expect(rows.contains { $0.id == "streaming-tool-0" })
    }

    @Test func collapsedStreamingShowsOnlyFinalAnswer() {
        let user = UserMessage(content: "hi")
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let stream = AssistantMessage(content: "partial answer")  // no thinking, no tools
        // streamingTurnID NOT in expandedTurns -> collapsed
        let rows = ChatTranscript.rows(
            for: turn, ctx(generating: true, expandedTurns: [], stream: stream))
        #expect(rows.map(\.id) == [user.id.uuidString, "streaming-answer"])
        guard case .streamingText(let streaming) = rows[1].kind else {
            Issue.record("no streaming text"); return
        }
        #expect(streaming.content == "partial answer")
    }

    /// The generating indicator must survive a committed assistant that renders
    /// *zero* rows. An assistant whose content is whitespace-only commits anyway
    /// (the commit guard uses a raw `isEmpty`, so `"   "` is admitted) yet yields
    /// no steps and no answer. The fallback is gated on "no committed rows
    /// produced", not "no assistant message present", so the indicator still
    /// shows while generation runs.
    @Test func activeTurnWithBlankCommittedAssistantStillShowsIndicator() {
        let user = UserMessage(content: "hi")
        let blank = AssistantMessage(content: "   ")  // whitespace-only -> no rows
        let turn = ChatTranscript.Turn(id: user.id, messages: [user, blank])
        let rows = ChatTranscript.rows(for: turn, ctx(generating: true, stream: nil))
        #expect(rows.map(\.id) == [user.id.uuidString, "streaming-indicator"])
    }

    /// A streaming tool row honors `expandedDetails` (keyed `streaming-tool-<i>`)
    /// rather than hard-coding collapsed, so a detail the user opened stays open
    /// across the ~20×/sec tail-patch reprojections instead of snapping shut.
    @Test func streamingToolRowDetailHonorsExpandedDetails() {
        let user = UserMessage(content: "hi")
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let stream = AssistantMessage(
            content: "", toolCalls: [toolCall(id: "tc1", name: "read_file", path: "/tmp/a.txt")])
        let rows = ChatTranscript.rows(
            for: turn,
            ctx(
                generating: true,
                expandedTurns: [ChatTranscript.streamingTurnID],
                expandedDetails: ["streaming-tool-0"],
                stream: stream
            )
        )
        guard let row = rows.first(where: { $0.id == "streaming-tool-0" }),
            case .toolCall(let tool) = row.kind
        else { Issue.record("no streaming tool row"); return }
        #expect(tool.isDetailExpanded == true)
    }

    /// The header's "(n steps)" badge (`streamingStepCount`) and the rendered
    /// step rows (`streamingStepRows`) encode the same step-emission rule; pin
    /// them in lockstep so adding or reordering a streaming step kind in one but
    /// not the other fails here. For a streaming-only active turn the header
    /// count must equal the number of rendered streaming *step* rows — everything
    /// the overlay emits except the header and the final `streaming-answer` row
    /// (which is the answer, not a step).
    @Test func streamingHeaderCountMatchesRenderedStepRows() {
        let user = UserMessage(content: "hi")
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let streams: [AssistantMessage] = [
            AssistantMessage(content: "", thinking: "t"),
            AssistantMessage(content: "", toolCalls: [toolCall(id: "a", name: "ls", path: "/x")]),
            AssistantMessage(
                content: "interim", toolCalls: [toolCall(id: "a", name: "ls", path: "/x")]),
            AssistantMessage(
                content: "", thinking: "t",
                toolCalls: [
                    toolCall(id: "a", name: "ls", path: "/x"),
                    toolCall(id: "b", name: "read_file", path: "/y"),
                ]),
        ]
        for stream in streams {
            let rows = ChatTranscript.rows(
                for: turn,
                ctx(
                    generating: true, expandedTurns: [ChatTranscript.streamingTurnID],
                    stream: stream)
            )
            guard case .turnHeader(let header) = rows[1].kind else {
                Issue.record("no streaming header"); return
            }
            let stepRows = rows.filter {
                $0.id != "streaming-header" && $0.id != "streaming-answer"
                    && $0.id != user.id.uuidString
            }
            #expect(header.stepCount == stepRows.count)
        }
    }

    // MARK: - Zero-Turn

    @Test func zeroTurnGeneratingNoStreamShowsIndicator() {
        let projection = ChatTranscript.rows(from: [], ctx(generating: true, stream: nil))
        #expect(projection.rows.count == 1)
        #expect(projection.rows[0].id == "streaming-indicator")
        #expect(projection.activeTurnStart == 0)
    }

    @Test func zeroTurnNotGeneratingIsEmpty() {
        let projection = ChatTranscript.rows(from: [], ctx(generating: false))
        #expect(projection.rows.isEmpty)
        #expect(projection.activeTurnStart == 0)
    }

    // MARK: - Image-only convergence (the one intentional behaviour change)

    /// An image-only user Turn renders its user row on the active-turn (tail
    /// patch) path — today's streaming patch drops it. This is the converged
    /// behaviour.
    @Test func imageOnlyUserRowRendersOnActiveTurnPatchPath() {
        let image = ImageAttachment(data: Data([0x01]), mimeType: "image/png", filename: "p.png")
        let user = UserMessage(content: "", images: [image])
        let turn = ChatTranscript.Turn(id: user.id, messages: [user])
        let rows = ChatTranscript.rows(for: turn, ctx(generating: true, stream: nil))
        #expect(rows.first?.id == user.id.uuidString)
        guard case .user(let userRow) = rows.first?.kind else {
            Issue.record("no user row"); return
        }
        #expect(userRow.content.isEmpty)
        #expect(userRow.images.count == 1)
    }

    // MARK: - Detail expansion passthrough

    @Test func expandedDetailMarksToolRow() {
        let user = UserMessage(content: "read")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("data")])
        let a2 = AssistantMessage(content: "done")
        let toolRowID = "\(a1.id.uuidString)-tool-0"
        let rows = ChatTranscript.rows(
            from: [user, a1, result, a2],
            ctx(expandedTurns: [user.id], expandedDetails: [toolRowID])
        ).rows
        guard let row = rows.first(where: { $0.id == toolRowID }),
            case .toolCall(let tool) = row.kind
        else {
            Issue.record("tool row missing"); return
        }
        #expect(tool.isDetailExpanded == true)
    }

    // MARK: - Detail pruning support

    /// A collapsed Turn's tool rows are not rendered, but their ids are still
    /// reported in `detailRowIDs` so the coordinator's full-rebuild detail-pruning
    /// preserves tool-detail expansion across a collapse/re-expand.
    @Test func detailRowIDsIncludeCollapsedTurnTools() {
        let user = UserMessage(content: "read")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("data")])
        let a2 = AssistantMessage(content: "done")
        let projection = ChatTranscript.rows(from: [user, a1, result, a2], ctx())  // collapsed

        let toolRowID = "\(a1.id.uuidString)-tool-0"
        #expect(!projection.rows.contains { $0.id == toolRowID })  // collapsed: not rendered
        #expect(projection.detailRowIDs.contains(toolRowID))  // but tracked for pruning
    }

    /// The projection reports the still-valid turn ids — committed Turn ids, plus
    /// ``ChatTranscript/streamingTurnID`` while generating — so the coordinator
    /// prunes stale `expandedTurns` without re-grouping the log a second time.
    @Test func projectionExposesValidTurnIDs() {
        let u1 = UserMessage(content: "a"); let a1 = AssistantMessage(content: "A")
        let u2 = UserMessage(content: "b")

        let committed = ChatTranscript.rows(from: [u1, a1, u2], ctx())
        #expect(committed.validTurnIDs == [u1.id, u2.id])
        #expect(!committed.validTurnIDs.contains(ChatTranscript.streamingTurnID))

        let generating = ChatTranscript.rows(from: [u1, a1, u2], ctx(generating: true))
        #expect(generating.validTurnIDs == [u1.id, u2.id, ChatTranscript.streamingTurnID])
    }

    // MARK: - Splice point

    @Test func activeTurnStartPointsAtLastTurnStart() {
        let u1 = UserMessage(content: "a"); let a1 = AssistantMessage(content: "A")
        let u2 = UserMessage(content: "b"); let a2 = AssistantMessage(content: "B")
        let projection = ChatTranscript.rows(from: [u1, a1, u2, a2], ctx())
        // rows: [u1, a1-answer, u2, a2-answer]; the last turn begins at index 2
        #expect(projection.activeTurnStart == 2)
        #expect(projection.rows[projection.activeTurnStart].id == u2.id.uuidString)
    }

    // MARK: - Purity

    @Test func identicalInputsYieldIdenticalRows() {
        let user = UserMessage(content: "hi")
        let call = toolCall(id: "tc1", name: "read_file", path: "/tmp/foo.txt")
        let a1 = AssistantMessage(content: "", toolCalls: [call])
        let result = ToolResultMessage(
            toolCallId: "tc1", toolName: "read_file", content: [.text("data")])
        let a2 = AssistantMessage(content: "done")
        let messages: [any AgentMessageProtocol] = [user, a1, result, a2]
        let context = ctx(expandedTurns: [user.id])

        let first = ChatTranscript.rows(from: messages, context)
        let second = ChatTranscript.rows(from: messages, context)
        #expect(first.rows == second.rows)
        #expect(first.activeTurnStart == second.activeTurnStart)
    }
}
