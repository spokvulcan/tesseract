import Foundation

/// The **Chat Transcript**: the pure projection of the agent message log into
/// the flat `[ChatRow]` the chat list renders, grouped into **Turn**s.
///
/// A stateless namespace of pure value logic — no `@MainActor`, no `@Observable`,
/// no coordinator references. It reads no coordinator state and has no side
/// effects: expansion state, the live `stream`, and a timestamp formatter are
/// passed in via ``Context``. The coordinator projects every Turn for a full
/// rebuild (``rows(from:_:)``) and only the last Turn for the streaming
/// tail-patch (``rows(for:_:)``), then splices — so the fast path avoids
/// re-running tool display formatting for all history while sharing one fold.
///
/// Distinct from the **Generation Projection** (which maps one turn's Generation
/// Accumulator state to a caller's output): the Chat Transcript projects the
/// whole committed log plus the live stream into the rendered row sequence.
nonisolated enum ChatTranscript {

    /// Stable synthetic id for the active (streaming) turn's header — survives
    /// rebuilds so its expansion toggle persists. Shared with the coordinator's
    /// expansion tracking and toggle handlers.
    static let streamingTurnID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    // MARK: - Turn

    /// The grouping unit: a contiguous run of messages from one user message (or
    /// compaction marker) through the assistant's complete response. The
    /// projection derives user / compaction / assistant content from `messages`.
    struct Turn {
        let id: UUID
        var messages: [any AgentMessageProtocol]
    }

    // MARK: - Context

    /// Everything the projection needs beyond the messages — all read-only.
    ///
    /// Deliberately **not** `Sendable`: it carries a `formatTimestamp` closure,
    /// and the app's instance closes over a `DateFormatter`. A `Context` is built
    /// and consumed within a single isolation domain — the coordinator builds one
    /// on the main actor per rebuild / tail-patch, and the pure-value tests build
    /// one synchronously off-actor — and is never shared across concurrent tasks.
    /// The projection being `nonisolated` lets either caller run it *in place*; it
    /// does not promise that one `Context` is safe to hand to `Task.detached`.
    struct Context {
        /// Whole-transcript generating flag (drives the active turn + the
        /// zero/empty-Turn fallback).
        var isGenerating: Bool
        /// Expanded turn ids. The module resolves expansion per Turn from this
        /// set, checking ``streamingTurnID`` for the active Turn and `turn.id`
        /// for committed Turns. The coordinator is responsible for the
        /// "auto-expand the streaming header unless the user collapsed it"
        /// decision before projecting (by inserting ``streamingTurnID`` here).
        var expandedTurns: Set<UUID>
        /// Tool-call row ids whose argument/result detail is expanded.
        var expandedDetails: Set<String>
        /// The live streaming message — active turn only.
        var stream: AssistantMessage?
        /// Injected for deterministic tests; the coordinator passes its formatter.
        var formatTimestamp: (Date) -> String

        init(
            isGenerating: Bool = false,
            expandedTurns: Set<UUID> = [],
            expandedDetails: Set<String> = [],
            stream: AssistantMessage? = nil,
            formatTimestamp: @escaping (Date) -> String = { _ in "" }
        ) {
            self.isGenerating = isGenerating
            self.expandedTurns = expandedTurns
            self.expandedDetails = expandedDetails
            self.stream = stream
            self.formatTimestamp = formatTimestamp
        }
    }

    // MARK: - Projection

    /// The full-rebuild result: the rows plus the splice point the coordinator
    /// retains for the next streaming tail-patch.
    struct Projection {
        var rows: [ChatRow]
        /// Index where the last Turn's rows begin (`0` when there are zero
        /// Turns) — the coordinator uses it verbatim as the `replaceSubrange`
        /// splice point, with no recomputation of layout.
        var activeTurnStart: Int
        /// Every committed tool-call row id across all Turns, **regardless of
        /// whether the Turn is expanded** (a collapsed Turn's tool rows are not
        /// in `rows`). The coordinator prunes stale `expandedDetails` against
        /// this so collapsing then re-expanding a Turn preserves tool-detail
        /// expansion state.
        var toolRowIDs: Set<String>
        /// Every still-valid turn id — the committed Turn ids, plus
        /// ``streamingTurnID`` when generating. The coordinator prunes stale
        /// `expandedTurns` against this, so it never has to re-group the log a
        /// second time just to learn which turns survived the rebuild.
        var validTurnIDs: Set<UUID>
    }

    // MARK: - Grouping

    /// A Turn boundary: a user message or a compaction marker. The single source
    /// of the grouping rule, shared by ``turns(from:)`` (forward) and
    /// ``activeTurn(from:)`` (the bounded backward scan), so the full rebuild and
    /// the streaming fast path cannot drift on where a Turn begins.
    static func isTurnBoundary(_ msg: any AgentMessageProtocol) -> Bool {
        msg.asUser != nil || msg is CompactionSummaryMessage
    }

    /// Groups the message log into Turns. A Turn boundary is a user message or a
    /// compaction marker. Leading non-boundary messages (malformed input — the
    /// first message is always a user message in practice) are grouped into a
    /// Turn whose id is the first message's uuid.
    static func turns(from messages: [any AgentMessageProtocol]) -> [Turn] {
        var result: [Turn] = []
        var current: Turn?
        for msg in messages {
            let isBoundary = isTurnBoundary(msg)
            if isBoundary {
                if let current { result.append(current) }
                current = Turn(id: msg.messageUUID, messages: [msg])
            } else {
                if current == nil {
                    current = Turn(id: msg.messageUUID, messages: [])
                }
                current?.messages.append(msg)
            }
        }
        if let current { result.append(current) }
        return result
    }

    // MARK: - Full rebuild

    /// Projects every Turn and appends the generating fallback when there is no
    /// Turn to host it (including zero Turns). Returns the rows and the splice
    /// point (the index where the last Turn's rows begin).
    static func rows(from messages: [any AgentMessageProtocol], _ ctx: Context) -> Projection {
        let turns = turns(from: messages)
        var rows: [ChatRow] = []
        var activeTurnStart = 0
        var toolRowIDs = Set<String>()
        var validTurnIDs = Set<UUID>()

        for (index, turn) in turns.enumerated() {
            let isLast = index == turns.count - 1
            if isLast { activeTurnStart = rows.count }
            let projected = projectTurn(turn, isActive: ctx.isGenerating && isLast, ctx)
            rows += projected.rows
            toolRowIDs.formUnion(projected.toolRowIDs)
            validTurnIDs.insert(turn.id)
        }

        // Zero-Turn generating fallback: generation started but the log is still
        // empty (first message of a conversation, between `.agentStart` and the
        // prompt's `.messageEnd`). No Turn object exists to host the stream.
        if turns.isEmpty && ctx.isGenerating {
            activeTurnStart = rows.count
            rows += generatingFallback(ctx)
        }
        if ctx.isGenerating { validTurnIDs.insert(streamingTurnID) }

        return Projection(
            rows: rows, activeTurnStart: activeTurnStart,
            toolRowIDs: toolRowIDs, validTurnIDs: validTurnIDs
        )
    }

    // MARK: - Tail patch

    /// The active (last) Turn alone — the bounded backward scan the streaming
    /// tail-patch needs, without grouping the entire log into Turns just to take
    /// the last one. Walks back from the end to the first Turn boundary (a user
    /// message or compaction marker), matching ``turns(from:)``'s grouping rule
    /// (including the leading-non-boundary fallback). Equivalent to
    /// `turns(from:).last` but O(active Turn) rather than O(whole log).
    static func activeTurn(from messages: [any AgentMessageProtocol]) -> Turn? {
        guard !messages.isEmpty else { return nil }
        var start = 0
        var id = messages[0].messageUUID
        for index in stride(from: messages.count - 1, through: 0, by: -1) {
            let msg = messages[index]
            if isTurnBoundary(msg) {
                start = index
                id = msg.messageUUID
                break
            }
        }
        return Turn(id: id, messages: Array(messages[start...]))
    }

    /// Projects a single Turn as the active (streaming) turn — the fast path
    /// projects only the last Turn and the coordinator splices the result onto
    /// the stable prefix. The tail patch never prunes, so the projected tool-row
    /// ids are discarded.
    static func rows(for turn: Turn, _ ctx: Context) -> [ChatRow] {
        projectTurn(turn, isActive: true, ctx).rows
    }

    // MARK: - Per-Turn projection (the single shared fold)

    private static func projectTurn(
        _ turn: Turn, isActive: Bool, _ ctx: Context
    ) -> (rows: [ChatRow], toolRowIDs: Set<String>) {
        var rows: [ChatRow] = []
        var toolRowIDs = Set<String>()

        let boundary = turn.messages.first
        let user = boundary?.asUser
        let compaction = boundary as? CompactionSummaryMessage

        // User row — rendered whenever the turn opens with a user message, even
        // image-only (empty content). This converges both paths onto the
        // full-rebuild behaviour.
        if let user {
            rows.append(ChatRow(
                id: turn.id.uuidString,
                kind: .user(UserRow(
                    content: user.content,
                    images: user.images,
                    timestamp: ctx.formatTimestamp(user.timestamp),
                    messageID: turn.id
                ))
            ))
        }

        // System (compaction) row.
        if let compaction {
            rows.append(ChatRow(
                id: turn.id.uuidString + "-system",
                kind: .system(SystemRow(
                    content: compaction.displayText
                ))
            ))
        }

        // Committed assistant body: steps (thinking / intermediate text / tool
        // calls) and the final answer.
        var toolResultMap: [String: ToolResultMessage] = [:]
        for msg in turn.messages {
            if let tr = msg.asToolResult { toolResultMap[tr.toolCallId] = tr }
        }
        let lastAssistant = turn.messages.last(where: { $0.asAssistant != nil })?.asAssistant

        var steps: [ChatRow] = []
        var answer: (content: String, timestamp: Date, id: UUID)?

        for msg in turn.messages {
            if msg is CompactionSummaryMessage { continue }
            if msg.asUser != nil || msg.asToolResult != nil { continue }
            guard let asst = msg.asAssistant else { continue }

            if let thinking = asst.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
               !thinking.isEmpty {
                steps.append(ChatRow(
                    id: "\(asst.id)-thinking",
                    kind: .thinking(ThinkingRow(content: thinking, isLast: false))
                ))
            }

            let isFinalAnswer = asst.id == lastAssistant?.id && asst.toolCalls.isEmpty
            let trimmed = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)

            if !trimmed.isEmpty && !isFinalAnswer {
                steps.append(ChatRow(
                    id: "\(asst.id)-text",
                    kind: .toolText(ToolTextRow(content: trimmed, isLast: false))
                ))
            }

            for (index, info) in asst.toolCalls.enumerated() {
                let props = ToolDisplayHelpers.displayProps(for: info)
                let result = toolResultMap[info.id]
                let rowID = "\(asst.id)-tool-\(index)"
                toolRowIDs.insert(rowID)
                steps.append(ChatRow(id: rowID, kind: .toolCall(ToolCallRow(
                    displayTitle: props.title,
                    iconName: props.icon,
                    argumentsFormatted: props.argsFormatted,
                    resultContent: result?.content.textContent,
                    resultImages: result.map { $0.content.imageAttachments(namespace: $0.id) } ?? [],
                    isError: result?.isError ?? false,
                    isLast: false,
                    isDetailExpanded: ctx.expandedDetails.contains(rowID),
                    filePath: props.filePath
                ))))
            }

            if isFinalAnswer && !trimmed.isEmpty {
                answer = (trimmed, asst.timestamp, asst.id)
            }
        }

        // An active turn whose committed assistant body renders nothing yet: the
        // streaming overlay / indicator is the whole body. Unified — emitted once
        // (the old full rebuild + tail-patch had a latent duplicate-emit here).
        //
        // Gated on "no committed rows produced" (`steps.isEmpty && answer == nil`),
        // not "no assistant message present" — an assistant message whose content
        // is whitespace-only commits anyway (the commit guard uses a raw
        // `isEmpty`) yet yields no steps and no answer, so the "thinking…"
        // indicator must still show while generation is running.
        if isActive && steps.isEmpty && answer == nil {
            rows += generatingFallback(ctx)
            return (rows, toolRowIDs)
        }

        let liveSteps = isActive ? streamingStepCount(ctx.stream) : 0
        let activeExpanded = ctx.expandedTurns.contains(streamingTurnID)
        let committedExpanded = ctx.expandedTurns.contains(turn.id)

        // Header: the streaming header for the active turn, or the completed
        // turn header for a turn with steps.
        if isActive {
            if let header = streamingHeaderRow(
                totalStepCount: steps.count + liveSteps, isExpanded: activeExpanded
            ) {
                rows.append(header)
            }
        } else if !steps.isEmpty {
            rows.append(ChatRow(
                id: "\(turn.id)-header",
                kind: .turnHeader(TurnHeaderRow(
                    stepCount: steps.count,
                    isGenerating: false,
                    turnID: turn.id,
                    isExpanded: committedExpanded
                ))
            ))
        }

        // Committed step rows (when expanded). The last committed step is only
        // stamped `isLast` when no live streaming steps follow it.
        if !steps.isEmpty {
            let showSteps = isActive ? activeExpanded : committedExpanded
            if showSteps {
                let hasLiveStepsAfter = liveSteps > 0
                for (index, step) in steps.enumerated() {
                    let isLast = index == steps.count - 1 && !hasLiveStepsAfter
                    rows.append(isLast ? step.withIsLast(true) : step)
                }
            }
        }

        // Live streaming step rows for the active turn.
        if isActive {
            rows += streamingStepRows(ctx)
        }

        // Final answer row, appended after any live streaming rows above. Safe
        // only because a committed final answer and an active stream never
        // coexist within one Turn today: `messageEnd` nils the stream before the
        // answer commits, and follow-up turns aren't wired (`getFollowUpMessages`
        // drains empty; `prompt` guards on `.idle`). If a second loop turn in the
        // same transcript Turn is ever enabled, revisit this ordering — the older
        // committed answer would otherwise render below the newer live stream.
        if let answer {
            rows.append(ChatRow(
                id: answer.id.uuidString + "-answer",
                kind: .assistantText(AssistantTextRow(
                    content: answer.content,
                    timestamp: ctx.formatTimestamp(answer.timestamp),
                    messageID: answer.id,
                    hasStepsAbove: !steps.isEmpty
                ))
            ))
        }

        return (rows, toolRowIDs)
    }

    // MARK: - Streaming overlay

    /// The fallback body for a generating state with no committed assistant
    /// message to host the stream (zero Turns, or an active Turn that is still
    /// just a user message).
    private static func generatingFallback(_ ctx: Context) -> [ChatRow] {
        let liveSteps = streamingStepCount(ctx.stream)
        var rows: [ChatRow] = []
        if liveSteps > 0 {
            if let header = streamingHeaderRow(
                totalStepCount: liveSteps,
                isExpanded: ctx.expandedTurns.contains(streamingTurnID)
            ) {
                rows.append(header)
            }
            rows += streamingStepRows(ctx)
        } else if ctx.stream == nil {
            rows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
        } else {
            let streamed = streamingStepRows(ctx)
            rows += streamed
            // A stream exists but produced no rows yet — fall back to the bare
            // indicator. A structural emptiness check, not a row-id prefix sniff.
            if streamed.isEmpty {
                rows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
            }
        }
        return rows
    }

    /// Counts live streaming steps from the current stream — the "(n steps)"
    /// header badge. Encodes the same step-emission rule as ``streamingStepRows``
    /// (thinking + each tool call + trailing text when tools are present; a
    /// trailing text with no tools is the answer, not a step). The two are pinned
    /// in lockstep by `streamingHeaderCountMatchesRenderedStepRows` — keep them in
    /// step when changing what counts as a streaming step.
    private static func streamingStepCount(_ stream: AssistantMessage?) -> Int {
        guard let stream else { return 0 }
        var count = 0
        if let thinking = stream.thinking,
           !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            count += 1
        }
        count += stream.toolCalls.count
        let trimmed = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !stream.toolCalls.isEmpty && !trimmed.isEmpty { count += 1 }
        return count
    }

    /// The unified streaming header row. `nil` when there are no steps to head.
    private static func streamingHeaderRow(totalStepCount: Int, isExpanded: Bool) -> ChatRow? {
        guard totalStepCount > 0 else { return nil }
        return ChatRow(
            id: "streaming-header",
            kind: .turnHeader(TurnHeaderRow(
                stepCount: totalStepCount,
                isGenerating: true,
                turnID: streamingTurnID,
                isExpanded: isExpanded
            ))
        )
    }

    /// Live streaming step rows (thinking, tool calls, text). No header.
    private static func streamingStepRows(_ ctx: Context) -> [ChatRow] {
        guard let stream = ctx.stream else { return [] }
        var rows: [ChatRow] = []
        let trimmed = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)

        guard ctx.expandedTurns.contains(streamingTurnID) else {
            // Collapsed — only the final streaming answer (plain text, no tools).
            if !trimmed.isEmpty && stream.toolCalls.isEmpty {
                rows.append(ChatRow(
                    id: "streaming-answer",
                    kind: .streamingText(StreamingTextRow(content: trimmed))
                ))
            }
            return rows
        }

        // Streaming thinking.
        if let thinking = stream.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
           !thinking.isEmpty {
            rows.append(ChatRow(
                id: "streaming-thinking",
                kind: .thinking(ThinkingRow(
                    content: thinking,
                    isLast: stream.toolCalls.isEmpty && trimmed.isEmpty
                ))
            ))
        }

        // Streaming tool calls.
        for (index, info) in stream.toolCalls.enumerated() {
            let rowID = "streaming-tool-\(index)"
            let props = ToolDisplayHelpers.displayProps(for: info)
            rows.append(ChatRow(
                id: rowID,
                kind: .toolCall(ToolCallRow(
                    displayTitle: props.title,
                    iconName: props.icon,
                    argumentsFormatted: props.argsFormatted,
                    resultContent: nil,
                    resultImages: [],
                    isError: false,
                    isLast: index == stream.toolCalls.count - 1 && trimmed.isEmpty,
                    isDetailExpanded: ctx.expandedDetails.contains(rowID),
                    filePath: props.filePath
                ))
            ))
        }

        // Streaming text.
        if !trimmed.isEmpty {
            if stream.toolCalls.isEmpty {
                rows.append(ChatRow(
                    id: "streaming-answer",
                    kind: .streamingText(StreamingTextRow(content: trimmed))
                ))
            } else {
                rows.append(ChatRow(
                    id: "streaming-text",
                    kind: .toolText(ToolTextRow(content: trimmed, isLast: true))
                ))
            }
        }

        return rows
    }
}
