import Foundation
import MLXLMCommon

/// The agent driver's **Generation Projection** onto the parts model: folds the
/// `AgentGeneration` event stream into the ordered `[ContentPart]` of one
/// assistant turn and yields the pi-ai `AssistantMessageEvent`s the driver
/// emits (ADR-0024). Replaces `AssistantMessageProjection`, whose flat
/// content/thinking/toolCalls output the parts model supersedes.
///
/// A pure value with no side effects: the driver owns every emit and log. The
/// message id is minted once per turn and shared by every snapshot, so stream
/// consumers key rows off a stable (message id, part index) identity.
///
/// Part-boundary rule: a part stays open while events of its kind arrive;
/// an event of a different kind closes it (emitting `*_end`) and opens the
/// next (emitting `*_start`). Tool calls stream through the **Open Tool
/// Call**: the part is born at name-lock (`toolcallStart` with a real id and
/// name), grows raw fragments via `toolcallDelta`, and commits in place on
/// the parsed `.toolCall` (`toolcallEnd`, same id). An Open Tool Call lives
/// outside `parts` — snapshots append it — so every path that ends the turn
/// without a parse (raw-text fallback, malformed close, terminal close,
/// abort) retracts it without trace.
nonisolated struct AssistantPartsBuilder: Sendable {

    /// Stable identity for the turn's assistant message.
    let messageID = UUID()

    /// The model id stamped on every snapshot.
    var model: String = ""

    private(set) var parts: [ContentPart] = []
    private var usage = Usage()

    /// The in-flight tool call, born at name-lock. `buffer` accumulates the
    /// raw body fragments (pre-name characters included) — the committed
    /// part's normalized-JSON guarantee does not apply while open.
    private struct OpenToolCall {
        let id: String
        let name: String
        var buffer: String
    }

    private var openToolCall: OpenToolCall?

    /// Raw body accumulated before the name locks — no part exists yet.
    private var preNameBuffer = ""

    /// Whether generation stopped at the token limit (from the terminal
    /// `.info` metrics) — maps to pi-ai's `length` stop reason.
    private(set) var hitLengthLimit = false

    /// Index into `parts` of the currently-open (streaming) part, if any.
    private var openIndex: Int?

    /// What the driver should do for one folded event.
    enum Step: Sendable {
        /// Emit `messageUpdate` for each event, in order.
        case events([AssistantMessageEvent])
        /// A `<tool_call>` block failed to parse: the driver logs the warning
        /// and emits the distinct `.malformedToolCall` agent event.
        case malformed(raw: String)
        /// Nothing to emit.
        case silent
    }

    // MARK: - Fold

    mutating func ingest(_ event: AgentGeneration) -> Step {
        switch event {
        case .text(let chunk):
            return .events(appendText(chunk))

        case .thinkStart:
            retractOpenToolCall()
            var events = closeOpenPart()
            parts.append(.thinking(ThinkingPart(thinking: "")))
            openIndex = parts.count - 1
            events.append(.thinkingStart(contentIndex: openIndex!, partial: snapshot()))
            return .events(events)

        case .thinking(let chunk):
            return .events(appendThinking(chunk))

        case .thinkEnd:
            return .events(closeOpenPart())

        case .thinkReclassify:
            // `<think>` never closed: the buffered thinking is really visible
            // text. Convert the open thinking part in place (order preserved —
            // reclassify only fires at end of generation, so nothing follows).
            guard let index = openIndex, case .thinking(let t) = parts[index] else {
                return .silent
            }
            // Open-empty rule again: `textStart`'s snapshot shows an empty
            // part; the buffered thinking arrives as one delta.
            parts[index] = .text(TextPart(text: ""))
            let startPartial = snapshot()
            parts[index] = .text(TextPart(text: t.thinking))
            openIndex = index
            return .events([
                .textStart(contentIndex: index, partial: startPartial),
                .textDelta(contentIndex: index, delta: t.thinking, partial: snapshot()),
            ])

        case .thinkTruncate(let safePrefix):
            // Thinking-loop safeguard fired: the safe prefix becomes the
            // canonical reasoning; the part closes so any later thinking opens
            // a fresh part.
            guard let index = openIndex, case .thinking = parts[index] else { return .silent }
            parts[index] = .thinking(ThinkingPart(thinking: safePrefix))
            openIndex = nil
            return .events([
                .thinkingEnd(contentIndex: index, content: safePrefix, partial: snapshot())
            ])

        case .toolCall(let call):
            let normalized = ToolArgumentNormalizer.encode(call.function.arguments)
            if let open = openToolCall {
                // Commit the Open Tool Call in place: same id (row identity
                // and the Tool Clock key survive), authoritative name, and
                // the raw buffer replaced by the normalized arguments.
                openToolCall = nil
                let part = ToolCallPart(
                    id: open.id, name: call.function.name, argumentsJSON: normalized)
                parts.append(.toolCall(part))
                let index = parts.count - 1
                return .events([
                    .toolcallEnd(contentIndex: index, toolCall: part, partial: snapshot())
                ])
            }
            // No deltas preceded the parse (or the name never locked) — the
            // original atomic start + end pair.
            preNameBuffer = ""
            var events = closeOpenPart()
            let part = ToolCallPart(
                id: UUID().uuidString,
                name: call.function.name,
                argumentsJSON: normalized
            )
            parts.append(.toolCall(part))
            let index = parts.count - 1
            let partial = snapshot()
            events.append(.toolcallStart(contentIndex: index, partial: partial))
            events.append(.toolcallEnd(contentIndex: index, toolCall: part, partial: partial))
            return .events(events)

        case .malformedToolCall(let raw):
            // The block closed unparseable — the Open Tool Call vanishes
            // without trace (the malformed-buffer surfacing owns recovery).
            retractOpenToolCall()
            return .malformed(raw: raw)

        case .info(let info):
            usage.input = info.promptTokenCount
            usage.output = info.generationTokenCount
            usage.totalTokens = info.promptTokenCount + info.generationTokenCount
            if case .length = info.stopReason { hitLengthLimit = true }
            return .silent

        case .toolCallDelta(let name, let argumentsDelta):
            let events = appendToolCallDelta(name: name, delta: argumentsDelta)
            return events.isEmpty ? .silent : .events(events)
        }
    }

    /// Close any still-open part at end of stream, returning its `*_end`
    /// event. Call before `finalize` so the event protocol terminates cleanly.
    /// An Open Tool Call that never parsed is retracted, not closed — the
    /// terminal message must not carry an unexecutable call.
    mutating func closeForTerminal() -> [AssistantMessageEvent] {
        retractOpenToolCall()
        return closeOpenPart()
    }

    /// The vanish-without-trace rule: drop the in-flight tool-call state,
    /// emitting nothing — the next event's partial (or the terminal message)
    /// simply no longer contains the part, and consumers resync from the
    /// snapshot they carry. The driver's abort/error paths call this directly
    /// so a half-written call is never persisted or executed.
    mutating func retractOpenToolCall() {
        openToolCall = nil
        preNameBuffer = ""
    }

    /// The pi-ai stop reason for a successful end of turn.
    var terminalStopReason: StopReason {
        if parts.contains(where: { if case .toolCall = $0 { return true } else { return false } }) {
            return .toolUse
        }
        return hitLengthLimit ? .length : .stop
    }

    // MARK: - Snapshots

    /// The turn so far. Used for every per-event partial and for the
    /// cancel/error paths, which must preserve the partial message exactly.
    /// An Open Tool Call appears as the trailing part, its `argumentsJSON`
    /// holding the raw accumulated buffer (terminal paths retract it first).
    func snapshot(stopReason: StopReason = .stop, errorMessage: String? = nil) -> AssistantMessage {
        var content = parts
        if let open = openToolCall {
            content.append(
                .toolCall(ToolCallPart(id: open.id, name: open.name, argumentsJSON: open.buffer)))
        }
        return AssistantMessage(
            id: messageID,
            content: content,
            model: model,
            usage: usage,
            stopReason: stopReason,
            errorMessage: errorMessage
        )
    }

    /// The committed terminal message. When the shared malformed-buffer
    /// predicate fired (`GenerationAccumulator.surfacesMalformedBuffer`), the
    /// dropped `<tool_call>` buffer becomes a text part so the turn is
    /// persisted instead of dropped as contentless.
    func finalize(
        stopReason: StopReason,
        surfacingMalformedBuffer raw: String? = nil
    ) -> AssistantMessage {
        var finalParts = parts
        if let raw, !raw.isEmpty {
            finalParts.append(.text(TextPart(text: raw)))
        }
        return AssistantMessage(
            id: messageID,
            content: finalParts,
            model: model,
            usage: usage,
            stopReason: stopReason
        )
    }

    // MARK: - Private

    /// Fold one raw tool-call fragment. Name-lock belongs to the producers
    /// (`ToolCallParser` and `GenerationStreamLoop`'s vendor path, both via
    /// `ToolCallNameLock`): before it fires, nameless fragments pool in
    /// `preNameBuffer` silently; the first named fragment births the Open
    /// Tool Call and `toolcallStart` fires with a real id and name. Afterward
    /// every fragment grows the buffer and emits `toolcallDelta`.
    private mutating func appendToolCallDelta(
        name: String?, delta: String
    ) -> [AssistantMessageEvent] {
        if openToolCall != nil {
            openToolCall?.buffer += delta
            return [.toolcallDelta(contentIndex: parts.count, delta: delta, partial: snapshot())]
        }
        preNameBuffer += delta
        guard let name else { return [] }
        var events = closeOpenPart()
        openToolCall = OpenToolCall(id: UUID().uuidString, name: name, buffer: preNameBuffer)
        preNameBuffer = ""
        events.append(.toolcallStart(contentIndex: parts.count, partial: snapshot()))
        return events
    }

    private mutating func appendText(_ chunk: String) -> [AssistantMessageEvent] {
        retractOpenToolCall()
        if let index = openIndex, case .text(var t) = parts[index] {
            t.text += chunk
            parts[index] = .text(t)
            return [.textDelta(contentIndex: index, delta: chunk, partial: snapshot())]
        }
        // pi-ai boundary semantics: the part opens EMPTY at `*_start`; content
        // only ever arrives as deltas. A consumer that seeds from the start
        // snapshot and then applies deltas must never see the first chunk twice.
        var events = closeOpenPart()
        parts.append(.text(TextPart(text: "")))
        openIndex = parts.count - 1
        events.append(.textStart(contentIndex: openIndex!, partial: snapshot()))
        parts[openIndex!] = .text(TextPart(text: chunk))
        events.append(.textDelta(contentIndex: openIndex!, delta: chunk, partial: snapshot()))
        return events
    }

    private mutating func appendThinking(_ chunk: String) -> [AssistantMessageEvent] {
        retractOpenToolCall()
        if let index = openIndex, case .thinking(var t) = parts[index] {
            t.thinking += chunk
            parts[index] = .thinking(t)
            return [.thinkingDelta(contentIndex: index, delta: chunk, partial: snapshot())]
        }
        // Thinking without an explicit thinkStart — open a part defensively.
        // Same open-empty rule as `appendText`.
        var events = closeOpenPart()
        parts.append(.thinking(ThinkingPart(thinking: "")))
        openIndex = parts.count - 1
        events.append(.thinkingStart(contentIndex: openIndex!, partial: snapshot()))
        parts[openIndex!] = .thinking(ThinkingPart(thinking: chunk))
        events.append(.thinkingDelta(contentIndex: openIndex!, delta: chunk, partial: snapshot()))
        return events
    }

    /// Close the open part, emitting its `*_end` event. `openIndex` never
    /// points at a tool-call part — the in-flight call is the separately
    /// tracked Open Tool Call.
    private mutating func closeOpenPart() -> [AssistantMessageEvent] {
        guard let index = openIndex else { return [] }
        openIndex = nil
        switch parts[index] {
        case .text(let t):
            return [.textEnd(contentIndex: index, content: t.text, partial: snapshot())]
        case .thinking(let t):
            return [.thinkingEnd(contentIndex: index, content: t.thinking, partial: snapshot())]
        case .toolCall:
            return []
        }
    }
}
