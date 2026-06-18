import Foundation
import MLXLMCommon

/// The server's **Generation Projection**: maps a *terminal*
/// `GenerationAccumulator` (plus the turn's completion `info`, the request's
/// effective max-tokens, and the completion id) to the content both HTTP
/// completion paths emit — the `finish_reason`, the malformed-fallback-applied
/// text, the reasoning, the tool calls, the thinking-safeguard sidecar, and a
/// finish-reason **diagnostic**.
///
/// A pure `nonisolated Sendable` value: no logging, no I/O, no transport. Each
/// completion path builds it **once** and keeps only its own framing — the
/// streaming path chunks SSE, the non-streaming path encodes one JSON body. The
/// shared rules (finish-reason, empty-payload diagnostic, safeguard sidecar) live
/// here, so changing one means editing one place instead of two mirrored paths;
/// the malformed→text fallback *predicate* lives once on the accumulator
/// (`surfacesMalformedBuffer`), shared with the agent path's
/// `AssistantMessageProjection`, and this projection applies it to the text.
///
/// The diagnostic is classified **once, on pre-fallback state**, so both paths
/// classify identically; the caller logs it via `FinishReasonDiagnostic.emit`.
///
/// See `CONTEXT.md` → Language → *Generation Projection*.
nonisolated struct CompletionProjection: Sendable {
    /// The OpenAI `finish_reason` for this turn. One rule, one home.
    let finishReason: OpenAI.FinishReason

    /// Assistant text with the malformed→text fallback already applied: when the
    /// turn produced no text and no tool calls but captured a malformed
    /// `<tool_call>` buffer, that raw buffer is surfaced here.
    let textContent: String

    /// `accumulator.thinking ?? ""` — collapses the nil-vs-empty distinction the
    /// HTTP response cannot carry, but only at this boundary and no earlier.
    let thinkingContent: String

    /// Finalized tool calls in arrival order.
    let toolCalls: [ToolCall]

    /// The thinking-safeguard sidecar, present iff the safeguard fired.
    let safeguardReport: OpenAI.ThinkingSafeguardReport?

    /// True when `textContent` is the surfaced malformed buffer. The streaming
    /// path emits one extra SSE content chunk on this flag; the non-streaming
    /// path ignores it (the buffer is already in the JSON body). Both paths emit
    /// the shared "surfaced dropped tool-call buffer" info-log when it is set.
    let malformedFallbackSurfaced: Bool

    /// Completion metrics captured from the terminal `.info` event.
    let info: AgentGeneration.Info?

    /// Finish-reason diagnostic, classified on pre-fallback state. A value the
    /// caller logs — the projection itself stays pure.
    let diagnostic: FinishReasonDiagnostic

    init(
        accumulator: GenerationAccumulator,
        info: AgentGeneration.Info?,
        maxTokens: Int?,
        completionID: String
    ) {
        // Raw, pre-fallback turn state.
        let rawText = accumulator.text
        let calls = accumulator.toolCalls
        let reasoning = accumulator.thinking ?? ""
        let malformedRaw = accumulator.malformedToolCallRaw

        // 1. Compute the finish-reason once, from one rule.
        let finishReason = Self.finishReason(
            hasToolCalls: !calls.isEmpty,
            generationTokenCount: info?.generationTokenCount,
            maxTokens: maxTokens
        )

        // 2. Classify the diagnostic on RAW pre-fallback state — before the
        //    fallback moves the malformed buffer into the text and masks the
        //    empty-payload signal.
        let classification = Self.classify(
            finishReason: finishReason,
            rawText: rawText,
            toolCallCount: calls.count,
            reasoning: reasoning,
            malformedRaw: malformedRaw
        )

        // 3. THEN apply the malformed→text fallback to produce the surfaced text.
        //    The predicate lives once on the accumulator (`surfacesMalformedBuffer`),
        //    shared with the agent path's `AssistantMessageProjection`.
        let surfaced = accumulator.surfacesMalformedBuffer

        self.finishReason = finishReason
        self.textContent = surfaced ? malformedRaw : rawText
        self.thinkingContent = reasoning
        self.toolCalls = calls
        self.safeguardReport = accumulator.safeguardSafePrefixChars.map {
            OpenAI.ThinkingSafeguardReport(safePrefixChars: $0)
        }
        self.malformedFallbackSurfaced = surfaced
        self.info = info
        self.diagnostic = FinishReasonDiagnostic(
            classification: classification,
            completionID: completionID,
            finishReason: finishReason,
            textLen: rawText.count,
            toolCallCount: calls.count,
            reasoningLen: reasoning.count,
            malformedLen: malformedRaw.count,
            generationTokenCount: info?.generationTokenCount ?? 0,
            maxTokens: maxTokens ?? -1,
            stopReason: info.map { describeStopReason($0.stopReason) } ?? "nil"
        )
    }

    /// The single finish-reason rule: tool calls win, otherwise a generation
    /// that hit max-tokens is `.length`, otherwise `.stop`.
    static func finishReason(
        hasToolCalls: Bool,
        generationTokenCount: Int?,
        maxTokens: Int?
    ) -> OpenAI.FinishReason {
        if hasToolCalls {
            return .tool_calls
        }
        if let generationTokenCount, let maxTokens, generationTokenCount >= maxTokens {
            return .length
        }
        return .stop
    }

    /// Classify the terminal turn for the diagnostic log. Precedence matches the
    /// streaming path's historical `if / else if`: an empty-payload `.stop` with
    /// reasoning outranks a dropped malformed tool call.
    private static func classify(
        finishReason: OpenAI.FinishReason,
        rawText: String,
        toolCallCount: Int,
        reasoning: String,
        malformedRaw: String
    ) -> FinishReasonDiagnostic.Classification {
        let stopWithEmptyPayload =
            finishReason == .stop
            && rawText.isEmpty
            && toolCallCount == 0
        if stopWithEmptyPayload && !reasoning.isEmpty {
            return .emptyPayloadWithReasoning
        }
        if !malformedRaw.isEmpty && toolCallCount == 0 {
            return .malformedToolCallDropped
        }
        return .normal
    }
}

/// The finish-reason decision recorded for offline investigation: the
/// classification plus the state inputs that produced it. A `Sendable, Equatable`
/// value — the classification, the rendered line, and the severity (the parts
/// that drifted between the two completion paths) are pure and unit-tested via
/// `renderedLine`/`severity`; `emit` is the convenience that logs them through
/// `Log.server` so neither path repeats the level switch.
nonisolated struct FinishReasonDiagnostic: Sendable, Equatable {
    /// Why this finish-reason fired, and the severity the caller logs at.
    enum Classification: Sendable, Equatable {
        /// Ordinary completion — logged at `info`.
        case normal
        /// `.stop` with empty text and empty tool calls but non-empty reasoning
        /// — the jundot/omlx#825 stale-recurrent-state symptom. Logged at
        /// `warning`.
        case emptyPayloadWithReasoning
        /// A malformed `<tool_call>` buffer was captured and no tool call
        /// survived — the model attempted a call that was dropped. Logged at
        /// `warning`.
        case malformedToolCallDropped
    }

    let classification: Classification
    let completionID: String
    let finishReason: OpenAI.FinishReason
    let textLen: Int
    let toolCallCount: Int
    let reasoningLen: Int
    let malformedLen: Int
    let generationTokenCount: Int
    let maxTokens: Int
    let stopReason: String

    /// The log level the diagnostic is emitted at, derived purely from the
    /// classification. Exposed so the classification→severity mapping — part of
    /// what drifted between the two paths — is assertable without logging.
    enum Severity: Sendable, Equatable {
        case info
        case warning
    }

    var severity: Severity {
        switch classification {
        case .normal:
            return .info
        case .emptyPayloadWithReasoning, .malformedToolCallDropped:
            return .warning
        }
    }

    /// The rendered diagnostic line. `label` distinguishes the path (`"streaming"`
    /// / `"non-streaming"`); the line is otherwise identical, so the same turn
    /// leaves the same fingerprint regardless of which path served it. A pure
    /// function — the caller (or `emit`) does the logging.
    func renderedLine(label: String) -> String {
        let base =
            "HTTP \(label) finish_reason decision — "
            + "completionID=\(completionID) "
            + "finishReason=\(finishReason.rawValue) "
            + "textLen=\(textLen) "
            + "toolCalls=\(toolCallCount) "
            + "reasoningLen=\(reasoningLen) "
            + "malformedLen=\(malformedLen) "
            + "genTokens=\(generationTokenCount) "
            + "maxTokens=\(maxTokens) "
            + "stopReason=\(stopReason)"
        switch classification {
        case .normal:
            return base
        case .emptyPayloadWithReasoning:
            return "\(base) — EMPTY PAYLOAD WITH REASONING"
        case .malformedToolCallDropped:
            return "\(base) — MALFORMED TOOL CALL DROPPED"
        }
    }

    /// Render the line and log it via `Log.server` at the classification's
    /// severity. A thin convenience over the pure `renderedLine`/`severity`; the
    /// callers use this so neither completion path repeats the level switch.
    func emit(label: String) {
        let line = renderedLine(label: label)
        switch severity {
        case .info:
            Log.server.info(line)
        case .warning:
            Log.server.warning(line)
        }
    }
}
