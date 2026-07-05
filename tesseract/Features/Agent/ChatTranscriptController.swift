//
//  ChatTranscriptController.swift
//  tesseract
//
//  The **Chat Transcript Controller**: the `@Observable @MainActor` stateful
//  driver of the pure ``ChatTranscript`` fold, carved out of `AgentCoordinator`.
//  It owns the transcript's view-interaction state — expansion sets, the
//  streaming-throttle clock, the auto-expand bookkeeping, the active-turn splice
//  index, and the timestamp formatter — and publishes the `rows` /
//  `streamingRowVersion` the chat list renders.
//
//  Publisher-agnostic (ARCHITECTURE.md's panel-controller rule): the coordinator's
//  event dispatcher *feeds* it `messages`, the live `stream`, and `isGenerating`
//  per call, so it holds no `Agent` reference. The `isGenerating`-before-rebuild
//  ordering invariant lives in the dispatcher that drives it, not here.
//

import Foundation
import Observation
import os

@Observable @MainActor
final class ChatTranscriptController {

    // MARK: - Observable State

    /// Flat, pre-computed rows for the chat List. Equatable elements enable
    /// SwiftUI skip-rendering.
    private(set) var rows: [ChatRow] = []

    /// Bumped on each streaming row update — cheap change signal for scroll tracking.
    private(set) var streamingRowVersion: Int = 0

    // MARK: - Interaction State

    /// Expanded turn headers (step timeline visible).
    @ObservationIgnored private var expandedTurns: Set<UUID> = []
    /// Expanded tool call details (arguments/results visible), keyed by row ID.
    @ObservationIgnored private var expandedDetails: Set<String> = []
    /// Throttle streaming row updates.
    @ObservationIgnored private var lastStreamingUpdate: ContinuousClock.Instant = .now
    /// Turn that was auto-expanded for generation — auto-collapse on agentEnd.
    @ObservationIgnored private var autoExpandedTurnID: UUID?
    /// Tracks if user manually collapsed the streaming header during this generation.
    @ObservationIgnored private var streamingManuallyCollapsed: Bool = false
    /// Index in `rows` where the active (last) turn starts — for incremental streaming patches.
    @ObservationIgnored private var activeTurnRowIndex: Int = 0

    /// Minimum interval between streaming tail-patches. Injectable so tests can
    /// disable throttling (`.zero`) to exercise the splice deterministically.
    @ObservationIgnored private let streamingThrottle: Duration

    /// Shared date formatter for timestamps — avoids creating formatters per row.
    @ObservationIgnored private let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()

    // MARK: - Init

    init(streamingThrottle: Duration = .milliseconds(50)) {
        self.streamingThrottle = streamingThrottle
    }

    // MARK: - Full rebuild

    /// Projects the message log into rows via the pure ``ChatTranscript`` module.
    /// Makes the auto-expand decision, asks the module for the rows + splice
    /// point, then prunes stale expansion state — all explicit steps.
    func rebuild(
        messages: [any AgentMessageProtocol], stream: AssistantMessage?, isGenerating: Bool
    ) {
        let perfState = ChatViewPerf.signposter.beginInterval("rebuildRows")

        applyStreamingHeaderAutoExpand(isGenerating: isGenerating)

        let projection = ChatTranscript.rows(
            from: messages, makeContext(stream: stream, isGenerating: isGenerating))
        rows = projection.rows
        activeTurnRowIndex = projection.activeTurnStart
        pruneExpansionState(
            validTurnIDs: projection.validTurnIDs, detailRowIDs: projection.detailRowIDs)

        ChatViewPerf.signposter.endInterval("rebuildRows", perfState)
    }

    // MARK: - Streaming tail-patch

    /// Throttled streaming update — patches only the active turn's rows and bumps
    /// `streamingRowVersion`. Re-projects only the active (last) turn and splices
    /// it onto the stable prefix.
    func patchStreamingTail(
        messages: [any AgentMessageProtocol], stream: AssistantMessage?, isGenerating: Bool
    ) {
        let now = ContinuousClock.now
        guard now - lastStreamingUpdate >= streamingThrottle else { return }
        lastStreamingUpdate = now

        ChatViewPerf.signposter.emitEvent("updateStreamingRows")
        applyTailPatch(messages: messages, stream: stream, isGenerating: isGenerating)
        streamingRowVersion &+= 1
    }

    private func applyTailPatch(
        messages: [any AgentMessageProtocol], stream: AssistantMessage?, isGenerating: Bool
    ) {
        guard isGenerating, activeTurnRowIndex <= rows.count else {
            rebuild(messages: messages, stream: stream, isGenerating: isGenerating)
            return
        }

        applyStreamingHeaderAutoExpand(isGenerating: isGenerating)

        guard let activeTurn = ChatTranscript.activeTurn(from: messages) else {
            // Zero-Turn while generating — the full rebuild owns that fallback.
            rebuild(messages: messages, stream: stream, isGenerating: isGenerating)
            return
        }

        let tailRows = ChatTranscript.rows(
            for: activeTurn, makeContext(stream: stream, isGenerating: isGenerating))
        let spliceIndex = min(activeTurnRowIndex, rows.count)
        rows.replaceSubrange(spliceIndex..., with: tailRows)
    }

    // MARK: - Event-driven transitions (fed by the dispatcher)

    /// `.turnEnd`: auto-expand the last turn on the first turnEnd, then rebuild.
    func onTurnEnd(
        messages: [any AgentMessageProtocol], stream: AssistantMessage?, isGenerating: Bool
    ) {
        // By now the user message is committed and the assistant turn exists.
        if autoExpandedTurnID == nil {
            autoExpandLastTurn(messages: messages)
        }
        rebuild(messages: messages, stream: stream, isGenerating: isGenerating)
    }

    /// `.agentEnd`: clear the streaming-header expansion bookkeeping, auto-collapse
    /// the turn that generation auto-expanded, then rebuild. The agent has ended,
    /// so this rebuilds as **not generating** unconditionally — the terminal-state
    /// contract is encoded here, not left to the caller to pass `isGenerating: false`.
    func onAgentEnded(messages: [any AgentMessageProtocol], stream: AssistantMessage?) {
        streamingManuallyCollapsed = false
        expandedTurns.remove(ChatTranscript.streamingTurnID)
        autoCollapseIfNeeded()
        rebuild(messages: messages, stream: stream, isGenerating: false)
    }

    // MARK: - Expand / Collapse

    func toggleTurnExpanded(
        _ turnID: UUID, messages: [any AgentMessageProtocol], stream: AssistantMessage?,
        isGenerating: Bool
    ) {
        expandedTurns.formSymmetricDifference([turnID])
        // Track manual collapse of the streaming header.
        if turnID == ChatTranscript.streamingTurnID
            && !expandedTurns.contains(ChatTranscript.streamingTurnID)
        {
            streamingManuallyCollapsed = true
        }
        rebuild(messages: messages, stream: stream, isGenerating: isGenerating)
    }

    func toggleDetailExpanded(_ rowID: String) {
        expandedDetails.formSymmetricDifference([rowID])
        if let idx = rows.firstIndex(where: { $0.id == rowID }),
            case .toolCall(let data) = rows[idx].kind
        {
            rows[idx] = ChatRow(id: rowID, kind: .toolCall(data.togglingDetail()))
        }
    }

    // MARK: - Reset

    func reset() {
        expandedTurns.removeAll()
        expandedDetails.removeAll()
        autoExpandedTurnID = nil
        streamingManuallyCollapsed = false
        activeTurnRowIndex = 0
        rows = []
    }

    // MARK: - Private

    /// Builds the projection context from current controller state plus the fed
    /// `stream` / `isGenerating`. The injected timestamp formatter keeps the
    /// projection pure and deterministic.
    private func makeContext(stream: AssistantMessage?, isGenerating: Bool)
        -> ChatTranscript.Context
    {
        ChatTranscript.Context(
            isGenerating: isGenerating,
            expandedTurns: expandedTurns,
            expandedDetails: expandedDetails,
            stream: stream,
            formatTimestamp: { [timeFormatter] date in timeFormatter.string(from: date) }
        )
    }

    /// Auto-expand the streaming header unless the user manually collapsed it —
    /// computed before projecting.
    private func applyStreamingHeaderAutoExpand(isGenerating: Bool) {
        if isGenerating && !streamingManuallyCollapsed {
            expandedTurns.insert(ChatTranscript.streamingTurnID)
        }
    }

    /// Full-rebuild-only pruning of stale expansion state against the projection's
    /// valid-turn and committed tool-row id sets.
    private func pruneExpansionState(validTurnIDs: Set<UUID>, detailRowIDs: Set<String>) {
        expandedTurns = expandedTurns.intersection(validTurnIDs)
        if !expandedDetails.isEmpty {
            expandedDetails = expandedDetails.intersection(detailRowIDs)
        }
    }

    /// Auto-expand the last assistant turn when generation starts.
    private func autoExpandLastTurn(messages: [any AgentMessageProtocol]) {
        for msg in messages.reversed() {
            if let u = msg.asUser {
                if !expandedTurns.contains(u.id) {
                    expandedTurns.insert(u.id)
                    autoExpandedTurnID = u.id
                }
                return
            }
        }
    }

    /// Auto-collapse the turn that was auto-expanded, unless user manually toggled it.
    private func autoCollapseIfNeeded() {
        if let turnID = autoExpandedTurnID, expandedTurns.contains(turnID) {
            expandedTurns.remove(turnID)
        }
        autoExpandedTurnID = nil
    }
}
