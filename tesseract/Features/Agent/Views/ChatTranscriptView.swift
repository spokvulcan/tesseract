//
//  ChatTranscriptView.swift
//  tesseract
//
//  The flat document transcript (ADR-0024): committed `ChatItem` value rows
//  plus the one Live Part, laid out in a readable column. Committed rows
//  never re-render on a token delta — only the Live Part's `Text`
//  invalidates.
//
//  Scrolling contract (rewritten from scratch, per the user's spec):
//   1. Sending a message always lands it at the bottom, immediately.
//   2. While a response streams and the user sits at the bottom, the view
//      follows the growing content.
//   3. When the response finishes, settle smoothly at the absolute bottom.
//   4. The moment the user scrolls up, auto-follow disengages completely;
//      it re-arms only when *they* return to the bottom.
//
//  Mechanics: a `ScrollPosition` for programmatic scrolls (declarative — safe
//  to set mid-update, unlike `ScrollViewReader.scrollTo`), one geometry
//  observer deriving the near-bottom gate (`isPositionedByUser` distinguishes
//  the user's scrolls from our own), and a second geometry observer that
//  follows measured content *growth* — driving off geometry, not token
//  events, is what makes the follow land after layout has the true height.
//

import SwiftUI

struct ChatTranscriptView: View {
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(ChatSession.self) private var session

    @State private var scrollPosition = ScrollPosition()
    @State private var isNearBottom = true
    @State private var autoFollow = true

    var body: some View {
        ScrollView {
            // A plain (eager) VStack, deliberately: LazyVStack + bottom
            // anchoring is a documented macOS minefield (blank realization,
            // scrollTo mis-measure — FB threads 741406/761014), and swapping
            // container types mid-conversation resets the scroll position.
            // Rows are cheap value views; eagerness buys correctness.
            VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
                ForEach(session.items) { item in
                    ChatItemRow(
                        item: item,
                        speakingMessageID: $speakingMessageID,
                        isSpeechActive: isSpeechActive
                    )
                }

                // The Pending Row: the just-sent user message, rendered through
                // the same row view as its committed form — the event-spine
                // commit swaps it into `items` with no visual change.
                if let pending = session.pendingUserMessage {
                    ChatItemRow(
                        item: .user(pending),
                        speakingMessageID: $speakingMessageID,
                        isSpeechActive: isSpeechActive
                    )
                }

                LiveMessageSection()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .frame(maxWidth: ChatLayout.columnMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .scrollPosition($scrollPosition, anchor: .bottom)
        .defaultScrollAnchor(session.items.isEmpty ? .top : .bottom, for: .initialOffset)
        // Soft edge under the floating composer; the top edge gets the
        // same treatment from the hosting split-view detail column.
        .scrollEdgeEffectStyle(.soft, for: .bottom)
        // The auto-follow gate. Re-arms whenever the user is at the bottom;
        // disarms only on a *user* scroll away — content growing under a
        // programmatic pin also momentarily leaves the bottom, but that never
        // has `isPositionedByUser` set.
        .onScrollGeometryChange(for: Bool.self) { geo in
            geo.contentSize.height > 0
                && geo.visibleRect.maxY >= geo.contentSize.height - 80
        } action: { _, nearBottom in
            isNearBottom = nearBottom
            if nearBottom {
                autoFollow = true
            } else if scrollPosition.isPositionedByUser {
                autoFollow = false
            }
        }
        // The follow engine: any measured content growth (token deltas, part
        // commits, tool results, images) pins back to the bottom while the
        // gate is armed.
        .onScrollGeometryChange(for: CGFloat.self) { geo in
            geo.contentSize.height
        } action: { oldHeight, newHeight in
            if autoFollow, newHeight > oldHeight {
                scrollPosition.scrollTo(edge: .bottom)
            }
        }
        // A send always shows the new message: jump to the bottom and re-arm,
        // even if the user had scrolled up into history.
        .onChange(of: session.items.count) { _, _ in
            if case .user = session.items.last {
                autoFollow = true
                scrollPosition.scrollTo(edge: .bottom)
            }
        }
        // The Pending Row is the send signal now — it rises before any item
        // commits (the run may sit queued behind a cold-start model load).
        .onChange(of: session.pendingUserMessage?.id) { _, id in
            if id != nil {
                autoFollow = true
                scrollPosition.scrollTo(edge: .bottom)
            }
        }
        // Conversation switch (first item's identity changes): land at the
        // bottom of the loaded conversation.
        .onChange(of: session.items.first?.id) { _, _ in
            autoFollow = true
            scrollPosition.scrollTo(edge: .bottom)
        }
        // End of a run: settle smoothly at the absolute bottom.
        .onChange(of: session.isGenerating) { _, generating in
            if !generating, autoFollow {
                withAnimation(.smooth) {
                    scrollPosition.scrollTo(edge: .bottom)
                }
            }
        }
        .overlay {
            if session.items.isEmpty && !session.isGenerating {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(.quaternary)
            Text("Start a Conversation")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Committed rows

/// One committed transcript row. A pure switch over the `ChatItem` value —
/// tool results render nothing standalone (they appear under their tool row).
private struct ChatItemRow: View {
    let item: ChatItem
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(ChatSession.self) private var session

    var body: some View {
        switch item {
        case .user(let message):
            if let skill = SkillEnvelope.parse(message.content) {
                SkillInvocationRowView(
                    block: skill, images: message.images, timestamp: message.timestamp)
            } else {
                UserMessageRow(message: message)
            }

        case .assistant(let message) where message.content.allSatisfy(\.isBlankRow):
            // All parts blank — render nothing rather than a zero-height
            // stack that would still consume the Row Rhythm twice.
            EmptyView()

        case .assistant(let message):
            AssistantMessageView(
                message: message,
                isSpeaking: speakingMessageID == message.id && isSpeechActive,
                onPlay: {
                    speakingMessageID = message.id
                    session.speakMessage(message.id)
                },
                onStop: {
                    speakingMessageID = nil
                    session.stopSpeaking()
                }
            )

        case .toolResult:
            EmptyView()

        case .system(_, let text):
            SystemNoteRow(text: text)
        }
    }
}

// MARK: - Live message

/// The in-flight assistant message: committed parts as value rows, the
/// streaming part through its Live Part box. Only this section observes
/// `liveMessage`/`livePart`.
private struct LiveMessageSection: View {
    @Environment(ChatSession.self) private var session
    @Environment(AgentEngine.self) private var agentEngine

    var body: some View {
        // Render the live stack only when it has a visible row. `messageStart`
        // raises `liveMessage` with empty content before prefill; an empty
        // VStack is still a zero-height child of the transcript stack and
        // draws a phantom Row Rhythm gap above the Waiting Row that collapses
        // at the first delta — a visible jump.
        if let message = session.liveMessage,
            session.livePart != nil || message.content.contains(where: { !$0.isBlankRow })
        {
            VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
                ForEach(Array(message.content.enumerated()), id: \.offset) { index, part in
                    if let live = session.livePart, live.partIndex == index {
                        LivePartView(live: live)
                    } else if !part.isBlankRow {
                        AssistantPartView(part: part, messageID: message.id, partIndex: index)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        // The Waiting Row: queued behind the lease (cold start) or in a turn
        // prefill. Gated on `promptStartsThinking` — the turn's first tokens
        // are then guaranteed to be thinking, so the live "Thinking…" row
        // takes over with no geometry change. While the model is still
        // loading, the gate can't be read for the incoming model, so the row
        // shows for all models ("Loading model…" is true regardless); a
        // non-thinking model drops the row once loaded. How a prefill wait
        // should present for non-thinking models is deferred — every model
        // the app currently ships starts its turns thinking.
        if session.showsWaitingRow,
            !agentEngine.isModelLoaded || agentEngine.promptStartsThinking
        {
            WaitingRow(isModelLoaded: agentEngine.isModelLoaded)
        }
    }
}

/// The Waiting Row: the placeholder for a turn that is waiting on the model —
/// a spinner in the marker slot and a stage label, "Loading model…" until the
/// load completes, "Reading context…" through prefill. It clones the live
/// thinking row's collapsed header exactly — a hidden `CollapseMarker` keeps
/// the marker slot's real text metrics (a bare `ProgressView` has no text
/// baseline, so `.firstTextBaseline` would drop it to its bottom edge) and
/// the spinner is overlaid centered on it. The handoff to "Thinking…" then
/// changes nothing but the glyph and the words.
private struct WaitingRow: View {
    let isModelLoaded: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            CollapseMarker(isExpanded: false)
                .hidden()
                .overlay {
                    ProgressView()
                        .controlSize(.mini)
                }
            Text(isModelLoaded ? "Reading context…" : "Loading model…")
                .font(.system(size: chatBodyFontSize, weight: .medium))
                .foregroundStyle(.secondary)
                .contentTransition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: isModelLoaded)
            Spacer(minLength: 0)
        }
    }
}
