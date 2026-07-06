//
//  ChatTranscriptView.swift
//  tesseract
//
//  The flat document transcript (ADR-0024): committed `ChatItem` value rows
//  plus the one Live Part, laid out in a readable column. Auto-follows the
//  stream while the user is near the bottom; committed rows never re-render
//  on a token delta — only the Live Part's `Text` invalidates.
//

import SwiftUI

struct ChatTranscriptView: View {
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(ChatSession.self) private var session
    @State private var isNearBottom = true
    @State private var pendingScrollTask: Task<Void, Never>?

    private let bottomAnchorID = "chat-transcript-bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                transcriptStack
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .frame(maxWidth: ChatLayout.columnMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            .defaultScrollAnchor(session.items.isEmpty ? .top : .bottom)
            // Soft edge under the floating composer; the top edge gets the
            // same treatment from the hosting split-view detail column.
            .scrollEdgeEffectStyle(.soft, for: .bottom)
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.contentSize.height > 0 && geo.visibleRect.maxY >= geo.contentSize.height - 80
            } action: { _, nearBottom in
                isNearBottom = nearBottom
            }
            .onChange(of: session.items.count) { _, _ in
                if isNearBottom { scheduleScrollToBottom(proxy: proxy) }
            }
            // Streaming auto-follow, isolated so the transcript body doesn't
            // observe the Live Part.
            .overlay {
                LiveScrollTrigger(
                    proxy: proxy,
                    bottomAnchorID: bottomAnchorID,
                    isNearBottom: isNearBottom
                )
            }
            .overlay {
                if session.items.isEmpty && !session.isGenerating {
                    emptyState
                }
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Rows

    /// SwiftUI's lazy prefetch path still re-enters AppKit constraint updates
    /// under high-frequency row mutation + auto-scroll. Keep the stack eager
    /// while a generation is actively streaming, then return to lazy loading
    /// for idle conversations.
    @ViewBuilder
    private var transcriptStack: some View {
        if session.isGenerating {
            VStack(alignment: .leading, spacing: 16) {
                rowContents
            }
        } else {
            LazyVStack(alignment: .leading, spacing: 16) {
                rowContents
            }
        }
    }

    @ViewBuilder
    private var rowContents: some View {
        SystemPromptSection()

        ForEach(session.items) { item in
            ChatItemRow(
                item: item,
                speakingMessageID: $speakingMessageID,
                isSpeechActive: isSpeechActive
            )
        }

        LiveMessageSection()

        Color.clear
            .frame(height: 1)
            .id(bottomAnchorID)
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

    private func scheduleScrollToBottom(proxy: ScrollViewProxy) {
        pendingScrollTask = deferredScrollToBottom(
            proxy, anchor: bottomAnchorID, cancelling: pendingScrollTask)
    }
}

/// Defers a scroll-to-bottom until the current layout pass completes.
/// Triggering `scrollTo` synchronously while SwiftUI is still reconciling a
/// lazy stack can provoke AppKit constraint exceptions in Release builds.
/// Returns the deferred task; callers keep it so the next call (or
/// `onDisappear`) can cancel it.
@MainActor
private func deferredScrollToBottom(
    _ proxy: ScrollViewProxy, anchor: String, cancelling task: Task<Void, Never>?
) -> Task<Void, Never> {
    task?.cancel()
    return Task { @MainActor in
        await Task.yield()
        guard !Task.isCancelled else { return }
        proxy.scrollTo(anchor, anchor: .bottom)
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
            if let skill = SkillInvocationBlock.parse(message.content) {
                SkillInvocationRowView(
                    block: skill, images: message.images, timestamp: message.timestamp)
            } else {
                UserMessageRow(message: message)
            }

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

    var body: some View {
        if let message = session.liveMessage {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(message.content.enumerated()), id: \.offset) { index, part in
                    if let live = session.livePart, live.partIndex == index {
                        LivePartView(live: live)
                    } else {
                        AssistantPartView(part: part)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        if session.runPhase == .streaming, session.livePart == nil, session.isGenerating {
            // Waiting on the model (prefill, or between parts) — a quiet pulse.
            ProgressView()
                .controlSize(.small)
                .padding(.vertical, 2)
        }
    }
}

/// Reads only the Live Part's `displayText` — fires auto-follow at the
/// throttled republish rate without re-evaluating the transcript body.
private struct LiveScrollTrigger: View {
    let proxy: ScrollViewProxy
    let bottomAnchorID: String
    let isNearBottom: Bool

    @Environment(ChatSession.self) private var session
    @State private var pendingScrollTask: Task<Void, Never>?

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: session.livePart?.displayText) { _, newValue in
                if isNearBottom, newValue != nil {
                    scheduleScrollToBottom()
                }
            }
            .onChange(of: session.liveMessage?.content.count) { _, newValue in
                if isNearBottom, newValue != nil {
                    scheduleScrollToBottom()
                }
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
    }

    private func scheduleScrollToBottom() {
        pendingScrollTask = deferredScrollToBottom(
            proxy, anchor: bottomAnchorID, cancelling: pendingScrollTask)
    }
}

/// Reads only the System Prompt Inspector — changes don't diff the ForEach.
private struct SystemPromptSection: View {
    @Environment(AgentSystemPromptInspector.self) private var inspector

    var body: some View {
        if !inspector.assembledSystemPrompt.isEmpty {
            AgentSystemPromptView()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
