import SwiftUI
import os

struct AgentConversationListView: View {
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(AgentCoordinator.self) private var coordinator
    @State private var isNearBottom: Bool = true
    @State private var pendingScrollTask: Task<Void, Never>?

    private var lastRowID: String? { coordinator.rows.last?.id }
    private let bottomAnchorID = "agent-conversation-bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    SystemPromptSection()

                    EmptyStateSection()

                    ForEach(coordinator.rows) { row in
                        ChatRowView(
                            row: row,
                            speakingMessageID: $speakingMessageID,
                            isSpeechActive: isSpeechActive
                        )
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 2)
                    }

                    Color.clear
                        .frame(height: 1)
                        .id(bottomAnchorID)
                }
                .padding(.vertical, 8)
            }
            .defaultScrollAnchor(.bottom)
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.contentSize.height > 0 &&
                geo.visibleRect.maxY >= geo.contentSize.height - 80
            } action: { _, nearBottom in
                isNearBottom = nearBottom
            }
            // [Perf] Track content height changes — reveals layout instability causing scroll jumps
            .onScrollGeometryChange(for: CGFloat.self) { geo in
                geo.contentSize.height
            } action: { oldHeight, newHeight in
                let delta = newHeight - oldHeight
                if abs(delta) > 1 {
                    Log.agent.debug("[Perf] ScrollGeo contentHeight: \(String(format: "%.0f", oldHeight)) → \(String(format: "%.0f", newHeight)) (Δ\(String(format: "%.0f", delta))) | isNearBottom=\(isNearBottom)")
                    ChatViewPerf.signposter.emitEvent("ContentHeightChange")
                }
            }
            .onChange(of: lastRowID) { _, newID in
                if isNearBottom, newID != nil {
                    Log.agent.debug("[Perf] scrollTo(bottomAnchor) triggered")
                    scheduleScrollToBottom(proxy: proxy)
                }
            }
            // Separate view to isolate streamingRowVersion observation from the List body
            .overlay {
                StreamingScrollTrigger(
                    proxy: proxy,
                    bottomAnchorID: bottomAnchorID,
                    isNearBottom: isNearBottom
                )
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Defers the scroll until the current layout pass completes. Triggering
    /// `scrollTo` synchronously while SwiftUI is still reconciling a lazy stack
    /// can provoke AppKit constraint exceptions in Release builds.
    private func scheduleScrollToBottom(proxy: ScrollViewProxy) {
        pendingScrollTask?.cancel()
        pendingScrollTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
        }
    }
}

// MARK: - Isolated Observation Sub-Views

/// Reads only `assembledSystemPrompt` — changes don't trigger ForEach diffing.
private struct SystemPromptSection: View {
    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        if !coordinator.assembledSystemPrompt.isEmpty {
            AgentSystemPromptView()
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.vertical, 2)
        }
    }
}

/// Self-contained empty state — keeps logic out of the main List body.
private struct EmptyStateSection: View {
    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        if coordinator.rows.isEmpty && !coordinator.isGenerating {
            VStack(spacing: 8) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 40))
                    .foregroundStyle(.quaternary)
                Text("Start a conversation")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 80)
            .padding(.horizontal, 16)
            .padding(.vertical, 2)
        }
    }
}

/// Reads only `streamingRowVersion` — fires scroll-to-bottom without re-evaluating the List body.
private struct StreamingScrollTrigger: View {
    let proxy: ScrollViewProxy
    let bottomAnchorID: String
    let isNearBottom: Bool

    @Environment(AgentCoordinator.self) private var coordinator
    @State private var pendingScrollTask: Task<Void, Never>?

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: coordinator.streamingRowVersion) { _, _ in
                if isNearBottom, coordinator.rows.last != nil {
                    scheduleScrollToBottom()
                }
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
    }

    private func scheduleScrollToBottom() {
        pendingScrollTask?.cancel()
        pendingScrollTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
        }
    }
}
