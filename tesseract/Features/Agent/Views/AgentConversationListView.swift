import SwiftUI
import os

struct AgentConversationListView: View {
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(AgentCoordinator.self) private var coordinator
    @State private var isNearBottom: Bool = true

    private var lastRowID: String? { coordinator.rows.last?.id }

    var body: some View {
        ScrollViewReader { proxy in
            List {
                SystemPromptSection()

                EmptyStateSection()

                ForEach(coordinator.rows) { row in
                    ChatRowView(
                        row: row,
                        speakingMessageID: $speakingMessageID,
                        isSpeechActive: isSpeechActive
                    )
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 2, leading: 16, bottom: 2, trailing: 16))
                }
            }
            .listStyle(.plain)
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
                if isNearBottom, let id = newID {
                    Log.agent.debug("[Perf] scrollTo(lastRowID) triggered: \(id)")
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
            // Separate view to isolate streamingRowVersion observation from the List body
            .overlay { StreamingScrollTrigger(proxy: proxy, isNearBottom: isNearBottom) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Isolated Observation Sub-Views

/// Reads only `assembledSystemPrompt` — changes don't trigger ForEach diffing.
private struct SystemPromptSection: View {
    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        if !coordinator.assembledSystemPrompt.isEmpty {
            AgentSystemPromptView()
                .listRowSeparator(.hidden)
                .listRowInsets(EdgeInsets(top: 2, leading: 16, bottom: 2, trailing: 16))
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
            .listRowSeparator(.hidden)
            .listRowInsets(EdgeInsets(top: 2, leading: 16, bottom: 2, trailing: 16))
        }
    }
}

/// Reads only `streamingRowVersion` — fires scroll-to-bottom without re-evaluating the List body.
private struct StreamingScrollTrigger: View {
    let proxy: ScrollViewProxy
    let isNearBottom: Bool

    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: coordinator.streamingRowVersion) { _, _ in
                if isNearBottom, let id = coordinator.rows.last?.id {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
    }
}
