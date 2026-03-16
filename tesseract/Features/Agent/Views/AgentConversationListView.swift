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
                if !coordinator.assembledSystemPrompt.isEmpty {
                    AgentSystemPromptView()
                        .listRowSeparator(.hidden)
                        .listRowInsets(EdgeInsets(top: 2, leading: 16, bottom: 2, trailing: 16))
                }

                if coordinator.rows.isEmpty && !coordinator.isGenerating {
                    emptyState
                        .listRowSeparator(.hidden)
                        .listRowInsets(EdgeInsets(top: 2, leading: 16, bottom: 2, trailing: 16))
                }

                ForEach(coordinator.rows) { row in
                    ChatRowView(
                        row: row,
                        speakingMessageID: $speakingMessageID,
                        isSpeechActive: isSpeechActive
                    )
                    .onGeometryChange(for: CGFloat.self) { proxy in
                        proxy.size.height
                    } action: { _, newHeight in
                        if row.heightCacheable {
                            coordinator.cacheRowHeight(newHeight, for: row.id)
                        }
                    }
                    .frame(height: row.heightCacheable ? coordinator.cachedHeight(for: row.id) : nil)
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
            .onChange(of: coordinator.streamingRowVersion) { _, _ in
                // Fires when streaming content grows within a stable last row ID.
                if isNearBottom, let id = lastRowID {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Empty State

    private var emptyState: some View {
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
    }
}
