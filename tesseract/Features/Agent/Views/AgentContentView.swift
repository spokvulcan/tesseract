import SwiftUI
import os

struct AgentContentView: View {
    @Environment(AgentCoordinator.self) private var coordinator
    @Environment(AgentEngine.self) private var agentEngine
    @EnvironmentObject private var conversationStore: AgentConversationStore
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var inputText = ""
    @State private var showingHistory = false
    @State private var speakingMessageID: UUID?
    @Environment(SettingsManager.self) private var settings
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[settings.selectedAgentModelID] {
            return true
        }
        return false
    }

    private var isSpeechActive: Bool {
        if case .idle = speechCoordinator.state { return false }
        if case .error = speechCoordinator.state { return false }
        return true
    }

    var body: some View {
        VStack(spacing: 0) {
            if agentEngine.isLoading {
                AgentModelLoadingBanner()
            } else if !agentEngine.isModelLoaded && !isModelDownloaded {
                AgentModelNotDownloadedBanner()
            }

            if let error = coordinator.error {
                AgentErrorBanner(message: error, onDismiss: { coordinator.error = nil })
            }

            if case .error(let message) = coordinator.voiceState {
                AgentVoiceErrorBanner(message: message)
            }

            AgentConversationListView(
                speakingMessageID: $speakingMessageID,
                isSpeechActive: isSpeechActive
            )
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 0) {
                if isSpeechActive {
                    AgentSpeechIndicatorBar(onStop: {
                        coordinator.stopSpeaking()
                        speakingMessageID = nil
                    })
                }

                AgentInputBarView(inputText: $inputText)
            }
        }
        .navigationTitle("Agent")
        .onChange(of: speechCoordinator.state) { _, newState in
            if case .idle = newState { speakingMessageID = nil }
        }
        .animation(.easeInOut(duration: 0.2), value: isSpeechActive)
        .onExitCommand {
            if coordinator.voiceState == .recording {
                coordinator.cancelVoiceInput()
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    coordinator.newConversation()
                } label: {
                    Image(systemName: "plus.message")
                }
                .help("New conversation")
                .disabled(coordinator.isGenerating)

                Button {
                    showingHistory.toggle()
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                .help("Conversation history")
                .popover(isPresented: $showingHistory) {
                    conversationHistoryPopover
                }

                Button {
                    settings.agentAutoSpeak.toggle()
                } label: {
                    Image(systemName: settings.agentAutoSpeak ? "speaker.wave.2.fill" : "speaker.slash.fill")
                }
                .help(settings.agentAutoSpeak ? "Auto-speak responses (on)" : "Auto-speak responses (off)")
                
                Button {
                    useMarkdown.toggle()
                } label: {
                    Image(systemName: useMarkdown ? "text.alignleft" : "doc.plaintext")
                }
                .help(useMarkdown ? "Disable Markdown formatting" : "Enable Markdown formatting")
            }
        }
    }

    // MARK: - Conversation History Popover

    private var conversationHistoryPopover: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                if conversationStore.conversations.isEmpty {
                    Text("No past conversations")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding()
                        .frame(maxWidth: .infinity)
                } else {
                    ForEach(conversationStore.conversations) { summary in
                        conversationRow(summary)
                    }
                }
            }
        }
        .frame(maxWidth: 280, maxHeight: 360)
    }

    private func conversationRow(_ summary: AgentConversationSummary) -> some View {
        let isCurrent = conversationStore.currentConversation?.id == summary.id
        return Button {
            coordinator.loadConversation(summary.id)
            showingHistory = false
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(summary.title)
                        .font(.callout)
                        .lineLimit(1)
                        .foregroundStyle(isCurrent ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
                    Text(summary.updatedAt.formatted(.relative(presentation: .named)))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(summary.messageCount)")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(role: .destructive) {
                coordinator.deleteConversation(summary.id)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }
}
