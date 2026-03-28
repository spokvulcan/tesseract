import SwiftUI

struct AgentContentView: View {
    @Environment(AgentCoordinator.self) private var coordinator
    @EnvironmentObject private var conversationStore: AgentConversationStore
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(SchedulingService.self) private var schedulingService
    @State private var inputText = ""
    @State private var showingHistory = false
    @State private var speakingMessageID: UUID?
    @Environment(SettingsManager.self) private var settings
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var isSpeechActive: Bool {
        if case .idle = speechCoordinator.state { return false }
        if case .error = speechCoordinator.state { return false }
        return true
    }

    var body: some View {
        VStack(spacing: 0) {
            if coordinator.isViewingBackgroundSession {
                backgroundSessionBanner
            }

            AgentConversationListView(
                speakingMessageID: $speakingMessageID,
                isSpeechActive: isSpeechActive
            )
            .overlay(alignment: .bottom) {
                AgentInputStatusStrip()
                    .padding(.horizontal, Theme.Spacing.lg)
                    .padding(.bottom, Theme.Spacing.xs)
            }
        }
        .safeAreaInset(edge: .bottom) {
            if !coordinator.isViewingBackgroundSession {
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
        }
        .navigationTitle(coordinator.isViewingBackgroundSession
            ? (coordinator.viewingSessionName ?? "Background Session")
            : "Agent")
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
        .onAppear {
            consumePendingBackgroundSession()
        }
        .onChange(of: schedulingService.pendingBackgroundSessionId) { _, sessionId in
            guard sessionId != nil else { return }
            consumePendingBackgroundSession()
        }
    }

    // MARK: - Background Session

    private func consumePendingBackgroundSession() {
        guard let sessionId = schedulingService.pendingBackgroundSessionId else { return }
        schedulingService.pendingBackgroundSessionId = nil
        Task {
            let opened = await coordinator.openBackgroundSession(id: sessionId)
            if opened {
                schedulingService.markSessionRead(sessionId)
            }
        }
    }

    private var backgroundSessionBanner: some View {
        let task = coordinator.viewingSessionId.flatMap { id in
            schedulingService.tasks.first(where: { $0.sessionId == id })
        }
        
        return VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Image(systemName: "calendar.badge.clock")
                            .foregroundStyle(.secondary)
                        Text(task?.name ?? coordinator.viewingSessionName ?? "Background Session")
                            .font(.headline)
                        
                        if let count = task?.runCount {
                            Text("\(count) run\(count == 1 ? "" : "s")")
                                .font(.caption2.weight(.medium))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.15))
                                .clipShape(Capsule())
                                .foregroundStyle(.secondary)
                        }
                    }
                    if let schedule = task?.humanReadableSchedule {
                        Text(schedule)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Button("Dismiss") {
                    coordinator.dismissBackgroundSession()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(.bar)
            
            Divider()
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
