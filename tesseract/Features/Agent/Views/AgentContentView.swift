import SwiftUI
import UniformTypeIdentifiers

struct AgentContentView: View {
    @Environment(AgentCoordinator.self) private var coordinator
    @EnvironmentObject private var conversationStore: AgentConversationStore
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @State private var inputText = ""
    @State private var showingHistory = false
    @State private var speakingMessageID: UUID?
    /// True while an image-bearing drag hovers the window (slice #117).
    @State private var isDropTargeted = false
    @Environment(SettingsManager.self) private var settings
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var isSpeechActive: Bool {
        if case .idle = speechCoordinator.state { return false }
        if case .error = speechCoordinator.state { return false }
        return true
    }

    var body: some View {
        VStack(spacing: 0) {
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
            VStack(spacing: 0) {
                if isSpeechActive {
                    AgentSpeechIndicatorBar(onStop: {
                        coordinator.stopSpeaking()
                        speakingMessageID = nil
                    })
                }

                ZStack(alignment: .bottom) {
                    if coordinator.commandPalette.showCommandPopup {
                        Color.clear
                            .contentShape(Rectangle())
                            .onTapGesture {
                                coordinator.commandPalette.dismissCommandPopup()
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    VStack(spacing: 0) {
                        if coordinator.commandPalette.showCommandPopup, !coordinator.commandPalette.commandFilteredResults.isEmpty {
                            SlashCommandPopupView(
                                commands: coordinator.commandPalette.commandFilteredResults,
                                selectedIndex: coordinator.commandPalette.commandSelectedIndex,
                                onSelect: { command in
                                    selectCommandFromPopup(command)
                                }
                            )
                            .padding(.horizontal, Theme.Spacing.md + 16)
                            .padding(.bottom, 4)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                        }

                        AgentInputBarView(inputText: $inputText)
                    }
                }
                .animation(.easeOut(duration: 0.15), value: coordinator.commandPalette.showCommandPopup)
            }
        }
        .navigationTitle("Agent")
        .background(
            QuickLookContainer(
                request: coordinator.quickLookRequest,
                onClose: { coordinator.dismissQuickLook() }
            )
        )
        // Full-window image drop (slice #117): dropping an image anywhere lands it
        // in the composer's pending strip. `isTargeted` only flips for drags whose
        // items conform to `.image`, so non-image drags never dim the window.
        .onDrop(of: [.image], isTargeted: $isDropTargeted) { providers in
            coordinator.handleWindowImageDrop(providers)
        }
        .overlay {
            if isDropTargeted {
                ZStack {
                    Color.black.opacity(0.4)
                    VStack(spacing: 12) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 44, weight: .light))
                        Text("Drop image to attach")
                            .font(.title2.weight(.medium))
                    }
                    .foregroundStyle(.white)
                    .padding(32)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: isDropTargeted)
        .onChange(of: speechCoordinator.state) { _, newState in
            if case .idle = newState { speakingMessageID = nil }
        }
        .animation(.easeInOut(duration: 0.2), value: isSpeechActive)
        .onExitCommand {
            if coordinator.voiceInput.voiceState == .recording {
                coordinator.voiceInput.cancel()
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

    // MARK: - Slash Command Popup

    private func selectCommandFromPopup(_ command: SlashCommand) {
        inputText = coordinator.commandPalette.autocompleteCommand(command)
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
