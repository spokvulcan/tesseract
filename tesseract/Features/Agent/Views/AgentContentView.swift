import SwiftUI
import UniformTypeIdentifiers

struct AgentContentView: View {
    @Environment(AgentCoordinator.self) private var coordinator
    @Environment(AppshotController.self) private var appshot
    @EnvironmentObject private var conversationStore: AgentConversationStore
    @Environment(SpeechCoordinator.self) private var speechCoordinator
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
        // The drop-target flag lives on the draft controller so the composer
        // text view's AppKit drag tracking drives the same overlay (#167).
        @Bindable var imageDraft = coordinator.imageDraft

        return VStack(spacing: 0) {
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
                        if coordinator.commandPalette.showCommandPopup,
                            !coordinator.commandPalette.commandFilteredResults.isEmpty
                        {
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
                .animation(
                    .easeOut(duration: 0.15), value: coordinator.commandPalette.showCommandPopup)
            }
            // Min-size shield, load-bearing (macOS 26 framework bug): the
            // scene measures NavigationSplitView's detail minimum by probing
            // at near-zero width, where any wrapping `.fixedSize(vertical:)`
            // text in this inset reports a word-per-line height. When a
            // banner (or popup, or the speech bar) appears, that inflated
            // minimum makes the scene resize the window past the screen and
            // pin its min height there. Reporting a zero minimum here keeps
            // the window's frame the user's alone; real layout is unaffected.
            .frame(minHeight: 0)
        }
        .navigationTitle("Agent")
        .onChange(of: coordinator.editDraftRestore) { _, _ in
            applyEditDraftRestore(allowClobber: true)
        }
        .onChange(of: appshot.composerPrefill) { _, _ in
            applyAppshotPrefill()
        }
        .onAppear {
            applyEditDraftRestore(allowClobber: false)
            applyAppshotPrefill()
        }
        .background(
            QuickLookContainer(
                request: coordinator.imageDraft.quickLookRequest,
                onClose: { coordinator.imageDraft.dismissQuickLook() }
            )
        )
        // Full-window image drop (slice #117): dropping an image anywhere lands it
        // in the composer's pending strip. `isTargeted` only flips for drags whose
        // items conform to `.image`, so non-image drags never dim the window.
        .onDrop(of: [.image], isTargeted: $imageDraft.isDropTargeted) { providers in
            coordinator.imageDraft.handleWindowImageDrop(providers)
        }
        .overlay {
            if imageDraft.isDropTargeted {
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
        .animation(.easeInOut(duration: 0.15), value: imageDraft.isDropTargeted)
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
                    Image(
                        systemName: settings.agentAutoSpeak
                            ? "speaker.wave.2.fill" : "speaker.slash.fill")
                }
                .help(
                    settings.agentAutoSpeak
                        ? "Auto-speak responses (on)" : "Auto-speak responses (off)")

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

    /// Consume the one-shot **Edit & resend** text restore. At edit time
    /// (`onChange`, this view mounted) the composer is REPLACED outright with the
    /// edited message's text — even empty, for an image-only message, so a stale
    /// draft is cleared rather than mixed with the restored images. The `onAppear`
    /// fallback catches a restore set while this view was off-screen (a bare
    /// `onChange` would miss that transition and strand the text), but must not
    /// blank a draft the user has typed since, so it declines an empty restore
    /// over a non-empty composer.
    private func applyEditDraftRestore(allowClobber: Bool) {
        guard let restored = coordinator.editDraftRestore else { return }
        coordinator.editDraftRestore = nil
        if !allowClobber, restored.isEmpty, !inputText.isEmpty { return }
        inputText = restored
    }

    /// Consume the one-shot Appshot composer prefill. Unlike the edit restore,
    /// it never replaces typed text — the appshot label is a convenience and
    /// the user's draft always wins; the prefill is simply dropped then.
    private func applyAppshotPrefill() {
        guard let prefill = appshot.composerPrefill else { return }
        appshot.composerPrefill = nil
        guard inputText.isEmpty else { return }
        inputText = prefill
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
