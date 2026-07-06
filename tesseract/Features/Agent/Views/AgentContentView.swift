//
//  AgentContentView.swift
//  tesseract
//
//  The agent chat page: the flat document transcript with the glass composer
//  floating in the bottom safe-area inset. The chat's two custom glass
//  surfaces — the composer and the slash-command popup — share the one
//  GlassEffectContainer here; everything above them is content layer and
//  stays glass-free (HIG).
//

import SwiftUI
import UniformTypeIdentifiers

struct AgentContentView: View {
    @Environment(ChatSession.self) private var session
    @Environment(ComposerDraftController.self) private var composerDraft
    @Environment(SlashCommandPaletteController.self) private var commandPalette
    @Environment(AgentVoiceInputController.self) private var voiceInput
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var conversationStore: AgentConversationStore

    @State private var showingHistory = false
    @State private var speakingMessageID: UUID?
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var isSpeechActive: Bool {
        if case .idle = speechCoordinator.state { return false }
        if case .error = speechCoordinator.state { return false }
        return true
    }

    var body: some View {
        @Bindable var composerDraft = composerDraft

        return ChatTranscriptView(
            speakingMessageID: $speakingMessageID,
            isSpeechActive: isSpeechActive
        )
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 0) {
                if isSpeechActive {
                    AgentSpeechIndicatorBar(onStop: {
                        session.stopSpeaking()
                        speakingMessageID = nil
                    })
                }

                ZStack(alignment: .bottom) {
                    if commandPalette.showCommandPopup {
                        Color.clear
                            .contentShape(Rectangle())
                            .onTapGesture {
                                commandPalette.dismissCommandPopup()
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    // The one glass container: the popup and the composer
                    // morph within a shared sampling context.
                    GlassEffectContainer {
                        VStack(spacing: 0) {
                            if commandPalette.showCommandPopup,
                                !commandPalette.commandFilteredResults.isEmpty
                            {
                                SlashCommandPopupView(
                                    commands: commandPalette.commandFilteredResults,
                                    selectedIndex: commandPalette.commandSelectedIndex,
                                    onSelect: { command in
                                        composerDraft.text =
                                            commandPalette.autocompleteCommand(command)
                                    }
                                )
                                .padding(.horizontal, Theme.Spacing.md + 16)
                                .padding(.bottom, 4)
                                .transition(.move(edge: .bottom).combined(with: .opacity))
                            }

                            AgentComposerView()
                                .padding(Theme.Spacing.md)
                        }
                    }
                }
                .animation(.easeOut(duration: 0.15), value: commandPalette.showCommandPopup)
            }
            .frame(maxWidth: ChatLayout.columnMaxWidth + 2 * Theme.Spacing.md)
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
        .background(
            QuickLookContainer(
                request: composerDraft.quickLookRequest,
                onClose: { composerDraft.dismissQuickLook() }
            )
        )
        // Full-window image drop (slice #117): dropping an image anywhere lands
        // it in the composer's pending strip. `isTargeted` only flips for drags
        // whose items conform to `.image`, so non-image drags never dim the
        // window.
        .onDrop(of: [.image], isTargeted: $composerDraft.isDropTargeted) { providers in
            composerDraft.handleWindowImageDrop(providers)
        }
        .overlay {
            if composerDraft.isDropTargeted {
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
        .animation(.easeInOut(duration: 0.15), value: composerDraft.isDropTargeted)
        .onChange(of: speechCoordinator.state) { _, newState in
            if case .idle = newState { speakingMessageID = nil }
        }
        .animation(.easeInOut(duration: 0.2), value: isSpeechActive)
        .onExitCommand {
            if voiceInput.voiceState == .recording {
                voiceInput.cancel()
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    session.newConversation()
                } label: {
                    Image(systemName: "plus.message")
                }
                .help("New conversation")
                .disabled(session.isGenerating)

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
            session.loadConversation(summary.id)
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
                session.deleteConversation(summary.id)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }
}

// MARK: - Speech Indicator

/// Slim "Speaking…" strip above the composer while TTS plays, with a stop
/// control. Content-layer chrome — system materials only.
struct AgentSpeechIndicatorBar: View {
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "speaker.wave.2.fill")
                .foregroundStyle(.tint)
                .symbolEffect(.variableColor.iterative, options: .repeating)
            Text("Speaking\u{2026}")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                onStop()
            } label: {
                Image(systemName: "stop.circle.fill")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.tint.opacity(0.08))
    }
}
