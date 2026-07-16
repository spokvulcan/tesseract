//
//  AgentContentView.swift
//  tesseract
//
//  The agent chat page: the flat document transcript with the glass composer
//  floating in the bottom safe-area inset. The chat's three custom glass
//  surfaces live here: the composer and the slash-command popup share one
//  GlassEffectContainer; the Skill Cluster (ADR-0030) floats above the
//  composer's trailing corner in its own container so it never fuses with
//  the composer's glass. Everything above them is content layer and stays
//  glass-free (HIG).
//

import SwiftUI
import UniformTypeIdentifiers

struct AgentContentView: View {
    @Environment(ChatSession.self) private var session
    @Environment(ComposerDraftController.self) private var composerDraft
    @Environment(SlashCommandPaletteController.self) private var commandPalette
    @Environment(SkillPillController.self) private var skillPills
    @Environment(AgentVoiceInputController.self) private var voiceInput
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(SettingsManager.self) private var settings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var conversationStore: AgentConversationStore

    @State private var showingHistory = false
    @State private var speakingMessageID: UUID?
    /// The Skill Cluster's interaction state machine (ADR-0030). View-local:
    /// it has no dependencies and no life outside this page.
    @State private var skillCluster = SkillClusterController()

    private var isSpeechActive: Bool {
        if case .idle = speechCoordinator.state { return false }
        if case .error = speechCoordinator.state { return false }
        return true
    }

    /// The full-inset tap catcher behind a transient surface (the slash popup
    /// or a pinned Skill Cluster): any click outside the surface dismisses it.
    private func clickAwayCatcher(_ action: @escaping () -> Void) -> some View {
        Color.clear
            .contentShape(Rectangle())
            .onTapGesture(perform: action)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    var body: some View {
        @Bindable var composerDraft = composerDraft

        return ChatTranscriptView(
            speakingMessageID: $speakingMessageID,
            isSpeechActive: isSpeechActive
        )
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 0) {
                ZStack(alignment: .bottom) {
                    if commandPalette.showCommandPopup {
                        clickAwayCatcher { commandPalette.dismissCommandPopup() }
                    } else if skillCluster.phase == .pinned {
                        clickAwayCatcher { skillCluster.clickedAway() }
                    }

                    // The composer's glass container: the popup and the
                    // composer morph within a shared sampling context. (The
                    // Skill Cluster brings its own container — ADR-0030.)
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
                    // The Skill Cluster floats above the composer's trailing
                    // corner without reserving inset space — the fan overlays
                    // the transcript, so opening it never shifts layout. The
                    // zero-height frame pins an anchor line at the container's
                    // top edge and the cluster hangs entirely above it (an
                    // alignmentGuide override here silently failed through the
                    // conditional wrapper — don't reintroduce one). Faded out
                    // while the slash popup owns this area (the controller is
                    // suppressed then).
                    .overlay(alignment: .topTrailing) {
                        if skillPills.isClusterVisible {
                            SkillClusterView()
                                .padding(.bottom, Theme.Spacing.xs)
                                .frame(height: 0, alignment: .bottom)
                                .padding(.trailing, Theme.Spacing.md)
                                .opacity(commandPalette.showCommandPopup ? 0 : 1)
                                .allowsHitTesting(!commandPalette.showCommandPopup)
                        }
                    }
                }
                // `initial: true` seeds suppression on (re)appear — this view
                // is recreated by sidebar navigation, and a run can be
                // generating when the user navigates back.
                .onChange(
                    of: session.isGenerating || commandPalette.showCommandPopup, initial: true
                ) {
                    _, suppressed in
                    skillCluster.isSuppressed = suppressed
                }
                // Draft auto-open: text or an image landing in the composer
                // opens the cluster (pinned) to suggest the skills for it;
                // clearing the draft retires it. Order matters — suppression
                // must be seeded before the draft edge so a mid-generation
                // draft arms instead of opening.
                .onChange(
                    of: !composerDraft.text.isEmpty || !composerDraft.pendingImages.isEmpty,
                    initial: true
                ) { _, hasContent in
                    skillCluster.draftContentChanged(hasContent: hasContent)
                }
                .animation(
                    reduceMotion ? nil : .easeOut(duration: 0.15),
                    value: commandPalette.showCommandPopup
                )
                .environment(skillCluster)
            }
            .frame(maxWidth: ChatLayout.columnMaxWidth + 2 * Theme.Spacing.md)
            // Min-size shield, load-bearing (macOS 26 framework bug): the
            // scene measures NavigationSplitView's detail minimum by probing
            // at near-zero width, where any wrapping `.fixedSize(vertical:)`
            // text in this inset reports a word-per-line height. When a
            // banner (or the slash popup) appears, that inflated
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
            } else {
                skillCluster.escapePressed()
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
                    settings.agentUseMarkdown.toggle()
                } label: {
                    Image(
                        systemName: settings.agentUseMarkdown ? "text.alignleft" : "doc.plaintext")
                }
                .help(
                    settings.agentUseMarkdown
                        ? "Disable Markdown formatting" : "Enable Markdown formatting")
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
                    HStack(spacing: 6) {
                        Text(summary.title)
                            .font(.callout)
                            .lineLimit(1)
                            .foregroundStyle(
                                isCurrent ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
                        // The origin badge (#327 §3): the Companion's own
                        // turns are findable at a glance; typed chats stay
                        // unbadged.
                        if let badge = ConversationOriginBadge.label(for: summary.turnOrigin) {
                            Text(badge)
                                .font(.caption2)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(.tint.opacity(0.14), in: Capsule())
                                .foregroundStyle(.tint)
                        }
                    }
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

// MARK: - Origin badges (#327 §3)

/// The conversation-list origin vocabulary (#327 §2): `interactive` (and
/// legacy untagged) rows stay clean; the Companion's turn classes get a small
/// badge. Exhaustive over `TurnOrigin` so the vocabulary can't drift from
/// what the loop actually emits.
enum ConversationOriginBadge {
    static func label(for origin: TurnOrigin) -> String? {
        switch origin {
        case .interactive: nil
        case .beat: "beat"
        case .wake: "wake"
        case .ambient: "ambient"
        case .catchup: "catch-up"
        case .sleep: "sleep"
        }
    }
}
