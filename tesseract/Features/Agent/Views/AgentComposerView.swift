//
//  AgentComposerView.swift
//  tesseract
//
//  The composer: one of the chat's three custom glass surfaces (with the
//  slash-command popup, which shares its GlassEffectContainer, and the Skill
//  Cluster, ADR-0030, which floats above in its own). Hosts the text field,
//  the pending-image strip, the model button, and the single in-composer
//  notice slot every hint/error feeds — there is no separate status strip or
//  floating error banner.
//

import SwiftUI
import UniformTypeIdentifiers

/// Shared metrics for the composer action-row icons: one glyph size, one hit
/// frame, one spacing, so the six controls read as a single family. Two
/// deliberate exceptions keep their own glyph treatment inside the shared
/// frame: send/stop (the row's one primary action) and + (highlighted with a
/// circle background).
private let actionIconFont: Font = .system(size: 15, weight: .medium)
private let actionIconFrame: CGFloat = 26
private let actionIconSpacing: CGFloat = 10

struct AgentComposerView: View {
    @Environment(ChatSession.self) private var session
    @Environment(ComposerDraftController.self) private var composerDraft
    @Environment(SlashCommandPaletteController.self) private var commandPalette
    @Environment(SkillClusterController.self) private var skillCluster
    @Environment(AgentVoiceInputController.self) private var voiceInput
    @Environment(VisionAvailabilityController.self) private var visionAvailability
    @Environment(AppshotController.self) private var appshot
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @Environment(SettingsManager.self) private var settings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var permissions: PermissionsManager
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var isHoldingMic = false
    @State private var textHeight: CGFloat = 20

    /// Whether controls that depend on a loaded model should be disabled —
    /// during both generation and model (re)loading.
    private var isModelBusy: Bool {
        session.isGenerating || agentEngine.isLoading
    }

    var body: some View {
        @Bindable var draft = composerDraft

        VStack(alignment: .leading, spacing: 0) {
            if let notice = activeNotice {
                noticeBanner(notice)
            }

            ZStack(alignment: .topLeading) {
                if draft.text.isEmpty && composerDraft.pendingImages.isEmpty {
                    Text("Message…")
                        .font(.system(size: chatBodyFontSize))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 14)
                        .allowsHitTesting(false)
                }

                AgentScrollableTextField(
                    text: $draft.text,
                    dynamicHeight: $textHeight,
                    onCommit: { handleCommit() },
                    // Every Image Gesture (paste or composer drag) resolves on
                    // the draft controller: availability hint, cap, and
                    // rejection feedback all live there (issue #167).
                    onImageGesture: { payload in
                        composerDraft.handleGesture(payload)
                    },
                    onImageDragTargeted: { targeted in
                        composerDraft.isDropTargeted = targeted
                    },
                    isEnabled:
                        !(voiceInput.voiceState == .recording
                        || voiceInput.voiceState == .transcribing),
                    onArrowUp: {
                        guard commandPalette.showCommandPopup else { return false }
                        commandPalette.commandSelectedIndex = max(
                            0, commandPalette.commandSelectedIndex - 1)
                        return true
                    },
                    onArrowDown: {
                        guard commandPalette.showCommandPopup else { return false }
                        let count = commandPalette.commandFilteredResults.count
                        commandPalette.commandSelectedIndex = min(
                            count - 1, commandPalette.commandSelectedIndex + 1)
                        return true
                    },
                    onEscape: {
                        guard commandPalette.showCommandPopup else {
                            // No popup — Esc closes an open Skill Cluster.
                            return skillCluster.escapePressed()
                        }
                        commandPalette.dismissCommandPopup()
                        composerDraft.text = ""
                        return true
                    }
                )
                // Taller-than-one-line minimum — a one-line composer feels cramped.
                .frame(height: min(max(textHeight, 32), 150))
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, composerDraft.pendingImages.isEmpty ? 8 : 4)
            }

            // Image preview strip
            if !composerDraft.pendingImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(composerDraft.pendingImages) { attachment in
                            ImageThumbnailView(
                                attachment: attachment,
                                onRemove: {
                                    composerDraft.pendingImages.removeAll {
                                        $0.id == attachment.id
                                    }
                                },
                                onTap: {
                                    composerDraft.openQuickLook(
                                        clicked: attachment.id,
                                        includingPending: composerDraft.pendingImages)
                                }
                            )
                        }
                    }
                    .padding(.horizontal, 16)
                }
                .frame(height: 64)
                .padding(.bottom, 4)
            }

            HStack(spacing: actionIconSpacing) {
                if visionAvailability.imageInputAvailable {
                    Button {
                        openImagePicker()
                    } label: {
                        Image(systemName: "plus")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 22, height: 22)
                            .background(.quinary, in: Circle())
                            .frame(width: actionIconFrame, height: actionIconFrame)
                    }
                    .buttonStyle(.plain)
                    .help("Add image")
                    .disabled(isModelBusy)
                }

                Button {
                    settings.webAccessEnabled.toggle()
                } label: {
                    Image(systemName: "globe")
                        .font(actionIconFont)
                        .foregroundStyle(
                            settings.webAccessEnabled
                                ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary)
                        )
                        .frame(width: actionIconFrame, height: actionIconFrame)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(
                    settings.webAccessEnabled
                        ? "Web access enabled — click to disable"
                        : "Web access disabled — click to enable")

                Button {
                    if commandPalette.showCommandPopup {
                        commandPalette.dismissCommandPopup()
                    } else {
                        composerDraft.text = "/"
                        commandPalette.updateCommandPopup(for: "/")
                    }
                } label: {
                    Image(systemName: "slash.circle")
                        .font(actionIconFont)
                        .foregroundStyle(
                            commandPalette.showCommandPopup
                                ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary)
                        )
                        .frame(width: actionIconFrame, height: actionIconFrame)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Commands")

                Spacer()

                ModelButtonView()

                micButton

                if session.isGenerating {
                    Button {
                        session.cancelGeneration()
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 22))
                            .foregroundStyle(.red)
                            .frame(width: actionIconFrame, height: actionIconFrame)
                    }
                    .buttonStyle(.plain)
                    .help("Stop generating")
                } else {
                    Button {
                        send()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 22))
                            .foregroundStyle(
                                canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.tertiary)
                            )
                            .frame(width: actionIconFrame, height: actionIconFrame)
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .help("Send message")
                }
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
        // `.interactive()` matches the Skill Cluster's material — the two
        // surfaces sit 20pt apart and must not read as different colors.
        .glassEffect(
            .regular.interactive(), in: RoundedRectangle(cornerRadius: 16, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.quaternary, lineWidth: 0.5)
        }
        .onChange(of: draft.text) { _, newValue in
            commandPalette.updateCommandPopup(for: newValue)
        }
        // The Vision Availability leaf owns the verdict and its effects on the
        // draft; the view keeps only the refresh triggers for its inputs.
        .onChange(of: settings.selectedAgentModelID) { _, _ in
            visionAvailability.refresh()
        }
        .onChange(of: downloadManager.status(for: settings.selectedAgentModelID)) { _, _ in
            visionAvailability.refresh()
        }
        .onChange(of: settings.useVisionWhenAvailable) { _, _ in
            visionAvailability.refresh()
        }
        .onAppear {
            visionAvailability.refresh()
            voiceInput.onVoiceTranscription = { [weak composerDraft] text in
                guard let composerDraft else { return }
                if composerDraft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    composerDraft.text = text
                } else {
                    composerDraft.text += " " + text
                }
            }
        }
    }

    // MARK: - Notice slot (the one banner)

    /// Every notice the composer can carry, in priority order: generation
    /// errors, voice errors, the vision-switch hint, the Appshot permission
    /// explainer, attachment feedback, then the selected model's download
    /// state. One slot — never a stack of banners.
    private enum ComposerNotice: Equatable {
        case sessionError(String)
        case voiceError(String)
        case imageHint
        case appshotPermission
        case attachment(String)
        case modelDownloading(displayName: String, progress: Double)
        case modelMissing
    }

    private var activeNotice: ComposerNotice? {
        if let error = session.error { return .sessionError(error) }
        if case .error(let message) = voiceInput.voiceState { return .voiceError(message) }
        if composerDraft.showImageSwitchHint { return .imageHint }
        if appshot.showPermissionExplainer { return .appshotPermission }
        if let notice = composerDraft.attachmentNotice { return .attachment(notice) }
        if !agentEngine.isModelLoaded && !agentEngine.isLoading {
            switch downloadManager.status(for: settings.selectedAgentModelID) {
            case .downloading(let progress):
                return .modelDownloading(
                    displayName: selectedModelDisplayName, progress: progress)
            case .notDownloaded, .error:
                return .modelMissing
            case .downloaded, .verifying:
                break  // On disk; auto-load owns the gap and the slot stays quiet.
            }
        }
        return nil
    }

    @ViewBuilder
    private func noticeBanner(_ notice: ComposerNotice) -> some View {
        switch notice {
        case .sessionError(let message):
            composerBanner(
                icon: "exclamationmark.triangle", tint: .red, message: message
            ) { session.error = nil }

        case .voiceError(let message):
            composerBanner(icon: "mic.slash", tint: .orange, message: message) {
                voiceInput.cancel()
            }

        case .imageHint:
            composerBanner(
                icon: "eye.slash", message: visionAvailability.remedy.message,
                actionTitle: visionAvailability.remedy.actionTitle,
                action: { visionAvailability.applyRemedy() }
            ) {
                composerDraft.showImageSwitchHint = false
            }

        case .appshotPermission:
            // The lazy Screen Recording explainer (PRD #170): shown when an
            // Appshot failed on the missing permission. The grant lives in
            // System Settings and macOS applies it only after a relaunch.
            composerBanner(
                icon: "rectangle.dashed.badge.record",
                message:
                    "Appshots need Screen Recording permission. After granting it in "
                    + "System Settings, relaunch Tesseract to apply.",
                actionTitle: "Open System Settings",
                action: permissions.requestScreenRecordingPermission,
                onDismiss: { appshot.showPermissionExplainer = false }
            )

        case .attachment(let message):
            // Transient Image Gesture feedback (issue #167): what a paste,
            // drop, or pick could not attach.
            composerBanner(icon: "exclamationmark.circle", message: message) {
                composerDraft.attachmentNotice = nil
            }

        case .modelDownloading(let displayName, let progress):
            HStack(spacing: 8) {
                ProgressView(value: progress)
                    .controlSize(.small)
                    .frame(width: 80)
                Text("\(displayName) is downloading — \(Int(progress * 100))%")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
            }
            .padding(.horizontal, 16)
            .padding(.top, 10)
            .padding(.bottom, 2)

        case .modelMissing:
            composerBanner(
                icon: "arrow.down.circle",
                message: "No agent model on this Mac yet. Download one to start chatting.",
                actionTitle: "Open Models",
                action: { (NSApp.delegate as? AppDelegate)?.navigateToModels() },
                onDismiss: nil
            )
        }
    }

    /// Shared chrome for the notice slot: icon, message, optional action,
    /// optional dismiss.
    private func composerBanner(
        icon: String, tint: Color? = nil, message: String,
        actionTitle: String? = nil, action: @escaping () -> Void = {},
        onDismiss: (() -> Void)?
    ) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundStyle(tint.map(AnyShapeStyle.init) ?? AnyShapeStyle(.secondary))
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Spacer(minLength: 8)
            if let actionTitle {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderless)
                    .font(.system(size: 12, weight: .medium))
            }
            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Dismiss")
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 2)
    }

    // MARK: - Mic Button

    private var micButton: some View {
        let state = voiceInput.voiceState

        return micIcon(for: state)
            .font(actionIconFont)
            .frame(width: actionIconFrame, height: actionIconFrame)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isHoldingMic else { return }
                        isHoldingMic = true
                        voiceInput.start()
                    }
                    .onEnded { _ in
                        isHoldingMic = false
                        if voiceInput.voiceState == .recording {
                            voiceInput.finishCapture()
                        }
                    }
            )
            .disabled(!canUseVoice)
            .help(voiceButtonHelp)
    }

    @ViewBuilder
    private func micIcon(for state: AgentVoiceState) -> some View {
        switch state {
        case .idle:
            Image(systemName: "mic.fill")
                .foregroundStyle(
                    canUseVoice ? AnyShapeStyle(.secondary) : AnyShapeStyle(.quaternary))
        case .recording:
            Image(systemName: "stop.fill")
                .foregroundStyle(.red)
                .symbolEffect(.pulse, options: .repeating, isActive: !reduceMotion)
        case .transcribing:
            Image(systemName: "waveform")
                .foregroundStyle(.tint)
                .symbolEffect(
                    .variableColor.iterative, options: .repeating, isActive: !reduceMotion)
        case .error:
            Image(systemName: "mic.slash.fill")
                .foregroundStyle(.red)
        }
    }

    // MARK: - Computed State

    private var selectedModelDisplayName: String {
        ModelDefinition.all.first { $0.id == settings.selectedAgentModelID }?.displayName
            ?? settings.selectedAgentModelID
    }

    private var isWhisperAvailable: Bool {
        if transcriptionEngine.isModelLoaded { return true }
        // Any downloaded speech model counts: selection auto-heals onto a
        // downloaded variant, so voice input is usable the moment one exists.
        return !downloadManager.downloadedModels(in: .speechToText).isEmpty
    }

    private var canUseVoice: Bool {
        !session.isGenerating
            && voiceInput.voiceState != .transcribing
            && isWhisperAvailable
    }

    private var voiceButtonHelp: String {
        if !isWhisperAvailable {
            return "Download Whisper model to use voice input"
        }
        if session.isGenerating {
            return "Voice input unavailable during generation"
        }
        switch voiceInput.voiceState {
        case .recording: return "Release to transcribe"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return msg
        default: return "Hold to speak"
        }
    }

    private var canSend: Bool {
        (!composerDraft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !composerDraft.pendingImages.isEmpty)
            && !session.isGenerating
    }

    // MARK: - Slash Command Helpers

    private func handleCommit() {
        // If popup is showing, Enter autocompletes the selected command
        if commandPalette.showCommandPopup {
            let filtered = commandPalette.commandFilteredResults
            if filtered.indices.contains(commandPalette.commandSelectedIndex) {
                composerDraft.text = commandPalette.autocompleteCommand(
                    filtered[commandPalette.commandSelectedIndex])
                return
            }
        }
        // Otherwise send (command parsing happens in the Chat Session)
        send()
    }

    // MARK: - Actions

    private func send() {
        let text = composerDraft.text
        let images = composerDraft.drainImages()
        composerDraft.text = ""
        session.sendMessage(text, images: images)
    }

    private func openImagePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImageIngest.supportedUTTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Select images to attach"
        panel.begin { [weak composerDraft] response in
            guard response == .OK else { return }
            // Same funnel as paste/drop, so cap trims and unreadable files get
            // the same composer notice (issue #167).
            let payload = PasteboardImageReader.ingest(fileURLs: panel.urls)
            DispatchQueue.main.async {
                composerDraft?.handleGesture(payload)
            }
        }
    }
}

// MARK: - Model Button

/// The model selector: a quiet icon-only menu button in the composer's right
/// cluster, ChatGPT-style — model selection is a rare action, so it gets no
/// text at rest. A spinner replaces the icon while a model loads; the tooltip
/// carries the selected model's name and honest load state. Switching models
/// writes `selectedAgentModelID`; the engine's auto-load reacts to the setting.
private struct ModelButtonView: View {
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(SettingsManager.self) private var settings
    @Environment(ChatSession.self) private var session
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    var body: some View {
        Menu {
            let models = downloadManager.downloadedModels(in: .agent)
            ForEach(models) { model in
                Button {
                    settings.selectedAgentModelID = model.id
                } label: {
                    if model.id == settings.selectedAgentModelID {
                        Label(model.displayName, systemImage: "checkmark")
                    } else {
                        Text(model.displayName)
                    }
                }
            }
            if models.isEmpty {
                Button("Download Models…") {
                    (NSApp.delegate as? AppDelegate)?.navigateToModels()
                }
            }
        } label: {
            Image(systemName: "brain")
                .font(actionIconFont)
                .foregroundStyle(
                    session.isGenerating
                        ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.secondary)
                )
                // While loading, the glyph yields its footprint to the spinner
                // overlaid below — kept out of the label because a ProgressView
                // inside a Menu label does not render on macOS.
                .opacity(agentEngine.isLoading ? 0 : 1)
                .frame(width: actionIconFrame, height: actionIconFrame)
                .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(session.isGenerating)
        .help(loadStateHelp)
        .overlay {
            if agentEngine.isLoading {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.7)
                    .allowsHitTesting(false)
            }
        }
    }

    private var selectedDisplayName: String {
        ModelDefinition.all.first { $0.id == settings.selectedAgentModelID }?.displayName
            ?? settings.selectedAgentModelID
    }

    private var loadStateHelp: String {
        if agentEngine.isLoading {
            return agentEngine.loadingStatus.isEmpty
                ? "Loading model…" : agentEngine.loadingStatus
        }
        return agentEngine.isModelLoaded
            ? "\(selectedDisplayName) is loaded — click to switch models"
            : "\(selectedDisplayName) loads on the first message — click to switch models"
    }
}

// MARK: - Image Thumbnail

private struct ImageThumbnailView: View {
    let attachment: ImageAttachment
    let onRemove: () -> Void
    /// Click the thumbnail (not the ✕) to open it full size in Quick Look (#116).
    var onTap: (() -> Void)?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if let nsImage = NSImage(data: attachment.data) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 56, height: 56)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.quaternary)
                        .frame(width: 56, height: 56)
                        .overlay {
                            Image(systemName: "photo")
                                .foregroundStyle(.secondary)
                        }
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .onTapGesture { onTap?() }
            .help("Click to view full size")

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(.white, .black.opacity(0.6))
            }
            .buttonStyle(.plain)
            .offset(x: 6, y: -6)
        }
    }
}
