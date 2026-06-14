//
//  AgentInputBarView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct AgentInputBarView: View {
    @Binding var inputText: String
    @Environment(AgentCoordinator.self) private var coordinator
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var isHoldingMic = false
    @State private var textHeight: CGFloat = 20
    /// Whether the *selected* agent model can serve images. Probed off the view
    /// body (disk read via `ModelIdentity`) and cached here, refreshed only when
    /// the selection or its download status changes — never per keystroke.
    @State private var selectedModelIsVisionCapable = false
    @Environment(SettingsManager.self) private var settings

    // The pending-image queue and the "switch to a vision model" hint live on the
    // coordinator (`coordinator.pendingImages` / `coordinator.showImageSwitchHint`)
    // so a full-window drop, hosted above the composer, reaches the same queue
    // (slice #117).

    /// The image-input-availability projection (ADR-0013): show image affordances
    /// only when the model is vision-capable *and* the global vision opt-out is on.
    private var imageInputAvailable: Bool {
        ImageInputAvailability.showImageAffordance(
            isVisionCapable: selectedModelIsVisionCapable,
            useVisionWhenAvailable: settings.useVisionWhenAvailable
        )
    }

    /// Whether controls that depend on a loaded model should be disabled —
    /// during both generation and model (re)loading.
    private var isModelBusy: Bool {
        coordinator.isGenerating || agentEngine.isLoading
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if coordinator.showImageSwitchHint {
                imageHintBanner
            }

            ZStack(alignment: .topLeading) {
                if inputText.isEmpty && coordinator.pendingImages.isEmpty {
                    Text("Message…")
                        .font(.system(size: 15))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 16)
                        .allowsHitTesting(false)
                }

                AgentScrollableTextField(
                    text: $inputText,
                    dynamicHeight: $textHeight,
                    onCommit: { handleCommit() },
                    onImagePaste: { attachments in
                        // When the selected model can't see images, surface the
                        // one-tap switch hint instead of silently dropping (#115).
                        guard imageInputAvailable else {
                            coordinator.showImageSwitchHint = true; return
                        }
                        coordinator.attachImages(attachments)
                    },
                    isEnabled:
                        !(coordinator.voiceInput.voiceState == .recording
                        || coordinator.voiceInput.voiceState == .transcribing),
                    onArrowUp: {
                        guard coordinator.commandPalette.showCommandPopup else { return false }
                        coordinator.commandPalette.commandSelectedIndex = max(
                            0, coordinator.commandPalette.commandSelectedIndex - 1)
                        return true
                    },
                    onArrowDown: {
                        guard coordinator.commandPalette.showCommandPopup else { return false }
                        let count = coordinator.commandPalette.commandFilteredResults.count
                        coordinator.commandPalette.commandSelectedIndex = min(
                            count - 1, coordinator.commandPalette.commandSelectedIndex + 1)
                        return true
                    },
                    onEscape: {
                        guard coordinator.commandPalette.showCommandPopup else { return false }
                        coordinator.commandPalette.dismissCommandPopup()
                        inputText = ""
                        return true
                    }
                )
                .frame(height: min(max(textHeight, 20), 150))
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, coordinator.pendingImages.isEmpty ? 12 : 4)
            }

            // Image preview strip
            if !coordinator.pendingImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(coordinator.pendingImages) { attachment in
                            ImageThumbnailView(
                                attachment: attachment,
                                onRemove: {
                                    coordinator.pendingImages.removeAll { $0.id == attachment.id }
                                },
                                onTap: {
                                    coordinator.openQuickLook(
                                        clicked: attachment.id,
                                        includingPending: coordinator.pendingImages)
                                }
                            )
                        }
                    }
                    .padding(.horizontal, 16)
                }
                .frame(height: 64)
                .padding(.bottom, 4)
            }

            HStack(spacing: 16) {
                // Formatting and attachment actions
                HStack(spacing: 14) {
                    // Image attach button — shown only when the selected model can
                    // serve images and vision is enabled (ADR-0013). Hidden, not
                    // disabled: capability changes on model switch, not per
                    // keystroke, so there are no layout jumps to mask. The standalone
                    // vision toggle is gone — vision is on by default for capable
                    // models, governed globally in Settings.
                    if imageInputAvailable {
                        Button {
                            openImagePicker()
                        } label: {
                            Image(systemName: "plus")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.secondary)
                                .frame(width: 24, height: 24)
                                .background(.quinary, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .help("Add image")
                        .disabled(isModelBusy)
                    }

                    Button {
                        settings.webAccessEnabled.toggle()
                    } label: {
                        Image(systemName: "globe")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(
                                settings.webAccessEnabled
                                    ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                    }
                    .buttonStyle(.plain)
                    .help(
                        settings.webAccessEnabled
                            ? "Web search enabled — click to disable"
                            : "Web search disabled — click to enable")

                    Button {
                        if coordinator.commandPalette.showCommandPopup {
                            coordinator.commandPalette.dismissCommandPopup()
                        } else {
                            inputText = "/"
                            coordinator.commandPalette.updateCommandPopup(for: "/")
                        }
                    } label: {
                        Image(systemName: "slash.circle")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(
                                coordinator.commandPalette.showCommandPopup
                                    ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                    }
                    .buttonStyle(.plain)
                    .help("Commands")
                }

                Spacer()

                // Active input controls
                HStack(spacing: 12) {
                    micButton

                    if coordinator.isGenerating {
                        Button {
                            coordinator.cancelGeneration()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.system(size: 20))
                                .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                        .help("Cancel generation")
                    } else {
                        Button {
                            send()
                        } label: {
                            Image(systemName: "paperplane.fill")
                                .font(.system(size: 18))
                                .foregroundStyle(
                                    canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.tertiary))
                        }
                        .buttonStyle(.plain)
                        .disabled(!canSend)
                        .help("Send message")
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
        .glassEffect(in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 4)
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.quaternary, lineWidth: 0.5)
        }
        .onChange(of: inputText) { _, newValue in
            coordinator.commandPalette.updateCommandPopup(for: newValue)
        }
        .onChange(of: imageInputAvailable) { _, available in
            // Mirror availability to the coordinator so the full-window drop
            // (hosted above the composer) can decide attach-vs-hint (#117).
            coordinator.imageInputAvailable = available
            if available {
                // Vision input just became available (model switched / opt-in) —
                // the hint is moot.
                coordinator.showImageSwitchHint = false
            } else {
                // Clear any queued images when image input becomes unavailable
                // (model switched to text-only, or vision opted out) — the LLM
                // container would silently drop them.
                coordinator.pendingImages = []
            }
        }
        .onChange(of: settings.selectedAgentModelID) { _, _ in
            refreshVisionCapability()
        }
        .onChange(of: downloadManager.statuses[settings.selectedAgentModelID]) { _, _ in
            refreshVisionCapability()
        }
        .onAppear {
            refreshVisionCapability()
            coordinator.imageInputAvailable = imageInputAvailable
            coordinator.voiceInput.onVoiceTranscription = { text in
                if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    inputText = text
                } else {
                    inputText += " " + text
                }
            }
        }
        .padding(Theme.Spacing.md)
    }

    // MARK: - Mic Button

    private var micButton: some View {
        let state = coordinator.voiceInput.voiceState

        return micIcon(for: state)
            .font(.title2)
            .frame(width: 28, height: 28)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isHoldingMic else { return }
                        isHoldingMic = true
                        coordinator.voiceInput.start()
                    }
                    .onEnded { _ in
                        isHoldingMic = false
                        if coordinator.voiceInput.voiceState == .recording {
                            coordinator.voiceInput.finishCapture()
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
                .symbolEffect(.pulse, options: .repeating)
        case .transcribing:
            Image(systemName: "waveform")
                .foregroundStyle(.tint)
                .symbolEffect(.variableColor.iterative, options: .repeating)
        case .error:
            Image(systemName: "mic.slash.fill")
                .foregroundStyle(.red)
        }
    }

    // MARK: - Computed State

    private var isWhisperAvailable: Bool {
        if transcriptionEngine.isModelLoaded { return true }
        // Any downloaded speech model counts: selection auto-heals onto a
        // downloaded variant, so voice input is usable the moment one exists.
        return ModelDefinition.all.contains { model in
            guard model.category == .speechToText else { return false }
            if case .downloaded = downloadManager.statuses[model.id] { return true }
            return false
        }
    }

    private var canUseVoice: Bool {
        !coordinator.isGenerating
            && coordinator.voiceInput.voiceState != .transcribing
            && isWhisperAvailable
    }

    private var voiceButtonHelp: String {
        if !isWhisperAvailable {
            return "Download Whisper model to use voice input"
        }
        if coordinator.isGenerating {
            return "Voice input unavailable during generation"
        }
        switch coordinator.voiceInput.voiceState {
        case .recording: return "Release to transcribe"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return msg
        default: return "Hold to speak"
        }
    }

    private var canSend: Bool {
        (!inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !coordinator.pendingImages.isEmpty)
            && !coordinator.isGenerating
    }

    // MARK: - Slash Command Helpers

    private func selectCommand(_ command: SlashCommand) {
        inputText = coordinator.commandPalette.autocompleteCommand(command)
    }

    private func handleCommit() {
        // If popup is showing, Enter autocompletes the selected command
        if coordinator.commandPalette.showCommandPopup {
            let filtered = coordinator.commandPalette.commandFilteredResults
            if filtered.indices.contains(coordinator.commandPalette.commandSelectedIndex) {
                selectCommand(filtered[coordinator.commandPalette.commandSelectedIndex])
                return
            }
        }
        // Otherwise send (command parsing happens in coordinator.sendMessage)
        send()
    }

    // MARK: - Actions

    private func send() {
        let text = inputText
        let images = coordinator.pendingImages
        inputText = ""
        coordinator.pendingImages = []
        coordinator.sendMessage(text, images: images)
    }

    /// Re-probe whether the selected agent model is vision-capable and cache the
    /// result in `selectedModelIsVisionCapable`. Called on selection/status
    /// changes and on appear — never from the view body — so the `ModelIdentity`
    /// disk read stays off the per-keystroke render path.
    private func refreshVisionCapability() {
        let modelID = settings.selectedAgentModelID
        guard case .downloaded = downloadManager.statuses[modelID],
            let directory = downloadManager.modelPath(for: modelID)
        else {
            selectedModelIsVisionCapable = false
            return
        }
        selectedModelIsVisionCapable = ModelVisionCapability.isVisionCapable(directory: directory)
    }

    // MARK: - Image Switch Hint (slice #115)

    @ViewBuilder
    private var imageHintBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "eye.slash")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Text(visionSwitch.message)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            if let title = visionSwitch.actionTitle {
                Button(title) { applyVisionSwitch() }
                    .buttonStyle(.borderless)
                    .font(.system(size: 12, weight: .medium))
            }
            Button {
                coordinator.showImageSwitchHint = false
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 4)
    }

    /// How the user can make image input available from the current state.
    private enum VisionSwitch: Equatable {
        /// The selected model is vision-capable but the global opt-out is off.
        case turnOnSetting
        /// A different downloaded model can see images — offer to switch to it.
        case switchModel(id: String, name: String)
        /// No vision-capable model is downloaded — nothing to switch to.
        case noVisionModel

        var message: String {
            switch self {
            case .turnOnSetting:
                "Vision is turned off. Turn it on to attach images."
            case .switchModel(_, let name):
                "This model can’t see images. Switch to \(name) to attach images."
            case .noVisionModel:
                "This model can’t see images. Download a vision model from Settings → Models."
            }
        }

        var actionTitle: String? {
            switch self {
            case .turnOnSetting: "Turn On"
            case .switchModel: "Switch"
            case .noVisionModel: nil
            }
        }
    }

    private var visionSwitch: VisionSwitch {
        if selectedModelIsVisionCapable && !settings.useVisionWhenAvailable {
            return .turnOnSetting
        }
        if let model = firstDownloadedVisionModel() {
            return .switchModel(id: model.id, name: model.displayName)
        }
        return .noVisionModel
    }

    /// The first downloaded agent model that can serve images, if any.
    private func firstDownloadedVisionModel() -> ModelDefinition? {
        ModelDefinition.all.first { model in
            guard model.category == .agent else { return false }
            guard case .downloaded = downloadManager.statuses[model.id],
                let directory = downloadManager.modelPath(for: model.id)
            else { return false }
            return ModelVisionCapability.isVisionCapable(directory: directory)
        }
    }

    /// Apply the one-tap switch: turn vision on, or switch to a vision model.
    private func applyVisionSwitch() {
        switch visionSwitch {
        case .turnOnSetting:
            settings.useVisionWhenAvailable = true
        case .switchModel(let id, _):
            settings.selectedAgentModelID = id
            if !settings.useVisionWhenAvailable {
                settings.useVisionWhenAvailable = true
            }
        case .noVisionModel:
            break
        }
        coordinator.showImageSwitchHint = false
    }

    private func openImagePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImageIngest.supportedUTTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Select images to attach"
        panel.begin { response in
            guard response == .OK else { return }
            let attachments = panel.urls.compactMap { url -> ImageAttachment? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                let uti =
                    UTType(filenameExtension: url.pathExtension)?.identifier ?? url.pathExtension
                return try? ImageIngest.ingest(
                    data: data, typeIdentifier: uti, filename: url.lastPathComponent
                ).get()
            }
            DispatchQueue.main.async {
                coordinator.attachImages(attachments)
            }
        }
    }
}

// MARK: - Image Thumbnail

private struct ImageThumbnailView: View {
    let attachment: ImageAttachment
    let onRemove: () -> Void
    /// Click the thumbnail (not the ✕) to open it full size in Quick Look (#116).
    var onTap: (() -> Void)? = nil

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
