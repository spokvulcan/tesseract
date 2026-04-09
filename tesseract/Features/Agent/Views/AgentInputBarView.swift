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
    @State private var pendingImages: [ImageAttachment] = []
    @Environment(SettingsManager.self) private var settings

    private static let supportedImageTypes: [UTType] = [.png, .jpeg, .gif, .webP, .tiff]

    /// Whether controls that depend on a loaded model should be disabled.
    /// The vision toggle disables itself during both generation and model loading
    /// (model loading happens during a vision mode switch).
    private var isModelBusy: Bool {
        coordinator.isGenerating || agentEngine.isLoading
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topLeading) {
                if inputText.isEmpty && pendingImages.isEmpty {
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
                        // Silently drop pasted images when vision mode is off —
                        // the container doesn't support them.
                        guard settings.visionModeEnabled else { return }
                        pendingImages.append(contentsOf: attachments)
                    },
                    isEnabled: !(coordinator.voiceState == .recording || coordinator.voiceState == .transcribing),
                    onArrowUp: {
                        guard coordinator.showCommandPopup else { return false }
                        coordinator.commandSelectedIndex = max(0, coordinator.commandSelectedIndex - 1)
                        return true
                    },
                    onArrowDown: {
                        guard coordinator.showCommandPopup else { return false }
                        let count = coordinator.commandFilteredResults.count
                        coordinator.commandSelectedIndex = min(count - 1, coordinator.commandSelectedIndex + 1)
                        return true
                    },
                    onEscape: {
                        guard coordinator.showCommandPopup else { return false }
                        coordinator.dismissCommandPopup()
                        inputText = ""
                        return true
                    }
                )
                .frame(height: min(max(textHeight, 20), 150))
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, pendingImages.isEmpty ? 12 : 4)
            }

            // Image preview strip
            if !pendingImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(pendingImages) { attachment in
                            ImageThumbnailView(attachment: attachment) {
                                pendingImages.removeAll { $0.id == attachment.id }
                            }
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
                    // Image attach button — kept visible in both modes to avoid
                    // layout jumps when the vision toggle flips. Disabled when
                    // vision mode is off since the text-only container drops images.
                    Button { openImagePicker() } label: {
                        Image(systemName: "plus")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 24, height: 24)
                            .background(.quinary, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .help(settings.visionModeEnabled
                          ? "Add image"
                          : "Enable vision mode to attach images")
                    .disabled(!settings.visionModeEnabled || isModelBusy)

                    Button {
                        coordinator.setVisionModeEnabled(!settings.visionModeEnabled)
                    } label: {
                        Image(systemName: "photo")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(settings.visionModeEnabled ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                    }
                    .buttonStyle(.plain)
                    .help(settings.visionModeEnabled
                          ? "Vision mode enabled — click to switch to fast text-only"
                          : "Vision mode disabled (text-only, fast prefill) — click to enable image support")
                    .disabled(isModelBusy)

                    Button {
                        settings.webAccessEnabled.toggle()
                    } label: {
                        Image(systemName: "globe")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(settings.webAccessEnabled ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                    }
                    .buttonStyle(.plain)
                    .help(settings.webAccessEnabled ? "Web search enabled — click to disable" : "Web search disabled — click to enable")

                    Button {
                        if coordinator.showCommandPopup {
                            coordinator.dismissCommandPopup()
                        } else {
                            inputText = "/"
                            coordinator.updateCommandPopup(for: "/")
                        }
                    } label: {
                        Image(systemName: "slash.circle")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(coordinator.showCommandPopup ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
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
                                .foregroundStyle(canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.tertiary))
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
            coordinator.updateCommandPopup(for: newValue)
        }
        .onChange(of: settings.visionModeEnabled) { _, newValue in
            // Clear any queued images when the user disables vision — the
            // LLM container would silently drop them.
            if !newValue {
                pendingImages = []
            }
        }
        .onAppear {
            coordinator.onVoiceTranscription = { text in
                if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    inputText = text
                } else {
                    inputText += " " + text
                }
            }
        }
        .padding(Theme.Spacing.md)
        .onDrop(of: [.image], isTargeted: nil) { providers in
            // Silently ignore image drops when vision mode is off.
            guard settings.visionModeEnabled else { return false }
            handleDrop(providers)
            return true
        }
    }

    // MARK: - Mic Button

    private var micButton: some View {
        let state = coordinator.voiceState

        return micIcon(for: state)
            .font(.title2)
            .frame(width: 28, height: 28)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isHoldingMic else { return }
                        isHoldingMic = true
                        coordinator.startVoiceInput()
                    }
                    .onEnded { _ in
                        isHoldingMic = false
                        if coordinator.voiceState == .recording {
                            coordinator.stopVoiceInputAndSend()
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
                .foregroundStyle(canUseVoice ? AnyShapeStyle(.secondary) : AnyShapeStyle(.quaternary))
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
        if case .downloaded = downloadManager.statuses[WhisperModel.modelID] { return true }
        return false
    }

    private var canUseVoice: Bool {
        !coordinator.isGenerating
            && coordinator.voiceState != .transcribing
            && isWhisperAvailable
    }

    private var voiceButtonHelp: String {
        if !isWhisperAvailable {
            return "Download Whisper model to use voice input"
        }
        if coordinator.isGenerating {
            return "Voice input unavailable during generation"
        }
        switch coordinator.voiceState {
        case .recording: return "Release to transcribe"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return msg
        default: return "Hold to speak"
        }
    }

    private var canSend: Bool {
        (!inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingImages.isEmpty)
            && !coordinator.isGenerating
    }

    // MARK: - Slash Command Helpers

    private func selectCommand(_ command: SlashCommand) {
        inputText = coordinator.autocompleteCommand(command)
    }

    private func handleCommit() {
        // If popup is showing, Enter autocompletes the selected command
        if coordinator.showCommandPopup {
            let filtered = coordinator.commandFilteredResults
            if filtered.indices.contains(coordinator.commandSelectedIndex) {
                selectCommand(filtered[coordinator.commandSelectedIndex])
                return
            }
        }
        // Otherwise send (command parsing happens in coordinator.sendMessage)
        send()
    }

    // MARK: - Actions

    private func send() {
        let text = inputText
        let images = pendingImages
        inputText = ""
        pendingImages = []
        coordinator.sendMessage(text, images: images)
    }

    /// 10 MB max per image — larger files are silently skipped.
    private static let maxImageBytes = 10 * 1024 * 1024

    private func openImagePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = Self.supportedImageTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Select images to attach"
        panel.begin { response in
            guard response == .OK else { return }
            let attachments = panel.urls.compactMap { url -> ImageAttachment? in
                guard let data = try? Data(contentsOf: url),
                      data.count <= Self.maxImageBytes else { return nil }
                let mimeType = url.mimeTypeForImage ?? "image/png"
                return ImageAttachment(data: data, mimeType: mimeType, filename: url.lastPathComponent)
            }
            DispatchQueue.main.async {
                pendingImages.append(contentsOf: attachments)
            }
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) {
        for provider in providers {
            let registeredTypes = provider.registeredTypeIdentifiers
            let mimeType = registeredTypes.lazy
                .compactMap { UTType($0)?.preferredMIMEType }
                .first ?? "image/png"

            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data, data.count <= Self.maxImageBytes else { return }
                let attachment = ImageAttachment(data: data, mimeType: mimeType, filename: "dropped-image")
                DispatchQueue.main.async {
                    pendingImages.append(attachment)
                }
            }
        }
    }
}

// MARK: - Image Thumbnail

private struct ImageThumbnailView: View {
    let attachment: ImageAttachment
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
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

// MARK: - URL Image MIME Type Helper

private extension URL {
    var mimeTypeForImage: String? {
        guard let utType = UTType(filenameExtension: pathExtension) else { return nil }
        return utType.preferredMIMEType
    }
}
