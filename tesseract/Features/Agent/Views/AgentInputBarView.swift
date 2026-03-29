//
//  AgentInputBarView.swift
//  tesseract
//

import SwiftUI

struct AgentInputBarView: View {
    @Binding var inputText: String
    @Environment(AgentCoordinator.self) private var coordinator
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var isHoldingMic = false
    @State private var textHeight: CGFloat = 20
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topLeading) {
                if inputText.isEmpty {
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
                    onCommit: { send() },
                    isEnabled: !(coordinator.voiceState == .recording || coordinator.voiceState == .transcribing)
                )
                .frame(height: min(max(textHeight, 20), 150))
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, 12)
            }
            
            HStack(spacing: 16) {
                // Formatting and attachment actions (mocked for visual fidelity)
                HStack(spacing: 14) {
                    Button { } label: {
                        Image(systemName: "plus")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 24, height: 24)
                            .background(.quinary, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Add attachment")

                    Button {
                        settings.webAccessEnabled.toggle()
                    } label: {
                        Image(systemName: "globe")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(settings.webAccessEnabled ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                    }
                    .buttonStyle(.plain)
                    .help(settings.webAccessEnabled ? "Web search enabled — click to disable" : "Web search disabled — click to enable")

                    Button { } label: {
                        Image(systemName: "at")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Mention")
                    
                    Button { } label: {
                        Image(systemName: "slash.circle")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.secondary)
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
        .padding(Theme.Spacing.md)
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
        case .recording: return "Release to send"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return msg
        default: return "Hold to speak"
        }
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !coordinator.isGenerating
    }

    // MARK: - Actions

    private func send() {
        let text = inputText
        inputText = ""
        // Model loading is handled by the InferenceArbiter inside sendMessage
        coordinator.sendMessage(text)
    }
}
