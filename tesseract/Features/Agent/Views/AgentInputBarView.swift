//
//  AgentInputBarView.swift
//  tesseract
//

import SwiftUI

struct AgentInputBarView: View {
    @Binding var inputText: String
    @EnvironmentObject private var coordinator: AgentCoordinator
    @EnvironmentObject private var agentEngine: AgentEngine
    @EnvironmentObject private var transcriptionEngine: TranscriptionEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var isHoldingMic = false
    @AppStorage("selectedAgentModelID") private var agentModelID: String = "qwen3-4b-instruct-2507"

    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[agentModelID] {
            return true
        }
        return false
    }

    var body: some View {
        HStack(spacing: 8) {
            TextField("Message…", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .onSubmit { send() }
                .disabled(coordinator.voiceState == .recording || coordinator.voiceState == .transcribing)

            micButton

            if coordinator.isGenerating {
                Button {
                    coordinator.cancelGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
                .help("Cancel generation")
            } else {
                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary))
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .help("Send message")
            }
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

        Task {
            await loadModelIfNeeded()
            guard agentEngine.isModelLoaded else { return }
            coordinator.sendMessage(text)
        }
    }

    private func loadModelIfNeeded() async {
        guard isModelDownloaded,
              !agentEngine.isModelLoaded,
              !agentEngine.isLoading,
              let path = downloadManager.modelPath(for: agentModelID)
        else { return }

        do {
            try await agentEngine.loadModel(from: path)
        } catch {
            coordinator.error = "Failed to load model: \(error.localizedDescription)"
        }
    }
}
