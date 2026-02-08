//
//  SpeechContentView.swift
//  tesseract
//

import SwiftUI

struct SpeechContentView: View {
    @ObservedObject var speechCoordinator: SpeechCoordinator
    @ObservedObject var speechEngine: SpeechEngine
    @ObservedObject private var settings = SettingsManager.shared

    @State private var inputText: String = ""

    private var isActiveState: Bool {
        switch speechCoordinator.state {
        case .playing, .streaming, .streamingLongForm, .paused:
            true
        default:
            false
        }
    }

    var body: some View {
        HSplitView {
            // Main content area
            mainContent
                .frame(minWidth: 400)

            // Parameters sidebar
            TTSParametersSidebar()
        }
        .navigationTitle("Speech")
    }

    // MARK: - Main Content

    private var mainContent: some View {
        VStack(spacing: 16) {
            // Status indicator
            SpeechStatusView(
                state: speechCoordinator.state,
                isModelLoaded: speechEngine.isModelLoaded,
                isLoading: speechEngine.isLoading,
                loadingStatus: speechEngine.loadingStatus
            )

            // Voice design
            VoiceDesignView(voiceDescription: $settings.ttsVoiceDescription)

            // Language picker
            HStack {
                Text("Language")
                    .font(.headline)
                Spacer()
                Picker("", selection: $settings.ttsLanguage) {
                    ForEach(TTSLanguage.allCases) { lang in
                        Text("\(lang.flag) \(lang.displayName)").tag(lang.rawValue)
                    }
                }
                .frame(width: 180)
            }

            // Text input area
            VStack(alignment: .leading, spacing: 8) {
                Text("Text")
                    .font(.headline)

                ZStack(alignment: .topLeading) {
                    TextEditor(text: $inputText)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 80, maxHeight: .infinity)
                        .padding(8)
                        .background(.fill.quaternary)
                        .clipShape(RoundedRectangle(cornerRadius: 8))

                    if inputText.isEmpty {
                        Text("Enter text to speak, or use \(settings.ttsHotkey.displayString) to speak selected text from any app")
                            .font(.body)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 13)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                }
            }
            .layoutPriority(1)

            // Long-form progress
            if case .streamingLongForm(let segment, let total) = speechCoordinator.state {
                VStack(spacing: 4) {
                    ProgressView(value: Double(segment), total: Double(total))
                    Text("Segment \(segment) of \(total)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if case .paused(let segment, let total) = speechCoordinator.state {
                VStack(spacing: 4) {
                    ProgressView(value: Double(segment), total: Double(total))
                    Text("Paused at segment \(segment) of \(total)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Controls
            HStack(spacing: 12) {
                Button {
                    if isActiveState {
                        speechCoordinator.stop()
                    } else {
                        speechCoordinator.speakText(inputText)
                    }
                } label: {
                    Label(
                        isActiveState ? "Stop" : "Speak",
                        systemImage: isActiveState ? "stop.fill" : "play.fill"
                    )
                    .frame(minWidth: 80)
                }
                .buttonStyle(.borderedProminent)
                .disabled(inputText.isEmpty && !isActiveState)
                .disabled(!speechEngine.isModelLoaded && !speechEngine.isLoading && speechCoordinator.state == .idle)

                if case .streamingLongForm = speechCoordinator.state {
                    Button {
                        speechCoordinator.pause()
                    } label: {
                        Label("Pause", systemImage: "pause.fill")
                            .frame(minWidth: 70)
                    }
                    .buttonStyle(.bordered)
                }

                if case .paused = speechCoordinator.state {
                    Button {
                        speechCoordinator.resume()
                    } label: {
                        Label("Resume", systemImage: "play.fill")
                            .frame(minWidth: 70)
                    }
                    .buttonStyle(.bordered)
                    .tint(.green)
                }

                if !speechEngine.isModelLoaded && !speechEngine.isLoading {
                    Button {
                        Task {
                            try? await speechEngine.loadModel()
                        }
                    } label: {
                        Label("Download Model", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                Text("Shortcut: \(settings.ttsHotkey.displayString)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
        .padding(.bottom, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}

// MARK: - Speech Status View

private struct SpeechStatusView: View {
    let state: SpeechState
    let isModelLoaded: Bool
    let isLoading: Bool
    let loadingStatus: String

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(statusText)
                .font(.subheadline)
                .fontWeight(.medium)

            if isLoading || isStreamingState {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .frame(height: 24)
        .animation(.easeInOut(duration: 0.2), value: statusText)
    }

    private var isStreamingState: Bool {
        switch state {
        case .streaming, .streamingLongForm:
            true
        default:
            false
        }
    }

    private var statusColor: Color {
        switch state {
        case .idle:
            isModelLoaded ? .green : .secondary
        case .capturingText, .loadingModel:
            .orange
        case .generating:
            .blue
        case .streaming, .streamingLongForm:
            .cyan
        case .paused:
            .yellow
        case .playing:
            .green
        case .error:
            .red
        }
    }

    private var statusText: String {
        switch state {
        case .idle:
            if isLoading {
                return loadingStatus.isEmpty ? "Loading model..." : loadingStatus
            }
            return isModelLoaded ? "Ready" : "Model not loaded"
        case .capturingText:
            return "Capturing text..."
        case .loadingModel:
            return "Loading model..."
        case .generating(let progress):
            return progress.isEmpty ? "Generating speech..." : progress
        case .streaming:
            return "Streaming..."
        case .streamingLongForm(let segment, let total):
            return "Streaming segment \(segment)/\(total)..."
        case .paused(let segment, let total):
            return "Paused (\(segment)/\(total))"
        case .playing:
            return "Playing"
        case .error(let message):
            return message
        }
    }
}
