//
//  SpeechContentView.swift
//  tesseract
//

import SwiftUI

struct SpeechContentView: View {
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(SpeechEngine.self) private var speechEngine
    @Environment(SettingsManager.self) private var settings

    @AppStorage("ttsParametersPanelVisible") private var isParametersPanelVisible: Bool = true
    @State private var inputText: String = ""
    @Namespace private var glassNamespace

    private var isActiveState: Bool {
        switch speechCoordinator.state {
        case .playing, .streaming, .streamingLongForm, .paused:
            true
        default:
            false
        }
    }

    var body: some View {
        mainContent
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .trailing) {
                if isParametersPanelVisible {
                    TTSParametersSidebar()
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                        .padding(12)
                }
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            isParametersPanelVisible.toggle()
                        }
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .help("Toggle Parameters")
                }
            }
            .navigationTitle("Speech")
    }

    // MARK: - Main Content

    private var mainContent: some View {
        @Bindable var settings = settings
        return VStack(spacing: 16) {
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

                    if inputText.isEmpty {
                        Text("Enter text to speak, or use \(settings.ttsHotkey.displayString) to speak selected text from any app")
                            .font(.body)
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .padding(8)
                .background(.fill.quaternary)
                .clipShape(RoundedRectangle(cornerRadius: 8))
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
                GlassEffectContainer(spacing: 12) {
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
                    .buttonStyle(.glassProminent)
                    .tint(isActiveState ? .red : .accentColor)
                    .disabled(inputText.isEmpty && !isActiveState)
                    .glassEffectID("primary", in: glassNamespace)

                    if case .streamingLongForm = speechCoordinator.state {
                        Button {
                            speechCoordinator.pause()
                        } label: {
                            Label("Pause", systemImage: "pause.fill")
                                .frame(minWidth: 70)
                        }
                        .buttonStyle(.glass)
                        .glassEffectID("pauseResume", in: glassNamespace)
                    }

                    if case .paused = speechCoordinator.state {
                        Button {
                            speechCoordinator.resume()
                        } label: {
                            Label("Resume", systemImage: "play.fill")
                                .frame(minWidth: 70)
                        }
                        .buttonStyle(.glass)
                        .tint(.green)
                        .glassEffectID("pauseResume", in: glassNamespace)
                    }
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
            isLoading ? .orange : .green
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
            return "Ready"
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
