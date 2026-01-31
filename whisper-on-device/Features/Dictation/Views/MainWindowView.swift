//
//  MainWindowView.swift
//  whisper-on-device
//

import SwiftUI

struct MainWindowView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var history: TranscriptionHistory
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var audioCapture: AudioCaptureEngine

    var body: some View {
        VStack(spacing: 20) {
            // Status Header
            StatusHeader(
                state: coordinator.state,
                isModelLoaded: transcriptionEngine.isModelLoaded,
                modelName: transcriptionEngine.loadedModel?.displayName
            )

            // Recording Button
            RecordingButtonView(
                state: coordinator.state,
                onToggle: { coordinator.toggleRecording() }
            )
            .disabled(!transcriptionEngine.isModelLoaded || permissionsManager.microphonePermission != .granted)

            // Waveform Visualizer
            WaveformVisualizer(
                audioCapture: audioCapture,
                state: coordinator.state
            )

            // Last Transcription
            if !coordinator.lastTranscription.isEmpty {
                LastTranscriptionView(text: coordinator.lastTranscription)
            }

            Divider()

            // History
            TranscriptionHistoryView(history: history)

            Spacer()
        }
        .padding()
        .frame(minWidth: 400, minHeight: 500)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                SettingsLink {
                    Image(systemName: "gear")
                }
            }
        }
    }
}

// MARK: - Status Header

struct StatusHeader: View {
    let state: DictationState
    let isModelLoaded: Bool
    let modelName: String?

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                    .accessibilityHidden(true)

                Text(state.statusText)
                    .font(.headline)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Status: \(state.statusText)")

            if let modelName {
                Text("Model: \(modelName)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !isModelLoaded {
                Text("No model loaded")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .accessibilityLabel("Warning: No transcription model loaded")
            }
        }
        .padding(.vertical)
    }

    private var statusColor: Color {
        switch state {
        case .idle:
            return .green
        case .listening:
            return .yellow
        case .recording:
            return .red
        case .processing:
            return .orange
        case .error:
            return .red
        }
    }
}

// MARK: - Recording Button

struct RecordingButtonView: View {
    let state: DictationState
    let onToggle: () -> Void

    @State private var isPulsing = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Button(action: onToggle) {
            ZStack {
                Circle()
                    .fill(buttonBackgroundColor)
                    .frame(width: 80, height: 80)

                if state == .recording && !reduceMotion {
                    Circle()
                        .stroke(Color.red.opacity(0.5), lineWidth: 4)
                        .frame(width: 90, height: 90)
                        .scaleEffect(isPulsing ? 1.2 : 1.0)
                        .opacity(isPulsing ? 0 : 1)
                        .animation(
                            .easeInOut(duration: 1.0).repeatForever(autoreverses: false),
                            value: isPulsing
                        )
                }

                Image(systemName: buttonIcon)
                    .font(.system(size: 32))
                    .foregroundStyle(.white)
                    .symbolEffect(.pulse, isActive: state == .processing && !reduceMotion)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(state == .recording ? "Double tap to stop recording" : "Double tap to start recording")
        .onChange(of: state) { _, newState in
            isPulsing = newState == .recording
        }
        .sensoryFeedback(.impact, trigger: state)
    }

    private var accessibilityLabel: String {
        switch state {
        case .idle: return "Start Recording"
        case .listening: return "Listening for voice"
        case .recording: return "Recording in progress"
        case .processing: return "Processing transcription"
        case .error: return "Recording error"
        }
    }

    private var buttonBackgroundColor: Color {
        switch state {
        case .idle:
            return .secondary
        case .listening:
            return .yellow
        case .recording:
            return .red
        case .processing:
            return .orange
        case .error:
            return .red.opacity(0.5)
        }
    }

    private var buttonIcon: String {
        switch state {
        case .idle, .error:
            return "mic.fill"
        case .listening:
            return "ear.fill"
        case .recording:
            return "stop.fill"
        case .processing:
            return "waveform"
        }
    }
}

// MARK: - Last Transcription

struct LastTranscriptionView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Last Transcription")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help("Copy to clipboard")
                .accessibilityLabel("Copy transcription to clipboard")
            }

            Text(text)
                .font(.body)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.regularMaterial)
                .cornerRadius(8)
                .accessibilityLabel("Transcription: \(text)")
        }
    }
}

// MARK: - History View

struct TranscriptionHistoryView: View {
    @ObservedObject var history: TranscriptionHistory
    @State private var searchText = ""

    var filteredEntries: [TranscriptionEntry] {
        if searchText.isEmpty {
            return history.entries
        }
        return history.search(searchText)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("History")
                    .font(.headline)

                Spacer()

                if !history.entries.isEmpty {
                    Button("Clear") {
                        history.clear()
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.red)
                    .accessibilityLabel("Clear all history")
                    .accessibilityHint("Double tap to delete all transcription history")
                }
            }

            TextField("Search...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel("Search transcription history")

            if filteredEntries.isEmpty {
                Text("No transcriptions yet")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                List {
                    ForEach(filteredEntries) { entry in
                        TranscriptionEntryRow(entry: entry)
                    }
                    .onDelete { offsets in
                        history.delete(at: offsets)
                    }
                }
                .listStyle(.inset)
                .accessibilityLabel("Transcription history, \(filteredEntries.count) items")
            }
        }
    }
}

struct TranscriptionEntryRow: View {
    let entry: TranscriptionEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.text)
                .lineLimit(2)

            HStack {
                Text(entry.timestamp, style: .relative)
                Text("•")
                    .accessibilityHidden(true)
                Text(String(format: "%.1fs", entry.duration))
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.text), recorded \(String(format: "%.1f", entry.duration)) seconds")
        .accessibilityHint("Right-click to copy")
        .contextMenu {
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(entry.text, forType: .string)
            }
        }
    }
}

#Preview {
    let audioCapture = AudioCaptureEngine()
    return MainWindowView(
        coordinator: DictationCoordinator(
            audioCapture: audioCapture,
            transcriptionEngine: TranscriptionEngine(),
            textInjector: TextInjector(),
            history: TranscriptionHistory()
        ),
        transcriptionEngine: TranscriptionEngine(),
        history: TranscriptionHistory(),
        permissionsManager: PermissionsManager(),
        audioCapture: audioCapture
    )
}
