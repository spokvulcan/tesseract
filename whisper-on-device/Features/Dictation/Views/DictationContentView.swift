//
//  DictationContentView.swift
//  whisper-on-device
//

import SwiftUI

struct DictationContentView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var history: TranscriptionHistory
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var audioCapture: AudioCaptureEngine

    private let contentMaxWidth: CGFloat = 820

    var body: some View {
        VStack(spacing: 16) {
            VStack(spacing: 16) {
                StatusHeader(
                    state: coordinator.state,
                    isModelLoaded: transcriptionEngine.isModelLoaded,
                    modelName: transcriptionEngine.loadedModel?.displayName
                )

                RecordingButtonView(
                    state: coordinator.state,
                    onToggle: { coordinator.toggleRecording() }
                )
                .disabled(!transcriptionEngine.isModelLoaded || permissionsManager.microphonePermission != .granted)

                WaveformVisualizer(
                    audioCapture: audioCapture,
                    state: coordinator.state
                )
                .frame(height: 72)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.thinMaterial)
                )

                Text("Shortcut: Shift+Command+D")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(.regularMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(.quaternary, lineWidth: 1)
                    )
            )
            .frame(maxWidth: contentMaxWidth)

            if !coordinator.lastTranscription.isEmpty {
                LastTranscriptionView(text: coordinator.lastTranscription)
                    .frame(maxWidth: contentMaxWidth)
            }

            TranscriptionHistoryView(history: history)
                .frame(maxWidth: contentMaxWidth)
                .frame(maxHeight: .infinity, alignment: .top)
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
        .padding(.bottom, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .navigationTitle("Dictation")
    }
}

#Preview {
    let audioCapture = AudioCaptureEngine()
    return DictationContentView(
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
