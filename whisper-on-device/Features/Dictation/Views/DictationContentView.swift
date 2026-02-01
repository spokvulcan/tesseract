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
