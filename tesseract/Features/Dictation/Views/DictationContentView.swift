//
//  DictationContentView.swift
//  tesseract
//

import SwiftUI

struct DictationContentView: View {
    @EnvironmentObject private var coordinator: DictationCoordinator
    @EnvironmentObject private var transcriptionEngine: TranscriptionEngine
    @EnvironmentObject private var history: TranscriptionHistory
    @EnvironmentObject private var permissionsManager: PermissionsManager
    @EnvironmentObject private var audioCapture: AudioCaptureEngine
    @ObservedObject private var settings = SettingsManager.shared

    private let contentMaxWidth: CGFloat = Theme.Layout.contentMaxWidth

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Compact recording control - no card, centered layout
                VStack(spacing: 4) {
                    // Button (centered, fixed 96pt)
                    RecordingButtonView(
                        state: coordinator.state,
                        onToggle: { coordinator.toggleRecording() }
                    )
                    .disabled(!transcriptionEngine.isModelLoaded || permissionsManager.microphonePermission != .granted)
                    .frame(height: 96)

                    // Status indicator below button (fixed 36pt)
                    StatusIndicator(state: coordinator.state)
                        .frame(height: 36)

                    // Shortcut hint (fixed 16pt)
                    Text("Shortcut: \(settings.hotkey.displayString)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .frame(height: 16)
                }
                .frame(maxWidth: contentMaxWidth)

                TranscriptionHistoryInlineView(history: history)
                    .frame(maxWidth: contentMaxWidth)
            }
            .padding(.horizontal, 24)
            .padding(.top, 16)
            .padding(.bottom, 20)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Dictation")
    }
}
