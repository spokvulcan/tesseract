//
//  DictationContentView.swift
//  tesseract
//

import SwiftUI

/// Dictation page surface constants (design language §2: one type size and
/// one spacing rhythm per surface; hierarchy comes from weight and color).
enum DictationPageStyle {
    static let bodySize: CGFloat = 15
    static let rhythm: CGFloat = 12
}

struct DictationContentView: View {
    @Environment(DictationCoordinator.self) private var coordinator
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @Environment(TranscriptionHistory.self) private var history
    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings

    private let contentMaxWidth: CGFloat = Theme.Layout.contentMaxWidth

    var body: some View {
        ScrollView {
            VStack(spacing: DictationPageStyle.rhythm) {
                VStack(spacing: DictationPageStyle.rhythm) {
                    RecordingButtonView(
                        state: coordinator.state,
                        onToggle: { coordinator.toggleRecording() }
                    )
                    .disabled(
                        !transcriptionEngine.isModelLoaded
                            || permissionsManager.microphonePermission != .granted
                    )
                    .frame(height: 96)

                    statusLine
                        .frame(height: 44)

                    Text("Shortcut: \(settings.hotkey.displayString)")
                        .font(.system(size: DictationPageStyle.bodySize))
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: contentMaxWidth)

                TranscriptionHistoryInlineView(history: history)
                    .frame(maxWidth: contentMaxWidth)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, DictationPageStyle.rhythm)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Dictation")
    }

    /// While the mic permission or the model load is still pending, the
    /// dictation state reads "Ready" even though the button is disabled —
    /// surface the blocker instead (design language §2: quiet loading).
    @ViewBuilder
    private var statusLine: some View {
        if permissionsManager.microphonePermission != .granted {
            StatusIndicator(
                badge: .dot(.secondary),
                title: "Microphone access needed",
                detail: "Grant access in System Settings › Privacy & Security › Microphone."
            )
        } else if !transcriptionEngine.isModelLoaded {
            StatusIndicator(
                badge: .spinner,
                title: "Loading dictation model…",
                detail: nil
            )
        } else {
            StatusIndicator(state: coordinator.state)
        }
    }
}
