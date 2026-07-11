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
    @Environment(CorrectionPairStore.self) private var pairs
    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings

    private let contentMaxWidth: CGFloat = Theme.Layout.contentMaxWidth

    /// The entry whose correction editor is open — page-owned so the overlay
    /// "edit" affordance's reveal drives scroll and expansion together.
    @State private var expandedEntryID: UUID?

    var body: some View {
        ScrollViewReader { proxy in
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

                    TranscriptionHistoryInlineView(
                        history: history, expandedEntryID: $expandedEntryID
                    )
                    .frame(maxWidth: contentMaxWidth)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, DictationPageStyle.rhythm)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: history.focusEntryID, initial: true) { _, focused in
                // The overlay "edit" affordance staged this reveal: open the
                // entry's editor, scroll to it, consume the request.
                guard let focused else { return }
                expandedEntryID = focused
                withAnimation {
                    proxy.scrollTo("entry-\(focused.uuidString)", anchor: .center)
                }
                history.focusEntryID = nil
            }
        }
        .toolbar {
            ToolbarItem {
                Button {
                    exportCorrections()
                } label: {
                    Label("Export Corrections…", systemImage: "square.and.arrow.up")
                }
                .disabled(pairs.pairs.isEmpty)
                .help("Export the correction pairs as JSONL")
            }
        }
        .navigationTitle("Dictation")
    }

    /// Writes the Correction Pair corpus as JSONL wherever the owner points —
    /// the flywheel's export half (fine-tuning itself is out of scope, #294).
    private func exportCorrections() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "correction-pairs.jsonl"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try pairs.exportJSONL().write(to: url, options: .atomic)
        } catch {
            Log.transcription.error("Correction export failed: \(error.localizedDescription)")
        }
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
