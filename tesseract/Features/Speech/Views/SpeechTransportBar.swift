//
//  SpeechTransportBar.swift
//  tesseract
//

import SwiftUI

/// Floating Liquid Glass transport for the Speech page: a morphing
/// Speak/Stop control, long-form pause/resume, and a status capsule.
struct SpeechTransportBar: View {
    let state: SpeechState
    let isModelLoading: Bool
    let modelLoadingStatus: String
    let hasText: Bool
    let hotkeyHint: String
    let onSpeak: () -> Void
    let onStop: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void

    @Namespace private var glassNamespace

    /// Tall enough for the large glass buttons and the status capsule alike,
    /// so the bar (and the composer above it) never resizes as elements
    /// come and go with playback state.
    private let controlRowHeight: CGFloat = 40

    var body: some View {
        VStack(spacing: Theme.Spacing.sm) {
            GlassEffectContainer(spacing: Theme.Spacing.md) {
                HStack(spacing: Theme.Spacing.md) {
                    primaryButton

                    if case .streamingLongForm = state {
                        pauseButton
                    }

                    if case .paused = state {
                        resumeButton
                    }

                    if let status {
                        statusCapsule(status)
                    }
                }
                .frame(height: controlRowHeight)
            }
            .animation(.smooth(duration: 0.25), value: state)
            .animation(.smooth(duration: 0.25), value: isModelLoading)

            Text("\(hotkeyHint) speaks the selected text in any app")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Controls

    /// Anything that can be stopped counts as active; `idle` and `error` offer Speak.
    private var isActive: Bool {
        switch state {
        case .idle, .error:
            false
        default:
            true
        }
    }

    private var primaryButton: some View {
        Button {
            isActive ? onStop() : onSpeak()
        } label: {
            Label(
                isActive ? "Stop" : "Speak",
                systemImage: isActive ? "stop.fill" : "play.fill"
            )
            .frame(minWidth: 90)
        }
        .buttonStyle(.glassProminent)
        .tint(isActive ? .red : .accentColor)
        .controlSize(.large)
        .disabled(!isActive && !hasText)
        .keyboardShortcut(isActive ? KeyboardShortcut(.escape) : KeyboardShortcut(.return, modifiers: .command))
        .help(isActive ? "Stop speaking (Esc)" : "Speak text (⌘↩)")
        .glassEffectID("primary", in: glassNamespace)
    }

    private var pauseButton: some View {
        Button(action: onPause) {
            Label("Pause", systemImage: "pause.fill")
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.glass)
        .controlSize(.large)
        .help("Pause after the current segment")
        .glassEffectID("pauseResume", in: glassNamespace)
    }

    private var resumeButton: some View {
        Button(action: onResume) {
            Label("Resume", systemImage: "play.fill")
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.glass)
        .controlSize(.large)
        .help("Resume from the next segment")
        .glassEffectID("pauseResume", in: glassNamespace)
    }

    // MARK: - Status

    private struct Status: Equatable {
        var text: String
        var isSpinning = false
        var isError = false
        var isPlaying = false
        var progress: (segment: Int, total: Int)?

        static func == (lhs: Status, rhs: Status) -> Bool {
            lhs.text == rhs.text && lhs.isSpinning == rhs.isSpinning
                && lhs.isError == rhs.isError && lhs.isPlaying == rhs.isPlaying
                && lhs.progress?.segment == rhs.progress?.segment
                && lhs.progress?.total == rhs.progress?.total
        }
    }

    private var status: Status? {
        if isModelLoading {
            let text = modelLoadingStatus.isEmpty ? "Loading model…" : modelLoadingStatus
            return Status(text: text, isSpinning: true)
        }
        switch state {
        case .idle:
            return nil
        case .capturingText:
            return Status(text: "Capturing selected text…", isSpinning: true)
        case .generating(let progress):
            return Status(text: progress.isEmpty ? "Generating…" : progress, isSpinning: true)
        case .streaming:
            return Status(text: "Speaking", isPlaying: true)
        case .streamingLongForm(let segment, let total):
            return Status(text: "Segment \(segment) of \(total)", isPlaying: true, progress: (segment, total))
        case .paused(let segment, let total):
            return Status(text: "Paused · \(segment) of \(total)", progress: (segment, total))
        case .playing:
            return Status(text: "Speaking", isPlaying: true)
        case .error(let message):
            return Status(text: message, isError: true)
        }
    }

    private func statusCapsule(_ status: Status) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            if status.isSpinning {
                ProgressView()
                    .controlSize(.small)
            }
            if status.isPlaying {
                Image(systemName: "speaker.wave.3.fill")
                    .symbolEffect(.variableColor.iterative, isActive: true)
                    .foregroundStyle(.secondary)
            }
            if status.isError {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if let progress = status.progress {
                ProgressView(value: Double(progress.segment), total: Double(progress.total))
                    .frame(width: 56)
            }
            Text(status.text)
                .font(.callout)
                .foregroundStyle(status.isError ? .red : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.sm + 2)
        .glassEffect()
        .glassEffectID("status", in: glassNamespace)
    }
}
