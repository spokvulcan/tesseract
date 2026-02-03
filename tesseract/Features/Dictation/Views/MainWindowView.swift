//
//  MainWindowView.swift
//  tesseract
//

import SwiftUI

struct MainWindowView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var history: TranscriptionHistory
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var audioCapture: AudioCaptureEngine
    @ObservedObject private var settings = SettingsManager.shared

    private let contentMaxWidth: CGFloat = 820

    var body: some View {
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

            TranscriptionHistoryView(history: history)
                .frame(maxWidth: contentMaxWidth)
                .frame(maxHeight: .infinity, alignment: .top)
                .layoutPriority(1)
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
        .padding(.bottom, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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

// MARK: - Status Indicator (Compact, centered)

struct StatusIndicator: View {
    let state: DictationState

    // Fixed height: status line (18) + spacing (2) + detail line (16) = 36
    private let totalHeight: CGFloat = 36

    var body: some View {
        VStack(spacing: 2) {
            // Status line - centered with dot
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)

                Text(statusTitle)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            .frame(height: 18)
            .animation(.easeInOut(duration: 0.2), value: state)

            // Error detail - always reserves space
            Text(statusDetail ?? " ")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(height: 16)
                .opacity(statusDetail != nil ? 1 : 0)
        }
        .frame(height: totalHeight)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Status: \(statusTitle)")
        .accessibilityHint(statusDetail ?? "")
    }

    private var statusTitle: String {
        switch state {
        case .error:
            return "Error"
        default:
            return state.statusText
        }
    }

    private var statusDetail: String? {
        switch state {
        case .error(let message):
            return message
        default:
            return nil
        }
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

    // Compact button: 64px with pulse ring up to 72 × 1.2 = 86px
    private let buttonSize: CGFloat = 64
    private let containerSize: CGFloat = 96

    var body: some View {
        Button(action: onToggle) {
            ZStack {
                // Pulse ring - only rendered when recording to avoid idle CPU usage
                if state == .recording {
                    Circle()
                        .stroke(Color.red.opacity(0.5), lineWidth: 3)
                        .frame(width: 72, height: 72)
                        .scaleEffect(isPulsing ? 1.2 : 1.0)
                        .opacity(isPulsing ? 0 : 1)
                        .animation(
                            .easeInOut(duration: 1.0).repeatForever(autoreverses: false),
                            value: isPulsing
                        )
                }

                Circle()
                    .fill(buttonBackgroundColor.opacity(0.85))
                    .frame(width: buttonSize, height: buttonSize)
                    .glassEffect(.regular.tint(buttonBackgroundColor))
                    .shadow(color: Color.black.opacity(0.15), radius: 8, y: 3)
                    .animation(.easeInOut(duration: 0.2), value: state)

                Image(systemName: buttonIcon)
                    .font(.system(size: 26))
                    .foregroundStyle(.white)
                    .symbolEffect(.pulse, isActive: state == .processing)
            }
            .frame(width: containerSize, height: containerSize)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(state == .recording ? "Double tap to stop recording" : "Double tap to start recording")
        .onChange(of: state) { _, newState in
            isPulsing = newState == .recording
        }
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
            return .accentColor
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

// MARK: - History View

struct TranscriptionHistoryView: View {
    @ObservedObject var history: TranscriptionHistory

    // Layout constants for header alignment
    private let timeColumnWidth: CGFloat = 70
    private let timeToConnectorSpacing: CGFloat = 12
    private let connectorWidth: CGFloat = 8
    private let connectorToContentSpacing: CGFloat = 12

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if history.flattenedItems.isEmpty {
                // Empty state - minimal
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(history.flattenedItems) { item in
                            switch item {
                            case .header(let label, _):
                                HistorySectionHeader(
                                    label: label,
                                    leadingPadding: timeColumnWidth + timeToConnectorSpacing + connectorWidth + connectorToContentSpacing
                                )
                            case .entry(let entry, let isFirst, let isLast):
                                TimelineEntryRow(entry: entry, isFirst: isFirst, isLast: isLast)
                                    .equatable()
                            }
                        }
                    }
                    .padding(.horizontal, 4)
                }
                .scrollContentBackground(.hidden)
                .frame(maxHeight: .infinity)
                .accessibilityLabel("Transcription history, \(history.entries.count) items")
            }
        }
        .frame(maxHeight: .infinity)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - History Section Header

struct HistorySectionHeader: View, Equatable {
    let label: String
    let leadingPadding: CGFloat

    var body: some View {
        Text(label)
            .font(.caption)
            .fontWeight(.medium)
            .foregroundStyle(.tertiary)
            .padding(.top, 16)
            .padding(.bottom, 8)
            .padding(.leading, leadingPadding)
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.label == rhs.label && lhs.leadingPadding == rhs.leadingPadding
    }
}

// MARK: - Timeline Entry Row

struct TimelineEntryRow: View, Equatable {
    let entry: TranscriptionEntry
    let isFirst: Bool
    let isLast: Bool

    @State private var isHovered = false

    // Use cached formatter from TranscriptionHistory
    private var timeString: String {
        TranscriptionHistory.formattedTime(for: entry.timestamp)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Time column
            Text(timeString)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
                .frame(width: 70, alignment: .trailing)
                .padding(.top, 2)

            // Timeline connector
            TimelineConnector(isFirst: isFirst, isLast: isLast)

            // Content
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(entry.text)
                        .font(.system(size: 15))
                        .foregroundStyle(.primary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)

                    Text(String(format: "%.1fs", entry.duration))
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 8)

                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(entry.text, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
                .help("Copy to clipboard")
                .opacity(isHovered ? 1 : 0)
                .allowsHitTesting(isHovered)
                .accessibilityLabel("Copy transcription")
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isHovered ? Color.primary.opacity(0.04) : Color.clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onHover { isHovered = $0 }  // No animation - instant state change
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.text), recorded at \(timeString), duration \(String(format: "%.1f", entry.duration)) seconds")
        .accessibilityHint("Copy button appears on hover")
    }

    // Equatable conformance for efficient diffing
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.entry.id == rhs.entry.id &&
        lhs.isFirst == rhs.isFirst &&
        lhs.isLast == rhs.isLast
    }
}

// MARK: - Timeline Connector (Extracted for simplicity)

private struct TimelineConnector: View {
    let isFirst: Bool
    let isLast: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Line above dot
            Rectangle()
                .fill(isFirst ? Color.clear : Color.secondary.opacity(0.25))
                .frame(width: 1.5)
                .frame(height: 8)

            // Dot
            Circle()
                .fill(Color.secondary.opacity(0.5))
                .frame(width: 6, height: 6)

            // Line below dot
            Rectangle()
                .fill(isLast ? Color.clear : Color.secondary.opacity(0.25))
                .frame(width: 1.5)
                .frame(maxHeight: .infinity)
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
