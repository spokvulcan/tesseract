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
                    .fill(buttonBackgroundColor.gradient)
                    .frame(width: buttonSize, height: buttonSize)
                    .overlay(
                        Circle()
                            .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
                    )
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

    private var groupedEntries: [(String, [TranscriptionEntry])] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: history.entries) { entry -> String in
            if calendar.isDateInToday(entry.timestamp) {
                return "TODAY"
            } else if calendar.isDateInYesterday(entry.timestamp) {
                return "YESTERDAY"
            } else {
                let formatter = DateFormatter()
                formatter.dateFormat = "EEEE, MMM d"
                return formatter.string(from: entry.timestamp).uppercased()
            }
        }

        // Sort groups by most recent first
        let sortedKeys = grouped.keys.sorted { key1, key2 in
            let date1 = grouped[key1]?.first?.timestamp ?? Date.distantPast
            let date2 = grouped[key2]?.first?.timestamp ?? Date.distantPast
            return date1 > date2
        }

        return sortedKeys.map { key in
            (key, grouped[key]?.sorted { $0.timestamp > $1.timestamp } ?? [])
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("History", systemImage: "clock")
                    .font(.title3)

                Spacer()

                Button("Clear") {
                    history.clear()
                }
                .buttonStyle(.borderless)
                .disabled(history.entries.isEmpty)
                .accessibilityLabel("Clear transcription history")
            }

            if history.entries.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "text.bubble")
                        .font(.system(size: 32))
                        .foregroundStyle(.tertiary)
                    Text("No transcriptions yet")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 220)
                .padding(.vertical, 16)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(groupedEntries, id: \.0) { dateGroup, entries in
                            HistoryDateSection(dateLabel: dateGroup, entries: entries)
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

// MARK: - History Date Section

struct HistoryDateSection: View {
    let dateLabel: String
    let entries: [TranscriptionEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Date header
            Text(dateLabel)
                .font(.callout)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .padding(.top, 14)
                .padding(.bottom, 6)
                .padding(.leading, 2)

            // Timeline entries
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(entries.enumerated()), id: \.element.id) { index, entry in
                    TimelineEntryRow(
                        entry: entry,
                        isFirst: index == 0,
                        isLast: index == entries.count - 1
                    )
                }
            }
        }
    }
}

// MARK: - Timeline Entry Row

struct TimelineEntryRow: View {
    let entry: TranscriptionEntry
    let isFirst: Bool
    let isLast: Bool

    @State private var isHovered = false

    private var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: entry.timestamp)
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
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.text), recorded at \(timeString), duration \(String(format: "%.1f", entry.duration)) seconds")
        .accessibilityHint("Copy button appears on hover")
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
