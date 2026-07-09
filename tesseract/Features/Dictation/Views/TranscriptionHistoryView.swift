//
//  TranscriptionHistoryView.swift
//  tesseract
//

import SwiftUI

// MARK: - Timeline layout

private enum TimelineLayout {
    static let timeColumnWidth: CGFloat = 84
    static let timeToConnectorSpacing: CGFloat = 12
    static let connectorWidth: CGFloat = 8
    static let connectorToContentSpacing: CGFloat = 12

    /// Left edge of the content column, where section headers align.
    static var contentLeadingPadding: CGFloat {
        timeColumnWidth + timeToConnectorSpacing + connectorWidth + connectorToContentSpacing
    }
}

// MARK: - Inline History View (no ScrollView, for embedding in parent ScrollView)

struct TranscriptionHistoryInlineView: View {
    var history: TranscriptionHistory
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        if history.flattenedItems.isEmpty {
            ContentUnavailableView(
                "No transcriptions yet",
                systemImage: "waveform",
                description: Text(
                    "Press \(settings.hotkey.displayString) to start dictating in any app.")
            )
            .padding(.vertical, DictationPageStyle.rhythm)
        } else {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(history.flattenedItems) { item in
                    switch item {
                    case .header(let label, _):
                        HistorySectionHeader(label: label)
                    case .entry(let entry, let isFirst, let isLast):
                        TimelineEntryRow(
                            entry: entry,
                            isFirst: isFirst,
                            isLast: isLast,
                            onDelete: { history.delete(entry) }
                        )
                        .equatable()
                    }
                }
            }
            .padding(.horizontal, 4)
            .accessibilityLabel("Transcription history, \(history.entries.count) items")
        }
    }
}

// MARK: - History Section Header

struct HistorySectionHeader: View, Equatable {
    let label: String

    var body: some View {
        Text(label)
            .font(.system(size: DictationPageStyle.bodySize, weight: .medium))
            .foregroundStyle(.tertiary)
            .padding(.top, DictationPageStyle.rhythm)
            .padding(.leading, TimelineLayout.contentLeadingPadding)
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.label == rhs.label
    }
}

// MARK: - Timeline Entry Row

struct TimelineEntryRow: View, Equatable {
    let entry: TranscriptionEntry
    let isFirst: Bool
    let isLast: Bool
    let onDelete: () -> Void

    @State private var isHovered = false

    // Use cached formatter from TranscriptionHistory
    private var timeString: String {
        TranscriptionHistory.formattedTime(for: entry.timestamp)
    }

    var body: some View {
        HStack(alignment: .top, spacing: TimelineLayout.timeToConnectorSpacing) {
            // Time column
            Text(timeString)
                .font(.system(size: DictationPageStyle.bodySize, weight: .medium))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
                .frame(width: TimelineLayout.timeColumnWidth, alignment: .trailing)
                .padding(.top, 2)

            // Timeline connector
            TimelineConnector(isFirst: isFirst, isLast: isLast)

            // Content
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.text)
                    .font(.system(size: DictationPageStyle.bodySize))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                Text(String(format: "%.1fs", entry.duration))
                    .font(.system(size: DictationPageStyle.bodySize))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
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
        .contextMenu {
            actionMenuItems
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "\(entry.text), recorded at \(timeString), duration \(String(format: "%.1f", entry.duration)) seconds"
        )
        .accessibilityHint("Use the context menu to copy or delete")
    }

    @ViewBuilder
    private var actionMenuItems: some View {
        Button {
            copyEntry()
        } label: {
            Label("Copy", systemImage: "doc.on.doc")
        }

        Button(role: .destructive) {
            onDelete()
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }

    private func copyEntry() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(entry.text, forType: .string)
    }

    // Equatable conformance for efficient diffing
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.entry.id == rhs.entry.id && lhs.isFirst == rhs.isFirst && lhs.isLast == rhs.isLast
    }
}

// MARK: - Timeline Connector

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
        .frame(width: TimelineLayout.connectorWidth)
    }
}
