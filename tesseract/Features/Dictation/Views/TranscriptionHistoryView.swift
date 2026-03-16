//
//  TranscriptionHistoryView.swift
//  tesseract
//

import SwiftUI

// MARK: - History View (with ScrollView)

struct TranscriptionHistoryView: View {
    var history: TranscriptionHistory
    @Environment(SettingsManager.self) private var settings

    // Layout constants for header alignment
    private let timeColumnWidth: CGFloat = 70
    private let timeToConnectorSpacing: CGFloat = 12
    private let connectorWidth: CGFloat = 8
    private let connectorToContentSpacing: CGFloat = 12

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if history.flattenedItems.isEmpty {
                HistoryEmptyStateView(hotkey: settings.hotkey.displayString)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                    .padding(.vertical, 16)
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

// MARK: - Inline History View (no ScrollView, for embedding in parent ScrollView)

struct TranscriptionHistoryInlineView: View {
    var history: TranscriptionHistory
    @Environment(SettingsManager.self) private var settings

    private let timeColumnWidth: CGFloat = 70
    private let timeToConnectorSpacing: CGFloat = 12
    private let connectorWidth: CGFloat = 8
    private let connectorToContentSpacing: CGFloat = 12

    var body: some View {
        if history.flattenedItems.isEmpty {
            HistoryEmptyStateView(hotkey: settings.hotkey.displayString)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 16)
        } else {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(history.flattenedItems) { item in
                    switch item {
                    case .header(let label, _):
                        HistorySectionHeader(
                            label: label,
                            leadingPadding: timeColumnWidth + timeToConnectorSpacing + connectorWidth + connectorToContentSpacing
                        )
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

// MARK: - History Empty State

private struct HistoryEmptyStateView: View {
    let hotkey: String

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(Color.secondary.opacity(0.12))
                    .frame(width: 44, height: 44)

                Image(systemName: "waveform")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.secondary)
            }

            Text("No transcriptions yet")
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundStyle(.primary)

            Text("Press \(hotkey) to start dictating.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: 300)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("No transcriptions yet. Press \(hotkey) to start dictating.")
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
    let onDelete: () -> Void

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
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .padding(.trailing, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isHovered ? Color.primary.opacity(0.04) : Color.clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
            .overlay(alignment: .topTrailing) {
                if isHovered {
                    Menu {
                        actionMenuItems
                    } label: {
                        Label("More actions", systemImage: "ellipsis")
                            .labelStyle(.iconOnly)
                    }
                    .menuIndicator(.hidden)
                    .controlSize(.small)
                    .help("More actions")
                    .accessibilityLabel("More actions")
                    .padding(.top, 6)
                    .padding(.trailing, 8)
                }
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onHover { isHovered = $0 }  // No animation - instant state change
        .contextMenu {
            actionMenuItems
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.text), recorded at \(timeString), duration \(String(format: "%.1f", entry.duration)) seconds")
        .accessibilityHint("Actions menu appears on hover")
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
        lhs.entry.id == rhs.entry.id &&
        lhs.isFirst == rhs.isFirst &&
        lhs.isLast == rhs.isLast
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
    }
}
