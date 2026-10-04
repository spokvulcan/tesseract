//
//  TranscriptionHistoryView.swift
//  tesseract
//
//  The full take history (PRD #612): a timeline grouped by day, shown in the
//  History sheet the Dictation page opens from its toolbar. The page itself
//  keeps only today's takes. A row marks what the **Learned Words** caught;
//  its context menu opens the take in the **Lens** to fix a word, copies it
//  or deletes it. Fixing happens only in the Lens, never here.
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

// MARK: - History sheet

/// Every take the history keeps, opened from the Dictation page's toolbar.
/// Content layer only, no glass (design language §1).
struct TranscriptionHistorySheet: View {
    @Environment(TranscriptionHistory.self) private var history
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if history.flattenedItems.isEmpty {
                HistoryEmptyState()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // The inline view is the scroll view's direct content, so its
                // LazyVStack stays lazy over a long history.
                ScrollView {
                    TranscriptionHistoryInlineView()
                        .frame(maxWidth: Theme.Layout.contentMaxWidth)
                        .padding(.horizontal, 24)
                        .padding(.vertical, DictationPageStyle.rhythm)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .frame(minWidth: 520, idealWidth: 640, minHeight: 420, idealHeight: 640)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("History")
                .font(.system(size: DictationPageStyle.bodySize, weight: .semibold))
            if !history.entries.isEmpty {
                Text(takeCount)
                    .font(.system(size: DictationPageStyle.bodySize))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, DictationPageStyle.rhythm)
    }

    private var takeCount: String {
        let count = history.entries.count
        return count == 1 ? "1 take" : "\(count) takes"
    }
}

// MARK: - Inline History View (no ScrollView, for embedding in parent ScrollView)

/// The history timeline without a scroll view of its own: the parent scrolls
/// it. Reads the history, the **Correction Pairs** and the settings from the
/// environment.
struct TranscriptionHistoryInlineView: View {
    @Environment(TranscriptionHistory.self) private var history
    @Environment(CorrectionPairStore.self) private var pairs

    var body: some View {
        if history.flattenedItems.isEmpty {
            HistoryEmptyState()
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
                            pair: entry.pairID.flatMap { pairs.pair(withID: $0) },
                            isFirst: isFirst,
                            isLast: isLast,
                            onDelete: { history.delete(entry) }
                        )
                    }
                }
            }
            .padding(.horizontal, 4)
            .accessibilityLabel("Transcription history, \(history.entries.count) items")
        }
    }
}

// MARK: - Empty state

private struct HistoryEmptyState: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        ContentUnavailableView(
            "No transcriptions yet",
            systemImage: "waveform",
            description: Text(
                "Press \(settings.hotkey.displayString) to start dictating in any app.")
        )
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

struct TimelineEntryRow: View {
    let entry: TranscriptionEntry
    /// The entry's **Correction Pair** (nil for entries predating the
    /// flywheel, ticket #289). Its glyphs say the take was flagged wrong or
    /// fixed; the fixing itself happens in the **Lens**.
    let pair: CorrectionPair?
    let isFirst: Bool
    let isLast: Bool
    let onDelete: () -> Void

    @Environment(\.fixInLens) private var fixInLens
    @State private var isHovered = false

    // Use cached formatter from TranscriptionHistory
    private var timeString: String {
        TranscriptionHistory.formattedTime(for: entry.timestamp)
    }

    private var accessibilityText: String {
        let seconds = String(format: "%.1f", entry.duration)
        return "\(entry.text), recorded at \(timeString), duration \(seconds) seconds"
    }

    /// What the glyphs and the marked words show, for VoiceOver (the label
    /// replaces the combined children's).
    private var accessibilityState: String {
        var parts: [String] = []
        if let pair, pair.flaggedWrong { parts.append("flagged wrong") }
        if let pair, pair.correction != nil || !pair.fixes.isEmpty { parts.append("fixed") }
        let caught = CaughtText.help(for: entry.catches)
        if !caught.isEmpty { parts.append(caught) }
        return parts.joined(separator: ", ")
    }

    /// A take is fixed through its Correction Pair; entries from before the
    /// pairs have none, so nothing could keep their fix.
    private var canFix: Bool { entry.pairID != nil }

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
                // The row takes no click (fixing is in the context menu), so
                // its text stays selectable.
                CaughtText(text: entry.text, catches: entry.catches)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 6) {
                    Text(String(format: "%.1fs", entry.duration))
                        .font(.system(size: DictationPageStyle.bodySize))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    // Presence in ink, not badges (design language §2): a
                    // gold pair reads as one quiet glyph.
                    if let pair {
                        if pair.flaggedWrong {
                            Image(systemName: "flag.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.orange)
                                .accessibilityLabel("Flagged wrong")
                        }
                        if pair.correction != nil || !pair.fixes.isEmpty {
                            Image(systemName: "pencil")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                                .accessibilityLabel("Fixed")
                        }
                    }
                }
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
        .accessibilityLabel(accessibilityText)
        .accessibilityValue(accessibilityState)
        .accessibilityHint(
            canFix
                ? "Use the context menu to copy, fix a word, or delete"
                : "Use the context menu to copy or delete")
    }

    @ViewBuilder
    private var actionMenuItems: some View {
        if canFix {
            Button {
                fixAWord()
            } label: {
                Label("Fix a Word…", systemImage: "pencil")
            }
        }

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

    /// Opens the take in the Lens; it refuses while a take is being
    /// recorded or a fix is being put back in an app.
    private func fixAWord() {
        if !fixInLens(entry.lensTake) {
            NSSound.beep()
        }
    }

    private func copyEntry() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(entry.text, forType: .string)
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
