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
    /// The entry whose **Correction Pair** detail is open (raw / polished /
    /// correction, ticket #289). One at a time — the detail is an editor,
    /// not a comparison table. Owned by the page so the overlay "edit"
    /// affordance's reveal can drive it deterministically.
    @Binding var expandedEntryID: UUID?
    @Environment(SettingsManager.self) private var settings
    @Environment(CorrectionPairStore.self) private var pairs

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
                            pair: entry.pairID.flatMap { pairs.pair(withID: $0) },
                            isFirst: isFirst,
                            isLast: isLast,
                            isExpanded: expandedEntryID == entry.id,
                            onDelete: { history.delete(entry) },
                            onToggleExpand: {
                                expandedEntryID = expandedEntryID == entry.id ? nil : entry.id
                            },
                            onSaveCorrection: { text in
                                if let pairID = entry.pairID {
                                    pairs.setCorrection(text, for: pairID)
                                }
                            },
                            onFlagWrong: {
                                if let pairID = entry.pairID {
                                    pairs.flagWrong(pairID)
                                }
                            }
                        )
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

struct TimelineEntryRow: View {
    let entry: TranscriptionEntry
    /// The entry's **Correction Pair** (nil for pre-flywheel entries): the
    /// take's text lineage, edited right here — full editing lives in the
    /// history, never the overlay (ticket #289).
    let pair: CorrectionPair?
    let isFirst: Bool
    let isLast: Bool
    let isExpanded: Bool
    let onDelete: () -> Void
    let onToggleExpand: () -> Void
    let onSaveCorrection: (String) -> Void
    let onFlagWrong: () -> Void

    @State private var isHovered = false
    @State private var correctionDraft = ""

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
                        if pair.correction != nil {
                            Image(systemName: "pencil")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                                .accessibilityLabel("Has a correction")
                        }
                    }
                }

                if isExpanded, let pair {
                    CorrectionDetailView(
                        pair: pair,
                        draft: $correctionDraft,
                        onSave: { onSaveCorrection(correctionDraft) },
                        onFlagWrong: onFlagWrong
                    )
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
        .onChange(of: isExpanded, initial: true) { _, expanded in
            if expanded {
                correctionDraft = pair?.correction ?? pair?.committed ?? entry.text
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "\(entry.text), recorded at \(timeString), duration \(String(format: "%.1f", entry.duration)) seconds"
        )
        .accessibilityHint("Use the context menu to copy, correct, or delete")
    }

    @ViewBuilder
    private var actionMenuItems: some View {
        Button {
            copyEntry()
        } label: {
            Label("Copy", systemImage: "doc.on.doc")
        }

        if pair != nil {
            Button {
                onToggleExpand()
            } label: {
                Label(
                    isExpanded ? "Hide Details" : "Edit Correction",
                    systemImage: isExpanded ? "chevron.up" : "pencil")
            }
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
}

// MARK: - Correction detail (the pair editor)

/// The take's text lineage, side by side, with the correction editor — the
/// **Correction Pair**'s gold half is written here (ticket #289).
private struct CorrectionDetailView: View {
    let pair: CorrectionPair
    @Binding var draft: String
    let onSave: () -> Void
    let onFlagWrong: () -> Void

    private var hasUnsavedEdit: Bool {
        draft.trimmingCharacters(in: .whitespacesAndNewlines)
            != (pair.correction ?? pair.committed ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            lineageLine("Raw", pair.rawASR)
            if pair.cleaned != pair.rawASR {
                lineageLine("Cleaned", pair.cleaned)
            }
            if let proofread = pair.proofread {
                lineageLine("Polished", proofread)
            }
            if let reason = pair.rejectReason {
                lineageLine("Rejected", reason)
            }

            HStack(alignment: .center, spacing: 6) {
                TextField("Correction", text: $draft, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: DictationPageStyle.bodySize))
                    .lineLimit(1...4)
                Button("Save") { onSave() }
                    .disabled(!hasUnsavedEdit)
            }

            Button {
                onFlagWrong()
            } label: {
                Label(
                    pair.flaggedWrong ? "Flagged wrong" : "Flag as wrong",
                    systemImage: pair.flaggedWrong ? "flag.fill" : "flag")
            }
            .buttonStyle(.plain)
            .font(.system(size: DictationPageStyle.bodySize))
            .foregroundStyle(pair.flaggedWrong ? Color.orange : Color.secondary)
            .disabled(pair.flaggedWrong)
        }
        .padding(.top, 4)
    }

    private func lineageLine(_ label: String, _ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.system(size: DictationPageStyle.bodySize, weight: .medium))
                .foregroundStyle(.tertiary)
                .frame(width: 64, alignment: .trailing)
            Text(text)
                .font(.system(size: DictationPageStyle.bodySize))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
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
