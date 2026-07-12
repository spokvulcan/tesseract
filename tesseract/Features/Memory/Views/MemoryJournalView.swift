//
//  MemoryJournalView.swift
//  tesseract
//
//  The consolidation journal (ADR-0035 §7): every mutation sleep — or the
//  owner — makes to the derived layer, newest first. This is how a bad
//  consolidation becomes visible instead of silently rewriting a belief.
//

import SwiftUI

struct MemoryJournalView: View {
    let entries: [JournalEntry]
    /// The unfiltered filter string, for the no-matches empty state.
    let filter: String

    var body: some View {
        if entries.isEmpty {
            emptyState
        } else {
            List(entries) { entry in
                MemoryJournalRow(entry: entry)
                    .frame(maxWidth: MemoryWindowLayout.columnMaxWidth)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        if filter.isEmpty {
            ContentUnavailableView {
                Label("Nothing Journaled Yet", systemImage: "moon.zzz")
            } description: {
                Text(
                    """
                    Every change consolidation makes to my memories — additions, \
                    confirmations, promotions, retirements — is recorded here.
                    """)
            }
        } else {
            ContentUnavailableView.search(text: filter)
        }
    }
}

// MARK: - Row

struct MemoryJournalRow: View {
    let entry: JournalEntry

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: entry.mutation.symbol)
                .foregroundStyle(entry.mutation.color)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(entry.mutation.label)
                        .fontWeight(.medium)
                    Spacer()
                    Text(entry.at.memoryMoment)
                        .foregroundStyle(.secondary)
                }
                Text(entry.detail)
                if let before = entry.before {
                    quote("Before", before, style: .secondary)
                }
                if let after = entry.after {
                    quote("After", after, style: .primary)
                }
            }
        }
        .padding(.vertical, 6)
    }

    /// A quiet verbatim block. `before` reads secondary, `after` primary, so
    /// the direction of a rewrite is visible without any diff chrome.
    private func quote(_ label: String, _ text: String, style: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .foregroundStyle(.secondary)
            Text(text)
                .foregroundStyle(style)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}
