//
//  MemoryBeliefListView.swift
//  tesseract
//
//  The Memory window's list column: every derived memory, grouped by tier,
//  each row leading with its provenance. Rows are icon-light — actions live
//  in the context menu (design language §2) and in the detail pane.
//

import SwiftUI

struct MemoryBeliefListView: View {
    /// Already filtered by the window's filter field.
    let memories: [MemoryRecord]
    /// The unfiltered filter string, for the no-matches empty state.
    let filter: String
    let stats: MemoryStats
    @Binding var selection: UUID?
    let onContest: (MemoryRecord) -> Void
    let onDelete: (MemoryRecord) -> Void

    var body: some View {
        VStack(spacing: 0) {
            if memories.isEmpty {
                emptyState
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                list
            }
            Divider()
            statsFooter
        }
    }

    // MARK: - List

    /// Tiers in ladder order, empty tiers omitted.
    private var tiers: [(tier: MemoryTier, records: [MemoryRecord])] {
        let grouped = Dictionary(grouping: memories, by: \.tier)
        return MemoryTier.allCases.sorted().compactMap { tier in
            guard let records = grouped[tier], !records.isEmpty else { return nil }
            return (tier, records)
        }
    }

    private var list: some View {
        List(selection: $selection) {
            ForEach(tiers, id: \.tier) { group in
                Section(group.tier.label) {
                    ForEach(group.records) { memory in
                        MemoryBeliefRow(memory: memory)
                            .tag(memory.id)
                            .contextMenu {
                                Button("Copy Text") {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(
                                        memory.text, forType: .string)
                                }
                                Divider()
                                Button("Contest") { onContest(memory) }
                                    .disabled(memory.status == .contested)
                                Button("Delete…", role: .destructive) { onDelete(memory) }
                            }
                    }
                }
            }
        }
        .onDeleteCommand {
            if let memory = memories.first(where: { $0.id == selection }) {
                onDelete(memory)
            }
        }
    }

    // MARK: - Empty states

    @ViewBuilder
    private var emptyState: some View {
        if filter.isEmpty {
            ContentUnavailableView {
                Label("Nothing Remembered Yet", systemImage: "brain.head.profile")
            } description: {
                Text(
                    """
                    Episodes are collected as we talk; sleep distills them into \
                    memories. What I come to believe about you appears here.
                    """)
            }
        } else {
            ContentUnavailableView.search(text: filter)
        }
    }

    // MARK: - Stats footer (ADR-0035 §9: the quiet tier/stat overview)

    private var statsFooter: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(stats.episodes) episodes · \(stats.memories) memories")
            if !tierBreakdown.isEmpty {
                Text(tierBreakdown)
            }
        }
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, MemoryWindowLayout.rhythm)
        .padding(.vertical, 8)
    }

    private var tierBreakdown: String {
        MemoryTier.allCases.sorted()
            .compactMap { tier in
                guard let count = stats.byTier[tier], count > 0 else { return nil }
                return "\(count) \(tier.label.lowercased())"
            }
            .joined(separator: " · ")
    }
}

// MARK: - Row

/// One memory: its provenance first, the first-person text, and one quiet
/// metadata line. Usage shows both counters because "retrieved ≠ useful" is
/// a core claim of the design (ADR-0035 §3).
struct MemoryBeliefRow: View {
    let memory: MemoryRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                ProvenanceBadge(provenance: memory.provenance)
                Text(memory.text)
                    .lineLimit(2)
                    .foregroundStyle(memory.status == .superseded ? .secondary : .primary)
            }
            HStack(spacing: 8) {
                MemoryStatusBadge(status: memory.status)
                Text(metadata)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }

    private var metadata: String {
        "\(memory.kind.label) · \(memory.bornAt.memoryDay)"
            + " · useful \(memory.usefulUseCount) · seen \(memory.seenCount)"
    }
}
