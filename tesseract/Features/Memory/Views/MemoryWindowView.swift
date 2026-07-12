//
//  MemoryWindowView.swift
//  tesseract
//
//  The Memory window (ADR-0035 §9): the owner's window into what the
//  assistant believes about him, and why. Browse the derived memories with
//  their provenance, drill into the source episodes, contest what is wrong,
//  delete what must go — the one true-deletion path in the system — and
//  read the consolidation journal.
//
//  An on-demand singleton `Window` scene (Markdown Gallery precedent),
//  reached from the system Window menu. Singleton scene @State survives a
//  close/reopen, so the data reloads in `onAppear` — which does fire again
//  on reopen (measured for the gallery).
//

import SwiftUI

private enum MemoryWindowSection: String, CaseIterable {
    case memories = "Memories"
    case journal = "Journal"
}

struct MemoryWindowView: View {
    @Environment(MemoryEngine.self) private var engine

    @State private var section: MemoryWindowSection = .memories
    @State private var memories: [MemoryRecord] = []
    @State private var journalEntries: [JournalEntry] = []
    @State private var selection: UUID?
    @State private var filter = ""

    @State private var pendingDelete: MemoryRecord?
    @State private var isDeleteConfirmationPresented = false

    var body: some View {
        Group {
            switch section {
            case .memories:
                memoriesSplit
            case .journal:
                MemoryJournalView(entries: filteredJournal, filter: filter)
            }
        }
        .toolbar {
            ToolbarItem {
                Picker("Section", selection: $section) {
                    ForEach(MemoryWindowSection.allCases, id: \.self) { choice in
                        Text(choice.rawValue).tag(choice)
                    }
                }
                .pickerStyle(.segmented)
            }
            ToolbarItem {
                TextField("Filter", text: $filter)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
            }
        }
        .navigationTitle("Memory")
        .confirmationDialog(
            "Delete this memory?",
            isPresented: $isDeleteConfirmationPresented,
            titleVisibility: .visible,
            presenting: pendingDelete
        ) { memory in
            Button("Delete Forever", role: .destructive) { delete(memory) }
            Button("Cancel", role: .cancel) {}
        } message: { memory in
            Text(
                """
                “\(memory.text)”

                This is the only true deletion in the system: the memory and its \
                lifecycle are removed for good. Its source episodes remain.
                """)
        }
        .onAppear { reload() }
    }

    // MARK: - Memories split

    private var memoriesSplit: some View {
        HSplitView {
            MemoryBeliefListView(
                memories: filteredMemories,
                filter: filter,
                stats: engine.stats,
                selection: $selection,
                onContest: { contest($0) },
                onDelete: { requestDelete($0) }
            )
            .frame(minWidth: 300, idealWidth: 360, maxWidth: 520)
            detail
                .frame(minWidth: 420, maxWidth: .infinity)
        }
    }

    @ViewBuilder
    private var detail: some View {
        if let memory = selectedMemory {
            MemoryBeliefDetailView(
                memory: memory,
                supersededBy: memory.supersededBy
                    .flatMap { id in memories.first { $0.id == id } },
                onContest: { contest(memory) },
                onDelete: { requestDelete(memory) }
            )
        } else {
            ContentUnavailableView {
                Label("No Memory Selected", systemImage: "brain.head.profile")
            } description: {
                Text("Select a memory to see what it rests on.")
            }
        }
    }

    private var selectedMemory: MemoryRecord? {
        memories.first { $0.id == selection }
    }

    // MARK: - Filtering

    private var filteredMemories: [MemoryRecord] {
        let needle = filter.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return memories }
        return memories.filter {
            $0.text.localizedCaseInsensitiveContains(needle)
                || $0.kind.label.localizedCaseInsensitiveContains(needle)
        }
    }

    private var filteredJournal: [JournalEntry] {
        let needle = filter.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return journalEntries }
        return journalEntries.filter { entry in
            entry.detail.localizedCaseInsensitiveContains(needle)
                || (entry.before?.localizedCaseInsensitiveContains(needle) ?? false)
                || (entry.after?.localizedCaseInsensitiveContains(needle) ?? false)
        }
    }

    // MARK: - Data

    private func reload() {
        Task {
            memories = await engine.allMemories()
            journalEntries = await engine.journal(limit: 500)
            await engine.refreshStats()
            if selection == nil || !memories.contains(where: { $0.id == selection }) {
                // Auto-select the row the list will actually show first, so an
                // opened window always answers with content: sections run in
                // tier-ladder order, and rows keep the store's order within a
                // tier.
                let firstTier = filteredMemories.map(\.tier).min()
                selection = filteredMemories.first { $0.tier == firstTier }?.id
            }
        }
    }

    private func contest(_ memory: MemoryRecord) {
        Task {
            await engine.contest(memory)
            reload()
        }
    }

    private func requestDelete(_ memory: MemoryRecord) {
        pendingDelete = memory
        isDeleteConfirmationPresented = true
    }

    private func delete(_ memory: MemoryRecord) {
        Task {
            await engine.delete(memory)
            if selection == memory.id { selection = nil }
            pendingDelete = nil
            reload()
        }
    }
}
