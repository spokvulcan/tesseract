//
//  MemoryBeliefDetailView.swift
//  tesseract
//
//  The Memory window's detail pane: one memory in full — its provenance
//  banner, the belief text, the lifecycle facts, and the source episodes it
//  was derived from. The sources section is the "why do you believe that"
//  answer (ADR-0035 §9): verbatim episodes, with dates, always reachable.
//
//  There is deliberately no editing here. A hand-edited belief has no
//  provenance and no lifecycle; the owner's levers are contest and delete.
//

import SwiftUI

struct MemoryBeliefDetailView: View {
    @Environment(MemoryEngine.self) private var engine

    let memory: MemoryRecord
    /// The memory that replaced this one, when there is one — resolved by
    /// the window from the already-loaded list.
    let supersededBy: MemoryRecord?
    let onContest: () -> Void
    let onDelete: () -> Void

    @State private var episodes: [Episode] = []
    @State private var isLoadingEpisodes = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MemoryWindowLayout.rhythm) {
                provenanceBanner
                statusBanner
                Text(memory.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                factsSection
                sourcesSection
            }
            .frame(maxWidth: MemoryWindowLayout.columnMaxWidth, alignment: .leading)
            .padding(20)
            .frame(maxWidth: .infinity)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { actionBar }
        .task(id: memory.id) {
            isLoadingEpisodes = true
            episodes = await engine.episodes(for: memory)
            isLoadingEpisodes = false
        }
    }

    // MARK: - Banners

    /// The safety field, unmissable: who this belief came from.
    private var provenanceBanner: some View {
        banner(
            memory.provenance.bannerText, symbol: memory.provenance.symbol,
            color: memory.provenance.color)
    }

    /// The tinted banner shape provenance and contest share.
    private func banner(_ text: String, symbol: String, color: Color) -> some View {
        Label(text, systemImage: symbol)
            .fontWeight(.medium)
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(MemoryWindowLayout.rhythm)
            .background(color.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var statusBanner: some View {
        switch memory.status {
        case .live:
            EmptyView()
        case .contested:
            banner(
                "Contested — I'll reconcile this against its sources in my next sleep.",
                symbol: "exclamationmark.triangle", color: .red)
        case .superseded:
            VStack(alignment: .leading, spacing: 4) {
                Label(
                    "Superseded — a newer memory replaced this one.",
                    systemImage: "arrow.right.circle"
                )
                .fontWeight(.medium)
                if let supersededBy {
                    Text("Now: “\(supersededBy.text)”")
                }
            }
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(MemoryWindowLayout.rhythm)
            .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - Facts

    private var factsSection: some View {
        VStack(alignment: .leading, spacing: MemoryWindowLayout.rhythm) {
            Text("Lifecycle")
                .fontWeight(.semibold)
            Grid(
                alignment: .leadingFirstTextBaseline,
                horizontalSpacing: 16, verticalSpacing: 6
            ) {
                factRow("Kind", memory.kind.label)
                factRow("Tier", "\(memory.tier.label) — \(memory.tier.meaning)")
                factRow("Born", memory.bornAt.memoryDay)
                factRow("Need probability", needNow)
                factRow("Stability", stabilityText)
                factRow("Storage strength", String(format: "%.2f", memory.storageStrength))
                factRow(
                    "Useful uses",
                    "\(memory.usefulUseCount) · last \(memory.lastUsefulUseAt.memoryDay)")
                factRow("Seen", "\(memory.seenCount) · last \(memory.lastSeenAt.memoryDay)")
                factRow("Confirmations", "\(memory.confirmations)")
            }
            Text(
                """
                Seen counts every retrieval into context; useful counts only the \
                uses that actually helped. Only a useful use strengthens a memory.
                """
            )
            .foregroundStyle(.secondary)
        }
    }

    private func factRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(.secondary)
                .gridColumnAlignment(.trailing)
            Text(value)
        }
    }

    private var needNow: String {
        let need = MemoryLifecycle.needProbability(of: memory, now: Date())
        return need.formatted(.percent.precision(.fractionLength(0)))
    }

    private var stabilityText: String {
        String(format: "%.1f days", memory.stability)
    }

    // MARK: - Sources (the provenance drill-down)

    private var sourcesSection: some View {
        VStack(alignment: .leading, spacing: MemoryWindowLayout.rhythm) {
            Text(sourcesTitle)
                .fontWeight(.semibold)
            if episodes.isEmpty && !isLoadingEpisodes {
                Text(
                    memory.sourceEpisodeIDs.isEmpty
                        ? "No source episodes were recorded for this memory."
                        : "Its source episodes could not be loaded."
                )
                .foregroundStyle(.secondary)
            }
            ForEach(episodes) { episode in
                MemoryEpisodeCard(episode: episode)
            }
        }
    }

    private var sourcesTitle: String {
        episodes.isEmpty
            ? "Why I believe this"
            : "Why I believe this — \(episodes.count) episode\(episodes.count == 1 ? "" : "s")"
    }

    // MARK: - The owner's hand

    private var actionBar: some View {
        HStack {
            Spacer()
            Button("Contest", action: onContest)
                .disabled(memory.status == .contested)
                .help(
                    "Mark this as wrong. It gets reconciled against its sources in the next sleep.")
            Button("Delete…", role: .destructive, action: onDelete)
                .help("Remove this memory permanently — the only true deletion in the system.")
        }
        .padding(.horizontal, MemoryWindowLayout.rhythm)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }
}

// MARK: - Episode card

/// One source episode, verbatim, with where and when it happened. The
/// immutable layer showing through — this text is never rewritten by anyone.
struct MemoryEpisodeCard: View {
    let episode: Episode

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Label(episode.source.label, systemImage: episode.source.symbol)
                Text(episode.occurredAt.memoryMoment)
            }
            .foregroundStyle(.secondary)
            Text(episode.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(MemoryWindowLayout.rhythm)
        .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}
