//
//  SkillClusterView.swift
//  tesseract
//
//  The **Skill Cluster** (ADR-0030): the third custom glass surface — a
//  floating ✦ bubble above the composer's trailing corner that morphs open
//  into the fanned **Skill Pill**s. A dumb rendering of the
//  `SkillClusterController` phase; pill firing reuses the Chat Session's
//  draft ride-along contract. Hosts its *own* GlassEffectContainer (the
//  Landmarks badges pattern) so the bubble and pills morph together via
//  glassEffectID without liquid-fusing into the composer's glass edge.
//

import SwiftUI

// MARK: - Wrap computation

/// Pure geometry for the leftward fan: greedy fill of the bottom row (row 0,
/// at bubble height) from the bubble outward, wrapping upward when out of
/// width. Earlier indices are more used and sit further right (**Skill Usage
/// Ranking**).
nonisolated enum SkillClusterWrap {
    static func rows(
        itemWidths: [CGFloat], available: CGFloat, spacing: CGFloat
    ) -> [[Int]] {
        var rows: [[Int]] = []
        var currentRow: [Int] = []
        var usedWidth: CGFloat = 0
        for (index, width) in itemWidths.enumerated() {
            let widthIfAppended = currentRow.isEmpty ? width : usedWidth + spacing + width
            if !currentRow.isEmpty, widthIfAppended > available {
                rows.append(currentRow)
                currentRow = [index]
                usedWidth = width
            } else {
                currentRow.append(index)
                usedWidth = widthIfAppended
            }
        }
        if !currentRow.isEmpty { rows.append(currentRow) }
        return rows
    }
}

// MARK: - Trailing wrap layout

/// Lays subviews out right-to-left from the bottom-trailing corner, wrapping
/// upward — the fan's geometry. Subview order is the ranking order: subview 0
/// lands nearest the bubble.
nonisolated private struct TrailingWrapLayout: Layout {
    var spacing: CGFloat
    var rowSpacing: CGFloat

    /// Subview ideal sizes, measured once per layout pass — row assignment
    /// still recomputes per call because the available width differs between
    /// sizing (proposal) and placement (bounds).
    struct Cache {
        var sizes: [CGSize]
    }

    func makeCache(subviews: Subviews) -> Cache {
        Cache(sizes: subviews.map { $0.sizeThatFits(.unspecified) })
    }

    func updateCache(_ cache: inout Cache, subviews: Subviews) {
        cache.sizes = subviews.map { $0.sizeThatFits(.unspecified) }
    }

    func sizeThatFits(
        proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache
    ) -> CGSize {
        let sizes = cache.sizes
        let rows = SkillClusterWrap.rows(
            itemWidths: sizes.map(\.width),
            available: proposal.width ?? .infinity,
            spacing: spacing)
        guard !rows.isEmpty else { return .zero }
        let width =
            rows
            .map { row in
                row.reduce(0) { $0 + sizes[$1].width } + CGFloat(row.count - 1) * spacing
            }
            .max() ?? 0
        let height =
            rows
            .map { row in row.map { sizes[$0].height }.max() ?? 0 }
            .reduce(0, +) + CGFloat(rows.count - 1) * rowSpacing
        return CGSize(width: width, height: height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache
    ) {
        let sizes = cache.sizes
        let rows = SkillClusterWrap.rows(
            itemWidths: sizes.map(\.width), available: bounds.width, spacing: spacing)
        var rowBottom = bounds.maxY
        for row in rows {
            var trailingEdge = bounds.maxX
            for index in row {
                subviews[index].place(
                    at: CGPoint(x: trailingEdge, y: rowBottom),
                    anchor: .bottomTrailing,
                    proposal: ProposedViewSize(sizes[index]))
                trailingEdge -= sizes[index].width + spacing
            }
            let rowHeight = row.map { sizes[$0].height }.max() ?? 0
            rowBottom -= rowHeight + rowSpacing
        }
    }
}

// MARK: - SkillClusterView

struct SkillClusterView: View {
    @Environment(ChatSession.self) private var session
    @Environment(SkillPillController.self) private var skillPills
    @Environment(ComposerDraftController.self) private var composerDraft
    @Environment(SkillClusterController.self) private var cluster

    /// One namespace for the whole cluster — the bubble and every pill carry
    /// a glassEffectID in it, so open/close morphs instead of fading.
    @Namespace private var glassNamespace

    private static let bubbleSize: CGFloat = 38

    var body: some View {
        // The cluster's own sampling context (spacing per the Landmarks
        // badges): bubble and pills blend with each other during the morph,
        // never with the composer's glass below.
        GlassEffectContainer(spacing: 16) {
            HStack(alignment: .bottom, spacing: 8) {
                if cluster.isOpen {
                    TrailingWrapLayout(spacing: 6, rowSpacing: 6) {
                        ForEach(skillPills.pills) { pill in
                            pillCapsule(pill)
                        }
                    }
                }

                bubble
            }
        }
        .onHover { hovering in
            if hovering {
                cluster.pointerEntered()
            } else {
                cluster.pointerExited()
            }
        }
        .animation(.smooth(duration: 0.3), value: cluster.isOpen)
    }

    /// The collapsed ✦ bubble. Dimmed and inert while a run is generating
    /// (the controller is suppressed then, so hover/click are already no-ops —
    /// the opacity is the visible half of that state).
    private var bubble: some View {
        Button {
            cluster.buttonClicked()
        } label: {
            Image(systemName: "sparkles")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: Self.bubbleSize, height: Self.bubbleSize)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .glassEffect(.regular.interactive(), in: Circle())
        .glassEffectID("bubble", in: glassNamespace)
        .opacity(session.isGenerating ? 0.4 : 1)
        .pointerStyle(session.isGenerating ? nil : .link)
        .help("Skills")
    }

    /// One fanned Skill Pill — same firing contract as the retired composer
    /// strip: the draft rides along, and is restored whole if the fire failed.
    private func pillCapsule(_ pill: SkillPill) -> some View {
        Button {
            fire(pill)
        } label: {
            Text(pill.label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glassEffect(.regular.interactive(), in: Capsule())
        .glassEffectID(pill.name, in: glassNamespace)
        .pointerStyle(.link)
        .help(pill.description)
    }

    private func fire(_ pill: SkillPill) {
        let text = composerDraft.text
        let images = composerDraft.drainImages()
        composerDraft.text = ""
        if !session.fireSkillPill(pill, draftText: text, images: images) {
            composerDraft.restore(text: text, images: images)
        }
        cluster.pillFired()
    }
}
