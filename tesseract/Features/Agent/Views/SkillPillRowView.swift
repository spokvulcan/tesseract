//
//  SkillPillRowView.swift
//  tesseract
//
//  The **Skill Pill** strip (PRD #174, inline per PRD #183): capsules in the
//  composer's action row, one per pill skill, most-used leftmost. A dumb
//  rendering of the `SkillPillController` leaf — tapping fires the skill
//  instantly through the Chat Session with the composer's text and pending
//  images riding along. Sits *on* the composer's glass surface, so the pills
//  themselves are plain capsules — the composer stays one of exactly two
//  custom glass surfaces (HIG).
//

import SwiftUI

struct SkillPillRowView: View {
    @Environment(ChatSession.self) private var session
    @Environment(SkillPillController.self) private var skillPills
    @Environment(ComposerDraftController.self) private var composerDraft

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(skillPills.pills) { pill in
                    pillButton(pill)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// One capsule, quiet chrome matching the action row's buttons. Disabled —
    /// not hidden — while a run is active, so the layout never jumps.
    private func pillButton(_ pill: SkillPill) -> some View {
        Button {
            fire(pill)
        } label: {
            Text(pill.label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(.quinary, in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(session.isGenerating)
        .help(pill.description)
    }

    /// Drain the composer draft into the skill invocation; restore it whole if
    /// the fire failed (missing skill file, run already active).
    private func fire(_ pill: SkillPill) {
        let text = composerDraft.text
        let images = composerDraft.drainImages()
        composerDraft.text = ""
        if !session.fireSkillPill(pill, draftText: text, images: images) {
            composerDraft.restore(text: text, images: images)
        }
    }
}
