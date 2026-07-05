//
//  SkillPillRowView.swift
//  tesseract
//
//  The **Skill Pill** row (PRD #174): liquid-glass capsules floating above the
//  composer, one per pill skill, most-used leftmost. A dumb rendering of the
//  `SkillPillController` leaf — tapping fires the skill instantly through the
//  coordinator with the composer's text and pending images riding along.
//  Control layer only (glass never in the content layer, per HIG).
//

import SwiftUI

struct SkillPillRowView: View {
    @Binding var inputText: String

    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(coordinator.skillPills.pills) { pill in
                    pillButton(pill)
                }
            }
            // Breathing room so the capsules' shadows/edges never clip against
            // the scroll container.
            .padding(.horizontal, 2)
            .padding(.vertical, 2)
        }
        .scrollClipDisabled()
    }

    /// One capsule: the system glass button style gives the Liquid Glass
    /// treatment (press bounce, vibrancy, automatic dimming when disabled)
    /// without hand-built chrome. Disabled — not hidden — while a run is
    /// active, so the layout never jumps.
    private func pillButton(_ pill: SkillPill) -> some View {
        Button {
            let composerText = inputText
            inputText = ""
            coordinator.fireSkillPill(pill, composerText: composerText)
        } label: {
            Text(pill.label)
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 4)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.capsule)
        .controlSize(.small)
        .disabled(coordinator.isGenerating)
        .help(pill.description)
    }
}
