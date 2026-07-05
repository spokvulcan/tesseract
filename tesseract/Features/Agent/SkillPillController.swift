//
//  SkillPillController.swift
//  tesseract
//
//  The **Skill Pill** module (PRD #174): a publisher-agnostic leaf following
//  the Agent-coordinator-leaves pattern. It owns the pill list derivation
//  (membership × **Skill Usage Ranking** × curated default order), usage
//  recording, and skill-invocation argument assembly (including the translate
//  pill's default-target wiring). The SwiftUI pill row is a dumb rendering of
//  this module's state; skill *execution* stays on the coordinator spine.
//

import Foundation
import Observation

// MARK: - SkillPill

/// One pill in the Skill Pill row — presentation data only.
nonisolated struct SkillPill: Identifiable, Equatable, Sendable {
    var id: String { name }
    let name: String
    let label: String
    let description: String
    let filePath: String
}

// MARK: - SkillPillController

@Observable @MainActor
final class SkillPillController {

    // MARK: - Observable State

    /// The current pill row, leftmost first. Recomputed only by
    /// ``refreshPills()`` — at conversation start — and held stable within a
    /// conversation, so pills never shift under the cursor mid-session.
    private(set) var pills: [SkillPill] = []

    // MARK: - Dependencies

    /// Skill discovery, injected so tests feed fixture metadata and the app
    /// wires the sandbox + package-cache scan.
    private let discoverSkills: @MainActor () -> [SkillMetadata]
    private let settings: SettingsManager?

    // MARK: - Curated default order

    /// Zero-count skills follow this order (PRD #174); skills outside it sort
    /// after, alphabetically. Usage always outranks curation.
    static let curatedOrder = [
        "proofread", "proofread-tweet", "translate", "reply", "summarize", "explain",
    ]

    // MARK: - Init

    init(
        discoverSkills: @escaping @MainActor () -> [SkillMetadata],
        settings: SettingsManager? = nil
    ) {
        self.discoverSkills = discoverSkills
        self.settings = settings
        refreshPills()
    }

    // MARK: - Row visibility

    /// Whether the row should render: the "Show skill pills" Setting is on and
    /// at least one skill declares pill membership.
    var isRowVisible: Bool {
        (settings?.showSkillPills ?? true) && !pills.isEmpty
    }

    // MARK: - Pill list derivation

    /// Recompute the pill row: `composer-pill` skills, ordered by the Skill
    /// Usage Ranking (usage count descending; ties fall back to the curated
    /// order, then name). Called at conversation start — never mid-conversation.
    func refreshPills() {
        let candidates = discoverSkills().filter(\.composerPill)
        pills =
            candidates
            .map { skill in
                (
                    pill: SkillPill(
                        name: skill.name,
                        label: skill.displayLabel,
                        description: skill.description,
                        filePath: skill.filePath
                    ),
                    count: settings?.skillUsageCount(skillName: skill.name) ?? 0,
                    curatedIndex: Self.curatedOrder.firstIndex(of: skill.name) ?? Int.max
                )
            }
            .sorted { lhs, rhs in
                if lhs.count != rhs.count { return lhs.count > rhs.count }
                if lhs.curatedIndex != rhs.curatedIndex {
                    return lhs.curatedIndex < rhs.curatedIndex
                }
                return lhs.pill.name < rhs.pill.name
            }
            .map(\.pill)
    }

    // MARK: - Usage recording

    /// Record one *user-initiated* invocation (pill tap or slash command).
    /// Model-initiated `use_skill` calls never reach this module, so they are
    /// structurally excluded from the ranking. The persisted count feeds the
    /// next ``refreshPills()``; the current row stays put.
    func recordUserInvocation(skillName: String) {
        settings?.incrementSkillUsage(skillName: skillName)
    }

    // MARK: - Invocation assembly

    /// Assemble the argument text appended after the injected skill block:
    /// the user's composer text, plus — for the built-in `translate` pill —
    /// the configured default target language. The wiring is this one named
    /// special case, not a templating engine; the skill body's direction rule
    /// lets an explicit language in the user's text win.
    func assembleArguments(skillName: String, userText: String) -> String {
        var arguments = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        if skillName == "translate" {
            let target = (settings?.translateTargetLanguage ?? "")
                .trimmingCharacters(in: .whitespaces)
            if !target.isEmpty {
                if !arguments.isEmpty { arguments += "\n\n" }
                arguments += "Default target language: \(target)"
            }
        }
        return arguments
    }
}
