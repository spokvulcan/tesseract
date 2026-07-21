import Foundation

// MARK: - SkillInvocationBlock

/// The parsed shape of a user message that carries an injected skill block —
/// the `<skill name="…" location="…">…</skill>` wrapper `executeSkill` builds,
/// optionally followed by the user's argument text.
///
/// The Chat Transcript projection uses this to render a **Skill Invocation
/// Row** (PRD #174) instead of the raw injected instruction text, for every
/// invocation surface (Skill Pill or slash command).
///
/// This is the parsed *value*; the render/parse framing lives in
/// ``SkillEnvelope`` (`SkillEnvelope.parse(_:)` produces this type). Keeping the
/// producer and its inverse in one module is the point of #401 — a contract
/// that lives only as two matching literals drifts green.
nonisolated struct SkillInvocationBlock: Equatable, Sendable {
    /// The skill's name from the opening tag's `name` attribute.
    let skillName: String
    /// The user's argument text after the closing tag (trimmed; empty when the
    /// skill fired bare).
    let argumentText: String
    /// The full injected `<skill>…</skill>` block, for the expanded
    /// transparency view.
    let injectedBlock: String

    /// The human-readable title: the kebab-case skill name title-cased
    /// ("proofread-tweet" → "Proofread Tweet").
    var displayLabel: String { skillName.kebabTitleCased }
}

// MARK: - Kebab → Title Case

extension String {
    /// "proofread-tweet" → "Proofread Tweet". Shared by the Skill Pill label
    /// derivation and the Skill Invocation Row title.
    nonisolated var kebabTitleCased: String {
        split(separator: "-")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}
