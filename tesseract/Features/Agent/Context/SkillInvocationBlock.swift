import Foundation

// MARK: - SkillInvocationBlock

/// The parsed shape of a user message that carries an injected skill block —
/// the `<skill name="…" location="…">…</skill>` wrapper `executeSkill` builds,
/// optionally followed by the user's argument text.
///
/// The Chat Transcript projection uses this to render a **Skill Invocation
/// Row** (PRD #174) instead of the raw injected instruction text, for every
/// invocation surface (Skill Pill or slash command).
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

    /// Parse a user-message content string. Returns nil unless the content
    /// starts with a well-formed `<skill name="…" location="…">` opening tag
    /// and contains a closing `</skill>`. Requiring *both* attributes the
    /// injection path always writes keeps a user who literally types
    /// `<skill name=…>` from having their message re-rendered as an
    /// invocation row (best-effort — the expanded view shows the full text
    /// either way, so an exact-format collision stays transparent).
    static func parse(_ content: String) -> SkillInvocationBlock? {
        // Raw prefix gate before any allocation: this runs for every user
        // message on every projection pass, and the injection path always
        // writes the tag at byte zero — so a plain message bails without
        // paying for a full-content trim.
        guard content.hasPrefix("<skill ") else { return nil }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let openingEnd = trimmed.firstIndex(of: ">") else { return nil }
        let openingTag = trimmed[..<openingEnd]

        guard let name = attributeValue("name", in: openingTag), !name.isEmpty,
            attributeValue("location", in: openingTag) != nil
        else {
            return nil
        }

        guard let closingRange = trimmed.range(of: "</skill>") else { return nil }
        let injectedBlock = String(trimmed[..<closingRange.upperBound])
        let argumentText = String(trimmed[closingRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return SkillInvocationBlock(
            skillName: name,
            argumentText: argumentText,
            injectedBlock: injectedBlock
        )
    }

    /// Extract `key="value"` from an opening tag.
    private static func attributeValue(_ key: String, in tag: Substring) -> String? {
        guard let keyRange = tag.range(of: "\(key)=\"") else { return nil }
        let valueStart = keyRange.upperBound
        guard let valueEnd = tag[valueStart...].firstIndex(of: "\"") else { return nil }
        return String(tag[valueStart..<valueEnd])
    }
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
