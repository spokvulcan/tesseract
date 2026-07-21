import Foundation

// MARK: - SkillEnvelope

/// The one home for how injected skill content is *framed* — the wrapper text
/// that surrounds a skill's body when it reaches the model, and the parser that
/// reads the one framing that is a persistence contract.
///
/// Three renderers, one parser:
///
/// - ``injection(name:location:body:)`` — the user-message injection
///   (`<skill name="…" location="…">…</skill>`) `ChatSession.executeSkill`
///   sends. This one is a **persistence contract**: transcripts store it and
///   the **Skill Invocation Row** re-parses it forever after, which is why it
///   is the only format with an inverse (``parse(_:)``).
/// - ``toolResult(name:location:body:)`` — the `use_skill` load result
///   (`# Skill: …\nLocation: …\n\n…`).
/// - ``linkedFile(name:path:content:)`` — the `use_skill` linked-file result
///   (`# … / …\n\n…`).
///
/// **The tool-result formats are parser-less by design.** They are model-facing
/// prose — nothing consumes them structurally, so there is nothing to keep an
/// inverse honest against. Only ``injection(name:location:body:)`` is persisted
/// and re-read, so only it earns a parser. The asymmetry is a decision, not an
/// oversight.
///
/// **The round-trip law.** For inputs in the round-trip domain,
/// `parse(injection(name:location:body:))` recovers `skillName == name` and an
/// `injectedBlock` byte-equal to the rendered block. The domain is every
/// realistic skill (see ``parse(_:)`` for the exact boundary and why inputs
/// outside it — a `"`/`>` in the name, a literal `</skill>` in the body — are
/// documented-lossy rather than escaped: escaping would either change the bytes
/// the model reads or break the persisted-transcript byte contract).
nonisolated enum SkillEnvelope {

    // MARK: - Renderers

    /// The user-message injection wrapper (a user message, never a system-prompt
    /// mutation, so the prefix cache's stable prefix is untouched). `location`
    /// is the skill file's path; the "References are relative to …" line is its
    /// parent directory, derived here so the two never drift apart.
    ///
    /// Byte-identical to the literal `ChatSession.executeSkill` shipped before
    /// the envelope — old transcripts must keep parsing identically. Arguments
    /// are appended by the caller, outside the block, so they are not part of
    /// this string.
    static func injection(name: String, location: String, body: String) -> String {
        let skillDir = URL(fileURLWithPath: location).deletingLastPathComponent().path
        return """
            <skill name="\(name)" location="\(location)">
            References are relative to \(skillDir).

            \(body)
            </skill>
            """
    }

    /// The `use_skill` load result — model-facing prose, parser-less. `location`
    /// is the skill file's path; the linked-files listing is appended by the
    /// caller, outside this string.
    static func toolResult(name: String, location: String, body: String) -> String {
        var output = "# Skill: \(name)\n"
        output += "Location: \(location)\n\n"
        output += body
        return output
    }

    /// The `use_skill` linked-file result — model-facing prose, parser-less.
    static func linkedFile(name: String, path: String, content: String) -> String {
        "# \(name) / \(path)\n\n\(content)"
    }

    // MARK: - Parser

    /// Parse a user-message content string into a ``SkillInvocationBlock``.
    /// Returns nil unless the content starts with a well-formed
    /// `<skill name="…" location="…">` opening tag and contains a closing
    /// `</skill>`. Requiring *both* attributes the injection path always writes
    /// keeps a user who literally types `<skill name=…>` from having their
    /// message re-rendered as an invocation row.
    ///
    /// ## Round-trip domain
    ///
    /// `parse(injection(name:location:body:))` recovers `skillName` and the full
    /// `injectedBlock` for every realistic skill. Three input shapes fall
    /// **outside** the domain and are documented-lossy on purpose — the
    /// `SkillEnvelopeTests` "round-trip boundary" cases pin exactly what each
    /// does:
    ///
    /// - **A `"` in the name.** The value is delimited by `"`, so a quote inside
    ///   it truncates `skillName` at the quote. Skill names are kebab-case
    ///   (`[a-z0-9-]`) and never contain quotes.
    /// - **A `>` in the name or location.** The opening tag ends at the first
    ///   `>`, so a `>` inside an attribute cuts the tag short — a `>` in the name
    ///   truncates it; a `>` in the location makes the whole parse fail (nil).
    ///   Neither occurs in a real skill name or macOS skill path.
    /// - **A literal `</skill>` in the body.** `injectedBlock` ends at the *first*
    ///   `</skill>`, so a closing tag inside the body truncates the block and the
    ///   remainder bleeds into `argumentText`. No data is lost (the model still
    ///   receives the whole message — this parser only drives the transcript
    ///   row), and it stays cosmetic.
    ///
    /// These are **not escaped.** Escaping the body would change the bytes the
    /// model reads; escaping the name/location would change the persisted-block
    /// bytes for any transcript that already carries a benign name — either
    /// breaks a contract the envelope exists to hold. A byte-identical renderer
    /// plus a documented, test-pinned domain is the honest trade.
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
