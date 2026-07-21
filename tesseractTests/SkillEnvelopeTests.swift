//
//  SkillEnvelopeTests.swift
//  tesseractTests
//
//  `SkillEnvelope` (#401) — the one home for how injected skill content is
//  framed. Three renderers, one parser. These tests hold two contracts:
//
//  1. **Byte stability.** The three renderers emit exactly the literals that
//     shipped before the envelope. The injection format is persisted in
//     transcripts and re-parsed forever after, so a byte change is a
//     compatibility break — these goldens are copied from the pre-#401 output.
//  2. **The round-trip law.** `parse(injection(x))` recovers `x` for every
//     realistic skill, and the handful of inputs outside that domain are
//     pinned as documented-lossy rather than left to drift silently.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SkillEnvelopeTests {

    // MARK: - Byte stability (transcript / persistence compat)

    @Test func injectionMatchesTheShippedLiteral() {
        let rendered = SkillEnvelope.injection(
            name: "proofread",
            location: "/skills/proofread/SKILL.md",
            body: "Fix objective errors only.")
        let golden =
            "<skill name=\"proofread\" location=\"/skills/proofread/SKILL.md\">\n"
            + "References are relative to /skills/proofread.\n"
            + "\n"
            + "Fix objective errors only.\n"
            + "</skill>"
        #expect(rendered == golden)
    }

    @Test func toolResultMatchesTheShippedLiteral() {
        let rendered = SkillEnvelope.toolResult(
            name: "proofread",
            location: "/skills/proofread/SKILL.md",
            body: "Fix objective errors only.")
        let golden =
            "# Skill: proofread\n"
            + "Location: /skills/proofread/SKILL.md\n"
            + "\n"
            + "Fix objective errors only."
        #expect(rendered == golden)
    }

    @Test func linkedFileMatchesTheShippedLiteral() {
        let rendered = SkillEnvelope.linkedFile(
            name: "proofread", path: "references/api.md", content: "API details.")
        #expect(rendered == "# proofread / references/api.md\n\nAPI details.")
    }

    // MARK: - Round-trip law: parse(injection(x)) == x

    /// Assert the block round-trips: the name and the full rendered block come
    /// back out of the parser, and appended arguments (pre-trimmed) survive.
    private func expectRoundTrip(
        name: String,
        location: String,
        body: String,
        arguments: String = "",
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let block = SkillEnvelope.injection(name: name, location: location, body: body)
        var content = block
        if !arguments.isEmpty { content += "\n\n\(arguments)" }

        let parsed = try #require(SkillEnvelope.parse(content), sourceLocation: sourceLocation)
        #expect(parsed.skillName == name, sourceLocation: sourceLocation)
        #expect(parsed.injectedBlock == block, sourceLocation: sourceLocation)
        #expect(parsed.argumentText == arguments, sourceLocation: sourceLocation)
    }

    @Test func plainBodyRoundTrips() throws {
        try expectRoundTrip(
            name: "proofread",
            location: "/skills/proofread/SKILL.md",
            body: "Fix objective errors only.")
    }

    @Test func multilineBodyAndArgumentsRoundTrip() throws {
        try expectRoundTrip(
            name: "translate",
            location: "/skills/translate/SKILL.md",
            body: "Translate content.\n\nPreserve tone.",
            arguments: "guten Tag\n\nDefault target language: Ukrainian")
    }

    @Test func angleBracketBodyRoundTrips() throws {
        // Bodies may contain angle brackets and even opening-tag-looking text —
        // as long as they hold no literal closing `</skill>`, the real closing
        // tag is still the first one found.
        try expectRoundTrip(
            name: "doc",
            location: "/x/SKILL.md",
            body: "Use <tag> and <skill name=\"x\"> and a < b carefully.")
    }

    @Test func quotesInLocationRoundTrip() throws {
        // The parser never recovers the location's *value*, only checks the
        // attribute is present — so a quote inside the path can't disturb the
        // name or the block.
        try expectRoundTrip(
            name: "proofread",
            location: "/skills/it's \"here\"/SKILL.md",
            body: "Body.")
    }

    @Test func whitespaceAroundBodyIsPreserved() throws {
        try expectRoundTrip(
            name: "doc",
            location: "/x/SKILL.md",
            body: "  leading and trailing spaces  ")
    }

    @Test func argumentWhitespaceIsTrimmedByContract() throws {
        // Arguments are trimmed on the way out (the invocation row shows clean
        // argument text), so surrounding whitespace does not survive the trip —
        // pinned so the contract is a decision, not a surprise.
        let block = SkillEnvelope.injection(
            name: "doc", location: "/x/SKILL.md", body: "Body.")
        let parsed = try #require(SkillEnvelope.parse(block + "\n\n  spaced arg  "))
        #expect(parsed.argumentText == "spaced arg")
    }

    // MARK: - Round-trip boundary (documented-lossy, pinned so it can't drift)

    @Test func nameWithQuoteTruncatesAtTheQuote() throws {
        // A `"` in the name closes the attribute early; the block still parses,
        // but the recovered name stops at the quote. Skill names are kebab-case
        // and never contain quotes — this is the documented boundary.
        let block = SkillEnvelope.injection(
            name: "pro\"ofread", location: "/x/SKILL.md", body: "Body.")
        let parsed = try #require(SkillEnvelope.parse(block))
        #expect(parsed.skillName == "pro")
    }

    @Test func nameWithAngleBracketFailsToParse() {
        // A `>` in the name ends the opening tag early, so the name attribute
        // can't be read — the message stays a plain user block.
        let block = SkillEnvelope.injection(
            name: "a>b", location: "/x/SKILL.md", body: "Body.")
        #expect(SkillEnvelope.parse(block) == nil)
    }

    @Test func locationWithAngleBracketFailsToParse() {
        // A `>` in the location ends the opening tag before the closing quote,
        // so the whole parse fails. macOS skill paths do not contain `>`.
        let block = SkillEnvelope.injection(
            name: "proofread", location: "/a>b/SKILL.md", body: "Body.")
        #expect(SkillEnvelope.parse(block) == nil)
    }

    @Test func bodyWithClosingTagTruncatesTheBlock() throws {
        // A literal `</skill>` inside the body is found before the real one, so
        // the block truncates there and the remainder bleeds into the arguments.
        // No data is lost (the model still receives the whole message; this
        // parser only drives the transcript row) — but it is not a round-trip.
        let block = SkillEnvelope.injection(
            name: "doc", location: "/x/SKILL.md", body: "before </skill> after")
        let parsed = try #require(SkillEnvelope.parse(block))
        #expect(parsed.skillName == "doc")
        #expect(parsed.injectedBlock != block)
        #expect(
            parsed.injectedBlock
                == "<skill name=\"doc\" location=\"/x/SKILL.md\">\n"
                + "References are relative to /x.\n\nbefore </skill>")
        #expect(parsed.argumentText == "after\n</skill>")
    }
}
