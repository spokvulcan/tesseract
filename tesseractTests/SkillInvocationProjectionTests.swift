//
//  SkillInvocationProjectionTests.swift
//  tesseractTests
//
//  The **Skill Invocation Row** (PRD #174): pure tests for the injection-format
//  parser — the gate that decides whether a user message renders as a compact
//  invocation row or a plain user block. The old transcript-projection
//  assertions are gone with the projection layer (ADR-0024): the row view now
//  consumes the parsed block directly and owns its expansion as view state.
//
//  Since #401 the fixtures round-trip through the *producer* —
//  `SkillEnvelope.injection` — instead of a hand-duplicated literal, so a change
//  to the framing can no longer leave these green while real rows fall back to
//  raw text. The producer/parser inverse itself is pinned in `SkillEnvelopeTests`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SkillInvocationProjectionTests {

    // MARK: - Fixtures

    /// Build a fired-skill message the way `executeSkill` does: the producer's
    /// injection block, plus the user's arguments appended outside it.
    private func skillContent(
        name: String = "proofread",
        body: String = "Fix objective errors only.",
        arguments: String = ""
    ) -> String {
        var content = SkillEnvelope.injection(
            name: name, location: "/skills/\(name)/SKILL.md", body: body)
        if !arguments.isEmpty {
            content += "\n\n\(arguments)"
        }
        return content
    }

    // MARK: - SkillInvocationBlock parsing

    @Test func parsesNameBlockAndArguments() throws {
        let content = skillContent(name: "proofread-tweet", arguments: "my draft tweet")
        let block = try #require(SkillEnvelope.parse(content))
        #expect(block.skillName == "proofread-tweet")
        #expect(block.argumentText == "my draft tweet")
        #expect(block.injectedBlock.hasPrefix("<skill name=\"proofread-tweet\""))
        #expect(block.injectedBlock.hasSuffix("</skill>"))
        #expect(block.displayLabel == "Proofread Tweet")
    }

    @Test func bareInvocationParsesWithEmptyArguments() throws {
        let block = try #require(SkillEnvelope.parse(skillContent()))
        #expect(block.argumentText.isEmpty)
    }

    @Test func multilineArgumentsSurviveIntact() throws {
        let arguments = "line one\nline two\n\nDefault target language: Ukrainian"
        let block = try #require(SkillEnvelope.parse(skillContent(arguments: arguments)))
        #expect(block.argumentText == arguments)
    }

    @Test func plainTextIsNotASkillInvocation() {
        #expect(SkillEnvelope.parse("hello there") == nil)
        #expect(SkillEnvelope.parse("<skillet name=\"x\">no</skillet>") == nil)
        #expect(SkillEnvelope.parse("") == nil)
    }

    @Test func handTypedSkillTagWithoutLocationStaysAUserMessage() {
        // Only the injection path writes both `name` and `location`; a user
        // literally typing a bare `<skill name=…>` tag keeps their message.
        let handTyped = "<skill name=\"proofread\">how does this tag work?</skill>"
        #expect(SkillEnvelope.parse(handTyped) == nil)
    }

    @Test func unterminatedBlockIsNotASkillInvocation() {
        #expect(SkillEnvelope.parse("<skill name=\"proofread\">no closing tag") == nil)
    }
}
