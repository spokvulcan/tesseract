//
//  SkillRegistryTests.swift
//  tesseractTests
//
//  First test file for the Skill Registry (PRD #174). Exercises the public
//  `discover` seam against real files written to a temp directory — no mocks —
//  pinning the frontmatter contract the Skill Pill row depends on: the
//  `composer-pill` membership key, the optional `label` override, and the
//  derived display label. Plus a bundled-content validation test asserting the
//  shipped `essentials` package parses into the six pill skills.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SkillRegistryTests {

    // MARK: - Fixtures

    /// Write a skill directory `<root>/<name>/SKILL.md` with the given text.
    private func writeSkill(_ name: String, _ text: String, in root: URL) throws {
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try text.write(
            to: dir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)
    }

    private func makeTempRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("skill-registry-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    // MARK: - composer-pill key

    @Test func composerPillKeyParsesTrue() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "proofread",
            """
            ---
            name: proofread
            description: Fix objective errors only.
            composer-pill: true
            ---
            Body.
            """, in: root)

        let skills = SkillRegistry.discover(locations: [root])
        #expect(skills.count == 1)
        #expect(skills.first?.composerPill == true)
    }

    @Test func absentComposerPillKeyDefaultsFalse() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "memory",
            """
            ---
            name: memory
            description: Memory management.
            ---
            Body.
            """, in: root)

        let skills = SkillRegistry.discover(locations: [root])
        #expect(skills.first?.composerPill == false)
    }

    @Test func nonTrueComposerPillValueReadsFalse() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "notes",
            """
            ---
            name: notes
            description: Notes.
            composer-pill: nope
            ---
            Body.
            """, in: root)

        #expect(SkillRegistry.discover(locations: [root]).first?.composerPill == false)
    }

    // MARK: - label override

    @Test func labelOverrideIsHonored() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "proofread-tweet",
            """
            ---
            name: proofread-tweet
            description: Polish a tweet.
            composer-pill: true
            label: Tweet
            ---
            Body.
            """, in: root)

        let skill = try #require(SkillRegistry.discover(locations: [root]).first)
        #expect(skill.label == "Tweet")
        #expect(skill.displayLabel == "Tweet")
    }

    @Test func absentLabelDerivesTitleCaseFromKebabName() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "proofread-tweet",
            """
            ---
            name: proofread-tweet
            description: Polish a tweet.
            ---
            Body.
            """, in: root)

        let skill = try #require(SkillRegistry.discover(locations: [root]).first)
        #expect(skill.label == nil)
        #expect(skill.displayLabel == "Proofread Tweet")
    }

    // MARK: - Existing contract (pinned while touching the parser)

    @Test func skillWithoutDescriptionIsSkipped() throws {
        let root = try makeTempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeSkill(
            "broken",
            """
            ---
            name: broken
            ---
            Body.
            """, in: root)

        #expect(SkillRegistry.discover(locations: [root]).isEmpty)
    }

    @Test func duplicateNamesKeepFirstDiscovered() throws {
        let rootA = try makeTempRoot()
        let rootB = try makeTempRoot()
        defer {
            try? FileManager.default.removeItem(at: rootA)
            try? FileManager.default.removeItem(at: rootB)
        }
        try writeSkill(
            "dup",
            """
            ---
            name: dup
            description: First wins.
            composer-pill: true
            ---
            A.
            """, in: rootA)
        try writeSkill(
            "dup",
            """
            ---
            name: dup
            description: Second loses.
            ---
            B.
            """, in: rootB)

        let skills = SkillRegistry.discover(locations: [rootA, rootB])
        #expect(skills.count == 1)
        #expect(skills.first?.description == "First wins.")
        #expect(skills.first?.composerPill == true)
    }

    // MARK: - Bundled essentials package validation

    /// The shipped `essentials` package must parse into exactly the six pill
    /// skills of PRD #174, each with a name, a non-empty description, the
    /// `composer-pill` membership key, and `disable-model-invocation` (they are
    /// pill-only — triggered from the row, never auto-loaded by the model).
    /// Guards against a frontmatter typo silently dropping a pill from the row
    /// or leaking one into the model-facing skills prompt.
    @Test func bundledEssentialsSkillsParseWithPillKey() throws {
        let bundled = try #require(
            Bundle.main.url(forResource: "AgentPackages", withExtension: nil),
            "AgentPackages bundle resource missing")
        let skillsDir =
            bundled
            .appendingPathComponent("essentials", isDirectory: true)
            .appendingPathComponent("skills", isDirectory: true)

        let skills = SkillRegistry.discover(locations: [skillsDir])
        let byName = Dictionary(uniqueKeysWithValues: skills.map { ($0.name, $0) })

        let expected = [
            "proofread", "proofread-tweet", "reply", "summarize", "explain", "translate",
        ]
        #expect(Set(byName.keys) == Set(expected))
        for name in expected {
            let skill = try #require(byName[name], "missing bundled skill: \(name)")
            #expect(!skill.description.isEmpty)
            #expect(skill.composerPill == true)
            #expect(skill.disableModelInvocation == true)
        }
    }
}
