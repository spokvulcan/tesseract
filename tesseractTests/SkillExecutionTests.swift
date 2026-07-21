//
//  SkillExecutionTests.swift
//  tesseractTests
//
//  The **Skill Execution** leaf (ADR-0045 continuation, #408): decision-row
//  tests over the leaf in isolation — the injection render through the Skill
//  Envelope (#401), argument assembly from the drained composer draft, images
//  riding the fire, the load-failure nil, and usage recording. No `Agent`, no
//  arbiter, no store — the leaf never touches them. The session-level fire
//  (guarded, sent, recorded end to end) stays pinned by `AgentSkillExecutionTests`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SkillExecutionTests {

    // MARK: - Fixtures

    private func image(_ name: String) -> ImageAttachment {
        ImageAttachment(
            data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png", filename: name)
    }

    /// Write a real skill file the default loader can read.
    private func writeSkillFile(name: String, body: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("skill-execution-tests-\(UUID().uuidString)/\(name)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("SKILL.md")
        try """
        ---
        name: \(name)
        description: \(name) test skill
        ---
        \(body)
        """.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Envelope render

    @Test func renderWrapsTheBodyInTheSkillEnvelopeInjection() throws {
        let url = try writeSkillFile(name: "proofread", body: "Fix errors only.")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let leaf = SkillExecution()
        let injection = try #require(
            leaf.render(skillName: "proofread", filePath: url.path, userText: "", images: []))

        // The rendered message is a valid invocation block the Skill Envelope
        // parser recovers — body inside, name preserved, no stray arguments.
        let block = try #require(SkillEnvelope.parse(injection.message))
        #expect(block.skillName == "proofread")
        #expect(block.injectedBlock.contains("Fix errors only."))
        #expect(block.argumentText.isEmpty)
        #expect(
            injection.message
                == SkillEnvelope.injection(
                    name: "proofread", location: url.path, body: "Fix errors only."))
    }

    @Test func renderAppendsAssembledArgumentsOutsideTheBlock() throws {
        let url = try writeSkillFile(name: "explain", body: "Explain content.")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let leaf = SkillExecution()
        let injection = try #require(
            leaf.render(
                skillName: "explain", filePath: url.path, userText: "this error", images: []))

        let block = try #require(SkillEnvelope.parse(injection.message))
        #expect(block.argumentText == "this error")
    }

    // MARK: - Argument assembly (the drained-draft consumption)

    @Test func renderAssemblesTheUserTextThroughTheInjectedClosure() {
        // A leaf wired to a canned body and an assembler that transforms the
        // drained draft — the composer-text-becomes-arguments consumption.
        let leaf = SkillExecution(
            assembleArguments: { name, text in "assembled(\(name)):\(text.uppercased())" },
            loadSkillBody: { _ in "BODY" }
        )
        let injection = leaf.render(
            skillName: "translate", filePath: "/skills/translate/SKILL.md",
            userText: "guten Tag", images: [])

        #expect(injection?.message.hasSuffix("\n\nassembled(translate):GUTEN TAG") == true)
    }

    @Test func renderOmitsTheArgumentTailWhenAssemblyIsEmpty() {
        // An assembler that drains to nothing (an empty composer) appends no
        // tail — the block stands alone.
        let leaf = SkillExecution(
            assembleArguments: { _, _ in "" },
            loadSkillBody: { _ in "BODY" }
        )
        let injection = leaf.render(
            skillName: "summarize", filePath: "/skills/summarize/SKILL.md",
            userText: "ignored", images: [])

        // No argument tail: the message is exactly the envelope block.
        #expect(
            injection?.message
                == SkillEnvelope.injection(
                    name: "summarize", location: "/skills/summarize/SKILL.md", body: "BODY"))
    }

    // MARK: - Images ride the fire

    @Test func renderForwardsTheImagesUntouched() {
        let a = image("appshot")
        let b = image("paste")
        let leaf = SkillExecution(loadSkillBody: { _ in "BODY" })
        let injection = leaf.render(
            skillName: "describe", filePath: "/skills/describe/SKILL.md",
            userText: "", images: [a, b])

        #expect(injection?.images.map(\.id) == [a.id, b.id])
    }

    // MARK: - Load failure

    @Test func renderReturnsNilWhenTheFileCannotBeLoaded() {
        let leaf = SkillExecution()
        let injection = leaf.render(
            skillName: "ghost", filePath: "/nonexistent/ghost/SKILL.md",
            userText: "text", images: [image("x")])

        // Nil is the load-failure signal — the spine surfaces the error and
        // restores the draft; the leaf sets nothing itself.
        #expect(injection == nil)
    }

    @Test func renderReadsRealFrontmatterStrippedBody() throws {
        // The default loader strips YAML frontmatter (the registry read), so
        // only the body reaches the block.
        let url = try writeSkillFile(name: "summarize", body: "Summarize content.")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let leaf = SkillExecution()
        let injection = try #require(
            leaf.render(skillName: "summarize", filePath: url.path, userText: "", images: []))

        #expect(injection.message.contains("Summarize content."))
        #expect(!injection.message.contains("description: summarize test skill"))
    }

    // MARK: - Usage recording

    @Test func recordFiredForwardsToTheInjectedRecorder() {
        let recorded = Locked<[String]>([])
        let leaf = SkillExecution(
            recordInvocation: { name in recorded.value.append(name) },
            loadSkillBody: { _ in "BODY" }
        )

        leaf.recordFired("proofread")
        leaf.recordFired("proofread")
        leaf.recordFired("translate")

        // Recording is the leaf's job; the spine calls it only after a sent
        // injection, so the count is exactly the fires it was told about.
        #expect(recorded.value == ["proofread", "proofread", "translate"])
    }
}
