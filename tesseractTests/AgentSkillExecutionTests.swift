//
//  AgentSkillExecutionTests.swift
//  tesseractTests
//
//  Session-level tests for skill execution (PRD #174): a Skill Pill fire and a
//  skill slash command drive the PUBLIC Chat Session interface against a
//  scripted `Agent` (yield → finish), asserting the observable outcome — the
//  message the agent actually received (injected skill block, argument text,
//  forwarded images) and the recorded usage count. The session's skill hooks
//  are wired to a real `SkillPillController`, the production assembly path.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentSkillExecutionTests {

    // MARK: - Fixtures

    private func image(_ name: String) -> ImageAttachment {
        ImageAttachment(
            data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png", filename: name)
    }

    private func makeScriptedAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "skill-execution-test-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in
                AsyncThrowingStream { continuation in
                    continuation.yield(.text("done"))
                    continuation.finish()
                }
            }
        )
    }

    /// Write a real skill file the execution path can load.
    private func writeSkillFile(name: String, body: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("skill-exec-tests-\(UUID().uuidString)/\(name)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("SKILL.md")
        try """
        ---
        name: \(name)
        description: \(name) test skill
        composer-pill: true
        ---
        \(body)
        """.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func makeSession(agent: Agent, settings: SettingsManager) -> ChatSession {
        // The production wiring shape: assembly and usage recording resolve on
        // a real Skill Pill controller (DependencyContainer does the same).
        let skillPills = SkillPillController(discoverSkills: { [] }, settings: settings)
        return ChatSession(
            agent: agent,
            conversationStore: InMemoryAgentConversationStore(),
            arbiter: InMemoryInferenceArbiter(),
            settings: settings,
            skillExecution: SkillExecution(
                assembleArguments: { name, text in
                    skillPills.assembleArguments(skillName: name, userText: text)
                },
                recordInvocation: { name in
                    skillPills.recordUserInvocation(skillName: name)
                }
            ),
            liveMarkdownThrottle: .zero
        )
    }

    private func settle(_ session: ChatSession) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while session.isGenerating {
            try await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("Session did not settle within timeout")
                break
            }
        }
    }

    private func firstUserMessage(_ agent: Agent) -> UserMessage? {
        agent.state.messages.compactMap(\.asUser).first
    }

    // MARK: - Skill Pill fire

    @Test func pillFireSendsSkillBlockWithComposerTextAndImages() async throws {
        let skillURL = try writeSkillFile(name: "proofread", body: "Fix errors only.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let session = makeSession(agent: agent, settings: settings)

        let attachment = image("appshot")
        let pill = SkillPill(
            name: "proofread", label: "Proofread",
            description: "d", filePath: skillURL.path)

        // The pill row drains the composer draft and hands it to the fire.
        let sent = session.fireSkillPill(pill, draftText: "my draft text", images: [attachment])
        try await settle(session)

        #expect(sent == true)
        let user = try #require(firstUserMessage(agent))
        // The injected skill block wraps the body; the composer text rides as
        // arguments; the pending images are forwarded (the PRD's known gap).
        // Parse through the envelope so we assert the message is a *valid*
        // invocation block, not just that it starts with the right bytes.
        let block = try #require(SkillEnvelope.parse(user.content))
        #expect(block.skillName == "proofread")
        #expect(block.injectedBlock.contains("Fix errors only."))
        #expect(block.argumentText == "my draft text")
        #expect(user.images.map(\.id) == [attachment.id])
        // A user-initiated invocation records usage.
        #expect(settings.skillUsageCount(skillName: "proofread") == 1)
    }

    @Test func barePillFireWithEmptyComposerStillFires() async throws {
        let skillURL = try writeSkillFile(name: "summarize", body: "Summarize content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let session = makeSession(
            agent: agent, settings: SettingsManager(store: InMemorySettingsStore()))
        let pill = SkillPill(
            name: "summarize", label: "Summarize", description: "d", filePath: skillURL.path)

        let sent = session.fireSkillPill(pill, draftText: "", images: [])
        try await settle(session)

        #expect(sent == true)
        let user = try #require(firstUserMessage(agent))
        let block = try #require(SkillEnvelope.parse(user.content))
        #expect(block.skillName == "summarize")
        #expect(block.argumentText.isEmpty)
        #expect(user.images.isEmpty)
    }

    @Test func pillFireWhileGeneratingIsANoOp() throws {
        let skillURL = try writeSkillFile(name: "proofread", body: "Body.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let session = makeSession(agent: agent, settings: settings)
        let pill = SkillPill(
            name: "proofread", label: "Proofread", description: "d", filePath: skillURL.path)

        // Occupy the run envelope, then try to fire.
        session.sendMessage("first message")
        #expect(session.isGenerating == true)
        let sent = session.fireSkillPill(pill, draftText: "", images: [])

        // A refused fire returns false — the pill row keeps the draft — and
        // counts nothing.
        #expect(sent == false)
        #expect(settings.skillUsageCount(skillName: "proofread") == 0)
    }

    @Test func failedSkillLoadReturnsFalseAndCountsNothing() {
        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let session = makeSession(agent: agent, settings: settings)
        let attachment = image("appshot")
        let pill = SkillPill(
            name: "ghost", label: "Ghost", description: "d",
            filePath: "/nonexistent/ghost/SKILL.md")

        let sent = session.fireSkillPill(pill, draftText: "text", images: [attachment])

        // A failed fire surfaces the error and reports failure so the pill row
        // (which drained the draft) restores it whole — counting nothing and
        // sending nothing.
        #expect(sent == false)
        #expect(session.error?.contains("Failed to load skill") == true)
        #expect(settings.skillUsageCount(skillName: "ghost") == 0)
        #expect(agent.state.messages.isEmpty)
    }

    // MARK: - Slash command path

    @Test func skillSlashCommandRecordsUsage() async throws {
        let skillURL = try writeSkillFile(name: "explain", body: "Explain content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let session = makeSession(agent: agent, settings: settings)

        let command = SlashCommand(
            name: "explain", description: "d",
            source: .skill(filePath: skillURL.path), argumentHint: nil)
        session.executeCommand(command, arguments: "this error")
        try await settle(session)

        let user = try #require(firstUserMessage(agent))
        let block = try #require(SkillEnvelope.parse(user.content))
        #expect(block.skillName == "explain")
        #expect(block.argumentText == "this error")
        #expect(settings.skillUsageCount(skillName: "explain") == 1)
    }

    @Test func translateInvocationCarriesTheConfiguredTarget() async throws {
        let skillURL = try writeSkillFile(name: "translate", body: "Translate content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.translateTargetLanguage = "Ukrainian"
        let session = makeSession(agent: agent, settings: settings)
        let pill = SkillPill(
            name: "translate", label: "Translate", description: "d", filePath: skillURL.path)

        session.fireSkillPill(pill, draftText: "guten Tag", images: [])
        try await settle(session)

        let user = try #require(firstUserMessage(agent))
        let block = try #require(SkillEnvelope.parse(user.content))
        #expect(block.argumentText == "guten Tag\n\nDefault target language: Ukrainian")
    }
}
