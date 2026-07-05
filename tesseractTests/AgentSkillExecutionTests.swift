//
//  AgentSkillExecutionTests.swift
//  tesseractTests
//
//  Coordinator-level tests for skill execution (PRD #174): a Skill Pill fire
//  and a skill slash command drive the PUBLIC coordinator interface against a
//  scripted `Agent` (yield → finish, the dispatch-ordering fixture), asserting
//  the observable outcome — the message the agent actually received (injected
//  skill block, argument text, forwarded images) and the recorded usage count.
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

    private func makeCoordinator(
        agent: Agent, settings: SettingsManager, skills: [SkillMetadata] = []
    ) -> AgentCoordinator {
        let coordinator = AgentCoordinator(
            agent: agent,
            conversationStore: InMemoryAgentConversationStore(),
            settings: settings,
            batchEngine: InMemoryInferenceArbiter().makeBatchEngine(),
            discoverSkills: { skills }
        )
        coordinator.imageDraft.imageInputAvailable = true
        return coordinator
    }

    private func settle(_ coordinator: AgentCoordinator) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while coordinator.isGenerating {
            try await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("Coordinator did not settle within timeout")
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
        let coordinator = makeCoordinator(agent: agent, settings: settings)

        let attachment = image("appshot")
        coordinator.imageDraft.pendingImages = [attachment]
        let pill = SkillPill(
            name: "proofread", label: "Proofread",
            description: "d", filePath: skillURL.path)

        coordinator.fireSkillPill(pill, composerText: "my draft text")
        try await settle(coordinator)

        let user = try #require(firstUserMessage(agent))
        // The injected skill block wraps the body; the composer text rides as
        // arguments; the pending images are forwarded (the PRD's known gap).
        #expect(user.content.hasPrefix("<skill name=\"proofread\""))
        #expect(user.content.contains("Fix errors only."))
        #expect(user.content.hasSuffix("my draft text"))
        #expect(user.images.map(\.id) == [attachment.id])
        // The composer's pending strip was drained by the fire.
        #expect(coordinator.imageDraft.pendingImages.isEmpty)
        // A user-initiated invocation records usage.
        #expect(settings.skillUsageCount(skillName: "proofread") == 1)
    }

    @Test func barePillFireWithEmptyComposerStillFires() async throws {
        let skillURL = try writeSkillFile(name: "summarize", body: "Summarize content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let coordinator = makeCoordinator(
            agent: agent, settings: SettingsManager(store: InMemorySettingsStore()))
        let pill = SkillPill(
            name: "summarize", label: "Summarize", description: "d", filePath: skillURL.path)

        coordinator.fireSkillPill(pill, composerText: "")
        try await settle(coordinator)

        let user = try #require(firstUserMessage(agent))
        #expect(user.content.hasPrefix("<skill name=\"summarize\""))
        #expect(user.content.hasSuffix("</skill>"))
        #expect(user.images.isEmpty)
    }

    @Test func pillFireWhileGeneratingIsANoOp() throws {
        let skillURL = try writeSkillFile(name: "proofread", body: "Body.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let coordinator = makeCoordinator(agent: agent, settings: settings)
        let pill = SkillPill(
            name: "proofread", label: "Proofread", description: "d", filePath: skillURL.path)

        // Occupy the run envelope, then try to fire.
        coordinator.sendMessage("first message")
        #expect(coordinator.isGenerating == true)
        coordinator.fireSkillPill(pill, composerText: "should not send")

        #expect(settings.skillUsageCount(skillName: "proofread") == 0)
    }

    @Test func failedSkillLoadRestoresImagesAndCountsNothing() {
        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let coordinator = makeCoordinator(agent: agent, settings: settings)
        let attachment = image("appshot")
        coordinator.imageDraft.pendingImages = [attachment]
        let pill = SkillPill(
            name: "ghost", label: "Ghost", description: "d",
            filePath: "/nonexistent/ghost/SKILL.md")

        coordinator.fireSkillPill(pill, composerText: "text")

        // A failed fire surfaces the error, puts the Appshot back in the
        // pending strip, counts nothing, and sends nothing.
        #expect(coordinator.error?.contains("Failed to load skill") == true)
        #expect(coordinator.imageDraft.pendingImages.map(\.id) == [attachment.id])
        #expect(settings.skillUsageCount(skillName: "ghost") == 0)
        #expect(agent.state.messages.isEmpty)
    }

    // MARK: - Slash command path

    @Test func skillSlashCommandRecordsUsage() async throws {
        let skillURL = try writeSkillFile(name: "explain", body: "Explain content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        let coordinator = makeCoordinator(agent: agent, settings: settings)

        let command = SlashCommand(
            name: "explain", description: "d",
            source: .skill(filePath: skillURL.path), argumentHint: nil)
        coordinator.executeCommand(command, arguments: "this error")
        try await settle(coordinator)

        let user = try #require(firstUserMessage(agent))
        #expect(user.content.hasPrefix("<skill name=\"explain\""))
        #expect(user.content.hasSuffix("this error"))
        #expect(settings.skillUsageCount(skillName: "explain") == 1)
    }

    @Test func translateInvocationCarriesTheConfiguredTarget() async throws {
        let skillURL = try writeSkillFile(name: "translate", body: "Translate content.")
        defer { try? FileManager.default.removeItem(at: skillURL.deletingLastPathComponent()) }

        let agent = makeScriptedAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.translateTargetLanguage = "Ukrainian"
        let coordinator = makeCoordinator(agent: agent, settings: settings)
        let pill = SkillPill(
            name: "translate", label: "Translate", description: "d", filePath: skillURL.path)

        coordinator.fireSkillPill(pill, composerText: "guten Tag")
        try await settle(coordinator)

        let user = try #require(firstUserMessage(agent))
        #expect(user.content.hasSuffix("guten Tag\n\nDefault target language: Ukrainian"))
    }
}
