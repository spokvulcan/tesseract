//
//  CompanionTurnRunnerTests.swift
//  tesseractTests
//
//  The headless turn envelope's tool-sync contract (ADR-0048): the runner
//  caches its agent across turns, and the registry moves under it (the
//  browser MCP server connects asynchronously; servers reconnect) — so every
//  turn re-resolves the live tool set and prompt facts through the injected
//  sync, exactly like the interactive send path. Hermetic: no-op agent,
//  in-memory arbiter, disabled memory engine, scratch directories.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CompanionTurnRunnerTests {

    // MARK: - Fixtures

    private func tool(
        _ name: String, audience: AgentToolDefinition.Audience = .all
    ) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name, label: name, description: "",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            audience: audience,
            execute: { _, _, _, _ in .text("") })
    }

    /// A hermetic runner over the given agent, wired the way the container
    /// wires it: `syncActiveTools` re-resolves from a live registry surrogate
    /// (`allTools()`) under the companion gating.
    private func makeRunner(
        conversationDirectory: URL,
        agent: Agent,
        allTools: @escaping () -> [AgentToolDefinition]
    ) throws -> CompanionTurnRunner {
        let gating = ToolGating(consumer: .companionHeadless, webAccessEnabled: true)
        let engine = MemoryEngine(
            store: try scratchStore(),
            embedder: MemoryEmbedder(),
            isEnabled: { false },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { nil }
        )
        let recorder = scratchRecorder()
        return CompanionTurnRunner(
            makeAgent: { agent },
            syncActiveTools: { agent in
                let tools = ActiveToolSet.resolve(from: allTools(), gating: gating)
                agent.updateTools(tools)
                agent.syncSystemPrompt(facts: ActiveToolSet.promptFacts(for: tools))
            },
            arbiter: InMemoryInferenceArbiter(),
            conversationStore: AgentConversationStore(directory: conversationDirectory),
            memory: engine,
            recorder: recorder,
            settings: SettingsManager(store: InMemorySettingsStore()),
            context: CompanionTurnContext(),
            presence: CompanionPresence(recorder: recorder),
            isModelDownloaded: { _ in false }
        )
    }

    // MARK: - Per-turn re-resolve

    /// The late-connect defect, headless form (ADR-0048): the browser MCP
    /// server connects only after the cached agent's first turn built it. The
    /// next turn must see the refreshed registry — callable set *and* prompt —
    /// not the registry as it stood at build time.
    @Test func laterTurnsPickUpARegistryThatMovedUnderTheCachedAgent() async throws {
        let dir = makeTempDir("turn-runner")
        defer { try? FileManager.default.removeItem(at: dir) }

        var registry = [tool("read"), tool("track", audience: .companionOnly)]
        let agent = makeNoOpAgent(modelID: "turn-runner-test-model")
        agent.setSystemPromptReassembler(
            initialFacts: PromptToolFacts(hasSkillTool: false, carriesBrowserTools: false)
        ) { facts in
            facts.carriesBrowserTools ? "prompt+web" : "prompt"
        }
        let runner = try makeRunner(conversationDirectory: dir, agent: agent) { registry }

        // Turn 1: the browser MCP connection has not landed yet.
        let first = await runner.run(origin: .wake, opening: "first")
        #expect(first != nil)
        #expect(agent.state.tools.map(\.name) == ["read", "track"])

        // The connection lands: the registry refreshes under the cached agent.
        registry.append(tool("browser.search"))

        // Turn 2: the same cached agent carries the new tool, and the prompt
        // was rebuilt from the new facts — the ADR-0048 invariant, per turn.
        let second = await runner.run(origin: .event, opening: "second")
        #expect(second != nil)
        #expect(agent.state.tools.map(\.name).contains("browser.search"))
        #expect(agent.state.systemPrompt == "prompt+web")
    }
}
