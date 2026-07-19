//
//  ActiveToolSetTests.swift
//  tesseractTests
//
//  The **Active Tool Set** at its own seam (ADR-0048): pure decision tables
//  over `resolve` (consumer × audience × Web Access) and `promptFacts`, plus
//  the prompt/callable consistency invariant that used to be unrepresentable —
//  the callable set was filtered per turn while the prompt was assembled from
//  the unfiltered registry, and no test could pin the two to one universe.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ActiveToolSetTests {

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

    /// A registry-shaped universe: built-ins, a companion-only tool, a
    /// chat-only tool, a real browser tool name, and a non-browser
    /// extension tool.
    private var universe: [AgentToolDefinition] {
        [
            tool("read"),
            tool(skillToolName),
            tool("track", audience: .companionOnly),
            tool("report_back", audience: .chatOnly),
            tool("browser.search"),
            tool("files.list"),
        ]
    }

    private func names(
        _ consumer: ToolGating.Consumer, web: Bool
    ) -> [String] {
        ActiveToolSet.resolve(
            from: universe,
            gating: ToolGating(consumer: consumer, webAccessEnabled: web)
        ).map(\.name)
    }

    // MARK: - resolve: audience rules per consumer

    /// Every owner-facing chat carries the chat-only tools (`report_back` —
    /// ADR-0052's one contract) while companion-only delivery tools still
    /// never reach a chat the owner is looking at (ADR-0040 §10).
    @Test func chatKeepsChatOnlyDropsCompanionOnly() {
        let resolved = names(.chat, web: true)
        #expect(resolved.contains("report_back"))
        #expect(!resolved.contains("track"))
        #expect(resolved.contains("read"))
        #expect(resolved.contains("browser.search"))
    }

    /// The Companion's headless agent keeps its companion-only tools and never
    /// carries chat-only ones — a Mission Control turn has no conversation to
    /// report back from (#372).
    @Test func companionHeadlessKeepsCompanionOnlyDropsChatOnly() {
        let resolved = names(.companionHeadless, web: true)
        #expect(resolved.contains("track"))
        #expect(!resolved.contains("report_back"))
    }

    // MARK: - resolve: Web Access gate

    /// Web off strips exactly the browser tools, for every consumer; a
    /// non-browser extension tool is untouched, so the switch keeps meaning
    /// what it says (#190, US #16).
    @Test func webOffStripsBrowserToolsForEveryConsumer() {
        for consumer: ToolGating.Consumer in [
            .chat, .companionHeadless,
        ] {
            let resolved = names(consumer, web: false)
            #expect(!resolved.contains("browser.search"))
            #expect(resolved.contains("files.list"))
        }
    }

    @Test func webOnKeepsBrowserTools() {
        #expect(names(.chat, web: true).contains("browser.search"))
    }

    /// Registry order is the loop's dispatch precedence — resolve must keep it.
    @Test func resolvePreservesInputOrder() {
        let resolved = names(.chat, web: true)
        #expect(
            resolved == ["read", skillToolName, "report_back", "browser.search", "files.list"])
    }

    // MARK: - promptFacts

    @Test func promptFactsTrackSkillAndBrowserMembership() {
        let withBoth = ActiveToolSet.promptFacts(
            for: [tool(skillToolName), tool("browser.search")])
        #expect(withBoth == PromptToolFacts(hasSkillTool: true, carriesBrowserTools: true))

        let withNeither = ActiveToolSet.promptFacts(for: [tool("read"), tool("files.list")])
        #expect(
            withNeither == PromptToolFacts(hasSkillTool: false, carriesBrowserTools: false))
    }

    /// The ADR-0048 invariant: for every gating context, the prompt facts of
    /// the resolved set agree with the resolved set itself — the drift class
    /// that shipped (prompt instructing stripped browser tools) is
    /// unrepresentable through this seam.
    @Test func promptFactsAgreeWithResolvedSetForEveryGating() {
        for consumer: ToolGating.Consumer in [
            .chat, .companionHeadless,
        ] {
            for web in [true, false] {
                let resolved = ActiveToolSet.resolve(
                    from: universe,
                    gating: ToolGating(consumer: consumer, webAccessEnabled: web))
                let facts = ActiveToolSet.promptFacts(for: resolved)
                #expect(
                    facts.carriesBrowserTools
                        == resolved.contains {
                            ActiveToolSet.webGatedToolNames.contains($0.name)
                        })
                #expect(facts.hasSkillTool == resolved.contains { $0.name == skillToolName })
            }
        }
    }

    // MARK: - Agent.syncSystemPrompt

    /// A facts change rebuilds the prompt through the wired reassembler; the
    /// same facts again are a no-op, so the prompt (and the prefix cache
    /// riding it) is only invalidated by a real orientation change.
    @Test func syncSystemPromptRebuildsOnFactsChangeOnly() {
        let agent = makeNoOpAgent(modelID: "active-tool-set-test-model")
        var rebuilds = 0
        agent.setSystemPromptReassembler(
            initialFacts: PromptToolFacts(hasSkillTool: false, carriesBrowserTools: true)
        ) { facts in
            rebuilds += 1
            return facts.carriesBrowserTools ? "prompt+web" : "prompt"
        }

        // Same facts as initial: no rebuild, prompt untouched.
        agent.syncSystemPrompt(
            facts: PromptToolFacts(hasSkillTool: false, carriesBrowserTools: true))
        #expect(rebuilds == 0)
        #expect(agent.state.systemPrompt == "test")

        // Facts changed: one rebuild, both context and state updated.
        agent.syncSystemPrompt(
            facts: PromptToolFacts(hasSkillTool: false, carriesBrowserTools: false))
        #expect(rebuilds == 1)
        #expect(agent.state.systemPrompt == "prompt")
        #expect(agent.context.systemPrompt == "prompt")

        // Unchanged again: still one rebuild.
        agent.syncSystemPrompt(
            facts: PromptToolFacts(hasSkillTool: false, carriesBrowserTools: false))
        #expect(rebuilds == 1)
    }

    /// Without a wired reassembler the sync is a no-op — bench and test agents
    /// keep their fixed prompts.
    @Test func syncSystemPromptWithoutReassemblerIsNoOp() {
        let agent = makeNoOpAgent(modelID: "active-tool-set-test-model")
        agent.syncSystemPrompt(
            facts: PromptToolFacts(hasSkillTool: true, carriesBrowserTools: true))
        #expect(agent.state.systemPrompt == "test")
    }
}
