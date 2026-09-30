//
//  ActiveToolSetTests.swift
//  tesseractTests
//
//  The **Active Tool Set** at its own seam (ADR-0048): pure decision tables
//  over `resolve` (Web Access) and `promptFacts`, plus the prompt/callable
//  consistency invariant. Every conversation and every Companion moment
//  resolves the same way, so they share one cached system-and-tools prefix.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ActiveToolSetTests {

    // MARK: - Fixtures

    private func tool(_ name: String) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name, label: name, description: "",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            execute: { _, _, _, _ in .text("") })
    }

    /// A registry-shaped universe: built-ins, the skill tool, a real browser
    /// tool name, and a non-browser extension tool.
    private var universe: [AgentToolDefinition] {
        [
            tool("read"),
            tool(skillToolName),
            tool("agenda"),
            tool("browser.search"),
            tool("files.list"),
        ]
    }

    private func names(web: Bool) -> [String] {
        ActiveToolSet.resolve(
            from: universe, gating: ToolGating(webAccessEnabled: web)
        ).map(\.name)
    }

    // MARK: - resolve: Web Access gate

    /// Web off strips exactly the browser tools; a non-browser extension tool
    /// is untouched, so the switch keeps meaning what it says (#190, US #16).
    @Test func webOffStripsBrowserTools() {
        let resolved = names(web: false)
        #expect(!resolved.contains("browser.search"))
        #expect(resolved.contains("files.list"))
    }

    @Test func webOnKeepsEveryTool() {
        #expect(names(web: true) == universe.map(\.name))
    }

    /// Registry order is the loop's dispatch precedence — resolve must keep it.
    @Test func resolvePreservesInputOrder() {
        #expect(names(web: false) == ["read", skillToolName, "agenda", "files.list"])
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
        for web in [true, false] {
            let resolved = ActiveToolSet.resolve(
                from: universe, gating: ToolGating(webAccessEnabled: web))
            let facts = ActiveToolSet.promptFacts(for: resolved)
            #expect(
                facts.carriesBrowserTools
                    == resolved.contains {
                        ActiveToolSet.webGatedToolNames.contains($0.name)
                    })
            #expect(facts.hasSkillTool == resolved.contains { $0.name == skillToolName })
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
