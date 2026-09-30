import Foundation
import Testing
@testable import Tesseract_Agent

// The web-orientation block (ADR-0028) is assembled into the system prompt only
// when the turn carries browser tools. Tested at the pure `assemble` seam.

@MainActor
struct SystemPromptAssemblerTests {

    private static let emptyContext = ContextLoader.LoadedContext(
        contextFiles: [], systemOverride: nil, systemAppend: nil)

    private func tool(_ name: String) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name, label: name, description: "",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            execute: { _, _, _, _ in .text("") })
    }

    private func assemble(tools: [AgentToolDefinition]) -> String {
        SystemPromptAssembler.assemble(
            loadedContext: Self.emptyContext, skills: [], tools: tools, agentRoot: "/tmp/agent")
    }

    /// The web-orientation block appears when the turn carries a browser tool.
    @Test func includesWebBlockWhenBrowserToolsPresent() {
        let prompt = assemble(tools: [tool("browser.search"), tool("read")])
        #expect(prompt.contains("Web access:"))
        #expect(prompt.contains("browser.search to find candidate pages"))
    }

    /// …and is omitted with no browser tool, so a web-disabled or text-only turn
    /// doesn't carry it.
    @Test func omitsWebBlockWhenNoBrowserTools() {
        let prompt = assemble(tools: [tool("read"), tool("ls")])
        #expect(!prompt.contains("Web access:"))
    }

    /// Any single `browser.*` tool is enough to trigger the block.
    @Test func anyBrowserToolTriggersTheBlock() {
        let prompt = assemble(tools: [tool("browser.navigate")])
        #expect(prompt.contains("Web access:"))
    }

    /// The prefix-cache contract: the system prompt carries no time and
    /// nothing per-conversation, so assembling it at two different moments
    /// yields the same bytes and every chat shares one cached prefix. The time
    /// rides each user message as the Now Tag instead.
    @Test func systemPromptIsByteIdenticalAcrossTime() async throws {
        let first = assemble(tools: [tool("read"), tool("browser.search")])
        try await Task.sleep(for: .milliseconds(1100))
        let second = assemble(tools: [tool("read"), tool("browser.search")])
        #expect(first == second)
        #expect(!first.contains("Current date and time"))
        let year = Calendar.current.component(.year, from: Date())
        #expect(!first.contains(String(year)))
    }

    /// The prompt explains the Now Tag once, so the model knows the `<now>`
    /// line is the app's, not the owner's.
    @Test func systemPromptExplainsTheNowTag() {
        #expect(assemble(tools: []).contains("<now>"))
    }
}
