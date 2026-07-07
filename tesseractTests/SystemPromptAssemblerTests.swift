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
            loadedContext: Self.emptyContext, skills: [], tools: tools,
            dateTime: Date(timeIntervalSince1970: 0), agentRoot: "/tmp/agent")
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
}
