//
//  ExtensionToolOrderTests.swift
//  tesseractTests
//
//  Extension tools reach the system prompt in one order, whatever order their
//  dictionary iterates in. Swift seeds a dictionary's order per process, so
//  an unsorted walk changed the tool list — and every cached prefix after it —
//  with each relaunch.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ExtensionToolOrderTests {

    private func tool(_ name: String) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name, label: name, description: "",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            execute: { _, _, _, _ in .text("") })
    }

    private let names = [
        "browser.navigate", "browser.click", "browser.type", "browser.screenshot",
        "browser.read_page", "browser.find", "browser.search", "browser.back",
    ]

    @Test func extensionToolsComeInNameOrder() {
        let mcp = MCPToolsExtension()
        mcp.update(names.map(tool))
        let host = ExtensionHost()
        host.register(mcp)
        #expect(host.aggregatedTools().map(\.name) == names.sorted())
    }

    @Test func theSameToolsPushedInAnotherOrderListTheSame() {
        let mcp = MCPToolsExtension()
        let host = ExtensionHost()
        host.register(mcp)
        mcp.update(names.map(tool))
        let first = host.aggregatedTools().map(\.name)
        for _ in 0..<5 {
            mcp.update(names.shuffled().map(tool))
            #expect(host.aggregatedTools().map(\.name) == first)
        }
    }
}
