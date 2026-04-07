//
//  AgentEngineToolSpecTests.swift
//  tesseractTests
//

import Foundation
import Testing
import MLXLMCommon
import Tokenizers
@testable import Tesseract_Agent

@MainActor
struct AgentEngineToolSpecTests {

    private static let sampleToolSpecs: [ToolSpec] = [
        [
            "type": "function",
            "function": [
                "name": "bash",
                "description": "Execute shell commands",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "command": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["command"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    @Test func buildUserInputForwardsToolSpecs() {
        let input = AgentEngine.buildUserInput(
            systemPrompt: "You are a helper.",
            messages: [.user(content: "hi")],
            toolSpecs: Self.sampleToolSpecs
        )

        #expect(input.tools != nil)
        #expect(input.tools?.count == 1)

        let tool = input.tools![0]
        #expect(tool["type"] as? String == "function")
        let function = tool["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == "bash")
    }

    @Test func buildUserInputPassesNilToolSpecs() {
        let input = AgentEngine.buildUserInput(
            systemPrompt: "System",
            messages: [.user(content: "hi")],
            toolSpecs: nil
        )

        #expect(input.tools == nil)
    }

    @Test func generateToolSpecsOverloadReachesStartGeneration() throws {
        let engine = AgentEngine()

        #expect(throws: AgentEngineError.self) {
            _ = try engine.generate(
                systemPrompt: "System",
                messages: [.user(content: "hi")],
                toolSpecs: Self.sampleToolSpecs,
                parameters: .default
            )
        }
    }
}
