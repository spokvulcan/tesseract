//
//  tesseractTests.swift
//  tesseractTests
//
//  Created by Bohdan Ivanchenko on 31.01.2026.
//

import Testing
import MLXLMCommon
@testable import tesseract

@MainActor
struct ToolArgumentNormalizerTests {

    @Test func unwrapsWrappedStringArguments() async throws {
        let normalized = ToolArgumentNormalizer.normalize([
            "path": .string(#"string("conversations\/ABC-123.json")"#)
        ])

        #expect(normalized["path"] == .string("conversations/ABC-123.json"))
    }

    @Test func extractorHandlesWrappedScalarArguments() async throws {
        let args: [String: JSONValue] = [
            "path": .string(#".string("notes\/todo.md")"#),
            "offset": .string("int(42)"),
            "recursive": .string("bool(true)"),
        ]

        #expect(ToolArgExtractor.string(args, key: "path") == "notes/todo.md")
        #expect(ToolArgExtractor.int(args, key: "offset") == 42)
        #expect(ToolArgExtractor.bool(args, key: "recursive") == true)
    }

    @Test func boolExtractorHandlesWrappedCapitalizedStringBools() async throws {
        let args: [String: JSONValue] = [
            "recursive": .string(#"string("True")"#),
            "enabled": .string(#"string("off")"#),
        ]

        #expect(ToolArgExtractor.bool(args, key: "recursive") == true)
        #expect(ToolArgExtractor.bool(args, key: "enabled") == false)
        #expect(args.bool(for: "recursive") == true)
        #expect(args.bool(for: "enabled") == false)
    }
}

@MainActor
struct AgentChatMessageToolResultTests {

    @Test func preservesToolErrorStateFromCoreMessage() async throws {
        let toolResult = ToolResultMessage(
            toolCallId: "call-1",
            toolName: "read",
            content: [.text("Tool execution failed: File not found")],
            isError: true
        )

        let message = AgentChatMessage(from: toolResult)

        #expect(message.role == .tool)
        #expect(message.isError)
    }

    @Test func normalizesWrappedToolCallArgumentsForLegacyDisplayMessages() async throws {
        let rawArguments: [String: any Sendable] = [
            "recursive": #"string("True")"#,
            "path": #"string("tasks.md")"#,
        ]
        let rawToolCall = ToolCall(
            function: .init(
                name: "list",
                arguments: rawArguments
            )
        )
        let rawMessage = AgentChatMessage.assistant("", toolCalls: [rawToolCall])

        let normalized = rawMessage.normalizedForDisplay()
        let args = normalized.toolCalls[0].function.arguments

        #expect(args["recursive"] == JSONValue.string("True"))
        #expect(args["path"] == JSONValue.string("tasks.md"))
    }
}
