import Foundation
import Testing
import MLXLMCommon
@testable import Tesseract_Agent

@MainActor
struct ToolCallConverterTests {

    // MARK: - convertToOpenAI

    @Test func endToEndParserToOpenAIConversion() throws {
        let parser = ToolCallParser()
        let events = parser.processChunk(
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
        )

        let parsedToolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        #expect(parsedToolCalls.count == 1)

        let openAICalls = ToolCallConverter.convertToOpenAI(parsedToolCalls)

        #expect(openAICalls.count == 1)
        let call = openAICalls[0]
        #expect(call.id?.hasPrefix("call_") == true)
        #expect(call.type == "function")
        #expect(call.function?.name == "bash")
        #expect(call.index == 0)

        let argsData = try #require(
            JSONSerialization.jsonObject(with: Data((call.function?.arguments ?? "").utf8)) as? [String: Any]
        )
        #expect(argsData["command"] as? String == "ls")
    }

    @Test func convertsSingleToolCall() {
        let toolCall = ToolCall(
            function: .init(name: "read", arguments: ["path": "main.swift" as any Sendable])
        )

        let result = ToolCallConverter.convertToOpenAI([toolCall])

        #expect(result.count == 1)
        #expect(result[0].id?.hasPrefix("call_") == true)
        #expect(result[0].type == "function")
        #expect(result[0].function?.name == "read")
        #expect(result[0].index == 0)
        #expect(result[0].function?.arguments?.contains("main.swift") == true)
    }

    @Test func convertsMultipleToolCallsWithUniqueIDs() {
        let calls = [
            ToolCall(function: .init(name: "bash", arguments: ["command": "ls" as any Sendable])),
            ToolCall(function: .init(name: "read", arguments: ["path": "file.txt" as any Sendable])),
        ]

        let result = ToolCallConverter.convertToOpenAI(calls)

        #expect(result.count == 2)
        #expect(result[0].index == 0)
        #expect(result[1].index == 1)
        #expect(result[0].function?.name == "bash")
        #expect(result[1].function?.name == "read")
        #expect(result[0].id != result[1].id)
        #expect(result[0].id?.hasPrefix("call_") == true)
        #expect(result[1].id?.hasPrefix("call_") == true)
    }

    @Test func emptyArgumentsProduceValidJSON() {
        let toolCall = ToolCall(
            function: .init(name: "get_time", arguments: [:] as [String: any Sendable])
        )

        let result = ToolCallConverter.convertToOpenAI([toolCall])

        #expect(result[0].function?.arguments == "{}")
    }

    @Test func complexArgumentsAreJSONStringified() throws {
        let toolCall = ToolCall(function: .init(name: "edit", arguments: [
            "path": "file.swift" as any Sendable,
            "line": 42 as any Sendable,
            "insert": true as any Sendable,
        ]))

        let result = ToolCallConverter.convertToOpenAI([toolCall])
        let argsJSON = result[0].function?.arguments ?? ""

        let parsed = try #require(
            JSONSerialization.jsonObject(with: Data(argsJSON.utf8)) as? [String: Any]
        )
        #expect(parsed["path"] as? String == "file.swift")
        #expect(parsed["line"] as? Int == 42)
        #expect(parsed["insert"] as? Bool == true)
    }

    @Test func emptyToolCallsReturnsEmptyArray() {
        let result = ToolCallConverter.convertToOpenAI([])
        #expect(result.isEmpty)
    }

    // MARK: - mapToolCallIDs

    @Test func remapsToolCallIDOnToolMessages() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .user, content: .text("Run ls")),
            .init(role: .tool, content: .text("file1.txt"), tool_call_id: "call_client123"),
        ]
        let idMap = ["call_client123": "call_server456"]

        let result = ToolCallConverter.mapToolCallIDs(messages, idMap: idMap)

        #expect(result[0].tool_call_id == nil)
        #expect(result[1].tool_call_id == "call_server456")
    }

    @Test func remapsIDsInToolCallsArray() {
        let messages: [OpenAI.ChatMessage] = [
            .init(
                role: .assistant,
                content: .text(""),
                tool_calls: [
                    OpenAI.ToolCall(
                        id: "call_abc",
                        type: "function",
                        function: OpenAI.FunctionCall(name: "bash", arguments: "{}")
                    ),
                ]
            ),
        ]
        let idMap = ["call_abc": "call_xyz"]

        let result = ToolCallConverter.mapToolCallIDs(messages, idMap: idMap)

        #expect(result[0].tool_calls?[0].id == "call_xyz")
    }

    @Test func emptyIDMapPassesThroughUnchanged() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .tool, content: .text("result"), tool_call_id: "call_abc"),
        ]

        let result = ToolCallConverter.mapToolCallIDs(messages, idMap: [:])

        #expect(result[0].tool_call_id == "call_abc")
    }

    @Test func unmappedIDsLeftUnchanged() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .tool, content: .text("result"), tool_call_id: "call_unknown"),
        ]
        let idMap = ["call_other": "call_mapped"]

        let result = ToolCallConverter.mapToolCallIDs(messages, idMap: idMap)

        #expect(result[0].tool_call_id == "call_unknown")
    }
}
