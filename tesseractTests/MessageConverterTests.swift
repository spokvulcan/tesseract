//
//  MessageConverterTests.swift
//  tesseractTests
//

import Foundation
import Testing
@testable import Tesseract_Agent

@MainActor
struct MessageConverterTests {

    // MARK: - Message Conversion

    @Test func extractsSystemPrompt() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("You are a coding assistant.")),
            .init(role: .user, content: .text("Hello")),
        ]

        let (systemPrompt, converted) = MessageConverter.convertMessages(messages)

        #expect(systemPrompt == "You are a coding assistant.")
        #expect(converted.count == 1)
        if case .user(let content, _) = converted[0] {
            #expect(content == "Hello")
        } else {
            Issue.record("Expected .user message")
        }
    }

    @Test func concatenatesLeadingSystemMessages() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("First system")),
            .init(role: .system, content: .text("Second system")),
            .init(role: .user, content: .text("Hi")),
        ]

        let (systemPrompt, converted) = MessageConverter.convertMessages(messages)

        #expect(systemPrompt == "First system\n\nSecond system")
        #expect(converted.count == 1)
    }

    @Test func preservesMidConversationSystemMessages() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("Initial instructions")),
            .init(role: .user, content: .text("Hello")),
            .init(role: .system, content: .text("Updated instructions")),
            .init(role: .user, content: .text("Continue")),
        ]

        let (systemPrompt, converted) = MessageConverter.convertMessages(messages)

        #expect(systemPrompt == "Initial instructions")
        #expect(converted.count == 3)

        if case .user(let content, _) = converted[0] {
            #expect(content == "Hello")
        } else {
            Issue.record("Expected .user")
        }

        if case .system(let content) = converted[1] {
            #expect(content == "Updated instructions")
        } else {
            Issue.record("Expected .system")
        }

        if case .user(let content, _) = converted[2] {
            #expect(content == "Continue")
        } else {
            Issue.record("Expected .user")
        }
    }

    @Test func convertsMultiTurnConversation() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("System prompt")),
            .init(role: .user, content: .text("Read main.swift")),
            .init(
                role: .assistant,
                content: .text("I'll read that file."),
                tool_calls: [
                    OpenAI.ToolCall(
                        id: "call_abc123",
                        type: "function",
                        function: OpenAI.FunctionCall(name: "read", arguments: #"{"path":"main.swift"}"#)
                    ),
                ]
            ),
            .init(role: .tool, content: .text("file contents here"), tool_call_id: "call_abc123"),
            .init(role: .user, content: .text("Thanks")),
        ]

        let (systemPrompt, converted) = MessageConverter.convertMessages(messages)

        #expect(systemPrompt == "System prompt")
        #expect(converted.count == 4)

        // User message
        if case .user(let content, let images) = converted[0] {
            #expect(content == "Read main.swift")
            #expect(images.isEmpty)
        } else {
            Issue.record("Expected .user")
        }

        // Assistant with tool calls
        if case .assistant(let content, let toolCalls) = converted[1] {
            #expect(content == "I'll read that file.")
            #expect(toolCalls?.count == 1)
            #expect(toolCalls?[0].id == "call_abc123")
            #expect(toolCalls?[0].name == "read")
            #expect(toolCalls?[0].argumentsJSON == #"{"path":"main.swift"}"#)
        } else {
            Issue.record("Expected .assistant")
        }

        // Tool result
        if case .toolResult(let toolCallId, let content) = converted[2] {
            #expect(toolCallId == "call_abc123")
            #expect(content == "file contents here")
        } else {
            Issue.record("Expected .toolResult")
        }

        // Follow-up user
        if case .user(let content, _) = converted[3] {
            #expect(content == "Thanks")
        } else {
            Issue.record("Expected .user")
        }
    }

    @Test func convertsAssistantWithNoToolCalls() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .assistant, content: .text("Just text")),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        if case .assistant(let content, let toolCalls) = converted[0] {
            #expect(content == "Just text")
            #expect(toolCalls == nil)
        } else {
            Issue.record("Expected .assistant")
        }
    }

    @Test func generatesIDWhenToolCallMissingID() {
        let messages: [OpenAI.ChatMessage] = [
            .init(
                role: .assistant,
                content: .text(""),
                tool_calls: [
                    OpenAI.ToolCall(function: OpenAI.FunctionCall(name: "bash", arguments: "{}")),
                ]
            ),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        if case .assistant(_, let toolCalls) = converted[0] {
            #expect(toolCalls?.count == 1)
            #expect(toolCalls?[0].name == "bash")
            #expect(!toolCalls![0].id.isEmpty)
        } else {
            Issue.record("Expected .assistant")
        }
    }

    @Test func handlesNilContent() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .user),
            .init(role: .assistant),
            .init(role: .tool, tool_call_id: "call_x"),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        #expect(converted.count == 3)
        if case .user(let content, _) = converted[0] {
            #expect(content == "")
        }
        if case .assistant(let content, _) = converted[1] {
            #expect(content == "")
        }
        if case .toolResult(_, let content) = converted[2] {
            #expect(content == "")
        }
    }

    // MARK: - Multipart User Messages

    @Test func convertsMultipartUserMessage() {
        let pngData = Data([0x89, 0x50, 0x4E, 0x47])
        let base64 = pngData.base64EncodedString()

        let messages: [OpenAI.ChatMessage] = [
            .init(role: .user, content: .parts([
                OpenAI.ContentPart(type: .text, text: "What is this?"),
                OpenAI.ContentPart(type: .image_url, image_url: OpenAI.ImageURL(url: "data:image/png;base64,\(base64)")),
                OpenAI.ContentPart(type: .text, text: "Describe it."),
            ])),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        if case .user(let content, let images) = converted[0] {
            #expect(content == "What is this?\nDescribe it.")
            #expect(images.count == 1)
            #expect(images[0].mimeType == "image/png")
            #expect(images[0].data == pngData)
        } else {
            Issue.record("Expected .user")
        }
    }

    // MARK: - Image Content

    @Test func decodesDataURIImage() {
        let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
        let base64 = imageData.base64EncodedString()
        let part = OpenAI.ContentPart(
            type: .image_url,
            image_url: OpenAI.ImageURL(url: "data:image/jpeg;base64,\(base64)")
        )

        let attachment = MessageConverter.convertImageContent(part)

        #expect(attachment != nil)
        #expect(attachment?.mimeType == "image/jpeg")
        #expect(attachment?.data == imageData)
    }

    @Test func returnsNilForNonDataURI() {
        let part = OpenAI.ContentPart(
            type: .image_url,
            image_url: OpenAI.ImageURL(url: "https://example.com/image.png")
        )
        #expect(MessageConverter.convertImageContent(part) == nil)
    }

    @Test func returnsNilForTextPart() {
        let part = OpenAI.ContentPart(type: .text, text: "hello")
        #expect(MessageConverter.convertImageContent(part) == nil)
    }

    @Test func returnsNilForMalformedDataURI() {
        let part = OpenAI.ContentPart(
            type: .image_url,
            image_url: OpenAI.ImageURL(url: "data:image/pngbase64,abc")
        )
        #expect(MessageConverter.convertImageContent(part) == nil)
    }

    // MARK: - Tool Definitions

    @Test func convertsToolDefinitions() {
        let tools: [OpenAI.ToolDefinition] = [
            OpenAI.ToolDefinition(
                type: "function",
                function: OpenAI.FunctionDefinition(
                    name: "bash",
                    description: "Execute shell commands",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "command": .object(["type": .string("string")]),
                        ]),
                        "required": .array([.string("command")]),
                    ])
                )
            ),
        ]

        let specs = MessageConverter.convertToolDefinitions(tools)

        #expect(specs != nil)
        #expect(specs?.count == 1)

        let spec = specs![0]
        #expect(spec["type"] as? String == "function")

        let function = spec["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == "bash")
        #expect(function?["description"] as? String == "Execute shell commands")
        #expect(function?["parameters"] != nil)
    }

    @Test func returnsNilForNilTools() {
        #expect(MessageConverter.convertToolDefinitions(nil) == nil)
    }

    @Test func returnsNilForEmptyTools() {
        #expect(MessageConverter.convertToolDefinitions([]) == nil)
    }

    @Test func convertsToolWithNoParameters() {
        let tools: [OpenAI.ToolDefinition] = [
            OpenAI.ToolDefinition(
                type: "function",
                function: OpenAI.FunctionDefinition(name: "get_time")
            ),
        ]

        let specs = MessageConverter.convertToolDefinitions(tools)

        #expect(specs?.count == 1)
        let function = specs![0]["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == "get_time")
        #expect(function?["description"] == nil)
        #expect(function?["parameters"] == nil)
    }

    // MARK: - Tool Result Reordering

    @Test func reordersToolResultsToMatchToolCalls() {
        // Assistant calls tools A then B, but client returns B's result first
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .user, content: .text("Do two things")),
            .init(
                role: .assistant,
                content: .text("I'll call both tools."),
                tool_calls: [
                    OpenAI.ToolCall(id: "call_A", type: "function",
                        function: OpenAI.FunctionCall(name: "read", arguments: #"{"path":"a.txt"}"#)),
                    OpenAI.ToolCall(id: "call_B", type: "function",
                        function: OpenAI.FunctionCall(name: "read", arguments: #"{"path":"b.txt"}"#)),
                ]
            ),
            // Client sends B's result before A's
            .init(role: .tool, content: .text("contents of b"), tool_call_id: "call_B"),
            .init(role: .tool, content: .text("contents of a"), tool_call_id: "call_A"),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        // Tool results should be reordered to match tool_calls: A first, then B
        if case .toolResult(let id1, let content1) = converted[2] {
            #expect(id1 == "call_A")
            #expect(content1 == "contents of a")
        } else {
            Issue.record("Expected .toolResult at index 2")
        }

        if case .toolResult(let id2, let content2) = converted[3] {
            #expect(id2 == "call_B")
            #expect(content2 == "contents of b")
        } else {
            Issue.record("Expected .toolResult at index 3")
        }
    }

    @Test func preservesOrderWhenToolResultsAlreadyCorrect() {
        let messages: [OpenAI.ChatMessage] = [
            .init(role: .user, content: .text("Do two things")),
            .init(
                role: .assistant,
                content: .text(""),
                tool_calls: [
                    OpenAI.ToolCall(id: "call_1", type: "function",
                        function: OpenAI.FunctionCall(name: "bash", arguments: "{}")),
                    OpenAI.ToolCall(id: "call_2", type: "function",
                        function: OpenAI.FunctionCall(name: "bash", arguments: "{}")),
                ]
            ),
            .init(role: .tool, content: .text("result 1"), tool_call_id: "call_1"),
            .init(role: .tool, content: .text("result 2"), tool_call_id: "call_2"),
        ]

        let (_, converted) = MessageConverter.convertMessages(messages)

        if case .toolResult(let id, _) = converted[2] { #expect(id == "call_1") }
        else { Issue.record("Expected .toolResult at index 2") }

        if case .toolResult(let id, _) = converted[3] { #expect(id == "call_2") }
        else { Issue.record("Expected .toolResult at index 3") }
    }
}
