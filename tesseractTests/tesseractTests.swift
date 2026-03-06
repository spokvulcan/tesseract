//
//  tesseractTests.swift
//  tesseractTests
//
//  Created by Bohdan Ivanchenko on 31.01.2026.
//

import Foundation
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

@MainActor
struct ReadToolTests {

    @Test func readsNumberedTextFileContentsWithinLimits() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "Hello, world!\nLine 2\nLine 3".write(
            to: root.appendingPathComponent("test.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute("read-1", readToolArgs(path: "test.txt"), nil, nil)
        let output = result.content.textContent
        let lines = output.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

        #expect(lines.count == 3)
        #expect(lines[0].hasSuffix("\tHello, world!"))
        #expect(lines[1].hasSuffix("\tLine 2"))
        #expect(lines[2].hasSuffix("\tLine 3"))
        #expect(!output.contains("Use offset="))

        guard let details = result.details as? ReadToolDetails else {
            Issue.record("Expected ReadToolDetails")
            return
        }
        #expect(details.path == "test.txt")
        #expect(details.lineCount == 3)
        #expect(!details.wasTruncated)
        #expect(details.totalLines == 3)
    }

    @Test func showsContinuationNoticeWhenUserLimitLeavesMoreContent() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try makeLineFile(count: 100).write(
            to: root.appendingPathComponent("limit.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute(
            "read-2",
            readToolArgs(path: "limit.txt", limit: 10),
            nil,
            nil
        )
        let output = result.content.textContent

        #expect(output.contains("\tLine 1"))
        #expect(output.contains("\tLine 10"))
        #expect(!output.contains("\tLine 11"))
        #expect(output.contains("[90 more lines in file. Use offset=11 to continue.]"))
    }

    @Test func showsContinuationNoticeForOffsetAndLimit() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try makeLineFile(count: 100).write(
            to: root.appendingPathComponent("offset-limit.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute(
            "read-3",
            readToolArgs(path: "offset-limit.txt", offset: 41, limit: 20),
            nil,
            nil
        )
        let output = result.content.textContent

        #expect(!output.contains("\tLine 40"))
        #expect(output.contains("\tLine 41"))
        #expect(output.contains("\tLine 60"))
        #expect(!output.contains("\tLine 61"))
        #expect(output.contains("[40 more lines in file. Use offset=61 to continue.]"))
    }

    @Test func throwsWhenOffsetIsBeyondEndOfFile() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "Line 1\nLine 2\nLine 3".write(
            to: root.appendingPathComponent("short.txt"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await tool.execute(
                "read-4",
                readToolArgs(path: "short.txt", offset: 100),
                nil,
                nil
            )
            Issue.record("Expected read tool to throw for out-of-bounds offset")
        } catch {
            #expect(
                error.localizedDescription == "Offset 100 is beyond end of file (3 lines total)"
            )
        }
    }

    @Test func truncatesFilesThatExceedTheLineLimit() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try makeLineFile(count: 2_500).write(
            to: root.appendingPathComponent("large-lines.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute("read-5", readToolArgs(path: "large-lines.txt"), nil, nil)
        let output = result.content.textContent

        #expect(output.contains("\tLine 1"))
        #expect(output.contains("\tLine 2000"))
        #expect(!output.contains("\tLine 2001"))
        #expect(output.contains("[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]"))

        guard let details = result.details as? ReadToolDetails else {
            Issue.record("Expected ReadToolDetails")
            return
        }
        #expect(details.lineCount == 2_000)
        #expect(details.wasTruncated)
        #expect(details.totalLines == 2_500)
    }

    @Test func truncatesFilesThatExceedTheByteLimit() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        let longLines = (1...500).map { "Line \($0): " + String(repeating: "x", count: 200) }
        try longLines.joined(separator: "\n").write(
            to: root.appendingPathComponent("large-bytes.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute("read-6", readToolArgs(path: "large-bytes.txt"), nil, nil)
        let output = result.content.textContent

        #expect(output.contains("\tLine 1:"))
        #expect(output.contains("(50.0KB limit). Use offset="))

        guard let details = result.details as? ReadToolDetails else {
            Issue.record("Expected ReadToolDetails")
            return
        }
        #expect(details.wasTruncated)
    }

    @Test func truncatesOversizedSingleLinesWithANotice() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try String(repeating: "x", count: 60_000).write(
            to: root.appendingPathComponent("single-line.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute("read-7", readToolArgs(path: "single-line.txt"), nil, nil)
        let output = result.content.textContent

        #expect(output.contains("[Line 1 is "))
        #expect(output.contains("truncated to 50.0KB. Content may be incomplete.]"))
        #expect(!output.contains("Use offset="))

        guard let details = result.details as? ReadToolDetails else {
            Issue.record("Expected ReadToolDetails")
            return
        }
        #expect(details.wasTruncated)
    }

    @Test func detectsImagesByContentInsteadOfFilenameExtension() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        let imageURL = root.appendingPathComponent("image.txt")
        try Data(base64Encoded: tinyPNGBase64)!.write(to: imageURL)

        let result = try await tool.execute("read-8", readToolArgs(path: "image.txt"), nil, nil)

        #expect(result.content.textContent.contains("Read image file [image/png]"))

        let imageBlocks = result.content.compactMap { block -> String? in
            guard case .image(_, let mimeType) = block else { return nil }
            return mimeType
        }
        #expect(imageBlocks == ["image/png"])
    }

    @Test func treatsImageExtensionsWithTextContentAsText() async throws {
        let (tool, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "definitely not a png".write(
            to: root.appendingPathComponent("not-an-image.png"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute(
            "read-9",
            readToolArgs(path: "not-an-image.png"),
            nil,
            nil
        )
        let output = result.content.textContent
        let hasImage = result.content.contains { block in
            if case .image = block { return true }
            return false
        }

        #expect(output.contains("\tdefinitely not a png"))
        #expect(!hasImage)
    }
}

private let tinyPNGBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2Z0AAAAASUVORK5CYII="

private func makeReadToolTestRig() throws -> (tool: AgentToolDefinition, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-read-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    return (createReadTool(sandbox: sandbox), root)
}

private func removeReadToolTestRig(_ root: URL) {
    try? FileManager.default.removeItem(at: root)
}

private func readToolArgs(path: String, offset: Int? = nil, limit: Int? = nil) -> [String: JSONValue] {
    var args: [String: JSONValue] = ["path": .string(path)]
    if let offset {
        args["offset"] = .int(offset)
    }
    if let limit {
        args["limit"] = .int(limit)
    }
    return args
}

private func makeLineFile(count: Int) -> String {
    (1...count).map { "Line \($0)" }.joined(separator: "\n")
}
