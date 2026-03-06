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

@MainActor
struct WriteToolTests {

    @Test func writesFileContentsWithPiStyleSuccessMessage() async throws {
        let (tool, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("write-test.txt")
        let content = "Test content"

        let result = try await tool.execute(
            "write-1",
            writeToolArgs(path: "write-test.txt", content: content),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully wrote"))
        #expect(result.content.textContent.contains("write-test.txt"))
        #expect(result.details == nil)
        #expect(try readUTF8PreservingBytes(from: fileURL) == content)
    }

    @Test func createsParentDirectories() async throws {
        let (tool, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("nested/dir/test.txt")
        let content = "Nested content"

        let result = try await tool.execute(
            "write-2",
            writeToolArgs(path: "nested/dir/test.txt", content: content),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully wrote"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == content)
    }

    @Test func overwritesExistingFileContents() async throws {
        let (tool, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("overwrite.txt")
        try "Original content".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "write-3",
            writeToolArgs(path: "overwrite.txt", content: "Replacement"),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "Replacement")
    }

    @Test func reportsUtf16LengthLikePi() async throws {
        let (tool, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let content = "A😀B"
        let result = try await tool.execute(
            "write-4",
            writeToolArgs(path: "unicode.txt", content: content),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully wrote 4 bytes to unicode.txt"))
        #expect(
            try readUTF8PreservingBytes(from: root.appendingPathComponent("unicode.txt")) == content
        )
    }

    @Test func throwsOperationAbortedWhenAlreadyCancelled() async throws {
        let (tool, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let token = CancellationToken()
        token.cancel()

        do {
            _ = try await tool.execute(
                "write-5",
                writeToolArgs(path: "cancelled.txt", content: "Should not write"),
                token,
                nil
            )
            Issue.record("Expected write tool to throw when already cancelled")
        } catch {
            #expect(error.localizedDescription == "Operation aborted")
        }
    }
}

@MainActor
@Suite(.serialized)
struct EditToolTests {

    @Test func replacesTextInFile() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("edit-test.txt")
        try "Hello, world!".write(to: fileURL, atomically: true, encoding: .utf8)

        let result = try await tool.execute(
            "edit-1",
            editToolArgs(path: "edit-test.txt", oldText: "world", newText: "testing"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text in edit-test.txt."))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "Hello, testing!")

        guard let details = result.details as? EditToolDetails else {
            Issue.record("Expected EditToolDetails")
            return
        }
        #expect(details.path == "edit-test.txt")
        #expect(details.diff.contains("testing"))
        #expect(details.firstChangedLine == 1)
    }

    @Test func throwsIfTextNotFound() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "Hello, world!".write(
            to: root.appendingPathComponent("edit-test.txt"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await tool.execute(
                "edit-2",
                editToolArgs(path: "edit-test.txt", oldText: "nonexistent", newText: "testing"),
                nil,
                nil
            )
            Issue.record("Expected edit tool to throw when old_text is missing")
        } catch {
            #expect(
                error.localizedDescription
                    == "Could not find the exact text in edit-test.txt. The old text must match exactly including all whitespace and newlines."
            )
        }
    }

    @Test func throwsIfTextAppearsMultipleTimes() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "foo foo foo".write(
            to: root.appendingPathComponent("edit-test.txt"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await tool.execute(
                "edit-3",
                editToolArgs(path: "edit-test.txt", oldText: "foo", newText: "bar"),
                nil,
                nil
            )
            Issue.record("Expected edit tool to throw for duplicate matches")
        } catch {
            #expect(
                error.localizedDescription
                    == "Found 3 occurrences of the text in edit-test.txt. The text must be unique. Please provide more context to make it unique."
            )
        }
    }

    @Test func matchesTextWithTrailingWhitespaceStripped() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("trailing-ws.txt")
        try "line one   \nline two  \nline three\n".write(
            to: fileURL,
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute(
            "edit-4",
            editToolArgs(path: "trailing-ws.txt", oldText: "line one\nline two\n", newText: "replaced\n"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text in trailing-ws.txt."))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "replaced\nline three\n")
    }

    @Test func matchesSmartSingleQuotesToASCIIQuotes() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("smart-quotes.txt")
        try "console.log(‘hello’);\n".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "edit-5",
            editToolArgs(
                path: "smart-quotes.txt",
                oldText: "console.log('hello');",
                newText: "console.log('world');"
            ),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "console.log('world');\n")
    }

    @Test func matchesSmartDoubleQuotesToASCIIQuotes() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("smart-double-quotes.txt")
        try "const msg = “Hello World”;\n".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "edit-6",
            editToolArgs(
                path: "smart-double-quotes.txt",
                oldText: #"const msg = "Hello World";"#,
                newText: #"const msg = "Goodbye";"#
            ),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "const msg = \"Goodbye\";\n")
    }

    @Test func matchesUnicodeDashesToASCIIHyphen() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("unicode-dashes.txt")
        try "range: 1–5\nbreak—here\n".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "edit-7",
            editToolArgs(
                path: "unicode-dashes.txt",
                oldText: "range: 1-5\nbreak-here",
                newText: "range: 10-50\nbreak--here"
            ),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "range: 10-50\nbreak--here\n")
    }

    @Test func matchesNonBreakingSpaceToRegularSpace() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("nbsp.txt")
        try "hello\u{00A0}world\n".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "edit-8",
            editToolArgs(path: "nbsp.txt", oldText: "hello world", newText: "hello universe"),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "hello universe\n")
    }

    @Test func prefersExactMatchOverFuzzyMatch() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("exact-preferred.txt")
        try "const x = 'exact';\nconst y = 'other';\n".write(
            to: fileURL,
            atomically: true,
            encoding: .utf8
        )

        _ = try await tool.execute(
            "edit-9",
            editToolArgs(
                path: "exact-preferred.txt",
                oldText: "const x = 'exact';",
                newText: "const x = 'changed';"
            ),
            nil,
            nil
        )

        #expect(
            try readUTF8PreservingBytes(from: fileURL)
                == "const x = 'changed';\nconst y = 'other';\n"
        )
    }

    @Test func stillFailsWhenTextIsMissingAfterFuzzyMatching() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "completely different content\n".write(
            to: root.appendingPathComponent("no-match.txt"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await tool.execute(
                "edit-10",
                editToolArgs(path: "no-match.txt", oldText: "this does not exist", newText: "replacement"),
                nil,
                nil
            )
            Issue.record("Expected edit tool to throw when fuzzy matching cannot find a match")
        } catch {
            #expect(
                error.localizedDescription
                    == "Could not find the exact text in no-match.txt. The old text must match exactly including all whitespace and newlines."
            )
        }
    }

    @Test func detectsDuplicatesAfterFuzzyNormalization() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "hello world   \nhello world\n".write(
            to: root.appendingPathComponent("fuzzy-dups.txt"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await tool.execute(
                "edit-11",
                editToolArgs(path: "fuzzy-dups.txt", oldText: "hello world", newText: "replaced"),
                nil,
                nil
            )
            Issue.record("Expected edit tool to throw for fuzzy-normalized duplicates")
        } catch {
            #expect(
                error.localizedDescription
                    == "Found 2 occurrences of the text in fuzzy-dups.txt. The text must be unique. Please provide more context to make it unique."
            )
        }
    }

    @Test func matchesLFOldTextAgainstCRLFFileContent() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("crlf-test.txt")
        try Data("line one\r\nline two\r\nline three\r\n".utf8).write(to: fileURL)

        let result = try await tool.execute(
            "edit-12",
            editToolArgs(path: "crlf-test.txt", oldText: "line two\n", newText: "replaced line\n"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text in crlf-test.txt."))
    }

    @Test func preservesCRLFLineEndingsAfterEdit() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("crlf-preserve.txt")
        try Data("first\r\nsecond\r\nthird\r\n".utf8).write(to: fileURL)

        _ = try await tool.execute(
            "edit-13",
            editToolArgs(path: "crlf-preserve.txt", oldText: "second\n", newText: "REPLACED\n"),
            nil,
            nil
        )

        let actual = try Data(contentsOf: fileURL)
        let expected = Data("first\r\nREPLACED\r\nthird\r\n".utf8)
        #expect(actual == expected)
    }

    @Test func preservesLFLineEndingsForLFFiles() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("lf-preserve.txt")
        try "first\nsecond\nthird\n".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute(
            "edit-14",
            editToolArgs(path: "lf-preserve.txt", oldText: "second\n", newText: "REPLACED\n"),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "first\nREPLACED\nthird\n")
    }

    @Test func detectsDuplicatesAcrossCRLFAndLFVariants() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        try Data("hello\r\nworld\r\n---\r\nhello\nworld\n".utf8).write(
            to: root.appendingPathComponent("mixed-endings.txt")
        )

        do {
            _ = try await tool.execute(
                "edit-15",
                editToolArgs(path: "mixed-endings.txt", oldText: "hello\nworld\n", newText: "replaced\n"),
                nil,
                nil
            )
            Issue.record("Expected edit tool to throw for duplicate mixed line-ending matches")
        } catch {
            #expect(
                error.localizedDescription
                    == "Found 2 occurrences of the text in mixed-endings.txt. The text must be unique. Please provide more context to make it unique."
            )
        }
    }

    @Test func preservesUTF8BOMAfterEdit() async throws {
        let (tool, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("bom-test.txt")
        let initialData = Data([0xEF, 0xBB, 0xBF]) + Data("first\r\nsecond\r\nthird\r\n".utf8)
        try initialData.write(to: fileURL)

        _ = try await tool.execute(
            "edit-16",
            editToolArgs(path: "bom-test.txt", oldText: "second\n", newText: "REPLACED\n"),
            nil,
            nil
        )

        let actual = try Data(contentsOf: fileURL)
        let expected = Data([0xEF, 0xBB, 0xBF]) + Data("first\r\nREPLACED\r\nthird\r\n".utf8)
        #expect(actual == expected)
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

private func makeEditToolTestRig() throws -> (tool: AgentToolDefinition, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-edit-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    return (createEditTool(sandbox: sandbox), root)
}

private func makeWriteToolTestRig() throws -> (tool: AgentToolDefinition, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-write-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    return (createWriteTool(sandbox: sandbox), root)
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

private func editToolArgs(path: String, oldText: String, newText: String) -> [String: JSONValue] {
    [
        "path": .string(path),
        "old_text": .string(oldText),
        "new_text": .string(newText),
    ]
}

private func writeToolArgs(path: String, content: String) -> [String: JSONValue] {
    [
        "path": .string(path),
        "content": .string(content),
    ]
}

private func readUTF8PreservingBytes(from url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    return String(decoding: data, as: UTF8.self)
}

private func makeLineFile(count: Int) -> String {
    (1...count).map { "Line \($0)" }.joined(separator: "\n")
}
