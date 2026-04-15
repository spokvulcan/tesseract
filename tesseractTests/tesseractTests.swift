//
//  tesseractTests.swift
//  tesseractTests
//
//  Created by Bohdan Ivanchenko on 31.01.2026.
//

import Foundation
import Testing
import MLXLMCommon
@testable import Tesseract_Agent

@MainActor
struct ModelDefinitionCatalogTests {

    @Test func includesQwen35_27BParoInAgentCatalog() async throws {
        guard let model = ModelDefinition.all.first(where: { $0.id == "qwen3.5-27b-paro" }) else {
            Issue.record("Missing qwen3.5-27b-paro model definition")
            return
        }

        #expect(model.displayName == "Qwen3.5-27B PARO")
        #expect(model.category == .agent)
        #expect(model.repoID == "z-lab/Qwen3.5-27B-PARO")
        #expect(model.requiredExtension == "safetensors")
        #expect(
            ModelDefinition.byCategory()
                .first(where: { $0.0 == .agent })?
                .1
                .contains(where: { $0.id == model.id }) == true
        )
    }

    @Test func recognizes27BParoCheckpoints() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let config = """
        {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "quantization_config": {
            "quant_method": "paroquant"
          }
        }
        """
        try config.write(to: root.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        #expect(isParoQuantModel(directory: root))
    }
}

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
        let rawToolCall = ToolCallInfo(
            id: "call-1",
            name: "list",
            argumentsJSON: ToolArgumentNormalizer.encode([
                "recursive": .string(#"string("True")"#),
                "path": .string(#"string("tasks.md")"#),
            ])
        )
        let rawMessage = AssistantMessage(content: "", toolCalls: [rawToolCall])

        let normalized = AgentChatMessage(from: rawMessage)
        let args = normalized.toolCalls[0].function.arguments

        #expect(args["recursive"] == JSONValue.string("True"))
        #expect(args["path"] == JSONValue.string("tasks.md"))
    }
}

@MainActor
struct MessageConversionTests {

    @Test func reconstructsAssistantToolCallsUsingXMLFunctionFormat() async throws {
        let toolCalls = [
            ToolCallInfo(
                id: "call-1",
                name: "read",
                argumentsJSON: #"{"path":"notes/todo.md","offset":42,"recursive":true,"metadata":{"kind":"text","priority":1},"lines":[1,2],"fallback":null}"#
            )
        ]

        let messages = toLLMCommonMessages([
            .assistant(content: "Opening file.", toolCalls: toolCalls)
        ])

        #expect(messages.count == 1)
        #expect(messages[0].role == .assistant)
        let content = messages[0].content

        #expect(content.hasPrefix("Opening file.\n<tool_call>\n<function=read>\n"))
        #expect(content.contains("<parameter=path>\nnotes/todo.md\n</parameter>"))
        #expect(content.contains("<parameter=offset>\n42\n</parameter>"))
        #expect(content.contains("<parameter=recursive>\nTrue\n</parameter>"))
        #expect(content.contains(#"""
<parameter=metadata>
{"kind":"text","priority":1}
</parameter>
"""#))
        #expect(content.contains(#"""
<parameter=lines>
[1,2]
</parameter>
"""#))
        #expect(content.contains("<parameter=fallback>\nNone\n</parameter>"))
        #expect(content.hasSuffix("</function>\n</tool_call>"))
    }

    @Test func omitsParametersWhenToolCallArgumentsAreMalformed() async throws {
        let toolCalls = [
            ToolCallInfo(
                id: "call-2",
                name: "write",
                argumentsJSON: "{not json"
            )
        ]

        let messages = toLLMCommonMessages([
            .assistant(content: "", toolCalls: toolCalls)
        ])

        #expect(messages.count == 1)
        #expect(messages[0].role == .assistant)
        let content = messages[0].content

        #expect(content == "\n<tool_call>\n<function=write>\n</function>\n</tool_call>")
    }
}

@MainActor
struct ReadToolTests {

    @Test func readsNumberedTextFileContentsWithinLimits() async throws {
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
        let (tool, _, root) = try makeReadToolTestRig()
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
struct LsToolTests {

    @Test func listsDotfilesAndDirectoriesLikePi() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "secret".write(
            to: root.appendingPathComponent(".hidden-file"),
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(".hidden-dir"),
            withIntermediateDirectories: true
        )
        try "A".write(
            to: root.appendingPathComponent("Alpha.txt"),
            atomically: true,
            encoding: .utf8
        )
        try "Z".write(
            to: root.appendingPathComponent("zeta.txt"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await tool.execute("ls-1", lsToolArgs(path: "."), nil, nil)
        let lines = result.content.textContent.split(separator: "\n").map(String.init)

        #expect(lines == [".hidden-dir/", ".hidden-file", "Alpha.txt", "zeta.txt"])
        #expect(result.details == nil)
    }

    @Test func returnsEmptyDirectoryNotice() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        let result = try await tool.execute("ls-2", [:], nil, nil)

        #expect(result.content.textContent == "(empty directory)")
        #expect(result.details == nil)
    }

    @Test func throwsWhenPathDoesNotExist() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        do {
            _ = try await tool.execute("ls-3", lsToolArgs(path: "missing"), nil, nil)
            Issue.record("Expected ls tool to throw for a missing path")
        } catch {
            #expect(error.localizedDescription == "Path not found: \(root.appendingPathComponent("missing").path)")
        }
    }

    @Test func throwsWhenPathIsNotADirectory() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("file.txt")
        try "hello".write(to: fileURL, atomically: true, encoding: .utf8)

        do {
            _ = try await tool.execute("ls-4", lsToolArgs(path: "file.txt"), nil, nil)
            Issue.record("Expected ls tool to throw for non-directory paths")
        } catch {
            #expect(error.localizedDescription == "Not a directory: \(fileURL.path)")
        }
    }

    @Test func showsEntryLimitNoticeLikePi() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        try "a".write(to: root.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try "b".write(to: root.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
        try "c".write(to: root.appendingPathComponent("c.txt"), atomically: true, encoding: .utf8)

        let result = try await tool.execute("ls-5", lsToolArgs(path: ".", limit: 2), nil, nil)

        #expect(
            result.content.textContent
                == "a.txt\nb.txt\n\n[2 entries limit reached. Use limit=4 for more]"
        )

        guard let details = result.details as? LsToolDetails else {
            Issue.record("Expected LsToolDetails")
            return
        }
        #expect(details.entryLimitReached == 2)
        #expect(details.truncation == nil)
    }

    @Test func showsByteLimitNoticeLikePi() async throws {
        let (tool, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        for index in 0..<300 {
            let name = String(repeating: "a", count: 190) + String(format: "-%03d.txt", index)
            try "x".write(
                to: root.appendingPathComponent(name),
                atomically: true,
                encoding: .utf8
            )
        }

        let result = try await tool.execute("ls-6", lsToolArgs(path: "."), nil, nil)

        #expect(result.content.textContent.contains("[50.0KB limit reached]"))

        guard let details = result.details as? LsToolDetails else {
            Issue.record("Expected LsToolDetails")
            return
        }
        #expect(details.entryLimitReached == nil)
        #expect(details.truncation?.truncated == true)
        #expect(details.truncation?.truncatedBy == "bytes")
    }

    @Test func builtInToolsExposeLsInsteadOfList() async throws {
        let (_, root) = try makeLsToolTestRig()
        defer { removeReadToolTestRig(root) }

        let sandbox = PathSandbox(root: root)
        let names = BuiltInToolFactory.createAll(sandbox: sandbox).map(\.name)

        #expect(names.contains("ls"))
        #expect(!names.contains("list"))
    }
}

@MainActor
struct WriteToolTests {

    @Test func writesFileContentsWithPiStyleSuccessMessage() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
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
        let (tool, _, root) = try makeWriteToolTestRig()
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
        let (tool, tracker, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("overwrite.txt")
        try "Original content".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

        _ = try await tool.execute(
            "write-3",
            writeToolArgs(path: "overwrite.txt", content: "Replacement", overwrite: true),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "Replacement")
    }

    @Test func appendsToExistingFileByDefault() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("append-test.txt")
        try "Hello".write(to: fileURL, atomically: true, encoding: .utf8)

        let result = try await tool.execute(
            "write-6",
            writeToolArgs(path: "append-test.txt", content: ", World!"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully appended"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "Hello, World!")
    }

    @Test func reportsUtf16LengthLikePi() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
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
        let (tool, _, root) = try makeWriteToolTestRig()
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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("edit-test.txt")
        try "Hello, world!".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let editTestURL = root.appendingPathComponent("edit-test.txt")
        try "Hello, world!".write(to: editTestURL, atomically: true, encoding: .utf8)
        tracker.record(editTestURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let editTestURL = root.appendingPathComponent("edit-test.txt")
        try "foo foo foo".write(to: editTestURL, atomically: true, encoding: .utf8)
        tracker.record(editTestURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("trailing-ws.txt")
        try "line one   \nline two  \nline three\n".write(
            to: fileURL,
            atomically: true,
            encoding: .utf8
        )
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("smart-quotes.txt")
        try "console.log(‘hello’);\n".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("smart-double-quotes.txt")
        try "const msg = “Hello World”;\n".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("unicode-dashes.txt")
        try "range: 1–5\nbreak—here\n".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("nbsp.txt")
        try "hello\u{00A0}world\n".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

        _ = try await tool.execute(
            "edit-8",
            editToolArgs(path: "nbsp.txt", oldText: "hello world", newText: "hello universe"),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "hello universe\n")
    }

    @Test func prefersExactMatchOverFuzzyMatch() async throws {
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("exact-preferred.txt")
        try "const x = 'exact';\nconst y = 'other';\n".write(
            to: fileURL,
            atomically: true,
            encoding: .utf8
        )
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let noMatchURL = root.appendingPathComponent("no-match.txt")
        try "completely different content\n".write(to: noMatchURL, atomically: true, encoding: .utf8)
        tracker.record(noMatchURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fuzzyDupsURL = root.appendingPathComponent("fuzzy-dups.txt")
        try "hello world   \nhello world\n".write(to: fuzzyDupsURL, atomically: true, encoding: .utf8)
        tracker.record(fuzzyDupsURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("crlf-test.txt")
        try Data("line one\r\nline two\r\nline three\r\n".utf8).write(to: fileURL)
        tracker.record(fileURL.path)

        let result = try await tool.execute(
            "edit-12",
            editToolArgs(path: "crlf-test.txt", oldText: "line two\n", newText: "replaced line\n"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text in crlf-test.txt."))
    }

    @Test func preservesCRLFLineEndingsAfterEdit() async throws {
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("crlf-preserve.txt")
        try Data("first\r\nsecond\r\nthird\r\n".utf8).write(to: fileURL)
        tracker.record(fileURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("lf-preserve.txt")
        try "first\nsecond\nthird\n".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

        _ = try await tool.execute(
            "edit-14",
            editToolArgs(path: "lf-preserve.txt", oldText: "second\n", newText: "REPLACED\n"),
            nil,
            nil
        )

        #expect(try readUTF8PreservingBytes(from: fileURL) == "first\nREPLACED\nthird\n")
    }

    @Test func detectsDuplicatesAcrossCRLFAndLFVariants() async throws {
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let mixedURL = root.appendingPathComponent("mixed-endings.txt")
        try Data("hello\r\nworld\r\n---\r\nhello\nworld\n".utf8).write(to: mixedURL)
        tracker.record(mixedURL.path)

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
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("bom-test.txt")
        let initialData = Data([0xEF, 0xBB, 0xBF]) + Data("first\r\nsecond\r\nthird\r\n".utf8)
        try initialData.write(to: fileURL)
        tracker.record(fileURL.path)

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

private func makeReadToolTestRig() throws -> (tool: AgentToolDefinition, tracker: FileReadTracker, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-read-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    let tracker = FileReadTracker()
    return (createReadTool(sandbox: sandbox, readTracker: tracker), tracker, root)
}

private func makeLsToolTestRig() throws -> (tool: AgentToolDefinition, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-ls-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    return (createLsTool(sandbox: sandbox), root)
}

private func removeReadToolTestRig(_ root: URL) {
    try? FileManager.default.removeItem(at: root)
}

private func makeEditToolTestRig() throws -> (tool: AgentToolDefinition, tracker: FileReadTracker, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-edit-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    let tracker = FileReadTracker()
    return (createEditTool(sandbox: sandbox, readTracker: tracker), tracker, root)
}

private func makeWriteToolTestRig() throws -> (tool: AgentToolDefinition, tracker: FileReadTracker, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-write-tool-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sandbox = PathSandbox(root: root)
    let tracker = FileReadTracker()
    return (createWriteTool(sandbox: sandbox, readTracker: tracker), tracker, root)
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

private func writeToolArgs(path: String, content: String, overwrite: Bool? = nil) -> [String: JSONValue] {
    var args: [String: JSONValue] = [
        "path": .string(path),
        "content": .string(content),
    ]
    if let overwrite {
        args["overwrite"] = .bool(overwrite)
    }
    return args
}

private func lsToolArgs(path: String? = nil, limit: Int? = nil) -> [String: JSONValue] {
    var args: [String: JSONValue] = [:]
    if let path {
        args["path"] = .string(path)
    }
    if let limit {
        args["limit"] = .int(limit)
    }
    return args
}

private func readUTF8PreservingBytes(from url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    return String(decoding: data, as: UTF8.self)
}

private func makeLineFile(count: Int) -> String {
    (1...count).map { "Line \($0)" }.joined(separator: "\n")
}

// MARK: - FileReadTracker Tests

@MainActor
struct FileReadTrackerTests {

    @Test func recordAndHasRead() {
        let tracker = FileReadTracker()
        tracker.record("/tmp/a.txt")
        #expect(tracker.hasRead("/tmp/a.txt"))
    }

    @Test func hasReadReturnsFalseForUnrecorded() {
        let tracker = FileReadTracker()
        #expect(!tracker.hasRead("/tmp/x.txt"))
    }

    @Test func recordIsIdempotent() {
        let tracker = FileReadTracker()
        tracker.record("/tmp/a.txt")
        tracker.record("/tmp/a.txt")
        #expect(tracker.hasRead("/tmp/a.txt"))
    }

    @Test func differentPathsAreIndependent() {
        let tracker = FileReadTracker()
        tracker.record("/tmp/a.txt")
        #expect(!tracker.hasRead("/tmp/b.txt"))
    }

    @Test func pathNormalizationMatters() {
        let tracker = FileReadTracker()
        tracker.record("/tmp/a")
        #expect(!tracker.hasRead("/tmp/./a"))
    }
}

// MARK: - Write Tool Read Guard Tests

@MainActor
struct WriteToolReadGuardTests {

    @Test func writeNewFileWithoutReadSucceeds() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let result = try await tool.execute(
            "wg-1",
            writeToolArgs(path: "brand-new.txt", content: "hello"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully wrote"))
    }

    @Test func appendExistingFileWithoutReadSucceeds() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("existing.txt")
        try "original".write(to: fileURL, atomically: true, encoding: .utf8)

        let result = try await tool.execute(
            "wg-2",
            writeToolArgs(path: "existing.txt", content: " appended"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully appended"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "original appended")
    }

    @Test func overwriteExistingFileWithoutReadReturnsError() async throws {
        let (tool, _, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("existing.txt")
        try "original".write(to: fileURL, atomically: true, encoding: .utf8)

        let result = try await tool.execute(
            "wg-2b",
            writeToolArgs(path: "existing.txt", content: "overwrite", overwrite: true),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("must read"))
    }

    @Test func overwriteExistingFileAfterReadSucceeds() async throws {
        let (tool, tracker, root) = try makeWriteToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("existing.txt")
        try "original".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

        let result = try await tool.execute(
            "wg-3",
            writeToolArgs(path: "existing.txt", content: "overwrite", overwrite: true),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully wrote"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "overwrite")
    }
}

// MARK: - Edit Tool Read Guard Tests

@MainActor
struct EditToolReadGuardTests {

    @Test func editWithoutReadReturnsError() async throws {
        let (tool, _, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("guarded.txt")
        try "Hello, world!".write(to: fileURL, atomically: true, encoding: .utf8)

        let result = try await tool.execute(
            "eg-1",
            editToolArgs(path: "guarded.txt", oldText: "world", newText: "testing"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("must read"))
    }

    @Test func editAfterReadSucceeds() async throws {
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("guarded.txt")
        try "Hello, world!".write(to: fileURL, atomically: true, encoding: .utf8)
        tracker.record(fileURL.path)

        let result = try await tool.execute(
            "eg-2",
            editToolArgs(path: "guarded.txt", oldText: "world", newText: "testing"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "Hello, testing!")
    }

    @Test func editDifferentPathAfterReadReturnsError() async throws {
        let (tool, tracker, root) = try makeEditToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileA = root.appendingPathComponent("a.txt")
        let fileB = root.appendingPathComponent("b.txt")
        try "content A".write(to: fileA, atomically: true, encoding: .utf8)
        try "content B".write(to: fileB, atomically: true, encoding: .utf8)
        tracker.record(fileA.path)

        let result = try await tool.execute(
            "eg-3",
            editToolArgs(path: "b.txt", oldText: "content B", newText: "changed"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("must read"))
    }
}

// MARK: - Read Tool Tracker Integration Tests

@MainActor
struct ReadToolTrackerTests {

    @Test func readRecordsPathInTracker() async throws {
        let (tool, tracker, root) = try makeReadToolTestRig()
        defer { removeReadToolTestRig(root) }

        let fileURL = root.appendingPathComponent("tracked.txt")
        try "content".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await tool.execute("rt-1", readToolArgs(path: "tracked.txt"), nil, nil)

        #expect(tracker.hasRead(fileURL.path))
    }

    @Test func readThenEditIntegration() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tesseract-integration-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let sandbox = PathSandbox(root: root)
        let tracker = FileReadTracker()
        let readTool = createReadTool(sandbox: sandbox, readTracker: tracker)
        let editTool = createEditTool(sandbox: sandbox, readTracker: tracker)

        let fileURL = root.appendingPathComponent("flow.txt")
        try "Hello, world!".write(to: fileURL, atomically: true, encoding: .utf8)

        _ = try await readTool.execute("rt-2a", readToolArgs(path: "flow.txt"), nil, nil)

        let result = try await editTool.execute(
            "rt-2b",
            editToolArgs(path: "flow.txt", oldText: "world", newText: "flow"),
            nil,
            nil
        )

        #expect(result.content.textContent.contains("Successfully replaced text"))
        #expect(try readUTF8PreservingBytes(from: fileURL) == "Hello, flow!")
    }
}

// MARK: - Cron Test Helpers

private let cronTestTimeZone = TimeZone(identifier: "America/New_York")!

private func cronDate(_ str: String) -> Date {
    let fmt = DateFormatter()
    fmt.dateFormat = "yyyy-MM-dd HH:mm"
    fmt.timeZone = cronTestTimeZone
    return fmt.date(from: str)!
}

private func cronComponents(_ date: Date) -> DateComponents {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = cronTestTimeZone
    return cal.dateComponents([.year, .month, .day, .hour, .minute], from: date)
}

// MARK: - CronField Parsing Tests

@MainActor
struct CronFieldParsingTests {

    @Test func parsesAnyWildcard() throws {
        let field = try CronField.parse("*", validRange: 0...59, position: 0)
        #expect(field == .any)
    }

    @Test func parsesSingleValue() throws {
        let field = try CronField.parse("5", validRange: 0...59, position: 0)
        #expect(field == .value(5))
    }

    @Test func parsesRange() throws {
        let field = try CronField.parse("1-5", validRange: 0...59, position: 0)
        #expect(field == .range(1, 5))
    }

    @Test func parsesStepWithWildcard() throws {
        let field = try CronField.parse("*/15", validRange: 0...59, position: 0)
        #expect(field == .step(base: .any, 15))
    }

    @Test func parsesStepWithRange() throws {
        let field = try CronField.parse("1-30/5", validRange: 0...59, position: 0)
        #expect(field == .step(base: .range(1, 30), 5))
    }

    @Test func parsesList() throws {
        let field = try CronField.parse("1,15,30", validRange: 0...59, position: 0)
        #expect(field == .list([.value(1), .value(15), .value(30)]))
    }

    @Test func parsesListWithRanges() throws {
        let field = try CronField.parse("1-5,10,20-25", validRange: 0...59, position: 0)
        #expect(field == .list([.range(1, 5), .value(10), .range(20, 25)]))
    }

    @Test func normalizesDayOfWeek7To0() throws {
        let field = try CronField.parse("7", validRange: 0...6, position: 4)
        #expect(field == .value(0))
    }

    @Test func normalizesDayOfWeek7InRange() throws {
        let field = try CronField.parse("5-7", validRange: 0...6, position: 4)
        #expect(field == .range(5, 0))
    }

    @Test func wrapAroundRangeStepExpandsCorrectly() throws {
        // 5-7/1 with dow normalization: 5-0, step 1 → should expand to [5, 6, 0]
        let field = try CronField.parse("5-7/1", validRange: 0...6, position: 4)
        let expanded = field.expandedValues(in: 0...6)
        #expect(expanded == [5, 6, 0])
        // Verify matches works for each day
        #expect(field.matches(5, in: 0...6))  // Friday
        #expect(field.matches(6, in: 0...6))  // Saturday
        #expect(field.matches(0, in: 0...6))  // Sunday
        #expect(!field.matches(1, in: 0...6)) // Monday
    }

    @Test func wrapAroundRangeStepWithStep2() throws {
        // 5-7/2 with dow normalization: 5-0, step 2 → should expand to [5, 0]
        let field = try CronField.parse("5-7/2", validRange: 0...6, position: 4)
        let expanded = field.expandedValues(in: 0...6)
        #expect(expanded == [5, 0])
        #expect(field.matches(5, in: 0...6))   // Friday
        #expect(!field.matches(6, in: 0...6))  // Saturday (skipped)
        #expect(field.matches(0, in: 0...6))   // Sunday
    }

    @Test func throwsForOutOfRangeValue() throws {
        #expect(throws: CronError.self) {
            try CronField.parse("60", validRange: 0...59, position: 0)
        }
    }

    @Test func throwsForNegativeValue() throws {
        #expect(throws: CronError.self) {
            try CronField.parse("-1", validRange: 0...59, position: 0)
        }
    }

    @Test func throwsForZeroStep() throws {
        #expect(throws: CronError.self) {
            try CronField.parse("*/0", validRange: 0...59, position: 0)
        }
    }

    @Test func throwsForNonNumericValue() throws {
        #expect(throws: CronError.self) {
            try CronField.parse("abc", validRange: 0...59, position: 0)
        }
    }

    @Test func throwsForDescendingRangeOnNonDow() throws {
        // 17-9 is invalid for hour (position 1) — descending ranges only allowed for dow
        #expect(throws: CronError.self) {
            try CronField.parse("17-9", validRange: 0...23, position: 1)
        }
        // Also invalid for minute
        #expect(throws: CronError.self) {
            try CronField.parse("45-10", validRange: 0...59, position: 0)
        }
        // Full expression check
        #expect(throws: CronError.self) {
            try CronExpression(parsing: "0 17-9 * * *")
        }
    }

    @Test func allowsDescendingRangeOnDow() throws {
        // 5-0 is valid for dow (position 4) — wrap-around from Fri to Sun
        let field = try CronField.parse("5-0", validRange: 0...6, position: 4)
        #expect(field == .range(5, 0))
    }

    @Test func throwsForTooFewFields() throws {
        #expect(throws: CronError.self) {
            try CronExpression(parsing: "* * *")
        }
    }

    @Test func throwsForTooManyFields() throws {
        #expect(throws: CronError.self) {
            try CronExpression(parsing: "* * * * * *")
        }
    }
}

// MARK: - CronExpression Matching Tests

@MainActor
struct CronExpressionMatchingTests {

    @Test func matchesExactTime() throws {
        let cron = try CronExpression(parsing: "30 9 * * *")
        #expect(cron.matches(cronDate("2026-03-19 09:30"), in: cronTestTimeZone))
        #expect(!cron.matches(cronDate("2026-03-19 09:31"), in: cronTestTimeZone))
    }

    @Test func matchesWildcard() throws {
        let cron = try CronExpression(parsing: "* * * * *")
        #expect(cron.matches(cronDate("2026-03-19 14:22"), in: cronTestTimeZone))
    }

    @Test func matchesRange() throws {
        let cron = try CronExpression(parsing: "0 9-17 * * *")
        #expect(cron.matches(cronDate("2026-03-19 09:00"), in: cronTestTimeZone))
        #expect(cron.matches(cronDate("2026-03-19 17:00"), in: cronTestTimeZone))
        #expect(!cron.matches(cronDate("2026-03-19 18:00"), in: cronTestTimeZone))
    }

    @Test func matchesStep() throws {
        let cron = try CronExpression(parsing: "*/15 * * * *")
        #expect(cron.matches(cronDate("2026-03-19 09:00"), in: cronTestTimeZone))
        #expect(cron.matches(cronDate("2026-03-19 09:15"), in: cronTestTimeZone))
        #expect(cron.matches(cronDate("2026-03-19 09:30"), in: cronTestTimeZone))
        #expect(!cron.matches(cronDate("2026-03-19 09:07"), in: cronTestTimeZone))
    }

    @Test func matchesList() throws {
        let cron = try CronExpression(parsing: "0,30 * * * *")
        #expect(cron.matches(cronDate("2026-03-19 09:00"), in: cronTestTimeZone))
        #expect(cron.matches(cronDate("2026-03-19 09:30"), in: cronTestTimeZone))
        #expect(!cron.matches(cronDate("2026-03-19 09:15"), in: cronTestTimeZone))
    }

    @Test func dayOrSemanticsBothConstrained() throws {
        // dom=15 and dow=1 (Monday) — matches if either is true
        let cron = try CronExpression(parsing: "0 9 15 * 1")
        // 2026-03-15 is a Sunday — dom matches
        #expect(cron.matches(cronDate("2026-03-15 09:00"), in: cronTestTimeZone))
        // 2026-03-16 is a Monday — dow matches
        #expect(cron.matches(cronDate("2026-03-16 09:00"), in: cronTestTimeZone))
        // 2026-03-17 is a Tuesday, day 17 — neither matches
        #expect(!cron.matches(cronDate("2026-03-17 09:00"), in: cronTestTimeZone))
    }

    @Test func dayAndSemanticsOneWildcard() throws {
        // Only dow constrained (dom is *) — must match dow
        let cron = try CronExpression(parsing: "0 9 * * 1")
        // 2026-03-16 is a Monday
        #expect(cron.matches(cronDate("2026-03-16 09:00"), in: cronTestTimeZone))
        #expect(!cron.matches(cronDate("2026-03-17 09:00"), in: cronTestTimeZone))
    }

    @Test func matchesWeekdayRange() throws {
        let cron = try CronExpression(parsing: "0 17 * * 1-5")
        // 2026-03-19 is a Thursday (weekday)
        #expect(cron.matches(cronDate("2026-03-19 17:00"), in: cronTestTimeZone))
        // 2026-03-22 is a Sunday
        #expect(!cron.matches(cronDate("2026-03-22 17:00"), in: cronTestTimeZone))
    }
}

// MARK: - CronExpression Next Occurrence Tests

@MainActor
struct CronExpressionNextOccurrenceTests {

    @Test func nextMinute() throws {
        let cron = try CronExpression(parsing: "* * * * *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-19 09:30"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.hour == 9)
        #expect(c.minute == 31)
    }

    @Test func nextSpecificTime() throws {
        let cron = try CronExpression(parsing: "0 9 * * *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-19 09:30"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.day == 20)
        #expect(c.hour == 9)
        #expect(c.minute == 0)
    }

    @Test func nextWeekday() throws {
        let cron = try CronExpression(parsing: "0 9 * * 1-5")
        // 2026-03-20 is Friday. Next weekday is Monday 2026-03-23.
        let next = cron.nextOccurrence(after: cronDate("2026-03-20 09:30"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.month == 3)
        #expect(c.day == 23)
        #expect(c.hour == 9)
        #expect(c.minute == 0)
    }

    @Test func every15Minutes() throws {
        let cron = try CronExpression(parsing: "*/15 * * * *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-19 09:07"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.hour == 9)
        #expect(c.minute == 15)
    }

    @Test func monthBoundary() throws {
        let cron = try CronExpression(parsing: "0 9 * * *")
        // March 31 at 10AM — next day is April 1
        let next = cron.nextOccurrence(after: cronDate("2026-03-31 10:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.month == 4)
        #expect(c.day == 1)
        #expect(c.hour == 9)
    }

    @Test func leapYearFeb29() throws {
        let cron = try CronExpression(parsing: "0 0 29 2 *")
        // 2028 is a leap year
        let next = cron.nextOccurrence(after: cronDate("2026-03-01 00:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.year == 2028)
        #expect(c.month == 2)
        #expect(c.day == 29)
    }

    @Test func nonLeapYearSkipsFeb29() throws {
        let cron = try CronExpression(parsing: "0 0 29 2 *")
        // 2027 is not a leap year, should skip to 2028
        let next = cron.nextOccurrence(after: cronDate("2027-01-01 00:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.year == 2028)
        #expect(c.month == 2)
        #expect(c.day == 29)
    }

    @Test func yearBoundaryJan1() throws {
        let cron = try CronExpression(parsing: "0 0 1 1 *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-19 00:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.year == 2027)
        #expect(c.month == 1)
        #expect(c.day == 1)
    }

    @Test func lastMinuteOfYear() throws {
        let cron = try CronExpression(parsing: "59 23 31 12 *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-19 00:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.year == 2026)
        #expect(c.month == 12)
        #expect(c.day == 31)
        #expect(c.hour == 23)
        #expect(c.minute == 59)
    }

    @Test func dayOrSemanticsFindsEarlier() throws {
        // dom=15, dow=1 (Monday) — whichever comes first
        let cron = try CronExpression(parsing: "0 9 15 * 1")
        // After 2026-03-14 (Saturday): Monday 2026-03-16 or dom 2026-03-15 (Sunday)
        // 15th comes first
        let next = cron.nextOccurrence(after: cronDate("2026-03-14 10:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.day == 15)
        #expect(c.hour == 9)
    }
}

// MARK: - CronExpression DST Tests

@MainActor
struct CronExpressionDSTTests {

    @Test func springForwardSkips2AM() throws {
        // 2026-03-08 is US spring-forward: 2:00 AM doesn't exist
        let cron = try CronExpression(parsing: "0 2 8 3 *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-07 12:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        // Should resolve to 3:00 AM (the next valid time after the gap)
        #expect(c.month == 3)
        #expect(c.day == 8)
        #expect(c.hour == 3)
        #expect(c.minute == 0)
    }

    @Test func springForwardNonzeroMinuteResolvesToGapEdge() throws {
        // 2026-03-08 spring-forward: "30 2 8 3 *" means 2:30 AM which doesn't exist.
        // Spec: resolve to first valid wall-clock time (3:00 AM), not shifted 3:30 AM.
        let cron = try CronExpression(parsing: "30 2 8 3 *")
        let next = cron.nextOccurrence(after: cronDate("2026-03-07 12:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        #expect(c.month == 3)
        #expect(c.day == 8)
        #expect(c.hour == 3)
        #expect(c.minute == 0)
    }

    @Test func fallBackFirstOccurrence() throws {
        // 2026-11-01 is US fall-back: 1:30 AM occurs twice
        let cron = try CronExpression(parsing: "30 1 * * *")
        let next = cron.nextOccurrence(after: cronDate("2026-10-31 23:00"), in: cronTestTimeZone)!
        let c = cronComponents(next)
        // Calendar returns first occurrence (EDT) by default
        #expect(c.month == 11)
        #expect(c.day == 1)
        #expect(c.hour == 1)
        #expect(c.minute == 30)
    }
}

// MARK: - CronField Codable Tests

@MainActor
struct CronFieldCodableTests {

    @Test func roundTripsAllFieldCases() throws {
        let cases: [CronField] = [
            .any,
            .value(5),
            .range(1, 5),
            .step(base: .any, 15),
            .step(base: .range(1, 30), 5),
            .list([.value(1), .value(15), .range(20, 25)]),
        ]

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for field in cases {
            let data = try encoder.encode(field)
            let decoded = try decoder.decode(CronField.self, from: data)
            #expect(decoded == field)
        }
    }

    @Test func roundTripsFullExpression() throws {
        let expr = try CronExpression(parsing: "*/15 9-17 1,15 * 1-5")
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let data = try encoder.encode(expr)
        let decoded = try decoder.decode(CronExpression.self, from: data)
        #expect(decoded == expr)
    }

    @Test func producesDiscriminatedUnionJSON() throws {
        let field = CronField.step(base: .any, 15)
        let data = try JSONEncoder().encode(field)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(json["type"] as? String == "step")
        #expect(json["step"] as? Int == 15)
        let base = json["base"] as? [String: Any]
        #expect(base?["type"] as? String == "any")
    }
}

// MARK: - CronExpression Human Readable Tests

@MainActor
struct CronExpressionHumanReadableTests {

    @Test func everyMinute() throws {
        let cron = try CronExpression(parsing: "* * * * *")
        #expect(cron.humanReadable == "Every minute")
    }

    @Test func everyDayAtTime() throws {
        let cron = try CronExpression(parsing: "0 9 * * *")
        #expect(cron.humanReadable == "Every day at 9:00 AM")
    }

    @Test func every15Minutes() throws {
        let cron = try CronExpression(parsing: "*/15 * * * *")
        #expect(cron.humanReadable == "Every 15 minutes")
    }

    @Test func weekdaysAtTime() throws {
        let cron = try CronExpression(parsing: "0 17 * * 1-5")
        #expect(cron.humanReadable == "Weekdays at 5:00 PM")
    }

    @Test func expressionRoundTrip() throws {
        let expressions = ["* * * * *", "0 9 * * *", "*/15 * * * *", "0 17 * * 1-5", "0,30 9-17 1,15 * *"]
        for expr in expressions {
            let cron = try CronExpression(parsing: expr)
            #expect(cron.expression == expr)
        }
    }
}

// MARK: - TaskCreator Codable Tests

@MainActor
struct TaskCreatorCodableTests {

    @Test func roundTripsUserCase() throws {
        let creator = TaskCreator.user
        let data = try JSONEncoder().encode(creator)
        let decoded = try JSONDecoder().decode(TaskCreator.self, from: data)
        #expect(decoded == creator)
    }

    @Test func roundTripsAgentCase() throws {
        let creator = TaskCreator.agent(reason: "Detected recurring pattern")
        let data = try JSONEncoder().encode(creator)
        let decoded = try JSONDecoder().decode(TaskCreator.self, from: data)
        #expect(decoded == creator)
    }

    @Test func producesDiscriminatedUnionJSON() throws {
        let user = TaskCreator.user
        let userData = try JSONEncoder().encode(user)
        let userJSON = try JSONSerialization.jsonObject(with: userData) as! [String: Any]
        #expect(userJSON["type"] as? String == "user")
        #expect(userJSON["reason"] == nil)

        let agent = TaskCreator.agent(reason: "test reason")
        let agentData = try JSONEncoder().encode(agent)
        let agentJSON = try JSONSerialization.jsonObject(with: agentData) as! [String: Any]
        #expect(agentJSON["type"] as? String == "agent")
        #expect(agentJSON["reason"] as? String == "test reason")
    }
}

// MARK: - TaskRunResult Codable Tests

@MainActor
struct TaskRunResultCodableTests {

    @Test func roundTripsAllCases() throws {
        let cases: [TaskRunResult] = [
            .success(summary: "Completed check"),
            .noActionNeeded,
            .error(message: "Network failure"),
            .interrupted,
            .missed(at: Date(timeIntervalSince1970: 1742400000.123)),
        ]

        for result in cases {
            let data = try JSONEncoder().encode(result)
            let decoded = try JSONDecoder().decode(TaskRunResult.self, from: data)
            #expect(decoded == result)
        }
    }

    @Test func missedDatePreservesFractionalSeconds() throws {
        let original = TaskRunResult.missed(at: Date(timeIntervalSince1970: 1742400000.456))
        let data = try JSONEncoder().encode(original)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains(".456"))

        let decoded = try JSONDecoder().decode(TaskRunResult.self, from: data)
        #expect(decoded == original)
    }
}

// MARK: - ScheduledTask Tests

@MainActor
struct ScheduledTaskTests {

    @Test func createValidatesAndComputesNextRunAt() throws {
        let task = try ScheduledTask.create(
            name: "Test Task",
            cronExpression: "0 9 * * *",
            prompt: "Do something"
        )

        #expect(task.name == "Test Task")
        #expect(task.cronExpression == "0 9 * * *")
        #expect(task.prompt == "Do something")
        #expect(task.enabled)
        #expect(task.runCount == 0)
        #expect(task.nextRunAt != nil)
        #expect(task.createdBy == .user)
    }

    @Test func createThrowsForInvalidCron() {
        #expect(throws: ScheduledTaskError.self) {
            try ScheduledTask.create(
                name: "Bad",
                cronExpression: "invalid cron",
                prompt: "test"
            )
        }
    }

    @Test func isExhaustedLogic() throws {
        var task = try ScheduledTask.create(
            name: "Limited",
            cronExpression: "* * * * *",
            prompt: "test",
            maxRuns: 3
        )

        #expect(!task.isExhausted)
        task.runCount = 2
        #expect(!task.isExhausted)
        task.runCount = 3
        #expect(task.isExhausted)
    }

    @Test func parsedCronExpressionNilForGarbage() {
        let task = ScheduledTask(
            id: UUID(), name: "Bad", description: "",
            cronExpression: "not valid", prompt: "test",
            enabled: true, createdBy: .user, createdAt: Date(),
            lastRunAt: nil, lastRunResult: nil, nextRunAt: nil,
            runCount: 0, maxRuns: nil, tags: [], notifyUser: true,
            speakResult: false, sessionId: UUID()
        )
        #expect(task.parsedCronExpression == nil)
    }

    @Test func fullCodableRoundTrip() throws {
        var task = try ScheduledTask.create(
            name: "Round Trip",
            cronExpression: "30 8 * * 1-5",
            prompt: "Check status",
            description: "Morning check",
            createdBy: .agent(reason: "pattern detected"),
            maxRuns: 10,
            tags: ["monitoring"],
            notifyUser: true,
            speakResult: true
        )
        task.lastRunAt = Date()
        task.lastRunResult = .success(summary: "All clear")
        task.runCount = 2

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(task)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(ScheduledTask.self, from: data)

        #expect(decoded.id == task.id)
        #expect(decoded.name == task.name)
        #expect(decoded.cronExpression == task.cronExpression)
        #expect(decoded.createdBy == task.createdBy)
        #expect(decoded.lastRunResult == task.lastRunResult)
        #expect(decoded.tags == task.tags)
        #expect(decoded.maxRuns == task.maxRuns)
    }
}

// MARK: - ScheduledTaskIndex Codable Tests

@MainActor
struct ScheduledTaskIndexCodableTests {

    @Test func roundTripsWrappedFormat() throws {
        let summary = ScheduledTaskSummary(
            id: UUID(), name: "Test", cronExpression: "0 9 * * *",
            enabled: true, nextRunAt: Date(), createdBy: .user, sessionId: UUID()
        )
        let index = ScheduledTaskIndex(version: 1, tasks: [summary])

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(index)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(ScheduledTaskIndex.self, from: data)

        #expect(decoded.version == 1)
        #expect(decoded.tasks.count == 1)
        #expect(decoded.tasks[0].id == summary.id)
    }

    @Test func jsonContainsVersionField() throws {
        let index = ScheduledTaskIndex(version: 1, tasks: [])
        let data = try JSONEncoder().encode(index)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(json["version"] as? Int == 1)
        #expect(json["tasks"] as? [Any] != nil)
    }
}

// MARK: - ScheduledTaskStore Tests

@MainActor
struct ScheduledTaskStoreTests {

    private func makeTempStore() -> (store: ScheduledTaskStore, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "tesseract-scheduled-task-tests-\(UUID().uuidString)", isDirectory: true
            )
        return (ScheduledTaskStore(baseDirectory: root), root)
    }

    @Test func saveAndLoadTask() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task = try ScheduledTask.create(
            name: "Save Test", cronExpression: "0 9 * * *", prompt: "test prompt"
        )
        store.save(task)

        let loaded = store.loadTask(id: task.id)
        #expect(loaded != nil)
        #expect(loaded?.name == "Save Test")
        #expect(loaded?.prompt == "test prompt")
        #expect(store.tasks.count == 1)
        #expect(store.tasks[0].id == task.id)
    }

    @Test func loadAllReturnsTasks() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task1 = try ScheduledTask.create(
            name: "Task 1", cronExpression: "0 9 * * *", prompt: "p1"
        )
        let task2 = try ScheduledTask.create(
            name: "Task 2", cronExpression: "0 17 * * *", prompt: "p2"
        )
        store.save(task1)
        store.save(task2)

        let all = store.loadAll()
        #expect(all.count == 2)
        let names = Set(all.map(\.name))
        #expect(names == ["Task 1", "Task 2"])
    }

    @Test func deleteRemovesTaskAndRuns() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task = try ScheduledTask.create(
            name: "Delete Me", cronExpression: "0 9 * * *", prompt: "test"
        )
        store.save(task)

        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: UUID(),
            startedAt: Date(), completedAt: Date(), durationSeconds: 5,
            result: .success(summary: "done"), summary: "done",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        store.saveRun(run)

        store.delete(id: task.id)
        #expect(store.tasks.isEmpty)
        #expect(store.loadTask(id: task.id) == nil)
        #expect(store.loadRuns(for: task.id).isEmpty)
    }

    @Test func saveRunAndLoadRuns() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let taskId = UUID()
        let run1 = TaskRun(
            id: UUID(), taskId: taskId, sessionId: UUID(),
            startedAt: Date(timeIntervalSince1970: 1000), completedAt: nil,
            durationSeconds: nil, result: .success(summary: "r1"), summary: "r1",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        let run2 = TaskRun(
            id: UUID(), taskId: taskId, sessionId: UUID(),
            startedAt: Date(timeIntervalSince1970: 2000), completedAt: nil,
            durationSeconds: nil, result: .noActionNeeded, summary: "r2",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        store.saveRun(run1)
        store.saveRun(run2)

        let runs = store.loadRuns(for: taskId)
        #expect(runs.count == 2)
        // Sorted newest-first
        #expect(runs[0].summary == "r2")
        #expect(runs[1].summary == "r1")
    }

    @Test func updateAfterRunAdvancesState() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task = try ScheduledTask.create(
            name: "Updatable", cronExpression: "0 9 * * *", prompt: "test"
        )
        store.save(task)

        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: Date(), completedAt: Date(), durationSeconds: 3,
            result: .success(summary: "done"), summary: "done",
            notifiedUser: false, spokeResult: false, tokensUsed: 100
        )
        store.updateAfterRun(taskId: task.id, run: run)

        let updated = store.loadTask(id: task.id)
        #expect(updated?.runCount == 1)
        #expect(updated?.lastRunResult == .success(summary: "done"))
        #expect(updated?.lastRunAt != nil)
        #expect(updated?.nextRunAt != nil)
    }

    @Test func updateAfterRunAutoDisablesExhaustedTask() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task = try ScheduledTask.create(
            name: "One Shot", cronExpression: "* * * * *", prompt: "test", maxRuns: 1
        )
        store.save(task)

        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: Date(), completedAt: Date(), durationSeconds: 1,
            result: .success(summary: "final"), summary: "final",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        store.updateAfterRun(taskId: task.id, run: run)

        let updated = store.loadTask(id: task.id)
        #expect(updated?.runCount == 1)
        #expect(updated?.enabled == false)
    }

    @Test func storageVersionMismatchWipesData() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "tesseract-scheduled-task-tests-\(UUID().uuidString)", isDirectory: true
            )
        defer { try? FileManager.default.removeItem(at: root) }

        // Create store and save a task
        let store1 = ScheduledTaskStore(baseDirectory: root)
        let task = try ScheduledTask.create(
            name: "Survivor", cronExpression: "0 9 * * *", prompt: "test"
        )
        store1.save(task)
        #expect(store1.tasks.count == 1)

        // Corrupt the version file
        let versionFile = root.appendingPathComponent(".storage_version")
        try "999".write(to: versionFile, atomically: true, encoding: .utf8)

        // Re-init should wipe
        let store2 = ScheduledTaskStore(baseDirectory: root)
        #expect(store2.tasks.isEmpty)
    }

    @Test func indexPrunesOrphanedEntries() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "tesseract-scheduled-task-tests-\(UUID().uuidString)", isDirectory: true
            )
        defer { try? FileManager.default.removeItem(at: root) }

        let store1 = ScheduledTaskStore(baseDirectory: root)
        let task = try ScheduledTask.create(
            name: "Orphan", cronExpression: "0 9 * * *", prompt: "test"
        )
        store1.save(task)
        #expect(store1.tasks.count == 1)

        // Remove the backing file but leave the index
        let taskFile = root
            .appendingPathComponent("tasks", isDirectory: true)
            .appendingPathComponent("\(task.id.uuidString).json")
        try FileManager.default.removeItem(at: taskFile)

        // Re-init should prune the orphan
        let store2 = ScheduledTaskStore(baseDirectory: root)
        #expect(store2.tasks.isEmpty)
    }

    @Test func indexPrunesCorruptTaskFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "tesseract-scheduled-task-tests-\(UUID().uuidString)", isDirectory: true
            )
        defer { try? FileManager.default.removeItem(at: root) }

        let store1 = ScheduledTaskStore(baseDirectory: root)
        let task = try ScheduledTask.create(
            name: "Corrupt", cronExpression: "0 9 * * *", prompt: "test"
        )
        store1.save(task)
        #expect(store1.tasks.count == 1)

        // Corrupt the backing file (exists but not valid JSON)
        let taskFile = root
            .appendingPathComponent("tasks", isDirectory: true)
            .appendingPathComponent("\(task.id.uuidString).json")
        try "not json".write(to: taskFile, atomically: true, encoding: .utf8)

        // Re-init should prune the corrupt entry
        let store2 = ScheduledTaskStore(baseDirectory: root)
        #expect(store2.tasks.isEmpty)
    }

    @Test func loadAllReconcilesnilNextRunAtFromIndex() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(
            name: "Exhausted", cronExpression: "0 9 * * *", prompt: "test", maxRuns: 1
        )
        // Simulate an exhausted task: index has nil nextRunAt
        task.nextRunAt = nil
        task.enabled = false
        task.runCount = 1
        store.save(task)
        #expect(store.tasks[0].nextRunAt == nil)

        // Overwrite the task file with a stale nextRunAt
        var divergent = task
        divergent.nextRunAt = Date(timeIntervalSince1970: 9999)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(divergent)
        let taskFile = root
            .appendingPathComponent("tasks", isDirectory: true)
            .appendingPathComponent("\(task.id.uuidString).json")
        try data.write(to: taskFile, options: .atomic)

        // loadAll should use the index's nil, not the task file's stale date
        let loaded = store.loadAll()
        #expect(loaded.count == 1)
        #expect(loaded[0].nextRunAt == nil)
    }

    @Test func loadAllReconcilesnextRunAtFromIndex() throws {
        let (store, root) = makeTempStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let task = try ScheduledTask.create(
            name: "Reconcile", cronExpression: "0 9 * * *", prompt: "test"
        )
        store.save(task)

        // Simulate a crash-divergent state: manually overwrite the task file
        // with a different nextRunAt, while the index retains the original
        var divergent = task
        divergent.nextRunAt = Date(timeIntervalSince1970: 9999)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(divergent)
        let taskFile = root
            .appendingPathComponent("tasks", isDirectory: true)
            .appendingPathComponent("\(task.id.uuidString).json")
        try data.write(to: taskFile, options: .atomic)

        // loadAll should use the index's nextRunAt, not the task file's
        let loaded = store.loadAll()
        #expect(loaded.count == 1)
        #expect(loaded[0].nextRunAt == task.nextRunAt)
        #expect(loaded[0].nextRunAt != divergent.nextRunAt)
    }
}

// MARK: - Scheduling Actor Tests

@MainActor
struct SchedulingActorTests {

    @Test func detectsDueTasksAndExecutes() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Due", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.count == 1)
        #expect(runs[0].result == .noActionNeeded)
    }

    @Test func skipsFutureTasksAsNotDue() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Future", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: 3600)
        store.save(task)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.isEmpty)
    }

    @Test func skipsDisabledTasks() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Disabled", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        task.enabled = false
        store.save(task)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.isEmpty)
    }

    @Test func skipsExhaustedTasks() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(
            name: "Exhausted", cronExpression: "* * * * *", prompt: "test", maxRuns: 3
        )
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        task.runCount = 3
        store.save(task)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.isEmpty)
    }

    @Test func pausePreventsDueTaskExecution() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Paused", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)

        await scheduler.pause()
        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.isEmpty)
    }

    @Test func missedRunUnderOneHourEnqueuesCatchUp() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        let now = Date()
        let missedAt = now.addingTimeInterval(-1800) // 30 minutes ago
        var task = try ScheduledTask.create(name: "Recent Miss", cronExpression: "0 9 * * *", prompt: "test")
        task.nextRunAt = missedAt
        store.save(task)

        await scheduler.detectMissedRuns(now: now)
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.count == 1)
        #expect(runs[0].result == .noActionNeeded) // Caught up via execution

        let updated = store.loadTask(id: task.id)
        #expect(updated?.nextRunAt != missedAt) // nextRunAt was advanced
    }

    @Test func missedRunOverOneHourLogsMissed() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        let now = Date()
        let missedAt = now.addingTimeInterval(-7200) // 2 hours ago
        var task = try ScheduledTask.create(name: "Old Miss", cronExpression: "0 9 * * *", prompt: "test")
        task.nextRunAt = missedAt
        store.save(task)

        await scheduler.detectMissedRuns(now: now)
        #expect(await scheduler.awaitIdleForTesting())

        let runs = store.loadRuns(for: task.id)
        #expect(runs.count == 1)
        if case .missed = runs[0].result {
            // Expected
        } else {
            Issue.record("Expected missed run result, got \(runs[0].result)")
        }

        let updated = store.loadTask(id: task.id)
        #expect(updated?.nextRunAt != missedAt) // nextRunAt was advanced
    }

    @Test func catchUpOnlyOncePerTaskPerLaunch() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig()
        defer { try? FileManager.default.removeItem(at: root) }

        let now = Date()
        var task = try ScheduledTask.create(name: "One Catch-Up", cronExpression: "0 9 * * *", prompt: "test")
        task.nextRunAt = now.addingTimeInterval(-1800) // 30 minutes ago
        store.save(task)

        // First detectMissedRuns — should catch up
        await scheduler.detectMissedRuns(now: now)
        #expect(await scheduler.awaitIdleForTesting())
        var runs = store.loadRuns(for: task.id)
        #expect(runs.count == 1)

        // Reset nextRunAt to past again
        if var current = store.loadTask(id: task.id) {
            current.nextRunAt = now.addingTimeInterval(-1800)
            store.save(current)
        }

        // Second detectMissedRuns — should NOT catch up again
        await scheduler.detectMissedRuns(now: now)
        #expect(await scheduler.awaitIdleForTesting())
        runs = store.loadRuns(for: task.id)
        #expect(runs.count == 1) // Still only 1 run
    }

    @Test func autoPausesAfterConsecutiveFailures() async throws {
        let (scheduler, store, root) = makeSchedulingTestRig { _ in
            .error(message: "test failure")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Flaky", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)

        for _ in 0..<5 {
            if var current = store.loadTask(id: task.id) {
                current.nextRunAt = Date(timeIntervalSinceNow: -120)
                store.save(current)
            }
            await scheduler.checkAndRunDueTasks()
            #expect(await scheduler.awaitIdleForTesting())
        }

        let updated = store.loadTask(id: task.id)
        #expect(updated?.enabled == false)
    }

    @Test func successResetsFailureCount() async throws {
        let counter = SchedulingCallCounter()
        let (scheduler, store, root) = makeSchedulingTestRig { _ in
            let n = counter.increment()
            // Calls 1-3: error, call 4: success, calls 5-7: error
            if n == 4 { return .success(summary: "ok") }
            return .error(message: "fail")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "Recoverable", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)

        for _ in 0..<7 {
            if var current = store.loadTask(id: task.id) {
                current.nextRunAt = Date(timeIntervalSinceNow: -120)
                store.save(current)
            }
            await scheduler.checkAndRunDueTasks()
            #expect(await scheduler.awaitIdleForTesting())
        }

        // Task should still be enabled (max consecutive errors was 3, not 5)
        let updated = store.loadTask(id: task.id)
        #expect(updated?.enabled == true)
        let failures = await scheduler.consecutiveFailures(for: task.id)
        #expect(failures == 3)
    }

    @Test func skipsTaskDisabledAfterEnqueue() async throws {
        let recorder = SchedulingExecutionRecorder()
        let (scheduler, store, root) = makeSchedulingTestRig { task in
            recorder.record(task.id)
            return .noActionNeeded
        }
        defer { try? FileManager.default.removeItem(at: root) }

        var task = try ScheduledTask.create(name: "WillPause", cronExpression: "* * * * *", prompt: "test")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)

        // Enqueue by checking due tasks, but disable before processing
        // The actor re-reads from store before executing each queued task
        task.enabled = false
        store.save(task)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        #expect(recorder.ids.isEmpty)
    }

    @Test func executesSequentially() async throws {
        let recorder = SchedulingExecutionRecorder()
        let (scheduler, store, root) = makeSchedulingTestRig { task in
            recorder.record(task.id)
            return .noActionNeeded
        }
        defer { try? FileManager.default.removeItem(at: root) }

        var taskA = try ScheduledTask.create(name: "A", cronExpression: "* * * * *", prompt: "a")
        taskA.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(taskA)

        var taskB = try ScheduledTask.create(name: "B", cronExpression: "* * * * *", prompt: "b")
        taskB.nextRunAt = Date(timeIntervalSinceNow: -60)
        store.save(taskB)

        await scheduler.checkAndRunDueTasks()
        #expect(await scheduler.awaitIdleForTesting())

        #expect(recorder.ids.count == 2)
        #expect(recorder.ids[0] == taskA.id)
        #expect(recorder.ids[1] == taskB.id)
    }
}

// MARK: - Scheduling Test Helpers

private final class SchedulingExecutionRecorder: @unchecked Sendable {
    private(set) var ids: [UUID] = []
    func record(_ id: UUID) { ids.append(id) }
}

private final class SchedulingCallCounter: @unchecked Sendable {
    private var count = 0
    func increment() -> Int {
        count += 1
        return count
    }
}

@MainActor
private func makeSchedulingTestRig(
    executeTask: @escaping @Sendable (ScheduledTask) async -> TaskRunResult = { _ in .noActionNeeded }
) -> (scheduler: SchedulingActor, store: ScheduledTaskStore, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-scheduling-tests-\(UUID().uuidString)", isDirectory: true)
    let store = ScheduledTaskStore(baseDirectory: root)
    let scheduler = SchedulingActor(
        taskStore: store,
        executeTask: executeTask,
        executeHeartbeat: { _ in .noActionNeeded },
        persistInFlightSession: {}
    )
    return (scheduler, store, root)
}

@MainActor
private func makeSchedulingServiceTestRig(
    executeTask: @escaping @Sendable (ScheduledTask) async -> TaskRunResult = { _ in .noActionNeeded }
) -> (service: SchedulingService, store: ScheduledTaskStore, root: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-service-tests-\(UUID().uuidString)", isDirectory: true)
    let store = ScheduledTaskStore(baseDirectory: root)
    let settings = SettingsManager()
    let actor = SchedulingActor(
        taskStore: store,
        executeTask: executeTask,
        executeHeartbeat: { _ in .noActionNeeded },
        persistInFlightSession: {}
    )
    let notificationService = NotificationService(settings: settings)
    let speechCoordinator = SpeechCoordinator(
        textExtractor: TextExtractor(),
        speechEngine: SpeechEngine(),
        playbackManager: AudioPlaybackManager(),
        settings: settings
    )
    let service = SchedulingService(actor: actor, store: store, settings: settings, notificationService: notificationService, speechCoordinator: speechCoordinator)
    return (service, store, root)
}

// MARK: - SchedulingService Tests

@MainActor
struct SchedulingServiceTests {

    @Test func startLoadsTasks() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "Test", cronExpression: "0 9 * * *", prompt: "hello")
        store.save(task)

        await service.start()

        #expect(service.tasks.count == 1)
        #expect(service.tasks[0].id == task.id)
        await service.stop()
    }

    @Test func createTaskAddsToState() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()

        let task = try ScheduledTask.create(name: "New", cronExpression: "0 9 * * *", prompt: "prompt")
        try service.createTask(task)

        #expect(service.tasks.contains(where: { $0.id == task.id }))
        await service.stop()
    }

    @Test func deleteTaskRemovesFromState() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "Doomed", cronExpression: "0 9 * * *", prompt: "bye")
        store.save(task)
        await service.start()

        service.deleteTask(id: task.id)

        #expect(!service.tasks.contains(where: { $0.id == task.id }))
        #expect(service.runHistory[task.id] == nil)
        await service.stop()
    }

    @Test func pauseTaskDisablesTask() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "Active", cronExpression: "0 9 * * *", prompt: "test")
        store.save(task)
        await service.start()

        service.pauseTask(id: task.id)

        let updated = store.loadTask(id: task.id)
        #expect(updated?.enabled == false)
        await service.stop()
    }

    @Test func resumeTaskEnablesAndAdvancesNextRun() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        var task = try ScheduledTask.create(name: "Paused", cronExpression: "0 9 * * *", prompt: "test")
        task.enabled = false
        task.nextRunAt = nil
        store.save(task)
        await service.start()

        try service.resumeTask(id: task.id)

        let updated = store.loadTask(id: task.id)
        #expect(updated?.enabled == true)
        #expect(updated?.nextRunAt != nil)
        await service.stop()
    }

    @Test func pauseAllSetsFlags() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()

        service.pauseAll()

        #expect(service.isPaused == true)
        await service.stop()
    }

    @Test func resumeAllClearsFlags() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        service.pauseAll()

        service.resumeAll()

        #expect(service.isPaused == false)
        await service.stop()
    }

    @Test func startPopulatesRunHistory() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "WithRuns", cronExpression: "0 9 * * *", prompt: "test")
        store.save(task)

        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: Date(), completedAt: Date(), durationSeconds: 1,
            result: .success(summary: "ok"), summary: "ok",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        store.saveRun(run)
        await service.start()

        #expect(service.runHistory[task.id]?.count == 1)
        #expect(service.runHistory[task.id]?[0].id == run.id)
        await service.stop()
    }

    @Test func syncRefreshesCachedRunHistory() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "Tracked", cronExpression: "0 9 * * *", prompt: "test")
        store.save(task)
        await service.start()

        // Initial — no runs yet
        #expect(service.runHistory[task.id]?.count == 0)

        // Simulate a completed run saved to store
        let run = TaskRun(
            id: UUID(), taskId: task.id, sessionId: task.sessionId,
            startedAt: Date(), completedAt: Date(), durationSeconds: 1,
            result: .success(summary: "done"), summary: "done",
            notifiedUser: false, spokeResult: false, tokensUsed: nil
        )
        store.saveRun(run)

        // Trigger sync (e.g., from actor completing a run and saving to store)
        store.updateAfterRun(taskId: task.id, run: run)

        // Cached history should now include the new run
        #expect(service.runHistory[task.id]?.count == 1)
        #expect(service.runHistory[task.id]?[0].id == run.id)
        await service.stop()
    }

    @Test func runningTaskIdUpdatesOnActorRun() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig { _ in .noActionNeeded }
        var task = try ScheduledTask.create(name: "Runner", cronExpression: "* * * * *", prompt: "go")
        task.nextRunAt = Date(timeIntervalSinceNow: -120)
        store.save(task)
        await service.start()

        // Before any run, no task is running
        #expect(service.currentlyRunningTaskId == nil)
        await service.stop()
    }

    @Test func stopCancelsSubscription() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        await service.start()

        await service.stop()

        // After stop, store changes should not propagate
        let task = try ScheduledTask.create(name: "Ghost", cronExpression: "0 9 * * *", prompt: "test")
        store.save(task)

        // Service tasks should remain empty (no sync after stop)
        #expect(!service.tasks.contains(where: { $0.id == task.id }))
    }
}

// MARK: - CronToolTests

@MainActor
struct CronToolTests {

    // MARK: - cron_create

    @Test func cronCreateValidTask() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        let result = try await tool.execute("call-1", [
            "name": .string("Morning report"),
            "cron": .string("0 9 * * *"),
            "prompt": .string("Summarize today's tasks"),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("Created scheduled task 'Morning report'"))
        #expect(text.contains("0 9 * * *"))
        #expect(text.contains("Next run:"))
        #expect(service.tasks.count == 1)
        await service.stop()
    }

    @Test func cronCreateRejectsInvalidCron() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        let result = try await tool.execute("call-1", [
            "name": .string("Bad"),
            "cron": .string("not a cron"),
            "prompt": .string("test"),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("Invalid cron expression"))
        #expect(service.tasks.isEmpty)
        await service.stop()
    }

    @Test func cronCreateRejectsShortInterval() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        let result = try await tool.execute("call-1", [
            "name": .string("Too frequent"),
            "cron": .string("* * * * *"),
            "prompt": .string("spam"),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("5 minutes") || text.contains("300s"))
        #expect(service.tasks.isEmpty)
        await service.stop()
    }

    @Test func cronCreateRejectsMissingArgs() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        let result = try await tool.execute("call-1", [
            "cron": .string("0 9 * * *"),
            "prompt": .string("test"),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("Missing required parameter: name"))
        await service.stop()
    }

    @Test func cronCreateForcesNotifyTrue() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        _ = try await tool.execute("call-1", [
            "name": .string("Silent"),
            "cron": .string("0 9 * * *"),
            "prompt": .string("test"),
            "notify": .bool(false),
        ], nil, nil)

        #expect(service.tasks.count == 1)
        #expect(service.tasks[0].notifyUser == true)
        await service.stop()
    }

    @Test func cronCreateRejectsAt50Tasks() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        for i in 0..<50 {
            let task = try ScheduledTask.create(
                name: "Task \(i)", cronExpression: "0 \(i % 24) * * *", prompt: "p"
            )
            store.save(task)
        }
        await service.start()

        let tool = createCronCreateTool(schedulingService: service)
        let result = try await tool.execute("call-1", [
            "name": .string("Overflow"),
            "cron": .string("0 9 * * *"),
            "prompt": .string("test"),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("Maximum active tasks limit reached"))
        await service.stop()
    }

    @Test func cronCreateSetsAgentCreator() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        _ = try await tool.execute("call-1", [
            "name": .string("Agent task"),
            "cron": .string("0 9 * * *"),
            "prompt": .string("do something"),
        ], nil, nil)

        #expect(service.tasks.count == 1)
        if case .agent = service.tasks[0].createdBy {
            // expected
        } else {
            Issue.record("Expected .agent creator, got \(service.tasks[0].createdBy)")
        }
        await service.stop()
    }

    @Test func cronCreateRejectsZeroMaxRuns() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()
        let tool = createCronCreateTool(schedulingService: service)

        let result = try await tool.execute("call-1", [
            "name": .string("Zero runs"),
            "cron": .string("0 9 * * *"),
            "prompt": .string("test"),
            "max_runs": .int(0),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("max_runs must be at least 1"))
        #expect(service.tasks.isEmpty)
        await service.stop()
    }

    // MARK: - cron_list

    @Test func cronListAllTasks() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let t1 = try ScheduledTask.create(name: "A", cronExpression: "0 9 * * *", prompt: "a")
        let t2 = try ScheduledTask.create(name: "B", cronExpression: "0 18 * * *", prompt: "b")
        store.save(t1)
        store.save(t2)
        await service.start()

        let tool = createCronListTool(schedulingService: service)
        let result = try await tool.execute("call-1", [:], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("2 scheduled task(s)"))
        #expect(text.contains("A"))
        #expect(text.contains("B"))
        await service.stop()
    }

    @Test func cronListFiltersActive() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let active = try ScheduledTask.create(name: "Active", cronExpression: "0 9 * * *", prompt: "a")
        var paused = try ScheduledTask.create(name: "Paused", cronExpression: "0 18 * * *", prompt: "b")
        paused.enabled = false
        store.save(active)
        store.save(paused)
        await service.start()

        let tool = createCronListTool(schedulingService: service)
        let result = try await tool.execute("call-1", ["filter": .string("active")], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("1 scheduled task(s)"))
        #expect(text.contains("Active"))
        #expect(!text.contains("Paused"))
        await service.stop()
    }

    @Test func cronListFiltersPaused() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let active = try ScheduledTask.create(name: "Active", cronExpression: "0 9 * * *", prompt: "a")
        var paused = try ScheduledTask.create(name: "Paused", cronExpression: "0 18 * * *", prompt: "b")
        paused.enabled = false
        store.save(active)
        store.save(paused)
        await service.start()

        let tool = createCronListTool(schedulingService: service)
        let result = try await tool.execute("call-1", ["filter": .string("paused")], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("1 scheduled task(s)"))
        #expect(text.contains("Paused"))
        #expect(!text.contains("Name: Active"))
        await service.stop()
    }

    @Test func cronListFiltersMine() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let userTask = try ScheduledTask.create(
            name: "User", cronExpression: "0 9 * * *", prompt: "a", createdBy: .user
        )
        let agentTask = try ScheduledTask.create(
            name: "Agent", cronExpression: "0 18 * * *", prompt: "b",
            createdBy: .agent(reason: "test")
        )
        store.save(userTask)
        store.save(agentTask)
        await service.start()

        let tool = createCronListTool(schedulingService: service)
        let result = try await tool.execute("call-1", ["filter": .string("mine")], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("1 scheduled task(s)"))
        #expect(text.contains("Agent"))
        #expect(!text.contains("Name: User"))
        await service.stop()
    }

    @Test func cronListEmpty() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()

        let tool = createCronListTool(schedulingService: service)
        let result = try await tool.execute("call-1", [:], nil, nil)

        #expect(result.content.textContent == "No scheduled tasks found.")
        await service.stop()
    }

    // MARK: - cron_delete

    @Test func cronDeleteExistingTask() async throws {
        let (service, store, _) = makeSchedulingServiceTestRig()
        let task = try ScheduledTask.create(name: "Doomed", cronExpression: "0 9 * * *", prompt: "bye")
        store.save(task)
        await service.start()

        let tool = createCronDeleteTool(schedulingService: service)
        let result = try await tool.execute("call-1", [
            "task_id": .string(task.id.uuidString),
        ], nil, nil)

        let text = result.content.textContent
        #expect(text.contains("Deleted task 'Doomed'"))
        #expect(text.contains(task.id.uuidString))
        #expect(service.tasks.isEmpty)
        await service.stop()
    }

    @Test func cronDeleteInvalidUUID() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()

        let tool = createCronDeleteTool(schedulingService: service)
        let result = try await tool.execute("call-1", [
            "task_id": .string("not-a-uuid"),
        ], nil, nil)

        #expect(result.content.textContent.contains("Invalid UUID format"))
        await service.stop()
    }

    @Test func cronDeleteNonexistent() async throws {
        let (service, _, _) = makeSchedulingServiceTestRig()
        await service.start()

        let fakeId = UUID()
        let tool = createCronDeleteTool(schedulingService: service)
        let result = try await tool.execute("call-1", [
            "task_id": .string(fakeId.uuidString),
        ], nil, nil)

        #expect(result.content.textContent.contains("Task not found"))
        await service.stop()
    }
}

// MARK: - BackgroundSessionStore Tests

private func makeBackgroundSessionTestDir() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-bgsession-tests-\(UUID().uuidString)", isDirectory: true)
}

struct BackgroundSessionStoreTests {

    @Test func loadOrCreateNewSession() async throws {
        let store = BackgroundSessionStore(baseDirectory: makeBackgroundSessionTestDir())
        let sessionId = UUID()

        let session = await store.loadOrCreate(sessionId: sessionId)

        #expect(session.id == sessionId)
        #expect(session.sessionType == .cron)
        #expect(session.displayName == "")
        #expect(session.messages.isEmpty)
        #expect(session.lastRunAt == nil)
    }

    @Test func loadOrCreateExistingSession() async throws {
        let store = BackgroundSessionStore(baseDirectory: makeBackgroundSessionTestDir())
        let sessionId = UUID()
        let taskId = UUID()

        // Create and save a session with a message
        var session = await store.loadOrCreate(sessionId: sessionId)
        session.taskId = taskId
        session.displayName = "Test Task"
        session.sessionType = .cron
        let taggedMessage = TaggedMessage(
            type: "user",
            payload: ["content": .string("hello")]
        )
        session.messages = [taggedMessage]
        session.lastRunAt = Date()
        await store.save(session)

        // Load it back
        let loaded = await store.loadOrCreate(sessionId: sessionId)

        #expect(loaded.id == sessionId)
        #expect(loaded.messages.count == 1)
        #expect(loaded.messages[0].type == "user")
        #expect(loaded.lastRunAt != nil)
    }

    @Test func saveUpdatesIndex() async throws {
        let store = BackgroundSessionStore(baseDirectory: makeBackgroundSessionTestDir())
        let sessionId = UUID()
        let taskId = UUID()

        let session = BackgroundSession(
            id: sessionId, sessionType: .cron, displayName: "Indexed",
            taskId: taskId, messages: [], lastRunAt: nil, createdAt: Date()
        )
        await store.save(session)

        let all = await store.listAll()
        #expect(all.count == 1)
        #expect(all[0].id == sessionId)
        #expect(all[0].displayName == "Indexed")
        #expect(all[0].messageCount == 0)
    }

    @Test func deleteRemovesFileAndIndex() async throws {
        let dir = makeBackgroundSessionTestDir()
        let store = BackgroundSessionStore(baseDirectory: dir)
        let sessionId = UUID()
        let taskId = UUID()

        let session = BackgroundSession(
            id: sessionId, sessionType: .cron, displayName: "ToDelete",
            taskId: taskId, messages: [], lastRunAt: nil, createdAt: Date()
        )
        await store.save(session)
        #expect(await store.listAll().count == 1)

        await store.delete(sessionId: sessionId)

        #expect(await store.listAll().isEmpty)
        let fileExists = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("\(sessionId.uuidString).json").path
        )
        #expect(!fileExists)
    }

    @Test func listAllReturnsAllSessions() async throws {
        let store = BackgroundSessionStore(baseDirectory: makeBackgroundSessionTestDir())

        for i in 0..<3 {
            let session = BackgroundSession(
                id: UUID(), sessionType: .cron, displayName: "Task \(i)",
                taskId: UUID(), messages: [], lastRunAt: nil, createdAt: Date()
            )
            await store.save(session)
        }

        let all = await store.listAll()
        #expect(all.count == 3)
    }

    @Test func storageVersionMigration() async throws {
        let dir = makeBackgroundSessionTestDir()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Write a bad version file
        let versionURL = dir.appendingPathComponent(".storage_version")
        try "999".write(to: versionURL, atomically: true, encoding: .utf8)

        // Write a fake index that should be wiped
        let indexURL = dir.appendingPathComponent("index.json")
        try "{}".data(using: .utf8)!.write(to: indexURL, options: .atomic)

        let store = BackgroundSessionStore(baseDirectory: dir)
        let all = await store.listAll()
        #expect(all.isEmpty)

        // Version file should now match current
        let version = try String(contentsOf: versionURL, encoding: .utf8)
        #expect(version == "1")
    }

    @Test func rebuildIndexFromSessionFiles() async throws {
        let dir = makeBackgroundSessionTestDir()

        // Use one store to save two sessions (creates index.json + session files)
        let store1 = BackgroundSessionStore(baseDirectory: dir)
        let id1 = UUID(), id2 = UUID()
        let s1 = BackgroundSession(
            id: id1, sessionType: .cron, displayName: "Task A",
            taskId: UUID(), messages: [], lastRunAt: nil, createdAt: Date()
        )
        let s2 = BackgroundSession(
            id: id2, sessionType: .cron, displayName: "Task B",
            taskId: UUID(), messages: [], lastRunAt: Date(), createdAt: Date()
        )
        await store1.save(s1)
        await store1.save(s2)
        #expect(await store1.listAll().count == 2)

        // Delete index.json — simulates corruption/accidental deletion
        let indexURL = dir.appendingPathComponent("index.json")
        try FileManager.default.removeItem(at: indexURL)
        #expect(!FileManager.default.fileExists(atPath: indexURL.path))

        // New store should rebuild from session files
        let store2 = BackgroundSessionStore(baseDirectory: dir)
        let rebuilt = await store2.listAll()
        #expect(rebuilt.count == 2)

        let ids = Set(rebuilt.map(\.id))
        #expect(ids.contains(id1))
        #expect(ids.contains(id2))

        // Index should have been rewritten
        #expect(FileManager.default.fileExists(atPath: indexURL.path))
    }

    @Test func messageRoundTrip() async throws {
        let store = BackgroundSessionStore(baseDirectory: makeBackgroundSessionTestDir())
        let sessionId = UUID()

        // Encode a real UserMessage via SyncMessageCodec
        let userMsg = UserMessage.create("background prompt")
        let tagged = try SyncMessageCodec.encode(userMsg)

        var session = BackgroundSession(
            id: sessionId, sessionType: .cron, displayName: "RoundTrip",
            taskId: UUID(), messages: [tagged], lastRunAt: Date(), createdAt: Date()
        )
        await store.save(session)

        // Load back and decode
        let loaded = await store.loadOrCreate(sessionId: sessionId)
        #expect(loaded.messages.count == 1)

        let decoded = try SyncMessageCodec.decodeAll(loaded.messages)
        #expect(decoded.count == 1)
        let restoredUser = try #require(decoded[0] as? UserMessage)
        #expect(restoredUser.content == "background prompt")
    }
}

// MARK: - BackgroundPreamble Tests

@MainActor
struct BackgroundPreambleTests {

    @Test func includesTaskNameAndContext() async throws {
        let task = try ScheduledTask.create(
            name: "Daily Report",
            cronExpression: "0 9 * * *",
            prompt: "Summarize today's changes"
        )

        let preamble = SystemPromptAssembler.backgroundPreamble(for: task)

        #expect(preamble.contains("Daily Report"))
        #expect(preamble.contains("Background Task Execution Context"))
        // task.prompt is sent as a user message, not included in the preamble
        #expect(!preamble.contains("Summarize today's changes"))
    }

    @Test func omitsEmptyDescription() async throws {
        let task = try ScheduledTask.create(
            name: "No Desc",
            cronExpression: "0 9 * * *",
            prompt: "do something"
        )

        let preamble = SystemPromptAssembler.backgroundPreamble(for: task)

        #expect(!preamble.contains("Task description:"))
    }

    @Test func includesDescriptionWhenPresent() async throws {
        let task = try ScheduledTask.create(
            name: "With Desc",
            cronExpression: "0 9 * * *",
            prompt: "do something",
            description: "A helpful task"
        )

        let preamble = SystemPromptAssembler.backgroundPreamble(for: task)

        #expect(preamble.contains("Task description: A helpful task"))
    }

    @Test func includesSchedule() async throws {
        let task = try ScheduledTask.create(
            name: "Scheduled",
            cronExpression: "30 14 * * 1-5",
            prompt: "check status"
        )

        let preamble = SystemPromptAssembler.backgroundPreamble(for: task)

        #expect(preamble.contains("30 14 * * 1-5"))
        #expect(preamble.contains("Schedule:"))
    }
}
