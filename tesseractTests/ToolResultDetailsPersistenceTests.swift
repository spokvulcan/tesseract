//
//  ToolResultDetailsPersistenceTests.swift
//  tesseractTests
//
//  The persistence seam of PRD #200: typed tool-result details survive the
//  tagged-JSON round trip on `ToolResultMessage`, legacy payloads without a
//  details key still decode, and an unrecognized details shape degrades to
//  `nil` details instead of failing the whole message.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ToolResultDetailsPersistenceTests {

    private func roundTrip(_ message: ToolResultMessage) throws -> ToolResultMessage {
        let tagged = try SyncMessageCodec.encode(message)
        let decoded = try SyncMessageCodec.decode(tagged)
        return try #require(decoded as? ToolResultMessage)
    }

    @Test func editDetailsRoundTrip() throws {
        let details = ToolResultDetails.edit(
            EditToolDetails(
                path: "notes/todo.md",
                diff: "--- a\n+++ b\n-old\n+new",
                firstChangedLine: 12,
                oldText: "old",
                newText: "new"
            ))
        let message = ToolResultMessage(
            toolCallId: "call-1", toolName: "edit",
            content: [.text("Successfully replaced text in notes/todo.md.")],
            details: details
        )

        let decoded = try roundTrip(message)
        #expect(decoded == message)
        #expect(decoded.details == details)
    }

    @Test func readDetailsRoundTrip() throws {
        let details = ToolResultDetails.read(
            ReadToolDetails(
                path: "src/main.swift", lineCount: 40, wasTruncated: true, totalLines: 900)
        )
        let message = ToolResultMessage(
            toolCallId: "call-2", toolName: "read",
            content: [.text("     1\tline")],
            details: details
        )
        #expect(try roundTrip(message).details == details)
    }

    @Test func lsDetailsRoundTrip() throws {
        let details = ToolResultDetails.ls(
            LsToolDetails(
                truncation: LsToolTruncationDetails(
                    truncated: true, truncatedBy: "bytes", totalLines: 900, totalBytes: 90_000,
                    outputLines: 500, outputBytes: 51_200, lastLinePartial: false,
                    firstLineExceedsLimit: false, maxLines: 500, maxBytes: 51_200
                ),
                entryLimitReached: 500
            ))
        let message = ToolResultMessage(
            toolCallId: "call-3", toolName: "ls",
            content: [.text("a.txt\nb.txt")],
            details: details
        )
        #expect(try roundTrip(message).details == details)
    }

    @Test func nilDetailsRoundTripAndStayNil() throws {
        let message = ToolResultMessage(
            toolCallId: "call-4", toolName: "echo", content: [.text("ok")]
        )
        let decoded = try roundTrip(message)
        #expect(decoded.details == nil)
        #expect(decoded == message)
    }

    /// Conversations recorded before this feature have no `details` key at all.
    @Test func legacyPayloadWithoutDetailsKeyDecodes() throws {
        let legacyJSON = """
            {
                "id": "6F1BA1E0-0000-4000-8000-000000000001",
                "toolCallId": "call-5",
                "toolName": "read",
                "content": [{"type": "text", "text": "hello"}],
                "isError": false,
                "timestamp": 700000000
            }
            """
        let decoded = try JSONDecoder().decode(
            ToolResultMessage.self, from: Data(legacyJSON.utf8))
        #expect(decoded.details == nil)
        #expect(decoded.toolName == "read")
        #expect(decoded.content.textContent == "hello")
    }

    /// A details shape this build doesn't know (a future case, a corrupted
    /// payload) must cost only the details, never the message.
    @Test func unknownDetailsShapeDegradesToNil() throws {
        let futureJSON = """
            {
                "id": "6F1BA1E0-0000-4000-8000-000000000002",
                "toolCallId": "call-6",
                "toolName": "hologram",
                "content": [{"type": "text", "text": "rendered"}],
                "isError": false,
                "timestamp": 700000000,
                "details": {"hologram": {"frames": 3}}
            }
            """
        let decoded = try JSONDecoder().decode(
            ToolResultMessage.self, from: Data(futureJSON.utf8))
        #expect(decoded.details == nil)
        #expect(decoded.content.textContent == "rendered")
    }
}
