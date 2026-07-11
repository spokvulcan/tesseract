import Foundation
import Testing

@testable import Tesseract_Agent

/// #274 capture plumbing: fence-safe span elision + inbound message capture.
@MainActor
struct ServerTraceCaptureTests {

    // MARK: - Fence-safe elision

    /// Fence parity walked the same way the elider (and `PanelCap`) walks it:
    /// a line whose space-trimmed prefix is ``` or ~~~ toggles state.
    private func fenceParityIsBalanced(_ text: String) -> Bool {
        let fenceLines = text.split(separator: "\n", omittingEmptySubsequences: false)
            .count { line in
                let trimmed = line.drop(while: { $0 == " " })
                return trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
            }
        return fenceLines.isMultiple(of: 2)
    }

    /// The elision marker line must sit outside any code fence: the fence
    /// lines *before* it must be balanced.
    private func markerIsOutsideFences(_ text: String) -> Bool {
        guard let markerRange = text.range(of: "--- elided ") else { return false }
        return fenceParityIsBalanced(String(text[..<markerRange.lowerBound]))
    }

    @Test func elisionCutsOnLineBoundaries() {
        let line = String(repeating: "y", count: 79) + "\n"
        let text = String(repeating: line, count: 5_000)  // 400 KB
        let capped = RequestTrace.fenceSafeElide(
            text,
            headBytes: ServerGenerationLog.textHeadBytes,
            tailBytes: ServerGenerationLog.textTailBytes
        )

        // Both cut edges land on whole lines: every "y" line in the result
        // is intact (79 chars), never a partial line spliced to the marker.
        let lines = capped.split(separator: "\n", omittingEmptySubsequences: true)
        for fragment in lines where fragment.hasPrefix("y") {
            #expect(fragment.count == 79)
        }
        #expect(capped.contains("--- elided "))
    }

    @Test func headCutInsideFenceGetsClosed() {
        // A fence opens before the head budget and would span the head cut.
        let headBytes = ServerGenerationLog.textHeadBytes
        let filler = String(repeating: String(repeating: "a", count: 63) + "\n", count: 512)
        let text =
            filler  // 32 KiB of prose
            + "```swift\n"  // opens before the 64 KiB head cut
            + String(
                repeating: "let x = 1  // code line padding padding padding padding\n",
                count: 8_000)  // ~450 KB of code, spans both cuts
            + "```\n"
            + "after\n"
        #expect(text.utf8.count > headBytes + ServerGenerationLog.textTailBytes)

        let capped = RequestTrace.fenceSafeElide(
            text,
            headBytes: headBytes,
            tailBytes: ServerGenerationLog.textTailBytes
        )

        #expect(markerIsOutsideFences(capped))
        #expect(fenceParityIsBalanced(capped))
        // The tail re-opens with the original opener so highlighting resumes.
        let markerIdx = capped.range(of: "--- elided ")!.upperBound
        #expect(capped[markerIdx...].contains("```swift\n"))
    }

    @Test func plainTextGetsNoSyntheticFences() {
        let text = String(
            repeating: String(repeating: "p", count: 49) + "\n", count: 8_000)
        let capped = RequestTrace.fenceSafeElide(
            text,
            headBytes: ServerGenerationLog.textHeadBytes,
            tailBytes: ServerGenerationLog.textTailBytes
        )
        #expect(!capped.contains("```"))
        #expect(!capped.contains("~~~"))
        #expect(capped.contains("--- elided "))
    }

    @Test func singleGiantLineFallsBackToRawSlice() {
        // No newline anywhere — e.g. minified-JSON tool arguments.
        let text = String(repeating: "j", count: 400_000)
        let capped = RequestTrace.fenceSafeElide(
            text,
            headBytes: ServerGenerationLog.textHeadBytes,
            tailBytes: ServerGenerationLog.textTailBytes
        )
        let budget = ServerGenerationLog.textHeadBytes + ServerGenerationLog.textTailBytes
        #expect(capped.utf8.count < budget + 200)
        #expect(capped.contains("--- elided "))
        #expect(capped.hasPrefix("jjjj"))
        #expect(capped.hasSuffix("jjjj"))
    }

    @Test func repeatedSlicingStaysFenceBalanced() {
        // Stream a fenced block through cappedAppend in chunks, forcing
        // multiple re-slices; the capped span must stay balanced throughout.
        var content = "prose\n```python\n"
        let chunk = String(repeating: "print('pad pad pad pad pad')\n", count: 2_000)  // ~58 KB
        for _ in 0..<8 {
            content = RequestTrace.cappedAppend(content, chunk)
        }
        content = RequestTrace.cappedAppend(content, "```\ndone\n")
        #expect(fenceParityIsBalanced(content))
        #expect(markerIsOutsideFences(content))
    }

    // MARK: - Inbound capture

    @Test func captureFlattensRolesAndText() {
        let capture = RequestTrace.captureInbound([
            .init(role: .system, content: .text("You are helpful.")),
            .init(role: .user, content: .text("Hi there")),
        ])
        #expect(capture.messages.count == 2)
        #expect(capture.elidedMessages == 0)
        #expect(capture.messages[0].role == "system")
        #expect(capture.messages[0].content == "You are helpful.")
        #expect(capture.messages[1].role == "user")
        #expect(capture.messages[1].content == "Hi there")
    }

    @Test func imagePartsBecomePlaceholdersNeverDataURLs() {
        let dataURL = "data:image/png;base64," + String(repeating: "A", count: 50_000)
        let capture = RequestTrace.captureInbound([
            .init(
                role: .user,
                content: .parts([
                    .init(type: .text, text: "What is in this image?", image_url: nil),
                    .init(type: .image_url, text: nil, image_url: .init(url: dataURL)),
                ]))
        ])
        let content = capture.messages[0].content
        #expect(content.contains("What is in this image?"))
        #expect(content.contains("⟨image⟩"))
        #expect(!content.contains("base64"))
        #expect(content.utf8.count < 200)
    }

    @Test func priorTurnToolCallsAreFlattened() {
        let capture = RequestTrace.captureInbound([
            .init(
                role: .assistant,
                content: nil,
                tool_calls: [
                    .init(
                        id: "call_1", type: "function",
                        function: .init(name: "read", arguments: "{\"path\": \"a.md\"}"),
                        index: nil)
                ])
        ])
        #expect(capture.messages[0].content == "⟨tool call ▸ read⟩ {\"path\": \"a.md\"}")
    }

    @Test func oversizedMessageContentIsCappedFenceSafe() {
        let big =
            "intro\n```json\n"
            + String(repeating: "{\"k\": \"vvvvvvvvvvvvvvvvvvvvvvvvvvvv\"},\n", count: 3_000)
            + "```\n"
        #expect(
            big.utf8.count
                > RequestTrace.inboundMessageHeadBytes + RequestTrace.inboundMessageTailBytes)

        let capture = RequestTrace.captureInbound([
            .init(role: .tool, content: .text(big))
        ])
        let content = capture.messages[0].content
        #expect(content.contains("--- elided "))
        #expect(
            content.utf8.count
                < RequestTrace.inboundMessageHeadBytes + RequestTrace.inboundMessageTailBytes
                + 200)
        #expect(fenceParityIsBalanced(content))
    }

    @Test func totalBudgetDropsMiddleMessagesKeepsFirstAndRecent() {
        // 40 messages ≈ 30 KB each ≫ the 512 KiB total budget.
        let filler = String(repeating: "m", count: 30_000)
        var wire: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("SYSTEM PROMPT"))
        ]
        for i in 0..<40 {
            wire.append(.init(role: .tool, content: .text("result \(i)\n" + filler)))
        }
        wire.append(.init(role: .user, content: .text("LAST USER TURN")))

        let capture = RequestTrace.captureInbound(wire)
        #expect(capture.elidedMessages > 0)
        #expect(capture.messages.count + capture.elidedMessages == wire.count)
        #expect(capture.messages.first?.content == "SYSTEM PROMPT")
        #expect(capture.messages.last?.content == "LAST USER TURN")

        let totalBytes = capture.messages.reduce(0) { $0 + $1.content.utf8.count }
        #expect(totalBytes <= RequestTrace.inboundTotalBudgetBytes)
    }

    @Test func startRequestAttachesInboundToTrace() {
        let log = ServerGenerationLog()
        let inbound = RequestTrace.captureInbound([
            .init(role: .user, content: .text("hello"))
        ])
        log.startRequest(
            completionID: "id", model: "m", stream: false,
            sessionAffinity: nil, inbound: inbound
        )
        #expect(log.traces[0].inbound == inbound)
        #expect(log.traces[0].inbound.messages[0].content == "hello")
    }

    @Test func emptyWireListCapturesEmpty() {
        #expect(RequestTrace.captureInbound([]) == .empty)
        #expect(RequestTrace.InboundCapture.empty.isEmpty)
    }
}
