import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

// Disambiguate from the `ToolCallParser` protocol exported by
// `MLXLMCommon` (vendor). The app-level streaming parser is a concrete
// class in the Tesseract module.
private typealias Parser = Tesseract_Agent.ToolCallParser

/// Coverage for the `.toolCallDelta` event emitted by `ToolCallParser` when
/// content streams into a `<tool_call>` block before `</tool_call>` closes.
///
/// The invariants under test:
/// - Progressive chunks inside a tool-call block emit deltas carrying only
///   the newly-added body bytes (never the full buffer).
/// - `name` is `nil` until the first complete `"name":"X"` literal is
///   observed, then locks to that value for the remainder of the block.
/// - The authoritative `.toolCall` / `.malformedToolCall` event still fires
///   exactly once on `</tool_call>` close with the full parsed payload.
/// - If the whole `<tool_call>…</tool_call>` arrives in a single chunk (no
///   split), a single delta is emitted followed by the final event —
///   consumers that ignore deltas and only listen for `.toolCall` still
///   receive the same payload they did before.
struct ToolCallParserDeltaTests {

    @Test func splitChunkEmitsDeltasWithProgressiveNameAndFinalToolCall() {
        let parser = Parser()
        var events: [Parser.Event] = []
        events.append(contentsOf: parser.processChunk("<tool_call>\n{\"name\":"))
        events.append(contentsOf: parser.processChunk("\"read\", \"argu"))
        events.append(contentsOf: parser.processChunk("ments\":{\"path\":\"/x\"}}</tool_call>"))

        let deltas = events.compactMap { event -> (String?, String)? in
            if case .toolCallDelta(let name, let delta) = event {
                return (name, delta)
            }
            return nil
        }
        #expect(deltas.count >= 2, "expected at least two deltas, got \(deltas.count)")

        let deltaTexts = deltas.map { $0.1 }
        #expect(deltaTexts.joined().contains("\"name\":\"read\""))
        #expect(deltaTexts.joined().contains("\"path\":\"/x\""))

        // Name is nil until the first complete `"name":"X"` is observed.
        // The first delta ends with `"name":` (no closing quote yet) so name
        // must be nil. The delta that adds `"read"` completes the name match.
        #expect(deltas.first?.0 == nil)
        #expect(deltas.contains { $0.0 == "read" })

        // Exactly one `.toolCall` event with the parsed payload at the end.
        let tools = events.compactMap { event -> ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        #expect(tools.count == 1)
        #expect(tools.first?.function.name == "read")
        if case .string(let path) = tools.first?.function.arguments["path"] ?? .null {
            #expect(path == "/x")
        } else {
            Issue.record("expected string path argument in parsed tool call")
        }
    }

    @Test func splitChunkMalformedJSONFallsBackToMalformedEventAfterDeltas() {
        let parser = Parser()
        var events: [Parser.Event] = []
        events.append(contentsOf: parser.processChunk("<tool_call>\n{\"name\":"))
        events.append(contentsOf: parser.processChunk("\"broken\", \"argu"))
        // Missing closing brace on the arguments object, then close tag.
        events.append(contentsOf: parser.processChunk("ments\":{\"path\":\"/x\"</tool_call>"))

        let deltaCount = events.filter {
            if case .toolCallDelta = $0 { return true } else { return false }
        }.count
        #expect(deltaCount >= 2)

        let malformed = events.compactMap { event -> String? in
            if case .malformedToolCall(let raw) = event { return raw }
            return nil
        }
        #expect(malformed.count == 1)
        #expect(malformed.first?.contains("\"broken\"") == true)

        // No `.toolCall` event should fire on malformed JSON.
        let tools = events.filter {
            if case .toolCall = $0 { return true } else { return false }
        }
        #expect(tools.isEmpty)
    }

    @Test func wholeToolCallInSingleChunkStillEmitsSingleDeltaAndToolCall() {
        let parser = Parser()
        let events = parser.processChunk(
            "<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"/y\"}}</tool_call>"
        )

        let deltas = events.filter {
            if case .toolCallDelta = $0 { return true } else { return false }
        }
        // One delta for the body before close (emitted alongside the close
        // event so the building span is fully populated at finalize-time).
        #expect(deltas.count == 1)

        let tools = events.compactMap { event -> ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        #expect(tools.count == 1)
        #expect(tools.first?.function.name == "read")
    }

    @Test func deltaEventsArePrefixFreeOfAlreadyForwardedBytes() {
        // Consumers accumulate deltas by concatenation. If the parser
        // re-sent bytes it had already forwarded, the accumulated payload
        // would be wrong (duplication). This locks the append-only contract.
        let parser = Parser()
        var accumulated = ""
        var events: [Parser.Event] = []
        events.append(contentsOf: parser.processChunk("<tool_call>\n{\"name\":"))
        events.append(contentsOf: parser.processChunk("\"ls\","))
        events.append(contentsOf: parser.processChunk("\"arguments\":{}}</tool_call>"))

        for event in events {
            if case .toolCallDelta(_, let delta) = event {
                accumulated += delta
            }
        }

        // The accumulated payload must be exactly what was between the
        // `<tool_call>` open tag and the `</tool_call>` close, no more.
        #expect(accumulated == "\n{\"name\":\"ls\",\"arguments\":{}}")
    }

    @Test func twoConsecutiveToolCallsInOneStreamResetDeltaAccumulator() {
        let parser = Parser()
        var events: [Parser.Event] = []
        events.append(contentsOf: parser.processChunk(
            "<tool_call>\n{\"name\":\"a\",\"arguments\":{}}</tool_call>"
        ))
        events.append(contentsOf: parser.processChunk(
            "<tool_call>\n{\"name\":\"b\",\"arguments\":{}}</tool_call>"
        ))

        let deltas = events.compactMap { event -> String? in
            if case .toolCallDelta(_, let delta) = event { return delta }
            return nil
        }
        // Each tool call emits exactly one close-time delta because the
        // whole body arrives in a single chunk for both.
        #expect(deltas.count == 2)
        #expect(deltas[0].contains("\"name\":\"a\""))
        #expect(deltas[1].contains("\"name\":\"b\""))
        // Importantly, the second delta must NOT contain the first's body.
        #expect(deltas[1].contains("\"name\":\"a\"") == false)

        let tools = events.compactMap { event -> ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        #expect(tools.map { $0.function.name } == ["a", "b"])
    }

    @Test func textBeforeToolCallIsUnaffectedByDeltaPath() {
        let parser = Parser()
        let events = parser.processChunk(
            "pre-text <tool_call>\n{\"name\":\"x\",\"arguments\":{}}</tool_call> post"
        )

        let texts = events.compactMap { event -> String? in
            if case .text(let t) = event { return t }
            return nil
        }
        #expect(texts.contains("pre-text "))
        #expect(texts.contains(" post"))
    }
}
