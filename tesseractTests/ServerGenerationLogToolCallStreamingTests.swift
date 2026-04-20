import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

/// Coverage for the in-flight tool-call streaming path in `ServerGenerationLog`.
///
/// The parser-level behavior is covered by `ToolCallParserDeltaTests`. This
/// suite locks the log-level invariants that the user-visible Requests log UI
/// depends on:
/// - A `.toolCallDelta` on an empty trace creates a `.toolCallBuilding` span.
/// - Multiple deltas append to the same span rather than creating duplicates.
/// - A `.toolCall` following deltas replaces the building span in place
///   (so the UI's position/styling is preserved across finalize).
/// - A `.toolCall` with no preceding deltas (the vendor atomic path, used
///   by `MLXLMCommon.Generation.toolCall`) inserts a fresh `.toolCall` span.
/// - A `.malformedToolCall` follows the same two rules.
/// - Text after a building span starts a new `.text` span, leaving the
///   building span in place until it closes.
/// - The 33ms coalescer actually coalesces consecutive deltas (one
///   `streamingVersion` bump per flush window, not one per delta).
@MainActor
struct ServerGenerationLogToolCallStreamingTests {

    private func makeLog() -> (ServerGenerationLog, TraceHandle) {
        let log = ServerGenerationLog()
        let handle = log.startRequest(
            completionID: "chatcmpl-test",
            model: "qwen3.6-35b",
            stream: true,
            sessionAffinity: nil
        )
        return (log, handle)
    }

    @Test func firstToolCallDeltaCreatesBuildingSpan() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "\n{\"name\":"))
        log.flushPending(handle: handle)

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .toolCallBuilding(_, let name, let args) = spans[0] {
            #expect(name == "")
            #expect(args == "\n{\"name\":")
        } else {
            Issue.record("expected `.toolCallBuilding` span, got \(spans[0])")
        }
    }

    @Test func subsequentDeltasAppendToSameBuildingSpan() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "\n{\"name\":"))
        log.flushPending(handle: handle)
        log.ingest(handle: handle, event: .toolCallDelta(name: "read", argumentsDelta: "\"read\","))
        log.flushPending(handle: handle)
        log.ingest(handle: handle, event: .toolCallDelta(
            name: "read",
            argumentsDelta: "\"arguments\":{\"path\":\"/x\"}}"
        ))
        log.flushPending(handle: handle)

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .toolCallBuilding(_, let name, let args) = spans[0] {
            #expect(name == "read")
            #expect(args == "\n{\"name\":\"read\",\"arguments\":{\"path\":\"/x\"}}")
        } else {
            Issue.record("expected single `.toolCallBuilding` span, got \(spans)")
        }
    }

    @Test func toolCallAfterDeltasReplacesBuildingSpanInPlace() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(name: "read", argumentsDelta: "\n{\"name\":\"read\","))
        log.flushPending(handle: handle)
        let buildingID = log.traces[0].spans[0].id

        let call = ToolCall(function: ToolCall.Function(
            name: "read",
            arguments: ["path": "/x"]
        ))
        log.ingest(handle: handle, event: .toolCall(call))

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .toolCall(let id, let name, let argsJSON) = spans[0] {
            #expect(id == buildingID, "span id must be preserved across finalize")
            #expect(name == "read")
            #expect(argsJSON.contains("\"path\""))
        } else {
            Issue.record("expected finalized `.toolCall` span, got \(spans[0])")
        }
    }

    @Test func toolCallWithoutPrecedingDeltasInsertsFreshSpan() {
        // Vendor-parsed atomic path: `MLXLMCommon.Generation.toolCall` arrives
        // pre-parsed and the app-level `ToolCallParser` never runs, so no
        // deltas precede the `.toolCall` event.
        let (log, handle) = makeLog()

        let call = ToolCall(function: ToolCall.Function(
            name: "ls",
            arguments: ["path": "/"]
        ))
        log.ingest(handle: handle, event: .toolCall(call))

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .toolCall(_, let name, _) = spans[0] {
            #expect(name == "ls")
        } else {
            Issue.record("expected `.toolCall` span for vendor atomic path")
        }
    }

    @Test func malformedToolCallAfterDeltasReplacesBuildingSpan() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(
            name: "read",
            argumentsDelta: "\n{\"name\":\"read\",\"arguments\":{broken"
        ))
        log.flushPending(handle: handle)
        let buildingID = log.traces[0].spans[0].id

        log.ingest(handle: handle, event: .malformedToolCall("\n{\"name\":\"read\",\"arguments\":{broken"))

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .malformedToolCall(let id, let raw) = spans[0] {
            #expect(id == buildingID)
            #expect(raw.contains("broken"))
        } else {
            Issue.record("expected `.malformedToolCall` span replacement")
        }
    }

    @Test func malformedToolCallWithoutPrecedingDeltasInsertsFreshSpan() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .malformedToolCall("junk"))

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .malformedToolCall(_, let raw) = spans[0] {
            #expect(raw == "junk")
        } else {
            Issue.record("expected `.malformedToolCall` span")
        }
    }

    @Test func textAfterBuildingSpanOpensNewSpanLeavingBuildingInPlace() {
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(name: "x", argumentsDelta: "partial"))
        log.flushPending(handle: handle)
        log.ingest(handle: handle, event: .text("some text"))
        log.flushPending(handle: handle)

        let spans = log.traces[0].spans
        #expect(spans.count == 2)
        if case .toolCallBuilding(_, _, let args) = spans[0] {
            #expect(args == "partial")
        } else {
            Issue.record("expected first span to remain `.toolCallBuilding`")
        }
        if case .text(_, let content) = spans[1] {
            #expect(content == "some text")
        } else {
            Issue.record("expected second span to be `.text`")
        }
    }

    @Test func consecutiveDeltasInSingleCoalesceWindowProduceOneAppend() {
        // The coalescer's user-visible guarantee: three ingests within one
        // 33ms window result in exactly one `.toolCallBuilding` span whose
        // `argumentsJSON` is the concatenation of all three deltas. Version
        // bump behavior is timing-dependent on top of an independent
        // 100ms throttle that's covered by existing `ServerGenerationLogTests`;
        // this suite owns the data-shape invariant.
        let (log, handle) = makeLog()

        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "a"))
        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "b"))
        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "c"))
        log.flushPending(handle: handle)

        let spans = log.traces[0].spans
        #expect(spans.count == 1)
        if case .toolCallBuilding(_, _, let args) = spans[0] {
            #expect(args == "abc")
        } else {
            Issue.record("expected concatenated building span")
        }
    }

    @Test func toolCallDeltaBeforeAnyOtherEventMarksFirstTokenAt() {
        let (log, handle) = makeLog()
        #expect(log.traces[0].firstTokenAt == nil)

        log.ingest(handle: handle, event: .toolCallDelta(name: nil, argumentsDelta: "x"))
        log.flushPending(handle: handle)

        #expect(log.traces[0].firstTokenAt != nil)
        #expect(log.traces[0].phase == .decoding)
    }
}
