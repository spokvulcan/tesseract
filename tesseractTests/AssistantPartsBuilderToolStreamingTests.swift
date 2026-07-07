//
//  AssistantPartsBuilderToolStreamingTests.swift
//  tesseractTests
//
//  The Open Tool Call lifecycle in the parts builder: a tool-call part is
//  born at name-lock (real id + name from frame one), grows raw fragments
//  through `toolcallDelta`, commits in place on the parsed `.toolCall`
//  (same id — row identity and the Tool Clock key survive), and is
//  retracted without trace on every path that ends the turn without a
//  parse (text/thinking after an unclosed block, malformed close,
//  terminal close, explicit abort retraction).
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct AssistantPartsBuilderToolStreamingTests {

    // MARK: - Helpers

    private func toolCallPart(at index: Int, in message: AssistantMessage) -> ToolCallPart? {
        guard message.content.indices.contains(index),
            case .toolCall(let part) = message.content[index]
        else { return nil }
        return part
    }

    private func hasToolCallPart(_ message: AssistantMessage) -> Bool {
        message.content.contains { if case .toolCall = $0 { return true } else { return false } }
    }

    // MARK: - Name-lock birth

    @Test func preNameDeltasAreSilentAndPartIsBornAtNameLock() {
        var builder = AssistantPartsBuilder()

        // Fragment before the name literal completes: nothing to emit, no part.
        guard case .silent = builder.ingest(.toolCallDelta(name: nil, argumentsDelta: #"{"na"#))
        else {
            Issue.record("pre-name delta must be silent")
            return
        }
        #expect(!hasToolCallPart(builder.snapshot()))

        // Parser locks the name: the Open Tool Call is born with id + name.
        guard
            case .events(let events) = builder.ingest(
                .toolCallDelta(name: "read", argumentsDelta: #"me": "read", "arguments": {"#))
        else {
            Issue.record("name-lock delta must emit events")
            return
        }
        guard case .toolcallStart(let index, let partial) = events.last else {
            Issue.record("expected toolcallStart at name-lock")
            return
        }
        #expect(index == 0)
        let part = toolCallPart(at: 0, in: partial)
        #expect(part?.name == "read")
        #expect(part?.id.isEmpty == false)
        // The open part's arguments hold the raw accumulated body (pre-name
        // fragments included) — normalized JSON applies only to committed parts.
        #expect(part?.argumentsJSON == #"{"name": "read", "arguments": {"#)
    }

    @Test func nameLocksFromAccumulatedBufferWhenParserNeverNamesIt() {
        // The vendor-library path forwards deltas with `name: nil` always —
        // the builder extracts the name from its own accumulated buffer.
        var builder = AssistantPartsBuilder()

        _ = builder.ingest(.toolCallDelta(name: nil, argumentsDelta: #"{"name": "wri"#))
        #expect(!hasToolCallPart(builder.snapshot()))

        guard
            case .events(let events) = builder.ingest(
                .toolCallDelta(name: nil, argumentsDelta: #"te", "arguments": {"#))
        else {
            Issue.record("expected toolcallStart once the name literal closes")
            return
        }
        guard case .toolcallStart(_, let partial) = events.last else {
            Issue.record("expected toolcallStart")
            return
        }
        #expect(toolCallPart(at: 0, in: partial)?.name == "write")
    }

    @Test func nameLocksFromXMLFunctionBuffer() {
        var builder = AssistantPartsBuilder()

        guard
            case .events(let events) = builder.ingest(
                .toolCallDelta(name: nil, argumentsDelta: "<function=read><parameter=path>"))
        else {
            Issue.record("expected toolcallStart from XML function buffer")
            return
        }
        guard case .toolcallStart(_, let partial) = events.last else {
            Issue.record("expected toolcallStart")
            return
        }
        #expect(toolCallPart(at: 0, in: partial)?.name == "read")
    }

    @Test func nameLockClosesTheOpenTextPartFirst() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.text("Let me check."))

        guard
            case .events(let events) = builder.ingest(
                .toolCallDelta(name: "read", argumentsDelta: #"{"name": "read""#))
        else {
            Issue.record("expected events at name-lock")
            return
        }
        #expect(events.count == 2)
        guard case .textEnd(let textIndex, _, _) = events.first,
            case .toolcallStart(let callIndex, _) = events.last
        else {
            Issue.record("expected textEnd then toolcallStart")
            return
        }
        #expect(textIndex == 0)
        #expect(callIndex == 1)
    }

    // MARK: - Delta growth

    @Test func deltasAfterBirthEmitToolcallDeltaWithGrowingArguments() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read", "#))

        guard
            case .events(let events) = builder.ingest(
                .toolCallDelta(name: "read", argumentsDelta: #""arguments": {"path""#))
        else {
            Issue.record("expected toolcallDelta")
            return
        }
        guard case .toolcallDelta(let index, let delta, let partial) = events.first else {
            Issue.record("expected toolcallDelta event")
            return
        }
        #expect(index == 0)
        #expect(delta == #""arguments": {"path""#)
        #expect(
            toolCallPart(at: 0, in: partial)?.argumentsJSON
                == #"{"name": "read", "arguments": {"path""#)
    }

    // MARK: - Commit in place

    @Test func parsedToolCallCommitsInPlaceWithStableIdAndNormalizedArguments() {
        var builder = AssistantPartsBuilder()

        guard
            case .events(let startEvents) = builder.ingest(
                .toolCallDelta(name: "read", argumentsDelta: #"{"name": "read", "arg"#)),
            case .toolcallStart(_, let startPartial) = startEvents.last,
            let openPart = toolCallPart(at: 0, in: startPartial)
        else {
            Issue.record("expected toolcallStart")
            return
        }

        let call = GenerationFixtures.toolCall(
            name: "read", arguments: ["path": .string("notes.md")])
        guard case .events(let endEvents) = builder.ingest(.toolCall(call)) else {
            Issue.record("expected toolcallEnd events")
            return
        }
        #expect(endEvents.count == 1)
        guard case .toolcallEnd(let index, let committed, let partial) = endEvents.first else {
            Issue.record("expected toolcallEnd")
            return
        }
        #expect(index == 0)
        // Same part id from birth through commit — row identity and the Tool
        // Clock key survive.
        #expect(committed.id == openPart.id)
        #expect(committed.name == "read")
        // Committed arguments are the normalized JSON, not the raw buffer.
        #expect(committed.argumentsJSON == ToolArgumentNormalizer.encode(call.function.arguments))
        #expect(toolCallPart(at: 0, in: partial) == committed)
        #expect(builder.terminalStopReason == .toolUse)
    }

    @Test func toolCallWithoutPriorDeltasStaysAtomicStartEnd() {
        var builder = AssistantPartsBuilder()

        let call = GenerationFixtures.toolCall(name: "ls")
        guard case .events(let events) = builder.ingest(.toolCall(call)) else {
            Issue.record("expected events")
            return
        }
        #expect(events.count == 2)
        guard case .toolcallStart(0, _) = events[0], case .toolcallEnd(0, _, _) = events[1] else {
            Issue.record("expected atomic start + end pair")
            return
        }
    }

    @Test func secondToolCallOpensAFreshPartWithANewId() {
        var builder = AssistantPartsBuilder()

        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read"}"#))
        guard
            case .events(let firstEnd) = builder.ingest(
                .toolCall(GenerationFixtures.toolCall(name: "read"))),
            case .toolcallEnd(0, let first, _) = firstEnd.first
        else {
            Issue.record("expected first toolcallEnd")
            return
        }

        guard
            case .events(let secondStart) = builder.ingest(
                .toolCallDelta(name: "write", argumentsDelta: #"{"name": "write""#)),
            case .toolcallStart(let index, let partial) = secondStart.last,
            let second = toolCallPart(at: index, in: partial)
        else {
            Issue.record("expected second toolcallStart")
            return
        }
        #expect(index == 1)
        #expect(second.id != first.id)
        #expect(second.name == "write")
    }

    // MARK: - Retraction

    @Test func textAfterUnclosedToolCallRetractsTheOpenPart() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read""#))
        #expect(hasToolCallPart(builder.snapshot()))

        // Parser finalize turned the unclosed block back into raw text.
        guard case .events(let events) = builder.ingest(.text(#"<tool_call>{"name": "read""#))
        else {
            Issue.record("expected text events")
            return
        }
        guard case .textStart(0, let partial) = events.first else {
            Issue.record("expected textStart at the retracted index")
            return
        }
        #expect(!hasToolCallPart(partial))
        #expect(!hasToolCallPart(builder.snapshot()))
        #expect(builder.terminalStopReason == .stop)
    }

    @Test func thinkStartAfterUnclosedToolCallRetractsTheOpenPart() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read""#))

        guard case .events(let events) = builder.ingest(.thinkStart),
            case .thinkingStart(0, let partial) = events.first
        else {
            Issue.record("expected thinkingStart at the retracted index")
            return
        }
        #expect(!hasToolCallPart(partial))
    }

    @Test func malformedCloseRetractsTheOpenPart() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read", bad"#))

        guard
            case .malformed(let raw) = builder.ingest(.malformedToolCall(#"{"name": "read", bad"#))
        else {
            Issue.record("expected malformed step")
            return
        }
        #expect(raw == #"{"name": "read", bad"#)
        #expect(!hasToolCallPart(builder.snapshot()))
    }

    @Test func closeForTerminalRetractsTheOpenPart() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read""#))

        let events = builder.closeForTerminal()
        #expect(events.isEmpty)
        #expect(!hasToolCallPart(builder.finalize(stopReason: .stop)))
        #expect(builder.terminalStopReason == .stop)
    }

    @Test func explicitRetractionForAbortPathsRemovesThePart() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read""#))

        builder.retractOpenToolCall()
        let aborted = builder.snapshot(stopReason: .aborted)
        #expect(!hasToolCallPart(aborted))
        #expect(!aborted.hasContent)
    }

    @Test func committedToolCallSurvivesTerminalClose() {
        var builder = AssistantPartsBuilder()
        _ = builder.ingest(.toolCallDelta(name: "read", argumentsDelta: #"{"name": "read"}"#))
        _ = builder.ingest(.toolCall(GenerationFixtures.toolCall(name: "read")))

        _ = builder.closeForTerminal()
        let final = builder.finalize(stopReason: builder.terminalStopReason)
        #expect(hasToolCallPart(final))
        #expect(final.stopReason == .toolUse)
    }
}
