//
//  LivePreviewAssemblerTests.swift
//  tesseractTests
//
//  The **Live Preview**'s confirmation rule (PRD #612, ADR-0085): each
//  decode reads the audio after the last confirmed segment; when a decode
//  returns more than two segments, all but the last two are confirmed and
//  the next decode starts where they end. The rest is the provisional tail.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct LivePreviewAssemblerTests {

    private func take(seconds: Double, rate: Double = 16_000) -> AudioData {
        AudioData(
            samples: [Float](repeating: 0.1, count: Int(seconds * rate)), sampleRate: rate,
            duration: seconds)
    }

    private func decode(_ segments: [(String, Double, Double)]) -> TranscriptionResult {
        TranscriptionResult(
            text: segments.map(\.0).joined(),
            segments: segments.map {
                TranscriptionSegment(text: $0.0, startTime: $0.1, endTime: $0.2)
            },
            language: "en", processingTime: 0)
    }

    @Test func noDecodeBeforeTheMinimumAudio() {
        let assembler = LivePreviewAssembler()
        #expect(assembler.window(ofTail: take(seconds: 0.5)) == nil)
        #expect(assembler.window(ofTail: take(seconds: 0.6))?.duration == 0.6)
    }

    @Test func twoSegmentsOrFewerStayProvisional() {
        var assembler = LivePreviewAssembler()
        assembler.fold(decode([(" Ask Claude", 0, 1.2), (" why", 1.2, 1.6)]), windowDuration: 2)
        #expect(assembler.confirmed.isEmpty)
        #expect(assembler.tail == "Ask Claude why")
        #expect(assembler.confirmedEnd == 0)
        #expect(assembler.rawText == "Ask Claude why")
    }

    @Test func allButTheLastTwoSegmentsAreConfirmed() {
        var assembler = LivePreviewAssembler()
        assembler.fold(
            decode([(" Ask Claude", 0, 1.2), (" why the server", 1.2, 2.5), (" drops", 2.5, 3)]),
            windowDuration: 3)
        #expect(assembler.confirmed == "Ask Claude")
        #expect(assembler.tail == "why the server drops")
        #expect(assembler.confirmedEnd == 1.2)
    }

    @Test func theNextDecodeStartsWhereTheConfirmedSegmentsEnd() throws {
        var assembler = LivePreviewAssembler()
        assembler.fold(
            decode([(" Ask Claude", 0, 1.2), (" why the server", 1.2, 2.5), (" drops", 2.5, 3)]),
            windowDuration: 3)
        // The next decode reads from where the confirmed segments end.
        #expect(assembler.confirmedEnd == 1.2)
        let window = try #require(assembler.window(ofTail: take(seconds: 2.8)))

        // Times in the next decode are relative to its window.
        assembler.fold(
            decode([(" why the server", 0, 1.3), (" drops the", 1.3, 2.2), (" first", 2.2, 2.8)]),
            windowDuration: window.duration)
        #expect(assembler.confirmed == "Ask Claude why the server")
        #expect(abs(assembler.confirmedEnd - 2.5) < 0.001)
        #expect(assembler.tail == "drops the first")
    }

    @Test func aLongUnconfirmedStretchConfirmsAllButItsLastSegment() {
        var assembler = LivePreviewAssembler()
        assembler.fold(
            decode([(" a long sentence", 0, 12), (" still going", 12, 22)]),
            windowDuration: LivePreviewAssembler.maximumUnconfirmed + 2)
        #expect(assembler.confirmed == "a long sentence")
        #expect(assembler.tail == "still going")
        #expect(assembler.confirmedEnd == 12)
    }

    @Test func aDecodeWithoutSegmentsKeepsItsTextAsTheTail() {
        var assembler = LivePreviewAssembler()
        assembler.fold(
            TranscriptionResult(text: " Hello", segments: [], language: "en", processingTime: 0),
            windowDuration: 1)
        #expect(assembler.tail == "Hello")
        #expect(assembler.confirmed.isEmpty)
    }

    @Test func aSegmentEndPastTheWindowIsClampedToIt() {
        var assembler = LivePreviewAssembler()
        assembler.fold(
            decode([(" one", 0, 5), (" two", 5, 6), (" three", 6, 7)]), windowDuration: 3)
        #expect(assembler.confirmedEnd == 3)
    }

    @Test func aLongStretchWithOneSegmentIsConfirmedWhole() {
        var assembler = LivePreviewAssembler()
        let long = LivePreviewAssembler.maximumUnconfirmed + 2
        assembler.fold(decode([(" one long sentence", 0, long - 1)]), windowDuration: long)
        #expect(assembler.confirmed == "one long sentence")
        #expect(assembler.tail.isEmpty)
        #expect(assembler.confirmedEnd == long - 1)
    }

    @Test func aLongStretchWithNoSegmentsMovesOn() {
        var assembler = LivePreviewAssembler()
        let long = LivePreviewAssembler.maximumUnconfirmed + 2
        assembler.fold(
            TranscriptionResult(text: " hello", segments: [], language: "en", processingTime: 0),
            windowDuration: long)
        #expect(assembler.confirmed == "hello")
        #expect(assembler.confirmedEnd == long)
    }
}
