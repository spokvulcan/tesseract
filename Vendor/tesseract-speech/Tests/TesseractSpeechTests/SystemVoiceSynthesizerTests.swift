// The system voice behind the Speech Synthesizer port: its audio in the
// engine's frames, its word marks as the engine's word starts.

import Foundation
import Testing
@testable import TesseractSpeech

/// Plays back a fixed list of pieces, as a renderer would send them.
private struct ScriptedRenderer: SystemSpeechRendering {
    let pieces: [SystemSpeechPiece]

    func render(_ text: String, language: String?) -> AsyncThrowingStream<SystemSpeechPiece, Error> {
        AsyncThrowingStream { continuation in
            for piece in pieces { continuation.yield(piece) }
            continuation.finish()
        }
    }
}

private func request(_ text: String) -> SegmentRequest {
    SegmentRequest(
        text: text, voiceDescription: nil, language: "English", parameters: TTSParameters(),
        seed: 0)
}

private func events(_ text: String, _ pieces: [SystemSpeechPiece]) async throws -> [SynthesisEvent] {
    let synthesizer = SystemVoiceSynthesizer(renderer: ScriptedRenderer(pieces: pieces))
    var result: [SynthesisEvent] = []
    for try await event in await synthesizer.synthesizeSegment(request(text)) {
        result.append(event)
    }
    return result
}

private func samples(_ events: [SynthesisEvent]) -> [Float] {
    events.flatMap { event -> [Float] in
        if case .chunk(let samples) = event { return samples }
        return []
    }
}

@Suite struct SystemVoiceSynthesizerTests {

    @Test func audioAtTheEngineRateIsPaddedToWholeFrames() async throws {
        let tone = (0..<5_000).map { Float(sin(Double($0) * 0.05)) }
        let result = try await events("Hello.", [.audio(tone, sampleRate: 24_000)])
        let out = samples(result)
        // 5,000 samples, then silence to the end of their third frame.
        #expect(out.count == 3 * 1_920)
        #expect(Array(out.prefix(5_000)) == tone)
        #expect(out.dropFirst(5_000).allSatisfy { $0 == 0 })
        guard case .done(let take) = result.last else {
            Issue.record("the segment ends with done")
            return
        }
        #expect(take == nil, "the system voice never makes a Reference Take")
    }

    @Test func otherRatesAreResampledToTheEngines() async throws {
        // One second at the system voice's usual 22,050 Hz.
        let second = (0..<22_050).map { Float(sin(Double($0) * 2 * .pi * 220 / 22_050)) * 0.5 }
        let out = samples(try await events("Hello.", [.audio(second, sampleRate: 22_050)]))
        #expect(out.count % 1_920 == 0)
        // A second at 24 kHz, rounded up to a whole frame.
        #expect(out.count >= 24_000 - 1_920 && out.count <= 24_000 + 1_920)
        // Still a 220 Hz tone of the same loudness: count rising zero crossings.
        let body = Array(out.prefix(23_000))
        let crossings = zip(body, body.dropFirst()).filter { $0 < 0 && $1 >= 0 }.count
        #expect(abs(crossings - 211) <= 3)
        #expect(abs((body.map { $0 * $0 }.reduce(0, +) / Float(body.count)).squareRoot() - 0.354) < 0.02)
    }

    @Test func wordMarksBecomeWordStartsAfterTheirAudio() async throws {
        // "Hello there, world." — words 0, 1, 2 at UTF-16 offsets 0, 6, 13.
        let text = "Hello there, world."
        let frame = 1_920
        let result = try await events(
            text,
            [
                .audio([Float](repeating: 0.1, count: frame * 2), sampleRate: 24_000),
                .word(NSRange(location: 0, length: 5), sample: 0),
                .word(NSRange(location: 6, length: 6), sample: frame + 10),
                // In the third frame, before its audio has come.
                .word(NSRange(location: 13, length: 6), sample: frame * 2 + 100),
                .audio([Float](repeating: 0.1, count: frame), sampleRate: 24_000),
            ])
        var framesSent = 0
        var starts: [WordStart] = []
        for event in result {
            switch event {
            case .chunk(let samples): framesSent += samples.count / frame
            case .words(let words):
                for word in words { #expect(word.frame < framesSent, "never ahead of its audio") }
                starts += words
            case .done: break
            }
        }
        #expect(starts == [
            WordStart(word: 0, frame: 0), WordStart(word: 1, frame: 1), WordStart(word: 2, frame: 2),
        ])
    }

    @Test func aWordMarkedTwiceOrOutOfOrderIsTimedOnce() async throws {
        let result = try await events(
            "One two three.",
            [
                .audio([Float](repeating: 0.1, count: 1_920 * 3), sampleRate: 24_000),
                .word(NSRange(location: 4, length: 3), sample: 1_920),
                .word(NSRange(location: 5, length: 2), sample: 1_950),
                .word(NSRange(location: 0, length: 3), sample: 0),
            ])
        let starts = result.flatMap { event -> [WordStart] in
            if case .words(let words) = event { return words }
            return []
        }
        #expect(starts == [WordStart(word: 1, frame: 1)])
    }

    @Test func wordsSplitAsTheEngineSplitsThem() {
        let offsets = WordOffsets("  Hello\nworld,  again ")
        #expect(offsets.starts == [2, 8, 16])
        #expect(offsets.word(at: 0) == nil)
        #expect(offsets.word(at: 9) == 1)
        #expect(offsets.word(at: 20) == 2)
    }

    /// Through the engine, the system voice's segments start on exact frames.
    @Test func theEngineCountsItsFramesExactly() async throws {
        let renderer = ScriptedRenderer(pieces: [.audio([Float](repeating: 0.1, count: 3_000), sampleRate: 24_000)])
        let engine = SpeechEngine(
            model: .voiceDesign17B(.q8), synthesizer: SystemVoiceSynthesizer(renderer: renderer))
        let session = try await engine.session(
            SessionProfile(reference: .none, pacing: .eager), voice: .standard(language: "English"))
        let text = Array(repeating: "A sentence that is long enough to stand on its own here.", count: 40)
            .joined(separator: " ")
        let utterance = try await session.speak(text)
        var nextFrame = 0
        var segments = 0
        for try await event in utterance.events {
            switch event {
            case .segment(let script):
                #expect(script.startFrame == nextFrame)
                segments += 1
            case .audio(let chunk):
                nextFrame = chunk.frames.upperBound
            default: break
            }
        }
        #expect(segments > 1)
        #expect(nextFrame == segments * 2, "3,000 samples pad to two frames per segment")
    }
}
