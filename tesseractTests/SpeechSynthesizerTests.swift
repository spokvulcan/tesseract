//
//  SpeechSynthesizerTests.swift
//  tesseractTests
//
//  Light tests that the in-memory Speech Synthesizer adapter honors its
//  canned-result / call-recording contract, so tests built on it aren't a source
//  of false confidence. (The production `Qwen3SpeechSynthesizer` adapter is
//  exercised by the loaded-model suites, not here — it needs the real TTS model.)
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SpeechSynthesizerTests {

    @Test
    func returnsCannedSamplesAndRecordsLoadAndGenerate() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.1, 0.2, 0.3], sampleRate: 24_000)

        try await synth.load(modelRepo: "some/repo")
        let (samples, sampleRate) = try await synth.generate(
            text: "hello", voice: "narrator", language: "en", parameters: .default
        )

        #expect(samples == [0.1, 0.2, 0.3])
        #expect(sampleRate == 24_000)
        #expect(await synth.loadCount == 1)
        #expect(await synth.loadedRepos == ["some/repo"])
        #expect(await synth.generateCount == 1)
        #expect(await synth.recordedTexts == ["hello"])
    }

    @Test
    func throwsProgrammedLoadError() async {
        let synth = InMemorySpeechSynthesizer(loadError: FakeModelError(message: "no weights"))
        await #expect(throws: FakeModelError.self) {
            try await synth.load(modelRepo: "some/repo")
        }
    }

    @Test
    func throwsProgrammedGenerateErrorAfterRecordingTheCall() async {
        let synth = InMemorySpeechSynthesizer(generateError: FakeModelError(message: "boom"))
        await #expect(throws: FakeModelError.self) {
            _ = try await synth.generate(text: "t", voice: "v", language: "l", parameters: .default)
        }
        // The call is recorded even when it fails, so passthrough is observable.
        #expect(await synth.generateCount == 1)
        #expect(await synth.recordedVoices == ["v"])
        #expect(await synth.recordedLanguages == ["l"])
    }

    @Test
    func streamingYieldsCannedChunksAndRecordsTheCall() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.9, 1.0], sampleRate: 16_000)

        let (stream, sampleRate) = try await synth.generateStreaming(
            text: "stream", voice: nil, language: nil, parameters: .default, useVoiceAnchor: true
        )
        var chunks: [[Float]] = []
        for try await chunk in stream { chunks.append(chunk) }

        #expect(chunks == [[0.9, 1.0]])
        #expect(sampleRate == 16_000)
        #expect(await synth.streamingCount == 1)
        #expect(await synth.recordedStreamingTexts == ["stream"])
        #expect(await synth.recordedUseVoiceAnchor == [true])
    }

    @Test
    func programmedStreamingErrorTerminatesTheStream() async throws {
        let synth = InMemorySpeechSynthesizer(streamingError: FakeModelError(message: "decode failed"))

        let (stream, _) = try await synth.generateStreaming(
            text: "t", voice: nil, language: nil, parameters: .default, useVoiceAnchor: false
        )
        await #expect(throws: FakeModelError.self) {
            for try await _ in stream {}
        }
    }

    @Test
    func recordsVoiceAnchorCancelAndOffsets() async {
        let synth = InMemorySpeechSynthesizer(tokenOffsets: [1, 4])

        await synth.buildVoiceAnchor(referenceCount: 3, voice: nil, language: nil)
        await synth.clearVoiceAnchor()
        await synth.cancelGeneration()
        let offsets = await synth.computeTokenCharOffsets(text: "abc")

        #expect(await synth.buildVoiceAnchorCount == 1)
        #expect(await synth.recordedReferenceCounts == [3])
        #expect(await synth.clearVoiceAnchorCount == 1)
        #expect(await synth.cancelCount == 1)
        #expect(offsets == [1, 4])
        #expect(await synth.recordedOffsetTexts == ["abc"])
    }
}
