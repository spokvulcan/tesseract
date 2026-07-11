//
//  SpeechRecognizerTests.swift
//  tesseractTests
//
//  Light tests that the in-memory Speech Recognizer adapter honors its
//  canned-result / programmed-error / call-recording contract, so tests built on
//  it aren't a source of false confidence. (The production
//  `WhisperKitSpeechRecognizer` adapter is exercised by the loaded-model suites,
//  not here — it needs real model files.)
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SpeechRecognizerTests {

    @Test
    func returnsCannedResultAndRecordsLoadAndTranscribe() async throws {
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "canned", segments: [], language: "fr", processingTime: 0)
        )
        let path = URL(fileURLWithPath: "/tmp/model.mlmodelc")

        try await recognizer.load(modelPath: path)
        let result = try await recognizer.transcribe(
            AudioData(samples: [0.1], sampleRate: 16_000, duration: 1), language: "fr"
        )

        #expect(result.text == "canned")
        #expect(await recognizer.loadCount == 1)
        #expect(await recognizer.loadedPaths == [path])
        #expect(await recognizer.transcribeCount == 1)
        #expect(await recognizer.recordedLanguages == ["fr"])
    }

    @Test
    func throwsProgrammedTranscribeError() async throws {
        let recognizer = InMemorySpeechRecognizer(transcribeError: FakeModelError(message: "boom"))
        try await recognizer.load(modelPath: URL(fileURLWithPath: "/tmp/m"))

        await #expect(throws: FakeModelError.self) {
            _ = try await recognizer.transcribe(
                AudioData(samples: [0.1], sampleRate: 16_000, duration: 1), language: nil
            )
        }
    }

    @Test
    func throwsProgrammedLoadError() async {
        let recognizer = InMemorySpeechRecognizer(loadError: FakeModelError(message: "no weights"))
        await #expect(throws: FakeModelError.self) {
            try await recognizer.load(modelPath: URL(fileURLWithPath: "/tmp/m"))
        }
    }
}
