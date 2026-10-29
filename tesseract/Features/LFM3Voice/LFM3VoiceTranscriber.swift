//
//  LFM3VoiceTranscriber.swift
//  tesseract
//
//  **LFM3 Voice**'s ear behind the `Transcribing` port: its Voice Capture
//  Session runs every stage of a take with LFM3-Audio as the recognizer.
//

import Foundation

@MainActor
final class LFM3VoiceTranscriber: Transcribing {
    private let engine: LFM3VoiceEngine

    init(engine: LFM3VoiceEngine) {
        self.engine = engine
    }

    /// `language` is the dictation language's code; a code with no English
    /// name ("auto") lets the model tell the language itself.
    func transcribe(_ audioData: AudioData, language: String) async throws -> TranscriptionResult {
        let start = Date()
        let text = try await engine.transcribe(
            audioData.samples, sampleRate: audioData.sampleRate,
            language: Locale(identifier: "en").localizedString(forLanguageCode: language))
        return TranscriptionResult(
            text: text, segments: [], language: language,
            processingTime: Date().timeIntervalSince(start))
    }

    /// The take's task is the one channel: cancelling it stops the decode
    /// within a token.
    func cancelTranscription() {}
}
