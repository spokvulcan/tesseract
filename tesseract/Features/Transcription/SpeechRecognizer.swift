//
//  SpeechRecognizer.swift
//  tesseract
//
//  The model port (seam) for speech recognition (ASR), sitting *below* the
//  `TranscriptionEngine` facade. See CONTEXT.md → Language → "Speech model ports
//  and playback".
//
//  The port is `Sendable`: `TranscriptionEngine` races `transcribe` against a
//  timeout in an unstructured recognizer task, so the recognizer crosses a
//  `@Sendable` boundary. Above the port stay the timeout race, lazy
//  `ensureModelLoaded`, model-file (`.mlmodelc`) verification, the `@Observable`
//  lifecycle state, and the mapping of model failures onto `DictationError`.
//  Below the port: model load + inference only. What crosses: `AudioData` in,
//  `TranscriptionResult` out (both `Sendable`).
//

import Foundation

// `nonisolated` so actor adapters conform off the main actor (the build uses
// MainActor-by-default isolation, which would otherwise infer `@MainActor` here
// and force the recognizer onto the main actor — defeating the whole point of an
// actor-backed, `Sendable` model port raced inside a `withThrowingTaskGroup`).
nonisolated protocol SpeechRecognizer: Sendable {
    /// Loads the recognition model from a local `.mlmodelc` folder URL.
    func load(modelPath: URL) async throws

    /// Transcribes the given audio. `language` is `nil` for auto-detect.
    ///
    /// Cancellation is `Task` cancellation propagating into this call — the
    /// port's one cancellation channel (the former separate `cancel()`
    /// requirement was an empty body in the WhisperKit adapter and is gone,
    /// audit #285 item 10). Adapters stop at their next cooperative check;
    /// the engine's budget does not wait for them.
    func transcribe(_ audioData: AudioData, language: String?) async throws -> TranscriptionResult
}
