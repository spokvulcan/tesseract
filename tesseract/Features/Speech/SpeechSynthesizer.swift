//
//  SpeechSynthesizer.swift
//  tesseract
//
//  The model port (seam) for speech synthesis (TTS), sitting *below* the
//  `SpeechEngine` facade. See CONTEXT.md → Language → "Speech model ports and
//  playback". Mirrors the `SpeechRecognizer` port on the recognition side.
//
//  The port is `Sendable`: the adapter is actor-backed, so synthesis runs off the
//  main actor while `SpeechEngine` (`@Observable @MainActor`) stays free to drive
//  UI state. Above the port stay the default model repo and the `@Observable`
//  load lifecycle (`isModelLoaded` / `isLoading` / `loadingStatus`). Below the
//  port: model load + inference / streaming / voice-anchor / token alignment
//  only. What crosses: text + voice + language + `TTSParameters` in (all
//  `Sendable`), `[Float]` samples (plus a sample rate) or an
//  `AsyncThrowingStream<[Float], Error>` out.
//
//  Error contract: `SpeechEngine` maps only *load* failures onto a `SpeechError`
//  (`.modelLoadFailed`). `generate` / `generateStreaming` are passthroughs — a
//  failure from the model surfaces unchanged (synchronously for `generate`, or as
//  the stream's terminating error for `generateStreaming`), not re-wrapped.
//

import Foundation

// `nonisolated` so the actor adapter conforms off the main actor (the build uses
// MainActor-by-default isolation, which would otherwise infer `@MainActor` here
// and force the synthesizer onto the main actor — defeating the point of an
// actor-backed, `Sendable` model port).
nonisolated protocol SpeechSynthesizer: Sendable {
    /// Loads the synthesis model from a Hugging Face-style repo identifier.
    func load(modelRepo: String) async throws

    /// Synthesizes `text` to audio samples in one shot. `voice` / `language` are
    /// `nil` for model defaults.
    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (samples: [Float], sampleRate: Int)

    /// Synthesizes `text` to a stream of audio-sample chunks.
    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters,
        useVoiceAnchor: Bool
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int)

    /// Builds a voice anchor from the first `referenceCount` generated chunks so
    /// later long-form segments stay timbre-consistent.
    func buildVoiceAnchor(referenceCount: Int, voice: String?, language: String?) async

    /// Discards any voice anchor built by `buildVoiceAnchor`.
    func clearVoiceAnchor() async

    /// Cooperatively cancels in-flight generation.
    func cancelGeneration() async

    /// Token → character offsets for the given text, for word-level alignment.
    func computeTokenCharOffsets(text: String) async -> [Int]
}
