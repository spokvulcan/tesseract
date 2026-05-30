//
//  Qwen3SpeechSynthesizer.swift
//  tesseract
//
//  The framework-backed Speech Synthesizer adapter — the only production code
//  that touches MLXAudioTTS for synthesis. Formerly `TTSActor`. An actor, so
//  `Sendable` is free and synthesis runs off the main actor below the
//  `SpeechEngine` facade.
//

import Foundation
import MLX
import MLXLMCommon
import MLXAudioTTS
import MLXAudioCore

actor Qwen3SpeechSynthesizer: SpeechSynthesizer {
    // Actor-isolated: every method that touches `model` is actor-isolated, and the
    // sole `nonisolated` helper (`extractSamples`) never reads it — so this needs
    // no `nonisolated(unsafe)` escape hatch. (The former `TTSActor` carried one;
    // it was unnecessary.)
    private var model: (any SpeechGenerationModel)?
    private static let ttsCacheLimitMB: Int = {
        let raw = ProcessInfo.processInfo.environment["QWEN3TTS_CACHE_LIMIT_MB"] ?? ""
        return max(16, Int(raw) ?? 100)
    }()

    func load(modelRepo: String) async throws {
        let loadedModel = try await TTSModelUtils.loadModel(modelRepo: modelRepo)
        model = loadedModel
    }

    private func configureTTSMemoryCacheLimit() {
        Memory.cacheLimit = Self.ttsCacheLimitMB * 1024 * 1024
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (samples: [Float], sampleRate: Int) {
        guard let model else {
            throw SpeechError.modelNotLoaded
        }

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty
        )

        configureTTSMemoryCacheLimit()
        model.seed = parameters.seed
        MLXRandom.seed(parameters.seed)
        let audioArray = try await model.generate(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: language,
            generationParameters: genParams
        )

        let samples = audioArray.asArray(Float.self)
        Memory.clearCache()
        return (samples, model.sampleRate)
    }

    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters,
        useVoiceAnchor: Bool
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int) {
        guard let model else {
            throw SpeechError.modelNotLoaded
        }

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty
        )

        configureTTSMemoryCacheLimit()
        model.seed = parameters.seed
        MLXRandom.seed(parameters.seed)
        let sampleRate = model.sampleRate
        let modelStream = model.generateStream(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: language,
            generationParameters: genParams,
            useVoiceAnchor: useVoiceAnchor
        )

        let outputStream = convertAudioStream(modelStream)
        return (outputStream, sampleRate)
    }

    func buildVoiceAnchor(referenceCount: Int, voice: String?, language: String?) {
        guard let model else { return }
        model.buildVoiceAnchor(
            referenceCount: referenceCount,
            instruct: voice,
            language: language
        )
    }

    func clearVoiceAnchor() {
        guard let model else { return }
        model.clearVoiceAnchor()
    }

    func cancelGeneration() {
        guard let model else { return }
        model.cancelGeneration()
    }

    func computeTokenCharOffsets(text: String) -> [Int] {
        guard let model else { return [] }
        return model.tokenizeForAlignment(text: text)
    }

    private func convertAudioStream(
        _ modelStream: AsyncThrowingStream<AudioGeneration, Error>
    ) -> AsyncThrowingStream<[Float], Error> {
        let (stream, continuation) = AsyncThrowingStream<[Float], Error>.makeStream()

        Task {
            do {
                for try await event in modelStream {
                    extractSamples(from: event, continuation: continuation)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    private nonisolated func extractSamples(
        from event: AudioGeneration,
        continuation: AsyncThrowingStream<[Float], Error>.Continuation
    ) {
        switch event {
        case .audioChunk(let chunk):
            let samples = chunk.asArray(Float.self)
            if !samples.isEmpty {
                continuation.yield(samples)
            }
        case .audio(let audio):
            let samples = audio.asArray(Float.self)
            if !samples.isEmpty {
                continuation.yield(samples)
            }
        case .token, .info:
            break
        }
    }
}
