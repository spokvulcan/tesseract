//
//  SpeechEngine.swift
//  tesseract
//

import Foundation
import Combine
import os
import MLX
import MLXLMCommon
import MLXAudioTTS

@MainActor
final class SpeechEngine: ObservableObject {
    private enum Defaults {
        static let modelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    }

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadingStatus: String = ""

    private var ttsActor: TTSActor?

    func loadModel() async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Downloading model..."

        do {
            let actor = TTSActor()
            try await actor.loadModel(repo: Defaults.modelRepo)
            ttsActor = actor
            isModelLoaded = true
            loadingStatus = ""
            Log.speech.info("TTS model loaded successfully")
        } catch {
            loadingStatus = ""
            isLoading = false
            Log.speech.error("Failed to load TTS model: \(error)")
            throw SpeechError.modelLoadFailed(error.localizedDescription)
        }

        isLoading = false
    }

    func unloadModel() {
        ttsActor = nil
        isModelLoaded = false
        loadingStatus = ""
        Log.speech.info("TTS model unloaded")
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (samples: [Float], sampleRate: Int) {
        guard let actor = ttsActor else {
            throw SpeechError.modelNotLoaded
        }

        return try await actor.generate(
            text: text,
            voice: voice,
            language: language,
            parameters: parameters
        )
    }
}

actor TTSActor {
    private nonisolated(unsafe) var model: (any SpeechGenerationModel)?

    func loadModel(repo: String) async throws {
        let loadedModel = try await TTSModelUtils.loadModel(modelRepo: repo)
        model = loadedModel
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> ([Float], Int) {
        guard let model else {
            throw SpeechError.modelNotLoaded
        }

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize
        )

        let audioArray = try await model.generate(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: language,
            generationParameters: genParams
        )

        let samples = audioArray.asArray(Float.self)
        return (samples, model.sampleRate)
    }
}
