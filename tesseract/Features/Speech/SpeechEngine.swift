//
//  SpeechEngine.swift
//  tesseract
//

import Foundation
import Observation

@Observable @MainActor
final class SpeechEngine {
    private enum Defaults {
        static let modelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    }

    private(set) var isModelLoaded = false
    private(set) var isLoading = false
    private(set) var loadingStatus: String = ""

    private let makeSynthesizer: @Sendable () -> any SpeechSynthesizer
    private var synthesizer: (any SpeechSynthesizer)?

    init(
        makeSynthesizer: @escaping @Sendable () -> any SpeechSynthesizer = {
            Qwen3SpeechSynthesizer()
        }
    ) {
        self.makeSynthesizer = makeSynthesizer
    }

    func loadModel() async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Downloading model..."

        do {
            // Build a fresh adapter via the factory and load it. The default model
            // repo stays on the facade — the port is repo-agnostic.
            let synthesizer = makeSynthesizer()
            try await synthesizer.load(modelRepo: Defaults.modelRepo)
            self.synthesizer = synthesizer
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
        synthesizer = nil
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
        guard let synthesizer else {
            throw SpeechError.modelNotLoaded
        }

        return try await synthesizer.generate(
            text: text,
            voice: voice,
            language: language,
            parameters: parameters
        )
    }

    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters,
        useVoiceAnchor: Bool = false
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int) {
        guard let synthesizer else {
            throw SpeechError.modelNotLoaded
        }

        return try await synthesizer.generateStreaming(
            text: text,
            voice: voice,
            language: language,
            parameters: parameters,
            useVoiceAnchor: useVoiceAnchor
        )
    }

    func buildVoiceAnchor(
        referenceCount: Int,
        voice: String?,
        language: String?
    ) async {
        guard let synthesizer else { return }
        await synthesizer.buildVoiceAnchor(
            referenceCount: referenceCount,
            voice: voice,
            language: language
        )
    }

    func clearVoiceAnchor() async {
        guard let synthesizer else { return }
        await synthesizer.clearVoiceAnchor()
    }

    func cancelGeneration() async {
        guard let synthesizer else { return }
        await synthesizer.cancelGeneration()
    }

    func computeTokenCharOffsets(text: String) async -> [Int] {
        guard let synthesizer else { return [] }
        return await synthesizer.computeTokenCharOffsets(text: text)
    }
}
