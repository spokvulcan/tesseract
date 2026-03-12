//
//  ModelDefinition.swift
//  tesseract
//

import Foundation

enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case speechToText = "Speech-to-Text"
    case textToSpeech = "Text-to-Speech"
    case agent = "Agent"
    case imageGeneration = "Image Generation"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .speechToText: "mic.fill"
        case .textToSpeech: "speaker.wave.3.fill"
        case .agent: "brain"
        case .imageGeneration: "photo.fill"
        }
    }
}

enum ModelSource: Sendable {
    case huggingFace(repo: String, requiredExtension: String, pathPrefix: String? = nil)
}

struct ModelDefinition: Identifiable, Sendable {
    let id: String
    let displayName: String
    let description: String
    let category: ModelCategory
    let source: ModelSource
    let sizeDescription: String
    let dependencies: [String]

    var cacheSubdirectory: String? {
        guard case .huggingFace(let repo, _, _) = source else { return nil }
        return repo.replacingOccurrences(of: "/", with: "_")
    }

    var requiredExtension: String? {
        guard case .huggingFace(_, let ext, _) = source else { return nil }
        return ext
    }

    var repoID: String? {
        guard case .huggingFace(let repo, _, _) = source else { return nil }
        return repo
    }

    var pathPrefix: String? {
        guard case .huggingFace(_, _, let prefix) = source else { return nil }
        return prefix
    }
}

extension ModelDefinition {
    static let defaultAgentModelID = "qwen3.5-4b"

    // Image generation models are kept here but excluded from `all` until the feature is ready.
    static let all: [ModelDefinition] = [
        ModelDefinition(
            id: "whisper-large-v3-turbo",
            displayName: "Whisper Large V3 Turbo",
            description: "State-of-the-art multilingual transcription model. 100+ languages.",
            category: .speechToText,
            source: .huggingFace(
                repo: "argmaxinc/whisperkit-coreml",
                requiredExtension: "mlmodelc",
                pathPrefix: "openai_whisper-large-v3-v20240930_turbo"
            ),
            sizeDescription: "~1.5 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3-tts-voicedesign",
            displayName: "Qwen3-TTS VoiceDesign",
            description: "1.7B parameter text-to-speech with voice design capabilities.",
            category: .textToSpeech,
            source: .huggingFace(
                repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~3.6 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "nanbeige4.1-3b",
            displayName: "Nanbeige4.1-3B",
            description: "3B parameter bilingual agent model with tool calling and reasoning.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Nanbeige4.1-3B-8bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~4.2 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-4b",
            displayName: "Qwen3.5-4B",
            description: "4B hybrid-attention model with 256K context. Tool calling with think blocks.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.5-4B-MLX-8bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~5 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-4b-paro",
            displayName: "Qwen3.5-4B PARO (INT4)",
            description: "ParoQuant INT4 — near-FP16 quality at half the size of 8-bit. Hybrid-attention, 256K context.",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.5-4B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~3.8 GB",
            dependencies: []
        ),
    ]

    static let imageGenerationModels: [ModelDefinition] = [
        ModelDefinition(
            id: "flux2-klein-4b",
            displayName: "FLUX.2-klein-4B",
            description: "4B parameter distilled image generation. 4-step inference, 1024×1024.",
            category: .imageGeneration,
            source: .huggingFace(
                repo: "black-forest-labs/FLUX.2-klein-4B",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~8 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "z-image",
            displayName: "Z-Image",
            description: "6B parameter image generation from Tongyi Lab. 50-step inference, CFG guidance, excellent text rendering.",
            category: .imageGeneration,
            source: .huggingFace(
                repo: "Tongyi-MAI/Z-Image",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~12 GB",
            dependencies: []
        ),
    ]

    static func byCategory() -> [(ModelCategory, [ModelDefinition])] {
        ModelCategory.allCases.compactMap { category in
            let models = all.filter { $0.category == category }
            return models.isEmpty ? nil : (category, models)
        }
    }
}
