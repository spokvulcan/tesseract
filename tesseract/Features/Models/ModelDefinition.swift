//
//  ModelDefinition.swift
//  tesseract
//

import Foundation

enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case speechToText = "Speech-to-Text"
    case textToSpeech = "Text-to-Speech"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .speechToText: "mic.fill"
        case .textToSpeech: "speaker.wave.3.fill"
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
            dependencies: ["snac-24khz"]
        ),
        ModelDefinition(
            id: "snac-24khz",
            displayName: "SNAC 24kHz Codec",
            description: "Neural audio codec for decoding TTS output. Required by Qwen3-TTS.",
            category: .textToSpeech,
            source: .huggingFace(
                repo: "mlx-community/snac_24khz",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~87 MB",
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
