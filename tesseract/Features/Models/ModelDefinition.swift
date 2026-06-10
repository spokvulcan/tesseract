//
//  ModelDefinition.swift
//  tesseract
//

import Foundation

enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case speechToText = "Speech-to-Text"
    case textToSpeech = "Text-to-Speech"
    case agent = "Agent"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .speechToText: "mic.fill"
        case .textToSpeech: "speaker.wave.3.fill"
        case .agent: "brain"
        }
    }
}

enum ModelSource: Sendable {
    case huggingFace(repo: String, requiredExtension: String, pathPrefix: String? = nil)
}

/// A file the model needs at inference time that lives in a *different*
/// Hugging Face repo than the weights (e.g. WhisperKit reads the tokenizer
/// from the original OpenAI repo). Downloaded into the model folder so
/// loading never falls back to a network fetch — the app stays offline-
/// deterministic after the explicit model download.
struct CompanionFile: Sendable {
    let repo: String
    let path: String
}

struct ModelDefinition: Identifiable, Sendable {
    let id: String
    let displayName: String
    let description: String
    let category: ModelCategory
    let source: ModelSource
    let sizeDescription: String
    let dependencies: [String]
    var companionFiles: [CompanionFile] = []

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
    static let defaultAgentModelID = "qwen3.5-4b-paro"

    static let all: [ModelDefinition] = [
        ModelDefinition(
            id: "whisper-large-v3-turbo",
            displayName: "Whisper Turbo",
            description: "Fast and accurate voice-to-text. Supports 100+ languages.",
            category: .speechToText,
            source: .huggingFace(
                repo: "argmaxinc/whisperkit-coreml",
                requiredExtension: "mlmodelc",
                pathPrefix: "openai_whisper-large-v3-v20240930_turbo"
            ),
            sizeDescription: "~1.5 GB",
            dependencies: [],
            // WhisperKit resolves the large-v3 tokenizer from the OpenAI repo;
            // bundling it into the model folder keeps first dictation offline.
            companionFiles: [
                CompanionFile(repo: "openai/whisper-large-v3", path: "tokenizer.json"),
                CompanionFile(repo: "openai/whisper-large-v3", path: "tokenizer_config.json"),
            ]
        ),
        ModelDefinition(
            id: "qwen3-tts-voicedesign",
            displayName: "Voice Engine",
            description: "Natural-sounding text-to-speech with customizable voice.",
            category: .textToSpeech,
            source: .huggingFace(
                repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~4.2 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-4b-paro",
            displayName: "Qwen3.5-4B PARO",
            description: "Compact and fast agent. Great balance of quality and speed.",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.5-4B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~3.5 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-9b-paro",
            displayName: "Qwen3.5-9B PARO",
            description: "Larger, smarter agent. Needs more memory (~10 GB with voice).",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.5-9B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~8 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-27b-paro",
            displayName: "Qwen3.5-27B PARO",
            description: "Highest-quality PARO agent. Requires a high-memory Mac.",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.5-27B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~19 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-4b",
            displayName: "Qwen3.5-4B",
            description: "Standard agent model. Reliable general-purpose assistant.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.5-4B-MLX-8bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~5 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.6-35b-a3b-ud",
            displayName: "Qwen3.6-35B-A3B UD (MLX 4bit)",
            description: "Large sparse-MoE agent (3B active / 35B total) from Unsloth. Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5_moe.",
            category: .agent,
            source: .huggingFace(
                repo: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~20 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.6-27b",
            displayName: "Qwen3.6-27B (MLX 4bit)",
            description: "Dense 27B agent (uniform 4-bit MLX). Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.6-27B-4bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~16 GB",
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
