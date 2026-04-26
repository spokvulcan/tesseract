//
//  ModelDefinition.swift
//  tesseract
//

import Foundation

enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case speechToText = "Speech-to-Text"
    case textToSpeech = "Text-to-Speech"
    case agent = "Agent"
    case draft = "Draft"
    case imageGeneration = "Image Generation"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .speechToText: "mic.fill"
        case .textToSpeech: "speaker.wave.3.fill"
        case .agent: "brain"
        case .draft: "bolt.fill"
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
    /// Optional pointer to a DFlash speculative-decoding draft for this
    /// target. The draft is its own catalog entry (gated download); we
    /// don't list it in `dependencies` so the target can be downloaded
    /// without forcing draft acceptance. Resolution happens at load time
    /// in `AgentEngine.resolveDFlashLoadConfig` when the user enables
    /// DFlash in settings.
    var dflashDraftID: String? = nil

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

    // Image generation models are kept here but excluded from `all` until the feature is ready.
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
            dependencies: []
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
            dependencies: [],
            dflashDraftID: "qwen3.6-27b-dflash"
        ),
        ModelDefinition(
            id: "qwen3.6-27b-dflash",
            displayName: "Qwen3.6-27B DFlash Draft",
            description: "Block-diffusion speculative decoding draft for Qwen3.6-27B. BF16 (~3.2 GB). Requires gate acceptance at https://huggingface.co/z-lab/Qwen3.6-27B-DFlash before download.",
            category: .draft,
            source: .huggingFace(
                repo: "z-lab/Qwen3.6-27B-DFlash",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~3.2 GB",
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
