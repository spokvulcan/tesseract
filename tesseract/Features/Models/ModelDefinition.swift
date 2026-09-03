//
//  ModelDefinition.swift
//  tesseract
//

import Foundation

enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case speechToText = "Speech-to-Text"
    case textToSpeech = "Text-to-Speech"
    case agent = "Agent"
    /// The dictation **Proofread Pass**'s small co-resident model (ADR-0034)
    /// — its own category so it never appears in the agent-model picker.
    case proofread = "Proofread"
    /// The memory system's embedding model (ADR-0035) — same reasoning as
    /// `.proofread`: a co-resident, non-generative worker that must never
    /// appear in the agent-model picker.
    case embedding = "Memory"
    /// A speculative-decoding draft model (e.g. DFlash2) — never generative
    /// on its own, downloaded as a dependency of its target, and never in
    /// the agent-model picker.
    case draft = "Draft"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .speechToText: "mic.fill"
        case .textToSpeech: "speaker.wave.3.fill"
        case .agent: "brain"
        case .proofread: "text.badge.checkmark"
        case .embedding: "brain.head.profile"
        case .draft: "bolt.fill"
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
    /// The **Text-Only Override** (CONTEXT.md): the entry withholds the image
    /// input its checkpoint declares, so the app serves it text-only — no
    /// image affordances, the text-class load, no image keying. Today's use is
    /// the two Qwen3.8-27B entries: DFlash2 pairs only with the MLXLLM text
    /// classes, and a vision-mode load would silently never speculate. Lifted
    /// by map #457 once the VLM wrapper hosts the MLXLLM engine (#459).
    var textOnlyOverride: Bool = false

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
    static let defaultSpeechToTextModelID = "whisper-large-v3-turbo"
    static let defaultTextToSpeechModelID = "qwen3-tts-voicedesign"
    static let defaultProofreadModelID = "qwen3.5-0.8b-proofread"
    static let defaultEmbeddingModelID = "qwen3-embedding-0.6b"

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
            id: "whisper-large-v3-turbo-compact",
            displayName: "Whisper Turbo Compact",
            description:
                "Same fast voice-to-text at less than half the size. Best for Macs with limited memory.",
            category: .speechToText,
            source: .huggingFace(
                repo: "argmaxinc/whisperkit-coreml",
                requiredExtension: "mlmodelc",
                // Argmax's compressed turbo (<1% WER delta vs. the f16 default).
                pathPrefix: "openai_whisper-large-v3-v20240930_turbo_632MB"
            ),
            sizeDescription: "~650 MB",
            dependencies: [],
            // Both turbo variants share the large-v3 tokenizer; see the
            // companion-file comment on the default variant above.
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
            id: "qwen3.5-0.8b-proofread",
            displayName: "Dictation Proofreader",
            description:
                "Polishes dictation: punctuation, casing, and misheard words. Runs alongside the agent model.",
            category: .proofread,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.5-0.8B-4bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~0.5 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3-embedding-0.6b",
            displayName: "Memory Embedder",
            description:
                "Turns what you say into vectors so the assistant can find the right memory. Runs alongside the agent model.",
            category: .embedding,
            source: .huggingFace(
                repo: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~335 MB",
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
            id: "qwen3.6-27b-paro",
            displayName: "Qwen3.6-27B PARO",
            description:
                "Highest-quality PARO agent, built on Qwen3.6. Requires a high-memory Mac (48 GB+ recommended).",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.6-27B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~19 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.6-35b-a3b-paro",
            displayName: "Qwen3.6-35B-A3B PARO",
            description:
                "Large sparse-MoE PARO agent (3B active / 35B total), built on Qwen3.6. Requires a high-memory Mac (48 GB+ recommended).",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.6-35B-A3B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~21 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.6-35b-a3b-ud",
            displayName: "Qwen3.6-35B-A3B UD (MLX 4bit)",
            description:
                "Large sparse-MoE agent (3B active / 35B total) from Unsloth. Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5_moe.",
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
            description:
                "Dense 27B agent (uniform 4-bit MLX). Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.6-27B-4bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~16 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.8-27b-paro",
            displayName: "Qwen3.8-27B PARO",
            description:
                "Dense 27B agent built on Qwen3.8 with pairwise-rotation INT4 (near-FP16 accuracy). Served text-only for now so the DFlash2 draft engages; image input returns with #457. Requires a high-memory Mac (48 GB+ recommended); the first load writes a ~19 GB Prepared Checkpoint beside the download.",
            category: .agent,
            source: .huggingFace(
                repo: "z-lab/Qwen3.8-27B-PARO",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~19 GB",
            dependencies: ["qwen3.8-27b-dflash2-draft"],
            textOnlyOverride: true
        ),
        ModelDefinition(
            id: "qwen3.8-27b",
            displayName: "Qwen3.8-27B (MLX 4bit)",
            description:
                "Dense 27B agent from the Qwen3.8 generation (uniform 4-bit MLX). Served text-only: its legacy tensor layout loads as the text class, which is also what the DFlash2 draft pairs with; image input returns with #457. Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Qwen3.8-27B-4bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~16 GB",
            dependencies: ["qwen3.8-27b-dflash2-draft"],
            textOnlyOverride: true
        ),
        ModelDefinition(
            id: "qwen3.8-27b-dflash2-draft",
            displayName: "Qwen3.8-27B DFlash2 Draft",
            description:
                "Block-parallel speculative-decoding draft for Qwen3.8-27B (incoai/Qwen3.8-27B-DFlash2). Downloads alongside either Qwen3.8-27B target and accelerates decode ~2× at greedy and sampling presets alike. Never usable on its own.",
            category: .draft,
            source: .huggingFace(
                repo: "incoai/Qwen3.8-27B-DFlash2",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~3.9 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "qwen3.5-2b",
            displayName: "Qwen3.5-2B (bf16)",
            description:
                "Tiny official Qwen3.5 agent at full bf16 precision. The only small checkpoint that ships the MTP head, so greedy decoding can run speculatively.",
            category: .agent,
            source: .huggingFace(
                repo: "Qwen/Qwen3.5-2B",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~4.6 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "nanbeige4.2-3b-8bit",
            displayName: "Nanbeige4.2-3B (MLX 8-bit)",
            description:
                "Compact agentic model with a Looped Transformer (22 layers run twice, shared weights). Strong tool use and reasoning for its size.",
            category: .agent,
            source: .huggingFace(
                repo: "MercuriusDream/Nanbeige4.2-3B-mlx-8bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~4.4 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "ornith-9b",
            displayName: "Ornith 1.0 9B (MLX 6-bit)",
            description:
                "Compact agentic-coding agent from DeepReinforce (Qwen3.5 dense, vision-capable). Retains the Qwen3.5-VL vision tower; image understanding is inherited from the base and unvalidated by DeepReinforce's text/code post-training.",
            category: .agent,
            source: .huggingFace(
                repo: "mlx-community/Ornith-1.0-9B-6bit",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~8.2 GB",
            dependencies: []
        ),
        ModelDefinition(
            id: "ornith-35b",
            displayName: "Ornith 1.0 35B (MLX 4-bit)",
            description:
                "Large vision-capable MoE agent from DeepReinforce (Qwen3.5-A3B). Requires a high-memory Mac (48 GB+ recommended). Ships as qwen3_5_moe.",
            category: .agent,
            source: .huggingFace(
                repo: "leonsarmiento/Ornith-1.0-35B-4bit-mlx",
                requiredExtension: "safetensors"
            ),
            sizeDescription: "~21 GB",
            dependencies: []
        ),
    ]

    static func withID(_ id: String) -> ModelDefinition? {
        all.first { $0.id == id }
    }

    /// Every known model in a category, downloaded or not — the static facet.
    /// Runtime "which of these are on disk" is the **Model Catalog** join on
    /// `ModelDownloadManager`, not this. See `CONTEXT.md` → Model catalog.
    static func models(in category: ModelCategory) -> [ModelDefinition] {
        all.filter { $0.category == category }
    }

    /// The ids of every known model in a category — lets a caller tell an
    /// *unknown* model id from a known-but-not-downloaded one.
    static func ids(in category: ModelCategory) -> [String] {
        models(in: category).map(\.id)
    }

    static func byCategory() -> [(ModelCategory, [ModelDefinition])] {
        ModelCategory.allCases.compactMap { category in
            let models = all.filter { $0.category == category }
            return models.isEmpty ? nil : (category, models)
        }
    }
}
