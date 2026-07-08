import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

// App-side ParoQuant surface: checkpoint detection, container loading, and
// input-processor wiring. The quantization machinery itself — AutoAWQ weight
// conversion, rotation-layer patching, `RotateQuantizedLinear` — lives solely
// in the vendored package (`Vendor/mlx-swift-lm`, `MLXLMCommon/ParoQuant/`),
// reached through `MLXLMCommon.loadParoQuantModel`. Modify it there.

// MARK: - Detection

/// Returns `true` if the model directory contains a ParoQuant checkpoint.
nonisolated func isParoQuantModel(directory: URL) -> Bool {
    guard let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    else {
        return false
    }
    return isSupportedParoQuantModel(directory: directory, configData: configData)
}

/// The custom loader is meant for z-lab Qwen3.5/3.6 PARO models (4B, 9B, and 27B) with VLM wrappers.
nonisolated private func isSupportedParoQuantModel(directory: URL, configData: Data) -> Bool {
    guard let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
        let qc = json["quantization_config"] as? [String: Any],
        let method = qc["quant_method"] as? String,
        method == "paroquant"
    else { return false }

    let supportedDirectoryNames: Set<String> = [
        "z-lab_Qwen3.5-4B-PARO",
        "Qwen3.5-4B-PARO",
        "z-lab_Qwen3.5-9B-PARO",
        "Qwen3.5-9B-PARO",
        "z-lab_Qwen3.6-27B-PARO",
        "Qwen3.6-27B-PARO",
    ]
    if supportedDirectoryNames.contains(directory.lastPathComponent) {
        return true
    }

    let architectures = json["architectures"] as? [String] ?? []
    return architectures.contains("Qwen3_5ForConditionalGeneration")
}

// MARK: - UserInputProcessor

/// Local UserInputProcessor for ParoQuant models.
/// Mirrors the private `LLMUserInputProcessor` from MLXLLM.
nonisolated private struct ParoQuantInputProcessor: UserInputProcessor {
    let tokenizer: MLXLMCommon.Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            Log.agent.warning(
                "Tokenizer is missing a chat template for the ParoQuant model; falling back to plain text prompt formatting"
            )
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

// MARK: - Load Entry Points

/// Install a text-only `ParoQuantInputProcessor` on the container, using the
/// model's own `messageGenerator` when available. Used by the LLM loader and
/// as the VLM loader's fallback when `preprocessor_config.json` is absent.
nonisolated private func installTextOnlyProcessor(on container: ModelContainer) async {
    await container.update { context in
        let messageGenerator: MessageGenerator =
            if let llmModel = context.model as? LLMModel {
                llmModel.messageGenerator(tokenizer: context.tokenizer)
            } else {
                DefaultMessageGenerator()
            }
        context.processor = ParoQuantInputProcessor(
            tokenizer: context.tokenizer, configuration: context.configuration,
            messageGenerator: messageGenerator
        )
    }
}

/// Load a text-only ParoQuant container via the LLM type registry.
///
/// Qwen3.5 resolves to `MLXLLM.Qwen35Model`, which inherits `LLMModel.prepare`
/// with chunked prefill (~1300 tok/s on Qwen3.5-4B PARO). Uses
/// `ParoQuantInputProcessor` (text-only) — this container should never receive
/// an image; the caller is responsible for routing image-bearing turns to the
/// VLM container loaded by `loadParoQuantVLMContainer`.
///
/// Delegates to MLXLMCommon's `loadParoQuantModel` which handles AutoAWQ weight
/// conversion, rotation layer patching, and pre-rotation caching.
///
/// `loadParoQuantModel` is generic over a concrete `LanguageModel` type, so the
/// shared factory registries (`ModelTypeRegistry<LanguageModel>`, existential
/// element) don't satisfy it. ParoQuant supports exactly one architecture
/// (Qwen3.5), so a single-creator registry mirroring the factory entry is
/// passed instead.
nonisolated func loadParoQuantLLMContainer(
    from directory: URL, toolCallFormat: ToolCallFormat?
) async throws -> ModelContainer {
    let typeRegistry = ModelTypeRegistry<MLXLLM.Qwen35Model>(creators: [
        "qwen3_5": { data in
            MLXLLM.Qwen35Model(
                try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
    ])
    let container = try await MLXLMCommon.loadParoQuantModel(
        from: directory,
        typeRegistry: typeRegistry,
        tokenizerLoader: #huggingFaceTokenizerLoader(),
        toolCallFormat: toolCallFormat
    )
    await installTextOnlyProcessor(on: container)
    return container
}

/// Load a vision-capable ParoQuant container via the VLM type registry.
///
/// Qwen3.5/3.6 resolves to `MLXVLM.Qwen35`, whose `prepare` handles image-text
/// merging. Text prefill measures on par with the text-only container — the
/// retired "~390 tok/s un-chunked on long text prompts" claim was disproven by
/// measurement (ADR-0013); the only standing cost is the resident vision tower.
/// Loaded eagerly for vision-capable models (ADR-0013); text-only models and the
/// vision-opted-out path use `loadParoQuantLLMContainer`.
nonisolated func loadParoQuantVLMContainer(
    from directory: URL, toolCallFormat: ToolCallFormat?
) async throws -> ModelContainer {
    let typeRegistry = ModelTypeRegistry<MLXVLM.Qwen35>(creators: [
        "qwen3_5": { data in
            MLXVLM.Qwen35(
                try JSONDecoder.json5().decode(MLXVLM.Qwen35Configuration.self, from: data))
        }
    ])
    let container = try await MLXLMCommon.loadParoQuantModel(
        from: directory,
        typeRegistry: typeRegistry,
        tokenizerLoader: #huggingFaceTokenizerLoader(),
        toolCallFormat: toolCallFormat
    )

    if let vlmProcessor = await loadVLMProcessor(from: directory, container: container) {
        await container.update { context in
            context.processor = vlmProcessor
        }
    } else {
        await installTextOnlyProcessor(on: container)
    }

    return container
}

/// Attempts to create a VLM processor from the model directory's `preprocessor_config.json`.
/// Returns `nil` if the config is absent (expected for text-only models).
private func loadVLMProcessor(
    from directory: URL, container: ModelContainer
) async -> (any UserInputProcessor)? {
    let configURL = directory.appendingPathComponent("preprocessor_config.json")
    guard let configData = try? Data(contentsOf: configURL),
        let baseConfig = try? JSONDecoder().decode(
            BaseProcessorConfiguration.self, from: configData)
    else {
        return nil
    }
    do {
        let tokenizer = await container.perform { $0.tokenizer }
        return try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: configData,
            processorType: baseConfig.processorClass,
            tokenizer: tokenizer)
    } catch {
        Log.agent.warning("VLM processor creation failed, using text-only fallback: \(error)")
        return nil
    }
}
