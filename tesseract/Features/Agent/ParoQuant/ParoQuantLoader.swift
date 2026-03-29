import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXNN
import Tokenizers

// MARK: - Detection

/// Returns `true` if the model directory contains a ParoQuant checkpoint.
nonisolated func isParoQuantModel(directory: URL) -> Bool {
    guard let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json")) else {
        return false
    }
    return isSupportedParoQuantModel(directory: directory, configData: configData)
}

// MARK: - Config

nonisolated struct ParoQuantConfig: Sendable {
    let bits: Int
    let groupSize: Int
    let krot: Int
}

/// Reads ParoQuant quantization config from config.json data.
nonisolated private func readParoQuantConfig(_ configData: Data) -> ParoQuantConfig? {
    guard let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
          let qc = json["quantization_config"] as? [String: Any]
    else { return nil }

    let bits = qc["bits"] as? Int ?? 4
    let groupSize = qc["group_size"] as? Int ?? 128
    let krot = qc["krot"] as? Int ?? 8
    return ParoQuantConfig(bits: bits, groupSize: groupSize, krot: krot)
}

/// Detects quant_method from config.json. Returns nil if not ParoQuant.
nonisolated private func detectQuantMethod(directory: URL) -> String? {
    let configURL = directory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let qc = json["quantization_config"] as? [String: Any],
          let method = qc["quant_method"] as? String
    else { return nil }
    return method
}

/// The custom loader is meant for z-lab Qwen3.5 PARO models (4B and 9B) with VLM wrappers.
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
    ]
    if supportedDirectoryNames.contains(directory.lastPathComponent) {
        return true
    }

    let architectures = json["architectures"] as? [String] ?? []
    return architectures.contains("Qwen3_5ForConditionalGeneration")
}

// MARK: - AutoAWQ Conversion

nonisolated private enum AWQ {
    static let bits = 4
    static let packFactor = 32 / bits  // 8 values per uint32
    static let mask: Int32 = (1 << bits) - 1
    static let shifts: [Int32] = (0..<8).map { Int32($0 * bits) }
    /// Inverse of AutoAWQ reorder [0,2,4,6,1,3,5,7] → [0,4,1,5,2,6,3,7]
    static let inverseReorder = [0, 4, 1, 5, 2, 6, 3, 7]

    // Pre-computed MLXArrays (created once, reused across calls)
    nonisolated(unsafe) static let shiftsArray = MLXArray(shifts.map { Int64($0) }).reshaped(1, 1, 8)
    nonisolated(unsafe) static let reorderIndices = MLXArray(inverseReorder.map { Int32($0) })
}

/// Unpack AutoAWQ int32 → raw uint8 values, undoing the [0,2,4,6,1,3,5,7] reorder.
///
/// Input: `[rows, cols]` of int32 (each packing 8 × 4-bit values)
/// Output: `[rows, cols * 8]` of uint8 (raw 4-bit values)
nonisolated private func unpackAndReorder(_ packed: MLXArray) -> MLXArray {
    let rows = packed.dim(0)
    let cols = packed.dim(1)

    let expanded = packed.asType(.int64).expandedDimensions(axis: 2)
    let raw = ((expanded >> AWQ.shiftsArray) & Int64(AWQ.mask)).asType(.uint8)
    let reordered = raw.take(AWQ.reorderIndices, axis: 2)

    return reordered.reshaped(rows, cols * 8)
}

/// Pack raw uint8 values into uint32 (MLX sequential layout).
///
/// Input: `[rows, cols]` of uint8 where cols is divisible by 8
/// Output: `[rows, cols / 8]` of uint32
nonisolated private func packMLX(_ w: MLXArray) -> MLXArray {
    let rows = w.dim(0)
    let reshaped = w.reshaped(rows, -1, AWQ.packFactor)  // [rows, cols/8, 8]

    var packed = reshaped[0..., 0..., 0].asType(.uint32)
    for i in 1..<AWQ.packFactor {
        packed = packed | (reshaped[0..., 0..., i].asType(.uint32) << UInt32(i * AWQ.bits))
    }
    return packed
}

/// Convert AutoAWQ checkpoint weights to MLX quantized format in-place.
///
/// For each layer that has both `.qweight` and `.theta`:
/// - `.qweight` → `.weight` (unpack, undo reorder, transpose, repack as MLX uint32)
/// - `.qzeros` + `.scales` → `.biases` (compute `-scales * zeros`, transpose)
/// - `.scales` → `.scales` (transpose)
/// - `.theta`, `.pairs`, `.channel_scales`, `.bias` → pass through
nonisolated private func convertAutoAWQ(
    _ weights: inout [String: MLXArray], groupSize: Int
) {
    // Find prefixes that have both .qweight and .theta
    let prefixes = Set(
        weights.keys
            .filter { $0.hasSuffix(".qweight") }
            .compactMap { key -> String? in
                let pfx = String(key.dropLast("qweight".count))
                return weights["\(pfx)theta"] != nil ? pfx : nil
            }
    )

    guard !prefixes.isEmpty else { return }

    // Pass 1: compute biases from qzeros + scales BEFORE scales are transposed.
    // Dict iteration order is non-deterministic, so qzeros must be processed
    // while scales are still in their original [groups, outDims] layout.
    for pfx in prefixes {
        guard let qzeros = weights.removeValue(forKey: "\(pfx)qzeros") else { continue }
        let zeros = unpackAndReorder(qzeros).asType(.float32)
        let scales = weights["\(pfx)scales"]!.asType(.float32)
        weights["\(pfx)biases"] = (-scales * zeros).transposed().asType(.float16)
    }

    // Pass 2: convert remaining keys (qweight, scales, channel_scales)
    let keysToConvert = weights.keys.filter { key in
        prefixes.contains(where: { key.hasPrefix($0) })
    }

    for key in keysToConvert {
        guard let pfx = prefixes.first(where: { key.hasPrefix($0) }) else { continue }
        let suffix = String(key.dropFirst(pfx.count))

        switch suffix {
        case "qweight":
            let val = weights.removeValue(forKey: key)!
            weights["\(pfx)weight"] = packMLX(unpackAndReorder(val).transposed())

        case "scales":
            weights[key] = weights[key]!.transposed()

        case "channel_scales":
            if let val = weights[key], val.ndim == 1 {
                weights[key] = val.reshaped(1, -1)
            }

        default:
            break  // theta, pairs, bias — keep as-is
        }
    }
}

// MARK: - Layer Patching

nonisolated private func requireTensor(
    _ key: String, weights: [String: MLXArray]
) throws -> MLXArray {
    guard let tensor = weights[key] else {
        throw ParoQuantError.missingTensor(key)
    }
    return tensor
}

nonisolated private func verifyTensorShape(
    _ tensor: MLXArray, key: String, expected: [Int]
) throws {
    guard tensor.shape == expected else {
        throw ParoQuantError.invalidTensorShape(
            key: key,
            expected: expected,
            actual: tensor.shape
        )
    }
}

nonisolated private func rotationLeafModules(model: Module) -> [String: Module] {
    Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
}

nonisolated private func rotationModuleSpec(
    prefix: String,
    leafModules: [String: Module],
    weights: [String: MLXArray],
    bits: Int,
    groupSize: Int
) throws -> (inputDims: Int, outputDims: Int, hasBias: Bool, krot: Int) {
    guard let original = leafModules[prefix] else {
        throw ParoQuantError.rotationLayerNotFound(prefix)
    }
    guard let linear = original as? Linear else {
        throw ParoQuantError.rotationLayerTypeMismatch(
            path: prefix,
            actualType: String(describing: type(of: original))
        )
    }

    let outputDims = linear.shape.0
    let inputDims = linear.shape.1
    let groups = inputDims / groupSize
    let packedInputDims = inputDims * bits / 32
    let expectsBias = linear.bias != nil

    let theta = try requireTensor("\(prefix).theta", weights: weights)
    let pairs = try requireTensor("\(prefix).pairs", weights: weights)
    let channelScales = try requireTensor("\(prefix).channel_scales", weights: weights)
    let weight = try requireTensor("\(prefix).weight", weights: weights)
    let scales = try requireTensor("\(prefix).scales", weights: weights)
    let biases = try requireTensor("\(prefix).biases", weights: weights)

    let krot = theta.dim(0)
    try verifyTensorShape(theta, key: "\(prefix).theta", expected: [krot, inputDims / 2])
    try verifyTensorShape(pairs, key: "\(prefix).pairs", expected: [krot, inputDims])
    try verifyTensorShape(channelScales, key: "\(prefix).channel_scales", expected: [1, inputDims])
    try verifyTensorShape(weight, key: "\(prefix).weight", expected: [outputDims, packedInputDims])
    try verifyTensorShape(scales, key: "\(prefix).scales", expected: [outputDims, groups])
    try verifyTensorShape(biases, key: "\(prefix).biases", expected: [outputDims, groups])

    if expectsBias {
        _ = try requireTensor("\(prefix).bias", weights: weights)
    }

    return (inputDims, outputDims, expectsBias, krot)
}

/// Replace Linear layers with RotateQuantizedLinear where rotation parameters exist.
nonisolated private func patchRotationLayers(
    model: Module, weights: [String: MLXArray],
    bits: Int, groupSize: Int
) throws {
    // Find all layer prefixes that have .theta weights
    let prefixes = weights.keys
        .filter { $0.hasSuffix(".theta") }
        .map { String($0.dropLast(".theta".count)) }
        .sorted()

    guard !prefixes.isEmpty else { return }

    let leafModules = rotationLeafModules(model: model)
    var updates = [(String, Module)]()

    for prefix in prefixes {
        let spec = try rotationModuleSpec(
            prefix: prefix,
            leafModules: leafModules,
            weights: weights,
            bits: bits,
            groupSize: groupSize
        )

        let replacement = RotateQuantizedLinear(
            inputDims: spec.inputDims,
            outputDims: spec.outputDims,
            hasBias: spec.hasBias,
            groupSize: groupSize,
            bits: bits,
            krot: spec.krot
        )

        updates.append((prefix, replacement))
    }

    if !updates.isEmpty {
        try model.update(modules: ModuleChildren.unflattened(updates), verify: [.noUnusedKeys])

        let patchedLeaves = rotationLeafModules(model: model)
        for (path, _) in updates {
            guard patchedLeaves[path] is RotateQuantizedLinear else {
                throw ParoQuantError.rotationLayerPatchFailed(path)
            }
        }
    }
}

/// Predicate for the native MLX quantization pass.
///
/// ParoQuant checkpoints store rotation-aware INT4 weights for transformer projections,
/// but the tied IO embedding path still needs standard MLX quantization.
nonisolated private func isParoQuantIOLayer(path: String, module: Module) -> Bool {
    guard module is Quantizable else { return false }
    return path.hasSuffix("embed_tokens") || path.hasSuffix("lm_head")
}

/// Layers already represented in MLX quantized checkpoint form.
///
/// These need their module types swapped before `update(parameters:)` applies
/// quantized `weight` / `scales` / `biases` tensors.
nonisolated private func isCheckpointQuantizedLayer(
    path: String, weights: [String: MLXArray]
) -> Bool {
    weights["\(path).scales"] != nil && weights["\(path).theta"] == nil
}

// MARK: - UserInputProcessor

/// Local UserInputProcessor for ParoQuant models.
/// Mirrors the private `LLMUserInputProcessor` from MLXLLM.
nonisolated private struct ParoQuantInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch TokenizerError.missingChatTemplate {
            Log.agent.warning(
                "Tokenizer is missing a chat template for the ParoQuant model; falling back to plain text prompt formatting"
            )
            let prompt = messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

// MARK: - Load Entry Point


/// Load a ParoQuant model, returning a ModelContainer.
///
/// Delegates to MLXLMCommon's `loadParoQuantModel` which handles:
/// - AutoAWQ weight conversion
/// - Rotation layer patching
/// - Pre-rotation of weights (bakes Givens rotation into quantized weights for +23% throughput)
/// - Caching pre-rotated weights to disk for fast subsequent loads
nonisolated func loadParoQuantModel(
    from directory: URL, toolCallFormat: ToolCallFormat?
) async throws -> ModelContainer {
    // Use VLM type registry so Qwen3.5 loads with its vision tower.
    let container = try await MLXLMCommon.loadParoQuantModel(
        from: directory,
        typeRegistry: VLMModelFactory.shared.typeRegistry,
        toolCallFormat: toolCallFormat
    )

    // Create VLM processor for vision support; falls back to text-only if unavailable.
    let vlmProcessor = await loadVLMProcessor(from: directory, container: container)

    await container.update { context in
        if let vlmProcessor {
            context.processor = vlmProcessor
        } else {
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
              BaseProcessorConfiguration.self, from: configData) else {
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

// MARK: - Errors

nonisolated enum ParoQuantError: LocalizedError {
    case missingConfig
    case unsupportedModel
    case missingTensor(String)
    case invalidTensorShape(key: String, expected: [Int], actual: [Int])
    case rotationLayerNotFound(String)
    case rotationLayerTypeMismatch(path: String, actualType: String)
    case rotationLayerPatchFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingConfig:
            return "Missing quantization_config in config.json for ParoQuant model"
        case .unsupportedModel:
            return "The custom ParoQuant loader only supports z-lab Qwen3.5 PARO models (4B and 9B)"
        case .missingTensor(let key):
            return "Missing required ParoQuant tensor: \(key)"
        case .invalidTensorShape(let key, let expected, let actual):
            return "Invalid ParoQuant tensor shape for \(key): expected \(expected), got \(actual)"
        case .rotationLayerNotFound(let path):
            return "Unable to find ParoQuant rotation layer in model: \(path)"
        case .rotationLayerTypeMismatch(let path, let actualType):
            return "ParoQuant rotation layer \(path) is not a Linear-compatible module: \(actualType)"
        case .rotationLayerPatchFailed(let path):
            return "Failed to replace ParoQuant layer with RotateQuantizedLinear: \(path)"
        }
    }
}
