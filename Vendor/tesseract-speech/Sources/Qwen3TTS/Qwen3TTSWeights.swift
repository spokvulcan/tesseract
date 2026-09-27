import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

/// Loading the talker (with its code predictor) from a checkpoint.
///
/// Three things differ from reading the tensors straight in:
/// - the text embedding table stays on disk (`Qwen3TTSTextEmbedding`);
/// - q/k/v and gate/up are stacked into single projections, when their
///   quantization matches;
/// - the weights are evaluated a layer at a time, so loading never holds the
///   file's arrays and the stacked copies of the whole model at once.
enum Qwen3TTSWeights {

    static func loadTalker(
        config: Qwen3TTSModelConfig, directory: URL
    ) throws -> (talker: Qwen3TTSTalker, textEmbedding: Qwen3TTSTextEmbedding) {
        let talkerConfig = config.talkerConfig ?? .defaults
        var weights: [String: MLXArray] = [:]
        for file in try safetensorsFiles(in: directory) {
            // Lazy: nothing is read until evaluated.
            for (key, value) in try MLX.loadArrays(url: file) where key.hasPrefix("talker.") {
                weights[String(key.dropFirst("talker.".count))] = value
            }
        }

        let quantization = QuantizationLookup(config: config)
        let textEmbedding = try takeTextEmbedding(
            from: &weights, config: talkerConfig, quantization: quantization,
            directory: directory)

        var fusion = Qwen3TTSFusion()
        var stackedQuantization: [String: (Int, Int, QuantizationMode)] = [:]
        fusion.qkv = stack(
            ["q_proj", "k_proj", "v_proj"], into: "qkv_proj", under: "self_attn", in: &weights,
            quantization, &stackedQuantization)
        fusion.gateUp = stack(
            ["gate_proj", "up_proj"], into: "gate_up_proj", under: "mlp", in: &weights,
            quantization, &stackedQuantization)

        let talker = Qwen3TTSTalker(config: talkerConfig, fusion: fusion)
        if quantization.isQuantized {
            quantize(model: talker) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                return stackedQuantization[path] ?? quantization.parameters(for: path)
            }
        }
        try talker.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        weights.removeAll()

        // A layer at a time: each stacked projection is built from the
        // file's arrays, which are released before the next layer loads.
        for layer in talker.model.layers { eval(layer.parameters()) }
        for layer in talker.codePredictor.model.layers { eval(layer.parameters()) }
        eval(talker.parameters())
        Memory.clearCache()
        return (talker, textEmbedding)
    }

    /// `directory`'s safetensors files, by name.
    static func safetensorsFiles(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    // MARK: - Text embedding

    private static func takeTextEmbedding(
        from weights: inout [String: MLXArray], config: Qwen3TTSTalkerConfig,
        quantization: QuantizationLookup, directory: URL
    ) throws -> Qwen3TTSTextEmbedding {
        let path = "model.text_embedding"
        guard let weight = weights.removeValue(forKey: "\(path).weight") else {
            throw AudioGenerationError.modelNotInitialized(
                "The checkpoint has no talker text embedding (talker.\(path).weight).")
        }
        let scales = weights.removeValue(forKey: "\(path).scales")
        let biases = weights.removeValue(forKey: "\(path).biases")
        let count = config.textVocabSize
        let dimensions = config.textHiddenSize
        if scales == nil,
            let table = Qwen3TTSTextEmbedding(key: "talker.\(path).weight", directory: directory),
            table.count == count, table.dimensions == dimensions
        {
            return table
        }
        // Quantized (or not readable in place): keep it in memory.
        let embedding: Embedding
        if let scales, let (groupSize, bits, mode) = quantization.parameters(for: path) {
            embedding = QuantizedEmbedding(
                embeddingCount: count, dimensions: dimensions, groupSize: groupSize, bits: bits,
                mode: mode)
            var parameters: [String: MLXArray] = ["weight": weight, "scales": scales]
            parameters["biases"] = biases
            try embedding.update(
                parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        } else {
            embedding = Embedding(weight: weight)
        }
        eval(embedding.parameters())
        return Qwen3TTSTextEmbedding(embedding: embedding, count: count, dimensions: dimensions)
    }

    // MARK: - Stacking projections

    /// The layers that hold `parts` under `module` (`...self_attn`, `...mlp`).
    private static func prefixes(of part: String, under module: String, in weights: [String: MLXArray])
        -> [String]
    {
        let suffix = ".\(module).\(part).weight"
        return weights.keys.filter { $0.hasSuffix(suffix) }
            .map { String($0.dropLast(suffix.count - module.count - 1)) }
            .sorted()
    }

    /// Whether every layer's `parts` can be stacked: all present, and the
    /// same storage (dtype, packed width, quantization) in each part.
    private static func canStack(
        _ parts: [String], under module: String, in weights: [String: MLXArray],
        _ quantization: QuantizationLookup
    ) -> Bool {
        let layers = prefixes(of: parts[0], under: module, in: weights)
        guard !layers.isEmpty else { return false }
        for layer in layers {
            let first = "\(layer).\(parts[0])"
            for part in parts {
                let p = "\(layer).\(part)"
                guard let w = weights["\(p).weight"], let w0 = weights["\(first).weight"],
                    w.dtype == w0.dtype, w.dim(1) == w0.dim(1),
                    (weights["\(p).scales"] == nil) == (weights["\(first).scales"] == nil),
                    (weights["\(p).biases"] == nil) == (weights["\(first).biases"] == nil),
                    (weights["\(p).bias"] == nil) == (weights["\(first).bias"] == nil),
                    sameQuantization(
                        quantization.parameters(for: p), quantization.parameters(for: first))
                else { return false }
            }
        }
        return true
    }

    private static func sameQuantization(
        _ a: (Int, Int, QuantizationMode)?, _ b: (Int, Int, QuantizationMode)?
    ) -> Bool {
        a?.0 == b?.0 && a?.1 == b?.1 && a?.2 == b?.2
    }

    /// Replaces each layer's `parts` with one projection whose output rows
    /// are theirs, in order, when they can be stacked (`canStack`). Returns
    /// whether they were.
    private static func stack(
        _ parts: [String], into name: String, under module: String,
        in weights: inout [String: MLXArray], _ quantization: QuantizationLookup,
        _ stackedQuantization: inout [String: (Int, Int, QuantizationMode)]
    ) -> Bool {
        guard canStack(parts, under: module, in: weights, quantization) else { return false }
        for layer in prefixes(of: parts[0], under: module, in: weights) {
            for tensor in ["weight", "scales", "biases", "bias"] {
                let keys = parts.map { "\(layer).\($0).\(tensor)" }
                guard keys.allSatisfy({ weights[$0] != nil }) else { continue }
                weights["\(layer).\(name).\(tensor)"] = concatenated(
                    keys.map { weights.removeValue(forKey: $0)! }, axis: 0)
            }
            stackedQuantization["\(layer).\(name)"] = quantization.parameters(
                for: "\(layer).\(parts[0])")
        }
        return true
    }

    // MARK: - Quantization

    /// The checkpoint's quantization for a module path under `talker.`.
    struct QuantizationLookup {
        let global: BaseConfiguration.Quantization?
        let perLayer: BaseConfiguration.PerLayerQuantization?

        init(config: Qwen3TTSModelConfig) {
            global = config.quantization
            perLayer = config.perLayerQuantization
        }

        var isQuantized: Bool { global != nil || perLayer != nil }

        func parameters(for path: String) -> (Int, Int, QuantizationMode)? {
            if let perLayer {
                // Per-layer configs name paths from the checkpoint root.
                return (perLayer.quantization(layer: "talker.\(path)")
                    ?? perLayer.quantization(layer: path))?.asTuple
            }
            return global?.asTuple
        }
    }

    // MARK: - tokenizer.json

    /// Qwen3-TTS repos ship a slow tokenizer (vocab.json + merges.txt), and
    /// swift-transformers reads only the fast format: this writes
    /// tokenizer.json from them, once, next to them.
    static func generateTokenizerJSONIfMissing(in directory: URL) {
        let fm = FileManager.default
        let output = directory.appendingPathComponent("tokenizer.json")
        guard !fm.fileExists(atPath: output.path) else { return }
        let vocab = directory.appendingPathComponent("vocab.json")
        let merges = directory.appendingPathComponent("merges.txt")
        guard fm.fileExists(atPath: vocab.path), fm.fileExists(atPath: merges.path) else { return }
        try? writeTokenizerJSON(
            vocab: vocab, merges: merges,
            tokenizerConfig: directory.appendingPathComponent("tokenizer_config.json"),
            to: output)
    }

    private static func writeTokenizerJSON(
        vocab vocabURL: URL, merges mergesURL: URL, tokenizerConfig: URL, to output: URL
    ) throws {
        let vocab =
            try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as? [String: Int]
            ?? [:]
        // Skip the "#version: ..." header.
        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
            .components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        var addedTokens: [[String: Any]] = []
        if let data = try? Data(contentsOf: tokenizerConfig),
            let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let decoder = config["added_tokens_decoder"] as? [String: [String: Any]]
        {
            for (id, info) in decoder {
                guard let id = Int(id), let content = info["content"] as? String else { continue }
                addedTokens.append([
                    "id": id,
                    "content": content,
                    "single_word": info["single_word"] as? Bool ?? false,
                    "lstrip": info["lstrip"] as? Bool ?? false,
                    "rstrip": info["rstrip"] as? Bool ?? false,
                    "normalized": info["normalized"] as? Bool ?? false,
                    "special": info["special"] as? Bool ?? true,
                ])
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        // Qwen2's byte-level BPE with the GPT-2-style pre-tokenizer regex.
        let tokenizer: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": [
                            "Regex":
                                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        ],
                        "behavior": "Isolated",
                        "invert": false,
                    ] as [String: Any],
                    [
                        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                        "use_regex": false,
                    ] as [String: Any],
                ] as [[String: Any]],
            ] as [String: Any],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                "use_regex": true,
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            ] as [String: Any],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys])
            .write(to: output)
    }
}
