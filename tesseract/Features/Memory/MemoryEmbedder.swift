//
//  MemoryEmbedder.swift
//  tesseract
//
//  The memory system's embedding worker (ADR-0035 §5, §8).
//
//  A third co-resident MLX model, following ADR-0034's proofreader precedent
//  exactly: an `actor` so inference runs off the main actor, its own
//  `ModelContainer`, deliberately OUTSIDE the arbiter's `.llm`/`.tts` slots.
//  ADR-0034 established there is no single-container assertion in the stack —
//  a second (or third) co-resident model is architecturally safe.
//
//  Qwen3-Embedding-0.6B-4bit-DWQ: ~335 MB, 1024 dims. Measured on this
//  machine: 0.4 s to load, 334 texts/sec warm.
//
//  Like the proofreader, this NEVER touches the process-global
//  `MLX.Memory.cacheLimit` — that knob belongs to the agent's `LLMActor`.
//
//  Unlike the proofreader, it does not skip-when-busy. Embedding is a tiny
//  forward pass with no decode; it costs milliseconds and contending for it
//  would cost more in complexity than it saves in GPU time.
//

import Foundation
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
// `#huggingFaceTokenizerLoader()` expands to code referencing
// `Tokenizers.AutoTokenizer`, so the module must be in scope at the call site
// — same as `ProofreadModel`.
import Tokenizers

actor MemoryEmbedder {

    /// The embedding *scheme* this code produces — model, preprocessing,
    /// pooling, normalization, all of it. Bump it whenever any of those
    /// change and every stored vector is regenerated on the next prewarm
    /// (`MemoryEngine.reconcileEmbeddingScheme`, #332). Scheme 1 was the
    /// pre-#332 era: min-16 EOS padding, pooled at the batch's last position.
    static let scheme = 2

    private var container: EmbedderModelContainer?
    private var loadedDirectory: URL?
    private var inFlightLoad: Task<Void, Error>?

    var isLoaded: Bool { container != nil }

    /// Single-flight: concurrent callers await the same load.
    func load(from directory: URL) async throws {
        if loadedDirectory == directory, container != nil { return }
        if let inFlightLoad {
            try await inFlightLoad.value
            if loadedDirectory == directory, container != nil { return }
        }
        let load = Task { () throws in
            let loaded = try await EmbedderModelFactory.shared.loadContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
            self.container = loaded
            self.loadedDirectory = directory
        }
        inFlightLoad = load
        defer { if inFlightLoad == load { inFlightLoad = nil } }
        try await load.value
        Log.memory.info("Embedding model loaded from \(directory.lastPathComponent)")
    }

    func unload() {
        container = nil
        loadedDirectory = nil
    }

    /// Embed a batch of **documents** — the stored side of retrieval. Returns
    /// L2-normalized vectors, so cosine similarity is a plain dot product
    /// downstream.
    ///
    /// Returns `[]` when the model isn't loaded — every caller treats an empty
    /// result as "no vector available" and degrades to keyword-only retrieval
    /// rather than failing. Memory must never take the primary flow down.
    /// `applyLayerNorm` is the eval's A/B seam (#332): the HF reference does
    /// not layer-norm after pooling, the shipped configuration does, and the
    /// eval measures the gap rather than anyone asserting it. Product callers
    /// never pass it.
    func embed(_ texts: [String], applyLayerNorm: Bool = true) async -> [[Float]] {
        await encodeAndPool(texts, applyLayerNorm: applyLayerNorm)
    }

    func embed(_ text: String) async -> [Float]? {
        await embed([text]).first
    }

    /// Embed **queries** — the asking side. Qwen3-Embedding is
    /// instruction-aware: queries carry the model card's reference instruction,
    /// documents never do ("No need to add instruction for retrieval
    /// documents"; skipping it on queries costs ~1–5% retrieval quality).
    func embedQueries(_ texts: [String], applyLayerNorm: Bool = true) async -> [[Float]] {
        await encodeAndPool(
            texts.map { Self.queryInstruction + $0 }, applyLayerNorm: applyLayerNorm)
    }

    func embedQuery(_ text: String) async -> [Float]? {
        await embedQueries([text]).first
    }

    /// The reference retrieval instruction from the Qwen3-Embedding model
    /// card, verbatim — including the missing space after `Query:`.
    private static let queryInstruction =
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

    /// The one true embed path (#332).
    ///
    /// Qwen3-Embedding pools the hidden state of each sequence's trailing EOS
    /// — the token its tokenizer's own post-processor appends. Everything here
    /// exists to make sure `.last` pooling actually reads that position:
    ///
    /// - lengths are captured **before** padding, and the pooling mask is
    ///   built from them — never from token values: a `token != pad` mask
    ///   would also erase the genuine trailing EOS, the very position
    ///   last-token pooling exists to read.
    /// - the batch pads to its longest member, no arbitrary floor. Trailing
    ///   pads are invisible to the causal model, so a text's vector cannot
    ///   depend on its batch-mates.
    ///
    /// Getting this wrong is not a quality nuance: the first version pooled a
    /// pad position for every text under 16 tokens, and every short memory in
    /// the owner's store collapsed onto one content-free direction (pairwise
    /// cosine 0.96 between "I love cats." and "He is from Europe.").
    private func encodeAndPool(_ texts: [String], applyLayerNorm: Bool = true) async -> [[Float]] {
        guard let container, !texts.isEmpty else { return [] }
        return await container.perform {
            (model: EmbeddingModel, tokenizer: MLXLMCommon.Tokenizer, pooling: Pooling) -> [[Float]]
            in
            let eos = tokenizer.eosTokenId ?? 151_643
            let inputs = texts.map { text in
                var tokens = tokenizer.encode(text: Self.truncate(text), addSpecialTokens: true)
                // The tokenizer's post-processor already appends EOS; the
                // guard keeps this path correct under one that doesn't.
                if tokens.last != eos { tokens.append(eos) }
                return tokens
            }
            let maxLength = inputs.map(\.count).max() ?? 1
            let padded = stacked(
                inputs.map { MLXArray($0 + Array(repeating: eos, count: maxLength - $0.count)) })
            let lengths = MLXArray(inputs.map { Int32($0.count) })
            let positions = MLXArray((0..<maxLength).map { Int32($0) })
            let mask =
                positions.expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                mask: mask,
                normalize: true, applyLayerNorm: applyLayerNorm
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }
    }

    /// The encoder has a context limit and a memory record is short by
    /// construction; a runaway episode (a pasted logfile) must not blow the
    /// batch's padding out to its length.
    private static let characterBudget = 2_000

    private static func truncate(_ text: String) -> String {
        guard text.count > characterBudget else { return text }
        return String(text.prefix(characterBudget))
    }
}
