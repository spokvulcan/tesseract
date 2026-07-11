//
//  ProofreadModel.swift
//  tesseract
//
//  The MLX adapter of the **Proofread Pass** (map #283, ADR-0034): a second,
//  small co-resident causal LM (Qwen3.5-0.8B, 4-bit) that polishes dictation
//  transcriptions. Deliberately outside the agent's LLMActor/prefix-cache
//  machinery — the model is tiny, so a single persistent KV cache trimmed to
//  the common prompt prefix (the fixed system prompt) is all the caching it
//  needs to make idle-GPU passes feel instant.
//
//  An actor, so inference runs off the main actor; `Sendable` is free. The
//  process-global MLX `Memory.cacheLimit` is intentionally NOT touched here:
//  the agent's LLMActor owns that knob, and a 0.8B pass rides whatever limit
//  is in force (ADR-0034).
//

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import os

actor ProofreadModel {

    /// The persistent KV cache and the token ids it currently holds. Each
    /// call trims the cache back to its common prefix with the new prompt —
    /// across calls that prefix is the rendered system prompt, so only the
    /// short user turn ever prefills.
    ///
    /// `@unchecked Sendable` box because `KVCache` is not `Sendable` and
    /// `ModelContainer.perform` only returns `Sendable` results: the box is
    /// touched *exclusively inside `perform` closures*, which the container
    /// serializes, so access is single-threaded by construction.
    private final class CacheBox: @unchecked Sendable {
        var cache: [KVCache] = []
        var tokens: [Int] = []
    }

    private var container: ModelContainer?
    private var loadedDirectory: URL?
    private var inFlightLoad: Task<Void, Error>?
    private var cacheBox = CacheBox()

    var isLoaded: Bool { container != nil }

    /// Loads (or reuses) the model from a local directory. Single-flight:
    /// concurrent callers await the same load.
    func load(from directory: URL) async throws {
        if loadedDirectory == directory, container != nil { return }
        if let inFlightLoad {
            try await inFlightLoad.value
            if loadedDirectory == directory, container != nil { return }
        }
        let load = Task { () throws -> Void in
            // Non-PARO Qwen3.5 checkpoints ship as VLM bundles; force the
            // text-only LLM factory, same as LLMActor's non-vision path.
            let loaded = try await LLMModelFactory.shared.loadContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
            self.container = loaded
            self.loadedDirectory = directory
            self.cacheBox = CacheBox()
        }
        inFlightLoad = load
        defer { if inFlightLoad == load { inFlightLoad = nil } }
        try await load.value
        Log.transcription.info("Proofread model loaded from \(directory.lastPathComponent)")
    }

    func unload() {
        container = nil
        loadedDirectory = nil
        cacheBox = CacheBox()
        Log.transcription.info("Proofread model unloaded")
    }

    /// One proofread completion: renders `[system, user]` through the chat
    /// template (thinking disabled), prefills only the tokens past the cached
    /// common prefix, and decodes until EOS or the output budget. Returns the
    /// raw model reply; parsing into a verdict is `ProofreadReply`'s job.
    func run(system: String, text: String) async throws -> String {
        guard let container else { throw ProofreadModelError.notLoaded }
        let box = cacheBox

        let reply: String = try await container.perform { context in
            let messages: [[String: any Sendable]] = [
                ["role": "system", "content": system],
                ["role": "user", "content": text],
            ]
            // Qwen3.5's template honors `enable_thinking`; a template
            // that ignores it just renders normally, and the reply-side
            // think-strip below covers that case.
            let promptTokens = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: nil,
                additionalContext: ["enable_thinking": false])

            // Reuse the cached prefix: trim the persistent cache back to
            // the longest common prefix with this prompt (across calls that
            // is the rendered system prompt), then prefill only the rest.
            var cache = box.cache
            var commonCount = 0
            if !cache.isEmpty, cache.allSatisfy(\.isTrimmable) {
                while commonCount < min(box.tokens.count, promptTokens.count),
                    box.tokens[commonCount] == promptTokens[commonCount]
                {
                    commonCount += 1
                }
                // Never resume at the very end of the prompt: the last
                // token must go through the iterator to produce logits.
                commonCount = min(commonCount, promptTokens.count - 1)
                let excess = box.tokens.count - commonCount
                if excess > 0 {
                    for entry in cache where entry.trim(excess) != excess {
                        // A cache that refuses the trim is rebuilt.
                        commonCount = 0
                    }
                    if commonCount == 0 { cache = [] }
                }
            } else {
                cache = []
            }

            let parameters = GenerateParameters(
                maxTokens: promptTokens.count * 2 + 64,
                temperature: 0.0
            )
            if cache.isEmpty {
                cache = context.model.newCache(parameters: parameters)
                commonCount = 0
            }

            let suffix = Array(promptTokens[commonCount...])
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(suffix)),
                model: context.model,
                cache: cache,
                parameters: parameters
            )

            var stopIDs = context.configuration.eosTokenIds
            if let eos = context.tokenizer.eosTokenId { stopIDs.insert(eos) }

            var generated: [Int] = []
            while let token = iterator.next() {
                if token == context.tokenizer.unknownTokenId || stopIDs.contains(token) {
                    break
                }
                generated.append(token)
                if generated.count % 16 == 0, Task.isCancelled { break }
            }

            // The cache now holds prompt + generated tokens; remember them so
            // the next call can trim back to the shared system-prompt prefix.
            box.cache = cache
            box.tokens = promptTokens + generated

            return context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
        }

        try Task.checkCancellation()
        return Self.strippingThinkBlock(reply)
    }

    /// Defense in depth for templates that ignore `enable_thinking`: drop a
    /// leading `<think>…</think>` block from the reply.
    static func strippingThinkBlock(_ reply: String) -> String {
        let trimmed = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<think>") else { return trimmed }
        guard let closeRange = trimmed.range(of: "</think>") else { return trimmed }
        return String(trimmed[closeRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

nonisolated enum ProofreadModelError: Error {
    case notLoaded
}
