import Foundation
import MLX
import MLXLMCommon

/// The app-owned chunked-prefill driver (ADR-0006): runs forward passes over
/// token chunks split at the planned checkpoint offsets, capturing
/// `HybridCacheSnapshot`s between chunks, and leaves the warmed cache ready
/// for a `TokenIterator`. Replaces the fork's `prepareWithCheckpoints` model
/// hook with direct calls against the upstream public model surface.
///
/// Also used checkpoint-free on VLM-class models (2D token tensors), whose
/// upstream `prepare` runs the whole prompt in a single forward pass — the
/// driver restores the fork's chunking there so long prompts keep bounded
/// peak memory.
///
/// **Metal-affinity contract:** must run inside `ModelContainer.perform`.
nonisolated enum PrefillExecutor {

    struct Output {
        let snapshots: [HybridCacheSnapshot]
        /// Unprocessed tail of the input. When `consumeAll` is false this is
        /// at least one token — the prompt tail the `TokenIterator` primes on.
        let remainder: LMInput.Text
        /// Model state after the last executed chunk (Qwen3.5-VLM carries its
        /// RoPE deltas here). Callers that keep forwarding on the same cache
        /// thread this into their next call instead of starting from nil.
        let state: LMOutput.State?
    }

    /// Chunk-prefill `text` into `cache`.
    ///
    /// - Parameters:
    ///   - checkpoints: absolute token offsets to capture at, with their types.
    ///   - checkpointBaseOffset: number of leading prompt tokens the cache
    ///     already covers (0 cold; the restored snapshot offset on a hit).
    ///   - consumeAll: when true every input token is processed (leaf
    ///     re-prefill); when false the final token is left for the iterator.
    ///   - initialState: model state the first chunk forwards with. On a
    ///     restored cache the VLM container recomputes positions from zero
    ///     when this is nil; seeding the restored conversation's RoPE delta
    ///     here keeps M-RoPE positions anchored at the restore offset.
    static func run(
        model: any LanguageModel,
        text: LMInput.Text,
        cache: [any KVCache],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType] = [:],
        checkpointBaseOffset: Int = 0,
        prefillStepSize: Int,
        consumeAll: Bool = false,
        initialState: LMOutput.State? = nil
    ) throws -> Output {
        let ndim = text.tokens.ndim
        let total = text.tokens.dim(-1)
        var y = text

        // One chunk forward pass. Mirrors upstream `LLMModel.prepare`: 1D
        // inputs get the batch axis added here, 2D (VLM conditional-
        // generation) inputs are already batched; `LMOutput.State` threads
        // from each chunk into the next (Qwen3.5-VLM carries its RoPE deltas
        // there — without it every chunk after the first restarts positions
        // at 0). Checked evaluation keeps MLX runtime failures on this path
        // as Swift throws instead of process-fatal handler dispatches.
        // Checkpoint captures synchronize explicitly before copying.
        //
        // Cancellation is observed before every chunk (issue #97): a client
        // abort mid-prefill stops within one chunk instead of running the
        // remaining prompt to the end. The cache keeps every completed
        // chunk — left un-evaluated (the last chunks may still be
        // pipelined), so a catcher that wants the partial state must
        // `eval(cache)` before reading it (**Salvage-on-cancel**).
        var chunkState: LMOutput.State? = initialState
        func forward(_ chunkSize: Int) throws {
            try Task.checkCancellation()
            let chunk = ndim >= 2 ? y[0..., ..<chunkSize] : y[.newAxis, ..<chunkSize]
            let output = model(chunk, cache: cache.isEmpty ? nil : cache, state: chunkState)
            chunkState = output.state
            try MLXCheckedEvaluation.eval(cache)
            y = ndim >= 2 ? y[0..., chunkSize...] : y[chunkSize...]
        }

        let keepBack = consumeAll ? 0 : 1
        guard total > keepBack else {
            return Output(snapshots: [], remainder: y, state: chunkState)
        }

        // The checkpoint-aware loop never consumes the final token (its main
        // loop and tail drain both stop short), so a checkpoint at offset
        // total-1 is still captured while the iterator's prime token survives.
        let (consumed, snapshots) = try HybridCacheSnapshot.chunkedPrefill(
            totalTokens: total,
            prefillStepSize: prefillStepSize,
            checkpoints: checkpoints,
            checkpointBaseOffset: checkpointBaseOffset,
            cache: cache,
            processChunk: forward
        )

        // Drain the checkpoint-free tail (at most one prefill step) in one
        // final chunk, keeping back the iterator's prime token if requested.
        let tail = (total - keepBack) - consumed
        if tail > 0 {
            try forward(tail)
        }

        // Flush the pipelined chunks and release the prefill's scratch
        // buffers (the fork cleared the MLX buffer pool after prefill too) —
        // long cold prefills otherwise accumulate per-chunk allocations.
        try MLXCheckedEvaluation.eval(cache)
        Memory.clearCache()

        return Output(snapshots: snapshots, remainder: y, state: chunkState)
    }

    /// Build the post-prefill `TokenIterator`.
    ///
    /// The iterator only sees the unconsumed `remainder`, but upstream seeds
    /// its penalty processors from its own input (`processor?.prompt`) —
    /// which would shrink the repetition/presence/frequency context to the
    /// remainder's single token. When penalties are configured, the processor
    /// is created here and seeded with the full prompt suffix instead.
    ///
    /// The explicit-processor `TokenIterator` init has no in-iterator cache
    /// quantization, so that path quantizes once up front — the same boundary
    /// the per-step path would hit on its first decode step
    /// (`quantizedKVStart` defaults to 0).
    static func makeIterator(
        model: any LanguageModel,
        fullText: LMInput.Text,
        remainder: LMInput.Text,
        cache: inout [any KVCache],
        parameters: GenerateParameters
    ) throws -> TokenIterator {
        guard let penaltyProcessor = parameters.processor() else {
            return try TokenIterator(
                input: LMInput(text: remainder),
                model: model,
                cache: cache,
                parameters: parameters
            )
        }
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )
        return try TokenIterator(
            input: LMInput(text: remainder),
            model: model,
            cache: cache,
            processor: PrefillSeededLogitProcessor(
                inner: penaltyProcessor,
                fullPrompt: fullText.tokens
            ),
            sampler: parameters.sampler(),
            prefillStepSize: parameters.prefillStepSize,
            maxTokens: parameters.maxTokens
        )
    }
}

/// Wraps the parameters' logit processor so penalty rings are seeded with the
/// full prompt suffix: `TokenIterator.prepare` calls
/// `processor?.prompt(input.text.tokens)` with the iterator's own input, and
/// after chunked prefill that input is only the final prompt token.
/// `TokenRing.loadPrompt` keeps the trailing context-size window, so seeding
/// with the full suffix reproduces the unchunked behavior.
private nonisolated struct PrefillSeededLogitProcessor: LogitProcessor {
    private var inner: any LogitProcessor
    private let fullPrompt: MLXArray

    init(inner: any LogitProcessor, fullPrompt: MLXArray) {
        self.inner = inner
        self.fullPrompt = fullPrompt
    }

    mutating func prompt(_ prompt: MLXArray) {
        inner.prompt(fullPrompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        inner.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        inner.didSample(token: token)
    }
}
