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
    }

    /// Chunk-prefill `text` into `cache`.
    ///
    /// - Parameters:
    ///   - checkpoints: absolute token offsets to capture at, with their types.
    ///   - checkpointBaseOffset: number of leading prompt tokens the cache
    ///     already covers (0 cold; the restored snapshot offset on a hit).
    ///   - consumeAll: when true every input token is processed (leaf
    ///     re-prefill); when false the final token is left for the iterator.
    static func run(
        model: any LanguageModel,
        text: LMInput.Text,
        cache: [any KVCache],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType] = [:],
        checkpointBaseOffset: Int = 0,
        prefillStepSize: Int,
        consumeAll: Bool = false
    ) throws -> Output {
        let ndim = text.tokens.ndim
        let total = text.tokens.dim(-1)
        var y = text

        // One chunk forward pass. Mirrors the fork's prepare loops: 1D inputs
        // get the batch axis added here, 2D (VLM conditional-generation)
        // inputs are already batched. The fork passed `state: nil` per chunk;
        // hybrid models carry recurrent state inside the cache (MambaCache),
        // not in LMOutput.State.
        func forward(_ chunkSize: Int) {
            let chunk = ndim >= 2 ? y[0..., ..<chunkSize] : y[.newAxis, ..<chunkSize]
            _ = model(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = ndim >= 2 ? y[0..., chunkSize...] : y[chunkSize...]
        }

        let keepBack = consumeAll ? 0 : 1
        guard total > keepBack else {
            return Output(snapshots: [], remainder: y)
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
            forward(tail)
        }

        return Output(snapshots: snapshots, remainder: y)
    }
}
