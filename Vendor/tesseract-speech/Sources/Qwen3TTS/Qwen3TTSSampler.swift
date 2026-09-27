import Foundation
@preconcurrency import MLX

/// Draws a token the way Qwen's `generate` does: temperature, then top-k,
/// then top-p, then a categorical draw. Temperature 0 is greedy. Everything
/// stays on the GPU; nothing waits for a value.
struct Qwen3TTSSampler {
    let temperature: Float
    let topK: Int
    let topP: Float

    /// `logits` `[1, vocab]` to a `[1, 1]` int32 token.
    func callAsFunction(_ logits: MLXArray, random: MLXRandom.RandomState) -> MLXArray {
        guard temperature > 0 else {
            return argMax(logits, axis: -1, keepDims: true).asType(.int32)
        }
        if Qwen3TTSKernels.canSample(logits, temperature: temperature, topK: topK, topP: topP) {
            // The same draw in one kernel instead of about fourteen.
            return Qwen3TTSKernels.topKSample(
                logits, temperature: temperature, topK: topK, random: random)
        }
        let filtered = Self.filter(logits / temperature, topK: topK, topP: topP)
        return categorical(filtered, key: random).asType(.int32).reshaped(1, 1)
    }

    /// Top-k then top-p over `[1, vocab]` logits: a dropped token gets -inf,
    /// the rest keep their logits. Top-k keeps every logit at least the k-th
    /// largest, so ties at the cut stay (as in transformers' TopKLogitsWarper).
    static func filter(_ logits: MLXArray, topK: Int, topP: Float) -> MLXArray {
        var x = logits
        let vocab = x.dim(-1)
        let negativeInfinity = MLXArray(-Float.infinity).asType(x.dtype)
        if topK > 0, topK < vocab {
            let kth = -partitioned(-x, kth: topK - 1, axis: -1)[0..., (topK - 1) ..< topK]
            x = which(x .< kth, negativeInfinity, x)
        }
        if topP > 0, topP < 1 {
            // Drop the tokens whose cumulative probability, smallest first,
            // stays within 1 - top-p: what is left is the nucleus.
            let order = argSort(x, axis: -1)
            let cumulative = cumsum(takeAlong(softmax(x, axis: -1), order, axis: -1), axis: -1)
            let drop = zeros(like: cumulative).asType(.bool)
            let dropSorted = cumulative .<= (1 - topP)
            let dropped = putAlong(drop, order, values: dropSorted, axis: -1)
            x = which(dropped, negativeInfinity, x)
        }
        return x
    }
}

/// The talker's sampler: Qwen's logits processors, then `Qwen3TTSSampler`.
///
/// - The top 1,024 codec ids, the control tokens, are suppressed, except EOS.
/// - EOS is suppressed for the first `minFrames` frames (Qwen's
///   `min_new_tokens = 2`).
/// - The repetition penalty divides (or, below zero, multiplies) the logit of
///   every token among the last `window` talker tokens. Qwen counts the whole
///   generation; the engine counts a window (ADR-0072), so a long segment's
///   common tokens don't all end up penalized. The window lives on the GPU.
struct Qwen3TTSTalkerSampler {
    let base: Qwen3TTSSampler
    let penalty: Float
    let minFrames: Int
    private let suppress: MLXArray
    private let suppressWithEOS: MLXArray
    private let vocabulary: MLXArray
    /// The last `window` tokens, oldest first; -1 where none yet.
    private(set) var recent: MLXArray

    init(
        sampling: Qwen3TTSSampling, vocabSize: Int, eosTokenID: Int, dtype: DType,
        window: Int = Qwen3TTSSampling.repetitionWindow, minFrames: Int = 2
    ) {
        base = Qwen3TTSSampler(
            temperature: sampling.temperature, topK: Qwen3TTSSampling.topK, topP: sampling.topP)
        penalty = sampling.repetitionPenalty
        self.minFrames = minFrames
        var mask = [Float](repeating: 0, count: vocabSize)
        for id in max(0, vocabSize - 1024) ..< vocabSize where id != eosTokenID {
            mask[id] = -.infinity
        }
        suppress = MLXArray(mask).reshaped(1, vocabSize).asType(dtype)
        mask[eosTokenID] = -.infinity
        suppressWithEOS = MLXArray(mask).reshaped(1, vocabSize).asType(dtype)
        vocabulary = MLXArray(Int32(0) ..< Int32(vocabSize)).reshaped(1, vocabSize)
        recent = MLXArray([Int32](repeating: -1, count: max(1, window))).reshaped(max(1, window), 1)
    }

    mutating func callAsFunction(
        _ logits: MLXArray, frame: Int, random: MLXRandom.RandomState
    ) -> MLXArray {
        var x = logits
        if penalty != 1 {
            let seen = (recent .== vocabulary).any(axis: 0, keepDims: true)
            x = which(seen, which(x .< 0, x * penalty, x / penalty), x)
        }
        x = x + (frame < minFrames ? suppressWithEOS : suppress)
        let token = base(x, random: random)
        recent = concatenated([recent[1..., 0...], token], axis: 0)
        return token
    }
}
