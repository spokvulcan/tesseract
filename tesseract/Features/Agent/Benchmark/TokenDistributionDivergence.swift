import Foundation
import MLX

/// Per-token distributional-divergence measurements for the ASR-vs-gold gate
/// (issue #134). The spike's verdict rests on these: the trim-and-restore
/// diagnostic (`HybridCacheCorrectnessRunner.swift` test 9) and the production
/// offset-alignment guard (`ServerCompletion.swift`) both show argmax stays
/// stable at `trim = 1` while the sampled distribution drifts ~10×, so
/// argmax/greedy agreement alone is a false-success mode. KL divergence +
/// top-k agreement over the real serving temperature is the gate's required
/// measurement.
nonisolated enum TokenDistributionDivergence {

    /// Symmetric sanity tolerance for a probability sum (softmax over a large
    /// vocab in float32 can drift slightly off 1.0).
    static let probabilitySumTolerance: Float = 1e-4

    /// `KL(p ‖ q)` in nats, with `p`, `q` derived by softmax over the given
    /// logits at the real serving temperature. Returns `nil` if either
    /// distribution is degenerate (non-finite or empty). Both arrays must be
    /// the same 1-D shape over the vocabulary.
    ///
    /// `p` is the **gold** (the full, consistent re-prefill of the stripped
    /// stretch); `q` is the **ASR** distribution. KL(p‖q) penalizes ASR most
    /// where gold assigned mass that ASR did not — the direction that surfaces
    /// stale-recurrent-state drift as a single number per token.
    static func klDivergence(gold: MLXArray, asr: MLXArray, temperature: Float) -> Float? {
        guard gold.size == asr.size, gold.size > 0 else { return nil }
        let goldF = gold.asType(.float32) / max(temperature, Float.ulpOfOne)
        let asrF = asr.asType(.float32) / max(temperature, Float.ulpOfOne)
        let p = MLX.softmax(goldF, axis: -1, precise: true)
        let q = MLX.softmax(asrF, axis: -1, precise: true)
        // KL(p‖q) = Σ p · log((p + ε) / (q + ε)) — the epsilon keeps the log
        // finite where q has zero mass (a token gold can still sample).
        let eps = MLXArray(Float.ulpOfOne)
        let ratio = (p + eps) / (q + eps)
        let terms = p * MLX.log(ratio)
        let sum = terms.sum()
        let value = sum.item(Float.self)
        return value.isFinite ? value : nil
    }

    /// Top-k agreement: the fraction of the gold top-k set that also appears in
    /// the ASR top-k set (intersection-over-k). `1.0` means ASR preserves
    /// gold's entire top-k; `0.0` means total disagreement. Returns `nil` if
    /// the inputs are degenerate.
    static func topKAgreement(gold: MLXArray, asr: MLXArray, k: Int) -> Float? {
        guard gold.size == asr.size, gold.size > 0, k > 0 else { return nil }
        let goldTop = Set(topIndices(gold, k: k))
        let asrTop = Set(topIndices(asr, k: k))
        let overlap = goldTop.intersection(asrTop).count
        return Float(overlap) / Float(min(k, goldTop.count))
    }

    /// Argmax agreement — recorded as a *lower bound only* (the trim-and-restore
    /// diagnostic and the offset-alignment guard show argmax survives while the
    /// distribution drifts); never the verdict.
    static func argmaxAgrees(gold: MLXArray, asr: MLXArray) -> Bool {
        guard gold.size == asr.size, gold.size > 0 else { return false }
        return topIndices(gold, k: 1) == topIndices(asr, k: 1)
    }

    private static func topIndices(_ logits: MLXArray, k: Int) -> [Int] {
        let flat = logits.asType(.float32).reshaped(logits.size)
        eval(flat)
        let values = flat.asArray(Float.self)
        let count = min(k, values.count)
        // Indices of the `count` largest values, ranked descending. Vocab-sized
        // selection is fine for the offline harness (not a hot path).
        let ranked = values.indices.sorted { values[$0] > values[$1] }
        return Array(ranked.prefix(count))
    }
}
