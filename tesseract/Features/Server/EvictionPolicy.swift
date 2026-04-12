import Foundation
import MLXLMCommon

/// Marconi-style FLOP-aware eviction policy.
///
/// Scores eligible snapshot nodes by `utility = norm(R) + alpha * norm(F/B)`,
/// where `R = 1 / ageSeconds`, `F` is the parent-relative FLOPs the node's
/// snapshot saves on a future hit, and `B = snapshot.memoryBytes`. Both `R`
/// and `F/B` are min-max normalized across the candidate set, then summed
/// with `alpha` weighting the FLOP-efficiency term.
///
/// Caller is responsible for filtering to the Marconi-eligible candidate
/// set (snapshot present AND `childCount <= 1`); multi-child branch nodes
/// hold shared cache state and must not be scored here.
/// FLOP/state-size constants for one transformer architecture. Pure value
/// type — non-isolated so non-MainActor callers (e.g. `LLMActor` reading
/// `config.json` at model load) can construct one and hand it to the
/// `@MainActor`-isolated `EvictionPolicy.modelProfile`.
nonisolated struct ModelFlopProfile: Equatable, Sendable {
    let attentionLayers: Int
    let ssmLayers: Int
    let mlpLayers: Int
    /// Hidden size `D`.
    let hiddenSize: Int
    /// SSM state dim `N` for the F_ssm formula. For GatedDeltaNet this is
    /// `linearNumValueHeads * linearKeyHeadDim`.
    let ssmStateDim: Int

    /// Build a profile for a Qwen3.5-style hybrid architecture (alternating
    /// linear-attention/full-attention blocks with one MLP per transformer
    /// block). The attention/SSM split is derived from
    /// `fullAttentionInterval`: every `interval`-th layer is attention.
    static func qwen35(
        hiddenLayers: Int,
        hiddenSize: Int,
        linearNumValueHeads: Int,
        linearKeyHeadDim: Int,
        fullAttentionInterval: Int
    ) -> ModelFlopProfile {
        let interval = max(fullAttentionInterval, 1)
        let attentionLayers = hiddenLayers / interval
        return ModelFlopProfile(
            attentionLayers: attentionLayers,
            ssmLayers: hiddenLayers - attentionLayers,
            mlpLayers: hiddenLayers,
            hiddenSize: hiddenSize,
            ssmStateDim: linearNumValueHeads * linearKeyHeadDim
        )
    }

    /// Fallback profile matching the Qwen3.5-4B-PARO `text_config.json`
    /// shipped by `z-lab/Qwen3.5-4B-PARO`. Used when the loaded model's
    /// config can't be parsed.
    static let qwen35_4B_PARO = ModelFlopProfile.qwen35(
        hiddenLayers: 32,
        hiddenSize: 2560,
        linearNumValueHeads: 32,
        linearKeyHeadDim: 128,
        fullAttentionInterval: 4
    )
}

@MainActor
enum EvictionPolicy {

    /// FLOP weighting. `0` (the default) collapses utility to pure recency,
    /// equivalent to LRU within the eligible set.
    static var alpha: Double = 0.0

    /// Active model FLOP profile. Production code (`LLMActor`) overrides
    /// this at model-load time after parsing the loaded model's
    /// `config.json`. The default is the Qwen3.5-4B-PARO shape and exists
    /// only as a fallback for unit tests and unknown architectures.
    static var modelProfile: ModelFlopProfile = .qwen35_4B_PARO

    // MARK: - FLOP formulas (Marconi Appendix A / reference repo `utils.py`)

    /// Parent-relative FLOPs saved by reusing a snapshot at `nodeOffset`.
    ///
    /// SSM term uses the linear delta directly because `F_ssm` is linear in
    /// `L`; attention/MLP use absolute differences because the `L^2` term
    /// in `F_attn` does not factor cleanly into a parent-relative form.
    static func parentRelativeFlops(
        nodeOffset: Int,
        parentOffset: Int
    ) -> Double {
        let total = max(nodeOffset, 0)
        let parent = max(min(parentOffset, total), 0)
        let delta = total - parent
        guard delta > 0 else { return 0 }

        let profile = modelProfile
        let D = profile.hiddenSize
        let N = profile.ssmStateDim

        let ssmTerm = Double(profile.ssmLayers) * flopSSM(length: delta, hidden: D, stateDim: N)
        let attnTerm = Double(profile.attentionLayers)
            * (flopAttention(length: total, hidden: D) - flopAttention(length: parent, hidden: D))
        let mlpTerm = Double(profile.mlpLayers)
            * (flopMLP(length: total, hidden: D) - flopMLP(length: parent, hidden: D))

        return ssmTerm + attnTerm + mlpTerm
    }

    /// `F_attn(L, D) = 8 * L * D^2 + 4 * L^2 * D`
    private static func flopAttention(length L: Int, hidden D: Int) -> Double {
        let Ld = Double(L)
        let Dd = Double(D)
        return 8 * Ld * Dd * Dd + 4 * Ld * Ld * Dd
    }

    /// `F_mlp(L, D) = 16 * L * D^2`
    private static func flopMLP(length L: Int, hidden D: Int) -> Double {
        let Ld = Double(L)
        let Dd = Double(D)
        return 16 * Ld * Dd * Dd
    }

    /// `F_ssm(L, D, N) = 12 * L * D^2 + 16 * L * D * N + 10 * L * D`
    private static func flopSSM(length L: Int, hidden D: Int, stateDim N: Int) -> Double {
        let Ld = Double(L)
        let Dd = Double(D)
        let Nd = Double(N)
        return 12 * Ld * Dd * Dd + 16 * Ld * Dd * Nd + 10 * Ld * Dd
    }

    // MARK: - Min-max normalization

    /// Min-max normalize to `[0, 1]`. Returns `[1, 1, ...]` for all-equal or
    /// single-element inputs — degenerate case where the term carries no
    /// signal and every candidate is tied.
    static func normalize(_ values: [Double]) -> [Double] {
        guard values.count > 1 else {
            return Array(repeating: 1.0, count: values.count)
        }
        guard let minValue = values.min(),
              let maxValue = values.max(),
              minValue != maxValue
        else {
            return Array(repeating: 1.0, count: values.count)
        }
        let span = maxValue - minValue
        return values.map { ($0 - minValue) / span }
    }

    // MARK: - Scoring

    /// Score a candidate set. Returns scores in the same order as `candidates`.
    /// `now` is passed in so a multi-iteration eviction loop can pin a single
    /// clock reading across passes.
    static func computeScores(
        candidates: [RadixTreeNode],
        now: ContinuousClock.Instant
    ) -> [EvictionScore] {
        guard !candidates.isEmpty else { return [] }

        let rawRecencies: [Double] = candidates.map { node in
            let age = max((now - node.lastAccessTime).seconds, 1e-6)
            return 1.0 / age
        }
        let normRecencies = normalize(rawRecencies)
        let alpha = Self.alpha

        // Fast path: alpha == 0 collapses utility to pure recency. Skip the
        // FLOP work entirely — `parentRelativeFlops` runs `O(L²)` math per
        // candidate and the result is multiplied by zero.
        if alpha == 0 {
            return normRecencies.map { nR in
                EvictionScore(normalizedRecency: nR, normalizedFlopEfficiency: 0, utility: nR)
            }
        }

        let rawFlopEffs: [Double] = candidates.map { node in
            guard let snapshot = node.snapshot, snapshot.memoryBytes > 0 else {
                return 0.0
            }
            let parentOffset = node.parent?.tokenOffset ?? 0
            let deltaF = parentRelativeFlops(
                nodeOffset: node.tokenOffset,
                parentOffset: parentOffset
            )
            return deltaF / Double(snapshot.memoryBytes)
        }
        let normFlopEffs = normalize(rawFlopEffs)

        return zip(normRecencies, normFlopEffs).map { nR, nF in
            EvictionScore(
                normalizedRecency: nR,
                normalizedFlopEfficiency: nF,
                utility: nR + alpha * nF
            )
        }
    }

    /// Pick the candidate with the lowest utility, or `nil` if empty.
    static func selectVictim(
        candidates: [RadixTreeNode],
        now: ContinuousClock.Instant = .now
    ) -> (node: RadixTreeNode, score: EvictionScore)? {
        guard !candidates.isEmpty else { return nil }

        let scores = computeScores(candidates: candidates, now: now)

        var lowestIdx = 0
        for i in 1..<scores.count where scores[i].utility < scores[lowestIdx].utility {
            lowestIdx = i
        }
        return (node: candidates[lowestIdx], score: scores[lowestIdx])
    }
}

/// Per-snapshot eviction score: normalized recency + FLOP-efficiency terms,
/// plus the final weighted utility. `Comparable` order is by `utility`, so
/// `scores.min()` returns the eviction victim's score.
struct EvictionScore: Comparable, Sendable {
    let normalizedRecency: Double
    let normalizedFlopEfficiency: Double
    let utility: Double

    static func < (lhs: EvictionScore, rhs: EvictionScore) -> Bool {
        lhs.utility < rhs.utility
    }
}

extension Duration {
    /// Total elapsed seconds as a `Double`. Combines integer seconds with
    /// the fractional attoseconds component (`1 atto = 1e-18 s`).
    var seconds: Double {
        let components = self.components
        return Double(components.seconds)
            + Double(components.attoseconds) * 1e-18
    }
}
