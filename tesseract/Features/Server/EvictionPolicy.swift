import Foundation
import MLXLMCommon

/// Recovery-cost eviction policy — Marconi's utility blend adapted to a
/// two-tier cache (ADR-0011).
///
/// Scores eligible snapshot nodes by `utility = norm(R) + alpha * norm(F/B)`,
/// where `R = 1 / ageSeconds`, `F` is the **Recovery Cost** in seconds —
/// what the next hit *pays* if the body leaves RAM — and
/// `B = snapshot.memoryBytes`. Tier-aware F: an SSD-backed body (any
/// surviving **Snapshot Ref**) costs hydration seconds
/// (`bytesOnDisk / hydration bytes/s`); an unbacked body is a terminal
/// loss costing re-prefill seconds (parent-relative FLOPs / prefill
/// FLOPs/s). Both rates come from the configuration's measured
/// `MeasuredSecondsEstimates` — never guessed constants.
///
/// Deliberate consequence (pinned by tests): among backed bodies,
/// recovery cost per byte is near-constant, the density term flattens
/// under min-max normalization, and ordering degenerates to recency —
/// **LRU among backed bodies is correct by design**, and the `alpha == 0`
/// fast path survives as its implementation. The blend only changes
/// outcomes where terminal losses are at stake.
///
/// Caller is responsible for filtering to the Marconi-eligible candidate
/// set (snapshot present AND `childCount <= 1`); multi-child branch nodes
/// hold shared cache state and must not be scored here.

/// FLOP/state-size constants for one transformer architecture. Pure value
/// type — non-isolated so non-MainActor callers (e.g. `LLMActor` reading
/// `config.json` at model load) can construct one and fold it into the
/// cache's **Eviction Configuration**.
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

    // Name mirrors the HuggingFace repo z-lab/Qwen3.5-4B-PARO.
    // swiftlint:disable identifier_name
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
    // swiftlint:enable identifier_name

    /// The profile assumed when the model architecture is unknown — no model
    /// loaded yet, or a config that can't be parsed. The **single home** for
    /// the "unknown ⇒ Qwen3.5-4B-PARO" default: `ModelIdentity`'s parse
    /// fallback, the `EvictionConfiguration` and `AlphaTuner` construct-time
    /// defaults, and `LLMActor`'s pre-load cache all resolve here, so a change
    /// of assumed-default model lands in exactly one place.
    static let fallback = qwen35_4B_PARO
}

/// The eviction policy's load-bearing inputs, owned per-cache by
/// `PrefixCacheManager` instead of published as process globals.
/// `flopProfile` is read once from **Model Identity** when the cache is
/// built and never changes for that model; `alpha` starts at the LRU
/// default and is adapted at runtime by the **AlphaTuner**.
/// `EvictionPolicy`'s scorers take this by value, so every caller — and
/// every test — crosses the same global-free seam.
///
/// Defaults match the retired `EvictionPolicy` statics
/// (`ModelFlopProfile.fallback`, `alpha = 0`), so eviction behavior is
/// unchanged. See `CONTEXT.md` → Eviction tuning (**Eviction Configuration**).
nonisolated struct EvictionConfiguration: Sendable {
    /// FLOP/state-size profile the recovery-cost term scores against.
    /// Immutable for the life of the cache — a model swap builds a new
    /// cache.
    let flopProfile: ModelFlopProfile

    /// Recovery-cost weighting. `0` (the default) collapses utility to
    /// pure recency, equivalent to LRU within the eligible set.
    var alpha: Double

    /// Rolling measured device estimates (prefill FLOPs/s, hydration
    /// bytes/s) that denominate **Recovery Cost** in seconds. Folded by
    /// the manager from real operations; carried here by value so every
    /// scorer and every test crosses the same global-free seam.
    var estimates: MeasuredSecondsEstimates

    init(
        flopProfile: ModelFlopProfile = .fallback,
        alpha: Double = 0.0,
        estimates: MeasuredSecondsEstimates = MeasuredSecondsEstimates()
    ) {
        self.flopProfile = flopProfile
        self.alpha = alpha
        self.estimates = estimates
    }
}

/// Marconi FLOP-aware scoring as a pure-function namespace. Holds no
/// mutable state: every scorer takes the **Eviction Configuration** by
/// value. Stays `@MainActor` because its candidates are MainActor-isolated
/// `RadixTreeNode`s, not because it owns any global.
@MainActor
enum EvictionPolicy {

    // MARK: - FLOP formulas (Marconi Appendix A / reference repo `utils.py`)

    // Single-letter names mirror the Marconi FLOP formulas (Appendix A) in the doc comments.
    // swiftlint:disable identifier_name

    /// Parent-relative FLOPs saved by reusing a snapshot at `nodeOffset`.
    ///
    /// SSM term uses the linear delta directly because `F_ssm` is linear in
    /// `L`; attention/MLP use absolute differences because the `L^2` term
    /// in `F_attn` does not factor cleanly into a parent-relative form.
    ///
    /// `nonisolated` — pure arithmetic, shared with off-MainActor callers
    /// (the prefill-throughput measurement inside `container.perform` and
    /// the **Snapshot Ledger**'s terminal-loss cut on the writer thread).
    nonisolated static func parentRelativeFlops(
        nodeOffset: Int,
        parentOffset: Int,
        profile: ModelFlopProfile
    ) -> Double {
        let total = max(nodeOffset, 0)
        let parent = max(min(parentOffset, total), 0)
        let delta = total - parent
        guard delta > 0 else { return 0 }

        let D = profile.hiddenSize
        let N = profile.ssmStateDim

        let ssmTerm = Double(profile.ssmLayers) * flopSSM(length: delta, hidden: D, stateDim: N)
        let attnTerm =
            Double(profile.attentionLayers)
            * (flopAttention(length: total, hidden: D) - flopAttention(length: parent, hidden: D))
        let mlpTerm =
            Double(profile.mlpLayers)
            * (flopMLP(length: total, hidden: D) - flopMLP(length: parent, hidden: D))

        return ssmTerm + attnTerm + mlpTerm
    }

    /// `F_attn(L, D) = 8 * L * D^2 + 4 * L^2 * D`
    private nonisolated static func flopAttention(length L: Int, hidden D: Int) -> Double {
        let Ld = Double(L)
        let Dd = Double(D)
        return 8 * Ld * Dd * Dd + 4 * Ld * Ld * Dd
    }

    /// `F_mlp(L, D) = 16 * L * D^2`
    private nonisolated static func flopMLP(length L: Int, hidden D: Int) -> Double {
        let Ld = Double(L)
        let Dd = Double(D)
        return 16 * Ld * Dd * Dd
    }

    /// `F_ssm(L, D, N) = 12 * L * D^2 + 16 * L * D * N + 10 * L * D`
    private nonisolated static func flopSSM(length L: Int, hidden D: Int, stateDim N: Int) -> Double
    {
        let Ld = Double(L)
        let Dd = Double(D)
        let Nd = Double(N)
        return 12 * Ld * Dd * Dd + 16 * Ld * Dd * Nd + 10 * Ld * Dd
    }
    // swiftlint:enable identifier_name

    // MARK: - Min-max normalization

    /// Min-max normalize to `[0, 1]`. Returns `[1, 1, ...]` for all-equal or
    /// single-element inputs — degenerate case where the term carries no
    /// signal and every candidate is tied. `nonisolated` — shared with the
    /// **Snapshot Ledger**'s terminal-loss cut on the writer thread.
    nonisolated static func normalize(_ values: [Double]) -> [Double] {
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

    // MARK: - Utility blend

    /// Raw recency weight: `1 / age`, with the epsilon that keeps a
    /// just-touched candidate finite. `nonisolated` — shared with the
    /// **Snapshot Ledger**'s terminal-loss cut on the writer thread.
    nonisolated static func recencyWeight(ageSeconds: Double) -> Double {
        1.0 / max(ageSeconds, 1e-6)
    }

    /// The Marconi utility blend — `norm(recency) + α · norm(density)` —
    /// in one place, so the RAM scorer and the **Snapshot Ledger**'s
    /// terminal-loss cut (the one shared α, ADR-0011) cannot drift.
    ///
    /// `rawDensities` is a closure because the `α == 0` fast path must
    /// skip the recovery-cost work entirely — `parentRelativeFlops`
    /// runs `O(L²)` math per candidate and the result would be
    /// multiplied by zero. This is the designed common case, not an
    /// optimization shortcut: among SSD-backed bodies the density term
    /// is flat and LRU already picks the right victim (ADR-0011).
    nonisolated static func blendedTerms(
        rawRecencies: [Double],
        alpha: Double,
        rawDensities: () -> [Double]
    ) -> [(normalizedRecency: Double, normalizedDensity: Double, utility: Double)] {
        let normRecencies = normalize(rawRecencies)
        guard alpha != 0 else {
            return normRecencies.map { ($0, 0, $0) }
        }
        let normDensities = normalize(rawDensities())
        return zip(normRecencies, normDensities).map { nR, nD in
            (nR, nD, nR + alpha * nD)
        }
    }

    // MARK: - Hydration gate (PRD #149 item 7, HiCache min-hit gate)

    /// Should a body-less SSD hit hydrate, or is re-prefilling from the
    /// best resident RAM body cheaper? HiCache guards this with a
    /// 256-token constant; here it falls out of recovery-cost pricing —
    /// the same measured estimates eviction scores with. `hitOffset` is
    /// the SSD hit's boundary, `alternativeOffset` the deepest resident
    /// RAM body on the same path (0 when none): hydration buys exactly
    /// the `alternativeOffset → hitOffset` span of prefill.
    ///
    /// Unmeasurable estimates (a zeroed throughput) admit — hydrating
    /// is today's behavior and the conservative default.
    nonisolated static func hydrationGateAdmits(
        hydrationBytes: Int,
        hitOffset: Int,
        alternativeOffset: Int,
        config: EvictionConfiguration
    ) -> Bool {
        let bytesPerSecond = config.estimates.hydrationBytesPerSecond
        let flopsPerSecond = config.estimates.prefillFlopsPerSecond
        guard bytesPerSecond > 0, flopsPerSecond > 0 else { return true }
        let hydrateSeconds = Double(max(hydrationBytes, 0)) / bytesPerSecond
        let recomputeSeconds =
            parentRelativeFlops(
                nodeOffset: hitOffset,
                parentOffset: alternativeOffset,
                profile: config.flopProfile
            ) / flopsPerSecond
        return hydrateSeconds <= recomputeSeconds
    }

    // MARK: - Scoring

    /// Score a candidate set. Returns scores in the same order as `candidates`.
    /// `now` is passed in so a multi-iteration eviction loop can pin a single
    /// clock reading across passes.
    static func computeScores(
        candidates: [RadixTreeNode],
        now: ContinuousClock.Instant,
        config: EvictionConfiguration
    ) -> [EvictionScore] {
        guard !candidates.isEmpty else { return [] }

        let rawRecencies: [Double] = candidates.map { node in
            recencyWeight(ageSeconds: (now - node.lastAccessTime).seconds)
        }
        // The density closure is only evaluated when α ≠ 0 — see
        // `blendedTerms` for the fast-path rationale.
        let terms = blendedTerms(rawRecencies: rawRecencies, alpha: config.alpha) {
            candidates.map { node in
                guard let snapshot = node.state.body, snapshot.memoryBytes > 0 else {
                    return 0.0
                }
                // Recovery Cost: seconds the *next hit* pays to get this body
                // back. Any surviving ref (pending or committed) means the body
                // drop is recovered — `evictToFitBudget` classifies outcomes by
                // the same predicate — so the cost is hydrating the on-disk
                // bytes. A **Chain-Prefix Restore** point (ADR-0012) is also
                // recovered: hydrating the owning chain's leading segments.
                // No backing at all means terminal loss: re-prefilling the
                // node's parent-relative span from scratch.
                let recoverySeconds: Double
                if node.state.ref != nil {
                    recoverySeconds =
                        Double(node.state.storageBytes)
                        / config.estimates.hydrationBytesPerSecond
                } else if let point = node.chainPrefixRestorePoint {
                    recoverySeconds =
                        Double(point.prefixBytes)
                        / config.estimates.hydrationBytesPerSecond
                } else {
                    let parentOffset = node.parent?.tokenOffset ?? 0
                    recoverySeconds =
                        parentRelativeFlops(
                            nodeOffset: node.tokenOffset,
                            parentOffset: parentOffset,
                            profile: config.flopProfile
                        ) / config.estimates.prefillFlopsPerSecond
                }
                return recoverySeconds / Double(snapshot.memoryBytes)
            }
        }

        return terms.map {
            EvictionScore(
                normalizedRecency: $0.normalizedRecency,
                normalizedFlopEfficiency: $0.normalizedDensity,
                utility: $0.utility
            )
        }
    }

    /// Pick the candidate with the lowest utility, or `nil` if empty.
    static func selectVictim(
        candidates: [RadixTreeNode],
        now: ContinuousClock.Instant = .now,
        config: EvictionConfiguration
    ) -> (node: RadixTreeNode, score: EvictionScore)? {
        guard !candidates.isEmpty else { return nil }

        let scores = computeScores(candidates: candidates, now: now, config: config)

        var lowestIdx = 0
        for i in 1..<scores.count where scores[i].utility < scores[lowestIdx].utility {
            lowestIdx = i
        }
        return (node: candidates[lowestIdx], score: scores[lowestIdx])
    }
}

/// Per-snapshot eviction score: normalized recency + recovery-cost-density
/// terms, plus the final weighted utility. `Comparable` order is by
/// `utility`, so `scores.min()` returns the eviction victim's score.
struct EvictionScore: Comparable, Sendable {
    let normalizedRecency: Double
    /// Normalized recovery cost per byte (seconds/B, min-max normalized).
    /// Field name kept from the Marconi-era F/B density — it is wired
    /// through diagnostics events and the KPI strip, and the quantity
    /// occupies the same slot in the utility blend.
    let normalizedFlopEfficiency: Double
    let utility: Double

    static func < (lhs: EvictionScore, rhs: EvictionScore) -> Bool {
        lhs.utility < rhs.utility
    }
}

extension Duration {
    /// Total elapsed seconds as a `Double`. Combines integer seconds with
    /// the fractional attoseconds component (`1 atto = 1e-18 s`).
    /// Nonisolated: pure value math, callable from any actor.
    nonisolated var seconds: Double {
        let components = self.components
        return Double(components.seconds)
            + Double(components.attoseconds) * 1e-18
    }
}
