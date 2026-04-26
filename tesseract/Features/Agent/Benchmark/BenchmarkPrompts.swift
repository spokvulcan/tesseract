import Foundation
import MLXLMCommon

enum BenchmarkPrompts {

    /// Repeats `base` until the tokenizer produces at least `targetTokens`
    /// tokens, then truncates exactly so each run yields identical bytes.
    /// The token sequence is sensitive to `base` content — different bases
    /// produce different decoding trajectories under speculative decoding,
    /// so callers that want comparability across runs must use the same base.
    nonisolated static func deterministic(
        targetTokens: Int, tokenizer: any Tokenizer, base: String
    ) -> [Int] {
        var combined = base
        var encoded = tokenizer.encode(text: combined, addSpecialTokens: false)
        while encoded.count < targetTokens {
            combined += " " + base
            encoded = tokenizer.encode(text: combined, addSpecialTokens: false)
        }
        return Array(encoded.prefix(targetTokens))
    }

    /// Long lorem-ipsum passage used by `HybridCacheCorrectnessRunner`.
    nonisolated static let hybridCacheBase = """
        The cache verification harness must produce a deterministic, \
        reproducible token sequence. We compose long passages from a \
        fixed lexicon so the same target length yields identical bytes \
        on every run, which in turn guarantees identical model state \
        and lets the bitwise equality assertions hold. Lorem ipsum dolor \
        sit amet, consectetur adipiscing elit. Sed do eiusmod tempor \
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim \
        veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip \
        ex ea commodo consequat. Duis aute irure dolor in reprehenderit \
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \
        Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
        officia deserunt mollit anim id est laborum.
        """

    /// Short passage used by `DFlashCorrectnessRunner`. Acceptance baseline
    /// in `docs/dflash-m2-progress-2026-04-26.md` was measured against this
    /// exact text — changing it shifts the decoding trajectory.
    nonisolated static let dflashShortBase = """
        The cache verification harness must produce a deterministic, \
        reproducible token sequence. We compose long passages from a \
        fixed lexicon so the same target length yields identical bytes \
        on every run, which in turn guarantees identical model state \
        and lets the bitwise equality assertions hold.
        """
}
