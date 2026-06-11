import MLX
import MLXLMCommon

/// The model-affine half of the **Position Anchor** (CONTEXT.md): turns the
/// rope delta the **Cache Key Space** reconstructs from its image table into
/// the `LMOutput.State` a warm forward on the Qwen3.5 vision container must
/// resume with. The vendor reads the delta under this key whenever the cache
/// offset is non-zero (`Qwen35.callAsFunction`'s continuation branch);
/// without it the container recomputes M-RoPE positions from zero.
///
/// The key string and the app-side delta reconstruction are both pinned
/// against vendor drift by the VLM smoke harness
/// (`ParoQuantVLMSmokeRunner.prepareStateCarriesRopeDelta`), which keeps its
/// own copy of the key on purpose — an independent tripwire, not shared code.
nonisolated enum PositionAnchor {
    /// The vendor-internal Qwen3.5 state key (public `LMOutput.Key(String)` init).
    static let qwen35RopeDeltas = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

    /// A fresh model state carrying `ropeDelta` as the M-RoPE continuation
    /// delta — zero for image-free prefixes, Σ(positionSpan − runLength) over
    /// the cached images otherwise.
    static func seededState(ropeDelta: Int) -> LMOutput.State {
        var state = LMOutput.State()
        state[qwen35RopeDeltas] = MLXArray([Int32(ropeDelta)])
        return state
    }
}
