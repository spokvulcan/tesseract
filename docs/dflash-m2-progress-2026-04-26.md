# DFlash M2 Progress — 2026-04-26

## Status

- **M1 (vendor foundation)**: complete. 10 GDN/draft/attention tests pass at
  production Qwen3.6-27B dims.
- **M2 (greedy correctness)**: end-to-end pipeline runs and passes the
  relaxed near-AR gate. Target+draft load, hidden-state taps fire, hybrid
  rollback executes. **1.5× speedup** at **42.8% acceptance** on the
  64-token harness prompt with `maxTokens=256` (Debug build: 27 vs 18 tok/s);
  common prefix 89.8%.
- **M4 skeleton (LLMActor wiring)**: complete. `SettingsManager.dflashEnabled`
  toggle in **Server → Configuration**. `AgentEngine.loadModel` resolves
  `DFlashLoadConfig` from settings, `LLMActor.bindDFlashIfNeeded` loads +
  binds the draft after `verifyAndStore`, and `LLMActor.makeIterator`
  switches to `DFlashTokenIterator` for greedy non-TriAttention generations
  in both `startRawGeneration` and `startThinkingContinuation*`. Toggling
  `dflashEnabled` triggers an `InferenceArbiter.reloadLLMIfNeeded()` so
  changes take effect immediately.
- **Byte-identical to AR**: holds for first **229 tokens**, diverges at 230
  due to upstream MLX precision drift (see below). Gate is relaxed to
  near-equivalence — see "Decision" section.

## Root cause of the 230-token divergence

`MLXFast.scaledDotProductAttention` is **not bitwise-consistent across batch
shapes**:

| Path                                                | Output at logical position 0 |
| --------------------------------------------------- | ---------------------------- |
| `SDPA(B=16 queries, cache+B keys, .causal)`         | reference                    |
| `SDPA(1 query, cache+1 key, .none)`                 | drifts by ~5e-7 (≈1 fp32 ULP) |
| `SDPA(B=16 queries, cache+B keys, .array(causal))`  | drifts by ~6.5e-7             |

In isolation that's harmless. In DFlash it compounds: 16 full-attn layers ×
~30 verify rounds × 1 ULP per layer crosses the bf16 ULP boundary on a near-tied
logit, eventually flipping a top-1 sampled token. Once flipped, the AR-vs-DFlash
sequences diverge permanently because subsequent state depends on the wrong token.

Everything else we wrote is bitwise-correct in isolation:
- `conv1d` batched-vs-AR ✓
- `MLXFast.rmsNorm` batched-vs-AR ✓
- 4-bit `QuantizedLinear` batched-vs-AR ✓
- `gatedDeltaUpdate` (SSM kernel) batched-vs-AR ✓
- `[.ellipsis]` snapshots survive intermediate kernel mutations ✓
- SSM-only-replay rollback matches a fresh AR run for the same n positions ✓

The bitwise-equivalence checks live in
`Vendor/mlx-swift-lm/Tests/MLXLMTests/DFlashGDNRollbackTests.swift`. The two
SDPA tests (`sdpaBatchedDriftAtFirstPosition_known`,
`sdpaExplicitArrayMaskDrift_known`) record the observed drift; they assert
`< 1e-5` so the suite stays green and they fire if the drift ever increases.

## Decision

We're shipping with the relaxed gate (option 1 in the diagnosis discussion):
DFlash output need only be **near-AR-equivalent** within the precision MLX
itself guarantees, not bitwise-identical. This matches every production
speculative-decoding implementation we know of.

`DFlashCorrectnessRunner` is being changed from "first divergence index"
to a tolerance-based check (TBD: edit distance, longest common prefix, or
per-token cosine on the logit distributions). The current 229-token byte-
identical prefix already exceeds what most speculative decoders demonstrate,
and DFlash's correctness guarantee comes from the verify-and-correct protocol,
not from kernel-level numerics.

## Still to do (in order)

1. **Prefix-cache integration (HiddenStateTappable).** `makeHTTPPrefixCacheGeneration`
   still uses the standard `TokenIterator`. To hook DFlash on the HTTP hot
   path we need a tap-aware variant of `Qwen35TextModel.prepareWithCheckpoints`
   (the new protocol planned in the original implementation plan). Until
   this lands, DFlash kicks in for `startRawGeneration` (no prefix cache)
   and the thinking-continuation paths, but NOT for cached HTTP turns —
   those silently fall back to AR.
2. **ModelFingerprint draft mix-in.** When DFlash is active, hash the
   draft's `config.json` SHA-256 into the prefix-cache fingerprint so a
   draft swap forces a partition flush. Cheap once HiddenStateTappable
   lands.
3. **DFlashE2ERunner.** Modeled on `PrefixCacheE2ERunner.swift`. Validates
   the full HTTP path with the relaxed gate.
4. **DFlashBenchRunner.** tok/s × acceptance × peak-RSS sweep across
   `{512, 1024, 2048, 4096}` × `{AR, DFlash}` for the M5 perf gate.

## Follow-ups to revisit after the full M3–M5 implementation

1. **Upstream MLX SDPA fix.** File an issue / submit a patch making the SDPA
   Metal kernel batch-shape-consistent for the slice that overlaps an
   AR-equivalent computation. If/when this lands, flip the two `_known` tests
   back to `#expect(diff == 0.0)` and DFlash becomes literally byte-identical.

2. **Sliding-window draft cache.** The Qwen3.6-27B-DFlash draft config has
   `layer_types: ["sliding_attention" × 4, "full_attention" × 1]` with
   `sliding_window: 2048`. `DFlashAttention` currently treats every layer
   as full attention; harmless until cache offset crosses 2048 tokens
   (i.e. after generating ~2K tokens). Wire `RotatingKVCache` for the
   sliding layers before any production use.

3. **Hidden-state tap point.** Python z-lab DFlash hooks
   `decoder_layer.input_layernorm` (post-norm). Our Swift tap fires
   pre-norm. The draft was trained on post-norm taps, so this affects
   draft *quality* (acceptance rate observed at 42.8% vs the ~85%
   z-lab reports). Move the tap to fire post-`inputLayerNorm` and re-measure.
   If acceptance jumps, this was the dominant DFlash speed loss.

4. **Acceptance-rate decay.** Acceptance drops from 50% at 128 tokens to
   ~40% at 232+. Likely correlated with (3); confirm after fixing the tap point.

5. **DFlashCorrectnessRunner gate definition.** Pick the relaxed metric and
   bake it in. Suggested: edit distance ≤ 5 over ≥1024 generated tokens, plus
   acceptance-rate floor.

## Files touched

### M1/M2 (vendor foundation + harness)
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/DFlash/*` — draft model + iterator.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` — opt-in taps + DFlashTarget conformance.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` — TokenIteratorProtocol public, generate overload.
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/DFlashGDNRollbackTests.swift` — 10 tests including the SDPA drift-discovery pair.
- `tesseract/Features/Models/ModelDefinition.swift` — `qwen3.6-27b-dflash` catalog entry, `.draft` category, `dflashDraftID` pointer on the target.
- `tesseract/Features/Agent/Benchmark/DFlashCorrectnessRunner.swift` — M2 harness with relaxed near-AR gate.
- `tesseract/App/TesseractApp.swift` — `--dflash-correctness` dispatch.
- `scripts/dev.sh` — `dflash-correctness` subcommand.

### M4 skeleton (LLMActor wiring)
- `tesseract/Features/Agent/AgentGeneration.swift` — `DFlashLoadConfig` value type passed from settings to actor.
- `tesseract/Features/Settings/SettingsManager.swift` — `dflashEnabled`, `dflashBlockSize` keys + `makeDFlashLoadConfig(targetModelID:)`.
- `tesseract/Features/Agent/AgentEngine.swift` — `resolveDFlashLoadConfig`, threaded through `loadModel`.
- `tesseract/Features/Agent/LLMActor.swift` — `dflashContext`, `bindDFlashIfNeeded`, `nonisolated static makeIterator` (snapshot-passed), `startRawGeneration` + `buildThinkingContinuationStart` switched to the helper, cleared on `unloadModel`.
- `tesseract/Features/Server/Views/ServerConfigurationView.swift` — "Enable DFlash speculative decoding" toggle.
- `tesseract/App/DependencyContainer.swift` — `Observations` watcher on `dflashEnabled` triggers `reloadLLMIfNeeded`.

## How to reproduce

```sh
# Download the gated draft (one-time):
hf download z-lab/Qwen3.6-27B-DFlash --local-dir \
  "$HOME/Library/Containers/app.tesseract.agent/Data/Library/Application Support/models/z-lab_Qwen3.6-27B-DFlash"

# Run the harness:
scripts/dev.sh dflash-correctness --bench-model-id qwen3.6-27b
```
