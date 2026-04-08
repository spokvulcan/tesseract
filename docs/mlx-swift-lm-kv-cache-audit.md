# mlx-swift-lm KV Cache Audit

Phase 2a research spike T14 for [HTTP_SERVER_SPEC.md](./HTTP_SERVER_SPEC.md).

Audit target:
- `mlx-swift-lm` pin from `tesseract.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`
- Repo: `https://github.com/spokvulcan/mlx-swift-lm.git`
- Branch / revision: `test/tesseract-integration` @ `85b3adda84fe4c95172c78e0fe99f9d8fe72a0db`

All `Libraries/...` source paths below are relative to that audited `mlx-swift-lm` revision.

## Executive Summary

The pinned fork already exposes more cache API than the spec assumed. Public `KVCache.state` / `metaState`, `savePromptCache`, `loadPromptCache`, `ChatSession.init(cache:)`, and `ChatSession.currentCache()` mean external cache save/restore is already possible without another fork.

The remaining gap is not basic extraction. The gap is getting the final, post-generation cache array back out of the low-level generation path when `TokenIterator` swaps cache objects internally, especially when `kvBits` is enabled and `KVCacheSimple` entries are replaced by `QuantizedKVCache`.

The other major finding is model-specific: Tesseract's ParoQuant path loads `model_type = "qwen3_5"` and that model is hybrid. It does not use only append-only attention KV tensors. Most layers use `MambaCache` fixed-state recurrent storage, and only every `fullAttentionInterval` layer uses an attention KV cache. Phase 2b cannot assume every layer is a simple `[B, H, L, D]` block cache.

## Cache Lifecycle

1. `ModelContainer.prepare(input:)` prepares `LMInput`, but `ModelContainer.generate(input:parameters:)` does not accept an external cache array.
2. The public free function `MLXLMCommon.generate(input:cache:parameters:context:)` does accept `cache: [KVCache]?` and constructs a `TokenIterator` with it.
3. `TokenIterator` stores `var cache: [KVCache]` and calls `model.prepare(input, cache: cache, windowSize: ...)` for prompt prefill.
4. Each decode step calls `model(..., cache: cache, state: state)`, so attention or recurrent layers mutate the cache objects in-place.
5. After each step, `maybeQuantizeKVCache(cache:&cache, ...)` can replace `KVCacheSimple` entries with `QuantizedKVCache`.

Relevant sources:
- `Libraries/MLXLMCommon/ModelContainer.swift:142`
- `Libraries/MLXLMCommon/ModelContainer.swift:172`
- `Libraries/MLXLMCommon/Evaluate.swift:579`
- `Libraries/MLXLMCommon/Evaluate.swift:633`
- `Libraries/MLXLMCommon/Evaluate.swift:666`
- `Libraries/MLXLMCommon/Evaluate.swift:1082`

## Cache Types And Shapes

### Generic attention cache types

`KVCacheSimple`
- State arrays: exactly 2, `[keys, values]`
- Shapes:
  - `keys`: `[B, kvHeads, L, kHeadDim]`
  - `values`: `[B, kvHeads, L, vHeadDim]`
- Storage grows in `step = 256` token chunks, but `state` trims to logical `offset`
- Public deep copy exists via `copy()`

`RotatingKVCache`
- State arrays: exactly 2, `[keys, values]`
- Tensor rank matches `KVCacheSimple`
- `metaState` is required to interpret the logical window:
  - `[keep, maxCacheSize, step, offset, idx]`
- Physical tensor order may be rotated after wraparound

`QuantizedKVCache`
- State arrays: 4 or 6 arrays
- Layout:
  - keys: `(wq, scales, biases?)`
  - values: `(wq, scales, biases?)`
- Shapes for an original unquantized cache of `[B, kvHeads, L, D]`:
  - `wq`: `[B, kvHeads, L, D * bits / 32]`
  - `scales`: `[B, kvHeads, L, D / groupSize]`
  - `biases`: same as `scales` when present
- Types:
  - `wq`: `uint32`
  - `scales` / `biases`: floating-point
- `metaState`: `[step, offset, groupSize, bits]`

Relevant sources:
- `Libraries/MLXLMCommon/KVCache.swift:38`
- `Libraries/MLXLMCommon/KVCache.swift:328`
- `Libraries/MLXLMCommon/KVCache.swift:373`
- `Libraries/MLXLMCommon/KVCache.swift:407`
- `Libraries/MLXLMCommon/KVCache.swift:816`
- `Libraries/MLXLMCommon/KVCache.swift:899`

The quantized shape formula is inferred from the cache code above plus MLX's quantization packing behavior and test expectations in `mlx-swift` (`wq.shape == [32, 64]` and `scales.shape == [32, 4]` for a `[32, 256]` matrix at 8-bit, group size 64).

### Qwen3.5 / ParoQuant target model

The ParoQuant loader rewrites VLM configs to `model_type = "qwen3_5"`, so Tesseract's target model uses the fork's `Qwen35` implementation.

`Qwen35TextModel.newCache(parameters:)` returns one cache object per decoder layer:
- `MambaCache()` for linear attention layers
- `KVCacheSimple()` for full-attention layers

This is controlled by:
- `isLinear = (layerIdx + 1) % fullAttentionInterval != 0`
- default `fullAttentionInterval = 4`

So the cache is heterogeneous.

For `Qwen35` linear layers (`MambaCache`):
- `cache[0]` is the convolution state:
  - shape `[B, linearConvKernelDim - 1, convDim]`
  - `convDim = linearNumKeyHeads * linearKeyHeadDim * 2 + linearNumValueHeads * linearValueHeadDim`
- `cache[1]` is the recurrent gated-delta state:
  - shape `[B, linearNumValueHeads, linearValueHeadDim, linearKeyHeadDim]`

For `Qwen35` attention layers:
- standard attention cache shapes apply:
  - `keys`: `[B, kvHeads, L, headDim]`
  - `values`: `[B, kvHeads, L, headDim]`

Relevant sources:
- `Libraries/MLXLMCommon/ParoQuant/ParoQuantLoader.swift:339`
- `Libraries/MLXLLM/Models/Qwen35.swift:228`
- `Libraries/MLXLLM/Models/Qwen35.swift:241`
- `Libraries/MLXLLM/Models/Qwen35.swift:264`
- `Libraries/MLXLLM/Models/GatedDelta.swift:271`
- `Libraries/MLXLLM/Models/Qwen35.swift:452`
- `Libraries/MLXLLM/Models/Qwen35.swift:585`

## What Is Already Public

Already public on the pinned fork:
- `KVCache.state`
- `KVCache.metaState`
- `KVCache.copy()`
- `savePromptCache(url:cache:metadata:)`
- `loadPromptCache(url:)`
- `generate(input:cache:parameters:context:)`
- `generateTokens(...)` / `generateTokensTask(...)`
- `ChatSession.init(cache:)`
- `ChatSession.currentCache()`
- `ChatSession.saveCache(to:)`
- `ModelContainer.perform(nonSendable:_:)`

Relevant sources:
- `Libraries/MLXLMCommon/KVCache.swift:48`
- `Libraries/MLXLMCommon/KVCache.swift:75`
- `Libraries/MLXLMCommon/KVCache.swift:1244`
- `Libraries/MLXLMCommon/KVCache.swift:1309`
- `Libraries/MLXLMCommon/Evaluate.swift:1082`
- `Libraries/MLXLMCommon/Evaluate.swift:1164`
- `Libraries/MLXLMCommon/ChatSession.swift:183`
- `Libraries/MLXLMCommon/ChatSession.swift:511`
- `Libraries/MLXLMCommon/ChatSession.swift:528`
- `Libraries/MLXLMCommon/ModelContainer.swift:114`

## Access Modifiers That Still Matter

`TokenIterator.cache` is still internal, not public. `ModelContainer.generate(...)` is public but does not expose a `cache:` parameter and does not return the final cache array.

That means:
- cache injection is possible today through the public free functions
- cache extraction after generation is only fully reliable if the caller owns the exact final cache array
- the convenience `ModelContainer.generate(...)` API is not enough for Tesseract's shared-cache scheduler

Relevant sources:
- `Libraries/MLXLMCommon/Evaluate.swift:522`
- `Libraries/MLXLMCommon/ModelContainer.swift:172`

## Fork Requirement

No additional fork appears necessary for T14 or T15 if Tesseract is willing to use the pinned fork's existing public free functions and/or `ChatSession` cache APIs.

If Tesseract wants a clean `ModelContainer`-level API instead of working through `perform(...)` plus `MLXLMCommon.generate(...)`, then a small follow-up patch to the existing fork is still justified. That would be an additive convenience patch, not a new architectural fork.

## Proposed Minimal API

Smallest additive API that closes the remaining gap:

1. Add `ModelContainer.generate(input:cache:parameters:...)` as a thin wrapper over the existing public free function.
2. Add a public prefill-only helper that returns the populated cache array.
   There is already a private implementation in `WiredMemoryUtils.prefillOnly(...)`.
3. Add one way to obtain the final cache array after generation completes.
   Options:
   - `TokenIterator.currentCache` or `TokenIterator.finalizedCache()`
   - `generateTokensTask(...)` returning `(stream, task, finalCache)`
   - `ModelContainer.generateWithCache(...) -> (stream, cacheHandle)`

Without item 3, dynamic `kvBits` quantization is awkward.

## Risk Assessment

1. Hybrid-state model risk:
   Qwen3.5 PARO mixes `MambaCache` and attention KV caches. The Phase 2b design cannot assume every layer is an append-only `[B, H, L, D]` tensor.

2. Quantization handoff risk:
   Inference from the code: `TokenIterator` stores its own `[KVCache]` array and `maybeQuantizeKVCache(cache:&cache, ...)` replaces entries in that local array. External holders of the original array will not see those replacements automatically.

3. Sliding-window quantization gap:
   `RotatingKVCache.toQuantized()` is still unimplemented.

4. Complex cache restore gap:
   `loadPromptCache(...)` warns that `CacheList` reconstruction may not preserve sub-cache structure correctly.

5. Concurrency risk:
   `ChatSession.currentCache()` returns live cache objects. They are not safe to use concurrently with active generation.

Relevant sources:
- `Libraries/MLXLMCommon/KVCache.swift:1369`
- `Libraries/MLXLMCommon/KVCache.swift:1636`
- `Libraries/MLXLMCommon/ChatSession.swift:507`

## Recommendation For T15

Use the existing pinned fork; do not create another fork yet.

For the prototype:
- start with explicit cache injection via `MLXLMCommon.generateTokensTask(...)` or `generate(...)`
- keep `kvBits = nil` for the first extract/restore proof so the cache array remains stable
- if the prototype must run with `kvBits != nil`, add a very small helper that returns the finalized cache array after generation

This keeps T15 focused on proving cache restore semantics before redesigning Tesseract around a new cache API.
