# DFlash research for Qwen3.6-27B MLX 4-bit in Tesseract/Seract

Date: 2026-04-26
Status: research note for a later implementation plan
Primary target: `qwen3.6-27b` / `mlx-community/Qwen3.6-27B-4bit`
Goal: evaluate whether DFlash can deliver at least 2x decode speedup over the current ~20 tok/s Qwen3.6-27B local baseline, and what it would take to integrate into Swift `mlx-swift-lm`.

Note on project naming: the local repository is `/Users/owl/projects/tesseract`, but the user referred to the Seract server. This document uses Tesseract/Seract interchangeably for the current macOS Swift local-inference app/server.

---

## Executive summary

1. DFlash is now directly relevant to the current target. The upstream DFlash README lists a `Qwen3.6-27B (Preview)` draft: `z-lab/Qwen3.6-27B-DFlash`. Hugging Face API confirms it is public but auto-gated and has 1,730,213,120 BF16 parameters. Direct `config.json`/README fetches return 401 without accepting the gate.

2. The best available evidence says 2x is realistic but not guaranteed at long output lengths. The closest public Apple Silicon benchmark is `bstnxbt/dflash-mlx` on an Apple M5 Max for `mlx-community/Qwen3.5-27B-4bit`: 33.55 tok/s baseline -> 79.02 tok/s at 1K generated tokens (2.37x), 33.10 -> 70.21 at 2K (2.12x), then 1.77x at 4K and 1.34x at 8K. If our baseline is 20 tok/s, that maps to roughly 47 tok/s at 1K, 42 tok/s at 2K, 35 tok/s at 4K, 27 tok/s at 8K using the same speedup curve.

3. There is one nominal Swift port, but it is not useful yet. GitHub repository search finds `beshkenadze/dflash-mlx-swift`, described as a Swift port of `dflash-mlx`, but it currently contains only a README and one commit. There is no reusable Swift implementation. Integration must be a real port into vendored `mlx-swift-lm`.

4. Python MLX support exists in two forms:
   - Official `z-lab/dflash` includes `dflash/model_mlx.py`, tested on Apple Silicon and using stock `mlx-lm` plus a GatedDelta rollback path.
   - Community `bstnxbt/dflash-mlx` is more production-oriented for Apple Silicon: OpenAI-compatible server wrapper, recurrent rollback cache, context-only draft KV, long-context verify optimizations, and an M=16 int4 verify-qmm kernel.

5. Current local Swift code already has normal autoregressive speculative decoding (`SpeculativeTokenIterator`) but not DFlash. Existing `SpeculativeTokenIterator` requires trimmable KV caches and an autoregressive draft model. DFlash requires a different draft model API: target hidden-state taps + masked block diffusion + block verification + rollback for hybrid recurrent layers.

6. Qwen3.6-27B is a better DFlash target than the previous PARO analysis. The current model definition is standard MLX 4-bit (`mlx-community/Qwen3.6-27B-4bit`) and ships as `qwen3_5`, not PARO. It likely still uses Qwen3.5-family hybrid GatedDeltaNet + attention internals, so rollback remains important, but we no longer have PARO-specific layer-distribution mismatch.

7. Memory is acceptable if the draft is quantized or lazily loaded. The gated Qwen3.6-27B DFlash draft has 1.73B BF16 params: ~3.22 GiB raw BF16 weights. Int4 with ~6% metadata is ~0.85 GiB. The target is listed in the app as ~16 GB. On 48 GB Macs, BF16 draft is workable but tight under app headroom; int4 draft is the right default.

8. Business verdict: DFlash is worth pursuing because it directly addresses the key product bottleneck: local 27B quality at usable interactive speed. It should be an experimental feature flag first, with a go/no-go benchmark gate: `>=2.0x speedup at 1024 and 2048 generated tokens`, `>=1.5x at 4096`, no output divergence for greedy decode, no server streaming/tool-call regressions.

---

## Sources audited

Primary upstream:
- Paper: `DFlash: Block Diffusion for Flash Speculative Decoding`, arXiv `2602.06036`.
- Official repo: `https://github.com/z-lab/dflash`.
- Official README checked 2026-04-26: lists Qwen3.6-27B Preview, Qwen3.6-35B-A3B, Qwen3.5 models, Kimi, GPT-OSS, etc.
- Official Python MLX implementation: `dflash/model_mlx.py` from `z-lab/dflash`.
- HF collection/API: `z-lab/dflash` collection and model APIs.

DFlash draft checkpoints:
- `z-lab/Qwen3.6-27B-DFlash`: public, auto-gated, 1,730,213,120 BF16 params, last modified 2026-04-26, 3,447 downloads, 98 likes at check time. Direct raw config/README requires gate acceptance.
- `z-lab/Qwen3.6-35B-A3B-DFlash`: open config, 473,995,264 BF16 params, target layer ids `[1,10,19,28,37]`, block size 16, 8 draft layers, hidden size 2048.
- `z-lab/Qwen3.5-27B-DFlash`: open config, 1.73B-ish params, 5 draft layers, hidden size 5120, target layer ids `[1,16,31,46,61]`, block size 16.

Community Apple Silicon implementation:
- `https://github.com/bstnxbt/dflash-mlx`, 578 stars / 33 forks at check time, latest commit last week, version `0.1.4.1`.
- README benchmark table for Qwen3.5/3.6 models on Apple M5 Max, MLX 0.31.1.
- Code inspected: `dflash_mlx/model.py`, `runtime.py`, `draft_backend.py`, `recurrent_rollback_cache.py`, `pyproject.toml`.

Swift availability:
- GitHub repository search for `"DFlash" "Swift"` found `beshkenadze/dflash-mlx-swift` only.
- That repo has one commit, 0 stars, and only README content: “Swift port of dflash-mlx.” No implementation files.
- GitHub code search for `"DFlashDraftModel" Swift` requires sign-in for code results and shows no repository results.

Local code audited:
- `tesseract/Features/Models/ModelDefinition.swift`: current Qwen3.6-27B model definition points to `mlx-community/Qwen3.6-27B-4bit`, display size `~16 GB`.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`: has `SpeculativeTokenIterator`, but it is classic autoregressive speculative decoding and requires trimmable caches.
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/SpeculativeDecodingTests.swift`: current tests check normal speculative decoding equivalence and TriAttention cache prefill behavior.
- Existing prior doc: `docs/dflash-qwen35-paro-research-2026-04-16.md` for PARO-specific DFlash constraints.

---

## What DFlash does

DFlash is speculative decoding, but the draft is not autoregressive.

Classic speculative decoding:
- Small draft model proposes K tokens sequentially.
- Target model verifies the K-token block in one forward.
- Speedup depends on draft being much cheaper and acceptance being high.

DFlash:
- Uses a lightweight block-diffusion draft.
- Input block is `[last_confirmed_token, MASK, MASK, ..., MASK]` of length B.
- The draft denoises the masked block in parallel.
- Target verifies `[last_confirmed_token] + drafted_tokens` in one forward.
- The accepted prefix is emitted, plus the target token at the first mismatch.
- Correctness is lossless for greedy decoding because every emitted token is verified against the target path.

Official MLX loop, simplified:

```text
1. Prefill target prompt and capture selected target hidden states.
2. Sample first target token normally.
3. For each block:
   a. Draft input = last committed token + mask tokens.
   b. DFlash draft predicts B-1 proposed tokens in parallel.
   c. Target verifies last committed + proposed tokens in one forward.
   d. Compare draft tokens to target tokens left-to-right.
   e. Emit matching prefix plus target token at first mismatch.
   f. Roll back target/draft caches for rejected positions.
   g. Keep target hidden states for accepted positions as next draft context.
```

Draft architecture from official `model_mlx.py`:
- `DFlashDraftModel`
- shared/frozen target `embed_tokens`
- `fc` projection from concatenated target hidden states to draft hidden size
- `hidden_norm`
- 5 or 8 `DFlashDecoderLayer`s depending on checkpoint
- each layer has DFlash attention with:
  - Q from current noisy/masked block tokens
  - K/V from projected target hidden context plus current block tokens
  - RoPE offsets aligned to committed sequence offset
- shared/frozen target `lm_head` or tied embedding head

The important implication for Swift: this is not just another `LanguageModel`. The draft needs target hidden-state features and shared target embedding/head access.

---

## Current DFlash model support relevant to us

Official upstream README supported models now include:

| Target family | Draft status | Relevance |
|---|---:|---|
| Qwen3.6-27B Preview | `z-lab/Qwen3.6-27B-DFlash` | Primary target; gated config but public model API confirms checkpoint exists. |
| Qwen3.6-35B-A3B | `z-lab/Qwen3.6-35B-A3B-DFlash` | Good reference for Qwen3.6-family config and benchmarks. |
| Qwen3.5-27B | `z-lab/Qwen3.5-27B-DFlash` | Closest open dense 27B config; likely same draft size as Qwen3.6-27B. |
| Qwen3.5-35B-A3B | `z-lab/Qwen3.5-35B-A3B-DFlash` | MoE reference. |
| Qwen3.5 4B/9B | supported | Smaller integration/profiling targets. |
| Qwen3-Coder / GPT-OSS / Kimi | supported | Confirms method is broader than one model family. |

The Qwen3.6-27B draft is auto-gated. Before implementation, accept the Hugging Face gate for `z-lab/Qwen3.6-27B-DFlash` from the developer account used by the app/tooling. After gate acceptance, fetch `config.json` and confirm:
- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `block_size`
- `dflash_config.target_layer_ids`
- `dflash_config.mask_token_id`

Planning assumption: Qwen3.6-27B-DFlash is architecturally close to Qwen3.5-27B-DFlash because the HF safetensors parameter count exactly matches the 1.73B-class 27B dense draft.

---

## Existing MLX/Python implementations

### Official `z-lab/dflash` MLX path

Pros:
- Simple and close to the paper.
- Uses `mlx_lm` model loading and target layer hooks.
- Has `load`, `load_draft`, and `stream_generate` helpers.
- Has built-in GatedDelta rollback support in latest `model_mlx.py`.
- Official README says tested on Apple M5 Pro with Qwen3 and Qwen3.5.

Important code details:
- `_patch_model(model, layer_ids)` replaces target layers with hooks storing hidden states.
- `DFlashDraftModel.bind(model)` reuses target embeddings and `lm_head`.
- `DFlashDraftModel.make_cache()` supports optional draft sliding window via `RotatingKVCache`.
- For non-trimmable target caches, `_GDNStateCapture` captures GatedDelta internals and recomputes accepted recurrent state on rollback.

Limitations for Swift:
- Python monkey-patching does not translate to Swift; we need explicit hidden-state capture APIs.
- It assumes Python `mlx_lm` internals such as `GatedDeltaNet.__call__` can be patched.
- No Swift code.

### Community `bstnxbt/dflash-mlx`

This is the most valuable engineering reference for Apple Silicon.

Observed features:
- pip package `dflash-mlx`, version `0.1.4.1`.
- Requires `mlx>=0.25.0`, `mlx-lm>=0.31.0`.
- CLI `dflash`, server `dflash-serve`, benchmark `dflash-benchmark`.
- Auto draft resolution.
- OpenAI-compatible server wrapper around `mlx_lm.server` semantics.
- `ContextOnlyDraftKVCache` with sink/window retention for draft cache growth control.
- `RecurrentRollbackCache` for GatedDeltaNet state rollback.
- Custom Metal kernels for tape replay and verify-specialized int4 qmm.

Community M5 Max benchmark table excerpts:

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B-4bit | 1024 | 33.55 tok/s | 79.02 tok/s | 2.37x | 90.04% |
| Qwen3.5-27B-4bit | 2048 | 33.10 tok/s | 70.21 tok/s | 2.12x | 89.60% |
| Qwen3.5-27B-4bit | 4096 | 31.47 tok/s | 55.68 tok/s | 1.77x | 88.38% |
| Qwen3.5-27B-4bit | 8192 | 33.88 tok/s | 45.29 tok/s | 1.34x | 85.97% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.26 tok/s | 300.33 tok/s | 2.20x | 91.02% |
| Qwen3.6-35B-A3B-4bit | 2048 | 139.03 tok/s | 252.93 tok/s | 1.82x | 89.60% |
| Qwen3.6-35B-A3B-4bit | 4096 | 134.50 tok/s | 208.40 tok/s | 1.56x | 88.43% |
| Qwen3.6-35B-A3B-4bit | 8192 | 133.20 tok/s | 177.45 tok/s | 1.33x | 87.01% |

Interpretation:
- Acceptance can remain high (~86-91%), but speedup still falls at longer outputs because verify cost, cache/window behavior, and long-context attention dominate.
- The 27B dense path is the relevant caution: 2x at 1-2K, below 2x beyond 4K on that implementation/hardware.
- For Seract's current ~20 tok/s baseline, a 2.12-2.37x speedup means ~42-47 tok/s, which matches the product goal.

### Swift ports

`beshkenadze/dflash-mlx-swift`:
- public repo
- one commit
- 0 stars / 0 forks at check time
- only README: “Swift port of dflash-mlx.”
- no Swift source code yet

Conclusion: there is no available Swift MLX DFlash port to vendor. We should use it only as a watch item.

---

## Local integration surface in `mlx-swift-lm`

Current local facts:

1. Model catalog:
   - `qwen3.6-27b`
   - display: `Qwen3.6-27B (MLX 4bit)`
   - repo: `mlx-community/Qwen3.6-27B-4bit`
   - size: `~16 GB`
   - description says it ships as `qwen3_5`.

2. Existing `SpeculativeTokenIterator` is not enough:
   - It uses a normal `LanguageModel` draft.
   - It pre-fills both main and draft models from the same prompt.
   - It requires `canTrimPromptCache(mainCache)` and `canTrimPromptCache(draftCache)`.
   - It is a port of classic `speculative_generate_step()`, not block diffusion.

3. DFlash needs new Swift abstractions:
   - `DFlashDraftModel` type in `MLXLLM` or `MLXLMCommon`.
   - `DFlashConfiguration` on `GenerateParameters` or separate generation path.
   - hidden-state capture from target `Qwen35` forward.
   - shared target embedding and LM head access.
   - draft cache type with context-only KV and optional sink/window.
   - DFlash-specific iterator.
   - recurrent rollback for Qwen3.5/Qwen3.6 GatedDelta layers.

4. The port should not mutate general generation semantics:
   - Streaming outputs remain in token order, but emitted in bursts.
   - Tool-call parsing should see the same token stream as AR greedy decode.
   - Prefix-cache snapshots should store only final target cache state.
   - Draft cache is per request and should not enter the prefix cache.

---

## Proposed Swift architecture

### Public configuration

Add something like:

```swift
public struct DFlashConfiguration: Sendable, Equatable {
    public var enabled: Bool
    public var draftModelID: String?
    public var blockSize: Int        // default 16
    public var draftWindowSize: Int? // e.g. 1024/4096 or nil
    public var draftSinkSize: Int    // e.g. 64
    public var quantizeDraft: Bool   // default true for 27B
    public var minOutputTokensToEnable: Int // e.g. 64
}
```

Add to `GenerateParameters` later, or keep as app-level config passed around the agent/server path until the API stabilizes.

### Model loading

Need a loader for DFlash draft checkpoints:
- Download `z-lab/Qwen3.6-27B-DFlash` alongside the target or lazily when feature enabled.
- Parse `config.json` into `DFlashDraftConfiguration`.
- Load safetensors into `DFlashDraftModel`.
- Bind draft to target model embedding and LM head.
- For Qwen3.6-27B, default `quantizeDraft = true` to keep memory delta under ~1 GiB.

Open issue: the target is MLX 4-bit. The draft's own weights are BF16 upstream. Swift `mlx-swift-lm` must either:
- load draft BF16 first for fastest correctness proof, then add draft int4 quantization; or
- quantize at load time using existing QuantizedLinear/MLX quantization path.

Recommended: v0 proof-of-correctness with BF16 draft on a 64 GB machine; v1 product default with int4 draft.

### Target hidden-state capture

Avoid Python-style monkey patching. Add explicit target call support:

```swift
struct ForwardWithHiddenResult {
    let logits: MLXArray
    let hiddenStates: [Int: MLXArray]
}
```

For Qwen3.6-27B, capture exactly `target_layer_ids` from the draft config. The DFlash Python uses `hidden_states[layer_id + 1]` semantics in community code; verify whether Swift layer outputs are before or after residual/norm and align with Python.

Important: hidden capture must not force CPU sync. It should keep arrays on GPU/MLX and only eval the final set needed by the draft.

### Draft model

Swift classes mirroring Python:
- `DFlashDraftModel`
- `DFlashDecoderLayer`
- `DFlashAttention`
- `ContextOnlyDraftKVCache`

DFlash attention flow:
- embed block tokens using target embedding
- project concatenated target hidden with `fc` + RMSNorm
- Q from noisy/current block hidden
- K/V from projected target context and current block
- RoPE offsets use committed sequence offset and context length
- attention over cached context keys plus block/noise keys
- output target LM head over draft hidden positions excluding first token

### Iterator

Create a new iterator rather than overloading `SpeculativeTokenIterator`:

```swift
public struct DFlashTokenIterator: TokenIteratorProtocol { ... }
```

Core state:
- target model/cache/state
- draft model/cache
- last committed token
- current target hidden context
- sampler and logit processor
- pending token buffer
- accepted-token stats for telemetry

Algorithm:
1. Prefill target as today. Capture selected hidden states on the last prompt position.
2. Sample first target token normally; emit it.
3. Loop while `maxTokens` not reached:
   - Build `[lastToken] + maskTail` of block length B.
   - Run draft and sample proposed tokens.
   - Run target verify on `[lastToken] + proposed` while capturing hidden states.
   - Compare proposed vs target samples.
   - Emit accepted prefix plus correction token.
   - Roll back target cache/state for rejected suffix.
   - Trim/roll draft context cache if needed.
   - Keep hidden states only through accepted+correction positions.

### Rollback

This is the make-or-break engineering piece.

If target cache is fully trimmable:
- use `trimPromptCache(targetCache, trim)`.

For Qwen3.6/Qwen3.5 hybrid GatedDeltaNet:
- classic trim is insufficient because recurrent state has advanced through rejected tokens.
- We need either:
  1. snapshot + replay accepted positions, or
  2. tape-replay rollback like `bstnxbt/dflash-mlx`.

Recommended path:
- v0: snapshot recurrent cache state before verify, and replay accepted+correction tokens if any rejection. Easier to implement and verify in Swift.
- v1: implement tape-replay Metal kernel if snapshot/replay becomes a bottleneck.

The official latest MLX reference already implements a GDN rollback capture. The community implementation argues tape replay is faster and preserves long-generation acceptance better.

### Server and prefix cache

DFlash should be orthogonal to HTTP prefix cache:
- target cache is still authoritative
- final stored snapshot should match AR final target state
- draft cache is per-request only
- cache partitioning does not need to include DFlash unless target final state differs, which it must not under greedy verification

Cancellation rule:
- cancel only at a clean token boundary
- if cancel occurs mid-block, restore/trim target cache before returning final state
- safest v0: disable leaf snapshot store if cancellation happens inside DFlash verify block

---

## Math model

Let:
- `C_t` = cost of one target AR decode step.
- `C_d` = cost of one DFlash draft block pass.
- `B` = block size, usually 16.
- `tau` = average accepted draft tokens before first mismatch.
- Each successful block emits `tau + 1` tokens: accepted draft tokens plus target correction token.
- `r = C_d / C_t`.

Idealized speedup:

```text
per_token_cost_DFlash = (C_t + C_d) / (tau + 1)
speedup = C_t / per_token_cost_DFlash = (tau + 1) / (1 + r)
```

Required acceptance for 2x:

```text
tau_required = 2 * (1 + r) - 1
```

| Draft cost ratio `r` | `tau` needed for 2x |
|---:|---:|
| 0.20 | 1.4 |
| 0.35 | 1.7 |
| 0.50 | 2.0 |
| 0.75 | 2.5 |
| 1.00 | 3.0 |

This explains why DFlash can hit 2x even when real implementation overhead is large: it only needs to accept ~2-3 draft tokens on average if the draft block is not more expensive than one target step.

However, real Apple Silicon long-context speedup is lower than this formula because:
- target verify over 16 tokens is not exactly one token cost
- hidden-state capture has overhead
- recurrent rollback has overhead
- sampling/comparison can force synchronization if implemented poorly
- long context attention and draft KV growth increase cost over time
- M=16 quantized matmul may be slower than stock decode unless specialized

### Scenario projections from 20 tok/s baseline

Using measured community speedups from dense Qwen3.5-27B-4bit:

| Generated tokens | Measured 27B speedup analog | Projected from 20 tok/s |
|---:|---:|---:|
| 1024 | 2.37x | 47.4 tok/s |
| 2048 | 2.12x | 42.4 tok/s |
| 4096 | 1.77x | 35.4 tok/s |
| 8192 | 1.34x | 26.8 tok/s |

Using the simple formula:

| `tau` | `r` | Speedup | From 20 tok/s |
|---:|---:|---:|---:|
| 2.0 | 0.50 | 2.00x | 40 tok/s |
| 3.0 | 0.75 | 2.29x | 45.7 tok/s |
| 4.0 | 0.75 | 2.86x | 57.1 tok/s |
| 4.6 | 0.75 | 3.20x | 64.0 tok/s |
| 5.14 | 0.75 | 3.51x | 70.2 tok/s |
| 6.73 | 0.75 | 4.42x | 88.3 tok/s |

The formula is optimistic relative to measured 27B long-context Apple runs. For planning, use the measured analog table, not the ideal table.

---

## Memory model

Qwen3.6-27B-DFlash HF API:
- safetensors BF16 params: 1,730,213,120
- raw BF16 bytes: `1,730,213,120 * 2 = 3,460,426,240 bytes = ~3.22 GiB`
- int4 + 6% metadata estimate: `1,730,213,120 * 0.5 * 1.06 = ~0.85 GiB`

Target app catalog size:
- Qwen3.6-27B MLX 4-bit: `~16 GB`

Draft KV rough estimate if Qwen3.6-27B draft matches Qwen3.5-27B config:
- 5 layers
- 8 KV heads
- head dim 128
- per token per layer K+V elems = `2 * 8 * 128 = 2048`
- BF16 bytes per token per layer = 4096
- all 5 layers = 20 KiB/token
- 4096 generated tokens = ~80 MiB draft KV if context-only/windowed; more if unbounded over long conversations

With context-only sink/window draft cache from community implementation, bound it to e.g. sink 64 + window 1024/4096 to keep draft KV predictable.

Memory recommendation:
- v0 experimental on high-memory Mac: BF16 draft allowed.
- product default: int4 draft, DFlash disabled automatically if memory pressure/headroom insufficient.
- expose “DFlash acceleration” as an advanced/experimental toggle with estimated extra memory shown.

---

## Business analysis

### Product value

Qwen3.6-27B appears to be the quality sweet spot under 30B, but ~20 tok/s makes agentic use feel sluggish. DFlash directly targets decode, the bottleneck for:
- long reasoning responses
- code generation
- local agent loops
- tool-call planning
- server clients expecting streaming responsiveness

A sustained jump from ~20 tok/s to ~40-50 tok/s changes perceived UX from “acceptable but slow” to “interactive local model.” That is a meaningful differentiator for a macOS local AI inference app.

### Market positioning

DFlash lets Seract/Tesseract claim:
- local 27B-class model with speculative block diffusion acceleration
- Apple Silicon optimized local inference
- lossless speculative decoding, not approximate output shortcuts
- faster agent loops without server/cloud dependency

This aligns well with privacy/local-first positioning.

### Competitive context

Alternatives:
- smaller model: faster but quality loss
- MoE Qwen3.6-35B-A3B: much faster active params but different quality/behavior; current app already includes it
- classic speculative decoding: existing Swift support, but requires a good small AR draft and still drafts sequentially
- TriAttention/prefix caching: helps long context / memory, not pure decode speed in the same way
- Metal kernels/qmm optimization: complementary, lower-level, harder to generalize

DFlash is attractive because it is algorithmic and model-specific drafts already exist.

### Risks

High risks:
- No mature Swift implementation exists.
- Qwen3.6-27B draft is gated; config and license/card details need acceptance before shipping.
- Rollback for GatedDelta/Qwen3.5-family hybrid layers is complex.
- 2x speedup may not hold beyond 2K-4K generated tokens unless we also port long-context verify optimizations.
- Sampling equivalence under temperature/top-p is harder than greedy. First milestone should be greedy only.

Medium risks:
- BF16 draft memory on 48 GB Macs may be acceptable but not user-friendly.
- Draft int4 quantization may reduce acceptance or require custom kernels.
- Token bursts may expose latent streaming/parser assumptions.
- Gate/model availability can change.

Low risks:
- Prefix cache integration if DFlash final target cache is kept authoritative.
- UI gating/settings.

### Build vs. defer

Build now if:
- we can accept an experimental feature flag
- first target is greedy/temperature 0 or deterministic benchmark path
- we are okay with v0 not shipping as default
- success gate is 2x at 1-2K, not 8K

Defer if:
- the app cannot tolerate vendored `mlx-swift-lm` changes
- model gate/legal review blocks bundling/downloading draft
- the priority is short chat latency/TTFT rather than long decode

Recommendation: start a proof-of-concept, not full productization.

---

## Implementation scenarios

### Scenario A: Use Python `dflash-mlx` out of process

Description:
- Run `dflash-serve` as a subprocess for accelerated completions.
- Seract server proxies requests to it.

Pros:
- fastest proof of speedup
- uses production-ish Apple Silicon Python path
- can benchmark Qwen3.6-27B-DFlash before porting Swift

Cons:
- not native Swift
- packaging Python + MLX + custom kernels is heavy for macOS app
- separate model loading duplicates memory unless replacing current engine
- less control over app prefix cache/tool internals

Use as a research benchmark only.

### Scenario B: Port official `z-lab/dflash/model_mlx.py` to Swift

Pros:
- closest to official algorithm
- smaller code surface
- good correctness starting point

Cons:
- lacks community long-context optimizations
- Python monkey patching must be redesigned
- initial speed may miss 2x on 27B long outputs

Recommended v0 implementation path.

### Scenario C: Port community `bstnxbt/dflash-mlx` concepts to Swift

Pros:
- best Apple Silicon performance reference
- has recurrent rollback cache, context-only draft KV, int4 verify-qmm concepts
- benchmarked on 27B/35B 4-bit MLX models

Cons:
- larger and more complex
- custom Metal kernels need Swift/MLX integration work
- more maintenance surface

Recommended v1/v2 optimization path after official-style Swift DFlash works.

### Scenario D: Wait for real Swift port

Pros:
- lower engineering effort if external port matures

Cons:
- current Swift repo has no code
- no timeline
- may not match app’s vendored `mlx-swift-lm`

Not recommended as the primary plan.

---

## Go/no-go benchmark plan for next phase

Before writing the full implementation plan, benchmark the Python path on the target Mac:

1. Accept HF gate for `z-lab/Qwen3.6-27B-DFlash`.
2. Install `dflash-mlx` in a clean venv.
3. Run baseline `mlx_lm.stream_generate` on `mlx-community/Qwen3.6-27B-4bit`.
4. Run DFlash with `z-lab/Qwen3.6-27B-DFlash`.
5. Use fixed prompts:
   - math reasoning prompt
   - code generation prompt
   - agent/tool-planning style prompt
   - short chat prompt
6. Measure 512, 1024, 2048, 4096 generated-token caps.
7. Record:
   - prefill tok/s
   - decode tok/s
   - accepted length / acceptance ratio
   - peak memory
   - exact greedy output equivalence where possible
   - failure cases / fallback triggers

Go criteria:
- `>=2.0x` at 1024 and 2048 generated tokens on Qwen3.6-27B.
- `>=1.5x` at 4096 generated tokens.
- no unverified-token emission
- no crashes under cancellation/interrupt
- peak memory acceptable on 48 GB target Macs or draft int4 path viable

No-go / defer criteria:
- `<1.5x` at 1024 tokens
- recurrent rollback instability
- memory overhead >4 GiB with no quantization path
- draft gate/license prevents app distribution

---

## Recommended development plan outline

This is not the full implementation plan, but the research-backed sequence.

Phase 0: external Python benchmark
- Use `dflash-mlx` to validate real Qwen3.6-27B speed and memory on Apple Silicon.
- Accept/inspect gated Qwen3.6-27B DFlash config.

Phase 1: Swift correctness prototype
- Add DFlash config structs.
- Port `DFlashDraftModel`, attention, decoder layer.
- Add Qwen3.6/Qwen35 hidden-state capture API.
- Load BF16 draft.
- Implement greedy-only `DFlashTokenIterator`.
- Support fully trimmable cache first if possible, then GatedDelta snapshot/replay.
- Unit test against Python-equivalent toy model if feasible.

Phase 2: Qwen3.6-27B integration
- Bind draft to target embedding/head.
- Use `z-lab/Qwen3.6-27B-DFlash` after gate acceptance.
- Add app feature flag.
- Add benchmark harness.
- Prove byte-identical greedy outputs vs AR on small prompts.

Phase 3: performance work
- Draft int4 quantization.
- Context-only draft KV with sink/window.
- Reduce CPU sync in token comparison and acceptance logic.
- Recurrent rollback optimization.
- Consider M=16 verify qmm Metal kernel only if profiling shows stock MLX matmul bottleneck.

Phase 4: server/productization
- UI setting and model availability checks.
- Memory pressure guard.
- Prefix-cache consistency tests.
- Cancellation tests.
- Tool-call streaming tests.
- Telemetry: accepted length, fallback count, DFlash tok/s.

---

## Key open questions

1. What is the exact `config.json` for gated `z-lab/Qwen3.6-27B-DFlash`?
2. Does Qwen3.6-27B target hidden layer indexing match Qwen3.5-27B exactly?
3. Can Swift `mlx-swift-lm` expose hidden states without slowing normal generation?
4. Is BF16 draft fast enough to hit 2x, or is draft int4 required for speed as well as memory?
5. Does the current MLX Swift quantized matmul handle verify length M=16 efficiently, or do we need a specialized kernel?
6. How should sampling be handled beyond greedy? DFlash can be lossless with shared sampler semantics, but reproducibility with random sampling needs careful RNG state management.
7. Should DFlash be disabled automatically for short outputs, high temperature, or long context beyond a measured threshold?
8. Can we keep prefix-cache byte identity under DFlash when recurrent rollback is involved?

---

## Final recommendation

Proceed with DFlash research-to-prototype.

The evidence is strong enough to justify a POC:
- official Qwen3.6-27B DFlash draft exists
- Apple Silicon Python implementations exist
- 27B 4-bit measured speedups cross 2x at 1K-2K generated tokens
- current product bottleneck is exactly decode speed

But do not promise universal 2x yet. The likely product claim after measurement is:

“Up to ~2x faster long-form generation on Qwen3.6-27B, with best gains on 1K-2K reasoning/code outputs; experimental on very long 4K+ generations.”

The safest engineering path is:
1. benchmark Python `dflash-mlx` on the exact Qwen3.6-27B model,
2. port official DFlash to Swift for correctness,
3. then port selected community optimizations only if the naive Swift port misses the 2x gate.
