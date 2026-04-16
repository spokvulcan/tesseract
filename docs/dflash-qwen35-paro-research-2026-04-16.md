# DFlash Research for Qwen3.5 PARO in Tesseract

**Date:** 2026-04-16
**Status:** Research note for later implementation planning.
**Question:** Can [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) be integrated into the Tesseract server via the vendored `mlx-swift-lm` package, what does it do to memory, speed, and correctness for the current Qwen3.5 PARO models (`4B`, `9B`, `27B`), and can it coexist with PARO quantization, [TriAttention](https://arxiv.org/abs/2604.04921), and the existing HTTP prefix cache?

This document records the investigation performed in the codebase and on upstream sources. It is intentionally detailed so a later session can turn it into an implementation plan without repeating the research.

---

## Executive Summary

1. **DFlash is implementable for Tesseract, but not as a drop-in.**
   Upstream ships a Python reference (`dflash/model_mlx.py`) against Python `mlx_lm`. Tesseract uses Swift `mlx-swift-lm`. The real work is porting the draft model, draft cache, and a speculative token iterator into `Vendor/mlx-swift-lm`.

2. **The published DFlash drafts target dense Qwen3.5, not PARO.**
   The Hugging Face checkpoints (`z-lab/Qwen3.5-4B-DFlash`, `9B-DFlash`, `27B-DFlash`) condition on hidden states from specific target layer indices (`[1,8,15,22,29]` for 4B/9B, `[1,16,31,46,61]` for 27B). In PARO those indices hit a mix of full-attention and Mamba/SSM layers, so the hidden-state semantics differ from what the draft was trained on. Drop-in use with PARO targets is not expected to give the paper's 4.9x speedup; purpose-built PARO drafts are required.

3. **PARO's non-trimmable Mamba state is the hardest architectural constraint.**
   Speculative decoding requires the target to verify a block of proposed tokens, then roll back whatever tokens were rejected. In the hybrid PARO stack, attention KV trims but Mamba recurrent state does not. Verification forwards the Mamba state past the rejection point. The only viable workaround is per-block Mamba state snapshotting.

4. **DFlash is orthogonal to the HTTP prefix cache.**
   The draft has its own small, per-request KV cache and does not participate in cross-request snapshotting. The existing radix tree + `HybridCacheSnapshot` plumbing does not need changes for the draft. The target cache path stays as-is.

5. **DFlash and TriAttention are compatible in principle, with a measurable acceptance-rate cost.**
   TriAttention changes the attention compute but preserves layer output shape. DFlash only reads mid-layer hidden states from the target. The two can coexist; expect a ~5-15% acceptance-rate drop relative to full-attention DFlash until a draft is recalibrated against the TriAttention target.

6. **Decode speedup is the only direction DFlash helps.**
   Prefill is unchanged. TTFT is unchanged. The win is 2-5x on output tokens/sec, concentrated on long-output turns (math, code, agent tool-chains).

7. **Draft memory cost is 0.25 GB - 3.5 GB depending on target and draft dtype.**
   The published drafts are `bf16`. The Qwen3.5-27B-DFlash model card calls it "2B params," which matches the computed ~1.73B. Quantizing the draft to Int4 at load time is necessary to keep combined 27B + draft in the Tesseract memory envelope.

8. **Best first target is the agent path, not the server.**
   Same conclusion as the TriAttention research: the server is coupled to prefix cache invariants, and adding a speculative loop that touches `TokenIterator` and Mamba state belongs behind an agent-path feature flag first.

---

## 1. Sources Audited

### Paper and upstream project

- Paper: <https://arxiv.org/abs/2602.06036>, <https://arxiv.org/pdf/2602.06036>, <https://arxiv.org/html/2602.06036v1>
- Repo: <https://github.com/z-lab/dflash>
- Repo MLX implementation: <https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py>
- Qwen3.5-27B draft card: <https://huggingface.co/z-lab/Qwen3.5-27B-DFlash>

### DFlash draft configs (Hugging Face)

- `Qwen3.5-4B-DFlash/config.json`
- `Qwen3.5-9B-DFlash/config.json`
- `Qwen3.5-27B-DFlash/config.json`

### Local Tesseract docs and code

- TriAttention research (prior cross-reference): [`docs/triattention-qwen35-paro-research-2026-04-16.md`](triattention-qwen35-paro-research-2026-04-16.md)
- Prefix-cache plan: [`docs/marconi-hybrid-prefix-cache-implementation-plan.md`](marconi-hybrid-prefix-cache-implementation-plan.md)
- `tesseract/Features/Agent/LLMActor.swift`
- `tesseract/Features/Agent/AgentGeneration.swift`
- `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift`
- `tesseract/Features/Models/ModelDefinition.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`

---

## 2. What DFlash Actually Does

### 2.1 Core idea

Conventional speculative decoding uses a small autoregressive draft model to propose `K` tokens, then one target forward to verify. Draft cost scales linearly with `K` because it is still autoregressive. EAGLE-3 is the current state of the art and still runs the draft sequentially.

DFlash replaces the autoregressive draft with a **block diffusion** draft. The draft model is given a block of `B` positions where only the first is the last confirmed token and the remaining `B-1` are mask tokens. A single draft forward denoises all `B` positions in parallel. So draft cost is `~1 forward` per block regardless of `B`.

### 2.2 Draft model architecture

From the `DFlashDraftModel` class in `dflash/model_mlx.py`:

- Input: `(tokens, target_hidden, cache)`
- `target_hidden` is the concatenation along the feature dimension of the target's hidden states at a fixed set of layer indices (`target_layer_ids`).
- Forward flow per step:
  1. `h = embed_tokens(tokens)` (shared frozen embedding with the target)
  2. `h_ctx = RMSNorm(fc(target_hidden))` where `fc: (num_target_layers_extracted * target_hidden_size) -> draft_hidden_size`
  3. For each of the 5 `DFlashDecoderLayer`s:
     - `DFlashAttention(input, x_ctx, rope, cache)` where queries are derived from the current tokens only and keys/values are derived from the concatenation `[x_ctx, x]`
     - `MLP`
  4. `lm_head(RMSNorm(h))` (shared frozen head with the target)

The draft shares `embed_tokens` and `lm_head` with the target. Draft parameters are only the 5 decoder layers plus `fc` and two RMSNorms.

### 2.3 Block filling loop (MLX reference)

Excerpt from `dflash/model_mlx.py`:

```python
while n < max_tokens:
    bs = min(block_size, max_tokens - n + 1)
    block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
    draft_logits = draft(block, hidden, draft_cache)
    if (trim_n := draft_cache[0].offset - (prompt.size + n - 1)) > 0:
        trim_prompt_cache(draft_cache, trim_n)
    draft_tokens = sampler(draft_logits[:, 1 - bs:])

    verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
    logits = model(verify_input, target_cache)
    hidden = mx.concatenate(model._hidden_states, axis=-1)
    target_tokens = sampler(logits)

    d_list, t_list = draft_tokens[0].tolist(), target_tokens[0].tolist()
    accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list))
    new_tokens = d_list[:accepted] + [t_list[accepted]]
```

Key observations:

- Verification is **linear greedy**, not tree-based. Mismatch index triggers accepting the prefix plus the target's own sampled token at the mismatch position.
- The target forward is run once per block. The target is given `[last_confirmed] + draft_tokens` and emits logits for all positions plus hidden states collected via a tap (`model._hidden_states`).
- The draft cache is trimmed on each step if the draft's offset ran past where the confirmed sequence actually ended.

### 2.4 Published hyperparameters

Paper and configs together:

- Draft depth: `5` layers for base models; `8` for Qwen3 Coder.
- Block size: `16` for Qwen3 (4B/9B/27B); `10` for LLaMA-3.1.
- Target feature extraction: 5 layers uniformly selected between the second and third-to-last layer of the target.
- Training: around 800K samples from NVIDIA Nemotron Post-Training Dataset V2 and CodeAlpaca, AdamW with LR `6e-4`, 6 epochs.
- Loss: per-position exponentially decayed with `w_k = exp(-(k-1)/γ)` where `γ = 7` for `B=16`, `5` for `B=10`, `4` for `B=8`.

Published speedups (Table 1 and 3, greedy, B200 hardware):

- Qwen3-4B, GSM8K: `5.15x`, acceptance length `τ=6.53`.
- Qwen3-8B, GSM8K: `5.15x`, `τ=6.54`.
- Qwen3-8B, MATH-500, temp 0: `4.64x`, `τ=5.82`.
- Qwen3-8B, MATH-500, temp 1: `4.03x`, `τ=5.06`.
- Qwen3-Coder-30B at concurrency 1 (SGLang): `3.5x`.
- Overall average: `4.9x` versus autoregressive baseline; `2.4-2.5x` versus EAGLE-3.

Higher-concurrency SGLang numbers drop to `1.3-3.6x` at concurrency 8-32. Tesseract serves concurrency-1 workloads today, so the single-stream numbers are the right planning reference.

### 2.5 What the paper does **not** document

- No explicit tok/s absolute numbers on Apple Silicon. The model card only says "tested on an Apple M5 Pro with Qwen3 and Qwen3.5 models."
- No memory footprint in MB for the drafts. The 27B card calls its draft "2B params."
- No statement about int4/int8/FP8 draft compatibility. Released dtype is `bf16`.
- No discussion of sliding-window or sparse attention compatibility for the draft itself. Target-side sliding window is mentioned via `--speculative-dflash-draft-window-size` in vLLM but not characterized for accuracy impact.
- No KV-rollback algorithm. The reference just calls `trim_prompt_cache`, which relies on trimmable attention caches — it says nothing about non-trimmable caches like SSMs.

---

## 3. Current Tesseract and Model Facts

### 3.1 Current PARO models in Tesseract

From `tesseract/Features/Models/ModelDefinition.swift:90-125`:

| Model ID | Display name | Size description |
|---|---|---:|
| `qwen3.5-4b-paro` | `Qwen3.5-4B PARO` | `~3.5 GB` |
| `qwen3.5-9b-paro` | `Qwen3.5-9B PARO` | `~8 GB` |
| `qwen3.5-27b-paro` | `Qwen3.5-27B PARO` | `~19 GB` |

PARO is a hybrid: `full_attention_interval = 4`, meaning every fourth decoder layer is full attention and the other three are Mamba-style linear/state-space.

| Model | Hidden layers | Full-attn layers | Linear/SSM layers | Hidden size |
|---|---:|---:|---:|---:|
| `4B` | 32 | 8 | 24 | 2560 |
| `9B` | 32 | 8 | 24 | 4096 |
| `27B` | 64 | 16 | 48 | 5120 |

Hidden size numbers are derived from the matching DFlash draft configs, which set `fc` input dim to `num_target_layers_extracted * target_hidden_size`.

### 3.2 DFlash draft configs

Extracted verbatim from Hugging Face.

| Draft | Layers | Hidden | Heads | KV heads | Head dim | Intermediate | `block_size` | `target_layer_ids` | `num_target_layers` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `Qwen3.5-4B-DFlash` | 5 | 2560 | 32 | 8 | 128 | 9728 | 16 | `[1,8,15,22,29]` | 32 |
| `Qwen3.5-9B-DFlash` | 5 | 4096 | 32 | 8 | 128 | 12288 | 16 | `[1,8,15,22,29]` | 32 |
| `Qwen3.5-27B-DFlash` | 5 | 5120 | 32 | 8 | 128 | 17408 | 16 | `[1,16,31,46,61]` | 64 |

Shared fields across all three drafts:

- `architectures = ["DFlashDraftModel"]`
- `dtype = "bfloat16"`
- `rope_theta = 10_000_000`
- `max_position_embeddings = 262144`
- `vocab_size = 248320`
- `mask_token_id = 248070`

### 3.3 Current Tesseract generation defaults

From `tesseract/Features/Agent/AgentGeneration.swift:17-26`:

- `kvBits = 8`
- `kvGroupSize = 64`
- `prefillStepSize = 1024`

The target already uses Q8 KV for attention layers during decode. The DFlash verification path has to be Q8-compatible on the target and `bf16` on the draft.

### 3.4 Current text throughput baselines

From the comment at `tesseract/Features/Agent/LLMActor.swift:93-123` and `ParoQuant/ParoQuantLoader.swift:370-417`:

- `Qwen3.5-4B PARO` text prefill: `~1300 tok/s` (chunked).
- `Qwen3.5-4B PARO` VLM unchunked prefill: `~390 tok/s` on long prompts.

Decode tok/s is not directly documented in the repo but is the metric DFlash moves.

---

## 4. Mapping DFlash onto Tesseract's Stack

### 4.1 What has to change in `mlx-swift-lm`

1. **New draft model type.** A Swift port of `DFlashDraftModel` with `DFlashAttention`, `DFlashDecoderLayer`, shared embed/LM head binding, and `fc` projection from concatenated target hidden states.
2. **Hidden-state tap in `Qwen35Model`.** The target's `forward(...)` must expose hidden states from a configurable set of layer indices. Today `LLMModel.prepare(...)` and `TokenIterator.step(...)` do not expose intermediates.
3. **New KV cache type for the draft.** Smaller, per-request, `isTrimmable = true`, not serialized into `HybridCacheSnapshot`, not covered by the radix tree.
4. **New `SpeculativeTokenIterator`.** Owns draft + target caches, runs the block-fill / verify / accept / rollback loop, emits tokens to the existing generation stream.
5. **PARO Mamba rollback strategy.** Per-block snapshotting of all linear-layer caches before the target verification forward, restore on rejection.
6. **New cache partition key input.** DFlash on/off, block size, draft checkpoint fingerprint. Only matters if DFlash-aware snapshots are ever persisted; for the recommended design (draft caches are per-request only) the partition key does not need a new field.
7. **Sampler plumbing.** The draft and target must share a sampler so that greedy + temperature/top_p behavior is consistent.

### 4.2 What does **not** have to change

- The HTTP prefix cache and `HybridCacheSnapshot`. DFlash does not touch persisted state; only the target cache is persisted.
- The radix tree, stable-prefix detector, and eviction logic.
- `AgentEngine`, `ToolRegistry`, `ContextManager`. DFlash is below the streaming boundary.
- The overall `CompletionHandler` -> `MessageConverter` -> `AgentEngine.generateServerTextCompletion` path. It observes the same token stream.

---

## 5. Memory Math

All numbers below use `bf16` = 2 bytes/elem, `int4` = 0.5 bytes/elem plus ~6% quant metadata.

### 5.1 Draft parameter count

Per decoder layer, parameter count is roughly:

```text
attn_params = hidden^2               (Q)
            + hidden * n_kv_heads * head_dim   (K)
            + hidden * n_kv_heads * head_dim   (V)
            + hidden^2               (O)
mlp_params  = 3 * hidden * intermediate          (SwiGLU)
norms       = 2 * hidden
per_layer   = attn_params + mlp_params + norms
```

Plus `fc: (n_target_layers_extracted * hidden) * hidden` and two RMSNorms (`hidden` each).

Applied to the three Qwen3.5 drafts:

| Draft | Per-layer params | 5 layers | `fc` | Total params | `bf16` bytes | `int4` bytes |
|---|---:|---:|---:|---:|---:|---:|
| 4B | ~93.0 M | ~465 M | 32.8 M | **~498 M** | **~0.99 GiB** | **~0.26 GiB** |
| 9B | ~193.1 M | ~965 M | 83.9 M | **~1.05 B** | **~2.10 GiB** | **~0.56 GiB** |
| 27B | ~319.9 M | ~1.60 B | 131 M | **~1.73 B** | **~3.46 GiB** | **~0.92 GiB** |

The 27B number matches the "2B params" claim on the Hugging Face model card. The 4B and 9B numbers are planning estimates with about a 10% uncertainty from final norms/bias layouts.

### 5.2 Draft KV cache per token

All three drafts share `num_kv_heads = 8`, `head_dim = 128`, `num_hidden_layers = 5`.

Per layer per token: `2 * 8 * 128 = 2048` elements. At `bf16`: 4096 bytes.

Across 5 layers: `5 * 4096 = 20480` bytes/token ≈ `20 KiB/token`.

Under Tesseract's `kvBits = 8` Q8 pattern (if the draft cache follows the target's Q8 plan): `2048 + 2*64 = ~2176` bytes per layer per token, total `~10.6 KiB/token`.

Growth table:

| Tokens | bf16 draft KV | Q8 draft KV |
|---:|---:|---:|
| 4,000 | 78.1 MiB | 41.4 MiB |
| 8,192 | 160.0 MiB | 84.8 MiB |
| 16,384 | 320.0 MiB | 169.7 MiB |
| 32,768 | 640.0 MiB | 339.4 MiB |
| 65,536 | 1.25 GiB | 678.9 MiB |
| 120,000 | 2.29 GiB | 1.21 GiB |

Because the draft cache is per-request and resets on new turns, typical agent decode lengths (<= 4K generated tokens) keep draft KV under 100 MiB.

### 5.3 Total memory delta vs today

Combined footprint per live session, holding model weights resident.

| Target | Target weights (live) | + DFlash draft `bf16` | + DFlash draft `int4` |
|---|---:|---:|---:|
| 4B | ~2.50 GiB | +0.99 GiB (= 3.49 GiB) | +0.26 GiB (= 2.76 GiB) |
| 9B | ~5.60 GiB | +2.10 GiB (= 7.70 GiB) | +0.56 GiB (= 6.16 GiB) |
| 27B | ~16.90 GiB | +3.46 GiB (= 20.36 GiB) | +0.92 GiB (= 17.82 GiB) |

`27B + bf16 draft` at 20.36 GiB of weights alone eats most of a 48 GiB Mac when stacked with the Tesseract 20 GiB headroom rule (`LLMActor.swift:101-110`). Running DFlash on 27B without quantizing the draft is not a safe default.

Decision implication: the first Tesseract implementation should load drafts in `int4` using the same Quant path as the target weights. `4B+int4 draft` is a `0.26 GiB` addition; `27B+int4 draft` is `0.92 GiB`.

### 5.4 Memory during verification (target side)

Each block the target gets a `17`-token input (`last_confirmed + 16 drafts`). Peak per-step activation scratch is proportional to `block_size`, not to the accepted count. Compared with `block_size = 1` autoregressive decode:

- Target attention compute scales as `block_size * prefix_len` (linear in block size for decode step).
- Target activation peak scales as `block_size * hidden_size`.

On 4B PARO at 120K context, one 16-token verification step increases decode-time activation peak by roughly `15 * hidden * 2 bytes ≈ 15 * 2560 * 2 ≈ 77 KiB` per layer. Negligible.

The real memory sensitivity is the **Mamba rollback snapshot** (Section 6.2), not scratch size.

---

## 6. PARO-Specific Correctness Issues

### 6.1 Layer-index alignment

The published drafts extract features from target layers `[1,8,15,22,29]` (4B/9B) and `[1,16,31,46,61]` (27B). In dense Qwen3.5 those are plain decoder layers. In PARO, with `full_attention_interval = 4`:

- PARO 4B full-attention layers (0-indexed, pattern-dependent): every 4th layer. If the pattern is `[3,7,11,15,19,23,27,31]`, then of the draft's tap points `[1,8,15,22,29]`, only **layer 15** is a full-attention layer. Layers 1, 8, 22, 29 are Mamba/linear.
- PARO 27B with 64 layers: full-attention at `[3,7,...,63]`. Of `[1,16,31,46,61]`, only **layer 31** is a full-attention layer.

Whether this is a blocker depends on a semantics question: is the hidden state at the **output** of a PARO Mamba layer a reasonable proxy for the hidden state at the output of a dense attention layer? Both are `[B, L, hidden]` tensors, shape-compatible. But the DFlash draft was trained to predict the next token given a specific distribution of hidden-state patterns produced by a dense transformer. Mamba outputs have a measurably different spectral signature and causal-mixing pattern.

Expected outcome if the published draft is used directly against a PARO target: correctness stays (verification guarantees losslessness), but acceptance length `τ` drops sharply - possibly from ~6.5 to ~2-3, meaning the "speedup" could shrink to 1.5-2x or less and the extra draft FLOPs could make the system net-slower.

The principled fix is: **train a new DFlash draft per PARO target.** Z-lab's README says the training recipe will be open-sourced "soon." Without that, the options are:
- Sweep `target_layer_ids` against PARO to find layer indices where the draft still accepts well.
- Fine-tune the published draft briefly against PARO hidden states using the same loss.
- Ship DFlash only for dense Qwen3.5 variants if Tesseract ever adds them.

### 6.2 Non-trimmable Mamba state under rejection

This is the hardest correctness issue.

A speculative verify step runs the target on `[last_confirmed] + 16 drafts`. The target's state (KV cache and Mamba state) advances by 17 token positions. If only `k < 16` tokens are accepted, the state has to be rewound to "after `last_confirmed + accepted[:k] + target_token[k]`" = `k + 1` positions, not 17.

From the local audit in `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` and `tesseract/Features/Agent/LLMActor.swift:589-607`:

- `KVCacheSimple` (full attention) supports trim.
- `MambaCache` does not. `BaseKVCache.isTrimmable` is `false` by default and `trimPromptCache(...)` bails if **any** layer is untrimmable.

Therefore:

- For the dense Qwen3.5 variants (not currently shipped by Tesseract), DFlash rollback is straightforward: `trim_prompt_cache(target_cache, 17 - (k + 1))`.
- For PARO, trimming the full cache is not possible. The Mamba state has already been advanced through all 17 tokens by the verification forward and cannot be walked back.

Viable strategies for Tesseract:

- **Strategy A - Per-block Mamba snapshot.**
  Before each target verification forward, deep-copy the Mamba state for every linear layer. After classification:
  - If all drafts accepted: discard the snapshot.
  - If `k < 16` accepted: restore the snapshot, then run the target on the `k + 1` correct tokens to re-advance Mamba to the right offset. Attention KV is simply trimmed.

  Memory cost per block, using the 4B/9B/27B geometry from the TriAttention research (Section 7.2 of that doc):
  - 4B/9B: `24 * 1.047 MiB = 25.1 MiB` per snapshot.
  - 27B: `48 * 1.559 MiB = 74.8 MiB` per snapshot.

  Runtime cost per rejection: one short target forward over `k + 1` tokens (not 17). For average `τ = 6.5` this is `~7` tokens, roughly `7x` decode cost instead of 16x. The rollback is still net-positive because the accepted tokens were drafted cheaply.

  This is the **recommended default**.

- **Strategy B - Restrict block size.**
  With `block_size = 1`, DFlash degrades to nothing. Not useful.

- **Strategy C - Accept all draft tokens unconditionally.**
  Lossy, no longer lossless spec decoding. Rejected.

- **Strategy D - Skip Mamba layers during verification.**
  Not possible in a sequential hybrid stack.

Strategy A's memory cost is bounded: one snapshot in flight per draft block. Even at 27B it is 74.8 MiB, well inside the Tesseract prefix-cache budget (Section 7.4).

### 6.3 Impact on prefix cache store

Prefix-cache snapshotting happens once per request, after generation completes. It captures the **final** target cache state. DFlash does not change that - the final cache after all accepted tokens have advanced through is identical to what a non-speculative decode would have produced. So `.leaf` and `.system` snapshots stay correct with DFlash enabled.

The only nuance: if a request is cancelled mid-block, the target cache could be in an intermediate state after a verification forward with no subsequent rollback. The leaf store in `LLMActor.swift:532-595` already tolerates "no final cache" via `httpPrefixCacheHasReusableState(finalCache)` checks; the DFlash speculative iterator just needs to deliver a state-consistent cache on cancellation (either rolled back or advanced to a clean token boundary).

### 6.4 Impact on cache partition key

If draft caches are never persisted, no new partition fields are needed. Keep this as the default.

If at some point draft-side snapshotting is added (unlikely, low value - draft cache is cheap to rebuild), the partition key would need:
- `dflashEnabled`
- draft checkpoint fingerprint
- `block_size`
- sampler version (draft + target sampler must match)

---

## 7. Interaction With TriAttention

TriAttention (see sister research doc) compresses the target attention KV cache by retaining the top-`B` keys per the paper's pre-RoPE Q/K statistics. It does not change the target's layer output shape. DFlash reads layer outputs, so from a shape standpoint the two are orthogonal.

Correctness and acceptance-rate implications:

1. TriAttention is approximately lossless for reasoning accuracy at `B >= 12000`, but not bitwise. The hidden state at layer `i` produced by a TriAttention target is slightly different from the hidden state produced by a full-attention target at the same position.
2. The DFlash draft was trained against a full-attention target. Using it against a TriAttention target is an **out-of-distribution** inference path.
3. The target is still authoritative - verification is lossless. So the combined system remains output-correct.
4. Acceptance rate `τ` will drop. Expected magnitude: 5-15% reduction in `τ`, translating to 10-20% reduction in DFlash speedup. Still net-positive.

Rollback compatibility:
- TriAttention sparsifies attention state per Section 4.4 of the TriAttention research. It should be treated as **non-trimmable**, like Mamba.
- With DFlash, that means rollback requires snapshotting TriAttention state too (same Strategy A as for Mamba).
- Memory cost per TriAttention snapshot at `B=12000` on PARO 27B: ~804 MiB of persistent cache equivalent, but the Mamba-style deep copy is only for the sparse retained state - closer to 200-300 MiB per rollback snapshot. Measure after the port.

Bottom line: DFlash + TriAttention + PARO is implementable but stacks three non-trimmable compromises. Recommended rollout order:

1. DFlash on a dense Qwen3.5 variant (if Tesseract ever ships one) - correctness is simple.
2. DFlash on PARO with Strategy A Mamba snapshots - the big unlock for current models.
3. TriAttention without DFlash - memory win.
4. TriAttention + DFlash together - last, after both are proven independently.

---

## 8. Interaction With the HTTP Prefix Cache

### 8.1 Cache hit path

On a prefix-cache hit, the target cache is restored from `HybridCacheSnapshot`. The target is warm at offset `N`. The draft cache is not restored - it starts empty.

First decode iteration:
- Target runs one forward over the current tokens (typically 1 token beyond the restored prefix) to produce the next token and layer hidden states.
- DFlash now has fresh `x_ctx` and enters its block-fill loop.

No correctness issue. The draft does not need prior context: `x_ctx` is always supplied from the target's most recent forward. Draft cache is tiny and rebuilds in `O(first_block_size)` draft forwards.

### 8.2 Cache miss path

Target prefill runs as today, chunked. During prefill the target hidden states are computed but typically discarded. For DFlash the draft does not need prefill hidden states - it only needs the hidden state after the last token of the prompt, which is available at the end of prefill.

Therefore prefill stays unchanged. TTFT is unaffected by DFlash.

### 8.3 Cache store path

Post-generation leaf store works as today. The target cache after all accepted tokens is what gets captured, identical to what a non-speculative run would produce. DFlash does not interfere.

The rollback snapshots taken during generation (Strategy A, Section 6.2) are per-block and transient. They are discarded at request end. They do not enter the radix tree.

### 8.4 Concurrency and `InferenceArbiter`

Per the project layout: HTTP arbitration uses `InferenceArbiter` for single-in-flight. DFlash does not change that. The speculative iterator is a replacement for the normal decode loop inside one in-flight request.

---

## 9. Interaction With PARO Quantization

PARO is 4-bit weight quantization for the target. The draft model is `bf16` in the published checkpoints. Three points matter:

1. **Draft dtype vs target dtype.** The DFlash reference code does not require matching dtypes between draft and target. `lm_head` and `embed_tokens` are shared frozen weights - they live in the target at 4-bit PARO. The draft consumes `embed_tokens(tokens)` (output in the activation dtype, bf16 or fp16) and calls `lm_head(h)` at the end (same story). This works.

2. **Draft weight quantization in Tesseract.** To stay in the memory envelope on 27B, the draft itself must be quantized. MLX-Swift-LM's `Quant` path already supports Int4 weight quantization for dense transformer layers. Applying it to the DFlash draft's five layers should be straightforward and is expected to cost ~1-3% acceptance rate.

3. **Shared embed/head binding.** In the reference, `draft.bind(model)` pulls `embed_tokens` and `lm_head` out of the target. On PARO those weights are in the 4-bit ParoQuant layout. The Swift port has to bind to ParoQuant-layout weights rather than a plain `nn.Embedding`. This is the main ParoQuant-specific porting task, and lines up with the existing ParoQuantLoader at `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift`.

---

## 10. Performance Model for Tesseract

### 10.1 Per-step cost breakdown

Without spec decode, one decode step = one target forward (cost `C_target`) producing one token.

With DFlash at block size `B`:

- One draft forward (cost `C_draft ≈ (L_draft / L_target) * C_target`, roughly `5 / target_layers`).
- One target forward over `B + 1` tokens (cost `≈ (1 + (B / L_context)) * C_target` for small `B` relative to `L_context`, i.e. ~`C_target` for 16 vs 120K).
- Produces on average `τ + 1` tokens (accepted drafts + one bonus target token).

Per-token amortized cost:

```text
T_per_token = (C_draft + C_target) / (τ + 1)
Speedup     = C_target / T_per_token = (τ + 1) / (1 + C_draft/C_target)
```

For `τ = 6.5`, `C_draft/C_target ≈ 0.16` (4B, 5 draft layers / 32 target layers, plus the fc overhead but both drafts run on smaller hidden):

```text
Speedup ≈ 7.5 / 1.16 ≈ 6.5x
```

Paper reports 4.9x average. The gap is mostly the x_ctx concatenation making draft attention 2x as expensive as a naive layer count suggests, plus kernel-launch overhead for the small draft forward.

### 10.2 Tesseract projections

These are planning estimates. Baseline decode tok/s on Apple Silicon M-series is taken from typical Qwen 4B class numbers on M3 Max / M4 Pro.

| Target | Baseline decode | With DFlash (dense target) | With DFlash + PARO rollback | With DFlash + PARO + TriAttention |
|---|---:|---:|---:|---:|
| 4B | ~45 tok/s | ~200-220 tok/s (4.5-5x) | ~120-160 tok/s (2.7-3.5x) | ~100-140 tok/s (2.2-3.1x) |
| 9B | ~30 tok/s | ~130-150 tok/s | ~80-110 tok/s | ~70-95 tok/s |
| 27B | ~15 tok/s | ~60-75 tok/s | ~40-55 tok/s | ~35-50 tok/s |

"PARO rollback" row discounts the projected speedup by 25-35% to cover:
- Acceptance-rate drop from OOD draft (~20-30%).
- Mamba snapshot deep-copy per block (~5-10%).

Prefill is unchanged in all columns.

### 10.3 Where DFlash helps most

- Long agent turns with a lot of tool-call reasoning (many tokens before `<tool_call>`).
- Math and code benchmarks with repetitive structure (paper reports highest acceptance lengths here: `τ = 7.87` on Qwen3-8B Math, temp 0).
- Server completions with temperature 0 greedy decoding - this is where τ is highest.

### 10.4 Where DFlash does not help

- TTFT and first-token latency - DFlash only speeds decoding.
- High-temperature, high-entropy generations - τ drops sharply because draft and target sample different tokens.
- Very short outputs (<32 tokens) - overhead of draft KV setup + first block outweighs savings.
- Batch > 1 serving - single-stream Tesseract is the right scale; higher batch sizes diminish relative speedup (paper shows 1.3-3.6x at concurrency 8-32).

---

## 11. Feasibility for the Tesseract Agent

### Verdict

**Yes,** with one caveat: the published drafts target dense Qwen3.5 and will underperform on PARO. A custom draft is needed to hit paper numbers.

### Why agent first

- Same reasoning as the TriAttention research. The agent path is less coupled to the HTTP prefix cache contract and easier to gate behind a flag.
- `AgentEngine` already has a compaction + streaming architecture that tolerates arbitrary token-production cadence.
- The double-loop tool-calling pattern (`Core/AgentLoop.swift`) benefits most from decode speedup because tool turns are typically several hundred tokens each.

### Scope for a first implementation

1. Port `DFlashDraftModel`, `DFlashAttention`, `DFlashDecoderLayer` to Swift.
2. Add hidden-state tap to `MLXLLM.Qwen35Model`.
3. Add a speculative `TokenIterator` wrapper with Strategy A Mamba snapshotting.
4. Int4 quantize the draft at load time.
5. Hardcode `block_size = 16` for the first pass.
6. Support only one model (start with `4B PARO`).
7. Skip TriAttention interaction in v1.

---

## 12. Feasibility for the Tesseract Server

### Verdict

**Yes, behind a feature flag, after the agent path is proven.**

### Why server second

- Server has more correctness invariants (prefix cache, partition keys, normalization-time trim guards).
- Server runs at `127.0.0.1:8321` with OpenAI-compatible semantics - any acceptance-rate regression is visible to external clients.
- The `InferenceArbiter` single-in-flight model is where DFlash can be safely inserted, but the existing `CompletionHandler -> MessageConverter -> AgentEngine.generateServerTextCompletion` chain needs to pass a DFlash-enable signal through.

### Order of changes

1. Feature flag plumbed through `DependencyContainer` and `SettingsManager`.
2. `LLMActor.makeHTTPPrefixCacheGeneration` accepts a `DFlashConfiguration` alongside the current `TriAttentionConfiguration`.
3. The speculative iterator observes cancel on the same `Task.isCancelled` path as today.
4. Benchmarks: add a `--dflash-e2e` runner analogous to `--prefix-cache-e2e` that does Request A (cold, no DFlash) -> Request B (warm, DFlash) -> assert byte-identical greedy output. Lossless guarantee is the verification gate.

### What absolutely must be preserved

- Streaming token equivalence (sampling params honored, same tokens emitted).
- Prefix-cache byte identity of the final leaf snapshot (DFlash run vs non-DFlash run produce identical `storedTokens.count` and final cache contents). This is critical because the radix tree must not fork.
- Cancellation produces a state-consistent cache (no mid-verification partial advance).

---

## 13. Pros

1. **Large decode speedup (2-5x realistic on PARO).** This is the biggest per-request win on the table for LLM decode.
2. **Lossless.** Verification guarantees identical output distribution to non-speculative decode (given matching samplers).
3. **Orthogonal to prefix cache.** No snapshot format changes, no partition-key changes, no radix-tree changes required.
4. **Small draft footprint at Int4.** 4B: +0.26 GiB; 9B: +0.56 GiB; 27B: +0.92 GiB.
5. **Reuses target embedding and LM head.** No duplicated vocab/embed matrices.
6. **Already has a Python MLX reference** with `mx.fast.scaled_dot_product_attention`, so the algorithm maps cleanly onto Metal.
7. **Stacks with TriAttention.** Both give independent wins; combined acceptance-rate penalty is bounded.
8. **Doesn't require retraining the target.**
9. **Upstream ships per-model drafts already trained for Qwen3.5 sizes and LLaMA-3.1.**
10. **Block size is a simple knob.** Paper tests `8`, `10`, `16`; drops to `8` trade acceptance for latency on memory-bound hardware.

## 14. Cons

1. **Not a drop-in for `mlx-swift-lm`.** Requires porting the draft class, adding the hidden-state tap, and writing a speculative iterator.
2. **Published drafts target dense Qwen3.5, not PARO.** Using them directly against PARO targets is expected to cut acceptance length roughly in half. A custom PARO draft is needed for paper-level performance.
3. **PARO's Mamba state is non-trimmable.** Requires per-block snapshotting (Strategy A), adding 25-75 MiB of transient memory per block and ~7-token re-advance cost per rejection.
4. **Needs target hidden-state tap.** Modifies vendor `Qwen35Model` forward path.
5. **Draft dtype mismatch with PARO target (bf16 vs 4-bit).** Solvable by Int4 quantizing the draft at load, but needs ParoQuant-side binding for shared embed/head.
6. **Training recipe not yet open-sourced.** Z-lab promises "soon." Until it lands, custom PARO drafts are blocked.
7. **No Apple Silicon tok/s numbers published.** All projections are extrapolations from NVIDIA B200 results.
8. **Acceptance rate drops at higher temperatures.** Tesseract defaults to temperature 0.7 in some agent flows; DFlash wins are biggest at temperature 0.
9. **Batch > 1 benefits are smaller.** Not a current issue for Tesseract's concurrency-1 server, but caps future horizontal scaling.
10. **Stacks non-trimmable state with TriAttention.** Both Mamba-like rollback semantics and TriAttention sparse state require per-block snapshots if DFlash is on.
11. **Short turns see negligible benefit.** Outputs < 32 tokens do not amortize draft setup.
12. **Kernel-launch overhead matters on M-series.** MLX small-batch kernels have non-trivial dispatch latency; block size 16 already mitigates this but smaller blocks lose to dispatch overhead faster than on CUDA.
13. **Cancellation semantics get harder.** The speculative loop must deliver a clean cache boundary on mid-block cancel, or the leaf store has to detect and drop inconsistent finals.
14. **VLM path not covered.** Like TriAttention, a text-only DFlash port does not cover the `MLXVLM.Qwen35` path. Image turns would be DFlash-disabled until the VLM path is ported.
15. **Greedy-only linear verification, not tree-based.** EAGLE-3-style tree verification is more token-accepting at high temperature; DFlash wins on parallel drafting and loses a little on verification shape. Net positive per the paper, but the tradeoff exists.

---

## 15. Open Questions for the Implementation Plan

1. **Does Z-lab's training recipe land before Tesseract needs paper-level speedup?** If not, we need a short fine-tune script in-repo to adapt the published drafts to PARO.
2. **Which PARO target layer indices maximize acceptance?** The published `target_layer_ids` are calibrated on dense. A sweep against PARO (all 8 full-attention layers for 4B/9B, all 16 for 27B) would likely give better defaults.
3. **Should draft KV be Q8 or bf16?** Target is Q8; draft is tiny. Keeping draft at bf16 simplifies the port; Q8 saves ~50% of the draft KV bytes at the cost of another quant path.
4. **What sampler contract?** The draft and target must sample identically for verification to be well-defined under temperature/top_p. Needs a shared sampler seed.
5. **Per-block Mamba snapshot: deep copy or copy-on-write?** MLX arrays are immutable; `deep copy` in Swift means new allocations. Measuring the real cost is required before committing to Strategy A as the default.
6. **Cancellation boundary.** Do we cancel at the nearest accepted-token boundary (drop the in-flight block) or at the end of the current block?
7. **Does DFlash interact safely with the existing `ChatSession`-style completion signaling?** Specifically `emitParserEvents(parser.finalize(), ...)` and the tool-call reconstruction in `AgentEngine` - both assume tokens arrive in order. DFlash emits tokens in bursts of `τ + 1`, which is still in-order but bursty.
8. **Benchmark harness.** Need a DFlash-aware version of `--benchmark` and an equivalent of `--prefix-cache-e2e` that asserts byte-identical greedy output with and without DFlash on the same prompts.

---

## 16. Recommended Direction for the Future Implementation Plan

Order:

1. **Agent-only, text-only, 4B PARO, Int4 draft, Mamba Strategy A, `block_size = 16`, no TriAttention.**
   Smallest possible first landing. Proves correctness (byte-identical greedy output) and measures real tok/s on Apple Silicon.

2. **Sweep `target_layer_ids` against 4B PARO.**
   Find PARO-native tap points that preserve acceptance length.

3. **Add fine-tune script for custom PARO drafts** once upstream releases the training recipe, or implement our own based on the paper's loss.

4. **Server feature flag.**
   Plumb `DFlashConfiguration` through `LLMActor.makeHTTPPrefixCacheGeneration`, gated off by default.

5. **27B support.**
   Validate memory envelope and Mamba snapshot cost at 64 layers / 48 linear layers.

6. **TriAttention coexistence.**
   After both DFlash and TriAttention are proven independently, measure combined acceptance-rate loss. If acceptable, keep both available; otherwise expose them as mutually exclusive runtime settings.

7. **VLM parity.**
   Port the hidden-state tap and the speculative iterator into `MLXVLM.Qwen35` so image turns are not DFlash-disabled.

This ordering mirrors the TriAttention research's agent-first / server-flag-second / hybrid-last structure and minimizes cross-cutting changes to the prefix cache plumbing.

---

## Appendix A. Verified DFlash Config Fields

### `Qwen3.5-4B-DFlash`

```text
architectures: ["DFlashDraftModel"]
model_type: qwen3
dtype: bfloat16
num_hidden_layers: 5
hidden_size: 2560
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 128
intermediate_size: 9728
max_position_embeddings: 262144
rope_theta: 10000000
vocab_size: 248320
tie_word_embeddings: true
block_size: 16
num_target_layers: 32
dflash_config.mask_token_id: 248070
dflash_config.target_layer_ids: [1, 8, 15, 22, 29]
```

### `Qwen3.5-9B-DFlash`

```text
architectures: ["DFlashDraftModel"]
model_type: qwen3
dtype: bfloat16
num_hidden_layers: 5
hidden_size: 4096
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 128
intermediate_size: 12288
max_position_embeddings: 262144
rope_theta: 10000000
vocab_size: 248320
tie_word_embeddings: false
block_size: 16
num_target_layers: 32
dflash_config.mask_token_id: 248070
dflash_config.target_layer_ids: [1, 8, 15, 22, 29]
```

### `Qwen3.5-27B-DFlash`

```text
architectures: ["DFlashDraftModel"]
model_type: qwen3
dtype: bfloat16
num_hidden_layers: 5
hidden_size: 5120
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 128
intermediate_size: 17408
max_position_embeddings: 262144
rope_theta: 10000000
vocab_size: 248320
tie_word_embeddings: false
block_size: 16
num_target_layers: 64
dflash_config.mask_token_id: 248070
dflash_config.target_layer_ids: [1, 16, 31, 46, 61]
```

---

## Appendix B. Local Code References Used in This Research

### Tesseract app code

- `tesseract/Features/Models/ModelDefinition.swift:90-125`
- `tesseract/Features/Agent/AgentGeneration.swift:17-26`
- `tesseract/Features/Agent/LLMActor.swift:89-165` (load path, TriAttention hooks, memory budget)
- `tesseract/Features/Agent/LLMActor.swift:450-700` (post-generation snapshot + leaf store path)
- `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift:370-417` (text vs VLM loader paths)

### Vendored `mlx-swift-lm`

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (layer interleaving, hidden-state tap site)
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` (`TokenIterator` — the class a speculative iterator needs to replace or wrap)
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` (`BaseKVCache.isTrimmable`, `KVCacheSimple`, `MambaCache`, `trimPromptCache`)
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` (snapshot capture — confirmed not on the DFlash hot path)

### Sibling research

- [`docs/triattention-qwen35-paro-research-2026-04-16.md`](triattention-qwen35-paro-research-2026-04-16.md) — shared PARO geometry, non-trimmable-state rationale, memory tables.
- [`docs/marconi-hybrid-prefix-cache-implementation-plan.md`](marconi-hybrid-prefix-cache-implementation-plan.md) — prefix cache plumbing confirmed orthogonal to DFlash.
