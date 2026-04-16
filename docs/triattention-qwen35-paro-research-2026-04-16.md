# TriAttention Research for Qwen3.5 PARO in Tesseract

**Date:** 2026-04-16  
**Status:** Research note for later implementation planning.  
**Question:** Can [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921) be implemented for the Tesseract server and Tesseract agent, and what would it do to memory use for the current Qwen3.5 PARO models (`4B`, `9B`, `27B`) across practical and full context lengths?

This document records the investigation performed in the codebase and on upstream sources. It is intentionally detailed so a later session can turn it into an implementation plan without repeating the research.

---

## Executive Summary

1. **TriAttention is implementable for Tesseract, but not as a drop-in.**
   The upstream project includes vLLM and experimental MLX support, but not a Swift `mlx-swift-lm` implementation. Tesseract will need a real vendor-port inside `Vendor/mlx-swift-lm`.

2. **The Tesseract agent is the best first target.**
   The agent path does not depend on the HTTP prefix cache contract as heavily as the server path does.

3. **The Tesseract server can support it, but prefix-cache semantics must change.**
   A TriAttention-compressed attention cache should be treated like Mamba/SSM state for prefix-cache purposes: reusable and serializable, but **not tail-trimmable**. This is the key architectural constraint.

4. **TriAttention only shrinks the attention portion of the cache.**
   Qwen3.5 PARO is a hybrid architecture: every fourth layer is full attention and the other three are Mamba-style linear/state-space layers. TriAttention helps the attention KV cache but does not reduce:
   - model weights
   - Mamba recurrent state
   - prefill scratch / activation spikes

5. **`4B` and `9B` have the same cache geometry.**
   They differ in weights and compute, but not in attention/SSM cache shape. Cache math for `4B` and `9B` is therefore the same.

6. **At current Tesseract-scale contexts, the memory win is large.**
   Under current prefix-cache semantics, one hot conversation at `120K` context is roughly:
   - `4B` / `9B`: `2.12 GiB` persistent prefix cache today, about `0.37 GiB` with TriAttention at `B=12000`
   - `27B`: `4.29 GiB` persistent prefix cache today, about `0.79 GiB` with TriAttention at `B=12000`

7. **TriAttention does not solve Tesseract's existing prefill-memory problem by itself.**
   The current text path chunk-prefills; the VLM path still does not. TriAttention mainly reduces long-running decode-time KV growth, not the transient prefill working set.

---

## 1. Sources Audited

### Paper and upstream project

- Paper abstract/PDF:
  - <https://arxiv.org/abs/2604.04921>
  - <https://arxiv.org/pdf/2604.04921>
- Upstream repository:
  - <https://github.com/WeianMao/triattention>
- Upstream docs used during this investigation:
  - Calibration guide: <https://github.com/WeianMao/triattention/blob/main/docs/calibration.md>
  - MLX support note: <https://github.com/WeianMao/triattention/blob/main/docs/mlx.md>

### Model configs

- `4B`: <https://huggingface.co/z-lab/Qwen3.5-4B-PARO/blob/main/config.json>
- `9B`: <https://huggingface.co/z-lab/Qwen3.5-9B-PARO/blob/main/config.json>
- `27B`: <https://huggingface.co/z-lab/Qwen3.5-27B-PARO/blob/main/config.json>

### Local Tesseract docs and code

- Prior local cache audit:
  - [`docs/mlx-swift-lm-kv-cache-audit.md`](docs/mlx-swift-lm-kv-cache-audit.md)
- Current model list:
  - `tesseract/Features/Models/ModelDefinition.swift`
- Generation defaults:
  - `tesseract/Features/Agent/AgentGeneration.swift`
- Agent context compaction:
  - `tesseract/Features/Agent/Context/ContextManager.swift`
- Server model metadata:
  - `tesseract/App/DependencyContainer.swift`
- PARO loaders:
  - `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift`
- Prefix-cache flow:
  - `tesseract/Features/Agent/LLMActor.swift`

### Vendored `mlx-swift-lm` audit

The current worktree did not have the submodule hydrated, so the vendor audit was performed against the matching vendored checkout at commit `06c5478d6f36b61de3cf5602e8c935e9c3998fd3`, corresponding to the repo's submodule pointer.

Relevant vendor files:

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`

---

## 2. What the Paper Actually Proposes

The paper's core claim is:

- Long reasoning is bottlenecked by KV-cache growth.
- Existing compression approaches often estimate importance from recent post-RoPE queries.
- RoPE rotates queries with position, so those queries are a poor proxy for which older keys matter.
- In pre-RoPE space, Q and K vectors cluster around stable non-zero centers.
- Those centers imply preferred attention distances via a trigonometric series.
- TriAttention uses those centers, plus Q/K norms, to score keys and decide what to retain.

The paper abstract reports that on AIME25 with `32K` generation:

- TriAttention matches full-attention reasoning accuracy,
- reaches `2.5x` higher throughput,
- or achieves `10.7x` KV-memory reduction,
- while staying far more accurate than prior KV-compression baselines at comparable efficiency.

The upstream repo README also gives deployment guidance that matters directly for Tesseract:

- use a larger runtime budget such as `12000` for multi-turn chat
- disable prefix caching in the current vLLM chat setup
- keep prefill chunks bounded, e.g. `1024`

That upstream prefix-caching warning is one of the reasons the Tesseract server path needs extra care.

---

## 3. Current Tesseract and Model Facts

### 3.1 Current PARO models in Tesseract

From `tesseract/Features/Models/ModelDefinition.swift:90-125`:

| Model ID | Display name | Size description |
|---|---|---:|
| `qwen3.5-4b-paro` | `Qwen3.5-4B PARO` | `~3.5 GB` |
| `qwen3.5-9b-paro` | `Qwen3.5-9B PARO` | `~8 GB` |
| `qwen3.5-27b-paro` | `Qwen3.5-27B PARO` | `~19 GB` |

### 3.2 Context limits in practice

There are three different context numbers in play:

1. **Model config max context**
   All three PARO configs report `max_position_embeddings = 262144`.

2. **Current agent practical window**
   `CompactionSettings.standard` in `tesseract/Features/Agent/Context/ContextManager.swift:14-20` is described as the default for `120K` context-window models.

3. **Current server-reported context**
   `tesseract/App/DependencyContainer.swift:407-413` currently reports `max_context_length: 131_072`.

This matters because "full context" can mean either:

- full model capacity: `262,144`
- current agent operational target: about `120,000`
- current server public contract: `131,072`

### 3.3 Current generation defaults

From `tesseract/Features/Agent/AgentGeneration.swift:17-26`:

- `kvBits = 8`
- `kvGroupSize = 64`
- `prefillStepSize = 1024`

Those settings are the baseline for all Q8 cache math below.

### 3.4 Hybrid architecture: every 4th layer is full attention

This is the most important architectural fact for this research.

From the PARO configs and the local audit:

- `full_attention_interval = 4`
- every fourth decoder layer is full attention
- the other three are Mamba-style linear/state-space layers

From the verified config fields:

| Model | Hidden layers | Full-attention layers | Linear/SSM layers |
|---|---:|---:|---:|
| `4B` | 32 | 8 | 24 |
| `9B` | 32 | 8 | 24 |
| `27B` | 64 | 16 | 48 |

This is also reflected in the vendored `Qwen35` implementation:

- `Qwen35TextConfiguration.fullAttentionInterval = 4`
- `Qwen35DecoderLayer` marks layers as linear except every `fullAttentionInterval`-th layer
- `Qwen35TextModel.newCache()` returns:
  - `MambaCache()` for linear layers
  - `KVCacheSimple()` for attention layers

### 3.5 Text path vs VLM path

From `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift:370-417`:

- text-only PARO loads through `MLXLLM.Qwen35Model`
  - inherits `LLMModel.prepare`
  - chunked prefill
  - comment notes about `~1300 tok/s` on `Qwen3.5-4B PARO`
- vision-capable PARO loads through `MLXVLM.Qwen35`
  - comment notes unchunked prefill
  - comment notes about `~390 tok/s` on long text prompts

This is relevant because a TriAttention port that only patches the text path will not automatically cover image turns.

---

## 4. Current Prefix-Cache Behavior and Why It Matters

### 4.1 What Tesseract persists today

The current HTTP prefix-cache flow persists:

- one stable-prefix `.system` snapshot
- one `.leaf` snapshot for the stored conversation prefix

The request-local "boundary" snapshots used to synthesize canonical leaves are transient helpers, not durable additional snapshot classes.

Relevant code:

- `tesseract/Features/Agent/LLMActor.swift:462-480`
- `tesseract/Features/Agent/LLMActor.swift:489-620`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift:62-66`

### 4.2 Snapshot capture is a deep copy of all cache arrays

`HybridCacheSnapshot.capture(...)` deep-copies `layer.state` for every layer and sums `array.nbytes` into `memoryBytes`.

This means snapshot memory is a direct function of:

- the live cache representation for each layer
- the exact dtype and serialization shape used at capture time

### 4.3 `trim()` semantics in the current stack

In the current vendor cache API:

- `BaseKVCache.isTrimmable` defaults to `false`
- `KVCacheSimple` is trimmable
- `MambaCache` is effectively not trimmable
- `trimPromptCache(...)` only trims when **all** layers are trimmable

This is important because `LLMActor.swift:589-607` already contains correctness guards around normalization-time trimming: Mamba recurrent state cannot be "unwound" to represent a shorter logical prefix.

### 4.4 Why TriAttention compressed attention state should be treated like SSM state

This was the key semantic conclusion from the investigation:

- dense attention KV cache means "I have every token up to offset `N`"
- a TriAttention-compressed attention cache means "I have an opaque sparse state produced by processing a prefix of length `N`"

Those are not the same thing.

Therefore, for prefix caching:

- a compressed TriAttention attention cache should be **serializable and restorable**
- but it should be treated as **non-trimmable**
- the server should not call tail-trim logic on it

This is analogous to Mamba state today.

### 4.5 Important inference about current snapshot dtype

This is an inference from the audited vendor flow, not an explicit source statement:

- `LLMModel.prepareWithCheckpoints(...)` captures snapshots during prefill
- that path evaluates the cache but does **not** call `maybeQuantizeKVCache(...)`
- `TokenIterator.step(...)` later calls `maybeQuantizeKVCache(...)` during decode

So the best reading of the current Tesseract path is:

- mid-prefill `.system` snapshots are captured in the native prefill cache representation
- post-generation `.leaf` snapshots are typically Q8-quantized in the attention layers

That assumption is used in the prefix-cache memory tables below because it matches the current code structure more closely than assuming both snapshot types are Q8.

---

## 5. Upstream TriAttention Constraints Relevant to Tesseract

### 5.1 Prefix-caching warning

The upstream README recommends disabling prefix caching in the current vLLM chat deployment. In other words, upstream already considers "compressed KV + prefix caching" a correctness-sensitive area.

This does **not** mean Tesseract cannot support both.

It means:

- Tesseract should not assume the current prefix-cache design is automatically safe once attention becomes compressed
- any Tesseract implementation needs explicit cache semantics for sparse attention state

### 5.2 Chat-specific budget guidance

The upstream README recommends:

- default research-style budgets like `2048` or `4096` for tighter compression
- a larger budget like `12000` for multi-turn chat sessions

This is why the memory tables below include all three budgets.

### 5.3 Calibration is cheap enough to be practical

From the upstream calibration guide:

- TriAttention requires precomputed Q/K center and norm statistics
- calibration can be done from ordinary coherent plain text
- it is domain-agnostic according to the upstream notes
- it is essentially a one-pass statistics collection step for the model

So for Tesseract:

- `4B`, `9B`, and `27B` each need their own stats artifact
- this is not a deployment blocker

### 5.4 Upstream MLX support is not the Swift implementation we need

The upstream repo has experimental MLX support, but it targets Python `mlx_lm`.

Tesseract uses `mlx-swift-lm`, so the actual work is:

- port algorithm and runtime cache changes into the Swift vendor
- not "copy the Python files"

---

## 6. Memory Methodology and Assumptions

Unless otherwise stated, all cache numbers below assume:

- batch size `1`
- current Tesseract default `kvBits = 8`
- current Tesseract default `kvGroupSize = 64`
- attention-layer Q8 affine quantization with FP16 scales and FP16 biases

If a specific build stores scales/biases as FP32 instead, the Q8 numbers increase by about `5.9%`.

### 6.1 Definitions used in the tables

- **Active cache**: the live cache for one in-flight request
- **Persistent prefix cache**: `.system + .leaf` for one hot conversation
- **Peak extra while serving**: `.system + .leaf + active request cache`
- **Total peak**: `estimated live weights + peak extra while serving`

### 6.2 Important caveat on total-peak numbers

The total-peak tables use:

- `4B` live weights: about `2.5 GiB` from the comment in `tesseract/Features/Agent/LLMActor.swift:93-114`
- `9B` live weights: estimated as `5.60 GiB`
- `27B` live weights: estimated as `16.90 GiB`

The `9B` and `27B` numbers are extrapolated planning constants from the `4B` note and rounded for readability; they should not be read as exact measured live-weight footprints or as evidence of a perfectly uniform per-parameter scaling law.

So these are planning estimates, not measured instrument traces for every model.

### 6.3 Important caveat on TriAttention byte estimates

The TriAttention tables below are **engineering estimates**, not byte-for-byte guarantees from upstream.

They assume:

1. TriAttention reduces the number of retained attention tokens from `N` to `min(N, B)`.
2. The retained attention state is stored using the same current Tesseract Q8 path for leaf/active cache accounting.
3. Position metadata for retained sparse entries is small enough not to dominate the retained K/V bytes.
4. `.system` snapshots continue to be captured in the current prefill path, so their attention portion stays in native dtype at capture time.

If a future implementation stores sparse positions more expensively, or if it quantizes `.system` snapshots earlier, the tables would shift accordingly.

---

## 7. Deriving the Current Cache Math

### 7.1 Attention-layer bytes per token

From the PARO configs for all three models:

- `num_key_value_heads = 4`
- `head_dim = 256`

Per full-attention layer, each token stores:

- keys: `4 * 256 = 1024` elements
- values: `4 * 256 = 1024` elements
- total: `2048` elements

At FP16/BF16:

```text
2048 elements * 2 bytes = 4096 bytes = 4 KiB / token / attention layer
```

At current Tesseract Q8 with `groupSize = 64` and affine quantization:

```text
data bytes: 2048 * 1 byte = 2048 bytes
groups per K or V tensor: 1024 / 64 = 16
metadata per group: scale + bias = 2 bytes + 2 bytes = 4 bytes
metadata for K: 16 * 4 = 64 bytes
metadata for V: 16 * 4 = 64 bytes
total: 2048 + 64 + 64 = 2176 bytes = 2.125 KiB / token / attention layer
```

### 7.2 Linear/SSM layer state

From the PARO configs:

#### `4B` / `9B`

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `linear_conv_kernel_dim = 4`

Derived dimensions:

```text
convDim = 2 * linear_num_key_heads * linear_key_head_dim
        + linear_num_value_heads * linear_value_head_dim
        = 2 * 16 * 128 + 32 * 128
        = 8192
```

Per linear layer:

```text
conv state elems      = (kernel_dim - 1) * convDim
                      = 3 * 8192
                      = 24576

recurrent state elems = linear_num_value_heads * linear_value_head_dim * linear_key_head_dim
                      = 32 * 128 * 128
                      = 524288

total elems           = 548864
bytes at fp16         = 548864 * 2
                      = 1,097,728 bytes
                      = 1.046875 MiB
```

There are `24` linear layers, so:

```text
fixed SSM state = 24 * 1.046875 MiB = 25.125 MiB
```

#### `27B`

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 48`
- `linear_value_head_dim = 128`
- `linear_conv_kernel_dim = 4`

Derived dimensions:

```text
convDim = 2 * 16 * 128 + 48 * 128 = 10240
```

Per linear layer:

```text
conv state elems      = 3 * 10240 = 30720
recurrent state elems = 48 * 128 * 128 = 786432
total elems           = 817152
bytes at fp16         = 1,634,304 bytes = 1.55859375 MiB
```

There are `48` linear layers, so:

```text
fixed SSM state = 48 * 1.55859375 MiB = 74.8125 MiB
```

### 7.3 Whole-model cache formulas

#### `4B` / `9B`

- full-attention layers: `8`
- linear/SSM layers: `24`

So:

```text
attention growth, FP16 = 8 * 4 KiB/token = 32 KiB/token
attention growth, Q8   = 8 * 2.125 KiB/token = 17 KiB/token

active FP16 cache = 25.125 MiB + 32 KiB * ctx
active Q8 cache   = 25.125 MiB + 17 KiB * ctx
```

#### `27B`

- full-attention layers: `16`
- linear/SSM layers: `48`

So:

```text
attention growth, FP16 = 16 * 4 KiB/token = 64 KiB/token
attention growth, Q8   = 16 * 2.125 KiB/token = 34 KiB/token

active FP16 cache = 74.8125 MiB + 64 KiB * ctx
active Q8 cache   = 74.8125 MiB + 34 KiB * ctx
```

### 7.4 Geometry summary

| Model | Full-attn layers | Linear/SSM layers | Fixed SSM state | Attention growth FP16 | Attention growth Q8 |
|---|---:|---:|---:|---:|---:|
| `4B` | 8 | 24 | `25.125 MiB` | `32 KiB/token` | `17 KiB/token` |
| `9B` | 8 | 24 | `25.125 MiB` | `32 KiB/token` | `17 KiB/token` |
| `27B` | 16 | 48 | `74.8125 MiB` | `64 KiB/token` | `34 KiB/token` |

The important practical conclusion is:

- `4B` and `9B` have identical cache geometry
- `27B` is larger because it has twice as many full-attention layers and twice as many linear layers

---

## 8. Current Active-Cache Memory

### `4B` / `9B`

| Context | Active FP16/BF16 | Active Q8 |
|---|---:|---:|
| 4,096 | 153.1 MiB | 93.1 MiB |
| 8,192 | 281.1 MiB | 161.1 MiB |
| 16,384 | 537.1 MiB | 297.1 MiB |
| 32,768 | 1.02 GiB | 569.1 MiB |
| 65,536 | 2.02 GiB | 1.09 GiB |
| 120,000 | 3.69 GiB | 1.97 GiB |
| 131,072 | 4.02 GiB | 2.15 GiB |
| 262,144 | 8.02 GiB | 4.27 GiB |

### `27B`

| Context | Active FP16/BF16 | Active Q8 |
|---|---:|---:|
| 4,096 | 330.8 MiB | 210.8 MiB |
| 8,192 | 586.8 MiB | 346.8 MiB |
| 16,384 | 1.07 GiB | 618.8 MiB |
| 32,768 | 2.07 GiB | 1.14 GiB |
| 65,536 | 4.07 GiB | 2.20 GiB |
| 120,000 | 7.40 GiB | 3.96 GiB |
| 131,072 | 8.07 GiB | 4.32 GiB |
| 262,144 | 16.07 GiB | 8.57 GiB |

---

## 9. Current Prefix-Cache Memory

Under the current Tesseract architecture, one hot conversation typically means:

- one `.system` snapshot at the stable-prefix boundary
- one `.leaf` snapshot for the stored conversation
- one active request cache while serving

Using the current observed stable-prefix assumption of about `4096` tokens:

- `.system` snapshot for `4B` / `9B` is about `153.125 MiB`
- `.system` snapshot for `27B` is about `330.8125 MiB`

So:

```text
persistent prefix cache = .system + .leaf
peak extra while serving = .system + .leaf + active request cache
```

### `4B` / `9B`

| Context | Persistent Prefix Cache | Peak Extra While Serving |
|---|---:|---:|
| 4,096 | 246.2 MiB | 339.4 MiB |
| 8,192 | 314.2 MiB | 475.4 MiB |
| 16,384 | 450.2 MiB | 747.4 MiB |
| 32,768 | 722.2 MiB | 1.26 GiB |
| 65,536 | 1.24 GiB | 2.32 GiB |
| 120,000 | 2.12 GiB | 4.09 GiB |
| 131,072 | 2.30 GiB | 4.45 GiB |
| 262,144 | 4.42 GiB | 8.70 GiB |

### `27B`

| Context | Persistent Prefix Cache | Peak Extra While Serving |
|---|---:|---:|
| 4,096 | 541.6 MiB | 752.4 MiB |
| 8,192 | 677.6 MiB | 1.00 GiB |
| 16,384 | 949.6 MiB | 1.53 GiB |
| 32,768 | 1.46 GiB | 2.59 GiB |
| 65,536 | 2.52 GiB | 4.72 GiB |
| 120,000 | 4.29 GiB | 8.25 GiB |
| 131,072 | 4.65 GiB | 8.97 GiB |
| 262,144 | 8.90 GiB | 17.47 GiB |

---

## 10. TriAttention Memory Model for Tesseract

### 10.1 Core approximation

For cache math, the simplest useful approximation is:

- keep the fixed SSM state unchanged
- replace attention token count `N` with `min(N, B)` for budget `B`

So:

#### `4B` / `9B`

```text
TriAttention active Q8 ≈ 25.125 MiB + 17 KiB * min(ctx, B)
```

#### `27B`

```text
TriAttention active Q8 ≈ 74.8125 MiB + 34 KiB * min(ctx, B)
```

### 10.2 Prefix-cache assumption used for apples-to-apples comparison

To compare against today's prefix-cache numbers, this research assumes:

- TriAttention-enabled `.system` snapshots are still captured during prefill in native dtype
- TriAttention-enabled `.leaf` and active request cache use the current Q8-style accounting on the retained attention state

That is why the tables below still show a nonzero `.system` cost even when the runtime budget is small.

This is an engineering estimate, not a hard implementation guarantee. The actual `.system` footprint after a real port could shift if:

- TriAttention retention is enforced per layer rather than as the simple whole-model `min(ctx, B)` approximation used here
- sparse retained-position metadata is materially larger than assumed
- the prefill checkpoint path changes dtype or quantization timing

The TriAttention tables are therefore appropriate for planning and prioritization, but they should be re-measured empirically after the vendor port before using them as hard deployment budgets.

### 10.3 TriAttention vs current prefix-cache numbers

Budgets included:

- `B=2048`
- `B=4096`
- `B=12000` for chat-oriented sessions

### `4B` / `9B`

| Context | Current Persistent | Current Peak Extra | Tri B=2048 Persistent | Tri B=2048 Peak | Tri B=4096 Persistent | Tri B=4096 Peak | Tri B=12000 Persistent | Tri B=12000 Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 246.2 MiB | 339.4 MiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 246.2 MiB | 339.4 MiB |
| 8,192 | 314.2 MiB | 475.4 MiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 314.2 MiB | 475.4 MiB |
| 16,384 | 450.2 MiB | 747.4 MiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |
| 32,768 | 722.2 MiB | 1.26 GiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |
| 65,536 | 1.24 GiB | 2.32 GiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |
| 120,000 | 2.12 GiB | 4.09 GiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |
| 131,072 | 2.30 GiB | 4.45 GiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |
| 262,144 | 4.42 GiB | 8.70 GiB | 148.2 MiB | 207.4 MiB | 246.2 MiB | 339.4 MiB | 377.5 MiB | 601.8 MiB |

### `27B`

| Context | Current Persistent | Current Peak Extra | Tri B=2048 Persistent | Tri B=2048 Peak | Tri B=4096 Persistent | Tri B=4096 Peak | Tri B=12000 Persistent | Tri B=12000 Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 541.6 MiB | 752.4 MiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 541.6 MiB | 752.4 MiB |
| 8,192 | 677.6 MiB | 1.00 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 677.6 MiB | 1.00 GiB |
| 16,384 | 949.6 MiB | 1.53 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |
| 32,768 | 1.46 GiB | 2.59 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |
| 65,536 | 2.52 GiB | 4.72 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |
| 120,000 | 4.29 GiB | 8.25 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |
| 131,072 | 4.65 GiB | 8.97 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |
| 262,144 | 8.90 GiB | 17.47 GiB | 345.6 MiB | 488.4 MiB | 541.6 MiB | 752.4 MiB | 804.1 MiB | 1.25 GiB |

### 10.4 What the TriAttention tables mean in practice

At `120K` context:

- `4B` / `9B`
  - current persistent prefix cache: `2.12 GiB`
  - TriAttention `B=12000`: `377.5 MiB`
  - reduction: about `5.7x`
- `27B`
  - current persistent prefix cache: `4.29 GiB`
  - TriAttention `B=12000`: `804.1 MiB`
  - reduction: about `5.4x`

At `262,144` full model context:

- `4B` / `9B`
  - current persistent prefix cache: `4.42 GiB`
  - TriAttention `B=12000`: still `377.5 MiB`
  - reduction: about `12.0x`
- `27B`
  - current persistent prefix cache: `8.90 GiB`
  - TriAttention `B=12000`: still `804.1 MiB`
  - reduction: about `11.3x`

The plateau is expected: once `ctx > B`, the retained attention state stops growing.

---

## 11. Fit on 48 GB / 64 GB / 128 GB Macs

These totals use the **current** prefix-cache architecture, not TriAttention.

Again, this is planning math:

- includes estimated live weights
- includes one hot conversation's `.system + .leaf + active request cache`
- excludes OS usage, transient prefill spikes, VLM image tensors, and non-LLM app pressure

Estimated live weights used in the totals:

- `4B`: `2.50 GiB`
- `9B`: `5.60 GiB`
- `27B`: `16.90 GiB`

| Model | Context | Total Peak | Free on 48 GB | Free on 64 GB | Free on 128 GB |
|---|---:|---:|---:|---:|---:|
| 4B | 120,000 | 6.59 GiB | 41.41 GiB | 57.41 GiB | 121.41 GiB |
| 4B | 131,072 | 6.95 GiB | 41.05 GiB | 57.05 GiB | 121.05 GiB |
| 4B | 262,144 | 11.20 GiB | 36.80 GiB | 52.80 GiB | 116.80 GiB |
| 9B | 120,000 | 9.69 GiB | 38.31 GiB | 54.31 GiB | 118.31 GiB |
| 9B | 131,072 | 10.05 GiB | 37.95 GiB | 53.95 GiB | 117.95 GiB |
| 9B | 262,144 | 14.30 GiB | 33.70 GiB | 49.70 GiB | 113.70 GiB |
| 27B | 120,000 | 25.15 GiB | 22.85 GiB | 38.85 GiB | 102.85 GiB |
| 27B | 131,072 | 25.87 GiB | 22.13 GiB | 38.13 GiB | 102.13 GiB |
| 27B | 262,144 | 34.37 GiB | 13.63 GiB | 29.63 GiB | 93.63 GiB |

Interpretation:

- `4B` and `9B` comfortably fit on `48 GB` even at full `262K` context by the steady-state cache math.
- `27B` also fits on `48 GB` by steady-state arithmetic, but `262K` is the only row that looks operationally risky once real prefill spikes and other processes are considered.
- `64 GB` and `128 GB` are comfortable for all three by steady-state cache math.
- With TriAttention enabled, fit becomes much easier because the per-conversation prefix-cache footprint stops growing after the configured budget.

---

## 12. Feasibility for the Tesseract Agent

### Verdict

**Yes.** The agent is the best first implementation target.

### Why

- The agent path already runs the relevant Qwen3.5 PARO models.
- It is less tightly coupled to the HTTP prefix-cache contract.
- The biggest direct win is reducing long-running conversation memory from retained attention KV.

### What must change

At minimum:

1. Add a new sparse/compressed attention-cache type to `mlx-swift-lm`.
2. Patch the Qwen3.5 attention path to use TriAttention scoring/pruning.
3. Load per-model TriAttention stats.
4. Preserve the existing `MambaCache` behavior unchanged.
5. Update snapshot serialization/restoration to understand the new cache class.

### Important caveat

A text-only implementation is not the same as full Tesseract support. If image turns must also work, the VLM Qwen3.5 path will need the same treatment.

---

## 13. Feasibility for the Tesseract Server

### Verdict

**Yes, but only if prefix-cache semantics are made explicit.**

### Why it is harder than the agent path

The server depends heavily on:

- exact cache restore
- stable token-offset semantics
- normalization-time correctness guards
- snapshot storage and eviction

Compressed attention state violates the old assumption that attention cache means "every token up to offset `N` is still present and can be tail-trimmed later."

### Correct mental model

Once attention layers are represented by TriAttention sparse state, they should be treated like SSM state **for prefix caching**:

- restorable
- serializable
- reusable
- **not trimmable**

That is the right conceptual translation of the user's question about treating attention like SSM state.

### Safest rollout options

#### Option A: feature-flag TriAttention and disable HTTP prefix caching

Pros:

- simplest correctness story
- aligns with upstream's current deployment guidance

Cons:

- leaves major server-side reuse benefits on the table

#### Option B: implement TriAttention-aware prefix caching

Pros:

- preserves most of the memory and reuse benefits
- better long-session server story

Cons:

- requires explicit handling of:
  - non-trimmable compressed attention cache
  - snapshot serialization for sparse state
  - cache-key partitioning by TriAttention config
  - normalization / offset-alignment rules

### Bottom-line recommendation

If the goal is the fastest path to a working result:

- do the **agent first**
- then do the **server** behind a feature flag
- and only then decide whether to keep prefix caching enabled for TriAttention sessions from day one

---

## 14. Concrete Vendor-Side Implications

This is not yet the implementation plan, but these are the code-level implications discovered during research.

### 14.1 New cache class

Tesseract will likely need something like:

- `TriAttentionSparseKVCache`

Properties:

- serializable
- restorable
- `isTrimmable = false`
- stores retained sparse positions and compressed K/V state

### 14.2 Snapshot serialization support

`HybridCacheSnapshot` and the prompt-cache serialization helpers will need to understand the new cache class:

- class-name mapping for save/restore
- state tensor layout
- any metadata required for sparse positions, retained offsets, and budget/config

### 14.3 Cache partition key must include TriAttention config

Any reusable server snapshot must be partitioned not just by model and KV quantization settings, but also by TriAttention runtime config, for example:

- TriAttention enabled/disabled
- stats file fingerprint or calibration version
- runtime budget
- any pruning schedule / hysteresis / implementation version

Otherwise the prefix cache could mix incompatible state layouts.

### 14.4 Attention-layer call sites need a real port

The work is not in Tesseract app code alone. The Qwen3.5 model implementation itself needs the attention path patched in the Swift vendor.

### 14.5 Both text and VLM paths matter

If Tesseract needs image turns under the PARO models, TriAttention support has to exist in:

- `MLXLLM.Qwen35Model`
- `MLXVLM.Qwen35`

not just one of them.

---

## 15. Things TriAttention Does Not Fix

This matters because it is easy to over-attribute wins to KV compression.

TriAttention does **not** directly fix:

1. **Model weight memory**
   The weights are unchanged.

2. **Mamba/SSM state**
   Those layers still keep their recurrent state.

3. **Prefill scratch / activation spikes**
   Long prefills can still blow up transient memory even if the eventual retained KV is smaller.

4. **The unchunked VLM prefill path**
   That is a separate issue in the current Tesseract stack.

So TriAttention is a strong solution for:

- active long-session decode memory
- persistent prefix-cache footprint

But it is not the whole memory story.

---

## 16. Open Questions to Resolve in the Later Implementation Plan

These were left intentionally open because this document is the research handoff, not the patch plan.

1. **Sparse state encoding**
   What exact tensors and metadata should the Swift cache class store for retained positions and compressed K/V?

2. **Quantization timing**
   Should sparse attention state be quantized immediately during prefill checkpoints, or continue to follow the current "prefill native / decode quantized" flow?

3. **Normalization / alignment rules**
   Today Tesseract sometimes skips leaf storage rather than trim mismatched hybrid caches. How should that behave once attention is also non-trimmable?

4. **Prefix-cache hit policy**
   Does any existing hit-selection logic assume dense attention semantics implicitly?

5. **VLM parity**
   Is the first implementation text-only, or do image turns need support immediately?

6. **Correctness harness**
   Which existing benchmark/correctness runners should be extended first to prove logit stability and cache-restore correctness?

7. **Budget policy**
   Should Tesseract default to something like `B=12000` for chat and expose lower budgets only as advanced tuning?

---

## 17. Recommended Direction for the Future Implementation Plan

The research supports the following order:

1. **Agent-only, text-only TriAttention prototype**
   Patch `MLXLLM.Qwen35Model`, ignore HTTP prefix caching, verify correctness and steady-state memory wins.

2. **Server feature flag**
   Add a server/runtime switch that enables TriAttention without assuming all existing prefix-cache behavior is still valid.

3. **TriAttention-aware prefix-cache design**
   Treat sparse attention state like Mamba state for trimming semantics and make snapshot serialization explicit.

4. **VLM support**
   Patch the `MLXVLM` Qwen3.5 path if image turns need parity.

This order minimizes correctness risk while still delivering the largest practical memory win early.

---

## Appendix A. Exact Verified Config Fields

### `Qwen3.5-4B-PARO`

```text
max_position_embeddings: 262144
num_hidden_layers: 32
full_attention_interval: 4
num_key_value_heads: 4
head_dim: 256
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 32
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
```

### `Qwen3.5-9B-PARO`

```text
max_position_embeddings: 262144
num_hidden_layers: 32
full_attention_interval: 4
num_key_value_heads: 4
head_dim: 256
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 32
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
```

### `Qwen3.5-27B-PARO`

```text
max_position_embeddings: 262144
num_hidden_layers: 64
full_attention_interval: 4
num_key_value_heads: 4
head_dim: 256
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 48
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
```

---

## Appendix B. Local Code References Used in This Research

### Tesseract app code

- `tesseract/Features/Models/ModelDefinition.swift:90-125`
- `tesseract/Features/Agent/AgentGeneration.swift:17-26`
- `tesseract/Features/Agent/Context/ContextManager.swift:14-20`
- `tesseract/App/DependencyContainer.swift:407-413`
- `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift:370-417`
- `tesseract/Features/Agent/LLMActor.swift:93-123`
- `tesseract/Features/Agent/LLMActor.swift:462-620`

### Vendored `mlx-swift-lm`

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  - `Qwen35TextConfiguration.fullAttentionInterval`
  - `Qwen35TextModel.newCache()`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift`
  - `prepare(...)`
  - `prepareWithCheckpoints(...)`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
  - `TokenIterator.prepare(...)`
  - `TokenIterator.step(...)`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
  - `BaseKVCache.isTrimmable`
  - `MambaCache`
  - `trimPromptCache(...)`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`
  - snapshot capture
  - `memoryBytes`
  - checkpoint types
