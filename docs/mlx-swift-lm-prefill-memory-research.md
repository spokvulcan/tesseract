# MLX-Swift-LM Prefill Memory & Prefix-Cache Research

**Date:** 2026-04-08
**Context:** Tesseract Agent crashed in Release with `EXC_BREAKPOINT` inside `mlx_async_eval` during prefill of a ~19K-token prompt for Qwen3.5-4B-paro on a 48 GB M3 Max. VM Summary at crash time: **27.4 GB MALLOC**. The user reports llama.cpp prefilling the same model uses minimal memory beyond weights+KV.

This document is a research note for resuming work in a fresh session. It is **not** a patch — it documents what's broken, why, the relevant prior art, and a concrete patch ladder.

---

## TL;DR (the smoking gun)

`mlx-swift-lm` has chunked prefill but **does not call `MLX.GPU.clearCache()` between chunks**. The Python `mlx-lm` does — that line was added in [mlx-lm PR #917](https://github.com/ml-explore/mlx-lm/pull/917) (merged 2026-02-23) and reduced peak memory from **50+ GB to 28 GB** on a 16K-token prompt with a comparable model. The Swift port never received the analogous patch.

Combined with two other issues (MLX's unbounded Metal buffer pool by default, and the VLM `Qwen35.prepare` path that doesn't even chunk), prefill of a 19K-token prompt on a 4B model can monotonically grow to 27+ GB before either OOM-killing the process or triggering an MLX assertion in `mlx_async_eval`.

**Minimum viable fix** (3 lines of Swift in our vendored fork) is documented in [§ 7](#7-patch-ladder).

---

## 1. The crash

Stack trace:

```
0  _assertionFailure (libswiftCore)
6  _mlx_error (Vendor/mlx-swift-lm/.../error.cpp:52)
7  mlx_async_eval (transforms.cpp:15)
8  eval(_:)
9  TokenIterator.prepare(input:windowSize:) (Evaluate.swift:647)
10 TokenIterator.init (Evaluate.swift:595-596)
11 LLMActor.makeHTTPPrefixCacheGeneration (LLMActor.swift:548)
```

VM Summary at crash time:

```
MALLOC                            27.4G     7085
TOTAL                             30.2G    12793
```

`activeMemMB` from the last STORE log line before the crash was **only 8.4 GB** (model + 3 cache entries + working set). The crash happened during a single prefill that needed ~19 GB of *transient* memory the system couldn't satisfy. macOS's low-power mode was active (`lowPowerMode: 1`), which further reduces GPU wired-memory limits.

The failing request was a continuation after a long subagent loop. The cache HIT logic worked (we matched the previous main-agent state), but the new tokens to prefill were still substantial.

---

## 2. Why MLX-Swift-LM blows up on long prefills

### 2.1 MLX's lazy evaluation accumulates the compute graph until `eval()`

From the official MLX docs (`docs/src/usage/lazy_evaluation.rst`):

> "When you perform operations in MLX, no computation actually happens. Instead a compute graph is recorded. The actual computation only happens if an `eval` is performed. […] the graph of `expensive_fun` is still built, and that has some cost associated to it."

Confirmation by `awni` (Apple/MLX lead) in [ml-explore/mlx #742](https://github.com/ml-explore/mlx/issues/742):

> "MLX has a memory buffer cache because device memory allocations are expensive. So MLX will not return 'freed' arrays to the system immediately. Rather they get held in the buffer cache and possibly reused."
>
> "everyone has run into the related problem which is that MLX can use way too much memory for token generation."

So **two independent mechanisms retain memory**:
1. The **lazy compute graph** keeps intermediate arrays alive until `mx.eval()` is called.
2. The **Metal buffer cache** holds "freed" buffers in a pool (`buffer_cache_`) so subsequent allocations can reuse them without hitting `newBuffer()`.

### 2.2 Direct confirmation in [mlx-lm PR #924](https://github.com/ml-explore/mlx-lm/pull/924)

> "The `batch.tokens` arrays are not always used and this can create some big graphs for long generations."

This is exactly the failure pattern: any reference to a produced array that isn't promptly evaluated keeps the entire upstream graph alive.

### 2.3 The Metal allocator does not bound pool growth on monotonically-growing workloads

[ml-explore/mlx #3350](https://github.com/ml-explore/mlx/issues/3350) — "Metal caching allocator retains unbounded buffer pool when cached buffers cannot be reused" (closed wontfix 2026-04):

> "When an MLX process allocates buffers of monotonically increasing sizes, the caching allocator accumulates Metal buffers in its pool indefinitely. Freed buffers from previous allocations are never returned to the OS because they are always smaller than the next allocation and can never be reused. The pool keeps growing until it hits `max_pool_size_`, which defaults to `block_limit_` (1.5 times the device's max recommended working set), roughly 192 GB on an M2 Ultra with 128 GB of RAM."

Inspection of `mlx/backend/metal/allocator.cpp` confirms: the GC path only triggers in `malloc()` when `get_active_memory() + get_cache_memory() + size >= gc_limit_`, and `gc_limit_` defaults to `0.95 × max_recommended_working_set_size`. On a 48 GB Mac the working set is ~36 GB, so **`gc_limit_` ≈ 34 GB** — the allocator does not begin to reclaim cache until the process is already deep in wired-memory territory.

Maintainer reply ([@zcbenz](https://github.com/ml-explore/mlx/issues/3350#issuecomment-4186693866)):

> "We can't just recommend clearing cache frequently as it will kill performance significantly, usually memory allocation would make the inference several times slower. Clearing cache on memory pressure is a practical solution but we should not do it in the framework's side, it would be a hidden performance killer."

**Apple's stance is explicit: the framework will not manage pool eviction automatically — the caller must call `mx.clear_cache()` at appropriate boundaries.**

### 2.4 The single most important upstream fix: [mlx-lm PR #917](https://github.com/ml-explore/mlx-lm/pull/917)

"Add `mx.clear_cache()` to piecewise prompt processing in server" (merged 2026-02-23):

> "Prompts that were being newly added to the server didn't have cache-clearing enabled while being processed. This lead to massive memory hang. **For prompts of length ~16K on GLM 4.7 Flash 6bit, this took peak memory from 50+ GB to a more reasonable 28 GB.**"

This is a near-direct analog of Tesseract's crash: ~19K prompt, 4B-class hybrid model, MALLOC ballooning to 27.4 GB. **The Swift port has never received the analogous patch.**

### 2.5 The Swift port's `prepare()` loop does not call `clearCache()`

`Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:22-37`:

```swift
public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
    -> PrepareResult
{
    let prefillStepSize = windowSize ?? 512
    var y = input.text

    // Prepare the prompt in chunks if larger than the prefill size
    while y.tokens.size > prefillStepSize {
        let input = y[.newAxis, ..<prefillStepSize]
        _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)            // evaluates cache, but NOT logits/intermediates
        y = y[prefillStepSize...]
        // ← MLX.GPU.clearCache() MISSING HERE
    }

    return .tokens(y)
}
```

Compare to current Python `mlx_lm/generate.py`:

```python
while total_prompt_tokens - prompt_processed_tokens > 1:
    …
    _model_call(input_tokens=prompt[:n_to_process][None], …)
    quantize_cache_fn(prompt_cache)
    mx.eval([c.state for c in prompt_cache])
    prompt_processed_tokens += n_to_process
    …
    mx.clear_cache()                    # <<< missing in Swift port
```

At `prefillStepSize = 256` on a 19K prompt that's **75 consecutive chunk forward passes** whose transient activations stay in the Metal cache pool. The first time any allocation exceeds the cache size, MLX goes to `device_->newBuffer()` for new wired memory — and the whole chain of intermediates is still in the pool. The crash signature matches exactly.

### 2.6 The VLM `Qwen35.prepare` path does not chunk at all

`Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1104-1165` overrides `prepare()` and processes the **entire input in one shot**:

```swift
public func prepare(
    _ input: LMInput,
    cache: [any KVCache],
    windowSize _: Int?               // ← windowSize ignored!
) throws -> PrepareResult {
    let inputIds = input.text.tokens
    // … vision/multimodal handling …
    let output = languageModel(
        inputIds,                     // ← passes WHOLE input.text.tokens, no chunking
        inputsEmbeds: inputEmbeddings,
        cache: typedCache,
        …
    )
    return .logits(output)
}
```

`Qwen3.5-4B-paro` is loaded via the ParoQuant path, which uses **VLM** `Qwen35`, not the LLM one. So `prefillStepSize` has no effect on this model — the entire 19K-token prompt goes through one forward pass, materializing all intermediates simultaneously.

This is a separate bug from §2.5 and is even more important for Tesseract's specific model.

### 2.7 [mlx-lm PR #943](https://github.com/ml-explore/mlx-lm/pull/943) confirms the speed↔memory tradeoff

"Proposal: `--prefill-step-size` as cmd line argument" (merged 2026-03-05):

> Reporter on M2 Pro 32GB running Qwen3.5-35B-A3B-4bit:
> "When the prompt length comes to 15K~20K (with Qwen3.5-35B-A3B-4bit), MLX would run out all memory and cause system stuck or crash almost every time even I limit the prompt cache size. However, llama.cpp could run in such case stably. […] When I change the prefill_step_size to 1024, I can run the same LLM with 20K ctx on MLX as llama.cpp does."

20K tokens on a 32 GB machine: default 2048-step crashes; 1024-step works. Tesseract's `prefillStepSize=256` is on the small side already, but we need the *clearCache* call to make it actually free between chunks.

### 2.8 [mlx-examples PR #931](https://github.com/ml-explore/mlx-examples/pull/931) — original chunked prefill: **8.4 GB → 2.6 GB**

The PR that introduced stepped prefill in mlx-lm reported, for a 4232-token prompt:
- Without stepped prefill: **Peak memory: 8.422 GB**
- With stepped prefill (512-chunk): **Peak memory: 2.640 GB**

A **3.2× peak memory reduction** from chunking alone — but only because that PR also added `mx.eval` on cache state between chunks.

### 2.9 Hybrid Mamba/attention compounds the memory pressure

Qwen3.5 is hybrid: 3 Mamba layers per 1 attention layer (every 4th is attention). Mamba's chunked-scan kernel allocates large temporary tensors proportional to `(seq_len × hidden_dim × state_dim)` for each forward pass. [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980) documents this systemically; [mlx-lm #903](https://github.com/ml-explore/mlx-lm/issues/903) confirms `Mamba cache cannot be trimmed`. From `angeloskath` in #903:

> "The lru prompt cache leaves the cache in memory when trimming. That was meant as an optimization since you can fork conversations for free. However it is obviously problematic when the cache is 10GB and then you copy it 10x so you spend 100GB on a cache..."

### 2.10 Related secondary failure modes (not primary, but worth knowing)

- [mlx #3302](https://github.com/ml-explore/mlx/issues/3302) — GPU watchdog kills process during long-context SDPA prefill at 65K+ keys (`steel_attention` dispatch exceeds Metal command-buffer time limit).
- [mlx #3186](https://github.com/ml-explore/mlx/issues/3186) — Kernel panic (`IOGPUMemory.cpp:550`) on M4 Max with large prefill (~173K tokens).
- [mlx-lm #883](https://github.com/ml-explore/mlx-lm/issues/883) — `mlx_lm.server` causes macOS kernel panic from `set_wired_limit` defaulting to ~75% RAM.

These don't apply at our 19K context but are good to know if context grows.

---

## 3. Comparison to llama.cpp (the 100× memory gap)

| Dimension | MLX | llama.cpp |
|---|---|---|
| **Chunked prefill** | In upper library (`mlx-lm`); requires caller to `clearCache()` between chunks. **Not ported into `mlx-swift-lm`.** | Built into core via `n_ubatch` (default **512**). Each ubatch is its own GGML graph with its own scratch arena, freed on graph teardown. |
| **Lazy graph vs eager** | Lazy: every op allocates and keeps arrays alive until `eval()`. | Eager: a `ggml_context` with a fixed-size scratch arena per ubatch. Reused across tokens; never grows. |
| **Buffer cache** | Unbounded `max_pool_size_` by default. Only shrinks at `gc_limit_ ≈ 0.95 × working_set` (≈34 GB on a 48 GB Mac). | No "buffer cache" at all — buffers belong to a ggml context, freed on context destroy. |
| **KV cache layout** | Contiguous `MLXArray` per layer. Resizing for prefill chunks reallocates. | Pre-allocated single contiguous slab for the full context at session start. Prefill writes into pre-allocated slots — **zero allocation during prefill**. |
| **FlashAttention** | `mx.fast.scaled_dot_product_attention` exists with `mask="causal"`, used by `MLXFast.scaledDotProductAttention` in Swift. **Not streaming** — the whole (Q,K,V) for the chunk is materialized. Chunked SDPA still unmerged ([mlx #3307](https://github.com/ml-explore/mlx/pull/3307)). | `-fa on` uses a streaming flash kernel that processes K/V in tiles and never materializes the full attention matrix. |
| **KV cache quantization** | `kv_bits=4/8` via `QuantizedKVCache`. Attention layers only — not Mamba. | `--cache-type-k q4_0 --cache-type-v q8_0` etc. Lossy but zero-overhead. |
| **Memory mapping** | `mx.load` mmaps safetensors but arrays enter the allocator's active accounting. | mmaps weights and never copies them into per-graph buffers. Weights are *not* counted toward the working set. |
| **`set_wired_limit`** | Default in `mlx-lm` server and `mlx-swift-lm` wires ~75% RAM. Wired memory cannot be swapped. | `mlock` is opt-in via `--mlock`. |

**Direct 4B comparison** ([mlx-lm #644](https://github.com/ml-explore/mlx-lm/issues/644), `gpt-oss-20b-MXFP4-Q8` on 16 GB M4 mini):
- llama.cpp `-c 32768 --no-mmap -fa on --n-cpu-moe 12`: works at ~26 tok/s, **13 GB RAM / 0.04 GB swap**
- `mlx_lm.chat`: **OOM crash** during first prompt, 12.9 GB RAM / 0.98 GB swap

**21K prompt with Qwen3.5-A17B 8-bit** ([mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980)):
- Stock mlx-lm: cold 169.9s, **warm 169.2s** (zero cache reuse — Mamba cache cannot be trimmed)
- llama.cpp `--swa-full` GGUF Q8_0: cold 230.7s, **warm 6.9s** (−97%)

---

## 4. State of the art on prefill and prefix caching

### 4.1 Chunked prefill — SARATHI (OSDI 2024)

**Paper:** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" — https://arxiv.org/abs/2308.16369 / https://arxiv.org/html/2403.02310v1

**Key insight:** "Sequences of 512 tokens are enough to saturate GPU compute." Chunking rarely hurts throughput and dramatically helps peak memory. Production token budgets:
- Strict latency SLOs: 256–512 per chunk
- Relaxed: 1024–2048 per chunk

**Reported peak memory reduction with chunking:**
- 16K prefill, 512-chunk: O(N²) attention scores reduced from ~400 MB → ~13 MB (**32×**)
- 16K prefill, 2048-chunk: ~52 MB (**8×**)
- 16K prefill, 4096-chunk: ~104 MB (**4×**)

These assume un-fused attention. With fused FlashAttention-style kernels, the N² matrix is never in HBM, so chunking mainly reduces activation/intermediate buffers (QKV projections, softmax workspace).

### 4.2 vLLM chunked prefill

Since vLLM V1, chunked prefill is **enabled by default**. Default `max_num_batched_tokens = 2048`. The scheduler prioritizes decode and batches all pending decodes before scheduling prefill, so prefill chunks are interleaved with decodes ("piggybacking"). Tesseract has no concurrent decode load to piggyback, so this benefit is small for us, but the chunking + memory bound is still relevant.

### 4.3 LM Studio finding: 512 → 8192 chunk size

[lmstudio-ai/lmstudio-js #507](https://github.com/lmstudio-ai/lmstudio-js/issues/507) — bumping `PROMPT_PROCESSING_CHUNK_SIZE` from 512 to 8192 on M1 Pro 32GB gave **1.5× faster prefill**.

> "The default chunk size of 512 processes prompts in small batches, requiring many iterations. Larger chunks (4096) reduce overhead and better utilize the GPU's parallel processing capabilities. However, very large chunks (8192+) can cause memory pressure and diminishing returns."

This is the speed/memory knob in the other direction. Tesseract's current `prefillStepSize=256` is on the small side; with `clearCache()` between chunks, we can probably bump to 1024–2048 and gain throughput.

### 4.4 PagedAttention (SOSP 2023)

**Paper:** Kwon et al. — https://arxiv.org/abs/2309.06180

Block-based KV cache (default block size **16 tokens** in vLLM). Per-sequence block table maps logical positions to physical block IDs. Sharing via reference-counted blocks with copy-on-write.

**Block size tradeoff:**
- Small (16): max prefix hit rate, more metadata, more kernel launches
- Large (256): low metadata, poor prefix hit rate

**Note:** `docs/HTTP_SERVER_SPEC.md` proposes **256-token blocks**. This is **16× larger than vLLM's default** and will hurt prefix hit rate substantially. Recommendation: revise to 16–32 tokens if we ever build the block-based cache.

**For Tesseract specifically: paged attention is overkill.** It's a throughput-at-scale optimization for serving many concurrent sequences. Tesseract serves 1–3 concurrent HTTP clients, all with long shared system prompts. Contiguous KV with explicit `copy()`-based prefix sharing is the right design.

### 4.5 RadixAttention / SGLang

**Paper:** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" — https://arxiv.org/pdf/2312.07104
**Blog:** https://lmsys.org/blog/2024-01-17-sglang/

Radix tree storing variable-length token-prefix edges. Token-level granularity (no quantization to block boundaries). LRU eviction of leaf nodes plus cache-aware scheduling.

**Performance:** "up to 5× higher throughput" vs baseline on multi-turn workloads.

**Why it matters for Tesseract:** Coding agents send the same system prompt (~4K tokens) every request. With token-level radix tree we'd hit on the system prompt + tools prefix even when the user message changes. Current Tesseract spike misses all of this because it keys on whole-conversation hashes.

### 4.6 FlashAttention on Apple Silicon

**Paper:** Dao et al., "FlashAttention" (NeurIPS 2022) — https://arxiv.org/abs/2205.14135

**Status on Apple Silicon:**
- Apple Silicon has **unified memory**, not the HBM/SRAM split that motivated FlashAttention. The HBM-tax win is muted.
- MLX team explicitly noted ([mlx #129](https://github.com/ml-explore/mlx/issues/129)): *"The HBM tax that Flash Attention solves doesn't quite apply in Apple Silicon architecture."*
- However: **avoiding materialization of the N² score matrix** still saves working memory. That's the win we need for long-context prefill.

**MLX has fused causal SDPA today**:
- `MLXFast.scaledDotProductAttention` with `mask="causal"` — this is what Tesseract is already using via `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:46-75`.
- [mlx PR #1924](https://github.com/ml-explore/mlx/pull/1924): fused causal masking. M3 Max benchmarks: 117% speedup at 2048 tokens, 132% at 4096.
- [mlx PR #1610](https://github.com/ml-explore/mlx/pull/1610): matrix attention kernel for query lengths ≥ 32. M3 Max, head_dim=64, fp16: 150% speedup at 32 tokens, 83% at 512.

**Metal FlashAttention** (Philip Turner's standalone Metal port — https://github.com/philipturner/metal-flash-attention) exists and Draw Things integrated it (Liu Liu reports "more than 20% faster than MLX SDPA at 4096 sequence length"), but **integrating MFA into MLX is not worth it** — large effort, modest improvement, MLX is actively optimizing its own kernel.

**Conclusion:** Tesseract is already on the right kernel. The fused causal SDPA path is what it should be. Don't pursue MFA integration.

### 4.7 Marconi — prefix caching for hybrid LLMs (MLSys 2025)

**This is the most important paper for Tesseract.**

**Paper:** Pan et al., "Marconi: Prefix Caching for the Era of Hybrid LLMs" — https://assets.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf

**Key claim:** **34.4× higher token hit rate; 71.1% / 617 ms lower P95 TTFT** vs SOTA prefix caching systems on hybrid LLMs.

**The fundamental problem Marconi solves:**

> "SSM State Properties:
> 1. SSM states are constant-sized regardless of how many tokens they represent.
> 2. SSM states are updated in place, so a sequence's states cannot be rolled back to represent its prefixes.
> 3. SSM states are orders of magnitude larger than the KVs of a single token."

Three design innovations:

1. **Speculative insertion at admission time.** Before prefilling a new request, walk the radix tree of stored sequences. If the new sequence creates a *branch point* with an existing one, that branch point is checkpointed during prefill. If the new sequence simply extends an existing path, only the tail is checkpointed.

2. **FLOP-aware eviction.** Score = `recency + α × flop_efficiency`, where `flop_efficiency = total_flops_across_layers / memory_consumption_of_states`. Longer sequences have higher FLOP-per-byte, so they're worth keeping even if accessed less recently.

3. **Two-pass prefill for chunk alignment.** If the matched prefix length doesn't coincide with a Mamba checkpoint, do two passes: first pass reaches the checkpoint, second pass continues to the request length.

**Marconi numbers on SWEBench (closest workload to Tesseract):**
- vs vLLM+: **34.4× hit rate** improvement
- vs SGLang+: **219.7% P95 TTFT** improvement
- 617 ms TTFT reduction at p95

SWEBench is the agentic-coding benchmark — exactly our workload. The win is concentrated on workloads with wide input length distributions (hundreds to tens of thousands of tokens), which is exactly what coding agents do.

**Marconi reference repo:** https://github.com/ruipeterpan/marconi

### 4.8 SGLang's `MambaRadixCache` (production reference impl)

**Blog:** https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/

SGLang shipped hybrid model support via `MambaRadixCache` in late 2025:

> **Match:** Return the best node where Mamba state value is not None and the key is a prefix of input. Copy the Mamba state from the radix tree into a new memory region.
>
> **Insert:** KV cache and Mamba states are inserted after chunked prefill or decoding stages.
>
> **Evict:** Two LRU lists (one for Mamba, one for KV). KV cache must be evicted from leaves to root; Mamba states can be evicted from any node.

The key innovation is the **dual LRU**: KV and Mamba evict independently because KV can be reconstructed from the Mamba state but not vice versa.

### 4.9 KV cache quantization

- **KIVI** — https://arxiv.org/abs/2402.02750 — 2-bit KV cache quantization
- MLX already supports `kvBits=4` and `kvBits=8` via `QuantizedKVCache`
- Community finding: **4-bit collapses on small models** (perplexity > 500 on sub-7B)
- **Stick with 8-bit on Qwen3.5-4B** unless willing to invest in TurboQuant

---

## 5. Open issues & PRs we should track

| URL | What | State |
|---|---|---|
| [mlx-lm #917](https://github.com/ml-explore/mlx-lm/pull/917) | `mx.clear_cache()` between prefill chunks; **50 GB → 28 GB** | merged 2026-02-23 — **port to Swift** |
| [mlx-lm #943](https://github.com/ml-explore/mlx-lm/pull/943) | `--prefill-step-size` CLI arg; documents speed/memory tradeoff | merged 2026-03-05 |
| [mlx-lm #1087](https://github.com/ml-explore/mlx-lm/pull/1087) | Adaptive prefill step size for long context | closed (author stepped back) — **could revive** |
| [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980) | Prefix cache reuse broken for hybrid models | open — taxonomy of failures |
| [mlx-lm #903](https://github.com/ml-explore/mlx-lm/issues/903) | Caching doesn't work for Qwen3.5; Mamba cache non-trimmable | open |
| [mlx-lm #1006](https://github.com/ml-explore/mlx-lm/pull/1006) | Per-layer checkpoint mechanism for hybrid models; 98% cache hit on Qwen3.5 | merged via #1072 — **read for design** |
| [mlx-lm PR #924](https://github.com/ml-explore/mlx-lm/pull/924) | Confirms unused arrays bloat lazy graph | merged 2026-02-23 |
| [mlx-lm #1015](https://github.com/ml-explore/mlx-lm/issues/1015) | `generate()` crashes on Metal OOM instead of recovering | open — workarounds documented |
| [mlx #3350](https://github.com/ml-explore/mlx/issues/3350) | Metal allocator pool unbounded by default | closed wontfix — **call clearCache ourselves** |
| [mlx #742](https://github.com/ml-explore/mlx/issues/742) | Foundational thread on MLX memory management | open |
| [mlx #3307](https://github.com/ml-explore/mlx/pull/3307) | Chunked SDPA with online softmax merge — fixes 65K+ context | open — **adopt when merged** |
| [mlx PR #1924](https://github.com/ml-explore/mlx/pull/1924) | Fused causal masking in SDPA | merged — already used |
| [mlx PR #1610](https://github.com/ml-explore/mlx/pull/1610) | Matrix attention kernel for q≥32 | merged — already used |
| [vllm #26201](https://github.com/vllm-project/vllm/issues/26201) | Tracking: Prefix Caching for Hybrid Models | open |

---

## 6. The spokvulcan/mlx-swift-lm fork (what's already patched, what isn't)

Tesseract vendors a fork at [`spokvulcan/mlx-swift-lm`](https://github.com/spokvulcan/mlx-swift-lm). Active feature branches as of research date:

| Branch | Purpose |
|---|---|
| `feat/paroquant-support` | ParoQuant load path. Transient ~17 GB peak during pre-rotation, optimized to ~9 GB in commit `afcf4fa0`. |
| `perf/qwen3.5-4b-decode-speed` | asyncEval checkpoints, MLP fusion, T=1 fast path. Most reverted. |
| `autoresearch/inference-speed` | Benchmark harness. **Commit `f13697b5` raised default `prefillStepSize` 512 → 1024 for +30% prefill speed.** Tesseract currently overrides to 256 in `AgentGenerateParameters`. |
| `feat/clustered-kv-cache` | KV bandwidth reduction via k-means. Helps decode, **not prefill**. |
| `gpu-only-penalties` | Sampler penalty processors GPU-resident. Merged. |
| `fix/token-ring-2d-prompt` | 2D prompt support for ring buffer. |
| `test/tesseract-integration` | Integration tests. |

**No branch in the spokvulcan fork adds `MLX.GPU.clearCache()` inside the prepare() prefill loop.** This is the missing patch — see § 7.1.

---

## 7. Patch ladder

Ranked from "trivial 3-line fix that you can ship today" to "research project that takes weeks".

### 7.1 [P0 — minimum viable, ship immediately] Add `clearCache()` between prefill chunks

**Files to patch (in our vendored fork at `Vendor/mlx-swift-lm/`):**

1. `Libraries/MLXLLM/LLMModel.swift:22-37` — the chunking loop:

```swift
public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
    -> PrepareResult
{
    let prefillStepSize = windowSize ?? 512
    var y = input.text
    while y.tokens.size > prefillStepSize {
        let input = y[.newAxis, ..<prefillStepSize]
        _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)
        y = y[prefillStepSize...]
        MLX.GPU.clearCache()    // ← ADD THIS LINE — matches mlx-lm/generate.py:278
    }
    return .tokens(y)
}
```

2. `Libraries/MLXVLM/Models/Qwen35.swift:1104-1165` — the VLM prepare path. **This one is more involved** because it currently doesn't chunk at all. Three options:
   - **Quick fix:** add explicit chunking that mirrors the LLMModel.swift loop above. Simplest.
   - **Better fix:** replace the body so it falls through to the parent `LLMModel.prepare` chunking logic.
   - **Best fix:** delete the override and rely on the inherited `LLMModel.prepare`. Verify the multimodal path still works.

3. `Libraries/MLXLMCommon/Evaluate.swift:633` — `TokenIterator.prepare`. After the prepare returns, call `MLX.GPU.clearCache()` once before `asyncEval(y.tokens)`.

**Expected impact:** Per [mlx-lm PR #917](https://github.com/ml-explore/mlx-lm/pull/917): peak prefill memory drops by ~40%. On our 27 GB crash, that's ~16 GB peak — well within the 48 GB budget even with low-power mode active.

**Risk:** Negligible. `MLX.GPU.clearCache()` returns buffers to the OS but the next allocation hits the OS instead of the cache pool. Per [mlx #3129](https://github.com/ml-explore/mlx/issues/3129) one user reported "didn't affect performance, in fact, it made the model run faster" because the OS allocator was less fragmented.

### 7.2 [P0 — defensive] Lower `Memory.cacheLimit` at startup

Currently we set `Memory.cacheLimit = 1024 MB` (Phase 8 fix). The MLX default is `gc_limit_ ≈ 0.95 × max_recommended_working_set_size`, which is ~34 GB on a 48 GB Mac. **Verify** our 1024 MB setting is actually being applied — search for `Memory.cacheLimit` writes in `LLMActor.swift` to make sure the value isn't being overridden by another code path.

### 7.3 [P0 — defensive] Reduce `set_wired_limit` from 75% to 50% (or remove)

The default in `mlx-lm` server (and possibly inherited by `mlx-swift-lm`) wires ~75% RAM. Wired memory cannot be swapped, so under low-power mode or memory pressure the process gets killed instead of degrading gracefully. References: [mlx-lm #883](https://github.com/ml-explore/mlx-lm/issues/883), [mlx #3186](https://github.com/ml-explore/mlx/issues/3186).

Search for `setWiredLimit` or `set_wired_limit` in `Vendor/mlx-swift-lm/`. If present, lower or remove. If not present (i.e., MLX defaults are not being touched), no action.

### 7.4 [P1 — soon] Tune `prefillStepSize` for the VLM hybrid path

Once `clearCache()` is in place, the chunk size becomes the speed/memory knob. Benchmark `prefillStepSize ∈ {128, 256, 512, 1024, 2048}` on a 16K-token Qwen3.5-4B-paro prefill on M3 Max 48 GB.

- **Measure:** peak `Memory.activeMemory`, peak `Memory.peakMemory`, total prefill wall-clock.
- **Expect:** the sweet spot is 1024 or 2048. The LM Studio finding suggests 8192 is too large; current 256 is too small for compute saturation.
- **Plumbing:** already in place via `AgentGenerateParameters.prefillStepSize`.

### 7.5 [P1 — soon] Replace whole-conversation cache keys with token-prefix radix tree

**Current Tesseract cache** (`HTTPPrefixCacheSpike.swift`) keys on the entire `(systemPrompt, messages, toolDigest, templateDigest)` tuple. A single character difference anywhere → full miss.

**Replace with:**
1. After `apply_chat_template` produces the token list, hash *that* token list.
2. Insert the (token-prefix, KV-cache-slice) into a radix tree (token-level, like SGLang).
3. On lookup: tokenize the new request → longest-prefix match → slice the cached KV to the match length → prefill only the divergent tail.

**Attention layers only initially.** Mamba layers re-prefill from scratch for now. For Qwen3.5 with 4× more Mamba than attention layers this is still a big win because attention prefill is the quadratic cost.

**Estimated effort:** medium. ~1 week including tests. Reuses existing `KVCacheSimple.copy()` for slicing.

**Expected impact:** hits on system prompt + tool definitions across all coding-agent requests, even when conversations diverge. For 4–8K-token system prompts that's 80%+ of prefill saved on every steady-state turn.

### 7.6 [P2 — medium-term] Marconi-style hybrid prefix caching

Implement the Marconi paper's three innovations on top of the radix tree from § 7.5:

1. **Speculative insertion:** at admission, walk the radix tree. If the new sequence creates a branch point, mark that as a Mamba checkpoint candidate.
2. **Mamba state checkpointing during prefill:** intercept the prefill loop to materialize Mamba state at the checkpoint position.
3. **FLOP-aware eviction:** score = `recency + α × (FLOPs / memory)`. Hard-code α from a one-time benchmark.
4. **Dual LRU:** attention and Mamba evicted independently.
5. **Two-pass prefill:** when the matched prefix doesn't align with a Mamba checkpoint, two-pass through the prefill to reach the exact prefix length.

**Reference implementations to study:**
- The Marconi repo: https://github.com/ruipeterpan/marconi
- SGLang's `MambaRadixCache` source

**Effort:** 2–3 weeks of focused work. This is research-level — start with a verification harness that proves logits-equivalence between (a) full prefill and (b) checkpointed-restored-then-resumed prefill.

**Expected impact** (per the Marconi paper on SWEBench): **34× hit rate**, **617 ms P95 TTFT** reduction. For a 16K coding conversation that's the difference between 3-second TTFT and 300 ms.

### 7.7 [P3 — defer] PagedAttention / batched prefill

Only worth doing if Tesseract serves >3 concurrent HTTP clients. Current target is 1–2 concurrent OpenCode sessions. Skip for now.

---

## 8. Action plan for the fresh session

**Immediate (P0, can be done in one sitting):**
1. Add `MLX.GPU.clearCache()` in 3 places (§ 7.1)
2. Verify `Memory.cacheLimit` is actually 1024 MB at runtime via the new STORE log line (§ 7.2)
3. Ensure no `set_wired_limit` call beyond 50% (§ 7.3)
4. Build Release, re-run the OpenCode subagent flow that crashed, confirm no OOM

**Soon (P1, ~1 week):**
5. Benchmark `prefillStepSize` and pick the sweet spot (§ 7.4)
6. Begin radix-tree prefix cache for attention layers (§ 7.5)

**Medium-term (P2, 2–3 weeks):**
7. Implement Marconi-style hybrid cache for Mamba layers (§ 7.6)

**Defer:**
8. PagedAttention / batched prefill (§ 7.7)

The fixes in § 7.1–7.3 should resolve the immediate crash. § 7.4–7.5 give the next 2× win. § 7.6 is the architectural investment that closes the gap with llama.cpp's prefix-cache effectiveness.

---

## 9. References

### Papers
- SARATHI — https://arxiv.org/abs/2308.16369
- Sarathi-Serve — https://arxiv.org/html/2403.02310v1
- PagedAttention — https://arxiv.org/abs/2309.06180
- FlashAttention — https://arxiv.org/abs/2205.14135
- KIVI — https://arxiv.org/abs/2402.02750
- Marconi — https://assets.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf
- SGLang — https://arxiv.org/pdf/2312.07104

### Blogs / docs
- vLLM blog — https://blog.vllm.ai/2023/06/20/vllm.html
- SGLang RadixAttention — https://lmsys.org/blog/2024-01-17-sglang/
- vLLM internals (Aleksa Gordić) — https://www.aleksagordic.com/blog/vllm
- TensorRT-LLM chunked prefill — https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
- Hybrid models in vLLM (PyTorch blog) — https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
- Hybrid models in SGLang (PyTorch blog) — https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
- Draw Things / MFA — https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- WWDC25 #298 Explore LLMs on Apple Silicon with MLX — https://developer.apple.com/videos/play/wwdc2025/298/
- MLX lazy evaluation — https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX memory management API — https://ml-explore.github.io/mlx/build/html/python/memory_management.html
- MLX `fast.scaled_dot_product_attention` — https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html

### GitHub issues / PRs (the most actionable)
- [mlx-lm #917](https://github.com/ml-explore/mlx-lm/pull/917) — **the smoking gun**, missing `clearCache()` between chunks
- [mlx-lm #943](https://github.com/ml-explore/mlx-lm/pull/943) — `--prefill-step-size` tradeoff
- [mlx-examples #931](https://github.com/ml-explore/mlx-examples/pull/931) — original chunked prefill: 8.4 → 2.6 GB
- [mlx #3350](https://github.com/ml-explore/mlx/issues/3350) — Metal allocator unbounded pool
- [mlx #742](https://github.com/ml-explore/mlx/issues/742) — foundational memory management thread
- [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980) — hybrid model prefix cache broken
- [mlx-lm #903](https://github.com/ml-explore/mlx-lm/issues/903) — Qwen3.5 caching specifically
- [mlx-lm #1006](https://github.com/ml-explore/mlx-lm/pull/1006) — checkpoint mechanism for hybrids
- [mlx PR #1924](https://github.com/ml-explore/mlx/pull/1924) — fused causal SDPA
- [mlx PR #1610](https://github.com/ml-explore/mlx/pull/1610) — matrix attention kernel
- [mlx-lm #644](https://github.com/ml-explore/mlx-lm/issues/644) — direct llama.cpp vs MLX comparison

### Repos
- MLX core — https://github.com/ml-explore/mlx
- mlx-lm Python — https://github.com/ml-explore/mlx-lm
- mlx-swift — https://github.com/ml-explore/mlx-swift
- Tesseract's pinned fork — https://github.com/spokvulcan/mlx-swift-lm
- Marconi — https://github.com/ruipeterpan/marconi
- SGLang — https://github.com/sgl-project/sglang
- vLLM — https://github.com/vllm-project/vllm
- Metal FlashAttention (Philip Turner) — https://github.com/philipturner/metal-flash-attention
- LMCache — https://github.com/LMCache/LMCache
- KIVI — https://github.com/jy-yuan/KIVI

### Local code touchpoints
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:22-37` — chunking loop (missing `clearCache`)
- `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1104-1165` — VLM prepare (no chunking at all)
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:595-651` — `TokenIterator.init` and `prepare`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:46-75` — confirms `MLXFast.scaledDotProductAttention` is used
- `tesseract/Features/Agent/LLMActor.swift:548` — crash site, our entry into MLX prefill
- `tesseract/Features/Agent/AgentGeneration.swift` — `AgentGenerateParameters.prefillStepSize: 256` (Phase 8c)
- `tesseract/Features/Server/HTTPPrefixCacheSpike.swift` — current prefix cache (whole-conversation keys, to be replaced in § 7.5)
- `docs/mlx-swift-lm-kv-cache-audit.md` — earlier audit of cache lifecycle (read this together with this doc)
