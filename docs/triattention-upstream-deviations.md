# TriAttention - Upstream vs Tesseract Swift Deviations

Author: Codex
Validated on: 2026-04-17
Status: rewritten after re-checking the local Swift runtime, local tests, current upstream TriAttention repos, current `mlx-vlm`, and the shipped 4B calibration artifact.

---

## 0. Why this rewrite exists

The previous version of this note mixed three different baselines:

1. an older PyTorch layer-patch implementation,
2. an experimental MLX helper,
3. Tesseract's current Swift runtime.

That produced some useful observations, but several headline conclusions were stale or too strong. The biggest example is the "Swift silently fixes an upstream bug in the extra/MLR term" claim: that is no longer true for current upstream.

This rewrite does four things:

1. validates the original claims against current code,
2. separates "real local deviation" from "upstream has since moved the same way",
3. replaces rough qualitative statements with explicit math,
4. turns the result into an implementation and benchmarking plan.

---

## 1. Sources re-checked

### Local Tesseract / vendored Swift

- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/TriAttentionRuntime.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/TriAttentionSparseKVCache.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/QuantizedTriAttentionSparseKVCache.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/TriAttentionQwen35RuntimeTests.swift`
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/TriAttentionQuantizationTests.swift`
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/TriAttentionSparseKVCacheTests.swift`
- `tesseract/Features/Server/ModelFingerprint.swift`
- `tesseract/Features/Agent/LLMActor.swift`
- `scripts/triattention_calibrate.py`
- shipped artifact: `TriAttention/v1/0b0d94803c53b186006c100dd12a26ad1f955399ce5b52100e5f607a5fcb00fe.pt`

### Current upstream TriAttention

- PyTorch helpers:
  - `https://github.com/WeianMao/triattention/blob/main/triattention/methods/pruning_utils.py`
  - `https://github.com/WeianMao/triattention/blob/main/triattention/methods/triattention.py`
- MLX helper:
  - `https://github.com/WeianMao/triattention/blob/main/triattention/mlx/triattention_mlx.py`
- Current vLLM runtime:
  - `https://github.com/WeianMao/triattention/blob/main/triattention/vllm/core/config.py`
  - `https://github.com/WeianMao/triattention/blob/main/triattention/vllm/core/scoring.py`
  - `https://github.com/WeianMao/triattention/blob/main/triattention/vllm/core/compressor.py`

### Current `mlx-vlm`

- `https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py`
- current repo search found no native TriAttention implementation and no current README section wiring TriAttention in-tree.

### Model/config validation

- `z-lab/Qwen3.5-4B-PARO` and `z-lab/Qwen3.5-9B-PARO` `config.json` pages on Hugging Face
- prior local geometry audit in `docs/triattention-qwen35-paro-research-2026-04-16.md`

---

## 2. Executive Summary

1. Swift is aligned with current upstream on more than the previous report claimed.
   The current upstream PyTorch and vLLM scorers both use the same nonzero additive term as Swift, and current upstream vLLM defaults also use `divide_length = 128`, `offset_max_length = 65536`, `sparse_normalize_scores = true`, `window_size = 128`, and `include_prefill_in_budget = true`.

2. The real local correctness bug is snapshot restore, not the score formula.
   `HybridCacheSnapshot.restore(...)` recreates TriAttention caches without `runtimeState`, so restored sparse caches stop pruning and lose cache-level GQA mask expansion.

3. The real local behavioral deviation is prefill handling.
   Swift does not distinguish prefill from decode when deciding to prune. If prefill is long enough, pruning can start during prefill itself. Neither the current upstream PyTorch layer patch nor the MLX helper does that by default.

4. The previous "just turn on prefill pinning" recommendation was incomplete.
   Current upstream vLLM defaults to `protect_prefill = true`, but its selector explicitly refuses to prune when `protect_prefill && include_prefill_in_budget && prefill_len > budget`. Tesseract frequently has prefill longer than `12000`, so blindly copying that default would make compression unavailable on exactly the workloads we care about.

5. The shipped 4B artifact uses all attention heads of each full-attention layer.
   The artifact contains `8` full-attention layers and `16` sampled heads per such layer, for `128` sampled heads total. For 4B/9B geometry, that is full head coverage, not sparse head sampling.

6. The current score path is likely more compute-bound than the old report implied.
   The dequantization cost is real, but the bigger issue is repeated phase / `cos` / `sin` work per sampled head. With the shipped 4B artifact and `B = 12000`, one prune evaluates about `3.34e9` offset-frequency terms before any MLX kernel fusion.

7. The old `partial_rotary_factor = 1.0` assumption was false.
   PARO Qwen3.5 uses `partial_rotary_factor = 0.25`. That means only `32` of `128` complex pairs are truly RoPE-driven; the remaining `96` are phase-invariant lanes. This is live behavior, not a latent corner case.

---

## 3. Current baselines that matter

The correct comparison is no longer "Swift vs one upstream." There are now four relevant baselines:

| Aspect | PyTorch layer patch | MLX helper | vLLM runtime | Tesseract Swift |
|---|---|---|---|---|
| Integration style | patches attention forward | helper called from generation loop | runtime selector / compressor | model-internal `pruneIfNeeded` |
| Nonzero additive term | yes | different MLR formula | yes | yes |
| Score normalization default | optional, default `false` | no | default `true` | hardcoded `true` |
| Recent-window protection | not in older patch path | no | default `128` | hardcoded `128` |
| Prefill in budget | configurable | effectively yes | default `true` | hardcoded `true` |
| Prefill protection | implicit decode-only behavior | `prefill_pin = true` | default `true` | hardcoded `false` |
| Prune trigger | budget or slack trigger, decode only | budget + modulo-128, decode only | compressor state driven | `budget + 128`, any forward |
| Offset max | `65536` | not geometric-offset based | `65536` | `65536` |
| Default pruning granularity | configurable | global shared keep set | configurable | per-KV-head keep set shared across full-attn layers |

The practical takeaway is:

- Swift is no longer "freehand improvisation" on the scoring side.
- Swift is still distinctly different on lifecycle and cache-management semantics.

---

## 4. Claim validation ledger

This section explicitly validates the important claims from the previous report.

| Original claim | Verdict | Corrected statement |
|---|---|---|
| `mlx-vlm` does not implement TriAttention in `mlx_vlm/models/qwen3_5/language.py` | `True` | Current `mlx-vlm` still uses standard attention + cache types there. |
| The current upstream reference for scoring has `extra = 0` by default and Swift "fixes" it | `False` | Current upstream PyTorch and vLLM both use `extra = (q_abs_mean - q_mean_abs) * k_abs` when `disable_mlr = false`, matching Swift. |
| Swift's trig term is mathematically equivalent to upstream | `True` | This still holds. Swift implements the same real-part identity directly in Cartesian form. |
| Z-score normalization is Swift-only | `False` | Current upstream vLLM runtime defaults `sparse_normalize_scores = true`. The older PyTorch patch exposes it but defaults it off. |
| `offsetMaxLength = 65536` is a Swift-only choice | `False` | Current upstream PyTorch and vLLM also use `65536`. |
| Prefill tokens are pinned upstream by default | `Partial` | The MLX helper pins prefill. Current vLLM defaults to protecting prefill, but refuses pruning if protected prefill already exceeds budget. Current PyTorch patch prunes only on decode steps by default, which is not the same as full pinning. |
| Swift's 128-token guard is a recent-window guard, not prefill pinning | `True` | The code and tests confirm this. |
| Swift prunes at `budget + 128` | `True` | This is deliberate and test-pinned. It mirrors upstream slack-trigger behavior, not a random bug. |
| Snapshot restore drops `runtimeState` | `True` | Still true and still high severity. |
| Quantized TriAttention dequantizes the whole retained key tensor per prune | `True` | Still true. |
| The head-group clamp can silently hide bad configs | `Partial` | This is a valid future-proofing concern, but it is not live for the shipped 4B artifact because the artifact covers all runtime attention heads cleanly. |
| The zero-padded non-rotary tail is a latent bug because Qwen3.5 uses `partial_rotary_factor = 1.0` | `False` | PARO uses `0.25`, not `1.0`. The concern should be reframed as a performance/clarity question, not a latent dead path. |
| `protectPrefill = true` should simply become the default in Swift | `False` as stated | That would make long Tesseract prefills incompatible with `includePrefillInBudget = true` unless budget policy also changes. |

---

## 5. Scoring math

### 5.1 Canonical form

For one retained key position `p`, one scoring offset `o`, one frequency pair `f`, and current round start `t`:

```text
q_f = q_mean_real_f + i * q_mean_imag_f
k_f = k_real_f      + i * k_imag_f
r_f = Re(q_f * conj(k_f))
j_f = Im(q_f * conj(k_f))
theta_f = omega_f * (t - p + o)
lambda_f = freq_scale_sq_f
```

Then the current upstream PyTorch / vLLM / Swift position-dependent term is:

```text
trig(p, o) = sum_f lambda_f * (r_f * cos(theta_f) - j_f * sin(theta_f))
```

and the additive magnitude term is:

```text
extra(p) = sum_f lambda_f * (E|Q_f| - |E[Q_f]|) * |K_f|
```

The per-position score is then:

```text
score(p) = aggregate_over_offsets( trig(p, o) + extra(p) )
```

where the current Swift code uses `mean` across geometric offsets.

### 5.2 Why Swift matches current upstream

The old report was right about the trig identity but wrong about the additive term.

Current Swift:

```text
relativeReal = keyReal * qMeanReal + keyImag * qMeanImag
relativeImag = keyReal * qMeanImag - keyImag * qMeanReal
extra        = (qAbsMean - qMeanAbs) * keyAbs
```

Current upstream PyTorch:

```text
relative = q_mean_complex * conj(k_complex)
amp      = |q_mean_complex| * |k_complex|
phi      = atan2(Im(relative), Re(relative))
extra    = (q_abs_mean - q_mean_abs) * k_abs
```

These are the same score written in different coordinates:

```text
Re(relative * exp(i * theta)) = Re(relative) * cos(theta) - Im(relative) * sin(theta)
```

So the scoring change that remains controversial is not the formula. It is the policy layer around it: normalization, windows, prefill treatment, aggregation, and restore behavior.

### 5.3 The MLX helper is still a different algorithm

The experimental MLX helper remains materially different:

- it builds phase only from the query-center statistics,
- it adds a log-ratio MLR term,
- it adds a small key-norm term,
- it uses one global keep set for all layers.

So "Swift disagrees with MLX helper" is still true, but that no longer implies "Swift disagrees with canonical upstream."

---

## 6. Geometry that actually drives the runtime

### 6.1 Qwen3.5 PARO facts used below

For 4B and 9B PARO:

- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `full_attention_interval = 4`
- `partial_rotary_factor = 0.25`

Implications:

- each full-attention layer has `4` KV heads,
- each KV head has `256` dims,
- only `64` of those dims are RoPE-active,
- in the complex-pair representation that means:
  - `freqCount = 256 / 2 = 128` pairs,
  - `rotaryPairs = floor(256 * 0.25) / 2 = 32` pairs,
  - `96 / 128 = 75%` of pairs are phase-invariant.

For the shipped 4B artifact:

- sampled heads total: `128`
- full-attention layers represented: `8`
- sampled heads per represented layer: `16`

So the shipped artifact covers every attention head of every full-attention layer for 4B/9B geometry.

### 6.2 What that means for the current Swift scorer

`makeOmega(...)` currently zero-pads the non-rotary tail. For PARO that is live behavior, not dead code:

```text
omega = [32 real frequencies] + [96 zeros]
```

That does two things:

1. for RoPE inversion, zero-frequency lanes correctly act as identity lanes;
2. for scoring, those lanes contribute phase-invariant `relativeReal` mass.

This is why the old "latent bug" framing is not justified anymore. The real issue is performance:

- Swift still constructs a `phase` tensor and runs `cos` / `sin` over all `128` lanes,
- but only `32` lanes need trigonometric work,
- so PARO gives a built-in `4x` opportunity to shrink the trig path.

---

## 7. Cache and memory math

### 7.1 Dense attention bytes per token per layer

For one full-attention layer:

```text
keys   = kvHeads * headDim = 4 * 256 = 1024 elems
values = kvHeads * headDim = 4 * 256 = 1024 elems
total  = 2048 elems
```

At FP16/BF16:

```text
2048 * 2 bytes = 4096 bytes = 4 KiB / token / layer
```

### 7.2 Current dense Q8 bytes per token per layer

Using the current Tesseract Q8 accounting from the earlier cache audit:

```text
data payload                 = 2048 bytes
K metadata                   = 64 bytes
V metadata                   = 64 bytes
dense Q8 total               = 2176 bytes = 2.125 KiB / token / layer
```

### 7.3 Sparse Q8 bytes per token per layer

TriAttention adds retained positions:

```text
retainedPositions            = kvHeads * 4 bytes = 4 * 4 = 16 bytes
sparse Q8 total              = 2176 + 16 = 2192 bytes
                            = 2.140625 KiB / token / layer
```

This is only `16 / 2176 = 0.735%` above dense Q8. The old memory tables slightly underestimated sparse cache size, but only by a few MiB at `B = 12000`.

### 7.4 Whole-model plateau at `B = 12000`

Using the existing fixed SSM-state math from `docs/triattention-qwen35-paro-research-2026-04-16.md`:

- 4B / 9B fixed SSM state: `25.125 MiB`
- 27B fixed SSM state: `74.8125 MiB`

Then:

#### 4B / 9B

```text
attention plateau = 8 full-attn layers * 12000 * 2192 bytes
                  = 200.68359375 MiB

active plateau    = 25.125 + 200.68359375
                  = 225.80859375 MiB
```

If the current `.system` snapshot remains a dense 4096-token checkpoint:

```text
persistent prefix cache ~= 153.125 + 225.80859375
                         = 378.93359375 MiB
```

#### 27B

```text
attention plateau = 16 * 12000 * 2192 bytes
                  = 401.3671875 MiB

active plateau    = 74.8125 + 401.3671875
                  = 476.1796875 MiB

persistent prefix cache ~= 330.8125 + 476.1796875
                         = 806.9921875 MiB
```

### 7.5 What changed from the earlier memory note

Compared with the earlier estimate that ignored retained-position overhead:

- 4B / 9B persistent plateau moves from about `377.5 MiB` to about `378.9 MiB`
- 27B persistent plateau moves from about `804.1 MiB` to about `807.0 MiB`

So the earlier planning conclusion still holds: TriAttention materially flattens the cache curve. The correction is numerical, not strategic.

---

## 8. Pruning cadence math

### 8.1 Swift trigger

Swift currently prunes when:

```text
retainedTokenCount >= budgetTokens + divideLength
```

with:

```text
budgetTokens = 12000
divideLength = 128
threshold     = 12128
```

After each prune, retained count returns to `budgetTokens`, so once saturated the runtime prunes once every `128` newly appended tokens.

### 8.2 Why Swift can prune during prefill

Swift calls `TriAttentionQwen35Runtime.pruneIfNeeded(...)` at the end of every model forward, not only after decode steps.

Tesseract's default text prefill step size is `1024`, so the first prefill-time prune occurs when the first prefill chunk crosses `12128`:

```text
12 * 1024 = 12288 >= 12128
overshoot = 12288 - 12128 = 160 tokens
```

That means any text prefill of length `>= 12288` can already be sparse before the first generated token.

This is a real local deviation from both upstream baselines checked here:

- current upstream PyTorch layer patch only prunes on decode steps,
- current MLX helper resets state on prefill and starts compression afterward.

---

## 9. Compute-cost math

### 9.1 One prune on the shipped 4B artifact

For the shipped 4B artifact:

- full-attention layers: `8`
- sampled heads per such layer: `16`
- offsets: geometric powers of two up to `65536`, so `17` offsets
- frequency pairs: `128`
- budget: `12000`

Raw score-term count per prune:

```text
8 layers * 16 heads/layer * 12000 tokens * 17 offsets * 128 pairs
= 3,342,336,000 offset-frequency terms
```

That number is not "FLOPs" in a strict profiler sense, but it is the right order-of-magnitude for the current work being materialized before reduction.

### 9.2 Dequantization cost per prune

For quantized sparse caches, one prune dequantizes all retained keys:

```text
8 layers * 4 kvHeads * 12000 tokens * 256 dims
= 98,304,000 key values
```

This is large, but it is still much smaller than the score-term count above. The old report was directionally right that dequantization is wasteful, but it understated the bigger hotspot: repeated per-head trig work.

### 9.3 Amortized steady-state cost

Because pruning happens every `128` appended tokens once saturated:

```text
amortized score terms per appended token ~= 3,342,336,000 / 128
                                        ~= 26,112,000

amortized dequantized key values per appended token ~= 98,304,000 / 128
                                                    ~= 768,000
```

This points to the right optimization order:

1. stop recomputing phase / `cos` / `sin` per sampled head,
2. exploit `partial_rotary_factor = 0.25` so only `32` pairs use trig,
3. then optimize incremental dequantization.

---

## 10. Validated local deviations that still matter

### 10.1 High severity: restored caches lose `runtimeState`

This remains the clearest correctness bug.

On restore, TriAttention caches are reconstructed from state arrays plus `metaState`, but `runtimeState` is not reattached. Consequences:

- restored sparse caches stop pruning,
- cache-level grouped-query mask expansion is unavailable,
- only the model-level mask-expansion safety net remains.

This is the first thing to fix.

### 10.2 High severity: prefill policy is underspecified for Tesseract workloads

The old report correctly noticed that Swift does not protect prefill. But the actionable conclusion is not "flip the flag."

Current upstream vLLM makes an explicit policy choice:

```text
if protect_prefill and include_prefill_in_budget and prefill_len > budget_total:
    return None
```

That is coherent for some server settings, but it does not fit Tesseract's long agent prefills.

So Tesseract needs a product policy, not a boolean:

1. protect all prefill:
   good for correctness, bad for memory once prefill > budget
2. protect no prefill:
   good for memory, risky for system/tool retention
3. protect only the durable stable prefix:
   preserves system prompt / tools / sandbox rules while still allowing long user history to compress

Recommendation: option 3 is the correct Tesseract policy to prototype first.

### 10.3 Medium severity: Swift prunes during prefill

This is different from upstream behavior and directly interacts with the prefix-cache architecture:

- `.system` snapshots may be captured before or after sparse compaction depending on checkpoint placement,
- long warm-start suffix-prefills can already be sparse before generation starts,
- "protect the stable prefix" needs to be defined relative to checkpoint boundaries, not just absolute token zero.

### 10.4 Medium severity: aggregation semantics are intentionally different

Swift currently does:

- score per sampled head,
- z-normalize per sampled head,
- take `max` across sampled heads that map to the same KV head,
- average across full-attention layers,
- select top-k independently per KV head,
- apply the same per-KV-head keep plan to all full-attention layers.

This is not the same as:

- old global-union PyTorch behavior,
- MLX helper's one global keep set,
- current vLLM's configurable pruning modes.

But it is tested and deliberate enough that it should be treated as a product/runtime choice, not a bug report.

### 10.5 Medium severity: artifact loading is still fragile

The custom pickle decoder works for:

- the minimal fixture,
- the shipped 4B artifact,

but that is still a narrow compatibility surface. If the artifact format is fully under Tesseract's control, moving to a simpler format remains worthwhile.

### 10.6 Low severity: guard the head-group assumptions explicitly

The shipped 4B artifact is clean:

- 16 attention heads,
- 4 KV heads,
- KV group size = 4,
- sampled heads = 0...15 on each full-attention layer.

So the current clamp is not live-dangerous for that artifact. Still, `attentionHeads % kvHeads == 0` should be a hard precondition in `makeState(...)`.

---

## 11. What should no longer be treated as bugs

These points from the earlier report should be downgraded or removed:

1. "Swift fixed upstream's broken additive term."
   Not true anymore against current upstream.

2. "Z-score normalization is a Swift-only invention."
   Not true against current upstream vLLM.

3. "`offsetMaxLength = 65536` is an odd local choice."
   Not true against current upstream.

4. "Partial-rotary handling is a latent path because Qwen3.5 uses full rotary."
   False for PARO.

5. "The main efficiency problem is whole-tensor dequantization."
   Only partially true; repeated trig work is at least as important.

---

## 12. Recommended action plan

### P0 - fix restore correctness

Implement:

1. Rebuild `runtimeState` on cache restore from:
   - effective `TriAttentionConfiguration`
   - already-loaded calibration artifact
   - model geometry (`attentionHeads`, `kvHeads`, `headDim`, `partialRotaryFactor`, `ropeTheta`, `ropeType`)

Add tests for:

1. restore sparse cache,
2. append enough tokens to cross the threshold again,
3. verify pruning still happens,
4. verify cache-level grouped-query mask expansion still works after restore.

### P1 - choose a Tesseract-specific prefix protection policy

Do not just copy upstream `protect_prefill = true`.

Prototype these three modes explicitly:

1. `protectNone`
2. `protectStablePrefixOnly(length: stablePrefixBoundary)`
3. `protectAllPrefill`

Recommendation:

- benchmark all three,
- default to `protectStablePrefixOnly` if quality wins are real,
- keep `includePrefillInBudget = true` for everything outside the protected prefix.

That is the only option that plausibly preserves Tesseract's system/tool semantics without giving away the whole memory win on long histories.

### P2 - optimize the scorer in the right order

1. Precompute `deltaGrid`, `phase`, `cos`, and `sin` once per KV-head position grid, not once per sampled head.
2. Split rotary and non-rotary lanes:
   - only `32` pairs need trig on 4B/9B PARO,
   - the remaining `96` pairs are direct phase-invariant accumulations.
3. Only after that, add incremental dequantization for quantized sparse caches.

This ordering attacks the biggest known cost first.

### P3 - add a benchmark matrix that answers the real product questions

For the shipped 4B artifact, benchmark at least:

1. current Swift behavior,
2. current Swift + restore fix,
3. restore fix + `protectStablePrefixOnly`,
4. restore fix + score-path optimization.

Measure:

- decode tok/s after saturation,
- peak memory,
- cache plateau size,
- output parity / reasoning quality on long agent prompts,
- prefix-cache hit behavior after restore.

### P4 - harden the artifact boundary

Either:

1. keep `.pt` and expand decoder compatibility aggressively, or
2. switch future artifacts to a simpler format under our control.

This is lower priority than restore correctness and prefix policy.

---

## 13. Bottom line

The clean picture is:

- Swift's core score math is not the problem.
- Snapshot restore is the immediate correctness bug.
- Prefill policy is the main open product/design problem.
- The dominant performance work is probably repeated trig materialization, not only dequantization.
- The earlier recommendation to "just pin prefill" was too shallow for Tesseract's long-prefix workloads.

If we fix restore, define a stable-prefix protection policy, and reduce the trig path to the actual rotary lanes, the Swift runtime can stay structurally close to current upstream while becoming materially more correct and cheaper on real Tesseract workloads.
