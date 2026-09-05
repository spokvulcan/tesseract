# DFlash2 performance investigation — 2026-09-04

The reproduced summary slowdown is dominated by low accepted tokens per
verification round. The authors' Python/MLX implementation also accepts few
tokens on the identical prompt. This investigation found small removable
CPU/GPU costs, but has not established a large end-to-end speedup from those
code changes. The benchmark loop is substantially cheaper now.

## Reproduction and comparison with the authors

Hardware: Apple M3 Max, 48 GiB unified memory. Local target: `mlx-community/Qwen3.8-27B-4bit`; drafter:
`incoai/Qwen3.8-27B-DFlash2`, quantized to 4 bits at load time. Measurements
use Release, greedy generation, and serialized GPU work. The initial
Tesseract commit was `3cb640abc427ef75d044488fcce03072ed84b0e6`; the vendored
Swift LM commit was `6d251c6cedff52ebdf9872129b32bebb5b2f9c32`.
This local vendor includes work beyond the linked PR; results describe this
checkout, not an independently rebuilt PR head.

`summary.txt` freezes the original live-docs prompt. Its SHA-256 is
`cd4da0882e8151f57b9e9f64dc465a7618728ddc96ecd6fd31656f22ab3d7750`.
Both implementations reported 5,976 formatted input tokens and generated
192 tokens, with block size 8:

| Implementation | Accepted / proposed draft tokens | Rounds | Generated tokens / round |
| --- | ---: | ---: | ---: |
| Swift, before changes | 115 / 532 | 76 | 2.53 |
| Authors' Python/MLX reference | 111 / 552 | 80 | 2.40 |

Swift decoded at 28.7 tok/s in the saved baseline; earlier AR runs measured
19.6–21.2 tok/s. With only about 2.5 output tokens per round, a roughly
88 ms draft/verify round yields about 28 tok/s. Improving a small component
of that round cannot by itself produce a 2× end-to-end gain.

The reference was run offline with `research/acceptance_probe.py canon-chat`
against the local authors' implementation (`research/dflash`, commit
`07ebd93`). Its output is saved in `results/2026-09-04/python-reference.log`.
Its throughput includes prefill and is **not** a decode-speed comparison.
The generated first 192 tokens are planning text in both implementations.

The [authors' blog](https://inco.ai/blog/dflash2/) reports an MT-Bench mean
acceptance length of 4.10 for Qwen3.8-27B, with block size 8. Their
[model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) measures
throughput on an H200 using SGLang/FlashAttention 3, the official target model,
temperature 1, top-p 0.95, top-k 20, and up to 4,096 generated tokens.
These dataset averages and hardware/runtime settings do not predict a
particular 192-token greedy continuation on a quantized Mac model.
The blog's oMLX setup also recommends runtime block size 5.

## Implemented changes

* Recurrent-state replay no longer reads queries, calculates discarded
  token outputs, reduces those outputs, or allocates/stores them. It retains
  the existing state arithmetic and the general fallback for unusual key
  dimensions. The new test compares every state element exactly for accepted
  prefixes 0–8, three input dtypes, and key dimensions 32, 128 and 33.
* Compiled drafter segments declare the layers and rotary state they actually
  read, instead of passing the entire drafter's state to every segment.
  Context projection likewise declares only its dependencies. A compiled vs
  eager test updates weights after traces exist and checks subsequent proposals.
* The benchmark avoids loading a second automatic drafter, supports a single
  DFlash pass, reuses a Release binary on request, shares loaded weights across
  sequential repeats, and prevents overlapping benchmark scripts.
* Saved JSON contains complete generated tokens without an additional GPU
  synchronization. Compare those with a recorded baseline on every experiment;
  use `--bench-check` periodically for a fresh AR comparison. The old check only
  compared eight tokens and could silently miss later divergence.

The isolated recurrent replay probe measured approximately 13–19% less time
for that component across the two probe runs. In the final run, 48 dependent
replays took 3.116 → 2.525 ms for prefix length 1, 2.416 → 2.076 ms for
length 3, and 2.494 → 2.114 ms for length 8. These include host dispatch.
See `replay_microbench.py` and the saved microbenchmark log.
Repeated warm short-travel measurements remained around 54 tok/s before and
after. That end-to-end difference is within run-to-run noise; do not advertise
the component gain as an overall generation gain.

## Precision experiments

The frozen short travel fixture has SHA-256
`d80ce7731b3321c8cee9786461491731dd81e0e834ec25b121f6b95e628b4534`
and 82 formatted tokens. It differs from the unspecified travel prompt in
the PR table. All rows below generated 192 tokens with width 8:

| Drafter policy | Decode tok/s, individual runs | Accepted / proposed per run |
| --- | --- | ---: |
| Existing 4-bit policy, updated code | 54.3, 54.0, 53.7 | 144 / 357 |
| 8-bit context projection and selector | 53.2, 53.9 | 143 / 357 |
| All 8-bit | 50.0, 51.0 | 141 / 371 |
| Unquantized | 43.1, 43.4 | 138 / 370 |

Higher precision did not help this fixture. These policies change proposals
and potentially the verification schedule, so the rates alone do not isolate
kernel cost. They remain benchmark-only controls; production precision is
unchanged.

## Block width and output length

A 768-token travel sweep initially measured width 3 at 40.6 tok/s, width 5
at 31.1, and width 8 at 28.5. Reversing the order and repeating produced
36.0/31.0 tok/s at width 8 and 37.8/35.9 at width 3. This demonstrates a
material order/run effect; the initial 42% apparent gain is not a reliable
code-speed claim. At each width the repeated full streams matched, but
widths 3 and 8 diverged at position 38. Neither 768-token stream had reached
`</think>` or EOS. Even these longer measurements still describe planning,
not the finished blog post. Production block size remains unchanged.

## Exactness limitation found in the original implementation

The original 192-token travel AR and DFlash streams diverge at generated
position 153 (zero-based). The recurrent replay change preserved each arm's
complete original stream. At the first divergence, a diagnostic using the
same preceding tokens observed:

| Token | AR logit | Batched verification logit |
| --- | ---: | ---: |
| 357 | 20.875 | 21.0 |
| 63422 | 21.0 | 21.0 |

AR selects 63422; verification selects 357 from the tie. This is evidence of
a finite-precision difference between sequential and batched target execution.
It predates these optimizations. The new full-stream gate correctly reports
failure and the shell runner returns nonzero. Do not describe this as a
passing AR identity test, or as proof of a quality regression caused by the
new replay kernel. No approximate acceptance rule or target quantization
change was introduced. The saved original binary also diverged from AR
at position 8 on the 147-token math fixture, so this limitation is not
confined to the travel example.

## Running the cheaper loop

See [README.md](README.md) for options. After the first build:

```sh
scripts/dflash2-bench.sh --no-build \
  --bench-prompt-file "$PWD/benchmarks/dflash2/travel.txt" \
  --bench-json /tmp/dflash-after.json
python3 scripts/dflash2-compare.py /tmp/dflash-before.json \
  /tmp/dflash-after.json --require-identity
```

Use the same frozen prompt, token count, width, model and drafter policy for
a code-speed comparison. The wrapper defaults to an 82-token prose fixture, reducing prefill to
roughly 0.6 seconds. Fast mode removes five of the previous six passes
by default. For the summary fixture it also avoids five repeated prefills,
each about 32 seconds. Source changes still require rebuilding; `--no-build`
is appropriate for runtime experiments only.

## Validation and limits

The vendor suite passed: 592 XCTest cases (one skipped, zero failures), plus
741 Swift Testing cases. The narrowed trace-state change also passed its
targeted compiled/eager test (one test, zero failures, final xcresult).
Release app builds, shell syntax, diff checks,
and the before/after full-stream comparator passed. Numerical equality tests
are stronger evidence for these small code changes than noisy whole-model
rates, but do not establish universal AR equivalence.

The final summary run measured 30.1 tok/s versus the saved 28.7 baseline
(+4.8%), with identical acceptance counts and prompt tokens. This is one
before/after pair; the old report retained only eight generated tokens, so
it is not a complete summary-stream identity check.

The final full-stream before/after comparisons passed for travel and math.
Both repeated final math runs preserved all 192 original DFlash tokens and
158/249 accepted/proposed counts. Math measured 76.8–77.2 tok/s after versus
72.4 in one original run; this single-baseline difference is not a robust
performance claim. The 98-token code fixture passed full AR/DFlash identity
for all 192 generated tokens: AR 22.2 tok/s, DFlash 53.3 tok/s (2.40×).

Raw measurements are under `results/2026-09-04/`. Single prompts are diagnostic
fixtures, not substitutes for GSM8K, HumanEval or MT-Bench evaluations. The
harness generates a fixed number of tokens and does not stop at EOS; inspect
long continuations before counting them as useful answer output. Further
performance headroom is unproven, not ruled out by this investigation.

The recursive vendor formatting hook encounters parse errors in ignored
`DerivedData/SourcePackages` dependency sources. The pinned 603.0.0 formatter
completed successfully on all 480 tracked Swift files; only the four intended
vendor files remain changed.

## 2026-09-05 optimization loop (00:20-10:20)

Fast loop: `scripts/dflash2-bench.sh` on the 82-token travel fixture, 192 new
tokens, block 8, 51 rounds; ms/round is the kernel-speed ruler (acceptance is
fixed at 144/357 for bit-identical changes, and every change below except the
noted ones kept it). Per-kernel GPU time comes from the mlx fork's
`MLX_KERNEL_PROFILE=1` probe (serialized command buffers, exact GPU time per
launch). Baseline at the start of the loop: 68.3-69.8 ms/round, 54-55 tok/s.

Decode is 100% GPU-bound; the round is ~72 ms of GPU time of which the
quantized matmuls are 60 ms (84%, ~250 GB/s on the small-M tile), the rest is
~3900 launches of small kernels. Both AR and verify diverge at +153 on the
travel fixture as documented above (pre-existing, unchanged by any change here).

Landed (bit-identical acceptance, AR identity MATCH):

- GDN verify: the scan kernel gained an output-only variant (the discarded
  3.1 MB state write per layer per pass is gone), the capture carries `g`/`beta`
  instead of `a`/`b`/`A_log`/`dt_bias`/`q` (no gate recomputation at replay),
  and the replay kernel takes the int32 valid count directly (no mask array,
  no `.<` launch). 4 launches fewer per GDN layer per round (~190/round).
- GDN q/k RMS norms run as one launch over the adjacent q|k channels (one
  threadgroup per row, so bit-identical), and the head-scale scalars are built
  once instead of two scalar casts per layer per pass.
- Replay conv rows are computed once per commit and shared by the 48 layers.
- Attention verify rows are written with a dynamic slice update instead of
  `putAlong` and the bench sets `MLX_DYNSLICE_INPLACE=1` (fork C9), removing
  the whole-KV-store copy per pass (matters at long context: 21 MB per layer
  at 6K tokens).

- The drafter's selector top-16 runs as a two-stage custom kernel
  (`topKIndices`, MLXLMCommon/TopKIndices.swift: per-4096-chunk top-k, then
  top-k over the chunk winners, ordering by the float's order-preserving bit
  pattern then index — exactly the stable merge sort's tail, verified order-
  exact on random bf16 logits with and without heavy ties) instead of
  `argPartition` over the 248K vocabulary: 88 us instead of 617 us per round.
- The drafter's dynamic conv builds its dtype-cast base taps once per model
  (warmed outside the compiled traces) and shifts with one `padded` instead
  of zeros + concatenate: ~7 launches fewer per convolution, 20 per round.

Result: 67.5-67.9 ms/round (from 68.3-69.8), 55.4-55.8 tok/s.

Residual add + RMS norm as one launch (`rmsNormResidual`,
MLXLMCommon/RMSNormResidual.swift, the post-attention norm of every layer,
64 per round): a textual replica of MLX's `rms_looped` with the add folded
into the row read was NOT bitwise at first (travel re-rolled to 139/363),
because MLX compiles JIT/custom kernels with fast math off while the
package's AOT metallib is built by Xcode with `MTL_FAST_MATH=YES`. The fork
now compiles custom kernels named `fastmath_...` with fast math on, which is
what the fused kernel uses; parity is checked by `DFLASH2_RMSRES_MICROBENCH=1`
(bitwise against `x + r` then `MLXFast.rmsNorm` on both MLX geometries).
The same launch also produces the next layer's input norm at every layer
boundary (the segment bodies carry the normed input across layers), so the
64 residual `Add`s of the pre-norm sites are gone too: 66.9-67.3 ms/round.

GDN conv → silu → q/k norm → head scale as one launch
(`gatedDeltaConvNormQKV`, MLXLMCommon/GatedDeltaConvNorm.swift, 48 per
pass): one simdgroup per head row computes the four depthwise taps and the
silu for its 128 channels, and for the q|k rows MLX's `rms_single_row`
reduction and the head scale. It replaces `Convolution`, the compiled silu,
the weightless `RMSNorm` (whose strided input MLX copied inside the norm),
the two scale multiplies, and the copies of the strided `v` view inside the
GDN scan and replay kernels (their `ensure_row_contiguous`). Parity
(`DFLASH2_GDNCONV_MICROBENCH=1`, S = 1/5/8 random inputs, bitwise on q, k
and v) over a 54-variant matrix: the conv and sum-of-squares accumulation
match in every contraction form and under both fast-math regimes, and the
silu matches only when written as MLX's own bf16 expression
(`1 / (1 + exp(|x|))` in the bf16 type); per-op or single-rounding float
replicas of it differ by an ulp. The production variant is the plain JIT
compile (no `fastmath_` prefix needed). GPU time 4.5 us per layer against
24.5 us for the chain in the serialized profile. Travel 66.4-66.7 ms/round
at 144/357, code 66.6 at 140/356, math 66.6 at 158/249 (80.0 tok/s).

Attention, bitwise: the query scale (256^-0.5 = 2^-4, a power of two) is
folded into the q-norm weight at `prepare()` (`w * 2^-4` is exact and
scaling commutes with every rounding in the norm, RoPE and the score
products), so the SDPA fallback chain's `q * scale` launch goes (the fork's
`fast.cpp` also skips the multiply for a unit scale); and the output gate
multiplies the head-major views of the attention output and the gate, so
the two `Reshape` copies per attention layer (merged heads, gate) are gone.
Travel 66.2-66.6 ms/round at 144/357.

GDN scan geometry: the scan kernels take `RPT` value rows per thread (each
row its own chain, same lanes and simd reductions, so bitwise;
`DFLASH2_GDN_MICROBENCH=1` checks y and the replay state against the one-
row geometry). RPT=2 is the default now: travel 66.19/66.24 ms/round against
66.29/66.51 at RPT=1 and 66.35/66.58 at RPT=4 (the synced microbench
exaggerates the gap, 34 vs 82 us for the y-only kernel, because single
launches run at low GPU clocks; the replay kernel is write-bound and does not
move). The drafter's `logits * outputMultiplier` launch is skipped when the
multiplier is 1 (13 us per round).

Gated output norm fused (`GatedDeltaNormGate.swift`): the GDN output's
`rmsNorm` + `silu(z.f32) * x.f32` (an RMSNorm launch plus a compiled
elementwise kernel, 9.8 + 12.9 us in the synced microbench) as one launch
(4.1 us) that reads the gate straight out of the fused projection row, so
the strided `z` view is never copied. Bitwise against the compiled chain at
S = 8, 1 and 5 (`DFLASH2_GDNGATE_MICROBENCH=1`), travel 144/357 at
66.04/66.40 ms/round (RPT=2 build 66.19/66.24): a launch-count change at
the noise floor, kept for the 48 x 2 launches it removes per round.

Verify attention at short context is the unfused chain (`use_fallback`: qL 8
x gqa 6 = 48 rows exceeds the 2-pass kernel's 32-row threadgroup below the
1024-key floor): per layer a 31.7 us QK matmul, a 25 us PV matmul, softmax,
mask select — ~1.1 ms/round serialized, the largest non-QMM item left.

The 1-pass `sdpa_vector` kernel has no such cap (one threadgroup per (head,
query) row), so the fork now routes qL x gqa > 32 below the 2-pass floor to
it (`MLX_SDPA_1PASS_ANY_QL=0` restores the chain). Microbench GPU us per
SDPA on the verify shape, chain (sum of its five profiled kernels) vs
1-pass, first profiled window of each process excluded as warm-up:

| keys | chain | 1-pass |
|-----:|------:|-------:|
| 64   | 45    | 27     |
| 128  | 54    | 33     |
| 256  | 42    | 44     |
| 512  | 108   | 67     |

The 1-pass kernel is also nearer the f32 reference (max error 0.001-0.002
vs 0.004-0.007). Model, single runs: travel 140/356 at 65.6-65.9 ms/round
(chain 144/357 at 66.0-66.4), code 140/356 at 65.5-66.0 (66.6), math
160/235 at 66.1 (158/249 at 66.6) — 57.4 / 57.5 / 85.5 tok/s. The block's
logits now come from the kernel the single-token decode uses, and the
identity check moved with it: travel full stream MATCH (was DIVERGED at
+153), code MATCH, math DIVERGED at +8 (as before). Travel's bitwise
reference is now 140/356. Lowering the 2-pass floor instead
(`MLX_SDPA_2PASS_MIN_N=64`) returns zeros at every partition count below
32: `sdpa_vector_2pass_2` merges `blocks / 32` groups of partials, so few
partitions merge nothing; at 32 partitions a 512-key block would run one
32-key block per partition. Not worth a merge-kernel change for the
512-1023 band (the b8 timings, 52/63 us at 512/768, would save ~0.3
ms/round there).

The 2-token target pass seen once per generation in the round profile is
the last round of a capped generation (`draftCount` shortens the block to
the remaining budget), not redundant work.

GDN projection row read in place (`GatedDeltaConvNorm.swift` v2 and the
scan kernels' fused gates in `GatedDelta.swift`): the conv+norm kernel now
reads the conv state rows and the q|k|v columns of the fused in_proj row
directly (no `concatenated([state, qkv])` launch per layer per pass) and
emits the replay's conv-input block and the next conv state itself (9.4 us
including those two writes against ~68 us for the ops chain in the synced
microbench); and the scan gates `g = exp(-exp(A_log) * softplus(a +
dt_bias))`, `beta = sigmoid(b)` are computed inside the scan and replay
kernels from the `a`/`b` columns of the same row (one thread per step into
a threadgroup table, written as MLX's own bf16 `LogAddExp` and `Sigmoid`
expressions, so bitwise at S = 8, 1 and 5), replacing the compiled gate
kernel and its casts per layer per pass; the capture holds the projection
row instead of `g`/`beta`. A first version computed the gates inside the
step loop and lost (30 vs 23 us per scan: the transcendentals sat on the
recurrence's latency chain); hoisted, the y kernel is 21.1 us against 23.7
for the precomputed-gate kernel. Travel 140/356 at 65.05/65.55 ms/round
(1-pass reference 65.58/65.91).

Drafter dynamic conv as one launch (`DFlash2DynamicConv.swift`): the
grouped `sum_tap (base[tap] + dyn[t, tap, group]) * x[t - tap]` ran as a
`padded` shift plus a compiled multiply-add chain per convolution (three
kernels in the serialized profile, 3.9 + 6.7 + 7.0 us); the kernel reads
the slot's taps straight out of the `[B, L, 2 * K * groups]` kernel
projection (no slot slice, and `finish` gets the whole projection instead
of a slot-1 view) and keeps the ops' order and per-op bf16 rounding,
including the multiply-by-zero arithmetic of the padded positions. Bitwise
against the eager and the compiled chain at S = 8/1/5/7 for both slots
(`DFLASH2_DCONV_MICROBENCH=1`); 3.7-5.1 us per launch. Ten convolutions per
round on the 5-layer drafter: travel 140/356 at 65.38/65.55 ms/round, at
the noise floor of the previous build (65.05/65.55).

Attention q/k norm + RoPE as one launch (`AttentionNormRope.swift`, the
target's 16 attention layers and the drafter's 5, block and context
projections): MLX's `rmsNorm` copied the strided q and k head views out of
the stacked projection row before its `rms_single_row` kernel, then the
`rope` kernel ran per tensor — four kernels per tensor pair in the
serialized profile (8.3 + 7.9 + 10.0 + 8.9 us) against 5.1 us for the
fused launch, which reads the heads straight from the row (the target's
q|gate interleave by a head stride of 2 x 256), reduces exactly as
`rms_single_row` does (4 values per thread, simd sums, the cross-simd
table) and rotates the first 64 dims from the rounded normed values with
the `rope` kernel's own theta (`exp2(-d * log2(base))`, `fast::cos/sin`).
Parity (`DFLASH2_NORMROPE_MICROBENCH=1`, S = 8/1/5 x offsets 0/1234/77777
on the target layout): bitwise only as the plain `x1 * cos - x2 * sin`
expression compiled with fast math (`fastmath_` prefix: `metal::exp2` is
the fast one in the AOT metallib); explicit fma orders and the
contraction-off form differ by an ulp on a few queries, as does the plain
form without fast math. Together with the next two items: travel 140/356
at 65.09/65.13 ms/round (from 65.38/65.55).

Selector greedy walk as one launch (`DFlash2GreedyWalk.swift`): the seven
positions' gather → bf16 add → argmax → gather ran as ~28 launches per
round inside the selector trace (the profile's 7 `ArgReduce` at 11 us and
14 `GatherAxis`); one simdgroup now walks the path with lane j holding
candidate j, the same bf16 add and MLX's argmax order (highest score,
lowest index on ties). 40 random and heavily tied seeds match the eager and
the compiled loop; 7.0 us per launch.

Replay conv state from the replay launch (`GatedDelta.swift`
`.stateAfterValid(conv: true)`): the commit's conv state — rows
`validCount ..< validCount + K - 1` of the capture's conv input block —
was a `takeAlong` (4.8 us) plus a `contiguous` per layer, then a dynamic
slice (`mlx-swift` `dynamicSlice`, 5.5 us, no better); the replay kernel's
threads now copy those 30720 elements themselves, one per thread, so the
commit is one launch per GDN layer (48 fewer per round).

Drafter residual adds folded into its norm launches (`rmsNormResidual`,
as the target already did; the segment carries the next layer's normed
input, the last one the final norm's output) and the block's K/V written
into the context cache's slack rows (`DFlash2ContextCache.withBlock`, a
dynamic slice update in place under `MLX_DYNSLICE_INPLACE`, the SDPA
reading context and block as one view) instead of a per-layer concat:
bitwise, travel 140/356 at 65.18/65.19 ms/round (noise-level at short
context; the concat copied the whole context per layer per round, ~500 MB
at 6K tokens). `MLX_DYNSLICE_INPLACE=1` had been set only by the bench
runner — the production app paid a whole-store copy per verify-row write;
`TesseractApp.init` now sets it.

Verification 04:43 (identity on all fixtures, `--bench-check`): travel
MATCH 140/356, code MATCH 140/356, math DIVERGED at +8 (as before) 160/235.
Per-round launches (fresh profile): ~890 real kernels from ~3900 at the
start of the loop; QMM 295 launches / 61 ms serialized, custom kernels 378
/ 3.7 ms, SDPA 21 / 0.8 ms, SwiGLU 69 / 0.3 ms, everything else < 0.3 ms.

Deferred GDN replay — REFUTED (05:20). The replay kernels (48 per round,
20.5 us each serialized) were folded into the next verify pass's scan: the
pass took the previous captures plus the accepted count as trace inputs,
the conv-norm kernel read its state rows out of the previous conv input
block at that offset, and a `.outputAfterReplay` scan variant stepped the
previous pass's accepted prefix from its initial state, stored that state
(the new capture's initial state) and went on with the pass. Bitwise
(travel 140/356; the microbench matched y and state for prev 8 and 3),
and the serialized profile agreed with the estimate — custom kernels 187
to 173 ms over 51 rounds, total 3390 to 3308 — but the real bench LOST
0.5 ms/round (65.61/65.82 vs 65.18/65.19). Why: MLX encodes with
`MTL::DispatchTypeConcurrent` and inserts a barrier only when a kernel
reads a buffer written since the last barrier (`device.cpp:243`, `:271`),
so the replay kernels — which depend only on the acceptance count and the
previous captures — ran concurrently with the drafter's and the next
layers' QMMs, off the critical path. The fold moved their steps onto the
critical path (the scan is a latency chain: 21.8 to 35.9 us per layer, x
48) and gave back more than the launches saved. Reverted; the variant is
in the session scratchpad. Rule for the remaining levers: the serialized
profile overstates kernels that have slack (replay, the acceptance graph),
and fusing such work into a critical-path kernel is a loss even when the
sum shrinks. Judge a fusion by what it takes OFF the chain (SwiGLU, norms,
the projections' epilogues are on it).

Bandwidth ruler (`DFLASH2_BW_MICROBENCH=1`, GPU us per launch): a 1 GiB
bf16 `sum` streams at 401 GB/s and `x + 1` (read + write) at 361 GB/s;
the 4-bit qmv at M = 1 reaches 381-395 GB/s on the verify's weight shapes,
the M = 8 `affine_qmm_mma8n16` tile 264-287 GB/s (356 us on gate_up, 190 on
down, 172 on in_proj, 2489 on lm_head). So the M = 8 QMMs — 57 ms of the
65 ms round — sit ~28% under the memory ceiling, the gap the tile study
above attributes to dequant + MMA issue; nothing new here, the study stays
closed.

1-pass SDPA at the verify shape (`DFLASH2_SDPA_MICROBENCH=1`, GPU us):
N = 16: 21, 32: 24, 64: 27, 250: 42, 512: 67, 1000: 112 — about 20 us
fixed plus 0.09 us per key; the 2-pass takes over at 1024 (87 us there,
108 at 1500, 156 at 2048, 210 at 4096). Staging four output values per
barrier pair in the merge (16 -> 4 barriers per threadgroup, identical
arithmetic) changed nothing (travel 64.81/65.17 vs 65.23/65.17), so the
fixed cost is the threadgroup structure — 192 threadgroups of 1024 threads
per launch (one per query row and head), each re-reading the 1 MB K/V —
not the barriers. Reverted.

Split-K for the low-N QMM shapes — refuted by a shape sweep before any
kernel was written (`DFLASH2_QMM_SHAPES`, M = 8, GPU us): the same 89 MB
as 5120x17408 (189 us), 10240x8704 (191) and 20480x4608 (195, 94 MB) run
at the same time per byte, and 5120x6144 (67), 6144x5120 (69), 10240x3072
(67), 12288x2560 (72) likewise — the tile's time depends on the bytes, not
on how many threadgroups the N axis yields, so more threadgroups from a
K split (with the same 8-partial sum order for bitwise output) have
nothing to recover. The sub-2048-N drafter shapes (1280 and 2048 columns,
29 and 35 us in the model, 15 launches per round) fall outside the mma8
window and re-stream from cache on the qmv path; ~0.2 ms/round at most.

1-pass SDPA structure (05:40, both REFUTED, reverted; the kernel is the
upstream loop again). The verify's `sdpa_vector` launch costs ~20 us fixed
plus 0.09 us per key (N=16: 21 us, 250: 42, 1000: 112): each simdgroup
walks its keys with three dependent device round trips per key (the mask
byte gates the key-row load; the value row loads behind the score), so the
kernel runs at the memory latency, a few GB/s. (a) A GQA-group kernel
(`sdpa_vector_gs`, GS query heads of one group per threadgroup, K/V read
once per group, GS times fewer threadgroups) dispatched only after the
eligibility test read the mask's head stride the way the argument code
does; per launch it was slower at every length (N=250: 80 us at GS=3, 58
at GS=2, against 42) and the travel round went 65.2 -> 65.7 ms (140/356
kept): the per-head chain is serial inside the simdgroup, so grouping
lengthens it. (b) A register prefetch of the next one or two keys' mask
byte, key row and value row (function constant, depth 1/2): slower in the
microbench at every length (N=250: 53 / 72 us against 43; N=1000: 147 /
204 against 112). The occupancy reading (unverified: the pipeline's
thread cap was not printed) is that the 24-48 extra 32-bit registers per
thread halve the resident 1024-thread threadgroups per core, and the
kernel is hiding latency with occupancy already, so prefetch and
occupancy trade one for one. Depth 1 also re-rolled the travel trajectory
(141/349 in 50 rounds) with the arithmetic written expression for
expression: under the AOT metallib's fast-math the compiler contracts
differently in a different loop body, so "same expressions" is not
"bitwise" for this kernel. Both are recorded so the next loop does not
retry them; the 1-pass kernel serves only N < 1024 (the first ~1000
tokens of a conversation), so its lever is small either way.

QMM v2 (scale-after-accumulate) revisited (05:48, JIT override, gate_up /
down / lm_head GPU us): shipped 356 / 188 / 2514; v2 371 / 197 / 2625; v2
with the next group's scale/bias pairs loaded one step ahead (8 more live
registers) 451 / 209 / 2736; the earlier ping-pong accumulators 404 and
persistent-G "no drain" 391. The hypothesis that the v2 epilogue pays for
exposed scale-load latency is refuted: hiding that latency costs
registers, and every added live register in this family costs more than it
hides (the v2's own +4% over the shipped tile is its c00..c11/rs
live-range). The tile is occupancy-bound on registers; the only untested
direction is a register diet of the shipped kernel (the 12 row pointers).

Per-round QMM census by shape (travel profile, per round): target gate_up
64 x 358 us + down 64 x 191 + GDN in_proj 48 x 176 + o_proj 64 x 74 +
attention qgkv 16 x 156 = ~51 ms; lm_head at M = 8 for the verify 2.47 ms
AND at M = 7 for the drafter 2.47 ms; drafter layers 5 x ~690 us + the
5-layer feature projection 282 us = ~6.2 ms. The drafter's head is the one
QMM whose exactness is optional: it feeds a top-K selector, and the
accepted tokens are the target's argmax whatever the draft was.

Drafter head over a vocabulary prefix — LANDED (06:00). The drafter's head
(the target's 248320 x 5120 lm_head at M = 7, 2.47 ms/round) only feeds a
top-K candidate selector, and accepted tokens are the target's argmax
whatever was drafted, so the drafter may predict over any subset. A BPE
vocabulary is merge-ordered: over every fixture run recorded today no
generated id exceeds 93742, 1.5-5.5% of generated ids are >= 65536, none
>= 98304. `DFLASH2_DRAFT_VOCAB` rows of the head (row slices of the
quantized weight, scales and biases, no copy) with the plain quantized
matmul, exact-class guarded (a rotated head must keep its own forward):

| prefix | travel ms/round (140/356) | code | math (160/235) |
|---|---|---|---|
| full 248320 | 64.42 / 64.83 | 67.76 / 72.59 | 65.96 / 68.85 |
| 131072 | 63.64 / 67.43 | | |
| 98304 | 63.08 / 63.32 | 65.47 / 69.01 | 63.44 / 64.43 |
| 65536 | 65.10 / 68.50 (139/361, 52 rounds) | | |

98304 is the default now (-1.3 to -2.5 ms/round, +2-4%, acceptance
identical on all three fixtures; the second run of a pair drifts up by
several ms at this point of the day — the box is warm — so read the first
run). 65536 lost one draftable token and re-rolled. Guard for text past
the prefix: the iterator hands each round's committed tokens to the
drafter (`observeCommitted`, a defaulted protocol method); a decayed miss
count (x0.75 per round) at or above 1.5 — two out-of-prefix commits within
a few rounds, never one isolated control token — puts the drafter on the
full head for 64 rounds. `DFLASH2_DRAFT_VOCAB=0` restores the full head.
Not tried: a prefix plus the special-token tail as a second launch (needs
a candidate-id remap inside the selector's compiled walk).

Verification 06:08 (vocabulary-prefix default, adaptive widen, identity
checked): travel 63.54 / 63.76 ms/round 140/356 MATCH, code 63.58 / 69.01
140/356 MATCH, math 70.42 / 74.22 160/235 DIVERGED at +8 (expected). The
math pair ran 7 ms/round slower than the same build's 63.44 / 64.43 twenty
minutes earlier and code's second run drifted 5 ms: the box throttles
intermittently by now, so compare first runs and interleave A/B. The GDN
scan microbench printed 3.5x its normal numbers once (79 us for the 21 us
y-scan) right after a QMM microbench and was normal on the rerun: treat a
lone microbench reading as suspect, rerun before concluding.

Small-M QMM register diet — LANDED (06:10, `affine_qmm_mma8n16` in the
fork checkout's quantized.h and the JIT preamble copy): one packed-weight
pointer and one scale/bias pointer pair per lane with the other three
columns indexed off them (row strides kw, kg) and the activation words
loaded inside the kt loop. Interleaved base / variant pairs, GPU us:
gate_up 356.1 / 345.9 and 356.2 / 343.6; down 187.9 / 183.7 and 188.0 /
184.3; lm_head 2516 / 2402 and 2519 / 2399. Same arithmetic in the same
order, so bitwise (identity run follows). Either half alone did not
measure as a win earlier (pointer diet alone 384, activations alone "within
noise"); the register allocator responds to the pair. Loading the packed
words per kt as scalars instead (e46) lost 30%.

Register diet landed and verified (06:13): built-in preamble 351 / 185 /
2397 us; travel 62.64 ms/round 140/356 identity MATCH (run 1 drifted to
64.93), code 62.54 / 65.31 140/356. Then the next diet step, interleaved
pairs (base = the landed kernel): (e50) the four per-column `nA < N`
guards around the packed-word and scale loads compiled out — gate_up
343.7 / 344.5 -> 318.0 / 318.0, down 184.8 / 184.3 -> 170.1 / 170.7,
lm_head 2430 / 2406 -> 2222 / 2214 (-7.6 to -8.3%); (e52) `extract_bits`
for the nibble instead of shift-and-mask: neutral (343.9 vs 345.8, one
417 outlier); (e58) both: same as e50. The guards are dead for every shape
the model runs (all N are multiples of 16, so a 16-column tile is never
ragged) but their predicates and the branchy load sequence cost 8% of the
kernel. Landing as a `full_tiles` template variant the host selects when
N % 16 == 0; the guarded kernel stays for other shapes.

GDN scan operand prefetch (06:18, refuted): loading the next step's
k/q/v words one step ahead inside the scan loop (arithmetic untouched,
`GDN_SCAN_PREFETCH=1` variant). Isolated kernel, three interleaved pairs on
an idle GPU: state-after 15.3 / 15.3 / 15.7 -> 19.2 / 18.9 / 19.6 us,
y-scan 21.1 / 21.3 / 21.1 -> 24.6 / 24.6 / 24.1 us — slower on every pair.
In-model travel 62.05 / 61.68 (pf) vs 62.42 / 63.08 ms/round, identity
MATCH, i.e. inside the run-to-run band. The scan's live set is already at
the occupancy edge; the extra in-flight registers cost more than the
latency they hide. Reverted. (The first two microbench pairs of this A/B
read 40-90 us on both arms — the transient-reading caveat again; only the
rerun on an idle GPU is quoted.)

Branchless full-tile QMM — LANDED (06:24). `affine_qmm_mma8n16` gained a
`full_tiles` template parameter; the host picks it when N % 16 == 0 (kname
suffix `_ft_1`, every shape this model runs) and the four per-column range
guards on the weight/scale loads compile out; ragged N keeps the guarded
kernel (`_ft_0`). Built-in kernel: gate_up 317.6, down 171.9, lm_head
2221 us (from 351 / 185 / 2397). Travel 58.18 / 56.97 ms/round 140/356
identity MATCH (from 62.64), code 58.44 / 61.38 140/356 MATCH, math 65.60
/ 68.61 160/235 DIVERGED +8 as before (math ran third in the sequence;
re-measure cool). -4.4 ms/round on travel, ~-7%.

Three more tile shapes after e50, each interleaved with the landed kernel
(gate_up / down / lm_head us): (e60) one cooperative uint4 load per lane
(the simdgroup's whole 512-byte step) with the fragment nibbles fetched
from the holding lane by `simd_shuffle` — 4 live words instead of 32 —
561 / 297 / 3930 vs 317 / 171 / 2210: +77%; the 32 shuffles per step cost
far more than the 28 registers they free (results bitwise, same maxerr).
(e63) two 32-k half-steps per group, 16 live words: 323 / 173 / 2240 vs
318 / 171 / 2218, +1.5%. (e65) `max_total_threads_per_threadgroup(256)`
on the kernel: 318 / 170 / 2207, neutral; (e66) both: neutral to worse.
So after the guard removal the tile is not register-bound any more; its
remaining gap to the bandwidth ruler (281 GB/s vs 395 for the M = 1 qmv)
is instruction issue — the 4-op dequant chain per element (extract, cvt,
fma, cvt-to-bf16) that the qmv does not pay in this form.

Load-ahead forms of the landed tile (06:31, interleaved with it; gate_up /
down / lm_head / o_proj-shape 8x5120x6144 us, base 317 / 170 / 2200 /
61): (e68) 32-k half-steps with the next half's four uint4 issued before
the current half's compute (32 live words, as the landed kernel) 329 / 180
/ 2278 / 62.7, +3.5%; (e67) the whole next group double-buffered (64 live
words) 353-461 / 186 / 2515 / 69, +10-45%; the 8-wide tile via
`MLX_QMM_MMA8_N16=0` on the same shapes 356 / 190 / 2540 / 69 — the
16-wide tile wins on the small-N shapes too (its half-count of
threadgroups is not a tail problem at N = 5120: 320 threadgroups x 8
simdgroups fill 40 cores). Any load issued ahead of its use costs more
than the latency it hides in this tile.

Two more load-side forms (06:35, interleaved, same shapes as above):
(e71) scale/bias for two groups per 4-byte load (K % 1024 == 0), 319 / 172
/ 2189 / 61.4 vs 317 / 172 / 2207 / 60.5 — neutral: the L1-hot 2-byte
scale loads cost nothing. (e72) transposed operands — the weight tile as
the MMA's A operand, so a lane's two fragment elements are adjacent
nibbles of one word: 4 uint4 loads per step instead of 8, one scale/bias
pair per row, activations as X^T by two 2-byte loads per step — 324 / 172
/ 2260 / 62.2 and drafter-head 7x98304 888 vs 868: +1-2%, results
identical (same maxerr). Halving the redundant weight requests does not
help either, so the tile is neither register-, request- nor
scale-load-bound: what is left is the dequant ALU chain against the MMA
and the DRAM latency, and the landed kernel is the study's end point
(seven shapes since e50, none faster).

Probe of the DRAM-line theory (06:36, NOT bitwise, microbench only): (e73)
the eight simdgroups take interleaved 64-k blocks (sg, sg+8, ...) so a
threadgroup-step reads 256 contiguous bytes of every row instead of eight
scattered 32-byte chunks — gate_up 313-320 (neutral), down 165 vs 170
(-3%), lm_head and o_proj neutral. Line utilisation is not the limiter
either; not pursued (a different split-K partition re-rolls the fixtures).

Cool re-measure after e50 (06:39, first fixture of the sequence each):
math 57.10 / 57.31 ms/round 160/235 (from 63.44 at 06:00 — the 65.60
above was thermal), travel 57.10 / 59.37 140/356 MATCH. Window so far:
travel 65.2 -> 57.1, math 66.0 -> 57.1, code 67.8 -> 58.4 ms/round.

(e74, 06:40) The 8-wide tile with the same diets (guards and the twelve
pointers gone, activation words loaded in the kt loop) via
`MLX_QMM_MMA8_N16=0`: gate_up 331 / 332, down 182 / 185 vs the 16-wide
318 / 173 — the 8-wide tile's extra A traffic costs more than its doubled
threadgroup count returns even on N = 5120. The 16-wide branchless tile
stays for every shape.

Dequant-chain ablation of the landed tile (06:43, timing-only probes,
wrong results; gate_up / down / lm_head us, base 317 / 171 / 2212): raw
packed word as the B element (no dequant at all) 257 / 138 / 1798 — the
four-op chain (extract, cvt, fma, cvt-to-bf16) is 19% of the tile; extract
only 272 / 146 / 1911; extract + cvt + add + cvt (fma -> add) 374* / 164 /
2148. So a 1-op chain can recover ~14% and the rest is the MMA, the loads
and DRAM latency. The v2 tile (`MLX_QMM_MMA8_V2=1`, B = 128 + q by the
bf16 bit trick, scale and bias applied per group from the accumulated
raw products and the activation row sum) has that 1-op chain but shipped
with the pre-diet baggage (371 / 197 / 2617 / 70.8 today). With the e42 +
e50 diets applied (e78): 298 / 161 / 2088 / 57.2 vs 317 / 171 / 2202 /
60.8 — -6% on every shape. NOT bitwise with the landed tile (the group's
products accumulate exactly in fp32 before the per-column scale, instead
of each element rounding to bf16 after its own scale; maxerr vs the float
reference 1.10 vs 0.99 of 313). Landed as the v2 kernel body behind the
existing opt-in knob (host gates v2 to N % 16 == 0); default unchanged.
In-model numbers with the knob follow.

Dieted v2 in the model (06:47, `MLX_QMM_MMA8_V2=1`, v2 first in each
pair so the default numbers are hot): travel 54.93 / 54.72 ms/round
140/356 identity MATCH (default 59.79 hot; 57.10 cool at 06:39); code
63.73 / 63.76 at 141/349 (50 rounds) MATCH (default 65.67 hot); math
59.40 / 60.91 at 158/249 (36 rounds) DIVERGED +8 as the default. The
generated streams are IDENTICAL to the default's on all three fixtures
(192/192 token ids equal per fixture, bench JSON fingerprints), and the AR
stream is untouched (M = 1 runs the qmv); only the drafter's acceptance
pattern re-rolls on code and math (its proposals see the v2 rounding). So
the target's argmax path is unchanged by v2 on the fixtures — the
"not bitwise" is confined to logit low bits. Cool interleaved pairs
follow before the default is decided.

Dieted v2 — DEFAULT (06:55). Cool interleaved pairs, 45 s idle between
runs, default then v2 per fixture: code 56.91 -> 54.36 ms/round
(140/356 -> 141/349, 50 rounds), math 57.10 -> 54.49 (160/235 -> 158/249,
36 rounds), travel 57.01 -> 54.36 (140/356 both): -2.6 ms/round, -4.6%,
on every fixture, identity unchanged (travel and code MATCH, math
DIVERGED +8 as always), generated streams identical to the default's.
The host now defaults `v2_mode` to 1 for 4-bit full tiles;
`MLX_QMM_MMA8_V2=0` is the kill-switch back to the per-element tile.
References from here: travel 140/356, code 141/349, math 158/249 — same
token streams, the draft acceptance re-rolled by the drafter's rounding.

Verification 06:59 (v2 default build, `--bench-check`, 30 s idle between
fixtures): travel 55.51 / 54.39 ms/round 140/356 MATCH, code 55.79 /
54.48 141/349 MATCH, math 54.52 / 54.67 158/249 DIVERGED +8. Microbench
sanity of the default route: gate_up 302 / o_proj 57.7 / drafter head
(M = 7) 818 us vs 317 / 61.2 / 870 with `MLX_QMM_MMA8_V2=0`; M = 3 shapes
take the same kernel either way (147 us, not this tile). (e79) the v2
group row sums by a third MMA against a ones matrix instead of the
per-lane adds and two shuffles: 376-549 / 199 / 2636 / 72 us and wrong
(the scalar constructor is not an all-ones matrix) — the extra MMA per
kt costs more than the 32 adds it replaces; refuted. Remaining v2 cost
is that row-sum accumulation (about 27 us of gate_up between the 1-op
ablation's 272 and v2's 299); precomputing the per-group row sums once
per QMM (extra kernel or producer-side) is the follow-up, worth ~-1.5
ms/round net of its launch.

Row-sum probe and precompute (07:06-07:10). (e80) v2 with the group row
sum taken from a constant (no per-lane adds, converts or shuffles; wrong
results, timing only): gate_up 283 vs 300, down 154 vs 161, lm_head 1975
vs 2088, o_proj 54.8 vs 58.1 — -5% is the ceiling of moving that work
out of the tile. Landed: `affine_qmm_rowsum` (one simdgroup per (row,
group), `simd_sum`, float32 `[M, K/64]` temporary) dispatched by the host
right before v2, which reads `rowsums[am * kg + g]` per group. Every tile
threadgroup used to recompute the same 8 x 80 sums (2176 times on
gate_up). NOT bitwise with the 06:55 v2 (the sum's order differs);
streams compared against the v2 fingerprints below.

The fused q/k-norm + RoPE kernel no longer reads `RoPE`'s internals:
`PlainRoPEParameters(dims:base:traditional:scalingConfig:)` mirrors
`initializeRope`'s default/linear branch and each attention stores it from
its config at init (Qwen35, DFlash2, the bench's microbench builds it
explicitly). The mlx-swift checkout's public-RoPE-fields edit is reverted;
the vendor's remaining checkout dependency is `dynamicSlice`.

Row-sum precompute result (07:15-07:20): MEASURED, NOT KEPT. Microbench
(built-in preamble): gate_up 292.6 / down 164.0 / lm_head 2025 / o_proj
60.1 / drafter head 796 us against 318 / 172 / 2218 / 60.7 / 868 with
`MLX_QMM_MMA8_V2=0`, i.e. the -2..-3% over v2 the probe predicted. In the
model it bought nothing measurable (travel 55.30 / 54.18, code 54.01 /
55.26, math 54.15 / 54.74 ms/round against the v2 default's 55.5 / 54.4,
55.8 / 54.5, 54.5 / 54.7) AND it moved the target's argmax path: travel
identity DIVERGED at +38, code at +146 (143/335 over 48 rounds), math +8
as always; the generated streams differ from v2's on travel and code
(fingerprints: AR equal, spec not equal), math unchanged. The `simd_sum`
tree reduces the group sum in a different order than the tile's two
shuffles, and the epilogue term `(b - 128 k) * rs` cancels against `k * G`
closely enough that a one-ulp change in `rs` flips argmaxes downstream.
Reverted in full (header, preamble, host dispatch, the .metal macro); the
v2 tile again sums its own rows. Re-verified 07:26 (rebuilt, `--bench-check`):
travel 54.26 / 54.27 ms/round 140/356 MATCH, code 54.32 / 54.47 141/349
MATCH, math 54.51 / 54.55 158/249 DIVERGED +8; all three spec streams equal
the 06:55 v2 fingerprints again. Lesson for any further v2 epilogue work:
the row sum's rounding is load-bearing, so a precompute has to reproduce
the tile's exact shuffle order (a `simd_shuffle_xor` 16/8 pair over the
same lane pairing), not a library reduction.

Round census after the v2 default (07:27, travel, `MLX_KERNEL_PROFILE=1`
serialized, spec window = 51 rounds, 2975 ms GPU = 58.3 ms/round serialized
against 54.3 real, so ~4 ms of the tiny kernels overlap in the concurrent
encoder): QuantizedMatmul 2677 ms = 90%; every other primitive family is
CustomKernel 198 ms (3.9 ms/round: GDN scan 23.8 us x 46, GDN replay 21.5
us x 47 off the critical path, residual+rms norm 6.0 us x 132, conv+norm
7.1 us x 46, norm+gate 4.9 us x 46, q/k norm+RoPE 6.4 us x 15), SDPA 44.6
ms (41.6 us x 21/round), SwiGLU 17.5 ms (5.0 us x 66), DynamicSliceUpdate
11.7 ms; the lm_head argmax is 48.5 us and the drafter's greedy walk 42 us
once per round. Per round the target's M = 8 QMMs are gate_up 69 x 321 us
(313 GB/s serialized, 334 in the microbench), down 69 x 174 (288 / 306),
GDN in_proj 48 x 158 (300 / 321), out_proj 64 x 67 (263 / 287), attention
qkv 16 x 141 (294 / 314), lm_head 2099 (341), drafter head 937 (302); the
drafter's small shapes run at 116-250 GB/s (N = 1280: 31.7 us x 10, N =
2048: 32.7 x 5, N = 5120 x K = 4096: 47 x 5). The two M = 1 target passes
and the M = 53 / 81 shapes are warm-up and prefill (the `all` window), not
per-round work; the spec window holds exactly 49 M = 8 rounds, one M = 7
and the capped M = 2 last round. N sweep at fixed K (microbench, GPU us):
K = 17408 N = 2560 / 5120 / 10240 / 20480 = 84 / 162 / 325 / 607 us (298 /
311 / 309 / 331 GB/s); K = 6144 N = 5120 / 10240 / 20480 = 58 / 114 / 217
(307 / 309 / 326); K = 5120 N = 1280 / 2048 / 5120 = 42.5 / 43.1 / 82 us.
So the N = 5120 shapes (down, out_proj: 15 ms/round) run ~7% under the
wide shapes and the sub-2560-column shapes are latency-bound (a 10-group
split-K chunk per simdgroup lives ~30 us whatever N is). A bitwise fix for
either is an order-preserving cross-threadgroup split-K (one chunk per
32-thread group, partials reduced in the tile's exact order by a second
launch). REFUTED before building it (08:33, equal-byte shapes in one
microbench): 8 x 5120 x 17408 (320 threadgroups) 165.4 us against
8 x 10240 x 8704 (640, the same 50 MB) 162.6 us (-1.7%) and
8 x 5120 x 16384 (320) 152.6 against 8 x 20480 x 4096 (1280) 146.0
(-4.3%). A 2-way split buys 1.7% and pays a reduce launch plus 0.8 MB
of partials (~3%); a 4-way split breaks even. The N = 5120 shapes' gap
to the wide ones is mostly not threadgroup count.

Vendor test pass (07:29-07:50, `xcodebuild test -scheme mlx-swift-lm-Package
-only-testing:MLXLMTests` with a temporary `.package(path:)` on the
DerivedData mlx-swift checkout for `dynamicSlice`, Package.swift restored
after each run). Three loop-introduced defects surfaced, all invisible in
the model (production shapes never hit them) and all fixed: (1) MLX binds
custom-kernel inputs under 8 elements in the `constant` address space, so
every `const device T* p = input + ...` in the fused-gate scan, greedy
walk, dynamic conv, conv+norm and norm+gate kernels failed to JIT for the
unit tests' tiny arrays ("Unable to build metal library from source") and
took the xctest process down twice per run — the pointers are `auto` now;
(2) the GDN scan's two rows per thread made the grid `Dv / 2` simdgroups,
zero for the Kahan test's `Dv = 1` state, so nothing ran and the test read
garbage — `rowsPerThread` now falls back to 1 unless it divides `Dv`;
(3) the fused-projection test read `.b`/`.a` off the LLM layer's
`projectInputs`, which now returns a `GatedDeltaGateSource` (`.gates.a`,
`.gates.b`); the VLM layer keeps the plain tuple; (4) the dynamic-conv
test's naive reference indexed the `prepare` kernel as `[1, L, K, G]`,
but the fused path hands back the raw `[1, L, 2 K G]` projection row
(the module's `finish` reshapes it itself) — the test now takes the
slot-1 taps out of the row before comparing (the crash was an
Index-out-of-range in `getItemND`, found via the xctest .ips report's
faulting thread, since Swift Testing runs everything in parallel and the
log cannot say which test died). One run also hung at 0% CPU after the
JIT crash restart and was killed. The ChatSession test
`testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns` (draft-model
speculation, untouched by the loop) failed once with prompt-token count
11 vs 10 and passed in the neighbouring run — flaky, not ours. After the
fixes the XCTest phase is green: 592 tests, 1 skipped, 0 failures (run 8,
07:53). The Swift Testing phase (Xcode runs its ~800 cases in parallel in
one process) completed in two runs (233 and 21 cases passed after the
crash restarts) and hung at 0% CPU in two others with ~600 cases started
and 165 finished — an in-process parallel hang that the loop could not
attribute in the time left. The 21 DFlash2 Swift Testing cases then ran
in isolation (`-only-testing:MLXLMTests/<function>()` — the identifier
needs the parentheses; the parametrized replay case needs its argument
labels and was not matched): 20 of 20 addressed passed in 0.3 s,
including the fixed dynamic-conv test. Settled 08:36: with
`-parallel-testing-enabled NO -test-timeouts-enabled YES
-default-test-execution-time-allowance 120` the whole target passes in
one go — XCTest 592 (1 skipped, 0 failures, 59 s) and Swift Testing 743
cases in 55 suites (73 s), no timeouts — so the hang belongs to Xcode's
parallel runner over this GPU-heavy target, not to a test. Replicate CI
with that flag.

Verification 08:30 (v2 default, row-sum reverted, GPU idle 25 min before,
`--bench-check`, 30 s idle between fixtures): travel 54.24 / 54.36
ms/round 140/356 MATCH, code 54.48 / 54.57 141/349 MATCH, math 54.50 /
54.57 158/249 DIVERGED +8 (pre-existing); AR and spec streams equal the
06:55 v2 reference fingerprints on all three. Window for the loop: travel
68.3-69.8 (00:20) -> 54.2-54.4 ms/round, 54-55 -> 69.4 tok/s at the same
39.3% acceptance; code 70.4 tok/s, math 97.8 tok/s.

Loop close (08:40, ten hours from 00:20 with the last experiments
refuted rather than landed). Kept, in order of size: the v2
scale-after-accumulate small-M QMM tile as default (-2.6 ms/round), the
`full_tiles` branchless variant (-4.4), the tile register diet (-1.8),
the drafter head over the 98304-row vocabulary prefix (-1.3 to -2.5), the
1-pass verify SDPA below 1024 keys (-0.5, and identity MATCH on travel
and code), and the fused GDN / norm / RoPE / conv / selector kernels of
the first hours (~-3.2 together). Measured and reverted or refused:
GDN prefetch, deferred replay, the GQA-grouped SDPA, SDPA register
prefetch, v2 scale prefetch, the 8-wide and 32-wide tiles, every
load-ahead form of the tile, the row-sum precompute, cross-threadgroup
split-K. What is left in the round is 90% DRAM streaming through a tile
at 85-88% of the M = 1 kernel's efficiency (the 4-bit dequant chain
against MMA issue) and ~380 launches near the 5 us floor; the levers
above 1% now are non-bitwise (a different weight/scale format, a
lower-precision epilogue) or algorithmic (tree verification), i.e.
owner decisions recorded in `docs/mlx-swift-lm-fork.md`.

GPU occupancy (fork probe `MLX_CB_PROFILE=1 MLX_CB_TRACE=1`, one completed
handler per command buffer recording its GPU interval): the decode window is
99.93% GPU-busy — 2.4 ms of idle across 51 rounds (0.05 ms/round), no bubble
at the round's host sync. Every remaining gain has to come out of GPU time;
CPU-side work (about 25 empty command buffers per round, encode cost) is
fully hidden. Launch removal is worth ~3-4 us of GPU time per small kernel
(batch 1: ~190 launches -> ~0.8 ms; measured, not the serialized profiler's
number).

Measured and rejected:

- Env knobs: `MLX_MAX_OPS_PER_BUFFER` 200/400, `MLX_MAX_MB_PER_BUFFER` 400,
  `MLX_MAX_ACTIVE_TASKS` 10/20/80: none beat the defaults (the production
  default of 10 measures the same as the bench's 40 on this shape).
- `MLX_SDPA_2PASS_MIN_KL=64` (fused SDPA for the verify block at short
  context): ms/round within noise on travel/code/math; acceptance re-rolls
  (travel 144/357 -> 140/356, code unchanged, math 158/249 -> 160/235). Not
  adopted then; the knob actually routed the block to the 1-pass kernel
  (the eval dispatch keeps the 2-pass kernel above 1024 keys), which the
  later microbench and identity runs above made the default.
- The small-M QMM tile study (`results/2026-09-05/qmm-kernel-study.md`): the
  shipped tile is ALU-issue-bound at the compute/bandwidth ridge; 2-op dequant
  variants lose their gain in the per-group scale epilogue; N32 tile and
  register prefetch fall off a register cliff. No kernel change shipped.

Long context (summary fixture, 5976 prompt tokens, 21.6% acceptance): GPU
time per round is 76.8 ms of which attention is 7.4 ms (21 SDPA launches at
~354 us each, ~6x the K/V bandwidth floor) — the 2-pass MMA kernel at 256
partitions amplifies traffic through its partials (25 MB written and read
back per layer).

SDPA verify-shape ruler (`DFLASH2_SDPA_MICROBENCH=1`: q `[1, 24, 8, 256]`
against k/v `[1, 4, N, 256]` with the verify's bool mask, one SDPA per
synced command buffer, GPU time per SDPA including the merge pass; the
model-based partition sweep was unusable because sustained 6K prefills
throttle the GPU and every later run gets slower):

| N      | 256 blocks | 128 | 64  | 32  | 256 + register prefetch |
|--------|-----------:|----:|----:|----:|------------------------:|
| 1536   | 338 us     | 157 | 215 | 177 | 235 |
| 4096   | 272        | 230 | 211 | 217 | 299 |
| 8192   | 480        | 401 | 383 | 393 | 488 |
| 16384  | 803        | 710 | 683 | 737 | 817 |

At 16384 keys the kernel moves 67 MB of K/V in 683 us (98 GB/s): it is not
bandwidth bound, and the register prefetch of the staged K/V lines (kernel
variant behind `MLX_SDPA_MMA_PREFETCH=1`, function constant 27) is neutral
to slower, so it stays off. 64 partitions beat the shipped 256 at every
length in this serial regime (the verify runs its attention layers one after
another). The partition count is NOT numerics-neutral: the merge pass sums
the partials in a different order, so acceptance re-rolls (4K fixture:
123/479 at 256 vs 129/430 at 64), and a 4-run interleaved model check
(83.9/84.8 vs 77.5/86.5 ms/round) could not resolve a ~1 ms effect under
thermal drift. A finer microbench sweep (GPU us per SDPA, 40 synced
launches each) settled the policy:

| N     | 256 | 128 | 64  | 32  |
|-------|----:|----:|----:|----:|
| 1024  | 306 | 127 | 127 |  98 |
| 2048  | 232 | 145 | 126 | 132 |
| 3072  | 222 | 189 | 157 | 175 |
| 4096  | 277 | 234 | 208 | 217 |
| 6144  | 415 | 308 | 294 | 301 |
| 8192  | 481 | 548 | 378 | 391 |
| 12288 | 631 | 716 | 531 | 562 |
| 16384 | 806 | 714 | 683 | 737 |
| 32768 |1398 |1266 |1276 |1421 |

The fork's default is now 32 partitions below 2K keys and 64 above
(`MLX_SDPA_MMA_BLOCKS` still overrides). Per round that is ~3 ms at 1-2K
context (16 layers x ~180 us) and ~2 ms at 6K; the fixtures' acceptance
re-rolls once for the new merge order.

Tooling added for the loop: `DFLASH2_QMM_MICROBENCH=1` runner mode (exact
per-shape GPU time for the verify's QMM shapes, plus max error against a
dequantized f32 matmul) and the fork's `MLX_QUANTIZED_KERNEL_FILE` override
(a quantized.h path compiled at runtime instead of the built-in string, so a
kernel edit runs without rebuilding the app).
