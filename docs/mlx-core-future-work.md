# mlx-core inference optimization — what remains

Status after the C1–C13 loop (2026-07-24). Everything below is measured
against the state in `benchmarks/experiments-ledger.md` (the ledger's
rules, measurement protocol, and rejected-experiment list are
prerequisites — do not retry a logged failure).

## Banked so far (all parity-gated token-identical, both PARO models)

| Metric | Before (2026-07-23, quiet) | After (C13) | Compounded A/B |
|---|---|---|---|
| MoE decode t/s @128 | 79.8–80.7 | ~107–108 | **+22%** |
| MoE decode t/s @8K | 75.6–76.4 | ~95.5–96 | **+24%** |
| MoE decode t/s @32K | 60.8–65.2 | ~67–72 (cool) | **~+12–15%** |
| MoE prefill t/s @8K | 1457.5 | ~1526–1532 | **~+4–5%** |
| MoE prefill t/s @32K | 1005–1158 | ~1100–1130 (cool) | **~+11%** |
| Dense decode t/s @128 | 108.2–108.3 | ~114.8 | ~+2% (GPU-bound) |
| Dense 8K peak | 4.18 GB | 3.56 GB | **−15%** |
| Dense 32K peak | 5.67 GB | 5.11 GB | **−10%** |
| MoE 8K/32K peak | 20.27 / 21.44 GB | 19.93 / 21.17 GB | −1.7% / −1.3% |

Cross-day absolute comparisons carry machine-state error (the ledger's
trap 2); the compounded A/B ratios are the rigorous numbers. Decode
target (+20–30%) and prefill target (+10–20%) are both met.

## Remaining opportunities, ranked by (value × probability) / effort

### 1. Full-step graph caching — the remaining structural decode prize

Decode is serial-chain-bound: ~4,400 dispatched ops/token (census in the
ledger). After C4–C13 the CPU has slack at 8K but the GPU still executes
hundreds of tiny kernels back-to-back per token. The blocks are now
compiled (C11/C12 proved the pattern is bitwise and pays +3–7%), but the
per-token graph still costs ~25% Swift graph-build + ~40% eval_impl walk
on the generation thread, plus the serial kernel chain.

A whole-decode-step compile (one traced step replayed per token) would
collapse the walk and most of the chain. Known obstacles, from this
session's analysis:

- `compile_replace` rebuilds the tape per call (O(tape) allocs) — a
  full-step tape is ~300 nodes, still ≫10× cheaper than the ~4,400-node
  walk. Fine.
- **KV-cache mutation is the blocker.** The shared `KVCache` infra
  mutates in place (`SliceUpdate`, `cache.advance`); a compiled step
  needs cache arrays as explicit inputs/outputs (the C12 GDN pattern
  generalized). Touches shared cache code + the app's HybridCache —
  the surgery is the project.
- Decode shape is stable ([1,1,H]) → one trace, replayed forever.
- Keep the sampler outside the compiled region (logit processors).
- Fusion is the E2/C11-proven bitwise class; the parity gate arbitrates.

Expected: decode 8K from ~96 t/s toward ~140–200 t/s (the bandwidth
floor for ~1.5 GB active weights/token at ~350–400 GB/s effective).
Start with a spike: pure-function decode step for ONE model, measure,
then decide.

### 2. Projection batching (QKV, in_proj_b+a) — ~1–2% decode, medium effort

q/k/v projections share one input; GDN `in_proj_b`/`in_proj_a` share
another. Concat along the output dim at load time → one GEMM instead of
3 (attention) / 2 (GDN). Per-output-element dot is output-index-
independent, so bitwise-plausible — must be probe-verified like C1
(qmv lane→data mapping per output element is preserved under output
concat; the gate decides). ~40–50 matmul dispatches/token saved.

### 3. Attention-block compile (C11/C12 pattern) — ~1% decode

Thin fusable soup (norms/rope are already primitive kernels); the win is
mostly dispatch+node count. KV cache state is the same blocker as #1's —
do #1's cache work first and this falls out.

### 4. C13 extension to axis ≤ 4096 — ~+0.3–0.5% prefill at 8K

C13's fused causal-mask+softmax engages only for kL > 4096 (the
`looped_softmax` replica). A `block_softmax`-body replica covers
kL ∈ (1024, 4096] — chunks 2–4 at 8K/32K. Same probe→port pattern;
small.

### 5. gather_qmm round 2 (occupancy) — research-grade, probe first

Post-C1 the gather_qmm kernel sits at ~40–50% of the dense-qmm anchor
(occupancy-limited at production B/E=32, not bandwidth-limited — C1
evidence). NB: the C1 ledger entry also records "the winner reaches 96%
of the anchor" from the same sweep — the two readings are at different
points (the 96% is the best-case large-B/E end, the 40–50% is the
production-shape point), but the entry does not pin the B/E of each.
**Re-establish the production-shape anchor ratio in the probe rig first**
— if the kernel is already near the anchor at B/E=32, this whole item is
dead. A different algorithm shape (persistent CTAs, different
rows-per-expert mapping) within the SAME per-element K-accumulation
order is the only legal axis (split-K changes rounding → dead). Probe
in `benchmarks/gather-sweep` before any app work. +2–4% prefill if a
geometry exists; unknown probability.

### 6. Speculative decoding for the dense model — big but needs a draft

Dense 32K decode is GPU-bound (weights + KV re-read) — no kernel-level
lever remains. Greedy-verified speculative decoding is output-identical
by construction (accepted tokens equal the target's argmax or they're
rejected). `SpeculativeDecoding.swift` exists in MLXLMCommon. Blocked
on a compatible PARO draft model; self-speculative variants change the
model (out of zero-loss scope).

### 7. M6 tokenizer path — TTFT only at very long context

0.29 s tokenize at 32K (~1% of TTFT there; seconds at 100K+). Encode-loop
optimization in swift-transformers. Deprioritized.

### 8. C6 hit path still copies the cached kernel source — micro, own A/B

The custom-kernel memo returns (kernel_name, kernel_source) by value on
every hit: multi-KB string copies per GDN-layer call that then move into
the CustomKernel primitive. Eliminating them means CustomKernel holding
`shared_ptr<const string>` members — a primitive-surface refactor, so it
needs its own measured iteration (deliberately NOT folded into the
2026-07-24 review-fix batch). Expected ≲1% decode; post-C10 rules say
spread-out CPU cuts may not convert — measure before believing.

## Dead ends (evidence in the ledger — do not retry)

Global op-cap raise (C2), gather_qmv rps geometry (C3), GPU-side
commit-regime detectors (C4/v4 — physics: MoE decode is boundary-limited,
GPU busy either way), metadata-only primitive fast path (C10 — CPU
slack), expert-weight prefetch (M8 — routing locality 2.4/8), fused
rotate+dequant+GEMM (M4 — bitwise-exact but 2× slower by qmv
threadgroup geometry; the two-kernel pipeline is the right design),
chunked/parallel GDN scan (rounding order), full-step `compile` *with*
fused replay assumptions from outside the E2-bitwise class.

## Banked meta-lessons (use them)

- **MLXFast JIT compiles with fast-math OFF** — verbatim arithmetic in a
  custom kernel reproduces bitwise output of the production kernel it
  replaces (M4/M5 probes, 14/14 + all configs IDENT). Fused-kernel
  replications are cheap to prove in the rig.
- Fusing pays only when the geometry doesn't multiply the fused prologue
  (M4) and when the eliminated traffic is real (M5: −45% on the chain).
- Post-C9, decode has CPU slack at 8K: spread-out CPU cuts don't convert;
  aim at the GPU serial chain (kernel count/latency) or commit
  boundaries.
- Probe protocol: one big lazy graph for timing, 32 disjoint input sets,
  ABBA; 32K-context metrics carry ±5–10% thermal variance — never
  verdict them on single runs (10-pair minimum).
