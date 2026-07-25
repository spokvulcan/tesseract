# mlx-core inference optimization — what remains

Status after the C1–C13 loop (2026-07-24). Everything below is measured
against the state in `benchmarks/experiments-ledger.md` (the ledger's
rules, measurement protocol, and rejected-experiment list are
prerequisites — do not retry a logged failure).

## Read this first: the decode budget is closed (2026-07-25)

A MoE 8K decode token is **10.5 ms**, and it now has a complete measured
account (ledger, session 2026-07-25 (b)):

- **~5.3 ms is weight streaming** of 2002 MB/token, already running at
  ~95% of the M3 Max's 400 GB/s (`lm_head` hits 371 GB/s *in-model*;
  cold-DRAM ceiling 388 GB/s in the rig). Irreducible without changing
  quantization — i.e. without changing numerics.
- **~4.0 ms is serialization** at 972 hazard barriers (1892 dispatches,
  average wave width 1.95). Measured directly: skipping the barriers
  takes decode 95.07 → 153.80 t/s (+62%), numerically garbage.
- The GPU is **98–99% busy** throughout decode — no idle bubbles, no CPU
  slack left to convert.

So **the "~2× decode / 180–270 t/s" goal is off the table under the
zero-output-change constraint.** The floor is ~5.3 ms of streaming plus
the serialization of a ~1000-deep graph.

**Update 2026-07-25 (c): the fusion class is now exhausted too.** Three
findings close it (ledger, session (c)):

1. **The schedule is already optimal.** A per-primitive barrier census
   that also computes the ASAP critical-path depth of the dispatch DAG
   reports `criticalPathDepth=199599` against `barriers=197773` — MLX's
   dispatch order already sits at the graph's depth. Tape reordering,
   list scheduling and interleaving independent chains are **dead**; the
   only way to remove a barrier is to remove a serial link.
2. **The marginal barrier is worth ~1.4 µs, not ~5.1 µs.** C19 removed
   exactly one dispatch and one barrier per MoE layer and nothing else,
   and bought 0.058 ms/token. The 4.14 µs figure came from switching
   *all* barriers off at once, which also lets every kernel overlap — a
   super-linear effect that does not decompose.
3. Re-priced at 1.4 µs, **no remaining fusion candidate clears 1%**: the
   residual-add/RMSNorm fusions are ~0.6% each, the
   `CompiledBroadcastMultiply` sites ~1.0%, everything else less.

A custom top-k turned out *not* to change output (C18 below): the sort it
replaces is stable, so the selection order is exactly reproducible. But
its win was deleting the sort's compute, not the barriers.

Rig-to-production conversion is about **40%** (C18: 0.397 ms predicted,
0.174 ms delivered). Price in the rig, halve it, then decide whether an
in-model round is worth it.

Also corrected there: this checkpoint has **130** rotations/token (not
~511) and quantizes **only** `embed_tokens`/`lm_head` — the router and
shared expert are F16, which is 294 MB/token of the budget.

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

### 1. Full-step graph caching — DONE, and it re-aimed the whole list (C14)

**Status 2026-07-25: built, ACCEPTED, and small — read this before planning
any further CPU-side decode work.** The premise below (~25% Swift graph
build + ~40% eval walk *on the critical path*) was measured and is wrong:
`sample` on the generation thread shows it blocked in
`Scheduler::wait_for_one` **33.4% of MoE decode and 13.8% of dense decode**
(`transforms.cpp:424`, `n_active_tasks() > MAX_ACTIVE_TASKS`), i.e. the CPU
already runs ahead of the GPU. Decode is **GPU-paced**; CPU savings cannot
pay more than that slack.

C14 built the thing anyway (whole decode step as 11 traced segments for the
MoE / 9 for the dense, split only where the KV cache is written) and
measured it: Swift graph build 14.1% → 9.8%, GPU wait 33.4% → **37.3%**,
and tok/s +1.33% (MoE 128 decode), +0.67% (MoE 8K), +1.73% (dense 8K).
Accepted, but the 8K "→ 140–200 t/s" estimate below was never reachable this
way. **The 2× decode prize was GPU-serial-chain work** — fewer/faster
kernels — not graph caching. The serial-chain follow-up this entry
originally pointed at (shared-input rotation batching) was then probed and
**rejected as C15**: the C14-session estimate of ~310 batchable rotation
launches was wrong — the checkpoint has 130 rotations/token, ~50 of them
batchable, ≈ 0.48% at the measured 1.00 µs/dispatch. See the no-go list at
the bottom of this file; what actually converted afterwards was C16 and
C18, and the "Read this first" section above is where the class ended.

Original entry, kept for the reasoning:

#### Full-step graph caching — the remaining structural decode prize

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

### 2. Projection batching (QKV, in_proj_b+a) — DEAD for the rotated set

Two independent reasons, both established 2026-07-25:

1. **Structural.** PARO rotates the *activation* per projection, with
   different coefficients each. Two projections that share `x` do not
   share `rot(x)`, so an output-dim weight concat cannot serve both.
   Block-diagonal weights would double the weight bytes. This is the
   "PARO projection fusion — NO-GO, structural" ledger row; it applies
   to q/k/v (all rotated) and to `in_proj_qkv`/`in_proj_z`. The
   unrotated projections (router, shared expert) *could* concat, but
   their output dims differ and see (2).
2. **Not worth it anyway.** Extra *independent* dispatches cost only
   ~1.75–3.5 µs each (rig: one N=8192 qmv vs 16 N=512 qmvs differ by
   28 µs total), because independent kernels overlap. Fusion pays only
   where the dispatches are *serial*, and these are not.

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
fused replay assumptions from outside the E2-bitwise class,
**shared-input rotation batching (C15 — probe was bitwise-identical and
2–3.5× on the group, but the lever is only ~50 dispatches ≈ 0.48%)**,
**serial dispatch instead of concurrent+barriers (−19%)**,
**resource-scoped hazard barriers (−7% once the bookkeeping is made
sound; the +1.7% first reading was a dropped-hazard race)**,
**tape reordering / list scheduling (the schedule is already at the
graph's critical-path depth — measured, not argued)**, and **folding the
router's softmax into the top-k kernel (C19 — bitwise, but +0.64% at 128
ctx / +0.19% at 8K, and it pins far more MLX internals than it is
worth)**.

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
- **Conversion factor: one dispatch ≈ 1.00 µs in the real decode
  pipeline** (measured by appending N serial no-op kernels to the decode
  step). A 1% decode win needs ~105 dispatches removed. Size any
  "fewer kernels" idea against this *before* building it.
- **The marginal barrier is ~1.4 µs, not the ~4.14 µs the all-barriers-off
  measurement suggests** (C19 removed one barrier + one dispatch per MoE
  layer and bought 0.058 ms/token). Averages from switching a whole class
  off do not decompose into per-instance prices.
- **A barrier can be load-bearing for a consumer other than the one it is
  counted against.** C18 deleted the router's `sum`/`divide` barriers and
  `GatherQMM`'s barrier count doubled — those barriers had been
  publishing `inds` to the expert matmuls for free. Price a fusion by
  what the *consumer* still has to wait for.
- **The barrier census is reusable** —
  `benchmarks/apply-census.py apply|revert` patches the mlx checkout to
  attribute every dispatch and barrier to its primitive and to compute the
  DAG's critical-path depth. `TESS_CENSUS=1 TESS_CENSUS_OUT=<path>`, and
  run the app binary directly (`open` does not forward env).
- **Independent kernels overlap; dependent ones do not.** Fusing
  independent dispatches buys ~2–3.5 µs each; the expensive thing is a
  *serial* link. Optimize graph depth, not graph width.
- A scheduling change can read positive, token-identical, and still be
  unsound (resource barriers, above) — verify the invariant, not just
  the parity gate.
