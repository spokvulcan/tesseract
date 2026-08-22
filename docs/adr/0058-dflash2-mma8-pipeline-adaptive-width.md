# ADR-0058 — DFlash2 round 2: mma8 verify kernel, single-sync pipeline, adaptive width

## Status

Accepted (2026-08-21, issue #441).

## Context

ADR-0057 shipped the DFlash2 port with `blockSize = 3` as the measured
optimum: the verify pass re-streamed the 11.9 GB of 4-bit weights per 5-row
`qmv_wide` tile, so width-8 blocks paid ~2.4 weight streams and went
net-negative (0.8×). The reference stacks that hit 44-70 tok/s (oMLX on M3
Ultra/M5 Max; mlx-dspark on M4 Pro) all run width 8 behind an MMA-tile
quantized matmul (avlp12's `qmm_mma4`) that is flat in M — the kernel this
stack lacked.

## Decisions

1. **`affine_qmm_mma8` (mlx fork `pin-tesseract`, 756fd8f7 + 691e42a1).** One
   8×8 `simdgroup_matrix` MMA tile per 8 output columns: each quantization
   group is read and dequantized once per threadgroup and reused by every row;
   split-K across the threadgroup's 8 simdgroups; the next iteration's
   packed-weight/scale/bias loads are prefetched into registers before the
   staging barrier so streaming loads overlap the MMA chain. The A tile reads
   x straight from device memory with per-lane fragment row-clamping (the
   Metal `simdgroup_matrix` 8×8 lane layout — the mapping documented in
   `conv.metal`'s winograd transform); no host padding, 10 KB threadgroup
   memory (3 resident threadgroups/core). Dispatch gate: affine, gs 64,
   4/8-bit, N ≥ 4096, K % 512 == 0, non-batched, bf16, M ∈ [5, 8] — M ≤ 4
   stays on `qmv_wide` (measured crossover).
   Rejected en route: `qmv_wide` with the tile cap raised to 8 (register
   pressure, 0.5-1.7× slower than mma8 at M=8); a 128-K staging chunk (18 KB
   threadgroup, occupancy loss, slower than the prefetch pipeline).
2. **Single-sync pipeline (mlx-swift-lm 53e52f8).** The iterator synced three
   times per round (anchor `.item`, draft ids, target argmax). The anchor is
   tracked host-side (the accept step already knows it), and greedy accept
   ships draft ids + target argmax in one packed D2H transfer. 3 syncs → 1.
3. **Adaptive width bandit.** The init `blockSize` is now a cap (app passes
   the checkpoint's 8). The reference's acceptance-threshold policy cannot
   express this stack's per-width cost curve (mma8 makes verify near-flat over
   5..8 while the draft pass prices width), so the objective is measured
   decode tok/s: an 8-round window per width over the {3, 4, cap} shortlist,
   settle on the argmax with 3% hysteresis, re-sweep every 24 settled windows
   for content drift. `adaptiveWidth: false` pins a width (bench arms `Nf`).
   Exploration costs ~16 rounds once per stream; verify-trace JIT per width is
   cached on the model and shared across streams.
4. **Bench output-identity gate.** Every DFlash2 bench arm prints its first-8
   token fingerprint and compares against the AR arm — the cheap always-on
   check that kernel-level numeric reordering never flips a greedy near-tie.
5. **Harness launches skip models migration** (SandboxMigration). A headless
   bench must not mutate user data, and a wedged old-container directory
   otherwise hangs the harness on the main thread before it can start (hit
   live: `contentsOfDirectory` on the retired sandbox container blocked
   indefinitely — later-reboot filesystem damage from the 2026-08-20 crash).

## Measured (M3 Max 48 GB, 6K docs-summary prompt, 192 tokens, ABBA)

Kernel probe (`/tmp/qmvprobe`, M=8 vs qmv_wide): mlp_gate_up 1.44-1.65×,
mlp_down 1.41-1.57×, lm_head 1.75-1.84×, gdn_in_proj 1.18-1.29×; flat across
M ∈ {5, 6, 8}; numerics rel-L∞ ≈ 0.008 vs per-row reference (same class as
qmv_wide). Kernel-only M=8 bandwidth ≈ 75-80% of the M=1 GEMV path — the MMA
tile pays staging+barrier overhead the warp-synchronous loop doesn't.

App bench (cool machine): AR 21.1 → bs3 **33.3 tok/s (1.58×)**, bs8-fixed
23.8 (1.13×); output identity MATCH on all arms. Validation run of the
shipping configuration (tok/s bandit under cap 8): bs3 31.4 (1.46×), bs8-fixed
25.2 (1.17×), **bs8-adaptive 28.7 (1.34×)** — the bandit settles mid-width on
this adversarial prompt and beats fixed-8 by 14%. Pre-round-2 baseline: bs3
31.0-31.3 (1.43×). Round anatomy at bs3: propose 10-12, verify 54-58, accept
0.4-0.5, reconcile 0.6-0.7 ms.

## Consequences / known limits

- The docs-summary bench prompt is acceptance-capped by content (59%→22%
  per-position decay, cross-stack identical with the Python reference): on it,
  narrow wins. On predictable content the bandit rides the cap — measured on
  the repeat-style prompt (`DFLASH2_BENCH_PROMPT=repeat`, 9.1K tokens): bs3
  acceptance 74% vs the docs prompt's decay, and the arms invert: bs8-adaptive
  32.8 tok/s > bs8-fixed 31.3 > bs3 26.5 (warm machine; ledger R7). The
  bandit exists because both regimes ship to users.
- Verify at S=8 still costs ~1.7× the S=1 forward (probe: 82-90 vs 47-48 ms):
  ~2/3 mma8 bandwidth economics, ~1/3 eager inter-segment glue (16 attention
  rope+KV+SDPA steps sit outside the compiled segments). Tracing the attention
  steps through the cache arrays (like GDN states already ride) is the next
  lever if more width-8 headroom is needed.
- 70 tok/s is M5 Max physics (614 GB/s, 1.54× this machine's 400); on M3 Max
  the realistic envelope for this workload is ~35-50 depending on content.

## Deferred (recorded, not built in round 2)

- Async draft prefetch behind the verify pass (spec story 13): the draft for
  round N+1 depends on round N's accepted hidden states, so only host-side
  bookkeeping/detok overlap is available — est. 2-5 ms/round, not the 15-20
  of a full hide.
- Start-reduced policy for >= 32K prompts (spec story 4): verify cost grows
  with context (KV reads + SDPA partials); the oMLX/ollama stacks gate wide
  blocks off at long context. Needs long-context bench arms first.
- Content-suite benching (code/JSON/prose arms) and 24K+ prompt arms: the
  two-prompt harness (docs adversarial / repeat predictable) brackets the
  regimes but does not sample between.
- Kernel parity gate as a standing test: the qmvprobe `check`/`sdpacheck`
  numerics live in the probe harness, not in a test suite.
- Width-8 fixture extension of the mock-drafter tests (they cap at block 4).
- Draft-forward compilation: the propose pass is ~150 eager dispatches
  (18-22 ms/round measured at bs8, 9K ctx); tracing the layer stack per
  block width is the remaining structural lever, blocked on trace keys for
  the variable-length context attention.

## Round 3 addendum (2026-08-21 evening)

SDPA query-axis tiling (kills the S>=6 fallback cliff: -15.5 ms at S=8) and
the direct-fragment mma8 (4-bit) shipped in mlx 84b99c29d / mlx-swift
4bcdf8b; buffered draft context cache + mask memo in mlx-swift-lm d0d40a5.
Bench (repeat prompt, warm): bs8-fixed 35.2 tok/s, 1.75x over AR 20.1;
output identity MATCH. The end-to-end parity gate moved to the seam-tie
invariant after the token-10 divergence measured as a dead tie
(20.7500 == 20.7500; ledger R12).

## Round 4 addendum (2026-08-22)

SDPA multi-query kernel `sdpa_vector_2pass_1_mq` (QPS=2: one simdgroup
carries two consecutive queries, halving K/V access issue count; per-query
accumulation order unchanged, so per-query numerics match the single-query
kernel exactly): SDPA verify pass 10.9 -> 9.6-9.8 ms at S=8/9216. The
bracket around it (ledger R16) refuted QPS=4 (register collapse, 2.4x
worse), a threadgroup-staged cooperative variant (Apple maps threadgroup
memory onto the same L1 the re-streams already hit; barriers make it 2x
worse), and a blocks x2 dispatch tweak — establishing that this kernel
family is instruction-issue-bound, not bandwidth-bound, at these shapes.
qL=1 (AR) keeps the original kernel.

Lever pricing before building (ledger R17): projection fusion is worth
only 0.47 ms/pass at M=8 (dead); the round-3 verify-compile is confirmed
engaged and worth 10.8 ms/pass. Draft-forward compilation (deferred item
above) was built with keyed per-segment traces and measured bench-neutral
— kept for parity/scaffolding, the propose cost is real GPU small-GEMM
time, not host dispatch.

Warm bench: bs8f 43.5 tok/s, identity MATCH, 2.09x over AR (ledger R18;
cooled canonical in R19). Ceiling: with every remaining measured pool
(QMM M=8 gap 12.5 ms, SDPA-above-floor ~7, post-compile glue ~10)
recovered in full, S=8 chain verification lands ~53 tok/s — the 60 tok/s
target requires more tokens per round (tree verification; S=16 needs an
mma16-class QMM kernel first — M=16 is a 148 ms/pass hole today) or
propose/verify overlap.
