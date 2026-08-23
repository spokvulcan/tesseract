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

## Round 5 addendum (2026-08-22)

Tokens-per-round was re-litigated to closure first (ledger R20-R21):
out-of-width chains, S>8 trees (an mma16 kernel was built and banked —
it closes the M in [9,16] QMM dispatch hole 148 -> 85.8 ms/pass — but
rows 9-16 still price at +31 ms against <= +0.9 tok/round from tree
shapes), offline walk policies (Viterbi/beam/lookahead <= greedy; the
drafter's deep scores are miscalibrated), and online n-gram forcing
(46.8% -> 45.5%; converting an early death phase-shifts the round and
the deep chain dies earlier). The S=8 chain is content-capped at ~4.3
tok/round; the program is round time.

Round-time levers landed (all identity MATCH, acceptance byte-stable):

- **Compiled selector walk** per width + **mma8 N-gate 2048**
  (draft_kv joins the mma8 window).
- **Rope-in-trace** (R23): the segment traces take the rope offset as a
  `[1]` array input (`mlx_fast_rope_dynamic`), so the eager attention
  boundary is just KV-append + SDPA. bs8f 33.2 -> 38.1 same-thermal.
- **affine_qmm_mma8n16** (R24): 16-wide-N mma8 tile — two adjacent
  8-column tiles share the A fragments, doubling per-lane loads in
  flight in a latency-bound family. Census M=8 full-pop 59.59 -> 54.68
  ms/pass; live bs8f 43.7 @ AR 21.1 (2.07x).
- **sdpa_vector_2pass_1_mma** (R25): GQA-packed MMA verify SDPA — all
  48 q-rows of a KV head in one threadgroup, split-D halves exchanging
  partial scores, uint4-staged K/V (fragment gathers straight from
  device are 1.5-2x slower — measured, twice). In-probe 9.84 -> 6.37
  ms/pass; live it needed 256 partitions (1024 threadgroups) — the
  probe overlaps 16 independent ops while the live verify runs them
  serially between dependent segments, so the kernel must fill the
  machine alone. +0.8 tok/s live at blocks=256. The raw-stream probe
  (`kvstream`, 247 GB/s vs the kernels' 62-95) shows ~2.5 ms/pass of
  physics headroom remains, gated by serial fill, not bandwidth.

Both kernels dispatch by default (`MLX_QMM_MMA8_N16=0` /
`MLX_SDPA_MMA=0` are the kill-switches). Canonical cooled set (R26):
bs8f 39.9 (1.90x), bs8 adaptive **40.2 (1.91x, now the best arm)**,
identity MATCH everywhere. Round time improved 106.8 -> 104.5 ms; on
the bs8f arm the SDPA-MMA reduction order flipped near-tie drafts one
extra round per run on this prompt (4.27 -> 4.17 tok/round), cancelling
the headline — prompt luck, not quality; the identity gate holds. The
60 tok/s goal remains out of reach inside the S=8 chain: the R20-R21
program closed every selector-side tokens-per-round route, so past the
remaining ~53 tok/s of measured round-time pools, the lever is drafter
quality (a training project).

## Round 5c addendum (2026-08-22, late): pipelined round construction

A zero-perturbation host timeline (`DFLASH2_HOST_PROFILE=1`, ledger
R32) reattributed the round: propose was ~11.7 ms of HOST graph
splice + schedule with the GPU idle (the draft's GPU compute is ~3 ms
— the drafter is 4-bit at load), on top of ~88 ms of host blocked in
the verify sync doing nothing. The fix (R33) restructures the round so
the NEXT round's propose is built during the CURRENT round's sync:

- The propose graph is made **accept-invariant**: the accept count is
  computed lazily on GPU (`sum(cumprod(draft == target-argmax))`), the
  bonus anchor rides a lazy take, ALL verify rows append to staging
  clones of the draft context caches (host-known RoPE positions —
  a round's verify positions don't depend on its accept outcome), and
  a lazy validity mask hides the rows past the accept count. Committed
  rows' K/V are bitwise the synchronous route's (row-wise math).
- Round order becomes: `asyncEval(packed)` (verify starts on GPU) →
  prebuild propose N+1 on the idle host → `asyncEval(prebuilt)` (draft
  queues behind verify on the GPU stream) → `eval(packed)` → accept →
  adopt the staged caches, resolving the appended rows' validity from
  the now-known accept count. Cache compaction is valid-aware.
- Bandit width switches take effect one round late (the prebuilt width
  governs its round). Accept-log / lattice-dump / advised-selector
  modes and T>0 keep the synchronous round. Kill: `DFLASH2_PIPELINE=0`.

Canonical cooled set (R33/R34b re-run, identity MATCH all arms):
ar 20.9-21.0, bs3f ~32, **bs8f 44.0 (2.11x — new record, fixed-8
overtakes adaptive)**, bs8 adaptive ~41 (run-to-run canonical variance
~±0.5). Round ~104.5 -> ~94-96 ms.

Verify-build streaming (asyncEval every Nth boundary so the GPU chases
the splice) was built and REFUTED by a bare same-process A/B — off
47.7 vs stride-3 45.0: buffer fragmentation beats the overlap once the
round is pipelined; it stays probe-only (`DFLASH2_VERIFY_STREAM`).
Measurement lesson recorded in the ledger (R34): AR-ratio thermal
normalization is invalid across thermal states — same-process A/B or
the cooled canonical only. Unbuilt stage 2: pipelining the verify
build (~7 ms) and reconcile (~2.9 ms) needs the TARGET cache rollback
made accept-invariant — projected floor ~88-90 ms (~46-48 tok/s).

## Round 5d addendum — stage 2: the accept-invariant verify (R36)

Stage 2 is built: the NEXT round's whole verify pass (and its packed
accept transfer) is constructed inside the current round's sync window
and scheduled behind the in-flight verify, so the GPU never drains at
the round seam. Every accept-dependent input rides lazily:

- **Tokens**: prebuilt draft outputs + a lazy bonus anchor.
- **RoPE**: the traces already take the offset as a `[1]` array —
  the lazy start position drops in unchanged.
- **GDN initial states**: stage-2a's masked full-width replay (the
  fused kernel's mask branch is an exact identity step, so the replay
  equals the accepted-prefix replay bitwise for every outcome, and
  equals the committed state on full acceptance).
- **Target KV**: rows written at the lazy true offset via a dynamic
  slice update (`Ops+DynamicSlice.swift` wrapper, C8). Rejected rows
  are overwritten by the next round's write; attention "trims" become
  host bookkeeping (`commitPipelined`). Opt-in in-place mode (C9,
  `MLX_DYNSLICE_INPLACE=1`) skips the functional full-store copy —
  safe because stream FIFO order + the masks below make the written
  region invisible to every earlier-encoded read.
- **SDPA visibility**: ONE lazy bool mask `col < start + row + 1`
  (`[S, worstLen]`, worstLen = committed + 2S host-known) encodes
  in-block causality, history visibility, and stale-row exclusion.
  The mma verify kernel gained a bool-mask variant (function
  constants; mlx-generated mirror updated) so the masked pass keeps
  the fast route.
- **Scheduling**: mlx-core's 10-buffer MAX_ACTIVE_TASKS cap
  re-throttles the deep pipeline (round-seam bubble, ~-6 same-window);
  the cap is now env-tunable (C10, `MLX_MAX_ACTIVE_TASKS`, lazily
  latched) and the bench harness defaults it to 40.

Escape hatches mirror the R33 pipeline: greedy-only,
`processor == nil` (the packed graph runs the logits path lazily),
accept-log/lattice/advised keep the synchronous round, a chain break
falls back cleanly, `DFLASH2_VERIFY_PREBUILD=0` kills.
`finalizeGeneration` rewinds from a per-round capture context.

Identity MATCH with bit-identical acceptance on the first build and
after every fix. **Canonical R36b (cooled, display asleep): ar 20.0,
bs3f 32.2, bs8f 46.7 (2.34x) — NEW RECORD, +2.7 over R34b — bs8
adaptive 40.5.** The stage-2 projection (~46-48) is reached; the
remaining exposed host per round is single-digit. 60 tok/s remains a
hardware generation (or a trained wider drafter), not a kernel.

**R37 addendum — compiled GDN replay.** The accept-invariant replay
(built every round under stage 2) was ~9 eager launches per GDN layer;
now one `compile` trace shared by every layer (mask-free path; eager
fallback kept; kill: `DFLASH2_COMPILED_REPLAY=0`). Bit-identical —
the trace preserves per-node dtype rounding, and the fused scan kernel
traces as an ordinary primitive. Canonical R37: **bs8f 47.2 (2.35x) —
record**, adaptive 42.4 (+1.9; launch overhead weighs more in
mixed-width rounds), 3f unchanged within spread. The S=16 two-chain
tree was re-priced under the pipelined round (break-even 19.3 ms, bill
+13-19 ms for <= +0.9 tok/round) — the wash stands a second time.

**R40/R42b addendum — same-input QMM stacking.** An op census (env-
gated dispatch counter in the mlx checkout, `MLX_OP_CENSUS=1`) showed
the verify pass dispatches ~2,663 primitives — QMM 496, because q/k/v,
gate/up, and the GDN's four in-projections all launch separately.
Same-input projections now concatenate along the output axis post-load
(128 groups; bitwise-exact per element — each output row keeps its own
K-accumulation and quant groups; kill: `DFLASH2_STACK_GATEUP=0`),
halving QMM launches to 256. The merge effect far exceeds the launch
arithmetic (fewer encode boundaries, single-grid streaming): canonical
**bs8f 47.9 (record, measured as a floor under active-use
contention)**, bs3f 34.2 — stacking finally moved the launch-bound
short-round arm.

**R44 addendum — the trajectory-sensitivity constraint.** Fusing the
GDN q/k RMS norms into the scan kernel (mathematically exact, f32
instead of eager's bf16 intermediate roundings, spec==AR MATCH by
construction) shifted one near-tie greedy token ~70 tokens into the
canonical generation; the diverged-but-equally-coherent continuation
drafts at 33.6% instead of 45.7%, costing ~13 tok/s net. Kill-switch
A/B restored the record trajectory exactly; the fusion defaults OFF
(`DFLASH2_FUSED_QKNORM=1` opts in). The binding lesson for every
future optimization: draft acceptance is a property of the exact
bitwise AR token stream, so a lever must preserve that stream
bit-for-bit (as the QMM stacking does) or be priced against an
acceptance re-roll that dwarfs per-launch savings.
