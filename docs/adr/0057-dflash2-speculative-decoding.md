# ADR-0057: DFlash2 block-parallel speculative decoding for Qwen3.8-27B

- Status: Accepted
- Date: 2026-08-19
- Relates to: ADR-0056 (MTP speculative decoding cold path), ADR-0053
  (penalties via the app logit processor), ADR-0016 (ModelSession port)

## Context

ADR-0056 shipped MTP speculative decoding, but the Qwen3.5-family MTP head is
shallow (one draft per round — `maximumBlockSize = 2`), greedy-only, and
absent from every quantized redistribution (it had to be grafted back by
`scripts/graft_mtp_head.py`). The measured ceiling was ~2× at best.

DFlash2 (inco.ai, `incoai/Qwen3.8-27B-DFlash2`) is a *block-parallel* draft:
5 bidirectional sliding-window layers that consume the target's layer
5/19/33/47/61 hidden states and propose a whole block of 7 tokens per round,
walked by a bigram-selector over the top-16 candidates per position. It
supports **rejection sampling** (the selector yields a proposal distribution
`q`), so sampling presets speculate identically — MTP's greedy-only
limitation is gone. Reference numbers (H200): 4.80 mean accepted at block 8,
2.7–3.4× decode speedup.

The draft is a *separate* 3.85 GB checkpoint (no `mtp.*` graft): 81 tensors,
no embedding/head of its own — both are borrowed from the target at bind time
(works quantized: `QuantizedEmbedding`/`QuantizedLinear` are subclasses).

## Decision

Port DFlash2 into the vendored mlx-swift-lm and engage it in the app on both
generation surfaces, **preferred over MTP** when both drafters are present.

1. **Vendor (`Vendor/mlx-swift-lm`).**
   - `DFlash2DraftModel` (MLXLLM): dynamic causal conv (kernel 2, group 16),
     bidirectional block attention over a sliding context window (2047 rows),
     bigram candidate selector. `loadDFlash2Draft(from:quantization:)` — the
     production draft runs 4-bit (reference: `nn.quantize(draft, 64, 4)`).
   - Hidden capture: `Qwen35TextModelInner.forward` records per-layer outputs
     pre-norm into a `DFlash2HiddenCaptureBox`, requested per-forward through
     `LMOutput.State` keys (`dflash2CaptureLayerIdsKey`), so any capture-
     emitting target pairs with any `DFlash2DrafterModel`.
   - Deep rewind: verify passes also record every GDN layer's inputs
     (`GDNCaptureContext`); rollback replays `accepted+1` positions from the
     captured pre-state (`rollbackSpeculativeHybridCaches`) while trimmable
     attention caches just trim. This is what lifts MTP's one-token rewind
     ceiling.
   - `DFlash2SpeculativeTokenIterator`: chunked capture prefill (mirrors the
     reference `_prefill_target`, rolling 2047-row window), greedy argmax
     compare or vectorized rejection sampling, per-round cache reconcile,
     `finalizeGeneration` rewinds undrained lookahead.
2. **App.**
   - The draft is a `ModelDefinition` with the new `.draft` category,
     auto-downloaded as a dependency of `qwen3.8-27b`.
   - `DFlash2Support` mirrors `MTPDrafterSupport`: folder detection, 4-bit
     load, engagement policy. Unlike MTP there is **no greedy gate** and **no
     scratch gate** — the iterator's prefill is chunked and the draft's
     context window is fixed, so prompt length does not change the memory
     shape. The `.directLeaf` requirement stays (the iterator runs its own
     prefill, forfeiting mid-prefill boundary snapshots).
   - Server: `makeDFlash2Generation` shares the speculative cold-path body
     with MTP (`makeSpeculativeGeneration`); the engagement check prefers
     DFlash2. Agent: `startRawGeneration` engages on text-only prompts on a
     pairing target. One settings toggle covers both engines
     (`mtpSpeculationEnabled`, relabelled "Speculative Decoding").

## Parity (evidence)

Verified against the Python reference (`research/dflash`, z-lab/dflash) on
the same clean 4-bit checkpoint (`research/models/qwen3.8-27b-4bit-clean`):

- Round-0 tensors (`testDFlash2Round0TensorParity`): draft tokens **exact**,
  target verify argmax **exact**, acceptance 3/7 **exact**; hidden windows
  within bf16 noise (rel-L2 0.128, dominated by layer-61 outlier channels the
  draft's norm+fc absorbs).
- 48-token greedy trace (`testDFlash2EndToEndParityWithPythonTrace[4BitDraft]`):
  exact for 45 of 48 tokens; the late divergence is a draft-side bf16
  near-tie (both runs accept exactly 32 drafts — statistically identical
  proposals), and every emitted token remains target-verified by
  construction. Parity testing caught two real bugs (maxTokens round-width
  clamp off-by-one; `finalizeGeneration` rewind off-by-one).
- Heavyweight parity tests are `.serialized` — three parallel 27B loads
  swapped a 48 GB machine to death (the 2026-08-19 crash). Follow-up
  (2026-08-20): even serialized, packing several heavyweight tests into ONE
  `swift test` process re-accumulates fixtures (swiftpm-testing-helper peaked
  at 64.5 GB footprint and had to be killed). Run them **one test per
  process**; the memcheck guard watches `memory_pressure`, which lags the
  helper's real footprint — watch the helper itself.
- The same gates were re-run against the vendored Cmlx **with** the qmv_wide
  carry applied: all green (one test per process for round-0/draft parity,
  the e2e pair in one process — the pattern that does not accumulate
  fixtures). Kernel correctness is additionally covered by the Cmlx-level
  probe (`/tmp/qmvprobe check`: batched-M vs per-row reference, rel-L∞ ≈ 1%
  at every M) and by the app benches (sane acceptance at every block size).

## Performance

All numbers: Release build, ABBA-interleaved arms against thermal drift,
decode-only timing (`--dflash2-bench`, 192 new tokens, deterministic greedy).
Machine: M3 Max 40-core (400 GB/s), 48 GB.

The headline claim of the reference post — 70 tok/s — was measured by Inco on
an **M5 Max (614 GB/s, 1.54× this machine)** running their own oMLX stack. The
physics of this port on this machine: AR decode streams the 11.9 GB of 4-bit
weights per token in ~45 ms (82% of bandwidth peak); a verify pass must stream
the same weights once, so the speculative ceiling at ~2.2–2.7 accepted tokens
per round is ~35–50 tok/s, not 70.

**What made verify cheap.** Stock MLX dispatches a 4-bit GEMM with M < 12 rows
to the `qmv` kernel, whose grid re-streams the weights once per row — an 8-row
verify pass cost ~8 single-token streams (167 ms). The fix is a backport of
upstream mlx#3764 (`qmv_wide`, added after our v0.31.1 pin): each weight group
is decoded once and reused across up to 5 input rows (M3 Max = GPU gen 15, so
affine mode qualifies). Probe-measured on the app's own Cmlx build: M = 5 costs
1.5 streams, M = 8 costs 2.4 (vs 5/8 with plain `qmv`). Alternatives measured
and rejected: the tiled `qmm_t` (flat in M but 91 GB/s ≈ 3.5 streams), raising
the vector-tile cap to 8 (register-bound, slower), `k_lanes` 16, smaller tiles.

**Block size.** Per-draft acceptance decays with depth (59% → 50% → 41% → 36%
→ 22% at bs 3→8) while verify cost grows superlinearly past M = 5 (second
tile). Block 3 is the measured optimum; the app default is now 3
(`DFlash2Support.blockSize`).

| arm | tok/s | speedup | acceptance | verify ms | propose ms |
| --- | --- | --- | --- | --- | --- |
| AR decode | 21.8–22.1 | 1.00× | — | — | — |
| DFlash2 bs3 | **31.0–31.3** | **1.42–1.43×** | 59.1% | 57–62 | 10.2 |
| DFlash2 bs5 | 26.4–27.9 | 1.20–1.30× | 41.3% | 82–97 | 13.0 |
| DFlash2 bs8 | 14.0–17.2 | 0.67–0.80× | 22.2% | 150–162 | 20.1 |
| pre-qmv_wide bs8 (baseline) | 16.2 | 0.79× | 32.1% | 166.6 | 27 |
| pre-qmv_wide bs5 (baseline) | 18.3 | 0.89× | 36.0% | 116 | 21 |

Longer context (23.7K-token prompt, same harness): AR drops to 17.6 tok/s
(KV-stream regime) while bs3 holds 22.8 tok/s — **1.29×** (acceptance 65.5%).
A 65K attempt exceeded this 48 GB machine's envelope (swap climbed to 13.5 GB
and both arms decayed; aborted per the crash discipline) — the 256K-regime
claim of the reference post needs a machine with more RAM and bandwidth.

Cross-stack validation (`research/bench_dflash.py`, Python reference on mlx
0.32 — which ships qmv_wide natively — same 6K prompt, same machine): AR 17.0,
bs3 **31.5 (1.85×)**, bs5 23.0 (1.35×). The reference also peaks at **block
3**, confirming the choice is algorithmic, not a port artifact. Absolute
DFlash2 decode matches across the two stacks (31.5 vs 31.3 tok/s); the ratio
differs because this app's AR baseline is 26% faster than the reference's
(21.4–22.1 vs 17.0 — the Cmlx carries at work), which is exactly the regime
where speculation has less headroom left to buy.

Compiled vs eager verify: within noise at bs5 (85.7 vs 86.4 ms), compiled
ahead at bs3 (58.1 vs 61.3 ms) — the compiled segment traces stay default.
Round floor at bs3 ≈ 45 ms (weight stream) + 2.1 (LM head M=3) + ~3 (draft
body) + ~2 (accept/reconcile) ⇒ ~52 ms vs the measured ~70 ms; the residual is
draft-body dispatch (9.3 ms for 5 eager layers vs a 2.4 ms bandwidth floor —
a compiled draft body is the known next lever, not taken here) and ~13 ms of
non-GEMM verify work (GDN segment scan, capture writes, S>1 attention).

Greedy output is unchanged by construction; acceptance-rate movement between
kernel variants (e.g. bs8 32% → 22%) is near-tie argmax noise (probe: qmv_wide
vs per-row rel-L∞ ≈ 1% at every M), not a correctness change — rejected drafts
are always re-sampled from the target.

## Consequences

- The vendored Qwen35 grew capture hooks (`captureLayers`, `captureBox`,
  `gdnCapture`) threaded through the decoder layer; the compiled decode paths
  stay capture-free and bit-identical (they gate on `gdnCapture == nil`).
- Greedy output is unchanged by construction (drafts are accepted only where
  they match the target argmax); sampled output keeps the target distribution
  via rejection sampling against the selector's `q`.
- MTP stays as the fallback when no DFlash2 draft is downloaded (it requires
  no separate 3.9 GB artifact and works on more model families).
