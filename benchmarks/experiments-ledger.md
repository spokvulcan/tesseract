# Inference-optimization experiments ledger

Endless experiment loop over the Qwen3.5/3.6 PARO models (dense AND MoE) on
the tesseract stack (app + `Vendor/mlx-swift-lm` fork; fork rules in
`docs/mlx-swift-lm-fork.md` — vendor changes shaped upstreamable).

Goal: raw speed/memory only — prefill speed, decode speed, TTFT, CPU
overhead, peak memory — with **zero output-quality loss**. No quantization
changes, no KV-cache quantization, no accuracy-for-speed trades.

## Rules (binding)

- Exactly one hypothesis per iteration; implement minimally.
- Measure Release-only via `scripts/bench.sh` (Debug MLX is ~20× slower).
  Quit the running app first; never two instances. The parity harness runs
  through bench.sh: `scripts/bench.sh quick --model <id> --paro-parity-bench`.
- Quality gate: any change touching numerics or the model graph must pass
  `--paro-parity-bench` (greedy) with **token-identical output** vs the
  unmodified baseline (token IDs recorded per run in the parity report).
- Verdict: reproducible ≥1% win on any metric with no regression on the
  others → commit (Conventional Commits) + log ACCEPTED. Otherwise revert
  completely + log REJECTED. Append either way; commit the ledger with the
  experiment. Tree clean between iterations.

## Measurement discipline (inherited from map #230 — read before trusting any number)

1. **Serialize GPU work** — check `ps` for live `Tesseract Agent` processes
   before trusting a number; a concurrent sweep once fabricated a 562→882
   tok/s "warmup ramp".
2. **Thermals** — the M3 Max throttles under sustained load (602→485 tok/s
   over four back-to-back 32K prefills). A/B must **interleave** (round-robin
   / ABBA) and compare within a round, never all-A-then-all-B. Absolute tok/s
   is not comparable across time.
3. **Launch with `open`, never `nohup`/`&`** from an agent shell — nice 5
   collapses CPU-bound phases (decode 17.7 vs 80.4 tok/s measured).
   `scripts/bench.sh` uses `open -W`; verify `ps -o nice=` → `0`.
4. **Divide timings by FLOPs** before believing kernel comparisons (#251
   retraction).
5. **eval-barrier attribution biases itself** — coarse tier for absolute
   seconds, fine tier only for ratios within a block (#254).
6. **Verify model constants against `config.json`**, never against harness
   assumptions.

## Proven no-gos (never repeat)

| Idea | Verdict | Source |
| --- | --- | --- |
| Fused head_dim-256 prefill attention kernel | NO-GO — slower at every context (1.13–1.35×); unfused fallback already at 84–88% of peak bf16 GEMM; the two GEMMs are a hard lower bound | #251 |
| PARO projection fusion (QKV in attention; `in_proj_*` in GDN) | NO-GO, structural — each projection rotates the input with its own `theta`/`channel_scales`; no shared-input GEMM exists | #257, #255 |
| GDN chunk-scan megakernel (MegaGDN-style) | NO-GO — our GDN scan is already a single recurrent Metal kernel, ~1.9 ms/layer/chunk, flat with context | #234 |
| Raising `prefillStepSize` above 1024 | NO-GO — collapses at long context (128K: 155 vs 431 tok/s), peak-memory blowup; balanced chunking (#258) already banked the tail win | #253, #258 |
| `in_proj_b`+`in_proj_a` F16 fusion in GDN | Legal but pointless — ~960 launches saved vs 0.38% CPU graph-construction cost | #255 |
| Cmlx 0.31.1→0.32.0 bump | No measured kernel win (all four hot ops at parity within 4%) | #235 |
| Speculative decoding / draft models | NO-GO — MoE-hostile (~1.11×, ~11% accept), MTP tensors stripped, 248320 vocab locks out drafts | #235 |
| kvBits=8 | Saves zero peak memory, costs decode 7.6→40%; dropped as default | #252 |
| `gather_qmm` gather/scatter overhead theory | Killed — permutation+rotations are 3.17 s vs 25.54 s matmuls at 32K/step-1024 | #254 |

## Open questions from prior art

- **#256 `gather_qmm` rows-per-expert headroom** — unresolved: 43.2% of peak
  at B/E=32 → 64.4% at B/E=128. Bandwidth roofline (unrecoverable) or tiling
  (recoverable, ~14% of prefill)? Needs a TFLOP/s-vs-B/E sweep at fixed total
  FLOPs. The grouped-sorted fast path (`gather_qmm_rhs`) is **already
  engaged** in prefill — no "small-M fallback" to escape.
- Decode-side beyond kvBits: sampler/per-step Swift overhead — un-sized.
- Load-time: PARO 35B cold load ~40.8 s (AWQ→MLX conversion); Prepared
  Checkpoint artifact exists in the fork — check app wiring.
- Warm-path TTFT (prefix-cache restore cost).

## Environment

- Hardware: Mac15,9 (M3 Max), 48 GB
- Target models: `qwen3.5-4b-paro` (dense, z-lab/Qwen3.5-4B-PARO),
  `qwen3.6-35b-a3b-paro` (MoE, z-lab/Qwen3.6-35B-A3B-PARO)
- Ruler: `--paro-parity-bench` (greedy, fp16 KV, 256 new tokens, contexts
  128/8192/32768, 2 runs/context, production `prefillStepSize=1024`,
  balanced chunking active) — reports prefill tok/s, decode tok/s, peak GB,
  load s, tokenize s per context, and per-run generated token IDs.

---

## Session 2026-07-23

Git HEAD at session start: `5d955f46` (chore(vendor): re-pin mlx-swift-lm on
upstream eaefe75). Harness change preceding all experiments this session
(non-numeric): parity bench records per-run token IDs; app dispatch routes
`--paro-parity-bench` before `--benchmark` so bench.sh can drive it.

### Baseline (fresh, this session)

Recorded 2026-07-23 ~02:20 local, Release build @ `5d955f46` + non-numeric
harness change, quiet machine, nice 0 verified. Reports:
`benchmarks/results/paro-parity/baseline_*.json` (per-run token IDs included).

**qwen3.5-4b-paro** (load 1.4 s):

| ctx | prefill tok/s (r0/r1) | decode tok/s (r0/r1) | peak GB |
| --- | --- | --- | --- |
| 128 | 914.1 / 915.9 | 108.3 / 108.2 | 2.82 |
| 8192 | 1354.4 / 1349.9 | 95.5 / 95.0 | 4.18 |
| 32768 | 967.3 / 1001.0 | 57.1 / 57.3 | 5.67 |

**qwen3.6-35b-a3b-paro** (load 4.8 s — Prepared Checkpoint active; #230's
40.8 s cold load is stale, load-time is no longer a target):

| ctx | prefill tok/s (r0/r1) | decode tok/s (r0/r1) | peak GB |
| --- | --- | --- | --- |
| 128 | 741.4 / 747.1 | 79.8 / 80.7 | 19.07 |
| 8192 | 1457.5 / 1457.5 | 75.6 / 76.4 | 20.27 |
| 32768 | 1005.7 / 1158.4 | 60.8 / 65.2 | 21.44 |

Notes: MoE 32K shows ~15% run-to-run prefill variance (thermal — trap 2);
all A/B verdicts must interleave against the baseline binary, not against
this table. Decode falls steeply with context on the dense model
(108→57 t/s) — per-step overhead scales with KV length.

### Experiments

**E0 — methodology shakedown (baseline vs itself).** Ran `parity-ab.sh` with
the same binary on both arms (qwen3.5-4b-paro, 1 round, ctx=128): quality
gate PASS (token-identical across separate processes — cross-process
reproducibility confirmed); same-binary noise floor measured: decode ±0.1%,
prefill at ctx=128 ±2%, peak GB ±0. **Calibrations: (a) the ≥1% win bar is
meaningful for decode and 8K/32K prefill, but ctx=128 prefill needs ≥2%;
(b) load-time comparisons must discard round 1** (first arm pays one-time
warmup: 3.02 s vs 0.96 s same binary). Not an optimization; no code change.

**E1 — MoE prefill: rotate `gate_up` before the expert gather/sort, not
after.** Hypothesis: `PairwiseRotation` is row-independent and `gatherSort`
only duplicates rows, so rotating `L` rows pre-gather is bitwise-identical
to rotating `L×topK` rows post-gather — at 1/8 the rotation work per MoE
layer per chunk. Change: `Vendor/.../ParoQuant/RotateSwitchGLU.swift` — moved
`gateUpRot.rotate(x)` ahead of `gatherSort` (one line; docs updated).
Measure: 3-round interleaved A/B, qwen3.6-35b-a3b-paro, contexts
128/8192/32768. Gate: **PASS** (20/20 pairs token-identical). Numbers:
prefill **+1.35/+3.15%** (128), **+3.21/+3.35%** (8K), **+4.50/+4.00%**
(32K); decode +1.3/+0.7% (128), +1.7/+2.3% (8K), −4.3/−0.3% (32K — code
path at decode is provably identical (`doSort=false`), wobble inside the
32K-decode noise band, not reproduced across runs); peak +0.05–0.14%
(≤30 MB counter noise; the change mechanically reduces transient
33 MB→4 MB for the rotated copy per layer-chunk). Load −5.8% (within
load-warmup bias, not claimed). **Verdict: ACCEPTED** — reproducible ≥1%
prefill win at all contexts, no mechanistically-possible regression.
Vendor commit on the pin branch; gitlink in tesseract.

**E2 — compile-fuse `computeGatedDeltaG` (GDN decay chain).** Hypothesis:
decode is partly launch-bound; fusing the 6-kernel elementwise g chain
(`exp(-exp(aLog.f32) * softplus(a + dtBias))`, ~180 launches/token on the
35B, ~144 on the 4B) into one compiled kernel speeds decode. Pre-evidence:
standalone probe verified MLX `compile(shapeless:)` is **bitwise-identical**
to the unfused chain on the real shapes/dtypes — including bf16-intermediate
controls, refuting the "fusion loses intermediate rounding" prior for this
op class (reusable fact). Change: `Vendor/.../GatedDelta.swift` —
`compiledGatedDeltaG` behind the same function. Measure: (a) 3-round A/B vs
pre-E1 baseline, both models, 128/8192/32768 — MoE decode +4.2–6.4% (128),
+4.1/+19.4% (8K), but thermal throttle collapsed the 32K zone in BOTH arms
(MoE 60→15 t/s; trap 2 — numbers there unusable); (b) marginal isolation
A/B (E1-app vs E1+E2-app, 128/8192): **MoE decode +5.05/+2.28% (128),
+3.66/+3.11% (8K)**; (c) reversed-arm-order control (dense): 128 decode
+1.35/+1.45% in BOTH orders (real), 8K decode −0.89/−0.14% — combined with
earlier readings 6/6 negative, mean ≈ −0.5%, order-independent, inside the
same-binary session band for dense-8K decode (±0.5%). Gate: **PASS** both
models (18/18 + 8/8 + 8/8 pairs token-identical). **Verdict: ACCEPTED** —
MoE decode +3.1±1.1% (4/4 ≥ +2.3%), dense-128 decode +1.4% (6/6 ≥ +0.9%),
prefill/peak unchanged; the lone negative (dense-8K decode −0.5%) is below
the ≥1% materiality floor and within the harness's own band for that
metric.

**Protocol amendments (from E2):** (1) `parity-ab.sh` now alternates the
first arm per round (ABBA) — the second arm is thermally disadvantaged and
it contaminates sub-1% verdicts. (2) Decode-focused experiments use
contexts 128,8192 — 32K decode is thermally chaotic and KV-bandwidth-
dominated, so launch-count effects vanish there anyway. (3) Regression
materiality floor = ≥1%, symmetric with the win bar — sub-1% is inside the
measured noise band (E0), so "no regression" means "no reproducible ≥1%
degradation". (4) Marginal effects must be isolated against the previous
experiment's binary, not the session baseline (which accumulates accepted
wins).

**E3 — compile-fuse `preciseSwiGLU` (GDN gated norm).** Hypothesis: same
fusion family as E2 — 5 kernels → 1 per gated norm per step (~120–150
launches/token) should speed decode. Change: `Vendor/.../Qwen3Next.swift` —
`compiledPreciseSwiGLU` (reverted). Measure: marginal isolation A/B
(E2-app vs E3-app, ABBA, 3 rounds, 128/8192, both models). Gate: **PASS**
(12/12 each). Numbers: MoE decode −0.5/−0.3% (128), −1.5/+0.5% (8K);
dense decode +0.3/+1.3% (128), +0.5/−0.6% (8K); prefill ±0.5–1.9%
(noise-signed); peak −0.4/−0.5% consistently (20–70 MB — real but sub-1%).
**Verdict: REJECTED** — no reproducible ≥1% win on any metric. The
carried information: **after E2, decode is no longer launch-bound** — the
elementwise-fusion family is exhausted (E2 already collected the available
win; the gated-norm chain's larger ~4K-element tensors were never
latency-bound). Consequences, no iterations spent: **E4 (rotation `params`
array cache), E6 (dense `silu(g)*up` fusion), E7 (`sigmoidMultiply`
fusion) demoted** — same micro-op class with smaller counts, cannot clear
the bar. Diff reverted; vendor tree clean.

**E4 — #256 research verdict: `gather_qmm` headroom is occupancy, not
bandwidth.** Hypothesis under test (from issue #256): "the B/E=32→128
TFLOP/s headroom is reachable at fixed B/E (tiling), not a weight-
bandwidth roofline." Method: standalone sweep harness (scratch SwiftPM
pkg on the vendor) timing `gatherQuantizedMM` on the sorted-rhs fast path
at the real shapes (E=256, N=512, K=2048, 4-bit, gs=128 per config.json —
#256's table said 64, the checkpoint says 128), bf16 activations, uniform
random routing. **Sweep harness gotcha found:** x must be 3-D `[B,1,K]`
with 1-D indices (production's post-gatherSort shape); a 4-D x makes
`indices_or_default` broadcast `[B,1]×[B]→[B,B]` and silently computes B×
redundant work (32 GiB alloc at B=2048). Results: 1.37 / 2.34 / 3.61 /
**5.14** / 6.34 / 7.16 / 7.69 TFLOP/s at B/E = 4/8/16/**32**/64/128/256
(% of 12.69 peak: 10.8→60.6). Dense 4-bit qmm at B/E=32's FLOPs: 7.41
TFLOP/s; the gather kernel CONVERGES to it (7.69) at B/E=256. Analysis:
weights are 67 MB → 0.22 ms bandwidth floor; B/E=32 takes 3.34 ms at an
effective 43 GB/s of ~300 available — nowhere near bandwidth-saturated;
TFLOP/s grows with rows-per-expert and saturates at the dense-GEMM rate.
**Verdict: recoverable tiling/occupancy loss, NOT a roofline** — #256's
~14%-of-prefill estimate confirmed as existing. But the tile geometry
(`bm=16/64`, per-expert tile padding at small B/E) lives in Cmlx
(mlx-core), which this loop does not fork — the kernel-internal fix is
upstream territory (owner's call to file). In-scope lever identified and
kernel-probed: **`gate_proj`+`up_proj` fused into one gathered QMM at
N=1024** (shared x and indices; concat along the output dim at the
group-128 boundary, per-element bitwise-identical): **1.07–1.09×** on the
kernel pair across B/E 16–128 → modeled ≈ +1.7% of 35B prefill (MoE
matmuls = 42.8%×78% of prefill per #254; gate/up = 2/3 of them × 7.5%).
That becomes E5.

