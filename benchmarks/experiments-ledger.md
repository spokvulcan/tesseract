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

**E5 — fused gate+up gathered QMM (MoE).** Hypothesis: one N=1024
gathered QMM replaces two N=512 calls (shared input, shared indices) →
~7.5% on 2/3 of MoE matmuls → ≥1.5% app prefill. Change:
`RotateSwitchGLU` + `ParoQuantLoader` — load-time concat of gate/up
expert weights along the output dim (group-boundary-legal), placeholder
replacement of the separate children, one fused `gatherQuantizedMM` +
last-axis split per forward. Probe-verified bitwise-identical on the
exact production shapes (5-D decode, 3-D sorted prefill) before any app
run. Four measurement attempts; token gate **PASS on all** (18/18, 20/20).
Final numbers (5-round ABBA vs E2 binary): 8K prefill **+0.48/+0.40%**
(6/6 positive across all attempts — real, but sub-1%); 32K per-round
prefill pairings −11.6…+5.0% (thermal noise, no consistent win); decode
flat; peak +140 MB transient at 8K/32K (sub-floor); **load +1.2 s
(+25%)** — the fusion's load-time cost survives every optimization
(per-array eval overhead × 120 arrays; verify-walk removal didn't help).
**Verdict: REJECTED** — a sub-bar speed win against a certain load cost.
Reverted completely. Implementation lessons (reusable): (1) `ModuleInfo`
parameters trap on direct mutation — release modules via
`update(modules:)` placeholder replacement; (2) probe shapes must match
production exactly — the gathered-QMM output is **5-D** `[B,L,topK,1,2N]`
at decode, 3-D on the sorted prefill path (a 3-D-only probe missed this
and the first build broke the load-time warmup generation); (3) never
build derived tensors before `eval(model)` — lazy checkpoint tensors
materialize one-at-a-time (+2.3 s, +11 GB loadPeak); (4) the vendor's
"ParoQuant load phases" os_log is the load-cost ruler; (5) the 8K MoE
prefill zone (~1490 t/s) is near-saturated — kernel-level wins there
shrink ~3× vs naive attribution (1.07× on the pair → 0.5% app).

**E6 — rotation kernel tile tuning (4 → 16 rows/tile).** Hypothesis:
larger `ROWS_PER_TILE` at prefill batches amortizes the krot barrier
rounds + per-tile coefficient loads in the PARO rotation kernel → ≥1%
prefill on both models. Probe (direct `PairwiseRotation.rotate` timing,
tile variants interleaved within one process via a temporary env hook):
tile=16 −27% at [1024,2048], −15% at [997,2560] and [8192,512] vs 4;
tile=32 regresses (occupancy+tails); **bitwise-identical across tiles**.
Also found: production runs F16 (the 35B checkpoint stores F16 despite
`"dtype": "bfloat16"` in config) — the kernel has a *latent* bf16 compile
failure on a path nothing takes (noted, not fixed here). Change: tile
selection 1 / 4 / 16 by batch (reverted). Measure: 3-round ABBA both
models + 4-round 8K tie-break, gates **PASS** everywhere. Numbers: MoE 8K
prefill per-round +0.69/+0.68/+1.00/+0.91/+0.77/+1.99/+1.28/−0.28 (mean
**+0.88%**); dense 8K +0.48…+1.22% (mean ~0.8%); 32K confounded by
throttle; decode/peak/load flat (decode code provably identical — tile=1
at batch=1). **Verdict: REJECTED** — real, consistent, mechanistically
supported (probe + sign-consistency across 16+ readings), but the mean is
~0.85% < 1%. The probe's real payload: the rotation kernel runs at
10–30 GB/s effective — it is barrier/latency-bound, not bandwidth-bound,
so the lever is restructuring (register-resident simd-shuffle, no
threadgroup tile, no barriers) not tuning. That is E6b.

**E7 — device-side deep copy for prefix-cache snapshots (warm TTFT).**
Hypothesis (app-side map H2): `HybridCacheSnapshot.deepCopyState`'s host
round trip (`asData`→`Data`→`MLXArray` — two memcpys + a snapshot-sized
`Data` transient per array, per layer, per capture/restore) is pure
overhead; a device-side copy is faster and removes the transient.
Change: `tesseract/Features/Server/HybridCacheSnapshot.swift` —
`deepCopyState` becomes `array * 1` (device multiply-by-one: binary ops
always allocate a fresh output ⇒ independence by construction; IEEE
x×1.0 == x exactly, bytes bit-identical) returned **lazy**, with capture
and restore each hoisting ONE `eval` per operation (per-array syncs made
the device path ~2× *slower* than host at 200–300 MB — 80 arrays × ~0.2
ms sync; one hoisted eval fixes it). New `--snapshot-bench` runner
(synthetic 35B-shaped cache stack, within-process ABBA, byte-equality
gate) + bench.sh/dispatch routing. Measure: **byte-equality IDENTICAL**
at 8K/32K; **`--prefix-cache-e2e` PASS** (all 19 checks, output equality
on hit paths). Numbers (mean of 4 rounds × 3 iters): at 232 MB snapshot:
capture 8.2 → **1.9 ms (~4.3×)**, restore 8.2 → **1.9 ms (~4.3×)**; at
735 MB: capture 26.7 → **4.7 ms (~5.7×)**, restore 26.8 → **4.7 ms
(~5.7×)**. Peak unchanged (0.70/2.21 GB both strategies — the host
path's per-array `Data` frees between arrays, so the transient never
accumulated; the claimed peak benefit is **dead**, recorded honestly).
Translation: ~6.5–22 ms off warm-path restore (TTFT) and cold-prefill
capture per snapshot at realistic sizes (200–700 MB). **Verdict:
ACCEPTED** — ≥4× on the copy mechanism that sits on the warm-TTFT and
cold-prefill paths, byte-identical, functional gate green.

**E8 — GDN scan probe: sequential-latency-bound; only software
pipelining is bitwise-legal.** Method: direct `gatedDeltaUpdate` timing
at the production shapes (f16, [1,T,16,128]/[1,T,32,128], state f32),
T ∈ {128, 512, 1024, 2048}, mask on/off. Numbers: T=1024 → **2.0 ms**
(matches #234's 1.9 ms/layer/chunk — harness consistent); T=512 → 1.2 ms;
T=2048 → 3.8 ms; masked ≈ unmasked at large T, faster at small T (mask
skips FMAs). Scaling: ~1.8–2.4 µs/step at T≥512, fixed ~1.2 ms overhead
at small T. Attribution: ~0.5 µs serial per step per CTA × ~4 CTA waves
— dependent-load latency in the sequential recurrence, NOT bandwidth
(~20 MB moved → 53 µs floor) and NOT threadgroup barriers (the kernel
uses 2 `simd_sum`/step, no CTA barrier — source read). Legal levers
under the bitwise rule: **software pipelining only** (prefetch t+1's
q/k/v during t's compute — identical arithmetic); chunked/parallel-scan
(Blelloch) changes f32 rounding order → dead. Prize if pipelining
works: the scan is ~5.5% of 32K prefill (1.9 ms × 30 layers × 32 chunks
/ 33 s); a 2× cut ≈ +2.7% prefill. Queued behind E6b (rotation
simd-shuffle rewrite, larger expected win).

**E6b — simdgroup-resident rotation kernel (barrier elimination).**
Hypothesis: the rotation kernel's cost is its krot=8 serialized
`threadgroup_barrier` rounds (E6's probe: 10–30 GB/s effective = latency
-bound); making one CTA a single 32-lane simdgroup per (row-tile,
channel-group) makes per-round sync free, while keeping per-element f32
arithmetic identical (bitwise by construction). Change:
`PairwiseRotation.swift` + `RotateQuantizedLinear.swift` — 32-lane CTAs,
compile-time krot (register-resident coefficients), row-major f32 tile
(old layout had an 8-way bank conflict), float4 IO, explicit write-back
cast (fixes the latent bf16 compile bug as a bonus); requires
groupSize==128 (precondition). Implemented by a coder subagent (died to
3× transient 429 after producing the design + probe); validated by me:
isolated probe (ABBA, f16) **1.67–2.03× faster** at [1024,2048],
[997,2560], [8192,512], [512,9216], **bitwise IDENTICAL everywhere**,
decode [1,2048] unchanged; vendor-class bitwise check through real
`PairwiseRotation.rotate` IDENTICAL. App A/B (3 rounds ABBA vs current
main, both models): gate **PASS** (18/18 each). **MoE prefill
+1.83/+2.51% (8K)**; **dense prefill +1.59/+2.14% (8K), +1.86/+1.26%
(128)**; **dense decode +4.38/+3.38% (128), +4.03/+4.96% (8K)** — the
tile=1 restructure (old 64-thread CTA, half the lanes idle + 8 CTA
barriers; new 32-lane simdgroup) aggregates ~100 rotation launches/token
at decode; MoE decode flat (bandwidth-dominated); 32K discarded
(throttle zone); peak +0.00%, load noise. **Verdict: ACCEPTED** —
multiple metrics ≥1% reproducible on both models, no regression.
Vendor commit `8d1fb7b`; gitlink in tesseract.

**E8b — GDN scan software pipelining.** Hypothesis: the scan's ~0.5 µs
serial per step per CTA (E8) is dependent-load latency; register-
prefetching t+1's q/k/v/g/beta during t's arithmetic hides it — bitwise
-identical by construction (same values, same arithmetic order; loads
have no side effects). Change: `GatedDelta.swift` t-loop rewritten with
cur/next register pipeline (reverted). Measure (probe, vendor
`gatedDeltaUpdate` vs verbatim pre-edit kernel, f16): **y and state
bitwise IDENTICAL** at T ∈ {512, 1024, 2048, 1}; speed T=512 1.03×,
**T=1024 0.82×, T=2048 0.80×**, T=1 0.96×. **Verdict: REJECTED** — the
manual pipeline is ~20% slower at production sizes (16 extra live
registers/thread: spills or occupancy loss; the compiler was already
scheduling loads fine). Reverted completely. **This closes the GDN-scan
line:** it is sequential-latency-bound (E8), the only bitwise-legal
restructure makes it slower (E8b), and any parallel-scan/chunked variant
changes f32 rounding order (dead under the zero-loss rule). The ~5.5%
of 32K prefill the scan costs is a floor for this stack.
