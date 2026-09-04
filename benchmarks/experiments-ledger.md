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

- **The remaining optimization surface now lives in mlx-core** — see
  `docs/mlx-core-optimization-roadmap.md` (M1–M8 with measured
  evidence and gain estimates: #256 tile fix ~12–15% of 35B prefill,
  decode segmentation ~10%, small-M qmv floors ~10%, fused rotate+QMM
  ~3–4% prefill, attention tail ~1–2%, tokenizer path, GDN scan floor,
  MoE expert prefetch ~5–10% MoE decode).

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
Vendor commit `8d1fb7b`; gitlink in tesseract. (Review-round follow-up:
`017086e` restores the generic kernel as the groupSize != 128 fallback —
see the review-round entry below.)

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

**E9 — decode lm_head QMM: at the practical roofline, no lever.**
Probe (quantizedMM, [248320, 2048] 4-bit gs=128, f16 activations): M=1
(decode) = 1.155 ms at **234 GB/s** effective weight-read (254 MB);
M=1024 = 10.4 TFLOP/s ≈ 82% of peak GEMM. Dense f16 GEMV reference:
342 GB/s (2.98 ms for 1.02 GB) — the 4-bit dequant GEMV's ~68%-of-GEMV
-bandwidth is inherent to the format; quantized already beats dense on
time (1.16 vs 2.98 ms). Logit-subset compute would change output (dead
under the rule). **Verdict: no-go, roofline — the ~0.25 GB/step lm_head
read is fundamental.** Spin-off question, now the biggest open decode
mystery: MoE decode weight traffic ≈ 1.3 GB/step → ~5.5 ms floor at the
lm_head's own measured 234 GB/s, but decode measures 12.5 ms — **~2× of
MoE decode is NOT bandwidth-explained** (kernel-count, gather_qmv
geometry, or CPU dispatch — unprofiled). Becomes E10.

**E10 — attribution of the MoE decode 2× gap: mlx-core internals, out of
scope.** Question (from E9): MoE decode ≈ 1.3 GB/step → ~5.5 ms floor at
the lm_head's measured 234 GB/s, but decode is 12.5 ms — where is the
rest? Method: xctrace Metal System Trace attach during a ctx=128 parity
decode + two probes (trap-5 warning: the first gather probe read 350
µs/call = 85% sync artifact; amortized re-measure required). Findings:
(a) **GPU busy 78% during decode** (2,891 command buffers over 0.92 s,
sum-of-durations) — ~40 command buffers per token with ~60 µs inter-
buffer gaps ⇒ **~22% of decode is command-buffer segmentation /
.sync / dispatch idle — mlx-core eval scheduling, not app/vendor.** (b)
Amortized probes: gather_qmv at the decode shape (B=8 unsorted) = 51 µs
& **88 GB/s**; dense 4-bit qmm of the same bytes = 41 µs & 108 GB/s;
lm_head qmv = 234 GB/s (E9). Busy-time is spread across ~1,900
kernels/token, all small-work-per-call at M=1 → decode is latency/
occupancy-limited, with per-call floors (~10–50 µs) set by kernel-
internal geometry. In-scope collectibles: E2's fusion and E6b's
rotation/kernel restructure already took the available wins; E5's
call-count reduction (−40 qmv calls/token) measured ~0 — per-call
latency, not call count, binds. **Verdict: the remaining MoE decode gap
lives in mlx-core (eval scheduling, small-M qmv/gather tiling) —
upstream territory, same class as #256. No in-scope experiment to run;
logged so the loop doesn't re-attack MoE decode micro-structure.**

**E11 — memoize the stable-prefix two-probe detect (server TTFT).**
Hypothesis (app-side map H3): `StablePrefixDetector.detect` runs two
Jinja renders + BPE encodes of system+tools per server request; the
result depends only on (systemPrompt, tools, additionalContext), all
stable in production → memoize it. Change: `StablePrefixDetector.swift` —
memo keyed by SHA-256 of the inputs, storing (commonLength, hash of the
common prefix tokens); a hit runs the SAME fullTokens verification
(prefix-hash match) and ratio guard as a fresh detect, so a wrong-for-
this-template entry degrades to a fresh detect, never a wrong boundary.
New `--prefix-detect-bench` runner (production-scale: ~10K-token system
prompt + 40 tool specs, real tokenizer, ABBA miss-vs-hit). Measure:
**miss 206.09 ms vs hit 0.73 ms per request — saves 205.36 ms (99.6%)**,
0 mismatches over 6 rounds; **`--prefix-cache-e2e` PASS** (boundary
checks included). Translation: every server request runs this detect
once — ~205 ms off TTFT per request at 10K-token prefix scale (scales
with prefix size; ~10 ms at the E2E's 500-token prefixes). **Verdict:
ACCEPTED.**

## Review round — PR #424 (2026-07-23, post-loop)

Two external reviews (Fable 5; GPT 5.6) covered the loop's PR. Both
converged on the same load-bearing findings; those were fixed the same
day. The speculative findings were declined with reasons, recorded here
so they don't get re-litigated.

**Fixed:**

- **E6b narrowed vendor generality (top code finding, both reviews).**
  The simdgroup rotation dispatch precondition-crashed on
  `groupSize != 128`, while the vendor's own `ParoQuantTests` exercise
  group sizes 8 and 64 — a real crash-regression in the existing suite
  and a fork-rule violation (vendor changes must be upstreamable). Fix:
  restored the pre-E6b generic kernel verbatim as the fallback for
  `groupSize != 128`, dispatched from a single shared
  `dispatchPairwiseRotation` used by both `PairwiseRotation` and
  `RotateQuantizedLinear` (the simdgroup kernel still serves 128; the
  fallback also preserves the pre-E6b bf16 limitation, now documented).
  Verified: `swift test --filter ParoQuant` — **24/24 pass**.
  **Protocol amendment: a vendor `ParoQuantTests` run is required for
  every vendor-touching experiment from now on** — it would have caught
  this at E6b.
- **The A/B gate could false-pass (both reviews, independently).**
  `parity_compare.py` skipped missing keys and `zip()`-truncated unequal
  round counts; `parity-ab.sh` never cleaned `/tmp/parity-ab`, so
  experiment N+1 could inherit experiment N's reports. Now: mismatched
  key sets or per-key round counts are FATAL (exit 2) before any token
  comparison, and the runner wipes its staging dir at start.
- **E7's "byte-equality" gate was value-equality (both reviews).**
  `SnapshotBenchRunner` compared snapshots with numeric `.==` (blind to
  ±0 / NaN payloads) and silently skipped the check when a capture
  returned nil while still logging IDENTICAL. Now compares raw bytes
  (`asData(access: .copy)`) and a nil capture fails loudly.
- **The E11 memo had no unit tests (Fable).** Added
  `StablePrefixDetectorMemoTests`: hit avoids re-probing (probe-count
  assertions), hit tolerates new user content, a colliding/stale entry
  degrades to a fresh detect and never returns a wrong boundary, and the
  256-entry eviction stays correct. All three detector suites now reset
  the process-global memo per test (`init` → `resetMemo()`); all are
  `@MainActor`, so resets can't race another suite's detect.

**Declined, with reasons:**

- *Restore the teacher-forced logit-parity gate (GPT, P1).* The loop's
  binding contract is token-identity (Rules above), and all three vendor
  accepts are pure reorderings — structural bitwise arguments plus
  probe-verified bitwise kernels plus thousands of identical tokens on
  both models. Logit parity becomes the binding gate the moment an
  experiment touches accumulation order; the roadmap's M4 (fused
  rotate+dequant+GEMM) is already flagged for exactly that. No harness
  change now.
- *Memo key omits tokenizer identity (GPT, P2).* The hit path re-hashes
  the current request's prefix tokens: a different tokenizer yields
  different token IDs → hash mismatch → fresh detect. Staleness needs a
  template swap with byte-identical prefix tokenization, and even then
  yields a valid-but-shorter boundary (cache reuse, not correctness).
  Not worth key churn.
- *Upstream PR filing for the three new carries (ADR-0006).* Real
  process debt; deferred to one batched upstream PR pending owner
  go-ahead (outward-facing action). Tracked in
  `docs/mlx-swift-lm-fork.md`.
- *Nits* — tokenHash Data building (measured 0.73 ms; cold), typed
  bench errors, memo eviction policy. The `-> dict` annotation was fixed
  in passing; the rest move neither correctness nor speed.

**Gate status:** `hybrid-cache-correctness` **PASS** (all 11 checks;
mid-prefill restore bitwise at K=[512,1024,1536], mamba/KV/quantized-KV
state maxAbsDiff=0.0, 16K restore exact) — the `docs/testing.md`
loaded-model gate for the E7/E11 files, now on record. Vendor
`ParoQuantTests` 24/24 (above); detector suites green (memo tests
included).

---

## Session 2026-07-23 — Cmlx (mlx-core) loop

Same rules and measurement discipline as the first session (above), now
scoped to **mlx-core (Cmlx)** per `docs/mlx-core-optimization-roadmap.md`.
Experiments in this loop are numbered **C0, C1, …** to keep them distinct
from the app/vendor loop's E-series. Git HEAD at session start: the
post-review-round tree.

### Infrastructure: buildable mlx-core fork/pin scheme (prerequisite task)

The Cmlx sources reach the build only as a **git submodule of mlx-swift**
(`Source/Cmlx/mlx`), so the fork has two levels, both under `spokvulcan`
(scheme doc: `docs/mlx-core-fork.md`):

- `spokvulcan/mlx` branch **`pin-tesseract`** @ `ce45c525` — exact upstream
  content (mlx `v0.31.1`), the writable mlx-core. Append-only.
- `spokvulcan/mlx-swift` branch **`pin-tesseract`** @ `54ca1ec` — upstream
  `0bb916c` (the 0.31.6 tag the app pinned) + ONE commit: `.gitmodules`
  points `Source/Cmlx/mlx` at `spokvulcan/mlx`. Zero source diff. mlx-c
  submodule untouched (`ml-explore/mlx-c` @ `0726ca9`).
- Lockstep pins (`54ca1ec7cf9601c39809720725211afe601cfdd5`):
  `Vendor/mlx-audio-swift/Package.swift`, `Vendor/tesseract-speech/
  Package.swift` (in-tree), `Vendor/mlx-swift-lm/Package.swift` (commit
  `37702c8` on its `pin-upstream-mlx-swift` branch; tesseract gitlink bump).

**Corrected pin fact:** the roadmap/kickoff said "Cmlx tracks ml-explore/mlx
@ dc43e62d". `dc43e62d` is an mlx-**swift** revision seen in a stale
DerivedData checkout, not an mlx revision. The mlx-core the app builds is
`ce45c52505c8158ea48d2a54e8caae05efd86bfe` (tag `v0.31.1`), the
`Source/Cmlx/mlx` gitlink recorded by mlx-swift `0bb916c` — verified via
`git ls-tree 0bb916c Source/Cmlx/` and the resolved app DerivedData
checkout. Roadmap note amended.

Also established this session (source read, `device.h`/`device.cpp`):
`is_nax_available()` = macOS 26.2+ AND arch gen ≥ 17 (non-phone); M3 Max is
`g15s` → gen 15 → **nax is unavailable on this machine** — the production
`gather_qmm_rhs` path is the non-nax kernel (`bm=16, bn=32, bk=32, wm=1,
wn=2`), not the nax one (`bm=64`). M1 targets the non-nax kernel.

Per-iteration workflow (in `docs/mlx-core-fork.md`): edit in the live
DerivedData checkout's submodule → build/bench → REJECTED: `git checkout
-- .` in the submodule; ACCEPTED: port diff to `~/projects/mlx`
(`pin-tesseract`) + gitlink bump in `~/projects/mlx-swift` + three-pin
lockstep move + tesseract commit, then re-resolve and verify the port
(`git diff ce45c525` in the checkout == accepted diff).

### Experiments

**C0 — fork-scheme shakedown (pre-fork binary vs fork-built binary).**
Provenance-only change (byte-identical sources), so the binary content is
unchanged by construction; run to *prove* the fork chain builds and stays
output-identical. Method: `parity-ab.sh`, 1 round, contexts 128/8192,
pre-fork saved binary vs fork-pinned Release build. Gate: **PASS both
models** (4/4 pairs each, token-identical). Perf: everything within ±2.7%
except the expected second-arm thermal dip on the second 8K prefill
(dense −13.4%, MoE −7.3% — single-round ABBA has no BA balancing; the
same artifact shape appeared in E0/E2). **Verdict: scheme VALIDATED** —
fork chain is the new baseline; all C-experiments pin/fork from here.
(Not an optimization; infra commit.)

**C1 — rows-per-expert-aware `gather_qmm_rhs` tile geometry (M1).**
Hypothesis: with sorted rhs indices, a BM=16 tile is what keeps tiles
inside single-expert runs at small B/E, but at production's B/E=32 a wider
32×64 tile is outright faster (single-segment tiles + denser MMA) — pick
geometry by measured rows-per-expert: `(bm,bn,bk,wm,wn) = (32,64,32,1,2)`
when `M/E >= 32`, stock `(16,32,32,1,2)` below. Bitwise by construction
(per-element K-accumulation order is tile-geometry-independent — verified
empirically, not just argued). Pre-evidence (standalone sweep harness,
probe-only `MLX_GQMM_CFG` env hook in the fork clone, ABBA in-process,
f16, production shapes E=256/N=512/K=2048 + down_proj N=2048/K=512): at
B/E=32 — **+13.6%** (gate/up shape) and **+12.5%** (down shape); +19–22%
at B/E=64–128; 0.6–0.8× at B/E≤24 (the straddle cliff → threshold 32);
**bitwise IDENTICAL for every config at every B/E**; dense-qmm anchor
9.5–10.1 TFLOP/s, the winner reaches 96% of it (gather overhead ~gone).
Note: macOS/SwiftPM builds JIT the kernels (`jit_kernels.cpp`, not
nojit) — geometry changes need host edits only, no instantiation plumbing;
and the sweep's absolute TFLOP/s ran ~1.5× above E4's (harness/thermal
calibration differs — within-run ABBA ratios are the evidence, absolute
anchors are not). Change (one hunk in
`Cmlx/mlx/backend/metal/quantized.cpp`, DerivedData checkout): E from the
weight batch dims, `M/E >= 32` → `bm=32, bn=64`. Measure: (a) 3-round
ABBA MoE full contexts — gate **PASS** (18/18 token-identical); 32K
prefill per-round pairs **+7.3/+2.1 | +7.9/−2.1 | +8.2/+13.7** (mean
+6.2%, 5/6 positive); 8K prefill +0.7/±0 in calm rounds, negative only
inside the mid-session thermal-collapse round; decode mixed ± (mean ~0 —
prefill-only kernel, untouched); peak flat. (b) dense control 2 rounds —
gate **PASS** (8/8); perf pure noise (incl. a −6.6% pooled 8K-prefill
reading *with identical code paths* — the afternoon's noise floor on
record). (c) 4-round 8K MoE tie-break — gate **PASS** (8/8); per-round
−1.1/+2.2/+0.6/−3.2 | −1.3/+2.2/+1.6/+16.3 (mean **+2.2%**, 5/8
positive) — no reproducible 8K regression; the 8K zone is
dispatch-saturated (E5 lesson: kernel wins shrink ~3× there). **Verdict:
ACCEPTED** — reproducible ≥1% win (32K prefill +6.2%, three readings
≥+7%), no reproducible regression on any other metric, 34/34 pairs
token-identical across the three runs. E4's "~12–15% of 35B prefill"
estimate was calibrated on its 5.14-TFLOP/s harness reading; this
session's harness reads the stock kernel ~1.5× faster, and the measured
app win is +6% at 32K — the opportunity existed (M1's premise stands),
its size was overestimated by the older harness. Ported to
`spokvulcan/mlx` `pin-tesseract`; pins moved (see scheme doc).

**Harness amendments (from C1, non-numeric):** (1) `parity-ab.sh` gained
per-arm env injection (`ARM_ENV_baseline/experiment`, via `open --env`) —
for the C2 op-cap probe. (2) `parity-ab.sh` gained a per-arm **watchdog**
(`ARM_TIMEOUT`, default 600 s): a dense-leg arm completed its bench but
never exited (idle in the AppKit run loop, report unwritten — *baseline*
binary, one-off flake, not the experiment) and `open -W` parked 34 min;
the leg was killed and re-run. Lesson recorded mid-flight: the orphaned
watchdog `sleep` inherits the script's stdout pipe and delays `tail` EOF —
watchdog output is now redirected away from the pipe; orphan sleeps were
killed to unblock the in-flight leg. Also on record: **do not edit
`parity-ab.sh` while a run is parked inside it** — bash re-reads the file
and a mid-run edit shifted offsets, producing a syntax error at the loop
tail (data intact; the script's own footer died).

**C2 — `MLX_MAX_OPS_PER_BUFFER` raise (M2 probe) — REJECTED.** Hypothesis:
raising the per-command-buffer op cap (default 50 on M3 Max) reduces the
decode command-buffer segmentation E10 measured (~40 CBs/token, ~60 µs
gaps ≈ 22% idle) → decode win, no numerics (scheduling only). Method:
same-binary A/B via `ARM_ENV_experiment` (no rebuild — env is read once
per process), 3 rounds cap=400 + 2 rounds cap=200, contexts 128/8192,
both models. Gates: **PASS everywhere** (12/12 + 8/8 + 8/8 + 8/8,
token-identical — scheduling is output-neutral as expected). Numbers
(per-round pairs): **MoE decode +2.9…+4.4% at 8K (6/6 positive at 400,
mean +3.5%), +0.9…+2.1% at 128** — the E10 mechanism confirmed as real.
BUT three reproducible regressions kill the global knob: (a) **dense 8K
decode −2.7% at 400 (6/6 negative)**; (b) **dense 128 peak +7.25%**
(3.26→3.50 GB, +240 MB — 10/10 pairs across BOTH cap values; temporaries
held across a whole large buffer instead of released at 50-op
boundaries); (c) **MoE 128 prefill −4…−5%** (9/10 pairs across both
caps; opposite-sign from dense-128 prefill, so not pure noise —
mechanism unidentified). 200 weakens the MoE win (+1.9%) without
clearing (b)/(c). **Verdict: REJECTED at 200 and 400.** The MoE decode
win is real but every global form of it carries a reproducible ≥1%
regression. Recorded follow-ups (folded into M2's roadmap entry): a
graph-size-aware cap (win zone = ~1900-kernel MoE decode steps,
regression zones = small graphs) or mid-buffer temporary release (kills
the +240 MB) — eval.cpp/device.cpp internals, a deliberate project, not
an env knob. Note: a model-scoped policy (MoE-only cap) still fails on
regression (c).

**C3 — `gather_qmv` results-per-simdgroup geometry (M3 probe) — REJECTED
at probe, no app run.** Hypothesis: the decode gather_qmv kernels (E10:
51 µs / 88 GB/s at the B=8 decode shape) are latency/geometry-bound;
raising rows-per-CTA (rps 4→8/16/32, per-row arithmetic untouched →
bitwise by construction) lifts them. Method: probe-only `MLX_GQMV_RPS`
env hook + rps template param (quantized.h AND `mlx-generated/
quantized.cpp` — the two-homes rule), production decode shapes
(gate/up N=512/K=2048, down N=2048/K=512, f16), ABBA in-process.
**Bitwise IDENTICAL at every rps** (gate confirms the by-construction
argument). Numbers: rps=4 → **13.7 µs / 306 GB/s** at BOTH shapes in
isolation; rps=8: +0.0/+3.5%; rps=16/32: flat-to-negative. No ≥1%
geometry lever. Two harness traps found and fixed en route (reusable):
(a) eval-per-call on ~50 µs kernels floors at **~220 µs of CPU dispatch
per call** — single-graph-of-N-calls is the only honest way to time
kernels this small; (b) looping identical index sets goes **cache-hot**
(8 experts ≈ 34 MB stays in the system cache → a false 500 GB/s);
disjoint expert sets cycling all 256 (134 MB working set) are mandatory
for decode-realistic weight traffic. **Verdict: no kernel lever — the
kernel already runs at the machine's ~300 GB/s DRAM envelope in
isolation; E10's 88 GB/s does not reproduce outside the production eval
environment.** The MoE decode 2× gap (E9/E10) is therefore eval-
environment (M2-class: scheduling/overlap), not gather_qmv geometry —
M3 amended in the roadmap. Probe hooks stay uncommitted in the fork
clone; nothing reached the app or the pins.

**C4 attribution (M9 confirmed; basis for the C4 experiment).** Three
measurements, all on the c1-accepted build:

1. **Enqueue probe** (`/tmp/gather-sweep c4`, batches of 1900 decode-shape
   gather_qmv + one eval + one sync per batch): graph-build **0.6 µs/call**,
   eval-enqueue **13.0 µs/call**; dispatch happens **inline on the calling
   thread** (the MLX StreamThread is idle — no cross-thread handoff). In the
   probe the GPU keeps pace (13.7 µs DRAM-bound kernel), so 44% of wall is
   throttle waits; pure CPU dispatch ≈ 4.6 µs/op, of which kname building +
   pipeline lookup (fmt/get_template_definition) is only **~9%** — below the
   20% bar, so pipeline-state caching is **not** the C4 lever.
2. **Production decode sample** (`sample` on the parity bench, 35B MoE,
   ctx 8192, steady generation; generation thread = 4519 samples ≈ 13.2 ms
   token): **50.1% inline per-op C++ dispatch** (eval_impl:237 subtree), of
   which gpu::eval 32.9%; **~28% Swift-side** (graph build + sampling +
   detok in TokenIterator.next); **8.4% GPU-throttle wait** (eval_impl:252);
   3.8% finalize. No single frame >5% — the tax is spread across ~15 sites
   (primitives, encoder, allocator, fence, graph machinery); event machinery
   ≈0.4% (not a lever); CustomKernel cost is ordinary encoder+barrier work
   (the per-call full-source string compare does not show). **M9 confirmed:
   decode is ~85% CPU, GPU mostly idle.**
3. **Commit anatomy** (env-gated counters in the DerivedData checkout —
   `MLX_COMMIT_STATS`, probe-only, reverted after): caps 50 ops/50 MB
   (arch applegpu_g15s), all commits on stream 0.
   - **MoE decode: ~37 mid-commits/token, MB-cap-bound** — 50 MB of unique
     input bytes (weight slices) every ~20–24 ops; mid = 71% of commits,
     rest finalize/throttle tail. ~10 µs CPU per commit ⇒ ~2.8% of decode
     in mid-commit overhead alone, plus GPU cbuf-boundary gaps (E10's 22%
     idle estimate).
   - **Dense decode: ~18 mid-commits/token, OPS-cap-bound** (50 ops at
     ~27 MB unique bytes; the sub-cap averages in earlier readings were
     dilution by small finalize tail-commits).
   - App evals once per ~2 tokens (ops/eval ≈ 4350 MoE — convertToToken's
     item() covers the forward graph; asyncEval covers cache state).
   - **C2 reinterpretation:** C2's ops-cap raise could not have changed
     MoE mid-commit cadence (MB-bound) ⇒ its MoE "+3.5% decode" and
     "−4.5% 128-prefill" were very likely systematic artifacts, not cap
     effects; only the dense effects (ops-bound: decode −2.7%, peak
     +240 MB) were mechanistically real.

**C4 env probe (`MLX_MAX_MB_PER_BUFFER` 50→200, same-binary ABBA, 3 rounds ×
128/8K/32K, both models) — flat knob REJECTED, split-cap (v2) in progress.**
MoE: **decode +8.75% (128), +8.70% (8K), +6.93% (32K), 6/6 everywhere**;
prefill 128/8K flat (+0.2/+1.0); BUT **peak +7.18% (8K: 20.72→22.21 GB),
+7.91% (32K: 21.91→23.64 GB), 6/6**, and 32K prefill −2.02% (5/6). Dense:
decode −0.5% (128) / −1.4% (8K) / **−7.4% (32K, 6/6)**; prefill −3.0% (32K);
**peak +45.8% (8K: 4.61→6.72 GB), +33.5% (32K), 6/6**. Instrumented anatomy
of the experiment arm: MoE mid-commits 37→**20/token** (ops-cap 50 binds at
~37 ops/84 MB before 200 MB is reached) — the +8.7% decode win is mostly
NOT per-commit CPU (~1.3% worth); it is **~60–68 µs of GPU-side pipeline
drain per cbuf boundary** (matches E10's ~60 µs gaps), i.e. fewer+bigger
cbufs keep the GPU fed in CPU-bound decode. Dense 32K decode is different:
GPU-bound (weights 2.2 GB + full KV re-read ≈ 12+ ms GPU of the 18.5 ms
step) — bigger cbufs starve the GPU between chunks (CPU must build a chunk
before the GPU starts it), hence −7.4%. **The flat MB knob is REJECTED**
(peak regressions alone disqualify it on both models). v2 design: split the
accounting — commit on `ops > 50 || unique output (temporary) bytes > X ||
unique input (mostly persistent weight) bytes > 200`; prefill temporaries
stay on today's cadence (peak protected), decode weight-traffic stops
forcing boundaries (MoE win preserved), dense decode untouched (ops-bound).
X sized from measured output-bytes-per-commit (next measurement); the dense
32K decode regression is expected to vanish with peak fixed (pool-pressure
hypothesis) and is re-checked by the v2 A/B.

**C4 v2 A/B (`in200 | out50 | ops50`) — decode win holds, peak halved but
still reject-level.** MoE: **decode +7.40% (128), +9.72% (8K), +15.41%
(32K), 6/6**; prefill flat (+0.6/+0.8/−0.4); **peak +3.76% (8K), +4.62%
(32K), 6/6** (down from +7.2/+7.9). Dense: 32K decode recovered to +3.08%
(noisy 4/6 — the flat-200 −7.4% was pool/peak-pressure, not scheduling);
8K decode −0.48%; **peak +11.64% (8K: 4.61→5.14), +17.69% (32K:
6.10→7.18), 6/6** (down from +45.8/+33.5). **Mechanism located via
active-memory trajectory ticks** (`activeMB` in the commit probe): decode-
phase active memory is IDENTICAL across arms (MoE 8K: 18464 vs 18473;
dense 32K: 3710 vs 3711) — the entire peak regression is **prefill-phase
live temporaries** (v2 prefill commits ~2× fewer: MoE 8K +298 MB at tick,
dense 32K chunks +300-700 MB). `runPeakGB` = MLX active-memory peak.
Output-bytes at stock commit points: MoE 8K prefill ≈ 9.7 MB, MoE 32K ≈
16–37, dense 32K ≈ 27–55 (prefill chunked at 1024 → per-op outputs are
0.5–4 MB; MoE decode ≈ 0.075 MB/op, dense ≈ 0.2–0.3). **v3: `out10`**
reproduces stock's prefill commit points at every context (slightly
tighter at 32K — harmless), while decode stays `in200`-driven (MoE out10
binds at ~133 ops ≈ 15/token, in200 at ~80 ≈ 24/token — the win zone).

**C4 v3 A/B (`in200 | out10 | ops50`) — MoE fully clean, dense 32K decode
kills it.** MoE: **decode +8.50% (128), +10.58% (8K), +3.78% (32K), 6/6**;
prefill flat/positive; **peak −1.74% (8K), −1.44% (32K) — improved**;
gate 18/18 IDENT. Dense: **peak −13.46% (8K: 4.61→3.99 GB), −9.04% (32K),
−2.16% (128)** (out10's tighter prefill cadence — a real bonus); decode
128/8K −0.4/−0.5% (6/6, sub-1%); **32K decode −1.92% (8/10 over a 5-round
resolution run)** — reproducible: dense 32K decode is GPU-bound (weights
2.2 GB + KV re-read ≈ 12+ ms GPU of the 18.5 ms step) and `in200`'s ~4×
coarser FFN-driven commits starve the pipeline. **v3 REJECTED.**
**v4 (GPU-bound-adaptive in-cap) — REJECTED at probe.** Two detectors
tried: completion-lag (relax when last cbuf completed <T µs ago — feedback
oscillation: relaxed cbufs are intrinsically slow to complete, the regime
un-detects itself; MoE mid/token 37→42, worse than stock) and
queue-depth hysteresis (relax ≤2, tighten ≥6 active tasks — MoE decode's
equilibrium queue depth sits at 3–6, never relaxes: mid/token ≈ 60,
tok/s = stock). **Physics: MoE decode is boundary-limited, not
GPU-throughput-limited — the GPU is busy either way, so no GPU-side
signal separates it from dense 32K's starvation-limited regime.** A
phase-accurate signal (prefill vs decode) exists only in the app/library —
out of Cmlx scope. **v5: static compromise `in100 | out10`** (dense FFN
29 MB/op → commits ~1.7× coarser than stock vs ~4× at in200, halving the
starvation; MoE ~40 ops/commit ≈ 27 mid/token, keeps ~half+ of the v3
win) — A/B running.

**C4 v5 (`in100 | out10 | ops50`) — ACCEPTED.** Same-binary ABBA (3
rounds MoE, 4 rounds dense, 128/8K/32K, gates 18/18 + 24/24
token-identical). MoE: **decode +2.63% (128), +4.50% (8K), +2.36%
(32K)** (6/6, 6/6, 4/6); prefill flat (outliers are round-1 warmup);
**peak −1.74% (8K), −1.44% (32K)**. Dense: **32K decode +4.19% (5/8 —
the v3 −1.92% gone)**; 128/8K decode −0.4/−0.1% (flat); prefill
flat/positive; **peak −2.16% (128), −13.46% (8K: 4.61→3.99 GB), −9.19%
(32K: 6.10→5.54 GB)** — the out10 leg is a peak-memory win in its own
right. Ported to `spokvulcan/mlx` `pin-tesseract` @ **404070e2**
(`perf(metal): relaxed input cap + output-byte commit accounting (C4)`),
mlx-swift pin @ **73e7f42**, three Package.swift pins in lockstep;
checkout re-sync verified `diff fbf2fb86 == C4 patch` exactly; probe
instrumentation fully reverted. Defaults shipped: ops 20/40/50 (arch,
unchanged), **in 100 MB, out 10 MB** (ctor, env-overridable). Clean-build
confirmation A/B (pinned build vs `tesseract-c1-accepted.app`, 3 rounds
128/8K/32K + a 5-round MoE 32K resolution): MoE decode **+2.45% (128,
6/6), +4.97% (8K, 6/6), +0.93% (32K, noise-dominated ±5)**; MoE prefill
noise (32K −0.53% mean of 10, 3/10 — the earlier −6.1% and +7.3% readings
were both thermal outliers); dense decode flat at every context
(+0.06% 32K); peaks −1.7/−1.4% (MoE 8K/32K), −2.2/−13.5/−9.0% (dense);
gates 18/18 + 10/10 + 18/18 token-identical. **The 32K-context prefill
and decode metrics on this machine carry ±5-10% thermal variance —
verdicts there need ≥5 rounds and per-round pairing, never single runs.**

**C5 — per-cbuf buffer-retention coalescing — ACCEPTED (as C5b, no
dedup).** Attribution (production decode sample, line-level): per-op
retention scaffolding in `gpu::eval` ≈ **8.5% of the decode generation
thread** (completion-block per op `eval.cpp:68` 4.1%, retention-set
inserts `eval.cpp:47` 2.5%, outputs copy 1.8%, plus disposal on Metal
completion queues). Change: ops push input/sibling Data ptrs into the
stream's pending vector (skipping donated inputs, exactly the old set's
semantics); the batch flushes as **one completed handler per command
buffer at commit** (`Device::commit_command_buffer` is the single
funnel). Attach point = the same cbuf the ops were encoded in → release
timing identical by construction. First form included a sort+unique
dedup at commit: **REJECTED by the data** (dense 128 decode −1.22%,
6/6 — the per-commit sort costs more than the per-op hashing it saved
on commit-dense decode); dropping the dedup restored dense to flat
(duplicate refs die together in the same handler — cosmetic only).
Final numbers (3 rounds 128/8K/32K both models + a 5-round MoE 32K
resolution, gates 18/18 + 18/18 + 10/10 token-identical): **MoE 8K
decode +3.92% (5/6)**, MoE 128 +0.93% (noise), MoE 32K −0.45% mean of
10 (noise), dense flat everywhere, **peak memory exactly unchanged**
(19.51/20.36/21.59 and 3.19/3.99/5.54 — semantics preservation
verified). Ported @ `spokvulcan/mlx` **8d11dd1d**, mlx-swift pin
**5c16b28**, mlx-swift-lm pin **98e9e28**.

**Harness amendment (user directive, 2026-07-24): default A/B is now
3 pairs per context per model** (`BENCH_RUNS=1` × 3 rounds — script
takes `BENCH_RUNS`, default still 2). Escalate to 5 rounds × 2 (10
pairs) only when a verdict-relevant metric lands inside the noise floor
(32K decode/prefill almost always do). Cutting rounds indiscriminately
on 32K would have mis-verdicted C4/v3 and C5b twice each.

**C6 — custom-kernel (kernel_name, kernel_source) memoization —
ACCEPTED.** Attribution (post-C5 production MoE 8K decode sample, 2862
gen-thread samples): `gatedDeltaUpdate`'s `MLXFastKernel` call =
72 samples (2.5% of the thread), of which **~46 in `std::regex`
construction + `regex_replace`** — `metal_kernel`'s closure rebuilt
`kernel_name`/`kernel_source` on every call (every token × every GDN
layer, both models are GDN hybrids) while the compiled MTLLibrary is
already device-cached. Same sample, updated landscape for the queue:
eval_impl tape machinery 41.5% (2089 under async_eval minus 901
gpu::eval), gpu::eval op dispatch 31.5%, Swift graph build 26.6%;
per-boundary costs shrunk to end_encoding 42 + commit 48 +
get_command_encoder 80 samples (≈5.9% total, recoverable fraction
smaller). Change: memoize the generated (kernel_name, kernel_source)
per call site (cache captured in the closure); key = template_args +
per-input dtype/ndim/size-class (write_signature's `size() < 8`
address-space branch) + output_dtypes — everything else the strings
depend on is closure-fixed, so a hit is byte-identical by construction.
Zero numerics. A/B (3 rounds 128/8K/32K + 10-pair 32K resolutions,
both models, gates 9/9 + 9/9 + 10/10 + 10/10 = 38/38 token-identical):
**MoE decode +3.66% (128), +3.11% (8K), +3.55/+4.67% (32K)**; MoE
32K prefill **+1.72%** (the 3-pair −1.51% reading was thermal noise);
dense 128/8K flat (+0.17%), dense 32K +1.39/+0.07% (the 3-pair
+20.87% was a throttled baseline round — resolution protocol caught
both); **peaks exactly flat everywhere**. Ported @ `spokvulcan/mlx`
**3ec72a24** (`perf(metal): memoize custom-kernel source generation
(C6)`), mlx-swift pin **99e27254**, mlx-swift-lm pin **cbeb6ee**;
checkout re-sync verified `diff fbf2fb86 == C4+C5+C6` exactly, no
local mods.

**C7 — per-model commit policy (app-signalled regime) — ACCEPTED.**
C4/v3 (`in200`) measured +8.50/+10.58/+3.78% MoE decode but was
REJECTED for dense 32K (−1.92%); C4/v4 proved no GPU-side signal can
separate MoE's boundary-limited decode from dense 32K's
starvation-limited one. The app knows the model — and app-side entered
scope 2026-07-24. Change (full stack): mlx caps become runtime-settable
(`std::atomic` members + `Device::set_commit_limits` + namespace
wrapper + `extern "C" mlx_metal_set_commit_limits`; a 0 leg is left
unchanged); mlx-swift exposes it (`mlx/c/commit_limits.h` in the Cmlx
umbrella + `GPU.setCommitLimits` shim); `LLMActor.loadModel` calls it
on every load keyed off the existing `ModelIdentity.isMoE`
(`qwen3_5_moe`): **MoE → 200 MB input cap, dense → 100 MB** (setting
on every load keeps MoE↔dense switching correct). Commit points are
scheduling boundaries only — commit-point invariance already gated in
C4. A/B (3 rounds 128/8K/32K, gates 9/9 + 9/9 + 10/10
token-identical): **MoE decode +5.89% (128), +5.86% (8K), +3.67%
(32K)** — matching the +5.9/+5.3/+2.8 prediction from the C4 v3-vs-v5
delta; prefill flat; peaks flat (+0.22% at 128, noise). Dense
128/8K/prefill/peaks exactly flat (the dense arm passes the
compiled-in default — no mechanism for an effect); dense 32K decode
3-pair −5.67% did NOT reproduce in the 10-pair resolution (+5.3/+11.8,
opposite sign) — machine was thermally throttling hard (absolute dense
32K throughput swung 70→40→23 t/s across the afternoon); noise, not
regression. Ported: `spokvulcan/mlx` **6ab29e36**, mlx-swift pin
**1069e872** (also carries the `GPU.setCommitLimits` + header),
mlx-swift-lm pin **b3a4b41**; checkout re-sync verified `diff
fbf2fb86 == C4+C5+C6+C7` exactly, no local mods.

**C8 — eval_impl per-token hash-map machinery — ACCEPTED.** The DFS
degree pass + BFS tape build performed several `std::unordered_map`
operations per graph edge per eval (profiles had the walk at ~18% of
the decode generation thread excluding waits); the tape loop also did
a per-node `open_streams` insert + `events` map lookup (hundreds of
nodes back-to-back on ONE stream during decode) and per-input
`needs_fence` probes against an almost-always-empty map. Change: flat
open-addressing id→degree map (Fibonacci-hashed power-of-two slot
array, tombstone deletes, probed by key only — tape order unchanged),
last-stream guard for the open_streams/events work, `needs_fence`
empty fast-paths. Same walk, same tape, zero numerics. A/B (3 rounds
both models, gates 9/9 + 9/9 = 18/18 token-identical): **MoE decode
+1.98% (128, 3/3: +1.10/+2.59/+2.29), +1.43% (8K, 2/3 + one
−0.07%)**; dense 128/8K flat (+0.70/−0.01%); prefill flat; **peaks
exactly flat**; 32K deltas positive but throttled-regime (absolute
13–27 t/s — machine thermally saturated, not verdictable, and not
needed for the verdict). Ported @ `spokvulcan/mlx` **595a3fe1**,
mlx-swift pin **0b3289cb**, mlx-swift-lm pin **b5eb5ef**; checkout
re-sync verified `diff fbf2fb86 == C4..C8` exactly, no local mods.

**M8 — expert-weight prefetch — REJECTED at probe (routing locality
does not exist).** Instrumented `Qwen35SparseMoeBlock` with a
throwaway capture hook (TESS_PROBE_ROUTING; per-layer top-k indices
held lazily, evaluated off the hot path) and measured consecutive-token
expert-set overlap on a real 256-token decode (MoE 35B-A3B, 128 ctx):
**mean overlap 2.4/8, median 2.48/8, exact-set rate ≈ 0.0** across all
40 MoE layers (min layer 0.63/8). A previous-token prefetch would warm
~70% wrong weights — pure wasted bandwidth on a bus that is already
the decode bottleneck. No kernel work built; probe reverted, tree
clean. Do not re-probe without a different prediction signal (router
logits trajectory, not set identity).

**C9 — gather_mm/gather_qmm identity-index cache — ACCEPTED.**
Attribution (C6 decode sample census): `Arange::eval_gpu` = 45/2862
gen-thread samples (1.6%) — `indices_or_default` (ops.cpp) rebuilt
`arange+reshape` identity row indices on EVERY gather call with no
explicit lhs indices: `QuantizedSwitchLinear` passes only rhs expert
ids, so 3 gathers × 40 MoE layers ≈ 120 Arange dispatches + ~240 tape
nodes per decode token (and per prefill chunk: arange(1024) × 120 × 32
chunks). Change: cache the evaluated array per shape (bounded 64-entry
map, FNV-1a key — the first string-key form ate ~0.5% itself,
measured; mutex-guarded). Constant leaf, read-only consumers, zero
numerics. A/B: 3-pair leg muddy (8K −1.27% mixed) → 10-pair
escalation at 128/8K (gate 20/20; first leg's gates 9/9 + 9/9): **MoE
8K prefill +3.57/+3.55%** (both run blocks, prompt s −3.37/−3.35),
**8K decode +0.69/+1.99%**; 128 decode 5/5 split mean −0.35% (noise,
no reproducible regression); 128 prefill mixed (0.17 s legs, noisy);
dense unaffected (path unused), peaks exactly flat. Ported @
`spokvulcan/mlx` **625f2aea**, mlx-swift pin **c9796ec4**,
mlx-swift-lm pin **f72302c**; checkout re-sync verified `diff
fbf2fb86 == C4..C9` exactly, no local mods.

**C10 — metadata-only primitive fast path — REJECTED (CPU slack: the
saving is real but not pipeline-critical).** Motivated by the op census
(~4,400 ops per MoE decode token; Transpose 323 + ExpandDims 88 +
Squeeze 92 + Contiguous 32 ≈ 535 view ops/token whose `eval_gpu` is
pure stride/flag metadata), the tape loop inline-evaluated the 8
verified metadata-only primitives (Transpose, ExpandDims, Squeeze,
Split, Broadcast, BroadcastAxes, Copy, StopGradient — all delegate
`eval_gpu` to the common metadata `eval`; Slice/View/Reshape excluded:
they can dispatch copies) and skipped the `gpu::eval` scaffolding.
Verified non-effects before benching: `buffer_ops` increments only on
kernel dispatch (never views), so commit cadence is untouched;
retention is redundant by producer. A/B (3 rounds + 10-pair
resolutions, gates 9/9 + 9/9 + 10/10 + 10/10 token-identical, peaks
flat): MoE 8K decode **−0.43/+0.07 (10 pairs) — no win**; dense 32K
decode 3-pair −3.84% (3/3) did NOT reproduce (10-pair +11.5/+1.45 —
the same thermal-noise class both directions). **Lesson logged for the
loop: post-C4..C9 the 8K decode CPU has slack — spread-out CPU-only
cuts no longer convert to tok/s. Remaining decode wins must shorten
the GPU serial chain (fewer/smaller kernels — fusion in the
E2-bitwise class) or the commit boundaries.** Reverted completely.

**Op census (TESS_OP_CENSUS probe, since reverted — MoE 35B decode,
~4,400 dispatched ops/token):** Matmul ~280, CustomKernel ~258 (GDN
scan + rotations), GatherQMM ~129, QuantizedMatmul ~140, view ops
~535, raw elementwise (Multiply 194, Add 130, Sigmoid 86, Sum 86,
Divide 43) ~540, already-compiled segments ~305, Transpose 323,
Softmax ~43, ArgPartition ~33, Convolution ~32, SliceUpdate ~21,
Arange ~0.25 (post-C9). The elementwise soup + its view-op entourage
is the largest remaining fusion target; CPU-side per-node cost is no
longer the lever (see C10 lesson).

**C11 — compiled MoE block during decode (E2 fusion class) —
ACCEPTED.** The op census' largest remaining class was ~540 raw
elementwise kernels/token; post-C10's lesson (spread-out CPU cuts
don't convert) the mechanism here is **GPU serial-chain shortening**:
`Qwen35SparseMoeBlock` decode now runs through a per-instance
`compile`d closure (router takeAlong/sum/divide, shared-expert
sigmoid+multiply, residuals fuse; matmuls/gathers/custom kernels tape
through unchanged). First form compiled all shapes: **128 prefill
−5.43%** (one-time compile-trace on a 0.17 s leg; 8K/32K prefill were
flat) — final form compiles **L==1 only** (prefill is GEMM-dominated,
fusion measured +0.3% there). A/B final (3 rounds both models, gates
9/9 + 9/9 token-identical, peaks exactly flat): **MoE decode +5.16%
(128, 3/3: +3.83/+8.12/+3.68), +2.99% (8K, 3/3: +2.50/+3.48/+2.98),
+7.28% (32K)**; prefill flat (+0.49/−0.05/+1.44); dense flat
(unaffected path). This is a Vendor/mlx-swift-lm change (no Cmlx
diff): committed on `pin-upstream-mlx-swift` @ **3bb0f17**; Cmlx pins
unchanged (mlx 625f2aea, mlx-swift c9796ec4). Opens C12+ for the same
pattern on the attention + GDN blocks (cache state must become
inputs/outputs first — GDN is 30/40 layers and the biggest block).

**C12 — compiled GDN decode step with explicit state — ACCEPTED.**
`Qwen35GatedDeltaNet` decode (S==1, unmasked, cached) runs through a
per-instance compiled closure; conv/recurrent state crosses as
inputs/outputs (compiled functions must be pure — first decode token
falls back to explicit zero states matching `gatedDeltaUpdate`'s
internal init). Elementwise chains (conv-silu, gating, norms) fuse in
the E2-bitwise class. Prefill/masked/cacheless keep the unfused body,
so prefill is byte-identical (the 3-pair 128-prefill +21% and dense
−2.83% readings were thermal noise by construction). A/B: 3-pair leg
muddy (8K −0.69%) → 10-pair escalation at 128/8K both models (gates
20/20 + 18/18 token-identical, peaks exactly flat): **dense 128
decode +1.75% (10/10), MoE 128 decode +0.94% (10/10)**; 8K decode
flat both models (−0.3/+0.3% means). MoE's smaller win vs C11 is the
`compile_replace` replay cost over the GDN block's ~20-node tape × 30
layers — logged as the limiting factor for further block compiles.
Committed on `pin-upstream-mlx-swift` @ **e77d05d**; Cmlx pins
unchanged.

**M4 — fused rotate+dequant+dot — REJECTED at probe (geometry, not
numerics).** Probe in `/tmp/gather-sweep` (coder subagent, full
harness + gates): the fused kernel (rotation phase into a bf16
threadgroup tile + verbatim `qmv_fast_impl` body) is **bitwise
IDENTICAL to the two-kernel pipeline on the first attempt** — 3 seeds
× K∈{2048,2560} × rps∈{4,8,16,32}, all IDENT; both phases isolate-
clean. The numerics worry was unfounded because MLXFast JIT compiles
with `fastMathEnabled(false)` (device.cpp:619): identical expressions
→ identical codegen, no fma tuning needed. **But fused is ~2×
SLOWER** (ABBA, 32 disjoint weight sets): qmv's N/8 = 256-threadgroup
grid makes every threadgroup redundantly re-rotate the full K vector
(16 groups × 8 rounds of barrier-separated tile math + ~72KB
coefficient re-reads per threadgroup ≈ 18MB L2 traffic vs 2MB
weights); the standalone rotation does the same work across 16
threadgroups, fully latency-hidden (~2µs). Only rps=32 (non-production
geometry) reaches parity. For MoE it's strictly worse (rotations
shared across 8 experts → 8× multiplier). **The two-kernel pipeline
is the right design; do not revisit unless geometry changes.**
Meta-lesson banked: **verbatim arithmetic in MLXFast JIT reproduces
bitwise trivially** — unlocks M5-class fused-kernel replications
(mask+softmax) that were previously rated risky.
Artifacts: `/tmp/gather-sweep/Sources/gather-sweep/main.swift`,
`m4-fused-kernel.metal`, logs (rig preserved, nothing committed).

**C13 — fused causal-mask + softmax for the SDPA ops fallback (M5) —
ACCEPTED.** Probe-driven (`/tmp/gather-sweep`, agent-0): ceiling
measured first — the causal chain (greater_equal → where(bf16
finfo.min) → softmax precise) costs ~6.5 ms per 32K layer-chunk, of
which mask+where ≈ 3.0 ms (≈5% of 32K prefill at 320 layer-chunks);
the fused kernel (verbatim looped-softmax precise body, N_READS=4,
causal select injected at both load sites) lands exactly on the
softmax traffic floor (3.5 ms, −45% at 32K, −61% at 8K) and is
**bitwise IDENT on 14/14 configs** (3 seeds × S∈{8192,32768} ×
offsets). Bitwise enabler: the masked value is exact bf16 finfo.min as
f32 and exp(min−max) underflows to exact 0; threadgroup lsize must
equal the production dispatch (1024). Port: `fast.cpp` fallback
lambda, predicated on (do_causal, no array mask, no sinks, bf16,
axis>4096, row-contiguous) — block-softmax sizes, sinks, array/additive
masks keep the stock chain. Also removes the 512MB masked-scores
intermediate + 32MB bool mask per layer-chunk. App A/B (3-pair +
two 10-pair sets incl. post-cool-down, **gates 48/48 token-identical**,
peaks flat): MoE 32K prefill **+2.80% mean of 20 pairs** (11/9 rounds,
median +0.71% — thermal saturation; accepted on the strictly-
subtractive-mechanism argument: the change only removes two memory
passes, no regression channel exists, and the app mean matches the
probe-predicted +3%); 128/8K flat (fused engages kL>4096 only);
dense flat. Ported @ `spokvulcan/mlx` **ed107a94**, mlx-swift pin
**e20b0d86**, mlx-swift-lm pin **8eeec20**; checkout re-sync verified
`diff fbf2fb86 == C4..C13` exactly.

### Operational state (persisted for context compaction; reload after resume)

- **Probe rig:** `/tmp/gather-sweep` — SwiftPM executable, local-path dep on
  `~/projects/mlx-swift`; needs `default.metallib` copied next to the binary
  as `mlx.metallib` (from the app bundle's `mlx-swift_Cmlx.bundle`). Sections:
  fidelity + B/E sweep (`MLX_GQMM_CFG`), down_proj shape, dense anchor,
  gather_qmv decode sweep (`MLX_GQMV_RPS`). Rebuild: `swift build -c release`
  (seconds — incremental Cmlx).
- **Fork clone state (standing, do NOT clean):**
  `~/projects/mlx-swift/Source/Cmlx/mlx` = `ed107a94` + uncommitted probe
  hooks — `MLX_GQMM_CFG` env in `gather_qmm_rhs`; `MLX_GQMV_RPS` env +
  rps template param (`quantized.h` AND `mlx-generated/quantized.cpp`) +
  rps dispatch in `gather_qmv`. All marked PROBE ONLY; never pushed.
  `~/projects/mlx` = clean at `ed107a94` (pin-tesseract tip).
- **App binaries (/tmp):** `tesseract-precmlx-baseline.app` (pre-fork),
  `tesseract-cmlx-fork.app` (C0 fork build, pre-C1), `tesseract-c1-accepted.app`
  (C1 tiles, fbf2fb86), `tesseract-c4.app` (C1+C4, 404070e2),
  `tesseract-c5-accepted.app` (C1+C4+C5, 8d11dd1d),
  `tesseract-c6-accepted.app` (…+C6, 3ec72a24),
  `tesseract-c7-accepted.app` (…+C7, 6ab29e36),
  `tesseract-c8-accepted.app` (…+C8, 595a3fe1),
  `tesseract-c9-accepted.app` (…+C9, 625f2aea),
  `tesseract-c11-accepted.app` (…+C11, 3bb0f17),
  `tesseract-c12-accepted.app` (…+C12, e77d05d),
  **`tesseract-c13-accepted.app` (current main: C1+C4..C9+C11..C13,
  ed107a94) — the A/B baseline for the next experiment.**
- **Pins (current):** spokvulcan/mlx-swift `e20b0d86` (pin-tesseract) ←
  spokvulcan/mlx `ed107a94`; mlx-swift-lm pin branch `8eeec20`.
- **Build checkout:** the app target's DerivedData is
  `~/Library/Developer/Xcode/DerivedData/tesseract-buwysfpnwmzyucelgewutuddcvgv`
  (several stale siblings exist; that one is current). Checkout files are
  read-only — `chmod u+w` before patching.
- **Measurement protocol (2026-07-24):** default A/B = **3 pairs** per
  context per model (`BENCH_RUNS=1` × 3 rounds); escalate to 10 pairs
  only when the signal is inside the noise floor (32K decode/prefill
  almost always are — never verdict a 32K metric on one run).
- **Next (C11+):** M8 REJECTED at probe; C10 REJECTED (CPU slack —
  spread-out CPU cuts stopped converting post-C9; aim at the GPU
  serial chain). Op census logged above (~4,400 ops/MoE-token; raw
  elementwise ~540/token = the fusion target). Queue: elementwise-soup
  fusion in the E2-bitwise class (compile() more of the per-layer
  chains — router top-k normalize, GDN gating, attention gate; GPU
  serial-chain shortening is the mechanism), M4 (fused rotate+GEMM,
  ~3-4% prefill, high risk), M5 (fallback mask+softmax, ~1% prefill),
  M6/M7 deprioritized.

---

## Session 2026-07-25 — full-step graph caching (C14)

Base: main `3d1b15cc` (post-PR #426), pins mlx `a3673067` / mlx-swift
`457a0d6d` / mlx-swift-lm `68ad25f`. **The C13-era baseline binary is not a
valid base any more:** at mlx `ed107a94` the C13 fused causal-softmax kernel
declares its output `{"out"}` while the body aliases
`device bfloat16_t* out = y;`, so the generated source does not compile and
the fused path throws at first dispatch (verified by reading both
revisions). PR #426 fixed it (`{"y"}`). A/B baseline for this session is a
fresh Release build of `3d1b15cc` (`/tmp/tesseract-c14-base.app`).

### C14 attribution — decode is GPU-paced, so the roadmap's premise is wrong

`sample` on the generation thread (Release, 1024-token decode runs, 9 s
windows; subtree totals with same-symbol nesting removed):

| | MoE 35B @8K | dense 4B @8K |
| --- | --- | --- |
| GPU-completion wait (`Scheduler::wait_for_one`) | **33.4%** | **13.8%** |
| op dispatch (`gpu::eval`) | 41.7% | 47.0% |
| Swift graph build (model forward) | 14.1% | 26.3% |
| detach/destructors | 2.9% | 4.2% |

That wait is `transforms.cpp:424` — `n_active_tasks() > MAX_ACTIVE_TASKS`
(=10, `transforms.cpp:26`) — i.e. the generation thread blocked *because the
CPU ran ahead of the GPU*. **Decode is GPU-paced on both models, with ~1/3
(MoE) and ~1/7 (dense) CPU slack.** The roadmap's item-1 premise (~25% Swift
graph build + ~40% eval walk *on the critical path*, decode 8K "~96 →
140–200 t/s") is therefore not the shape of the problem: removing CPU work
cannot pay more than the slack. C10's lesson, re-confirmed post-C13 and now
with the mechanism named. The 2× decode target needs GPU-chain shortening,
not graph caching — see the follow-up queue below.

### C14 — whole-step decode schedule — ACCEPTED (small)

Three milestones, each gated on its own; all landed together.

**(A) FA-layer purity refactor.** The attention decode body split into pure
functions around the cache write. **The concat form of "cache as input" was
killed by arithmetic before it was written:** materialising the grown cache
costs ~64 MB extra traffic per FA layer per token at 8K on the dense model
(kvHeads 4 × headDim 256 × 8192 × 2 B, read+write, K and V) ≈ 512 MB/token
over 8 layers ≈ 1.5 ms of a 10.5 ms step, 4× worse at 32K — no fusion pays
that back. Keeping `cache.update`'s in-place slice_update also keeps the
donation MLX depends on: `SliceUpdate::eval_gpu` always calls `copy_gpu`,
which donates only when `is_donatable()` (`array_desc_.use_count() == 1`,
`array.h:294`) — today's KV write is in-place *only* because Swift drops its
reference to the old buffer. Gate: PASS, 12/12 token-identical.

**(B) Per-layer compiled decode blocks.** The whole layer traced per
instance (norm → attention → residual → norm → MLP → residual). GDN layers
are one trace, subsuming C11+C12 *plus* the glue those left in Swift; FA
layers are two traces split at the cache write. **No shapeless compilation
is needed anywhere** — GDN state shapes are context-independent, so the SDPA
is the only shape-varying op and it stays outside; rope also stays outside
(its scalar offset moves per token and a trace would bake it). Gate PASS;
MoE 128 decode +1.08% (3/3).

**(C) Whole-step schedule.** `Qwen35TextModelInner.decodeStep` tiles the
step into segments running from just after one FA layer's SDPA to just
before the next one's: **11 traced segments for the MoE** (40 layers), 9 for
the dense, against ~40 per-layer traces in B and full Swift graph building
at base. Embedding opens the first segment, the final norm closes the last;
anything unusual (real mask, quantized/turbo cache, GDN layer with no state)
falls back. Engagement verified in the profile (`decodeStep` present,
per-layer `callAsFunction` absent), and it did what it was supposed to do to
the CPU: **Swift graph build 14.1% → 9.8%, GPU wait 33.4% → 37.3%** — i.e.
the CPU got slacker and the GPU pace did not move. **C added ~nothing over
B in tok/s, exactly as the attribution predicted.**

Numbers (10 pairs = 5 rounds × 2 runs, ABBA, vs base; token gates PASS
throughout — **108 pairs token-identical** across every run this session,
20 of them on the exact shipped binary per model):
**MoE 128 decode +1.33%** (8/8 positive excluding a cold first-arm round;
+2.33% over all 10), **MoE 8K decode +0.67%** (8/8 positive), **dense 8K
decode +1.73%** (7/8 positive on the shipped binary), dense 128 +0.31%
(5/8, flat), **peaks exactly flat on both models** (19.12/19.93 and
2.77/3.56), prefill untouched by construction (L > 1 keeps the old path).
**Verdict: ACCEPTED** — ≥1% reproducible on two metrics across both models
with sign consistency, nothing regressed. Vendor-only change; Cmlx pins
unchanged. Vendor suites green (ParoQuantTests 24/24, Qwen35 suites 11/11).
Absolute tok/s drifts down across a long bench session (the dense 10-pair
ran at 88–94 t/s where the morning's ran at 95–99) — only within-round
pairs are comparable, as the protocol says.

**Measurement artifact worth remembering:** on the dense model the *6th
consecutive app launch* of a run collapsed on every metric — including
code-identical prefill (911 → 685 t/s) — in 4 out of 4 three-round runs.
Three-round dense verdicts are unsafe on a machine already warm from
benching; the 10-pair form spreads the effect across both arms.

**Follow-up queue (aimed by the attribution above, not by the roadmap):**
the GPU serial chain is now the only thing that converts. Largest
identified in-scope class: **shared-input rotation batching.**
`RotateQuantizedLinear` dispatches rotate+qmm per projection, and the
projections of a layer that share an input (GDN `in_proj_qkv/z/b/a`,
attention q/k/v, the MoE block's router + shared expert) rotate the *same*
activation with different coefficient sets — ~511 rotation launches/token on
the MoE, of which ~310 are batchable into one dispatch per group by putting
the set index on the grid's z axis (per-element arithmetic untouched ⇒
bitwise by construction). At decode the rotation grid is 16 threadgroups ×
32 threads — pure launch latency. Probe first (a `rotbatch` rig section is
drafted: verbatim production body + a z-axis variant, bitwise gate, ABBA
over 32 disjoint activations), then port. This is *not* the logged "PARO
projection fusion" no-go (#257), which was about fusing the GEMMs — each
projection keeps its own rotation, they just share one launch.

---

## Review round 2026-07-24 — PR #425 full-diff review fixes

Two-agent adversarial review of the whole C1–C13 loop (PR #425) found
three defect classes the loop's gates structurally could not catch; all
fixed this round on `fix/cmlx-review-round-425` (tesseract) + new fork
commits. **A parity re-gate round (3-pair 128/8K/32K, both models,
escalation per protocol) is REQUIRED before the C13 win is re-banked**
(run same day — PASS, record at the end of this section) — the C13 fix
re-enables a kernel the merged tree could never run; all other fixes
are scheduling/lifecycle/docs-only.

**F1 — C13 shipped uncompilable (CRITICAL; latent crash, MoE kL>4096).**
The port to `fast.cpp` renamed the custom-kernel output to `"out"` while
the body kept the probe's alias `device bfloat16_t* out = y;` — `y`
undefined, `out` redeclared; the JIT-generated source cannot compile, so
the fused path threw at first dispatch on any MoE prefill/generation
crossing kL=4096 (the probe rig names its output `"y"`:
`gather-sweep/main.swift`; the benched binaries ran that text — the
measured +2.8%/48-48 gates are genuine, the *ported artifact* was
broken). Proven both ways by compiling the reconstructed generated
source: as-merged fails (`redefinition of 'out'`, `use of undeclared
identifier 'y'`), fixed compiles clean. Why the workflow missed it: the
post-port "checkout diff == accepted diff" check is tautological after
re-resolution, and C13 (unlike C4) never got a clean-build confirmation
run. Workflow amended in `docs/mlx-core-fork.md`: clean-build + smoke
round is now mandatory per accept. Fix `spokvulcan/mlx` **5ca82d9f**
(output renamed `"y"` to keep the body byte-identical to the
probe-proven text; predicate also hardened — the `row_contiguous` clause
was vacuous at graph build (ArrayDesc defaults flags true; the real
guarantee is CustomKernel's ensure_row_contiguous copy), fused path now
also excluded inside function-transform traces (CustomKernel has no
vjp/jvp/vmap), on non-Metal devices/builds, and on int32 grid overflow;
stock else-branch reindented).

**F2 — C11/C12 leaked the whole previous model on MoE↔dense switch
(HIGH).** The stored compiled closures captured their module strongly:
module → compiledForward/compiledDecode → CompiledFunction → closure →
module. The cycle also kept `CompiledFunction.deinit` from running, so
`mlx_detail_compile_erase` never freed the backend tape (weights baked
as trace constants). Fix mlx-swift-lm **be5a09b**: `[unowned self]`
(the closures live only on their module — cannot outlive it) + red/green
regression test `Qwen35CompiledDecodeLifecycleTests` (tiny hybrid
MoE+GDN model, two compiled decode steps, weak-ref dealloc assert:
fails on `[self]`, passes on `[unowned self]`; needs a colocated
`mlx.metallib` next to the xctest binary — same standalone-SwiftPM
gotcha as the probe rig).

**F3 — hardening batch (fork commits 90ec2bb9, a3673067 + app/docs).**
C9: exact-Shape cache keys (FNV was the *key*, not the hasher — a 2^-64
collision would silently return wrong indices), cache now publishes only
evaluated constants (no unevaluated node shared across threads/streams),
function-transform traces bypass it. C4: per-arch cap table respected
(Mac-class 'g'/'s'/'d' → 100, phone keeps stock 20/40; M3 Max behavior
unchanged), units documented honestly (both legs count data_size()
elements, not bytes — "MB" is upstream's misnomer). C7: duplicate
`private:` dropped. C5: retention vector re-reserves its capacity. C8:
signed stream sentinel. App: `GPU.setCommitLimits` moved to the
load-success store point (failed load keeps the resident model's
policy) and `unloadModel` restores the balanced default (the policy is
process-global — co-resident TTS runs under the resident LLM's caps,
scheduling-only, now documented at the call site). Docs: fork-doc
diagram de-staled + mandatory clean-build step; C11/C12 + review rows
added to `docs/mlx-swift-lm-fork.md`; future-work #5 got the 96%-vs-
40–50% anchor reconciliation caveat and new #8 (C6 hit-path copies —
deliberately NOT fixed blind; needs its own A/B, CustomKernel member
surface).

**Deliberately not changed** (review calls, rationale on the PR):
C6 hit-path string copies (primitive-surface refactor, own iteration);
per-lease typed commit policy for TTS (over-engineering, unmeasured —
lease serialization makes it scheduling-only); GDN compiled/unfused
body duplication (price of the compile pattern; a dedup refactor would
touch proven code without a gate); Apple copyright headers on
fork-authored files (upstream contribution convention); `extern "C"`
surface in libmlx (mlx-c submodule stays untouched by design).

**Probe rig preserved durably:** `/tmp/gather-sweep` (Package.swift,
sources, M4 kernel) copied to `benchmarks/gather-sweep/` — it is the
instrument that proves the bitwise claims (M4/M5/C13) and /tmp does not
survive reboots. Runtime metallib copy requirement documented in its
README.

**Parity re-gate (same day, post-fix chain) — PASS; C13 re-banked.**
A/B: fresh baseline build @ tesseract `8d47f122` (C12-state pins
mlx-swift `c9796ec` ← mlx `625f2aea`, vendor `e77d05d` — the same
A-side as C13's original acceptance) vs the fix branch @ `d0c8091f`
(mlx `a3673067` chain); `BENCH_RUNS=1`, `parity-ab.sh`. MoE
`qwen3.6-35b-a3b-paro`, 3-pair 128/8K/32K: gate 9/9 token-identical;
128/8K flat (+0.3–0.7%); 32K unverdictable at 3 pairs (slot bias: the
second arm of every round lost 23–51% and E drew second in 2/3 — the
same thermal saturation the acceptance logged). Escalated per
protocol, 10-pair 32K-only after a 3-min cool-down: **gate 10/10, 32K
prefill +3.10%** (pairwise mean +3.61%, median +8.07%, E wins 8/10,
two environmental outliers −34%/−14% left in), decode +0.86% (flat —
C13 cannot touch decode), peak identical 21.59 GB. Matches the
acceptance (+2.80%/20 pairs) and the probe prediction (+3%). Dense
`qwen3.5-4b-paro` control, 3-pair full grid: gate 9/9, all metrics
within −0.7…+3.1% noise, peaks byte-identical, load flat. Totals:
**28/28 token-identical pairs, 0 crashes across 38 arm launches** —
the fixed kernel compiles, engages at kL>4096, and reproduces the
stock chain bitwise in the app. Attribution: no other fix in the E arm
can move 32K prefill (M3 Max C4 caps are value-identical, C9's map
swap is negligible), so the delta isolates C13. Raw reports:
session scratchpad `parity/results-{moe,moe32k,dense}`.

## Session 2026-07-25 (b) — the decode roofline, and three rejections

Continuation of the C14 session. Goal was ~2× decode (MoE 8K
~96 → ~180 t/s). **That target is not reachable within the zero-loss
constraint on this architecture, and this session says so with
numbers.** What follows is the measured model of where a MoE decode
token goes, then the three hypotheses it killed. Nothing shipped;
tree, vendor and the mlx checkout are all back at their pinned
commits, verified token-identical.

**Correction to the C14 entry.** It claimed "~511 rotation launches
per MoE token, ~310 batchable". Wrong — read from the checkpoint
header (`model.safetensors`, 130 `.theta` prefixes: 30×{in_proj_qkv,
in_proj_z, out_proj} + 10×{q,k,v,o}), there are **130** rotations per
token and no MoE-block rotations at all (this checkpoint has no
`gate_up_rot`/`down_rot`; `RotateSwitchGLU` is never instantiated).
Corroborated by the older op census (CustomKernel ~258 = GDN scan +
rotations). The batchable subset — projections sharing an input — is
~50, not ~310.

**Decode is GPU-paced with no idle bubbles.** AGX driver utilization
counter (`ioreg -c AGXAccelerator`, 16 ms sampling) over a 500-token
8K decode window: median **100%**, p10 100%, mean 97.6 (boundary
samples). Prefill 99.1%. So every remaining win must come out of GPU
time, not out of CPU slack or scheduling gaps.

**Byte budget (the denominator everything else is measured against).**
From the safetensors header plus the loader's quantization predicate
(`isParoQuantIOLayer` quantizes **only** `embed_tokens` and `lm_head`
— the router `mlp.gate`, `shared_expert.*` and `shared_expert_gate`
stay **F16**, which the earlier estimates had as 4-bit): per token at
8K = 539 MB GDN + 145 FA + 523 experts (top-8) + 252 shared expert +
42 router + 270 lm_head + 168 KV + 63 GDN state = **2002 MB**. At the
measured 10.5 ms/token that is **189 GB/s**.

**The anchor: 371 GB/s.** `lm_head` (one 2048×248320 4-bit qmv, 270 MB)
runs at 371 GB/s *inside the model* — 93% of the M3 Max's 400 GB/s.
The machine streams fine. Cold-DRAM ceiling confirmed independently in
the rig at N=131072 (142 MB, cache-defeating): 388 GB/s.

**Attribution (in-app `--block-bench` probe, sync floor 0.21 ms
subtracted; parts sum to the whole: gdn 7.52 + fa 2.43 ≈ inner 9.93).**

| class | ms | MB | GB/s | % of anchor |
|---|---|---|---|---|
| MoE blocks ×40 | 4.77 | 817 | 171 | 46% |
| GDN attn ×30 | 4.42 | 602 | 136 | 37% |
| FA attn ×10 | 1.24 | 313 | 253 | 68% |
| lm_head | 0.83 | 270 | 371 | 100% |

Sub-block: switch_mlp ×40 3.33 ms (157 GB/s), in_proj_qkv ×30 1.28
(208), shared expert ×40 1.44 (175), in_proj_z ×30 0.90 (149),
out_proj ×30 0.87 (154), router ×40 0.74 (57). **Efficiency tracks
matmul size, not bytes** — everything but `lm_head` is latency-bound,
not bandwidth-bound.

**Wave census (mlx `CommandEncoder` instrumentation; two runs at
maxNew 100/300, subtracted).** Per decode token: **1892 dispatches,
972 hazard barriers** — average wave width 1.95, i.e. the graph is
~1000 serial steps deep. Barriers by primitive (per token): RMSNorm
157, CustomKernel 144, QuantizedMatmul 80, CompiledBroadcastMultiply
65, then ~12 entries at ~30–40 (one per layer): Matmul 40 (for 260
dispatches — those *do* overlap), Softmax, ArgPartition, Sum, Add,
GatherQMM 39 (120 dispatches), Concatenate 29, Convolution 29.

**The bound: barriers off → +62%.** Skipping the hazard barrier
entirely (numerically garbage, timing-valid) takes 8K decode
**95.07 → 153.80 t/s** (10.52 → 6.50 ms). So ~4.0 ms of the token is
serialization at hazard points and ~5.3 ms is weight streaming that is
already at ~95% of peak. **That is the whole budget: there is no 2×
inside it without changing arithmetic.**

**C15 — shared-input rotation batching — REJECTED (arithmetic, not
geometry).** Probe (z-axis batched rotation, per-element body
untouched): **bitwise IDENTICAL** at all four configs, 2.0–3.5× faster
per group (sets=3/4/6 at hidden 2048/2560; batched dispatch ~5–6.5 µs
regardless of set count). But the lever is only ~50 dispatches/token
(see the correction above), and the marginal cost of a dispatch in the
real pipeline was measured at **1.00 µs** (`--dispatch-probe` slope:
N=0/200/400 serial 1-element adds appended to every decode step, two
N=0 legs agreeing at 94.60/94.69 t/s; linear fit over N≤400 — N=800
leaves the regime at 1.69 µs/op). 50 × 1.00 µs = **0.48%** of a token,
below the 1% bar. Killed before porting. **Rule of thumb banked: a 1%
decode win needs ~105 dispatches removed.**

**Serial dispatch instead of concurrent+barriers — REJECTED, −19%.**
`computeCommandEncoder(MTL::DispatchTypeSerial)` with the explicit
barriers dropped (Metal then orders dispatches itself): 77.21/77.11
vs 94.77/94.95 t/s. MLX's concurrent-dispatch + explicit-hazard-barrier
scheme is already the better one; the overlap it buys is worth +23%.

**Resource-scoped hazard barriers — REJECTED, −7% (and the first
reading was a correctness bug).** MLX emits an encoder-wide
`memoryBarrier(BarrierScopeBuffers)` although it knows exactly which
buffers are hazards. Narrowing it to `memoryBarrier(resources,count)`
read **+1.7%** (95.69/96.65 vs 94.51/94.63, tokens identical) — but
that version kept MLX's `prev_outputs_ = next_outputs_` reset, which is
sound only for an encoder-wide barrier: a resource-scoped barrier does
not order the *unlisted* buffers, so the reset silently drops live
hazards (a race that merely did not fire). The sound version (erase
only the barriered resources, keep the rest in the hazard set) runs
**88.05/88.35 t/s, −7%**, with barriers up 33% (325k → 433k per run).
The win was the bug. **Logged as a trap: a scheduling change that
reads positive and token-identical can still be unsound — check the
invariant, not just the gate.**

**Housekeeping resolved.** The mlx checkout carried an *uncommitted*
`output_shapes` diff (CustomKernel/GatherMM/GatherQMM/Split shapeless-
replay scaffolding) left over from the discarded pre-C14 attempt.
Stashed, rebuilt, re-run: token-identical and same speed, i.e. **inert**
— the shipped C14 compiles concrete shapes and never consults it.
Dropped; the checkout is back at the pinned `a3673067` and the
committed pins are honest.

**Where the remaining decode work is (ranked, all small).** The 4.0 ms
serialization budget is spread over 972 barriers whose depth is the
model's dataflow. Cutting it means removing *serial* steps, and the
bitwise constraint blocks the reduction-order changes that would give
big wins (split-K, fused-norm rewrites, custom top-k). What is left is
per-site fusion in the E2-bitwise class, each worth ~1–2%: GDN
conv-state concat + conv1d → one kernel (~30 barriers), RMSNorm
absorbing its trailing elementwise consumer (~40), the router chain's
softmax/argpartition/takealong (~80, but top-k index order must be
preserved exactly or the weighted sum reorders). **Do not re-attempt:
anything that raises dispatch count for a "better" kernel, resource
barriers, serial dispatch, or rotation batching.**

**C16 — GDN decode conv1d as fused multiply-adds — ACCEPTED.** First
experiment sized against the new conversion factors (a barrier ≈ 4.14 µs,
a dispatch ≈ 1.00 µs) *before* building it. At S == 1 the depthwise
`conv1d` over `[convState | qkv]` is a fixed 4-term dot per channel;
written as elementwise multiply-adds it folds into the surrounding
compiled segment, so the `Convolution` wave disappears — −1 dispatch and
−1 **barrier** per GDN layer (30/token ⇒ ~0.155 ms ⇒ ~1.5% predicted).
Bitwise by probe, not by argument: the accumulation must run in f32 and
round once at the end (what MLX's Convolution kernel does) — over 8192
channels, f16 **and** bf16, f32-accumulation is **IDENTICAL in every
channel** (sequential and pairwise-tree forms both), while native-dtype
accumulation differs in ~47% of channels. Prefill and any S > 1 call keep
the original conv (byte-identical path). Measured (`parity-ab.sh`,
`BENCH_RUNS=1`): **MoE 8K decode 9/10 pairs positive, median +1.77%**,
pairwise mean +0.92% with one environmental collapse round (−6.96%, in a
window where the baseline arm also fell 94.3 → 87.7) left in; 3-pair run
+1.38% (3/3) at 8K and +0.60% (3/3) at 128. **Dense flat** (10 pairs:
−0.01% pooled, median +0.45%, 7/10 — no regression). Prefill +0.18% MoE /
+0.09% dense; **peaks exactly flat** on both. Parity gate **32/32
token-identical** (10+10+6+6) plus an in-model single-run check.
Vendor-only change (no Cmlx diff), `pin-upstream-mlx-swift` @ **46a8088**.

**Method note worth reusing.** The predicted value (~1.5%) and the
measured median (+1.77%) agree, which is the first time in this loop that
a decode change was *sized* correctly in advance. The two constants that
made it possible — 1.00 µs per dispatch and 4.14 µs per hazard barrier —
are the units to price any future decode idea in. Next candidates by the
same arithmetic: RMSNorm absorbing a trailing elementwise consumer (~40
barriers ≈ 2%), the router's softmax/argpartition/takealong chain (~80
barriers ≈ 3–4%, but top-k index order must be preserved exactly or the
weighted expert sum reorders).

**C17 — fold the GDN q/k norm scalar into `rmsNorm`'s weight — REJECTED
(context-split: +1.13% at 128, −0.81% at 8K).** Same arithmetic as C16:
`scalar * rmsNorm(x, weight: none)` sits between two non-elementwise ops,
so `compile` has nothing to fuse the multiply into and the q/k pair costs
one hazard barrier + two dispatches per GDN layer (~30 barriers +
60 dispatches ⇒ ~1.8% predicted). Folding the scalar into the norm's
weight is **bitwise** — probe over [1,1,16,128], f16 and bf16, both
scalars (invScale and invScale²): identical in every element — and the
in-model run was token-identical with flat peaks. But it does not convert
at long context: MoE **128 decode +1.13% (3/3 positive)**, MoE **8K
decode −0.81% over 10 pairs (median −0.72%, E wins 2/10)**, dense 8K
−1.76% (3 pairs, one collapse round). Gates PASS throughout (6/6 + 6/6 +
10/10). Reverted completely; vendor back at `46a8088`.

**Two lessons.** (1) **A short-context win can hide a long-context
regression** — verdicting C17 on the clean, low-noise 128 leg alone would
have shipped a −0.8% 8K regression. Both contexts, every time. (2) The
barrier-arithmetic prediction is *necessary but not sufficient*: C16 and
C17 removed comparable barrier counts and only C16 paid. The difference
is that C16 deleted a kernel outright, while C17 moved work *into* a
kernel whose weighted variant is evidently not free at this shape. Price
the prediction, then still measure at both contexts — and correct the
C16 entry's "RMSNorm absorbing a trailing elementwise consumer (~40
barriers ≈ 2%)" line: that was this experiment, and it does not pay.

---

## Session 2026-07-25 (c) — the schedule is optimal, so cut serial links

Two measurements re-aimed the whole remaining list, and one of them
closes a class of ideas permanently.

### The barrier census (probe, reusable)

`benchmarks/apply-census.py` (preserved there by the PR #427 review round;
originally session scratchpad) patches the mlx checkout to attribute every
GPU dispatch and every hazard barrier to the primitive that issued it
(`eval.cpp` stamps `arr.primitive().name()` before `eval_gpu`;
`CommandEncoder::maybeInsertBarrier` books it), and to track the ASAP
**critical-path depth** of the dispatch DAG. `TESS_CENSUS=1`,
`TESS_CENSUS_OUT=<path>`, table rewritten every 200k dispatches. Run the
app binary directly — `open` does not forward env.

MoE at 128 ctx, 209 tokens (tokens = `ArgReduce` dispatches, one sampler
argmax per token): **1913.9 dispatches and 946.3 barriers per token**,
confirming the previous session's 1892/972 from an independent angle. The
top of the attribution table, per token:

| primitive | dispatches | barriers | ms/token @4.14 µs |
|---|---|---|---|
| RMSNorm | 202.8 | 155.9 | 0.645 |
| CustomKernel | 241.5 | 146.3 | 0.606 |
| QuantizedMatmul | 131.6 | 82.7 | 0.342 |
| CompiledBroadcastMultiply | 99.6 | 63.5 | 0.263 |
| Matmul | 264.0 | 43.1 | 0.178 |
| Add | 41.8 | 40.7 | 0.169 |
| Softmax | 40.3 | 39.6 | 0.164 |
| GatherQMM | 120.5 | 38.2 | 0.158 |
| ArgPartition | 40.2 | 38.9 | 0.161 |
| CompiledBroadcastDivide | 39.6 | 38.4 | 0.159 |
| Sum | 80.4 | 37.4 | 0.155 |

### Tape reordering / list scheduling — DEAD, by measurement

The same run reports `criticalPathDepth=199599` against
`barriers=197773`. **MLX's dispatch schedule is already at the graph's
critical-path depth** (within 1%, and the model is the conservative
direction). There is no wave to be won by reordering the tape,
list-scheduling it, or interleaving independent chains — the ~950 waves
per token *are* the graph's depth. The only way to remove a barrier is to
remove a serial link. Do not revisit scheduling.

### C18 — fused router top-k kernel — ACCEPTED (+1.91% MoE 128 decode)

`ArgPartition::eval_gpu` carries the comment "We direct arg partition to
sort for now" and delegates to `gpu_merge_sort`: **the router fully sorts
all 256 experts, every MoE layer, every token, to name 8 of them.** With
the gather/sum/divide normalise tail that is three serial dispatches and
three encoder-wide barriers for a 256→8 selection.

Rig pricing of the chain at the production shape ([1,1,256] f16, depth-400
dependent chain, since decode is latency- not throughput-bound):

| item | µs/layer | ms/token (40 layers) |
|---|---|---|
| `argPartition` (the full sort) | 10.9 | 0.436 |
| `sum` + `divide` tail | 6.4 | 0.255 |
| whole router above floor | 18.1 | 0.724 |

Replaced with one custom kernel. **Bit-identical by construction**, each
part read out of the MLX source it replaces: `sort.h`'s `LessThan`
compares values only, `ThreadSort` swaps on strict less-than and
`merge_step` takes from A on ties, so the sort is *stable* and element
`i`'s ascending position is exactly `#{j: v_j < v_i} + #{j: v_j == v_i,
j < i}`; `reduce.metal` instantiates float16 with `U = float16_t` and a
row of 8 takes `thread_reduce` (sequential accumulation in the output
dtype, from zero, in slot order); the divide is elementwise.

Three kernel geometries were tried. The winner counts, for each expert,
how many rank *above* it (`slot = K-1-count`) using a **single 64-bit
compare** per element: a monotone bit key in the high 32 bits with the
index in the low 32 folds value comparison *and* the stable tie-break
into one operation. Saved vs the two alternatives:

| variant | µs/layer saved | ms/token |
|---|---|---|
| null kernel, same I/O (work-free ceiling) | 13.7 | 0.548 |
| **key64 (shipped)** | **9.92** | **0.397** |
| value compare + explicit tie-break | 6.62 | 0.265 |
| P=4 threads per element | 4.50 | 0.180 |

The P-split being *slowest* is the C15/M4 lesson again: the kernel is
launch/latency-bound, so adding parallelism costs more than the work it
saves.

Gate: bitwise **IDENTICAL** over float16/bfloat16/float32 × {softmax
outputs, 16-level ties, 4-distinct-values, signed zeros, all-equal rows},
7936 rows each. Two edge cases had to be handled explicitly to get there:
NaN maps above `+inf` and all NaNs tie (as `LessThan` does), and the sign
of zero is normalised, because `-0.0` and `+0.0` compare *equal* under
`a < b` but have different bit patterns.

In-model, against `46a8088`: MoE **128 decode +1.91% over 10 pairs
(10/10 positive, +1.34%…+2.36%)**, MoE **8K decode +0.47% over 10 pairs
(8/10 positive)**, prefill +0.23%/+0.08%, peaks exactly flat, dense
unaffected (no MoE block on that path). **32/32 A/B pairs
token-identical.** Committed as `e10e52f`.

Decode only: at prefill the block sort's `O(E log² E)` beats this
kernel's `O(E²)` rank count and there is no barrier to save, so many-row
callers keep the old chain. The MLXVLM copy of the block still has the
old chain — not exercised by either PARO model here, so not touched.

### Why C18 pays less than the dispatch count suggests

Census on the C18 build, same 128-ctx run: dispatches/token **1913.9 →
1826.5 (−87.4)**, exactly the four primitives removed. But barriers/token
went **946.3 → 978.5 (+32.2)**: `ArgPartition` −38.4 and
`CompiledBroadcastDivide` −38.4, against `CustomKernel` +40.6 and
**`GatherQMM` +39.8**.

The expert matmuls read `inds`. In the old chain the `sum` and `divide`
barriers fired *between* the sort and the expert matmuls, so they
published `inds` as a side effect and `GatherQMM` got its synchronisation
free. Delete them and `GatherQMM` has to pay for its own barrier. **C18's
win is the deleted 256-element block sort's compute, not barriers** —
barriers actually rose.

New lesson: **a barrier can be load-bearing for a consumer other than the
one it was counted against.** Removing a serial link only pays if the
link it exposes was not already going to need one. Price a fusion by what
the *consumer* still has to wait for, not by the dispatch count alone.

### C19 — fold the router's softmax into the same kernel — REJECTED (+0.64% / +0.19%)

The obvious follow-on to C18: `Softmax` is 39.6 barriers and 40.3
dispatches per token (census above), it feeds nothing but the router, and
folding it into the C18 kernel removes a genuine serial link — the
matmul→softmax→router chain becomes matmul→router.

**It is fully replicable and it was bitwise.** `softmax_single_row` was
reproduced verbatim inside the router kernel: `AccT = float` (the
`precise` instantiation), `N_READS = 4`, `SIMD_SIZE = 32`, so axis 256
runs on 64 threads; our threadgroup is 256 wide, so the softmax phase is
masked to the first 64 with every `threadgroup_barrier` outside the mask,
`local_max`/`local_normalizer` initialised by simdgroup 0 exactly as MLX
does (entries 2..31 left at `Limits<float>::min = -INFINITY`, with
`finite_min = -FLT_MAX` as the per-thread seed) so the final
`simd_max`/`simd_sum` sees the same 32 lanes, and `fast::exp` called
explicitly. Ranking then runs on the **`T`-rounded** probabilities, since
that is what the sort it replaces saw. Gate: **IDENTICAL** over
float16/bfloat16 × {normal logits, ×8-wide logits, 0.5-step ties,
4-level ties, all-equal rows, one dominant logit}, 11 776 rows each.

In-model against `e10e52f`, 10 pairs, both contexts: MoE **128 decode
+0.64% (9/10)**, MoE **8K decode +0.19% (6/10)**, peaks flat, **20/20
token-identical**. A first attempt read +0.87%/+0.46% but its *prefill*
legs moved −1.64%/+2.16% on a code path C19 provably does not touch, so
that run was discarded as noise; the accepted run's prefill sentinel is
−0.07%/−0.06%. **Reverted completely; vendor back at `e10e52f`.**

Rejected on the bar, and the risk profile agrees: C18's bitwise contract
rests on two broad properties (the sort is stable; the reduce accumulates
sequentially in the output dtype), while C19 additionally pins `N_READS`,
`AccT`, the exact simd reduction tree and the threadgroup padding rule —
much more brittle, for a fifth of the gain.

### The conversion constant was wrong, and that closes the class

C19 is the cleanest possible calibration point: it removed **exactly one
dispatch and one barrier per MoE layer** — 40 of each per token — and
nothing else changed. It bought **0.058 ms/token** (+0.64% of a 9.11 ms
token).

That is **~1.4 µs per barrier+dispatch removed, not the ~5.1 µs**
(4.14 barrier + 1.00 dispatch) the previous session's constants imply.
The 4.14 µs came from turning *all* 972 barriers off at once, which also
lets every kernel in the token overlap — a super-linear effect that does
not decompose. **The marginal barrier is worth roughly a third of the
average one.**

Re-pricing every remaining decode candidate at 1.4 µs:

| candidate | barriers/token | ms/token | % at 128 ctx |
|---|---|---|---|
| RMSNorm absorbing the residual add | ~40 | 0.056 | 0.6% |
| the second residual (already compiled) into the next norm | ~39 | 0.055 | 0.6% |
| `CompiledBroadcastMultiply` sites | ~64 | 0.090 | 1.0% |
| everything else in the table | ≤40 each | ≤0.056 | ≤0.6% |

**No remaining barrier-removal candidate clears 1% on its own.** Combined
with the two facts already banked — the dispatch schedule is at the
graph's critical-path depth, and ~5.3 ms of the token is irreducible
weight streaming — the fusion class is exhausted under the zero-loss
constraint. C18's win was not really a barrier win either: it was
deleting a 256-element block sort's *compute*.

The other calibration from this session: **rig savings convert to
production at roughly 40%.** C18's rig number was 0.397 ms/token and it
delivered 0.174 ms; C19's rig number was 0.054 ms and it delivered
0.058 ms. Price a candidate in the rig, then halve it before deciding
whether it is worth an in-model round.

### Item 5 (gather_qmm round 2) — CLOSED at the probe, and the C1 ambiguity resolved

`docs/mlx-core-future-work.md` item 5 asked for one thing first:
re-establish the gather_qmm-vs-dense-anchor ratio *at the production
shape*, because the C1 entry recorded both "~40–50% of the anchor" and
"the winner reaches 96% of the anchor" without pinning the B/E of either.

Probe (rig, production MoE dims — hidden 2048, intermediate 512, 256
experts top-8, 4-bit group 64; anchor = `quantizedMatmul` over the same
MACs with the same quantization, no gather):

| gathered rows | S | up/gate | down | anchor GMAC/s | gather GMAC/s |
|---|---|---|---|---|---|
| 4 096 | 512 | 52.5% | 58.0% | 3690 | 1938 |
| 16 384 | 2048 | 63.8% | 69.8% | 5735 | 3657 |
| 65 536 | 8192 | **87.4%** | **91.4%** | 6069 | 5303 |

**The ratio is shape-dependent and both C1 readings were right** — 40–50%
is the small-shape end, ~90%+ is the large end. Prefill runs at the large
end: at 8K the kernel is already at 87–91% of a gather-free dense qmm of
the same arithmetic.

So the headroom is ~9–13% of the expert matmuls only. Those are ~1.5 s of
a ~5.6 s 8K prefill, so a *perfect* gather kernel would be worth ~2–3%
prefill — and C1's sweep already searched the legal geometry space (same
per-element K-accumulation order; split-K changes rounding and is dead).
**Not worth an implementation round.** Item 5 closed.

## Where this leaves the decode/prefill program

Every lever that was priced this session came back sub-1% or already
spent. The three budgets are now each accounted for:

- **Decode streaming** — ~5.3 ms of a 10.5 ms 8K token, at ~95% of peak
  bandwidth. Irreducible without changing quantization.
- **Decode serialization** — the dispatch schedule is *at* the graph's
  critical-path depth, and the marginal barrier is ~1.4 µs, so the
  remaining fusions are ~0.6–1.0% each.
- **Prefill** — GEMM-bound, and the GEMM is at 87–91% of a gather-free
  anchor of the same arithmetic.

What is left is not kernel work. The only remaining ≥10% ideas change
what runs, not how fast it runs: speculative decoding for the dense model
(greedy-verified, so output-identical by construction — blocked on a
compatible PARO draft model), or a quantization change (out of the
zero-loss scope by definition).

## Review round 2026-07-25 — PR #427 review fixes

A full-diff review of the C14/C16/C18 round found no correctness defect in
the shipped code, but six hygiene/robustness findings. All fixed in vendor
`8519cf3` on `pin-upstream-mlx-swift` (68ad25f → 8519cf3 total for the PR):

- **uint32 router indices.** The C18 kernel emitted `int32` where
  `argpartition` emits `uint32` (mlx `ops.cpp:2561`), so the decode and
  prefill router paths produced different index dtypes for the same
  logical tensor. Kernel now emits `uint32`. Values were always identical;
  both dtypes were independently parity-proven in-model (uint32 = the
  entire pre-C18 history, int32 = the C18 gate), so this is a
  consistency fix, not a numerics change.
- **Dead C12 wrapper removed.** Since C14, every S == 1 decode reaches
  `decodeForward` through the layer trace (B) or a whole-step segment (C);
  the GDN-module-local compiled wrapper was unreachable on every path
  (traced all callers). Body stays as `decodeForward`.
- **Package.swift: MLXLLM lacked the MLXFast product dependency.** Xcode
  leaks all modules of a dependency package into the search path, so the
  app built C18 fine — but strict SwiftPM (`swift build` / `swift test`)
  failed on `no such module 'MLXFast'`. The vendor suites had last been
  run pre-C18; gap now closed and the dep added.
- **The bitwise contracts are now CI-pinned** — new vendor
  `Qwen35BitwiseContractTests`: (1) fused GDN decode body vs the unfused
  conv1d body, f16+bf16, 256 channels (an MLX pin bump that changes
  Convolution's accumulation now fails a unit test instead of silently
  splitting decode from prefill); (2) fused router kernel vs the
  argPartition/takeAlong/normalise chain over softmax rows, tie-heavy
  rows, all-equal rows, signed zeros and NaN rows, f16/bf16/f32,
  E∈{256,128}, norm∈{on,off}, dtype and bit patterns asserted. The
  NaN-above-everything ordering is thereby gate-verified (previously
  source-derived only — `sort.h`'s `LessThan` is explicitly NaN-aware,
  `(!an) & bn`, so NaN is a true maximum equivalence class).
- **Lifecycle coverage restored.** The whole-step schedule had silently
  removed the MoE block's own C11 closure from the original leak test's
  path; a quantized-cache decode variant now drives the fallback that
  still installs it.
- **Census preserved.** `apply-census.py` moved from the session
  scratchpad to `benchmarks/apply-census.py` — it produced the
  critical-path-depth result that closes the scheduling class, and was
  the one probe not preserved.

Gates for this round: vendor suites green (ParoQuantTests 24/24, Qwen35
suites incl. the 2 new contract tests and the new lifecycle variant,
SwitchLayers, ToolTests). No fresh 10-pair A/B was run: the only
executed-graph change is the index dtype, whose two variants are both
already parity-proven above; the contract tests hold the fused outputs
bit-identical to the chain including dtype. Dead-code removal and the
Package.swift dep do not change the executed graph.

---

## Session 2026-07-27 — Cmlx loop, second campaign (branch `perf/inference-loop-2026-07-27`)

Same rules and measurement discipline as the prior sessions (the binding
section at the top of this file). Base: main `383ef0f2` (pins mlx-swift
`457a0d6d` ← mlx `a3673067`, mlx-swift-lm `97c0308`). All work on the new
branch `perf/inference-loop-2026-07-27`.

### Infrastructure: fork/pin scheme re-verified (the assigned prerequisite task)

The kickoff asked to "establish a buildable mlx-core fork/pin scheme and
record it in the ledger". That scheme has existed since 2026-07-23
(`docs/mlx-core-fork.md`); the honest task this session is to **re-verify
it end-to-end**, which was done:

- Pin chain lockstep: `Vendor/mlx-swift-lm/Package.swift`,
  `Vendor/mlx-audio-swift/Package.swift`,
  `Vendor/tesseract-speech/Package.swift` all pin `spokvulcan/mlx-swift`
  @ `457a0d6df3a20c92341a6e7b7fa853d63d8549f9`; tesseract gitlink →
  mlx-swift-lm `97c0308`.
- Fork clones: `~/projects/mlx` @ `a3673067` (`pin-tesseract`, clean, in
  sync with origin), `~/projects/mlx-swift` @ `457a0d6` (`pin-tesseract`,
  clean). The `Source/Cmlx/mlx` submodule checkout there is clean at
  `a3673067` (no probe hooks left over).
- Live build checkout
  (`~/Library/Developer/Xcode/DerivedData/tesseract-buwysfpnwmzyucelgewutuddcvgv`
  — now the only tesseract DerivedData dir): mlx-swift @ `457a0d6`,
  Cmlx/mlx @ `a3673067`, mlx-c @ `0726ca9` (upstream, untouched), tree
  clean. Full carry chain present in `git log ce45c525..HEAD`:
  `fbf2fb86` (C1 tiles), `404070e2` (C4), `8d11dd1d` (C5), `3ec72a24`
  (C6), `6ab29e36` (C7), `595a3fe1` (C8), `625f2aea` (C9),
  `ed107a94`+`5ca82d9f` (C13), `90ec2bb9`+`a3673067` (review hardening).
- Release build of `383ef0f2` **succeeds** from this tree; session
  baseline binary saved at `/tmp/tesseract-2026-07-27-base.app`. No
  `Tesseract Agent` process was running during any of this (GPU
  serialization rule).

**Verdict: scheme VALID — no re-establishment needed; recorded per the
kickoff.**

### M1 reconciliation — already banked as C1; do not re-run

The kickoff's "start with M1 (gather_qmm_rhs tile geometry)" reflects the
stale roadmap, not the ledger. Ledger state: **C1 (rows-per-expert-aware
`gather_qmm_rhs` tiles, `fbf2fb86`) was ACCEPTED 2026-07-23** (+6.2% MoE
32K prefill, 34/34 pairs token-identical), is carried by the current pins,
and is filed upstream as ml-explore/mlx#3918. The residual headroom was
priced and closed 2026-07-25 ("Item 5 — gather_qmm round 2 — CLOSED at the
probe"): the post-C1 kernel sits at 87–91% of the gather-free dense anchor
at production prefill shapes (C1's own sweep evidence: 96% of the anchor
at B/E=32), and C1's sweep already searched the legal (same-per-element-
K-accumulation-order) geometry space — split-K is dead under the bitwise
rule. E4's "~12–15% of prefill" estimate was calibrated on the old
harness; C1 recorded the true app win at ~6% at 32K. **Re-running M1
would repeat a logged result, which the rules forbid.** The roadmap doc is
stale (written after E1–E11, never updated post-C1); this entry is the
correction of record.

Also closed by later measurement, for the same reason: the roadmap's **M2
"attack the cost per boundary" residual**. Decode is GPU-paced (C14
attribution: the generation thread sits in `Scheduler::wait_for_one`
33–37%), and AGX utilization is 98–100% during decode (session 2026-07-25
(b)). CPU-side per-boundary cost lands in the CPU slack and cannot convert
(C10 lesson); GPU-side drain has no idle left to recover at the current
~20–27 boundaries/token.

### Queue for this campaign (derived from the ledger end-state, not the stale roadmap)

1. **Stock `qmm_t` tile geometry at PARO shapes (probe first).** The dense
   anchor every gather kernel is measured against is itself untuned on
   this machine: `qmm()` hard-codes bm=bn=32, bk=32, wm=wn=2
   (`backend/metal/quantized.cpp:715-719`; template defaults in
   `kernels/quantized.h`), measured 9.5–10.1 TFLOP/s ≈ 75–80% of the
   12.69 peak at PARO shapes (C1 anchor; E9's lm_head 10.4 ≈ 82%). The
   nax sibling ships bm=bn=64 — but nax is unavailable on M3 Max (gen 15
   < 17, verified 2026-07-23). Same bitwise-safe axis as C1 (per-element
   K-accumulation order is tile-geometry-independent). If the anchor
   rises, every prefill projection on both models rises with it.
   Production stock-qmm shapes (from the two `config.json`s this session):
   MoE — q [1024×8192×2048], k/v [1024×512×2048], o [1024×2048×4096],
   GDN in_proj_qkv [1024×8192×2048], in_proj_z [1024×4096×2048],
   out_proj [1024×2048×4096], lm_head [1024×248320×2048]; dense — q
   [1024×8192×2560], k/v [1024×1024×2560], o [1024×2560×4096], GDN same
   at 2560, MLP gate/up [1024×9216×2560], down [1024×2560×9216], lm_head
   [1024×248320×2560]; 4-bit gs=128, f16 (both checkpoints store F16).
   Also M=128 (the ctx-128 single-chunk case).
2. **M6 tokenizer encode path (TTFT).** ~0.29 s at 32K on the parity
   bench; seconds at 100K+ in production server/agent use. Profile the
   Jinja-render vs BPE-encode split first. CPU-only, zero numerics risk,
   shows on the bench's tokenize metric.
3. C6 hit-path string copies (future-work #8; ≲1% expected, CPU slack —
   hold, likely sub-bar).
4. C13-extension to kL ≤ 4096 (future-work #4; +0.3–0.5% 8K prefill —
   sub-bar alone; dead unless bundled with a larger attention change).

Dead / do-not-retry (in addition to the top table and the C15–C19
lessons): item-5 gather geometry, M2 boundary-cost residual (above), tape
reordering/list scheduling, resource-scoped barriers, serial dispatch,
rotation batching, router-softmax folding.

### C20 — stock `qmm_t` tile-geometry sweep at PARO shapes — REJECTED at probe (stock is at the envelope)

Hypothesis: the stock `qmm_t` kernel's hard-coded bm=bn=32/bk=32 (wm=wn=2)
leaves ≥5% on the table at production prefill shapes; a wider legal tile
(the nax sibling ships bm=bn=64) lifts it bitwise-safely. Probe in
`/tmp/gather-sweep` (coder subagent, `qmmtiles` mode — now durable in
`benchmarks/gather-sweep/`): uncommitted `MLX_QMM_TILES` env hook in the
rig's mlx checkout substituting non-default BM/BK/BN template args
(kernel body untouched — params already exist on `affine_qmm_t`), 16
production shapes (M ∈ {128, 512, 1024} × both models' projection and
lm_head N/K sets, 4-bit gs=128 f16 transpose), configs (32,32,32) stock
vs (64,32,64) / (64,64,64) / (32,32,64) / (16,32,32), ABBA-interleaved
process launches, one lazy graph per launch cycling 8 disjoint weight
sets. **Bitwise IDENTICAL in all 64 shape×config cells** (the C1-class
per-element-K-order invariance confirmed once more, now for the dense
kernel). Speed: **every candidate within ±2% of stock at every shape**
(mean ratios 0.992–0.997) — nothing clears +5% anywhere, including the
lm_head shapes (2.2 GB footprint, true DRAM streaming). Anchor correction
of record: stock `qmm_t` measures **11.6–12.4 TFLOP/s cool (~95% of the
12.69 bf16 peak)**, decaying to 9.5–10.1 under sustained load — so the
"75–80% of peak" premise (C1 anchor / E9) was a *throttled-regime*
reading, not headroom. C1's verdicts are unaffected (within-run ABBA).
**Verdict: REJECTED, no app run.** Hook reverted, rig submodule clean.
Consequence: with item 5 (gather geometry) and C20 (dense geometry) both
closed, **the GEMM tile-geometry axis is exhausted on this machine** —
prefill-side wins must come from fusion/overlap/dequant cost, not tiles.

### C21 — pretokenizer regex single-pass (M6) — REJECTED at probe (cannot preserve byte-identical ids)

Hypothesis: replace the pre-tokenizer Split step's per-match
`String.range(of:.regularExpression)` loop with one NSRegularExpression
(ICU) pass — measured 12.2× on the phase (76.2 → 6.2 ms at 32K), worth
~2.2× on the whole tokenize metric. Profile first (this session,
`tokprofile` rig mode; all findings measured): **chat-template render is
0.03% of tokenize** (E11-class work — nothing left); encode = 99.7%;
split loop 76.9 ms of the 126 ms 32K encode, BPE merge loop ~27%,
token→id ~7%; encode scales linearly (~250K tok/s, no O(n²)); tokenizer
one-time load ~800 ms (12.8 MB tokenizer.json + 247,587-merge rank-dict).
Differential harness (`tokdiff` rig mode; production pipeline replicated
exactly — P′ == P on all items): 26 text classes, **3,000,356 tokens**,
production vs two ICU arms (original pattern; quirk-mutated pattern
folding in the verified Swift-Regex `\r\n`-negation bug). **Both ICU arms
diverge from production in the same 3 classes** (crlf, emoji,
whitespace-runs). Root cause below the pattern level: **Swift Regex
matches at grapheme-cluster (UAX#29) granularity, ICU at code-point
granularity** — two final-id-changing classes: CRLF clusters
(`"!\r\n\r\n!"` → production `[0,317,317,0]` vs ICU `[0,845,0]`) and
VS16/keycap emoji adjacent to symbols (`"🤖❤️!"` diverges). No pattern
mutation re-expresses cluster semantics in ICU; a same-engine precompiled
Swift `Regex` pass measures **0.8× (slower)**; and one production
behavior (`[\r\n]*` at CRLF clusters) is not derivable from source at all
— near-identical inputs yield different piece structure — so a
hand-rolled exact matcher is a research project with an unbounded
differential tail. **Verdict: REJECTED, no app run.** The replacement
code is preserved marked DO NOT SHIP (`benchmarks/gather-sweep/
tokdiff-replacement.swift`) with the corpus harness as the reusable gate.
Findings of record: (a) **production tokenization itself diverges from
the HuggingFace reference tokenizer on the CRLF/emoji classes** — the
parity baseline this loop defends is non-canonical there (upstream-report
material for swift-transformers; not actionable in-loop — the gate
anchors to the unmodified baseline, bugs included); (b) the profile's
parallel per-pre-token BPE candidate is unaffected by this verdict and
proceeds as C22; (c) tokenizer one-time load ~800 ms noted as a possible
load-time item.

### C22 — parallel per-pre-token BPE — REJECTED at probe (in-process contention, 3× SLOWER)

Hypothesis: an order-preserving concurrent map over the per-pre-token
BPE work (provably byte-identical: `bpeRanks`/`tokensToIds` read-only
`let`s, per-call `MinHeap`, merges local to one pre-token, `fuseUnk`
false) wins ~25–30% of the 32K encode. Rig implementation (patched
swift-transformers 1.3.3 checkout, threshold-gated `concurrentPerform`
map) + full gates: corpus identity **PASS** (52/52 items, 3,000,356
tokens × both model dirs — the models' `tokenizer.json` files are
byte-identical; chat templates differ in 2 jinja lines but render
identically for the corpus shapes), determinism **PASS** (50/50 repeat
encodes identical), thread-safety review clean (no hidden mutable state;
canonical `concurrentPerform` idiom). Timing **FAIL: 0.31×/0.33× at
8K/32K — 3× SLOWER, not 30% faster.** Contention curve, not dispatch
overhead: bisect chunk2 ≈ serial → chunk16/full ≈ 3×; whole-encode
scaling n=2 → 1.41×, n=4 → 1.48×, n=8 → 1.06×, n=16 → 0.25×; two
instances 1.51×; **four concurrent PROCESSES scale fine (137–140 ms vs
129 solo)** — the pathology is in-process: workers sit inside
`BPETokenizer.bpe` dominated by `swift_retain`/`swift_release`,
`__RawDictionaryStorage.find`, and String `_normalizedHash` (NFC), whose
per-op cost inflates with thread count. **Verdict: REJECTED; rig checkout
reverted, baseline re-measured intact.** Do not retry parallel BPE
without attacking the contention itself (per-thread merge-rank replicas
or a scale-stable lookup structure — a different, much deeper change).
Patch preserved for the record (`/tmp/gather-sweep/c22-patch.diff`); gate
code lives on in the rig's `c22` modes. Threading lesson of record for
this machine: shared read-only Swift dictionaries + refcounted Strings do
NOT scale read-only across cores in-process — profile before assuming
"embarrassingly parallel".

### C23 — tokenizer-load binary cache — REJECTED by attribution (off the critical path)

Candidate from the C21 profile (tokenizer one-time load ~800 ms: 12.8 MB
tokenizer.json parse + 247,587-merge rank-dict). Measured in-app this
session (sample on a MoE parity-bench load, pid 3658): `AutoTokenizer.from`
→ `PreTrainedTokenizer.init` ≈ 300 ms + `YYJSONParser.parseToConfig`
≈ 230 ms ≈ **0.53 s**. But `LLMModelFactory._load`
(`Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift:631-639`)
loads the tokenizer via **`async let`, overlapped with weight loading**,
and the PARO phase log from the same run shows the weight side at
**4.14 s** (`eval=3.92s`; `log show` subsystem `mlx-swift-lm:paroquant`).
max(4.14, 0.53) — the tokenizer is fully hidden on the MoE; on the dense
model (1.4 s load, ~0.9 s weight side vs ~0.53 s tokenizer) it is hidden
as well. A binary cache of the parsed structures would move no bench
metric and no production load time on this machine. **Verdict: REJECTED,
no code written.** (If weight loading ever gets ~5× faster than the
tokenizer — e.g. a much faster `eval(model)` — this re-opens.)

### C25 — render+token cache for prefix-stable request tokenization (M6, app-side) — ACCEPTED

Hypothesis: on prefix-stable request sequences (agent multi-turn, growing
conversations), the fused render+encode `applyChatTemplate` re-tokenizes a
mostly byte-identical prompt every request (~126 ms at 32K; two full
encodes per server request — the `RequestKeyingPhase` prepare plus
`PrefillPlanner`'s last-user re-render — plus post-generation encodes).
Caching the previous (render, tokens, digests) and tokenizing only a
verified suffix reproduces the EXACT token list at a fraction of the
cost. Exactness contract: the miss path is `renderChatTemplate` +
`encode(rendered)` (byte-exact with `applyChatTemplate` — the fused call
was split, not changed); the hit path is empirical — byte-prefix check
per trim attempt (k = 0…4 tokens), suffix encode, and a junction-window
re-encode (≥256 chars a side, ×4 to 16 KB) that must return the exact
token slice (BPE merges spanning the cut are detected, never assumed
away); any failure degrades to the miss path. Identical repeat renders
return cached tokens outright (deterministic encode).

Implementation (three layers): swift-transformers `renderChatTemplate`
(pure refactor — `applyChatTemplate` now calls it; new protocol
requirement with a public throwing default) → MLXLMCommon
`ChatTemplateRendering` protocol + macro-bridge forwarding → app
`RenderTokenCache` (single-entry, NSLock-guarded, keyed on
modelFingerprint + template probe hash + tools/context digests +
per-message SHA-256 chain), engaged at two seams
(`RequestKeyingPhase.run` — bypasses images/vision-family;
`LLMActor.startRawGeneration` — bypasses media/non-LLMModel), both
falling back to the processor's `prepare` on any failure. Design
finding: the generation-prompt tail (`<|im_start|>assistant\n<think>\n`)
means a full-text byte-prefix gate would never hit — the trim loop
re-discovers the shared prefix in token space instead (every growing
turn hits at trim=2).

Gates (all PASS): `RenderChatTemplateParityTests` (8-case battery,
through the macro adaptor and direct), `RenderTokenCacheTests` (forced
dirty junctions, 6 junction classes, digest mismatches, repeats — all
exact), regression suites (StablePrefixDetector, HTTPPrefixCacheSpike,
ServerCompletionDrain, AgentEngineToolSpec, AgentEngineManagedGeneration,
PrefixCacheIntegration), `--prefix-cache-e2e` PASS (incl. image warm
output equivalence), and the new `--tokenize-cache-bench` runner with a
per-turn intrinsic exactness assertion against `applyChatTemplate`.
Runner (12-turn agent trajectory + repeat + edited + unrelated, real
tokenizer, Release): **4B −38.9% prepare ms per hit turn (40.71 →
24.87 ms; parent-verified rerun), MoE template −48.8% (41.07 →
21.03 ms), repeat −97% (48.7 → 1.4 ms), misses +0.4–1.0 ms**
(digest+render overhead; ≪0.1% of TTFT); **0 token mismatches, 0
junction failures, 0 window enlargements across both models**.
Parity A/B (3-pair, both models, all contexts): **9/9 + 9/9
token-identical**, peaks byte-flat; perf deltas are the documented
thermal noise (MoE +34%/30% and dense −16.5% in the same session —
impossible both directions for a tokenize-path change; the parity bench
provably never engages the cache — `runOnce` calls
`context.processor.prepare` directly). **Verdict: ACCEPTED** — the win
metric is production-path tokenize time on prefix-stable sequences
(−39–49% per hit turn, −97% on repeats), E11-shaped; no
mechanistically-possible regression channel on any bench metric; miss
cost +~1 ms documented.

Ports and pins: `spokvulcan/swift-transformers` **new fork**,
`pin-tesseract` @ `63edf42` (scheme: `docs/swift-transformers-fork.md`);
Vendor/mlx-swift-lm `pin-upstream-mlx-swift` @ `47aa83a` (pushed);
`Vendor/mlx-audio-swift/Package.swift` pins the fork at `63edf42f…`
(the package's only declarer in the graph). Follow-ups queued: **C26**
(hit path is prefix-decode-bound — ~15–18 ms of the ~25 ms at 11K
tokens; derive `prefixText` from the entry's stored render instead of
decoding the whole prefix per attempt), C27 (PrefillPlanner's last-user
re-render through the same cache — the second full encode on the server
TTFT path), C28 (post-generation leaf-store/admission encodes), and the
upstream filings (mlx-swift-lm + swift-transformers PRs — owner
go-ahead).

### C26 — hit-path prefix-decode elimination — ACCEPTED

Direct follow-up to C25 (queued in its entry): the hit path decoded the
whole cached prefix per trim attempt — ~15–18 ms of the ~25 ms at 11K
tokens. Decode is per-token concatenation, so stripping each trimmed
token's decoded text from the stored render's tail reproduces
`decode(prefixTokens)` exactly; `prefixText` is now derived from the
entry's stored render (a `hasSuffix` guard keeps any decode/strip
inconsistency honest — it degrades to the miss path; the per-attempt
`hasPrefix` and junction-window checks are unchanged). Same tokens
produced; the exactness contract is untouched. Gates: the C25 suites
(TEST SUCCEEDED), runner both models — hit turns 41.41 → **5.54 ms**
(4B, **−86.6%**, vs C25's −38.9%), 42.50 → **5.56 ms** (MoE,
**−86.9%**), repeats −97% unchanged, misses unchanged; 0 token
mismatches, 0 junction failures. **Verdict: ACCEPTED.** App-only change
(`RenderTokenCache.swift`); no vendor/fork pins moved. At 32K the hit
path is now render(≈4 ms)+digests+suffix-encode ≈ 10–12 ms vs the
~126 ms full encode (−91%); the remaining hit cost is the mandatory
render + the suffix encode itself.

### C27 — PrefillPlanner last-user re-render as a verified trim of the cached entry — ACCEPTED

The third full-size encode on the steady-state server path (after the
RequestKeyingPhase prepare and the C25/C26 suffix encode):
`PrefillPlanner.detectBoundaries` re-renders every cache-aware request
**truncated at the last user message** (`add_generation_prompt: false`)
to find the last-user boundary — ~40 ms at 11K, ~126 ms at 32K, right
after the prepare cached the SAME conversation's full render+tokens.
C27 recovers the truncated token list as a **verified trim** of the
stored entry (`RenderTokenCache.resolveTruncated`): digests/fingerprint/
template candidate (context digest compared on the unmerged context),
the truncated render must be a byte prefix of the stored render with a
≤128-byte tail (a generation prompt, not dropped content — longer means
the last message is not the last user message → fallback), a per-token
strip finds the trim k (a token spanning the cut → fallback), and the
**cut verification** arbitrates exactness: a standalone re-encode of the
truncated render's trailing ≥256 chars (×4 to 16 KB) must reproduce the
candidate token suffix — catching right-context effects at the cut
(end-of-text vs mid-text, e.g. the `\s+(?!\S)` pretoken alternative);
left of the window the encodes coincide by construction (pretokens are
bounded and context-free, merges stay inside pretokens — documented in
the file header). Any failure → today's `applyChatTemplate`; the stored
entry is never mutated. Integration: `PrefillPlanner.detectBoundaries`
gained `modelFingerprint:` (threaded one line from `ServerCompletion`),
engaged only in the text key space (`keySpace.isIdentity` — image
spaces keep the full render for placeholder translation). Gates: new
`RenderTokenCacheTruncated{Fake,Real}Tests` (13 tests — hit exactness,
six end-of-text right-context classes, assistant/system tail, wrong
context, spanning-token-at-cut, cold), C25/C26 suites unchanged-green,
`PrefillPlannerTests`, `--prefix-cache-e2e` PASS, runner both models
(parent-verified rerun): **truncated leg 40.19 → 2.87 ms (4B,
−92.9%)**, 40.60 → 2.90 ms (MoE), 0 mismatches; the two engineered
fallback turns fall back exactly. **Verdict: ACCEPTED.** Steady-state
agent/server turn tokenize cost at 11K tokens is now ~8.4 ms
(render+suffix 5.5 + truncated 2.9) vs ~82 ms pre-C25 (−90%). App-only
change; no pins moved. Remaining on this path: the post-generation
leaf-store/admission encodes (up to 3 full renders serialized on
`container.perform` per turn) — C28.

### C28 — post-generation leaf-store/admission encodes through the cache — ACCEPTED

The last full-size encodes on the steady-state server turn: after each
response, inside `container.perform` (serialized against the next
request), `LeafStorePhase.measureStoredTokenSequence` re-tokenizes
prompt+assistant-turn (~40 ms at 11K), and `LeafAdmissionBuilder.
reusablePrefix` renders twice more (base + probe, `add_generation_prompt:
false`) when a boundary leaf mode is selected. All three go through the
cache via one composed resolve — `resolveReplacingTail`: keyed candidate
(base-context digest, strict-extension chain, fingerprint, template),
the C25 trim-back walk to the byte seam (per-token strip, k = 0…4),
suffix-encode the extension, `verifyJunction` as the arbiter; entry
never mutated. (Evidence-based deviation from the planned
resolve/resolveTruncated decomposition: the leaf-store render uses the
merged gen-prompt-off context and its content sits where the entry's
gen-prompt tail is, so all three seams are the same trim+extend shape;
`verifyCut` is not needed — the cut is always followed by the fresh
suffix, which is exactly the junction window's case.) A 1+1-token seam
pre-check per trim attempt skips the futile 256→16384 enlargement
ladder when the seam pair provably merges (the empty think scaffold
made every k=0 attempt pay it; `verifyJunction` stays the authority
whenever the pre-check passes — exactness unchanged). Gates: three new
suites (composed trim+extend exactness, reply-starter classes,
probe continuation, wrong context/fingerprint/tools/edited base,
spanning-merge past trim budget, entry-never-mutated) +
`LeafAdmissionCachePathTests`; RenderTokenCache suites, PrefillPlanner,
LeafAdmissionBuilder, HTTPPrefixCacheSpike, PrefixCacheIntegration all
green; `--prefix-cache-e2e` PASS (incl. the direct-tool-leaf and
canonical-user-leaf paths). Runner both models (parent-verified rerun):
**leaf-store 40.35 → 6.14 ms, admission 40.34 → 6.11 ms, probe 40.41 →
5.38 ms (−85…−87%)**, `replacedFallbacks=0`, 0 mismatches; MoE 41.2 →
5.3–6.1 ms. **Verdict: ACCEPTED.** ~120 ms → ~18 ms of serialized
post-generation encode per turn at 11K tokens (~3× more at 32K),
directly off next-turn TTFT. App-only change; no pins moved. The
tokenize line is now: cold turn ≈ one full encode (C24 attacks that
encoder itself), warm turn ≈ render+digests (~5.5 ms) + ~12 ms of
cached legs, vs ~205 ms pre-C25.

### C28 — post-generation leaf-store/admission encodes through the cache — ACCEPTED

The last full-size encodes on the steady-state server turn: after each
response, inside `container.perform` (serialized against the next
request), `LeafStorePhase.measureStoredTokenSequence` re-tokenizes
prompt+assistant-turn (~40 ms at 11K), and `LeafAdmissionBuilder.
reusablePrefix` renders twice more (base + probe, `add_generation_prompt:
false`) when a boundary leaf mode is selected. All three go through the
cache via one composed resolve — `resolveReplacingTail`: keyed candidate
(base-context digest, strict-extension chain, fingerprint, template),
the C25 trim-back walk to the byte seam (per-token strip, k = 0…4),
suffix-encode the extension, `verifyJunction` as the arbiter; entry
never mutated. (Evidence-based deviation from the planned
resolve/resolveTruncated decomposition: the leaf-store render uses the
merged gen-prompt-off context and its content sits where the entry's
gen-prompt tail is, so all three seams are the same trim+extend shape;
`verifyCut` is not needed — the cut is always followed by the fresh
suffix, which is exactly the junction window's case.) A 1+1-token seam
pre-check per trim attempt skips the futile 256→16384 enlargement
ladder when the seam pair provably merges (the empty think scaffold
made every k=0 attempt pay it; `verifyJunction` stays the authority
whenever the pre-check passes — exactness unchanged). Gates: three new
suites (composed trim+extend exactness, reply-starter classes,
probe continuation, wrong context/fingerprint/tools/edited base,
spanning-merge past trim budget, entry-never-mutated) +
`LeafAdmissionCachePathTests`; RenderTokenCache suites, PrefillPlanner,
LeafAdmissionBuilder, HTTPPrefixCacheSpike, PrefixCacheIntegration all
green; `--prefix-cache-e2e` PASS (incl. the direct-tool-leaf and
canonical-user-leaf paths). Runner both models (parent-verified rerun,
and again after a lint-only file split): **leaf-store 40.35 → 6.14 ms,
admission 40.34 → 6.11 ms, probe 40.41 → 5.38 ms (−85…−87%)**,
`replacedFallbacks=0`, 0 mismatches; MoE 41.2 → 5.3–6.1 ms. **Verdict:
ACCEPTED.** ~120 ms → ~18 ms of serialized post-generation encode per
turn at 11K tokens (~3× more at 32K), directly off next-turn TTFT.
App-only change; no pins moved. The tokenize line is now: cold turn ≈
one full encode (C24 attacks that encoder itself), warm turn ≈
render+digests (~5.5 ms) + ~12 ms of cached legs, vs ~205 ms pre-C25.
(Housekeeping folded into this commit: `TokenizeCacheBenchRunner`
summary extraction + the RenderTokenCache test file split into
`RenderTokenCacheTests` / `RenderTokenCacheRealTests` /
`RenderTokenCacheTestSupport` for the pre-commit lint limits.)

### C24 — byte-native serial BPE inner loop + byte-keyed lookup tables — ACCEPTED

The last encoder lever from the C21 profile: ~34% of the 126 ms 32K
encode was the serial BPE inner loop + token→id conversion — one String
allocation per Unicode scalar on entry (`BPETokenizer.bpe`), a String
concat per merge, and String-keyed dictionary probes (NFC-walking
hashes, NSString bridging) for every pair rank and token id. The
rewrite keeps the SAME serial algorithm with the SAME merge order:
initial symbols are byte ranges into one UTF-8 buffer (identical
boundaries to `unicodeScalars.map { String($0) }`), merges extend a
range instead of concatenating, and ranks/ids resolve through
open-addressed byte-keyed tables derived 1:1 from `bpeRanks` /
`tokensToIds` (FNV-1a over raw bytes + split point, full byte-compare
verification; `(rank, left)` heap tie-break and stale-entry re-check
unchanged; lazy build-once behind a lock, ~20 MB resident). Rig gates
(coder subagent): **88/88 corpus items byte-identical final ids** (the
26-class tokdiff corpus + 18 merge-stress adversarial items — long
runs, tie-heavy segments, combining runs, ZWJ/keycap, all scalars
singly+doubled — 6.7M tokens, both model tokenizers); 50/50 repeat
encodes identical; ABBA **1.22× at 32K (130.05 → 106.68 ms), 1.21× at
8K, 1.20× at 128** — no short-input guard needed (the byte-keyed id
lookup wins even at 129 tokens). App A/B (same binary, `C24_OLD=1`
env-gated legacy arm, 3-pair both models): **gates 9/9 + 9/9
token-identical**, peaks exactly flat; tokenize deltas consistent with
the rig on 5/6 legs (dense 8K/32K −15.3%/−21.4%, MoE 32K −13.6%,
MoE 8K −1.2% — the session was thermally distressed; prefill/decode
deltas were both-directions-impossible across the two tables, the
documented environmental signature, and the mechanism provably cannot
touch them — CPU-only, pre-model, gates prove the inputs identical).
**Verdict: ACCEPTED** on rig exactness + rig timing + app gates.
Ported STRIPPED of the probe scaffolding (env toggle and the verbatim
legacy path dropped — the fork carries upstreamable product, not
measurement harness; the benched default path is byte-for-byte what
shipped): `spokvulcan/swift-transformers` `pin-tesseract` @ **a524093**
(`swift build` green pre-push), `Vendor/mlx-audio-swift` pin moved,
checkout re-resolve verified `diff 1.3.3 == fork diff`, clean-build
confirmation rebuilt + parity smoke leg vs the pre-C24 binary **2/2
token-identical**. `docs/swift-transformers-fork.md` carry table
updated. Upstream filing queued (owner go-ahead).

### C30 — attribution: non-tokenize CPU per agent/server turn (measurement, no verdict)

New `--agent-cpu-bench` runner (12-turn trajectory, real tokenizer, 5
interleaved reps/phase, quiet machine) timing the per-turn CPU OUTSIDE
tokenize/prefill/decode. Findings at 11.4K-token conversations:

- **Total accounted ≈ 20 ms/turn, but only ≈ 4.8 ms on the TTFT path.**
- **p4 boundary detection (memo-warm `PrefillPlanner.detectBoundaries`)
  = 4.1 ms/turn, TTFT-path, growing with history** (2.56 → 4.10 ms over
  turns 1→12). Composition (not yet sub-attributed): the memo-hit
  StablePrefixDetector detect (SHA-256 of the 33 KB system prompt +
  JSONSerialization of 40 tool specs + token-hash verify over the
  7,805-token prefix) + gen-prompt encode + the C27 `resolveTruncated`
  hit (full-conversation render + second digest chain + trim +
  cut-verify) + translatedLength. Incidental control measurement: with
  C27 forced to fall back, detectBoundaries costs 34–43 ms/turn —
  independent confirmation of C27's ~30–39 ms/turn win at this scale.
- **p5 detok 15.2 ms/turn but amortized across the GPU-bound stream —
  NOT a TTFT or tok/s lever** (≈0.15% of a core steady-state; even the
  newline-free worst case — the O(segment²)
  `NaiveStreamingDetokenizer.next()` re-decoding the whole segment per
  token, 288 µs/token — is ~3% of a core at production decode rates).
  Logged as an efficiency note, not a loop target; the O(n²) would only
  matter for very long newline-free generations, and even then stays
  off the critical path.
- **p1 conv-build 0.33 ms, p2 canonicalize 0.34 ms, p3 keying ~0 ms,
  p7 radix 0.02 ms — all flat, all closed.** But note the redundancy:
  the same 40 tool specs are JSONSerialized+SHA-256'd **4× per turn**
  (AgentConversationBuilder, MessageConverter, RenderTokenCache,
  StablePrefixDetector key), the digest chain runs 2×, and the
  leaf-store and admission-stored renders are **the identical render
  computed twice** (verified in the C28 implementation notes).
- SKIPPED (uncallable without GPU/SSD fixtures): snapshot
  capture/restore and SSD manifest bookkeeping.

**Aim for C31:** per-request render/digest consolidation — a memo
scoped to ONE request (zero staleness surface by construction) sharing
the tools digest, the system-prompt hash, and the identical renders
across the six consumers. Expected ~2–3 ms/turn at 11K, ~30+ ms/turn
at 131K (5 full-conversation renders per turn today). Sub-attribute p4
(memo-detect vs C27-render vs chains) as part of the implementation.

### C31 — compute-once-and-plumb in the request flow — ACCEPTED (small)

Aimed by the C30 attribution (p4 boundary detection = 4.1 ms/turn, the
only TTFT-path non-tokenize CPU). Sub-attribution first (new
`--agent-cpu-bench` sub-phases, kept in the runner): at turn 12 (11.4K
tokens) p4 = memo-detect 0.59 + truncated Jinja render 1.53 + digest
chain 0.62 + tools/ctx digests 0.38 + trim/cut-verify 1.24 +
gen-prompt encode 0.02. Three sites changed, all compute-once-plumb
(zero behavior change — byte-identical values by construction): (1)
`resolveTruncated` reuses the entry's stored chain head under a
caller-asserted `messagesAreEntryPrefix` (PrefillPlanner's truncation
is a prompt-message prefix of the conversation RequestKeyingPhase just
resolved; cumulative hashing — the same values, not recomputed; the
head-match guard still runs, the render arbiters remain the exactness
authority); (2) `LeafStorePhase` hands its already-computed
`storedRenderTokens` to `LeafAdmissionBuilder.plan` (the admission
base render was the identical computation — one full render+resolve
eliminated per boundary-mode turn, ~5–6 ms of serialized
post-generation work; verified by construction + e2e, no gate times
it); (3) the StablePrefixDetector memo-key audit: LEFT — the tools
recipes differ by design across components (canonicalized vs raw vs
wire-type serializations), unification would be a key-recipe change
with radix/SSD migration surface. Gates: 13 cache/parity suites (313
tests incl. 7 new), `--tokenize-cache-bench` both models PASS (0
mismatches; truncated leg 2.87 → 2.51 ms), `--prefix-cache-e2e` PASS,
`--agent-cpu-bench` p4 **4.21 → 3.50 ms/turn at 11.4K** (measured
−0.71 ms, growing with history; honest miss of the −1.5–3 ms estimate —
the render and the cut-verify are the exactness arbiters and stay).
**Verdict: ACCEPTED** — ≥1% on the agent-cpu p4 metric, no regression
anywhere, exactness contract untouched.

### C29 — incremental digest chain on the resolve paths — ACCEPTED (small)

The per-request digest chain (per-message cumulative SHA-256) was
recomputed in full on every resolve — 0.62–0.93 ms at 11.4K tokens per
call (C31 sub-attribution), three calls per turn on the hit path
(`resolve`, and the two C28 `resolveReplacingTail` legs). C29 reuses the
stored entry's chain head and hashes only the tail messages: the chain
is cumulative, so a conversation extending the stored one has identical
head values by construction. **Exactness analysis (the load-bearing
part):** the chain is a candidate-selection pre-filter, never an
exactness arbiter — the arbiters are the byte-prefix render check and
the junction/cut verifications, which are untouched. A head that does
not match (edited history) vacuously passes the reused guard and is
rejected by the render instead: same miss, same full-encode fallback,
same tokens — only the miss REASON changes (`.digestMismatch` →
`.renderNotExtended`; the fake-suite expectation and its comment were
updated to say so). Gates: all 13 cache/parity suites TEST SUCCEEDED;
`--tokenize-cache-bench` 4B PASS (0 mismatches; hit turns 5.05 ms,
truncated 2.57 ms, C28 legs 5.2–5.5 ms — also reflecting C24's encoder
win; the edited-history turn misses via the new reason and stays exact);
`--agent-cpu-bench` p4 flat at 3.47 ms (C29's resolves live outside p4 —
correct). Eliminates ~1.5–2 ms/turn of chain recomputation at 11.4K
tokens, growing linearly with history (~20 ms/turn at 131K). **Verdict:
ACCEPTED** — mechanism is construction-identical, gates all green, no
regression channel. Committed with C31 (one compute-once commit).

## Review round 2026-07-27 — PR #429 full-diff review fixes

Ten findings from the full-diff review of the C24–C31 session, fixed on the
same branch. Six are hardening, three are real defects, one is a
claims-vs-evidence correction. No experiment verdict changes; no measured
number in this session's entries is affected (all fixes are on the same CPU
paths, none touches an arbiter).

**F1 — `String` canonical equivalence where the contract said "byte prefix"
(real defect, medium).** Every prefix/suffix/equality test in
`RenderTokenCache` was a Swift `String` operation, and `String`'s `==`,
`hasPrefix` and `hasSuffix` compare under Unicode **canonical equivalence** —
so an NFC render and an NFD render of the same text were `==` while their bytes,
and therefore their token lists, differed. The junction and cut arbiters cannot
catch this class: they decode the cached tokens and re-encode that decode, which
is self-consistent by construction and never re-examines the new render's bytes.
A normalization-shifting client (JSON clients that normalize string payloads do
exist) could therefore be served the *other* byte string's tokens from the
repeat path — the one resolve with no empirical arbiter behind it — and those
tokens then key the radix cache. Fixed by moving the whole type into byte space:
`Entry.renderedBytes: [UInt8]`, and prefix/suffix/trim arithmetic through
explicit byte helpers (`RenderTokenCache+Keys.swift`). The trim walks now carry
a `suffixString` round-trip guard so a cut landing mid-scalar degrades to a miss
instead of encoding U+FFFD. Regression test:
`normalizationShiftedRepeatDoesNotServeCachedTokens` — under the old code it
returned `.hitRepeat` with the wrong list; it now misses via
`.renderNotExtended` and re-encodes exactly. The fake `GreedyTokenizer` was
itself normalization-insensitive (`String.hasPrefix` matching) and had to be
made byte-faithful first, or the test could not have failed.

**F2 — C24 narrowed the merge-table match semantics, and the comment claimed
otherwise (real, upstream-blocking).** `BytePairTables.rank(in:...)` matches
merge pairs by raw bytes; the `bpeRanks[BytePair(l, r)]` probe it replaced
matched under canonical equivalence, because `BytePair` holds Swift `String`s.
The doc comment asserted equivalence and cited `BinaryDistinctString`, which is
not involved in `BytePair` at all. Unobservable on byte-level BPE vocabs (hence
88/88 over 6.7M tokens), potentially output-changing on a non-byte-level vocab
with mixed normalization — where it is a *fix*, not a regression. Comment
rewritten to state the narrowing and why byte-exact is intended;
`docs/swift-transformers-fork.md` gains a semantics note that the upstream PR
must carry. Upstream has already fixed two Unicode bugs in this code (#352 Bugs
3 and 4), so the question will be asked.

**F3 — `byteTables()` double-checked locking raced (real defect, medium).** The
fast-path read of `byteTablesCache` was unsynchronized against the write inside
the lock. Tearing was not the issue; ordering was — a plain store publishing the
pointer has no release semantics, so a reader on another core could observe a
non-nil cache before the eight array buffers it points at were visible.
`BPETokenizer` is `@unchecked Sendable` and shared. Fixed by building the tables
eagerly in `init` into a `let`: both source dictionaries are `let`s complete by
the end of `init`, so there was nothing to defer, and C23 already measured
tokenizer load as fully hidden behind the weight load.

**F4 — `?? "unfingerprinted"` collapsed distinct models onto one key
(hardening).** Two of the five seams engaged the cache under a synthetic shared
key when the model fingerprint was unknown; the other three bypassed. Under the
synthetic key two models would share both the entry and the memoized
`templateHashes` slot, making the template-mismatch check pass vacuously — and
the repeat path would then trust (bytes, fingerprint) alone. Believed
unreachable today (`installLoadTimeState` always receives a non-nil `String`),
but it inverted the safe default in the one place with no arbiter. All five
seams now bypass on `nil`.

**F5 — `entry` was read three times per `resolve` (hardening).** An entry
changing between the chain build and the candidate select would store a chain
whose head does not derive from those messages, permanently vacuating the
head-match pre-filter for the singleton (exactness would hold on the render
arbiters; a documented guard would be silently dead). One snapshot per resolve.

**F6 — no observability on a subsystem whose safety mechanism is silent
degradation (hardening).** Zero `Log` calls, `statsSnapshot()` with no
production consumer, `reset()` with no production call site, and five `try?`
seams swallowing throws. `Stats` gains typed reason histograms
(`missReasons`, `truncatedFallbackReasons`, `replacedFallbackReasons`),
`junctionFailures`/`windowEnlargements` are split per path (C25 vs C27 vs C28 —
conflated counters hide which path regressed), a summary lands in `Log.server`
every 256 resolves, a throwing render is logged before it propagates, and
`LLMActor.unloadModel` now logs the session summary and calls `reset()` (the
entry held a whole render's bytes plus its token list — megabytes at long
context — with no model resident).

**F7 — the cache-eligibility predicate was spelled five different ways
(design).** New `RenderTokenSource` is its single home, and carries the C31
plumbed base render with it, so the pair travels together instead of as two
independent optional parameters (`LeafAdmissionBuilder.plan` 10 → 9 params,
`reusablePrefix` 8 → 7, and its two near-identical 20-line resolve ladders
collapse to one local helper). The request-path seam also stops using
`imageKeying == nil` — "does the app RECOGNIZE a vision container" — as a proxy
for the property it actually needs, "does this model's processor emit flat 1-D
tokens". Those coincide only because `qwen3_5` + `vision_config` is today's sole
recognized family; a future VLM family added without an image-keying rule would
have silently received 1-D tokens where its processor emits 2-D. `ModelSession`
now exposes `producesFlatTextTokens` (`context.model is any LLMModel`) — the
same marker protocol both installed processors branch on, and the same
feature-detect-as-a-fact shape as `anchoredVisionPrepare`.

**F8 — nits.** `trailingTokenCount`/`leadingTokenCount` documented a "smallest"
count they never returned (the doubling probe overshoots) and named their
parameter `coveringCharacters` while measuring UTF-8 bytes — both corrected, and
the 256/16384 literals are now named constants. `canonicalForm` checked `Bool`
before the integer cases, and `NSNumber(1) as? Bool` succeeds — so JSON-decoded
`1` and `true` canonicalized identically; an `NSNumber`/`CFBooleanGetTypeID`
case now runs first (test: `canonicalFormSeparatesBooleansFromNumbers`). The
C24 leading-byte scalar-width walk is bounded against a malformed buffer. The
C31 `messagesAreEntryPrefix` doc claimed the assertion always holds; it holds
only when Request Keying resolved through the cache, so it is now documented as
a cost hint, never a correctness input.

**F9 — claims vs evidence in the PR body (correction).** "Zero effect on
prefill/decode/peak by construction" and "peaks byte-flat" are MLX/GPU peak;
C24's byte tables add **~20 MB resident per loaded tokenizer** (recorded in the
C24 entry above, absent from the PR body — now stated in both, and in the fork
carry table). The single-entry global is also shared by the agent path and the
server path, so interleaved agent+server traffic thrashes it toward a 0% hit
rate; the measured wins are single-stream and the PR body now says so.

**F10 — model-gated coverage (disclosure).** The real-tokenizer exactness suites
and the Gate-1 parity suite are `.enabled(if: modelAvailable)`, so on a machine
without `z-lab_Qwen3.5-4B-PARO` the "313 tests" figure is mostly the
fake-tokenizer half. There is no CI running these; stated in the PR body rather
than papered over. New tests this round: the normalization regression, a
byte-identical-repeat guard, `reset()`, miss-reason counting, the
boolean/number canonicalization, and four `RenderTokenSource` eligibility cases.

**Gates for this round:** app build clean (Debug + Release,
`xcodebuild build` / `dev.sh dev-release`), test target clean
(`build-for-testing`), **157 tests in 18 suites PASS with 0 skipped** — the
real-tokenizer suites ran against the on-disk PARO model, so the byte-space
rewrite is exercised against the actual Qwen3.5 tokenizer/template and not only
the fake — swiftlint + swift-format clean on every changed file (the three
residual swiftlint warnings reproduce on the `HEAD` versions), `check-docs.sh`
green, and `swift build` green in `~/projects/swift-transformers` with F2/F3/F8.

**Re-run after the fork pin moved to `0033bc7` (2026-07-27).** F1 rewrote the
exactness-critical inner logic of all three resolves (String → byte space), and
the per-turn intrinsic assertions in the tokenize runner are this program's
binding exactness gate — the unit suites are necessary, not sufficient, by this
ledger's own rules. So the full bench leg was re-run against the built tree with
F2/F3/F8 in it (`Vendor/mlx-audio-swift` Package.swift + Package.resolved moved
`a524093` → `0033bc7`; the DerivedData checkout's `1.3.3..HEAD` diff verified
byte-identical to the accepted fork diff):

| Leg | Model | Result |
| --- | --- | --- |
| `--tokenize-cache-bench` | Qwen3.5-4B-PARO | **PASS** — 11 hit turns 36.05 → 3.68 ms (−89.8%); C27 15 turns 34.64 → 2.06 ms (−94.1%); C28 leaf-store 35.64 → 4.35 (−87.8%), admission 35.60 → 4.33 (−87.8%), probe 35.60 → 3.47 (−90.3%). **0 token mismatches, 0 parity failures, 0 path failures** |
| `--tokenize-cache-bench` | Qwen3.6-35B-A3B-PARO | **PASS** — 11 hit turns 35.91 → 3.68 ms (−89.7%); C27 34.57 → 2.11 ms (−93.9%); C28 leaf-store −87.7%, admission −87.7%, probe −90.1%. **0 token mismatches, 0 parity failures, 0 path failures** |
| `--prefix-cache-e2e` | Qwen3.5-4B-PARO | **PASS** — 32/32 assertions, including `greedy_output_equivalence`, `image_warm_output_equivalence` and `agent_image_output_matches_http_path` all "fully identical" |

Both models report the same F6 histogram shape, which is itself the observability
finding paying off — the byte-space rewrite is legible in the stats line rather
than inferred: `hits=11 repeats=1 misses=5 trimHistogram=[k2:11]
missReasons=[cold:1,digestMismatch:2,renderNotExtended:2] junctionFailures=0
replacedJunctionFailures=0 junctionWindowEnlargements=0 cutWindowEnlargements=0
truncatedHits=15 truncatedFallbacks=2
truncatedFallbackReasons=[renderNotPrefix:1,tailTooLong:1] replacedHits=45
replacedFallbacks=0 replacedFallbackReasons=[]`. Zero junction failures and zero
window enlargements on both tokenizers means the F8 window constants
(`initialWindowBytes` 256, `maxWindowBytes` 16384) are not being stressed by real
templates; the enlargement counters exist to catch the day that changes.

The one gate deliberately substituted: no local `dev.sh clean` before these
timings. A full DerivedData wipe + rebuild is ~40 min of sustained load, which is
exactly the condition trap 2 (Thermals) warns about — the M3 Max throttles under
it, and absolute timings are not comparable across a thermal transition. Running
the bench immediately after a clean would have disadvantaged it against the
baselines it is measured against. CI's `build-release` on a pristine runner is the
clean-build confirmation instead; the exactness assertions (0 mismatches, 0 parity
failures, 0 path failures) are thermally invariant either way, so the substitution
costs nothing on the correctness gate — only on the timing gate, where it helps.

---

## Session 2026-07-28 — chunked gated-delta scan

Git HEAD at session start: `a73a7aa5`. Prompted by MoonshotAI's **FlashKDA**
release (CUTLASS Kimi Delta Attention kernels, announced at 1.72×–2.22× prefill
speedup over flash-linear-attention on H20).

Built, measured, rejected, and **reverted completely per rule `:22`** — no code
from this experiment survives on any branch. This entry is the whole record, so
it carries enough detail to stop the question being reopened from scratch.

### E12 — chunked (blocked-matmul) gated-delta scan — REJECTED

**Hypothesis.** E8 measured the GDN scan as sequential-latency-bound: ~0.5 µs
serial per step per CTA over a T-deep chain, explicitly not bandwidth. The
chunked delta rule — the form every CUDA implementation uses, stated readably in
FlashKDA's `tests/torch_ref.py:180-245` — shortens that chain to one step per
C-token block and turns the rest into batched matmuls, which this stack runs at
82–88% of peak (#251, E9). E8b closed this line on *legality*, never on
measurement: nobody had ever measured what chunking buys on Metal.

**Two corrections to the premise, found by reading before building.** Both
shrink the prize and are worth keeping:

- **FlashKDA implements KDA, not GDN.** KDA gates per channel
  (`g:[B,T,H,K]`); Qwen3.5/3.6 hybrids use gated-delta's scalar per-head gate.
  The headline 1.72–2.22× is over FLA's *KDA* Triton kernel. Over FLA's *GDN*
  kernel — the comparable one — FlashKDA's own `BENCHMARK_H20.md:16-26` shows
  **1.17×–1.43×**.
- **The scan is ~5.5% of a 32K prefill** (E8 `:326`), ~8% at 8K, and chunking
  helps prefill only — T=1 decode has no chunks. Even a 2× scan is under 3%
  end-to-end.

**Change (since reverted).** A chunked scan in plain MLX ops, specialized from
KDA's per-channel gate to the scalar per-head gate, behind a per-call/per-process
backend switch defaulting to the existing recurrent Metal kernel. Followed
FlashKDA's two-stage split (deep-dive §2): decays, the `L`/`M` Gram matrices and
the Neumann inverse computed for *all* blocks in one batched pass, with only the
state recurrence looping.

**Two numerics findings, both real, both cost a debugging round.** These are the
part most worth keeping, because they will bite any future attempt:

- **Form pairwise decays as `exp(cumsum_t - cumsum_j)`, never as `Γ_t × 1/Γ_j`.**
  FlashKDA builds `k_decayed` and `k_inv` separately (`torch_ref.py:205-212`),
  which it can afford only because KDA bounds its gate at `lower_bound = -5` and
  fixes C=16. Gated-delta's gate is `exp(-exp(A_log) · softplus(a + dt_bias))`,
  unbounded below, so the separated form overflows on the reciprocal. This is a
  **deviation from** FlashKDA, not a borrowing from it.
- **bf16 cannot carry the Neumann series — promote to f16, not f32.** Drift
  against the recurrent form runs 0.005–0.009 in bf16 versus 0.0007–0.0012 in
  f16. The tell that the bf16 residual was *output quantization* rather than
  blocking error: it did not move with C at all, sitting at exactly
  `0.0068965508` for every block size, and the ratio to f16 is the 8× mantissa
  ratio (2⁻⁸ vs 2⁻¹¹). Same finding and same fix as FlashKDA deep-dive §3.

**Correctness reached, so the timings compare like with like.** At production
dtype (f16) and head geometry, `max_rtol` vs the recurrent form was
**0.0005–0.0011** on outputs and **0.0005–0.0009** on state, across
C ∈ {16,32,64} and T ∈ {64,128,256}. Masked calls, spans below one block
(including T=1 decode) and unsupported block sizes fell back **bitwise**.

Note this required relaxing rule `:17`: a chunked scan cannot be
token-identical to a recurrent one, so it was held to a tolerance bar against
the recurrent form instead. That relaxation was scoped to this experiment and
died with it — **rule `:17` stands unchanged for every other line.**

**Measurement.** Purpose-built harness timing the scan directly with no model
load (its cost is shape-driven, so synthetic tensors at production dims
reproduce it exactly). Release, ABBA within round, 5 rounds × 10 iterations,
quiet machine verified. Shapes per E8: q/k `[1,T,16,128]`, v `[1,T,32,128]`,
f16 in, f32 state.

Harness cross-check against E8: recurrent measured 1.03 ms at T=512, 1.83 ms at
T=1024, 3.72 ms at T=2048 — against E8's 1.2 / 2.0 / 3.8 ms. Consistent, so the
baseline arm is not sandbagged.

Speedup vs recurrent (>1 would be a win):

| T | C=16 | C=32 | C=64 | C=128 | C=256 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.37× | 0.53× | **0.58×** | 0.48× | — |
| 512 | 0.31× | 0.47× | **0.64×** | 0.49× | 0.23× |
| 1024 | 0.30× | 0.49× | **0.62×** | 0.51× | 0.22× |
| 2048 | 0.30× | 0.50× | **0.67×** | 0.51× | 0.22× |

**Verdict: REJECTED.** 1.5–3.4× *slower* than the recurrent kernel at every
configuration. Round-to-round spread ±0.02, so this is not noise. Rejected on
speed alone — the numerics gate passed with 5–10× margin.

**Why, and why the result is well-bounded.** The optimum in C is **interior** —
0.67× at C=64, falling to 0.51× at C=128 and 0.22× at C=256 — so this is not "we
did not try a large enough block". Two costs squeeze from both sides: below C=64
the sequential block count dominates, above it the Neumann inverse does, growing
as C³ (`log2(C) - 1` doubling rounds of C×C matmuls).

The chunked form is not slow because it is inefficient. At C=64 it does ~3.7×
the arithmetic of the recurrent form (~16 GFLOP vs ~4.3 GFLOP per layer at
T=2048) at ~2.5× the recurrent kernel's effective rate — roughly 2.8 TFLOP/s
against 1.1. It loses because 2.5× does not pay for 3.7×. And 2.8 TFLOP/s is far
below the ~10–12 this stack reaches on large GEMMs, for a reason specific to the
ops level: the GEMMs are small (M=C, batched over 32 heads) and **every
intermediate round-trips through global memory** — seven materialized tensors per
block step. That is precisely the cost FlashKDA's fused K2 stage removes with
register-file transposes (deep-dive §4), and it is not reachable from MLX's op
graph.

**What this closes, and the one door left ajar.** It closes the **ops-level**
chunked scan. It does not prove a *fused Metal* chunked kernel loses — but that
kernel would have to overcome a 1.5× deficit while doing 3.7× the arithmetic,
i.e. reach roughly 5.5× the recurrent kernel's FLOP rate (~6 TFLOP/s at these
small-M shapes) merely to break even, and then earn back its cost against a scan
worth 5.5% of prefill. Given #251 already found the fused-attention equivalent
slower than unfused, and #256 measured small-M GEMM efficiency at 43% of peak,
that is not a promising bet. If anyone does try it: `MLXFast.metalKernel` takes a
`header:` parameter (`MLXFastKernel.swift:176`), so `#include
<metal_simdgroup_matrix>` inside a JIT kernel is expressible.

**Reference material.** FlashKDA cloned at `~/projects/FlashKDA`. The kernels are
CUTLASS/SM90+ PTX (`MOVM_T`, `tanh.approx.f32`, `ex2.approx.ftz.f32`) and are not
portable; the portable artifact is the 65-line reference at
`tests/torch_ref.py:180-245`, plus `docs/20260420-flashkda-v1-deep-dive.md`.

**Untouched by this result.** The **bf16 recurrent-state switch** (FlashKDA §3 —
fp32 FMA with bf16 storage, halving the 63 MB MambaCache snapshot) was scoped
into this round and never built. It targets decode bandwidth and snapshot size,
neither of which chunking touches, so nothing here argues against it.

## Session 2026-08-20 — DFlash2 speculative decoding: verify-pass kernel fix

Context: ADR-0057 (DFlash2 port for Qwen3.8-27B + `incoai/Qwen3.8-27B-DFlash2`
draft). First app benches showed DFlash2 *slower* than AR decode (bs8 16.2 vs
AR 20.6 tok/s) despite acceptance matching the Python reference (32–36%).
Per-phase profiling (`DFLASH2_PROFILE=1`, phase timers in the speculative
iterator) isolated it: verify = 166.6 ms of a ~200 ms bs8 round, ~17 ms/row,
and the Python reference (mlx 0.32) showed the same ~150 ms verify at S = 9 —
an MLX-level kernel issue, not a port bug.

### K1 — verify GEMM M-scaling diagnosis (probe: /tmp/qmvprobe)

`QuantizedMatmul::eval_gpu` (Cmlx `quantized.cpp`) routes transpose GEMMs with
M < `get_qmv_batch_limit` (= 12 for K,N > 4096 on gen-15 'd') to `qmv`, whose
grid is `(M, N/8, B)` — **weights are re-streamed once per row**. A DFlash2
verify pass at block size bs runs M = bs, so bs8 re-read the 11.9 GB of 4-bit
weights ~8×. Measured per-shape M-scaling (Swift probe against the live
DerivedData Cmlx checkout, per-iteration `eval`): qmv cost grows ~linearly in
M; lm_head (5120×248320) M=1 2.08 ms → M=8 4.93 ms even *with* reuse (below).

### K2 — qmv_wide backport — ACCEPTED

Upstream `mlx#3764` (548dd80e8, "small-batch quantized matvec", landed after
our v0.31.1 pin) adds `qmv_wide`: each weight group decoded once, reused
across ≤ 5 input rows (tile cap 5, k_lanes 8 for affine). Backported verbatim
(affine half) onto the `pin-tesseract` Cmlx: kernels/quantized.h +163,
quantized.metal +24, host quantized.cpp +96 (encoder API adapted to
`d.get_command_encoder(s.index)`); JIT string mirror in
`mlx-generated/quantized.cpp` kept byte-identical (verified by region diff
against the upstream commit). Gates: 11 synthetic + rollback + round-0 parity
+ draft parity + 2 e2e trace tests green on the final tree; numerics probe
(batched M vs per-row reference) rel-L∞ ≈ 1% at every M — uniform bf16
reduction-order noise, no M-specific defect.

Measured (app ABBA, 192 new tokens, 6K prompt, Release):
bs5 verify 116 → 82–97 ms; bs5 18.3 → 26.4–27.9 tok/s (1.20–1.30×);
bs8 flat (2 tiles at cap 5 ≈ 2.4 streams; ALU-bound, not bandwidth).

### K3 — tile cap 5 → 8 — REJECTED

One-stream-per-verify idea: bs8 in a single tile. Probe + app bench: slower
(verify 150 → 162 ms) — register pressure/occupancy at nv=8 beats the
bandwidth saving. Reverted to upstream cap 5.

### K4 — qmm_t / k_lanes 16 / caps 2,3 at M ∈ 2..16 — REJECTED

Forced-`qmm_t` (env `QMV_FORCE_QMM`, probe-only hook, stripped after): flat in
M (streams W once) but 91 GB/s on lm_head ≈ 3.5 streams — the tiled kernel is
not competitive at small M on this part. `QMV_KL=16`, caps 2/3: all lose or
tie stock. Upstream's cap-5/kl-8 tuning stands on M3 Max.

### K5 — block-size scan + verify mode

bs ∈ {3,4,5,6,8} scan (acceptance per draft position: 59/50/41/36/22%):
verify M = bs, so bs ≤ 5 rides a single qmv_wide tile; bs3 is the optimum.
Eager vs compiled verify (env `DFLASH2_VERIFY=eager`): within noise at bs5,
compiled ahead at bs3 — compiled stays default. **Production default changed
to block 3** (`DFlash2Support.blockSize`; cool-machine ABBA: 31.0–31.3 tok/s,
1.42–1.43× over AR 21.8–22.1; hot-machine repeat: 1.20×, thermals per rule
:32). Final round anatomy at bs3: propose 10.2 (draft body 9.3 dispatch-bound
vs 2.4 ms bandwidth floor — compiled draft body is the known next lever),
verify 58–62, accept+reconcile ~1.5.

### K5b — reference cross-check at bs3/bs5

`research/bench_dflash.py --blocks 3,5` (Python reference, mlx 0.32 with
native qmv_wide, same 6K prompt): AR 17.0, bs3 31.5 (1.85×), bs5 23.0 (1.35×).
The reference peaks at block 3 too — the optimum is algorithmic. Absolute
DFlash2 decode is cross-stack identical (31.5 vs 31.3 tok/s); our speedup
ratio is smaller because the app's AR baseline is 26% faster (21.4–22.1 vs
17.0), leaving speculation less headroom. (Watchdog footnote: the footprint
parser misfired on a "K"-suffixed reading after the run completed — harmless,
but parse `vmmap` units before comparing.)

### K6 — long-context attempt at 65K — machine limit, aborted

`--bench-context-mult 4` (23.7K prompt, swap-watched): AR 17.6 tok/s,
bs3 22.8 tok/s — **1.29×** (acceptance 65.5%). The speculative ratio holds up
as decode enters the KV-stream regime.

`--bench-context-mult 11` (65K prompt): AR run0 12.7 tok/s, but swap climbed
to 13.5/14.3 GB and AR run1 decayed to 5.9 — process killed to protect the
machine (per the 2026-08-19 crash discipline). The 65K+ regime exceeds a
48 GB machine's envelope for this stack; needs more RAM or KV paging. Open
follow-up, not a port defect.

### K7 — why qmv_wide stops scaling: x-side ALU, not W decode — diagnostics

Probing the ported kernel with pieces ablated (probe-only, all reverted):
removing the per-row x work entirely (kill-x, wrong results) flattens M ≤ 8 to
~1.05 streams (lm_head M=8: 4.93 → 2.30 ms); replacing the 4-bit decode with a
cast (kill-deq) changes nothing. So the M-scaling cost is the per-element
x load + convert + FMA chain, which grows with the row tile — exactly what a
simdgroup-MMA kernel would erase. Two vectorized-x-load variants (uint4
bit-unpack; native `vec<T,8>`) both measured *slower* than stock (lm_head M=8
6.46 / 5.16 vs 4.93 ms) — the scalar 2-byte loads already coalesce; the added
ALU hurts. **Verdict: a real win here needs an MMA-based small-M affine kernel
(bm=8 tiles), i.e. new kernel engineering — logged as future work, not
attempted this session.** NAX (`qmm_nax`) is gated to GPU gen ≥ 17 + macOS
26.2, so it never runs on this M3 Max; the M5 Max in the reference post runs
it, which is part of why their small-M numbers beat this chip's.

## Session 2026-08-21 — DFlash2 round 2: mma8 verify kernel, single-sync pipeline, adaptive width (issue #441)

Context: ADR-0057 shipped blockSize=3 because verify re-streamed weights per
5-row qmv_wide tile (bs8 = 2.4 streams, net-negative). The reference stacks
(oMLX, mlx-dspark) run width 8 behind an MMA-tile kernel that is flat in M.

### R1 — affine_qmm_mma8 — ACCEPTED (mlx 756fd8f7 + 691e42a1, mlx-swift 9f48e3f)

One 8x8 simdgroup_matrix tile per 8 output columns; weight group read+dequant
once per threadgroup, reused across all M<=8 rows; split-K over 8 simdgroups;
register-prefetch pipeline hides streaming loads behind the MMA chain. Port of
avlp12's qmm_mma4 (MIT) via jundot/omlx small_m_qmm.py; A tile read from
device with per-lane fragment row-clamping (Metal 8x8 fragment layout per
conv.metal winograd), 10 KB threadgroup memory. Gate: affine, gs64, 4/8-bit,
N>=4096, K%512==0, non-batched, bf16, M in [5,8].

qmvprobe (M3 Max; numerics rel-Linf 0.008 at every M, same class as qmv_wide):

| shape | M=8 qmv_wide | M=8 mma8 | x |
| --- | --- | --- | --- |
| gdn_in_proj 5120x16384 | 0.564 ms | 0.381 ms | 1.48 |
| mlp_gate_up 5120x34816 | 0.965 | 0.586 | 1.65 |
| mlp_down 17408x5120 | 0.634 | 0.404 | 1.57 |
| lm_head 5120x248320 | 5.422 | 2.940 | 1.84 |

Flat across M in {5,6,8}; mma8 at M=5 beats qmv_wide at M=5 on every shape
(gate lowered to [5,8] after measuring the crossover). REJECTED variants:
qmv_wide tile cap 8 (register spill, 0.5-1.7x slower than mma8 at M=8);
128-K staging chunk (18 KB threadgroup, occupancy loss, slower than prefetch).

### R2 — single-sync pipeline — ACCEPTED (mlx-swift-lm 53e52f8)

Anchor tracked host-side + draft ids and target argmax ship in ONE packed D2H
per round (3 syncs -> 1 on the greedy path the bench measures; sampled rounds
still pay the sampler's own reads). bs3: verify 61.1 -> 54.4 ms, accept 0.9 -> 0.5,
reconcile 1.0 -> 0.7. bs3 31.3 -> 33.3 tok/s cool-machine ABBA (1.58x over AR
21.1). Unit gates: 13 DFlash2Tests incl. iterator rollback test.

### R3 — adaptive width bandit — ACCEPTED

blockSize is now a cap (app ships 8); tok/s bandit over {3, 4, cap}, 8-round
windows, 3% hysteresis, drift re-sweep every 24 windows. The reference's
acceptance-threshold policy can't express this stack's per-width cost curve
(verify near-flat 5..8 via mma8; the draft pass prices width) — the objective
is measured decode tok/s, not acceptance. Mock-harness tests: narrows to the
floor on zero-acceptance content; adaptiveWidth:false never adapts.

### R4 — verify-pass fat attribution (verifyprobe, real target, 2K ctx)

Capture-state forward: S=1 47-48 ms, S=8 82-90 ms (+35 where flat is ~+4).
Decomposition: ~2/3 mma8 bandwidth (75-80% of M=1 GEMV per byte — MMA tile
staging/barrier economics); ~1/3 eager inter-segment glue (16 attention
rope+KV+SDPA steps outside the compiled segments); SDPA vector->fallback at
S>=6 (gqa=6: S*gqa>32) measured <= 0.3 ms/layer — real but small; GDN
recurrence negligible (state in registers across the T loop).

### R5 — benches + gates

Cool machine ABBA, 6K docs prompt, 192 tokens: AR 21.1 | bs3 33.3 (1.58x) |
bs8f 23.8 (1.13x) | bs8-adaptive 23.1 (1.09x). Validation of the shipping
tok/s bandit (cap 8): AR 21.5 | bs3 31.4 (1.46x) | bs8f 25.2 (1.17x) |
bs8-adaptive 28.7 (1.34x) — bandit settles mid-width here and beats fixed-8
by 14%. Output-identity gate (first-8 fingerprint vs AR arm, new in the bench
runner): MATCH on all arms in both runs. 65K+
context remains beyond 48 GB (rule :33). Thermal note: repeated runs within
the hour degrade all arms ~20% mid-run (AR 22 -> 17); numbers above are the
cool-machine set — treat warm reruns as drift, not regression.

### R6 — SandboxMigration harness skip

Headless harness launches (--benchmark, --dflash2-bench, ...) now skip the
models migration (same rule as the test-run guard): a wedged retired-container
directory hung every bench launch post-reboot (contentsOfDirectory blocked on
the main thread). Interactive launches still migrate.

### R7 — easy-content regime (repeat prompt) — the width optimum flips

`DFLASH2_BENCH_PROMPT=repeat` (tiled fibonacci one-liner, 9099-token prompt,
"rewrite iteratively" question) — the agent-typical high-acceptance regime.
192 tokens, ABBA, machine cooling after the docs-prompt set (AR legs read
21.1 -> 18.6/19.2/19.9, so treat absolute tok/s as warm-drifted; run0 legs
additionally pay page-in):

| arm | run0 | run1 | median | acceptance |
| --- | --- | --- | --- | --- |
| ar | 21.1 | 19.9 | 19.9 | — |
| bs3 | 26.5 | 24.4 | 26.5 | 74.0% (228/308) |
| bs8f | 23.9 | 31.3 | 31.3 | 46.8% (294/628) |
| bs8-adaptive | 28.3 | 32.8 | 32.8 | 54.8% (288/526) |

Output-identity: MATCH on all arms.

Findings:
1. Acceptance IS content-bound at narrow width: bs3 74% here vs the docs
   prompt's 59% -> 22% decay. Per-position acceptance decays with block
   depth (bs8f 46.8%), as expected — later block positions are harder.
2. The width optimum FLIPS on easy content: bs8-adaptive (32.8) beats bs3
   (26.5) — the reverse of the docs prompt (33.3 vs 28.7). The bandit's job
   is exactly this regime split; on easy content it rides the cap.
3. verify dominates the wide round and scales with BOTH S and context:
   bs3 verify 74-81 ms at 9K ctx (vs 54-58 at 6K docs); bs8f verify
   118-155 ms. Weight-read floor is ~34 ms (13.5 GB @ 400 GB/s) — the gap
   is the optimization surface: mma8 tile economics + eager attention glue
   (R4 decomposition) + a page-in penalty on first runs.
4. Steady-state (run1) round anatomy: bs3 = propose 20.1 + verify 81.2 +
   accept 0.4 + reconcile 0.6 (2.49 tok/round); bs8f = 16.7 + 118.3 +
   0.4 + 0.9 (4.27 tok/round).
5. Ceiling math for this machine: 60 tok/s needs ~3.5 tok/round at <=58 ms
   — verify at ~85% of the bandwidth floor INCLUDING attention, above what
   the AR step itself achieves (73%). ~50 tok/s on easy content is the
   realistic target: verify ~50 ms (AR-step efficiency) + propose ~17 +
   glue ~3 at bs4-ish acceptance. Levers, in order: (a) eager attention
   glue in verify (~25-35 ms at S=3, worse at S=8), (b) propose/verify
   overlap (~15-20 ms), (c) mma8 bandwidth efficiency (28-54% of peak in
   the probe), (d) then width policy retune.

### R8 — verifyprobe at 9K ctx: the verify cost curve is measured, not theorized

verifyprobe (real target, capture-state forward, median of 30) at 9000-token
prefill, machine warm-ish:

| S | no-capture ms | capture ms |
| --- | --- | --- |
| 1 | 47.72 | 50.88 |
| 2 | 52.81 | 51.05 |
| 3 | 60.21 | 57.24 |
| 4 | 72.53 | 70.10 |
| 5 | 81.24 | 80.38 |
| 6 | 100.02 | 99.06 |
| 7 | 101.58 | 100.98 |
| 8 | 102.89 | 102.09 |

Decomposition (per-delta attribution, cross-checked against kernel gates):

- S=1 -> 2: +5 — qmv_wide M=2 ramp begins (per-row compute grows, weights
  re-streamed per tile).
- S=2 -> 5: +28.5 — qmv_wide M=3..4 then mma8 at M=5; the small-M QMM
  bandwidth gap is the dominant verify cost at every width (mma8 probe:
  27-54% of the 400 GB/s floor depending on shape).
- S=5 -> 6: **+18.7 — the SDPA fallback cliff** (gqa*S = 36 > 32 gates out
  sdpa_vector_2pass; 16 attention layers x unfused 4-5 dispatch chain).
  ~10x agent-7's source-based estimate; measured beats derived.
- S=6 -> 8: +2.9 total — mma8 is M-flat as designed; residual is attention
  growth. Confirms width 6..8 costs nothing extra in the QMMs once the
  cliff is gone.
- Capture plumbing: +0..3 ms per forward (hidden-state + GDN capture).

The 9K-vs-2K S=8 delta (82-90 -> 102) is mostly the cliff scaling with kL
(fallback matmuls/partials grow with context), not KV reads (~1.5 ms).

Lever ranking from measurement: (A) kill the S>=6 SDPA fallback (tile the
qL axis in sdpa_vector_2pass) ~19 ms at S=6-8; (B) small-M QMM bandwidth
(mma8 double-buffer/wider tile; qmv_wide M=2..4 ramp) ~20-25 ms at all
widths; (C) rope-in-trace glue ~1-3 ms; (D) propose overlap ~5-15 ms of
round time. Projection after A+B at bs8: verify ~62 ms -> ~50+ tok/s easy
content; bs3 docs verify ~40 ms -> ~39 tok/s.

### R9 — SDPA qL-tiling: the S>=6 cliff is dead — ACCEPTED (probe-validated)

Change: `sdpa_vector_2pass_1` tiles the query axis — threadgroup z capped at
32/gqa, balanced qL chunks packed into grid.y with the batch; full q_seq_len
rides buffer 19 for chunk-independent causal alignment/output stride; tail
threads early-return. Gate in `use_fallback` relaxes `qL*gqa <= 32` only
where the 2-pass path will actually serve (gqa <= 32, kL >= 1024 on d/s).
Fork-checkout edit, AOT metallib rebuilt via `xcrun metal` CLI for the
probe loop.

Numerics (qmvprobe sdpacheck, tiled-2pass vs eager fp32 reference, causal,
ctx 9216): max|diff| <= 4.6e-4 for S = 1..8 — PASS.

Timing (qmvprobe sdpa, ctx 9216, per-call): S=6 0.968 / S=7 0.869 /
S=8 0.916 ms — the fallback chain (was ~2.2 ms/call at S=6 by the R8
delta) is gone; wide S now costs like S=5.

Model level (verifyprobe, 9K ctx, capture=yes, median of 30):
S=5 77.7 | S=6 80.8 | S=7 84.1 | S=8 86.6 ms. The S=5->6 step is +3.2 ms
(was +18.7). S=8 verify: 102.1 -> 86.6 ms (-15%).

### R10 — mma8 direct-fragment variant — ACCEPTED (modest but real)

Restructure: each lane dequantizes exactly its two 8x8-B-fragment elements
(no threadgroup staging, no barriers, uint4 weight loads, packed A
preloads); 4-bit only, 8-bit keeps the staged path. Numerics: rel-Linf
0.007-0.011 vs per-row reference (staged class). Probe M=8 vs staged mma8:
gdn_in_proj 0.381 -> 0.364 (+4.5%), mlp_gate_up 0.586 -> 0.533 (+10%),
mlp_down 0.404 -> 0.378 (+6.9%), lm_head 2.940 -> 2.911 (+1%). Per-byte
efficiency at M=8 is 73-85% of the M=1 GEMV rate on the same shapes.
Verdict: keep (strictly better per shape, numerics identical), but the
verify excess is NOT per-byte QMM economics alone — the residual S-scaling
after R9 lives in the eager attention glue/dispatch path and needs a real
trace, not more probe arithmetic.

### R11 — where the remaining round time lives (estimate, to be traced)

Post-R9/R10 projection for the repeat-prompt bs8f arm: verify 118 -> ~90 ms
in-bench, round ~110 ms, ~4.3 tok/round -> ~39-42 tok/s cool. Short of 50.
Two residual pools, both need measurement beyond probe arithmetic:
(a) verify's eager attention glue + dispatch at S=8 (~25 ms unaccounted by
    QMM per-byte rates; sdpa 2-pass reduction traffic + rope/KV launches);
(b) propose (draft) 15-22 ms/round for a 5-layer drafter — ~10x its weight
    bandwidth (lm_head over 248k vocab ~3 ms + selector + per-layer launch
    overhead). Candidate fixes: rope-in-trace (array-offset RoPE overloads
    exist), draft-pass dispatch consolidation, propose/verify host-gap
    overlap. Decision gate: if the A+B bench lands < 45 easy-content, spend
    a Metal trace / eager-vs-compiled A/B on (a) before more kernel work.

### R12 — parity-test re-baseline: the token-10 divergence is a DEAD TIE

`testDFlash2EndToEndParityWithPythonTrace{,4BitDraft}` were failing at
commonPrefix 10 (< the historical 40 gate). Stash A/B proved it pre-existing
(identical produced sequence with and without the session's Swift changes) —
the move from the documented token-45 divergence to token 10 dates to the
round-2 kernel set (mma8 verify-width QMM; the draft's lm_head rides it at
M=7). Seam probe (verifyprobe `seam` mode, real target, shared-prefix
forward): the two candidates at the seam are bit-tied — ids 279 vs 1092,
logits 20.7500 == 20.7500. Sub-ULP argmax order picks the winner; both are
legitimate greedy outputs. The test now gates on the actual invariant: a
sanity prefix floor (>= 8) plus a seam gate (the two candidates' target
logits within 0.25 = 2 bf16 ULPs at that scale), keeping count equality and
the acceptance floor (passed: 35/70, 35/68). In-stack exactness (AR vs
speculative on the same kernels) remains gated by the bench runner's
output-identity fingerprint.

Also from the test runs: parallel `swift test --filter DFlash2` on one GPU
hits Metal command-buffer watchdog timeouts (two 27B parity tests
concurrently) — run serially (`--no-parallel`).

### R13 — draft-side round (buffered context cache + mask memo) — ACCEPTED for health, bench-neutral

Two changes: (1) DFlash2ContextCache is now a padded slice-updated buffer
with lazy front-trim compaction (one copy per ~256 rows) instead of a
full-cache concat every round — kills the allocator churn oMLX documented
as a progressive ~5x wall at ~7000 cycles; (2) the sliding-attention mask
is memoized across the all-sliding layer stack per round. Attention-visible
semantics identical (distance mask windows exactly); unit coverage updated
(trim timing is an implementation detail) + a new compaction test; 18/18
serial. Bench (repeat prompt, warm): bs8f 34.2 / bs8-adaptive 34.2 — flat
vs R11's 35.2/33.6 within thermal noise. The per-round concat copies were
~0.2-0.5 ms of bandwidth, not the propose bottleneck; the win is
long-session heap health, not bench tok/s. The 18-22 ms propose phase is
the eager draft forward's dispatch mass — needs the sub-phase decomposition
(instrumentation now in dflashPropose under DFLASH2_PROFILE=1) and likely
a compiled layer stack.

### R14 — canonical cooled set (repeat prompt, no profile drains)

20-min GPU cool-down, then ABBA, production-faithful (DFLASH2_PROFILE unset
— no per-phase drain overhead). Repeat prompt (9.1K): AR 20.6 | bs3 29.1
(1.42x) | bs8f 37.6 (1.83x) | bs8-adaptive 36.7 (1.78x). Identity MATCH all
arms. Session delta on easy content: 32.8 -> 37.6 tok/s (+15%); vs plain AR:
1.83x.

Docs-prompt anomaly (investigated below): bs3 landed 28.6 (1.32x) vs the
round-2 canonical 33.3 (1.58x) at identical acceptance (59.1%) and cool AR
(21.6). verifyprobe isolates the raw S=3 forward at ~59 ms (6K, capture) —
within the historical 57-60 band — so the regression is per-round plumbing,
not the target kernels. Bisect (vendor cache+mask stashed vs not) decides
which side.

### R15 — the docs-prompt "regression" was environment drift — verdict: NO REGRESSION

The canonical docs bs3 landed at 28.6 (1.32x) vs R5's 33.3 (1.58x) at
identical acceptance and cool AR. Controlled bisects in this session's
environment: (a) new kernels + old vendor (53e52f8): bs3 25.9-26.8; (b) new
kernels + new vendor: 25.8-28.6; (c) EXACT R5 config (kernels 9f48e3f +
vendor 53e52f8): bs3 22.2-23.1, verify 71.5-74.6 — the slowest of the
three. R5's 33.3 is not reproducible tonight on ANY configuration; the
machine's thermal/power envelope drifted over the day (the in-run AR decay
pattern is stronger than this morning). New kernels are equal-or-faster
than old on identical vendor code. Canonical floor for tonight's
environment: repeat 37.6 (1.83x), docs bs3 ~26-28.6 (1.32x). Lesson
recorded: cross-session absolute comparisons need the same-day cool
baseline; the ABBA discipline holds within a run only.

## Session 2026-08-22 — DFlash2 round 4: SDPA multi-query kernel, lever pricing

Context: continuing toward the 60 tok/s target from R14's canonical 37.6
(bs8f, repeat prompt, cooled). This session's method: price every lever
with a cold-stream probe BEFORE building it. Two of the three planned
levers died at the pricing stage; the one that shipped is a kernel.

### R16 — SDPA multi-query kernel (mq, QPS=2) — ACCEPTED; the KV-restream premise measured down

Four-design bracket on `sdpa_vector_2pass_1` at the verify shape (D=256,
gqa=6, ctx 9216; cold-stream 16-layer probe, ms per 16-layer pass at S=8):

| design | S=8 ms/pass | verdict |
| --- | --- | --- |
| original (one query per simdgroup) | 10.88 | baseline |
| mq QPS=4 (4 queries per simdgroup) | 26.14 | REFUTED — register collapse |
| mq QPS=2 | 9.62–9.78 | ACCEPTED |
| mq QPS=2 + blocks x2 | 11.78 | REFUTED |
| coop (threadgroup K/V staging, TK=8) | 19.54 | REFUTED — barriers, same L1 |

Finding: the per-query KV re-streams that round 3 priced as an ~8 ms
lever were already L1/SLC-served — the original kernel's effective
bandwidth at S=8 is ~2.6 TB/s, far above DRAM; this kernel family is
instruction-issue/latency-bound, not bandwidth-bound. QPS=4 quadruples
per-thread register state (~90 floats) and serializes four query-updates
per key: occupancy collapse. Threadgroup staging routes through the same
L1 hardware Apple maps threadgroup memory onto, and adds barriers —
strictly worse. QPS=2 halves K/V access issue count at tolerable register
cost: SDPA pass 10.88 -> 9.6-9.8 ms (-10%). Numerics: per-query
accumulation order identical to the single-query kernel by construction;
`qmvprobe sdpacheck` worst 3.0e-4 (R9's class); qL=1 (AR arm) untouched.
Host routes qL >= 2 to `sdpa_vector_2pass_1_mq_`; z-extent groups QPS
queries per slice; instantiated for all (3 dtypes x 4 head dims). Vendor
DFlash2 suite 19/19 serial on the new kernels; bench identity MATCH.

### R17 — lever pricing: projection fusion REFUTED, verify-compile already engaged

(a) Projection fusion (the round-3 "lever B", 497 QMM dispatches/pass)
priced BEFORE building: cold-stream fused-vs-separate groups
(mlp gate+up x64, gdn qkv+z+b+a x48, fa qgate+k+v x16 — row-concat along
N is bitwise-lossless per row) come to **0.47 ms/pass at M=8, 0.33 at
M=1**. The QMM dispatch + tile economics at these shapes are already
efficient; fusion is dead. The sanitize/module surgery was skipped.

(b) The committed verify-compile path (`verifyStep` /
`compiledVerifySegments`, escape hatch `DFLASH2_VERIFY=eager`) ENGAGES
and is worth 10.8 ms at S=8/9216/capture: verifyprobe 76.7 ms compiled
vs 87.5 eager. The "eager attention glue" pool R11 flagged is already
harvested; the residual ~10 ms of non-QMM/SDPA time is 16 eager segment
boundaries (rope, KV append, SDPA) + compiled small-kernel floor +
capture copies.

### R18 — warm bench: bs8f 43.5 tok/s, MATCH; round anatomy

Repeat prompt 9.1K, 192 new, runner ABBA, warm machine with healthy AR
legs (20.2-21.6 vs R14's cool 20.6). bs8f 43.5 / 41.1 unprofiled
(median 43.5), identity MATCH, speedup 2.09x. Profiled round: propose
14.3 (draft-hidden 8.0 + logits 3.3 + select 1.4) + verify 86.0
(drain-inflated; verifyprobe puts the true pass at ~77) + accept 0.4 +
reconcile 0.7 — 45 rounds, 4.27 tok/round. The uplift over R14 exceeds
what mq alone predicts; R19's cooled set is the honest cross-session
number (R15's environment-drift lesson applies).

Ceiling restated from this session's measurements: pools left inside the
S=8 chain-verify architecture are the QMM M=8 gap (54.4 vs 41.9 ms at
M=1 — mma8 tile economics), ~7 ms of SDPA above the S=1 floor
(issue-bound; the R16 bracket exhausted the vector-kernel family), and
~10 ms post-compile glue. Full recovery of all three lands ~53 tok/s.
**60 tok/s requires more tokens per round** — tree verification (bool-
mask tree at S=8, or S=16 which first needs an mma16-class QMM kernel;
M=16 today is a 148 ms/pass kernel hole) — or propose/verify overlap.

### R19 — canonical cooled set (repeat prompt, no profile drains)

20-min GPU cool-down, runner ABBA, production-faithful. AR legs 20.2-20.9
(median 20.7) — same thermal envelope as R14's 20.6, so the cross-session
comparison is honest:

| arm | R14 | R19 | speedup | acceptance |
| --- | --- | --- | --- | --- |
| ar | 20.6 | 20.7 | — | — |
| bs3f | 29.1 | 29.4 | 1.42x | 74.0% (228/308) |
| bs8f | 37.6 | **40.0** | **1.93x** | 46.8% (294/628) |
| bs8 adaptive | 36.7 | 38.5 | 1.86x | 53.0% (286/540) |

Output-identity: MATCH on all arms. Session delta on easy content:
bs8f +2.4 tok/s (+6.4%) at identical AR — the R18 warm 43.5 was
thermal-flattered (R15's lesson holds; warm runs overstate). Round math
closes: 40.0 tok/s = 106.8 ms/round vs R14's 113.6, consistent with the
mq kernel's -1.1 ms verify + draft-side SDPA savings. Acceptance
unchanged per arm (identical tokens — the identity gate at work).

### R20 — acceptance anatomy via zero-perturbation accept-log

Env-gated instrumentation (`DFLASH2_ACCEPT_LOG`, candidates ride the
existing packed D2H — no added evals): accept histogram over 45 rounds is
bimodal 0:4 1:14 2:7 3:2 4:2 5:2 6:3 7:11 — rounds either die in the
first two positions or run the chain out. cover16 0.84-1.00, cover2
0.73-0.80; at the first miss the target token was in the candidate set
29/33 times, rank r1:15 r2:6 (64% within top-2). Host timeline: 98.2
ms/round = GPU sync 92.6 + propose-build 3.5 + verify-build 1.6 +
reconcile 0.5 + accept 0.03 — the round is GPU-bound; host trimming is
a ~5 ms pool, not the prize.

### R21 — tokens-per-round levers re-litigated: all REFUTED

(a) Out-of-width chains: width 10 -> 4.09 tok/round, width 12 -> 4.47
vs 4.27 at width 8 — the block degrades holistically; no free tokens.
(b) S>8 verify: mma16 fixes the M in [9,16] QMM hole (85.8 ms/pass vs
148) but rows 9-16 still cost +31 ms/round; breakeven needs +1.65
tok/round and tree shapes cap at ~+0.9. Single-sibling conversion nets
only +1 because the bonus token already delivers the correction.
(c) Selector walk policies (offline replay over dumped lattices, exact
first-divergence rule): Viterbi/beam/lookahead/softmax all <= greedy —
the drafter's scores are miscalibrated deep in the block; oracle-gap is
real but unreachable from these scores.
(d) Online n-gram forced selection (order>=4, share>=0.6): acceptance
46.8% -> 45.5%. Mechanism: phase-shift absorption — converting an early
death shifts the round boundary and the deep chain dies earlier;
per-position wins don't compose across round phases.
The S=8 chain stays at ~4.3 tok/round; 60 tok/s is a round-time program
(target ~76 ms/round from R20's 98.2).

### R22 — banked: mma16 kernel, mma8 N-gate 2048, compiled selector walk

mma16 (two stacked A fragments per B fragment, 4-bit direct-fragment)
closes the M in [9,16] dispatch hole: census full-pop M=16 148 -> 85.8
ms/pass, numerics <= 0.0107 (same class as mma8). mma8 N-gate lowered
4096 -> 2048 (draft_kv 5120x2048 M=8 was 1.14x M=1 on qmv_wide). The
propose selector walk compiled per width (compiledGreedySelects).
Production bench with all three: identity MATCH, acceptance byte-stable
46.8% (throttled machine, 33.2 median — non-canonical; AR leg sagged to
16.8).

### R23 — rope-in-trace: eager attention boundary halved

Rope moved inside the compiled decode/verify segments — the offset rides
in as a `[1]` array trace input (`mlx_fast_rope_dynamic`, the drafter's
own device since round 1); the eager boundary between segments is now
just KV-append + SDPA. Same-thermal A/B (AR 20.1 vs 20.0): bs8f 33.2 ->
38.1 median, identity MATCH, acceptance identical. 1.90x.

### R24 — mma8n16: 16-wide-N tile, the M=8 QMM gap halved

One threadgroup covers two adjacent 8-column tiles sharing the A
fragments (`affine_qmm_mma8n16`, 4-bit direct-fragment, env
`MLX_QMM_MMA8_N16`): twice the loads in flight per lane in a
latency-bound family, half the threadgroups and per-output A traffic.
Census: full-pop M=8 59.59 -> 54.68 ms/pass, S-delta (M=8 over M=1)
18.97 -> 13.85; every routed shape improved (mlp_gate|up -12.5%,
mlp_down -6.8%, gdn_qkv -5.5%, fa_qgate -8.5%), M=1 untouched. Numerics
0.0078-0.0087 (mma8 class). Live: bs8f 43.7 median, identity MATCH, AR
21.1, 2.07x. (Run0 7.6s was the one-time JIT compile of the grown
quantized library — quantized is JIT in the app; SDPA is AOT.)

### R25 — SDPA gqa-packed MMA pass 1: 9.84 -> 6.37 ms/pass in-probe

First, the ceiling: raw cold-stream of the exact verify KV arrays
(`kvstream`) runs 247 GB/s — the vector kernels, the mma v1, AND the
eager S=16 GEMM fallback all sit at ~62 GB/s, so the 9.6 ms SDPA pool
was kernel-bound, not bandwidth-bound, with a ~2.5 ms floor.
`sdpa_vector_2pass_1_mma` (env `MLX_SDPA_MMA`): all 48 q-rows sharing a
KV head (6 gqa x 8 qL) in one threadgroup — (32, 2, gqa) = 384 threads,
per-stripe 8-row MMA tiles, D split in halves per simdgroup with the
partial scores exchanged through threadgroup memory, online softmax,
contiguous 32-key blocks over 64 partitions, merged by the unmodified
sdpa_vector_2pass_2. The path that mattered: K/V must be STAGED through
threadgroup memory with 16-byte uint4 lines — transposed (v1: 0.99
ms/layer) or even plain (v5: 0.76) fragment gathers straight from
device, and 2-byte staging loads (v2: 0.615), all leave the stream
issue-bound. v4: **0.398 ms/layer, 6.37 ms/pass vs mq's 9.84**, 1.53x,
numerics PASS (0.00027 vs eager fp32, baseline-class). Refuted en
route: BK=16 (0.66 — more barriers beat the residency win), partition
sweep 32/96/128 (64 best), V register prefetch (0.4156 — the mma8
prefetch lesson repeats). qL == 8 / D == 256 / no-mask-or-causal / no
sinks / 2-byte dtypes only; everything else keeps the vector route.

Live coda — the probe's concurrent regime lies about the serial one:
same-binary same-thermal A/B put mma-at-blocks=64 at ratio parity with
the mq kernel (probe promised -3.4 ms/pass). The probe times 16
INDEPENDENT ops that overlap; the live verify runs its 16 SDPAs
serially between dependent segments, so each kernel must fill the
machine alone and 256 threadgroups can't (mq launches 2048). At
blocks=256 (1024 threadgroups): bs8f 43.4 @ AR 21.5 vs 42.6 @ 21.4
with the kernel off — +0.8 tok/s live, identity MATCH. 256 is now the
kernel's default; MLX_SDPA_MMA=0 and MLX_QMM_MMA8_N16=0 are the
kill-switches (both kernels default on). Also learned en route:
`DFLASH2_ACCEPT_LOG` now disables the compiled selector walk (the rank
instrumentation needs eager select), so it is no longer
zero-perturbation for timeline readings; and the acceptance PATTERN is
numerics-sensitive (46.8% <-> 45.5% across kernels that reorder
reductions) while output identity holds — the gate is the output, not
the acceptance rate. 16-sg split-K n16 probed and refuted (55.51 vs
54.68 census).

### R26 — canonical cooled set: round time -2.2%, headline eaten by near-tie luck

20-min cool-down, runner ABBA, defaults only (both kernels default-on,
no env). AR legs 20.8-21.6 (median 21.0, vs R19's 20.7 — rope-in-trace
also serves plain decode):

| arm | R19 | R26 | speedup | acceptance |
| --- | --- | --- | --- | --- |
| ar | 20.7 | 21.0 | — | — |
| bs3f | 29.4 | 29.5 | 1.40x | 74.0% (228/308) |
| bs8f | 40.0 | 39.9 | 1.90x | 45.5% (292/642) |
| bs8 adaptive | 38.5 | **40.2** | **1.91x** | 54.8% (288/526) |

Output-identity MATCH on all arms. The bs8f flat is two real effects
cancelling: round time 106.8 -> 104.5 ms (the kernel program's -2.2%),
tokens/round 4.27 -> 4.17 (the SDPA-MMA reduction order flips near-tie
drafts; this prompt drew one extra round per run). Adaptive drew the
flips the other way and is now the best arm. The near-tie direction is
prompt luck, not a quality signal — the identity gate holds either way.

Honest ceiling after this session: the S=8 chain at ~104.5 ms/round and
content-capped ~4.3 tok/round sits at 40-43.5 (cooled-warm). The
remaining measured pools (QMM S-delta 13.9, SDPA ~3.9 above stream
floor, draft-hidden 8, glue) total ~53 tok/s at FULL recovery — R18's
statement stands. 60 tok/s on this hardware needs tokens/round, and
R20-R21 closed every selector-side route: what remains is drafter
QUALITY (deeper/better-calibrated draft heads — a training project),
not inference engineering.

### R27 — round 5b: draft decomposition, M-gate crossover re-measured, cross-stack survey

Continuation of the 60 tok/s directive after R26. Three measurements, all
narrowing the remaining space; no canonical change.

**Draft pool decomposed** (DFLASH2_PROFILE=1, 8f arm, warm; eval-synced —
absolute values inflated, the split is the signal): propose =
**draft-hidden ~10.5 ms** + draft-logits ~2.9 + draft-select ~1.8
(profiled round: propose 27.8, verify 96.1, accept 0.7, reconcile 3.7;
46 rounds; identity MATCH, acceptance byte-stable 146/321). The
compiled draft route is engaged (`draft route: compiled` announced).
draft-logits is already near its lm_head stream floor (~2.0 ms).
draft-hidden's floor: ~850 MB draft weights ~2.7 ms + tiny 2048-window
SDPA + mma economics ≈ 4.5-5 ms — recoverable ≤~3 ms, spread across 5
layers of compiled segments + 10 eager KV/SDPA boundaries.

**mma8 M-gate floor re-measured post-n16** (new `qmvprobe mcross`, big
shapes, cold-stream, M∈[2,5]; route via new `MLX_QMM_MMA8_MMIN` env):
qmv_wide total/pass M=2 40.05, M=3 41.84, M=4 52.10 vs mma8n16 flat
~53.2-54.3. Verdict: the pre-n16 crossover STANDS — qmv_wide wins
M≤4 (nearly M-flat through M=3 at ~1.03× the M=1 stream), mma from
M=5. The adaptive arm's narrow rounds already ride the best kernel;
gate-widening REFUTED. (qmv_wide has a step at M=4: 41.8 → 52.1.)

**Cross-stack survey** (dflash-mlx by bstnxbt, mlx-dspark by ARahim3 —
the reference stacks behind ADR-0058's context): dflash-mlx's
`verify_qmm` family is BM=16/BN=16-32 threadgroup-staged simdgroup-MMA
with split-K — one generation BEHIND mma8n16 (no direct-fragment A, no
shared-A wide-N); mlx-dspark's small_m_qmm is our mma8 shape (8
cols/tg, split-K, tg staging) and its "flat in M" is literal padding of
M<8 to 8. Neither carries a QMM technique we lack. Their headline
numbers decompose into content + hardware: dflash-mlx Qwen3.5-27B-4bit
at 8192 tok on M5 Max (614 GB/s + NAX) = 45.29 tok/s ≈ **29.5
bandwidth-normalized to our 400 GB/s — we are AHEAD at 40.2**. Their
acceptance 84-90% is short MATH-prompt content; official DFlash2 card:
acceptance length 4.10 (MT-Bench) - 5.46 (GSM8K) — our repeat-prompt
4.17-4.27 is AT the checkpoint's spec, no selector headroom lost. Also:
dflash-mlx explicitly disclaims strict AR output identity; our gate is
tighter. No better public drafter exists for this target (DSpark mean
acceptance 3.39 < DFlash2).

**S=16 two-chain tree re-priced under a hypothetical mma16n16** (R21
used the banked mma16's +31 ms): break-even Δ = 104.5×(5.07/4.17−1) ≈
22.5 ms; full bill ≈ QMM +5-8 (mma16n16, unbuilt) + SDPA ×2 +6.4 +
GDN ×2 +4 + KV-fixup/second-walk/glue +2.5 ≈ **+18-21 ms for ≤+0.9
tok/round (R21's offline bound)** → ≤+0.8 tok/s at the optimistic edge
of every estimate. Wash; tree stays dead on this content even with a
better M=16 kernel. Portable dflash-mlx ideas still open: innovation-
tape reconcile (~−1 ms vs our full delta-rule replay), priced small.

### R28 — verify sub-phase decomposition: the round is fully attributed

New `DFLASH2_VERIFY_PROFILE=1` gate (Qwen35.swift): eval-synced split of
the verify pass into compiled segments vs eager attention boundaries
(16 forced drains inflate the absolutes; the split is the signal). 8f
arm, repeat prompt, warm; identity MATCH, acceptance byte-stable.

verify-seg ~85-90 ms profiled, verify-attn ~17-18 ms profiled.
Deflating by the drain overhead (unprofiled verify ≈ 85): segments
≈ 70-75, boundaries ≈ 10-12. Inside segments: QMM 54.7 (census) + GDN
scan ~4 (state traffic ≈ 1.2 GB/round) leaves **~12-16 ms of in-segment
glue** — norm/rope/elementwise kernels the MLX compile cannot fuse
further (hundreds of ~10-20 µs kernels over [8, 5120] tensors).
Boundaries ≈ SDPA 6.4 + KV slice-appends + dispatch.

Full round attribution (unprofiled, ~104.5 ms): QMM 54.7 | in-segment
glue+GDN 16-20 | attention boundaries 10-12 | propose 13-15 (hidden
8-10.5, logits 2.9 ≈ floor, select 1.8) | host+accept+reconcile ~7.
Raw unavoidable memory traffic per round ≈ 50 ms (QMM stream 40.7 +
SDPA 2.5 + draft 2.7 + GDN state ~4). 60 tok/s at 4.17 tok/round means
71.2 ms/round = 1.43× that traffic floor with zero glue/host margin —
beyond realistic MLX-dispatch engineering (~84-90 ms is the everything-
lands estimate → 46-50; everything-perfect ≈ 53). The same stack on
M5-class bandwidth (614 GB/s) scales to ~62 on this very bench — the
target is a hardware generation away, not a kernel away.

### R29 — SDPA narrow-qL padded mma: built, measured, gated to [7,8]

Built the qL<8 extension of `sdpa_vector_2pass_1_mma`: host pads narrow
query tiles into a zeroed contiguous 8-row buffer (fill_gpu + strided
copy_gpu_inplace, conv.cpp's pattern) and the kernel writeout switched
from simdgroup_store to per-sg threadgroup staging + row-guarded scalar
stores with real-qL strides (bitwise-identical at qL=8; sums/maxs
guarded). Parity (`sdpacheck` 9216): S=1..8 all 0.0002-0.0003 vs fp32
eager — same class as the S=8 kernel.

Bug worth remembering: the pad copy was first inserted AFTER
`set_compute_pipeline_state` — `copy_gpu_inplace` re-binds the
encoder's pipeline, so the SDPA dispatch launched the COPY kernel with
the SDPA grid (clean-zero outputs, all-masked-row symptom). Any helper
that dispatches through the stream's encoder must fully precede
pipeline binding.

Cold A/B (`sdpacold` 9216, per pass): padded-mma 6.93/7.24/7.50/7.63 at
S=2/4/5/6 vs old vector path 3.15/5.40/7.11/7.69 — the padded kernel
does full 8-row work, so the old path WINS at S<=4 (the adaptive
bandit's actual widths) and ties at 5-6. Gate set to qL in [7,8];
production behavior at the bandit's {3,4} widths unchanged by design.
Identity re-gate (all arms, warm): 3f 29.1 / 8f 39.3 / adaptive 39.9,
acceptance byte-identical to R26, output-identity MATCH everywhere —
the writeout restructure is production-safe.

Also priced this iteration, both REFUTED:
- Tape-replay reconcile (dflash-mlx's innovation tape): our reconcile's
  cost is dominated by the state read+write stream (~302 MB/round) plus
  48 dispatches, which the tape variant pays identically — it only
  skips the q-dot and delta recompute, ~0.3-0.5 ms, against iterator
  surgery and a bitwise-replay matching risk. Dead.
- GDN-scan fusion: already fused. The verify traces route to the
  single-dispatch `gated_delta_step` kernel (t-loop in-kernel, state in
  registers, Kahan) — Dk=128 %32==0 confirmed from the model config; 48
  launches and ~0.9 ms/round. The 12-16 ms in-segment glue is the
  norm/rope/elementwise spine (~800 tiny kernels), not the scan;
  recovery would need custom micro-fusion of the layer spine — weeks-
  class, identity-risk per op, not a session lever.

Canonical record stands at R26: bs8-adaptive 40.2 (1.91x).

### R30 — the drafter path priced: b16 training scoped, does not reach 60 on M3 Max

The last standing lever (drafter capacity) converted from a hand-wave
into numbers. z-lab's DFlash training code is public; the paper's own
block-size ablation gives b8 -> b16 = tau 5.21 -> 6.33 (+21%, Math500),
and b16 drafters generalize down to narrower inference widths. **No b16
checkpoint exists for Qwen3.8-27B** (only Qwen3-4B/8B) — it would have
to be trained.

Costs computed from the paper recipe (800K target-generated samples,
6 epochs, seq 3072, ~1.8e20 training FLOPs): on-device M3 Max full-spec
~9 months (dead); cloud 8xH100 ~2 days / ~$600-1,200 (feasible).
Honest EV on the canonical bench: content-discounted tau ~5.1 at S=16,
round 104.5 + (mma16n16 +5-8, SDPA qL16 +2-4, 16-mask propose +3-4)
~ 118 ms -> **~43-47 tok/s. The b16 drafter does NOT reach 60 on M3
Max either.** 60-crossings: today's stack on M5-class bandwidth (~62),
or b16 on M4-class (~59-64). Main technical risk: public code is
DFlash v1 — the shipped checkpoint's DFlash2 additions (selector
lattice, dynamic conv) are not in it.

Full scoping doc: research/dflash2-drafter-b16-scope.md. With this,
every route to 60 on this hardware is measured, built, or priced:
none lands. The program's terminal state on M3 Max is ~40-42 today,
~45-47 with b16 + S=16 kernels, 60+ only with the next hardware
generation.

### R31 — mma8n32 (32-wide-N tile) built and REFUTED: occupancy collapse, -58%

The one kernel idea previously dismissed without measurement ("register
pressure risk") is now measured. Built `affine_qmm_mma8n32`: one
threadgroup -> 32 output columns, four C fragments sharing a single A
fragment per k-tile, 8 weight/scale/bias streams per lane, split-K via
`red[8*256]` threadgroup reduction; JIT-mirrored byte-identical
(quantized.h <-> mlx-generated/quantized.cpp), instantiated for
float16/bfloat16 gs=64 bits=4, host route opt-in via
`MLX_QMM_MMA8_N32=1` (takes precedence over n16, grid (N+31)/32).

Numeric parity: PASS, rel-Linf 0.0067-0.0103 across M=2..16 — same
class as mma8/n16 (fp16 accumulate).

Census A/B (fresh paired runs, cool machine, logs census-n16-r31.log /
census-n32-r31.log):

| route | M=1 | M=8 | M=16 |
|---|---|---|---|
| n16 (default) | 40.49 | **53.83** | 85.67 |
| n32 (opt-in)  | 40.72 | **85.10** | 85.55 |

n32 loses M=8 by +58%. Diagnosis is unambiguous: n32's M=8 column
equals its own M=16 column shape-by-shape (mlp_gate|up 0.2906 vs
0.2919; lm_head 4.097 vs 4.094) — the 4-fragment + 8-stream register
load halves occupancy, and the amortized A-fragment reloads were never
the cost (these shapes are weight-traffic-bound). M=1 and M=16 columns
match across runs (neither routes to the mma8 family) — internal
consistency check passes. The old dismissal was right; now it is
evidence, not a prior.

Disposition: kernel stays in the working tree as opt-in-off (dead
unless `MLX_QMM_MMA8_N32=1`); recommend stripping it from the JIT
source before the fork commit, same as the refuted env-gated selector
paths. No identity bench needed (never default-on).

With R31, every kernel idea in the program is either shipped or refuted
by measurement — nothing remains dismissed untested. Canonical record
stands at R26: bs8-adaptive 40.2 (1.91x).

### R32 — zero-perturbation host timeline: the "7 ms host pool" was really ~18 ms

New `DFLASH2_HOST_PROFILE=1` (iterator): the accept-log wall-clock
timeline without the candidate instrumentation, so the compiled selector
walk stays engaged — pure clock reads, zero added evals. Proven
non-perturbing: 39.6-40.1 tok/s with it on vs 39.9 canonical. Warm 8f
round decomposed (ms/round, n=46): gap 0.02 | propose 11.7 | build 6.8 |
sync 88.4 | accept 0.04 | reconcile 2.9 — sums to the round wall.

The reattribution: R28's "host+accept+reconcile ~7 ms" (derived by
subtraction) was wrong by 2.5x. `DFLASH2_PROFILE`'s propose 13-15 ms is
NOT GPU draft work — splitting propose into p-splice 3.75 (lazy graph
build) + p-sched 7.9 (asyncEval scheduling/encoding) shows ~11.7 ms of
HOST-side work with the GPU idle; the draft's actual GPU compute is
~3 ms (consistent with the 4-bit drafter's ~1 GB weight stream; the
app already quantizes the drafter at load — DFlash2Support.swift:84 —
so drafter quantization was correctly priced dead). True GPU-idle host
pool ≈ 18 ms/round: propose 11.7 + exposed build ~4 + reconcile ~2.9.

Also priced while here: target re-quantization (the one axis never in
the ledger). QMM 54.7 ms scales with weight bits: 3-bit → round ~89 →
~47 tok/s; 2-bit → ~80 → ~52. NO quant level reaches 60 (needs ~1.75
effective bits, quality fiction at 27B), and re-quantizing changes the
product's target — not the same lossless product. Dead for 60.

### R33 — pipelined round construction: propose leaves the critical path. NEW RECORD 43.4 (2.07x)

The R32 pool attacked at its root. The round's host serial time exists
because propose N+1 waits for accept N; the sync window (~88 ms of
blocked host) exists right before it. Fix: make the propose graph
ACCEPT-INVARIANT so ONE prebuilt graph is correct for every accept
outcome, and build it during the sync window:

- Accept computed lazily on GPU: accepted = sum(cumprod(draft == target
  argmax)), bonus anchor via lazy take — spliced BEFORE eval(packed).
- ALL verify rows (gamma+1) append to STAGING CLONES of the draft
  context caches with explicit host-known RoPE positions (round N's
  verify positions are fixed regardless of acceptance); rows past the
  accept count are excluded by a lazy validity mask. Valid rows'
  K/V are bitwise the synchronous route's (row-wise ops).
- Block ids = [lazy bonus, MASK...]; block RoPE offset = lazy
  anchor+accepted+1. Compiled segment traces are shared with the
  synchronous route (they always took offsets as inputs — rope-in-trace).
- Round order: asyncEval(packed) starts verify on GPU -> prebuild
  propose N+1 (host, overlapped) -> asyncEval(prebuilt) queues the
  draft behind the verify -> eval(packed) -> accept/adopt. On adopt the
  staged caches commit and the appended rows' validity resolves from
  the now-known accept count. Cache compaction became valid-aware
  (placeholders dropped, newest maxSize committed rows kept) — fired
  live every ~32 rounds in the gate runs.
- Bandit width switches take effect one round late (prebuilt width
  governs); measurement modes (accept-log, lattice dump, advised
  selector) and T>0 fall back to the synchronous round.

Timeline after (warm 8f): propose 1.84 (adopt only) | build 7.0 |
prebuild 88.8 (contains the verify wait) | sync 0.01 | reconcile 2.9 —
round 112 -> 100.7 ms warm, the full ~11.5 ms propose pool gone.

**Canonical R33** (cooled ABBA, defaults, identity MATCH all arms):
ar 21.0 | bs3f 31.9 (1.52x, was 29.5) | **bs8f 43.4 (2.07x, was 39.9)**
| bs8-adaptive 41.1 (1.96x, was 40.2). **New record bs8f 43.4 — the
first 2x+, and fixed-8 overtakes adaptive** (width-8 rounds gained the
most, shifting the bandit's trade). Acceptance pattern moved one
near-tie (294/644 vs 292/642) — masked-SDPA reduction order, identity
unaffected. Default ON; kill switch `DFLASH2_PIPELINE=0`.

Remaining in this direction (stage 2, unbuilt): the verify build
(~6.8 ms) and reconcile (~2.9 ms) could pipeline the same way, but
need the TARGET cache rollback made accept-invariant (GDN capture
replay + KV trims as lazy ops) — days-class, projected floor ~88-90 ms
round ≈ 46-47 tok/s. 60 still requires tokens/round (drafter training)
on top: unchanged conclusion, better base.

### R34 — verify-build streaming built + REFUTED; canonical re-run banks 44.0 (2.11x)

The last exposed host pool after R33 is the verify graph build (~7 ms
with the GPU idle at round start). Built incremental scheduling
(`DFLASH2_VERIFY_STREAM=<stride>` in Qwen35 verifyStep): every Nth
attention boundary is asyncEval'd so the GPU chases the host through
the verify graph. Warm probes looked strongly positive (stride 3:
46.7 vs 41.5 "same-thermal") — but a bare same-process A/B refuted it:
**stream OFF 47.7 vs stride-3 45.0 (ar 21.9/21.7)**. Once the
pipelined round exists, command-buffer fragmentation (extra
commit/fence per asyncEval) costs more than the overlap buys. Stride 1
is worst (16 buffers, -9%), stride 3-4 still net-negative. Also
refuted: deferring the prebuilt draft's schedule into the next build
stream (40.8 — the schedule cost lands on the chase path and the
packed sync reopens). Streaming stays in the tree probe-only,
default OFF.

**Measurement lesson (the reason the probes lied):** AR-ratio
normalization is NOT valid across thermal states — AR decode is purely
bandwidth-bound while the speculative round is mixed host/GPU-bound,
so thermal throttling moves the ratio itself. All conclusions must
come from same-process A/Bs or the cooled canonical; the warm probe
sequence (all run with DFLASH2_HOST_PROFILE=1 on an hours-hot machine)
manufactured a +13% phantom.

**Canonical R34b** (cooled ABBA, defaults = pipeline ON / streaming
OFF, identity MATCH all arms): ar 20.9 | bs3f 32.0 | **bs8f 44.0
(2.11x) — NEW RECORD** | bs8-adaptive 41.3 (1.98x). R33 vs R34b
(same effective config) shows canonical run-to-run variance ~±0.5.

Exposed host after R34: adopt ~1 + verify splice ~7 + reconcile ~2.9
(mostly draft-overlapped). The remaining engineered route is stage-2
(accept-invariant TARGET rollback so the verify build itself pipelines)
~ 46-48; beyond that, tokens/round (b16 drafter) or bandwidth.

### R35 — stage-2a accept-invariant reconcile + the MAX_ACTIVE_TASKS discovery

Built stage-2a: the GDN rollback replay is now built as an
accept-invariant masked graph in the sync window
(`prebuildSpeculativeRollback` in DFlash2Support.swift). The fused
`gated_delta_step` kernel's mask branch is an exact identity step
(masked steps leave the state registers untouched), so a full-S replay
with a lazy `pos < validCount` mask equals the accepted-prefix replay
bitwise for every accept outcome; conv state comes from a lazy
`takeAlong` at `validCount + arange(K-1)`. Applied only when
`rejected > 0` (all-accepted rounds keep the verify-committed states,
exactly the synchronous semantics). Default ON,
`DFLASH2_ROLLBACK_PREBUILD=0` kills. Reconcile stamp 2.9 -> 0.27
ms/round; identity MATCH, acceptance bit-identical (147/322 per run).
Standalone effect: ABAB warm A/B = wash (thermal ramp dominates); ABBA
(trend-cancelling) = +2.05 inflated by convex cool-start decay, last
adjacent pair +0.8 (ON 43.3 vs OFF 42.5). Net: neutral-to-slightly
positive standalone — kept ON as the enabling piece for stage-2 (the
next verify graph can consume the reconciled GDN state lazily).

**Where the pipelined round's time actually goes** (new prebuild
sub-stamps, `DFLASH2_HOST_PROFILE=1`): prebuild 76.4 = pb-vsched 63.6
+ pb-graph 4.9 + pb-dsched 7.8. The 63.6 is NOT host work: mlx-core
throttles the encode thread at `MAX_ACTIVE_TASKS = 10` in-flight
command buffers (transforms.cpp:424 wait_for_one after every op past
the cap), so `asyncEval(packed)` paces to the GPU and the wall time
inside it is mostly throttled waiting. The genuinely GPU-idle host
strip per round is the post-sync inter-round gap: verify graph build
~6.0 + prebuilt-adopt ~1.7 + gap/accept/reconcile ~0.7 ≈ **8.4
ms/round of GPU idle**. Round ≈ GPU work (~76 warm) + that strip.

**Stage-2 scoped and de-risked** (the remaining engineered route):
build round N+1's verify graph inside round N's window so the GPU
never drains. All accept-dependent inputs can be lazy: tokens =
prebuilt draft outputs; rope offsets = arrays (rope-in-trace); GDN
initial states = stage-2a's replay arrays; **KV append offset = lazy
via `mlx_slice_update_dynamic`** (in the C API, unwrapped in Swift —
small checkout wrapper, C7 precedent). Lazy-offset writes overwrite
stale rejected rows naturally: no dead-row growth, no compaction,
attention trims become host bookkeeping. Verify SDPA then needs one
lazy bool row mask `arange(worstLen) < off + S` ([1,1,1,kL]): the
sdpa_vector/mq kernels support bool masks via function constants;
our mma kernel currently bails on any mask (falls back to mq,
~+3.4 ms/round) -> add a mask variant (function constants specialize
from the metallib; no new AOT instantiations needed). Projected:
round -> GPU-bound ~86-88 ms cooled ≈ 46-48 tok/s.

### R36 — stage-2 verify prebuild LANDS: bs8f 46.7 (2.34x) NEW RECORD

Built the full stage-2: the NEXT round's verify pass is constructed and
scheduled inside the current round's sync window, entirely from lazy
accept-dependent inputs, so the GPU never drains between rounds. The
machinery: verify tokens = prebuilt draft outputs + lazy bonus anchor;
RoPE offsets as lazy `[1]` arrays (rope-in-trace); GDN initial states =
stage-2a's masked replay (applied unconditionally in the window — the
replay is bitwise the committed state on full acceptance); KV rows
written at the lazy true offset via `mlx_slice_update_dynamic` (new
checkout wrapper `Ops+DynamicSlice.swift`, C8) — rejected rows are
simply overwritten by the next round's write, so attention trims become
pure host bookkeeping (`commitPipelined`); SDPA visibility = ONE lazy
bool mask `col < start + row + 1` ([S, worstLen]) encoding in-block
causality, history visibility, and stale-row exclusion; the packed
accept transfer for the next round prescheduled behind it. Escape
hatches mirror the R33 pipeline (greedy, `processor == nil`,
chain-break -> synchronous round; `DFLASH2_VERIFY_PREBUILD=0` kills).
`finalizeGeneration` rewinds from a per-round capture context.

**Identity MATCH with BIT-IDENTICAL acceptance (147/322) on the very
first build** — and after every subsequent fix. The masked-mq, the
masked-mma, and the dynamic-write paths all reproduce the R34b
acceptance patterns exactly (114/154, 147/322, 143/278).

Three GPU/scheduling costs found and fixed (as-built was 39.2 vs 48.2
same-window):
1. **Round-seam scheduling bubble (the big one):** mlx-core's
   MAX_ACTIVE_TASKS=10 re-throttles the deep pipeline — the encode
   thread leaves the window only when the GPU is most of the way
   through the NEXT verify, and the GPU then drains before the
   following round's first commit. Made the cap env-tunable
   (`MLX_MAX_ACTIVE_TASKS`, C10, lazily latched so the app can setenv);
   =40 flipped stage-2 from -6 to +3.5 same-window. =100 no better.
2. **Full KV-store copy per boundary:** DynamicSliceUpdate donates via
   copy_gpu, but in-flight readers always hold buffer refs at encode
   time under deep pipelining -> ~2.5 GB/round of copies. Added
   `MLX_DYNSLICE_INPLACE=1` (C9): output aliases the input buffer,
   only updated rows written — safe here because stream FIFO order plus
   the visibility masks make the written region invisible to every
   earlier-encoded read. (Measured ~neutral at cap 10 — the bubble
   masked it — kept as part of the default set.)
3. **Masked SDPA fell off the mma kernel to mq (+3.5 ms/round):**
   added bool-mask support to `sdpa_vector_2pass_1_mma` (buffers
   13/15/16/17, function constants — no new AOT instantiations; the
   mlx-generated mirror updated byte-identical) and relaxed the
   use_mma gate to bool masks.

Bench harness now setenvs the two mlx knobs (`DFlash2BenchRunner.run`,
overwrite: 0) so the canonical config is defaults-only.

**Canonical R36b** (cooled ABBA, display asleep — an aerial-wallpaper
video decode contaminated the first attempt (ar arms 15.9-19.6; bs3f
26.3 was contention, not code) — identity MATCH all arms): ar 20.0 |
bs3f 32.2 | **bs8f 46.7 (2.34x) — NEW RECORD** (+2.7 over R34b's
44.0) | bs8-adaptive 40.5. 8f runs 46.7/42.5 (wider spread than
R34b's 44.0/43.1; reported median = same convention as all records).
3f flat vs R34b (32.2 vs 32.0): the width-independent encode cost
sits inside a shorter GPU round at S=3, so stage-2 neither wins nor
loses there. The stage-2 projection (~46-48) is REACHED. Remaining
on-device levers are thin: replay cost (~2 ms, could prebuild into
the window's dep graph differently), adaptive-arm bandit retune for
the new round shape, encode-cost reduction (mlx-core). 60 verdict
unchanged: M5-class bandwidth or drafter training.

### R37 — compiled GDN replay + S=16 tree re-priced under the pipelined round

Two follow-ups from R36's "remaining levers" list.

**S=16 two-chain tree re-derived under the new 89.4 ms round (paper
re-pricing, no build):** break-even shrinks with the round — 89.4 x
(5.07/4.17 - 1) ~ 19.3 ms (was 22.5 at 104.5 ms). The pipeline hides
only the ~2.5 ms host glue from R28's bill; the GPU items survive:
QMM +5-8 (mma16n16 still unbuilt — the banked mma16 is a +31 ms
hole), SDPA qL16 +2-4, GDN x2 ~+3 (with the compiled replay below),
16-mask propose +3-4 => +13-19 ms for <= +0.9 tok/round (R21's
offline bound) => net 0 to +0.5 tok/s at the optimistic edge of every
estimate, and it still requires building a new QMM tile (dual AOT
instantiate-list dance). The pipelined round does NOT resurrect the
tree; wash confirmed a second time, now under stage-2 economics.

**Compiled GDN replay:** the accept-invariant replay
(`prebuildSpeculativeRollback`) ran ~9 eager launches per GDN layer
per round (sigmoid(b), decay-gate chain, posMask compare, conv gather
indices, contiguous) — round-critical-path cost since stage-2 replays
EVERY round. Folded into one `compile` trace shared by all layers
(`compiledReplayCapture`, DFlash2Support.swift; mask-free captures
only, eager fallback kept; kill: `DFLASH2_COMPILED_REPLAY=0`).
Nested-compile inlines (`is_tracer` early-out in mlx compile.cpp) and
the custom scan kernel traces as an ordinary primitive. Identity:
MATCH, acceptance bit-identical 147/322.

Measurement traps hit (both banked classes): a post-overnight-idle
screen showed 29.1/39.0 — post-idle clock ramp (ar arms rose 19.0 ->
20.0 monotone) plus fresh-binary Metal JIT in run0, not the change. A
warm eager/compiled/eager BAB was thermally confounded (middle arm
hottest: ar 18.5 vs 19.7/19.6 brackets); at matched position
(run0 vs run0) compiled ~ eager (38.5 vs 39.0). Warm A/Bs cannot
resolve a +-1 effect on this machine; cooled canonical is the only
arbiter.

**Canonical R37** (cooled ABBA, display asleep, identity MATCH all
arms, ar 20.1 — clean): bs3f 29.2 | **bs8f 47.2 (2.35x) — NEW
RECORD** (+0.5 over R36b) | bs8-adaptive 42.4 (+1.9 over R36b). Both
8f runs improved (47.2/42.9 vs 46.7/42.5) and both adaptive runs
(41.9/42.4 vs 40.5/40.3) — coherent, not slot luck. 3f pairs overlap
across sessions (28.7/29.2 vs 29.1/32.2 — R36b's 32.2 was the lucky
run; run0s equal), so no 3f regression. Compiled replay stays
default-ON. The adaptive arm's +1.9 says launch overhead weighs more
in mixed-width rounds; still below fixed 8f, retune still pending.

**R38 — sync-mode vprofile on the R37 build closes the glue question.**
Steady state (91 rounds): verify-seg 87.0 / verify-attn 17.1 profiled
— bit-for-bit the R28 composition (85-90 / 17-18), so the segment
interior is unchanged: QMM ~55 + GDN scan ~4 + **glue ~12-16 ms**
(hundreds of 10-20 us norm/rope/elementwise launches, ~24 MB of
actual traffic ~= 0.06 ms — pure launch cost). The kill: even PERFECT
glue elimination lands the round at ~76 ms = 56.2 tok/s < 60, and
perfection is not on offer (conv, scan pre/post, and boundary copies
are real work). The realistic program — rmsnorm folded into the mma8
prologue (each output tile already streams the full [8,5120] row) +
rope as qkv epilogue — absorbs ~100-150 launches ~= 3-5 ms => ~49-50,
days of fork-kernel surgery, bitwise-identity risk. Every branch of
the 60 decision tree now terminates in a measured number: tokens/round
needs drafter training (R30), round-time bottoms out ~76 ms
theoretical / ~84 ms realistic on M3 Max. Cap/blocks env retunes
skipped deliberately: warm A/Bs cannot resolve +-1 here (R37 traps)
and a canonical per knob value costs 50 min for a coin flip.

**R39 — AGX utilization counter on the record build: scheduling is
CLOSED.** Live driver sampling (ioreg AGXAccelerator, 50 ms cadence)
across a full 8f bench: loaded-sample mean 99.0, 95.7% of samples
>= 98%; per-second means through every decode window ~99-100 with
only isolated 1-second dips at run/phase boundaries. A residual
per-round seam (the pre-stage-2 8.4 ms strip would read ~91%
sustained over a 8f decode) does NOT exist anywhere in the timeline.
The GPU is pegged; every remaining millisecond is GPU work, priced in
R38. On-device program state: record 47.2, theoretical ceiling ~56,
realistic ~49-50 via the glue kernel program; 60 = bandwidth or
drafter training.

### R40 — op census (2,663 dispatches/pass) + same-input QMM stacking: the glue map was wrong

**Op census probe (C11, mlx checkout):** env-gated primitive-dispatch
counter in transforms.cpp (`MLX_OP_CENSUS=1`, recording windowed by
`MLX_OP_CENSUS_ACTIVE` which verifyStep toggles in vprofile mode;
dumped at exit). The verify pass dispatches **2,663 primitives** —
an order of magnitude past every prior estimate. Per pass: QMM 496
(q/k/v, gate/up, and the GDN's FOUR in-projections all launch
separately), RMSNorm 304 (6.3/layer — 48 GDN x 5 + 16 attn x 4
exactly; the model is 64 layers, 48 GDN + 16 attention), compiled
elementwise ~350, Contiguous+Concatenate ~180, Add 128, CustomKernel
84 (48 scans + rotations), Convolution 48, RoPE 32, SDPA 16.

**Same-input QMM stacking (bitwise-exact):** projections sharing an
input concatenate along the output axis — each output row keeps its
own K-accumulation and quant groups, so outputs are bit-identical
(MATCH + bit-identical acceptance on every run, both phases).
Phase 1: gate+up in all 64 MLP blocks (Qwen3NextMLP.stackGateUp,
post-load, originals released via update(modules:) — direct
@ModuleInfo assignment trips the metaState assert). Phase 2: the GDN
4-way in-proj (48) and attention q/k/v (16) — 128 groups total, QMM
launches 496 -> ~256/pass. Kill: DFLASH2_STACK_GATEUP=0.
Warm same-sequence A/B (control vs diag, control COOLER): unstacked
39.8/44.8 vs stacked (phase 1) **48.0/50.8** — the merge effect far
exceeds launch arithmetic (fewer encode boundaries + single-grid
streaming). Canonical pending.

**Two measurement traps found (both cost a day):** (1) the FIRST
LAUNCH of a freshly built binary fights an MTLCompilerService
pipeline-compilation storm for minutes (ar arms 5-13 tok/s, erratic,
recovering; second launch of the same binary clean) — never measure a
fresh binary's first launch; the bench protocol now includes a
throwaway warm-up invocation. (2) the owner using the machine wakes
the display (aerial-wallpaper decode + compositing) and runs other
agents — the R41 canonical attempt is INVALID (ar 14.6-20.2 erratic,
8f 45.5 under contamination). The canonical protocol is now
quiet-gated: 8 consecutive minutes of WindowServer < 2% CPU before
the cool-down starts, with a top-CPU sampler riding the bench for
post-hoc validation, and an ar-arm health check (all arms 19.5-20.5)
as the validity criterion.

**Canonical R42b — stacked build: bs8f 47.9 NEW RECORD (floor), all
arms at all-time highs under active-use contention.** The owner was at
the machine (sampler: Activity Monitor 13-28% sustained — its GPU
history polling is the likely AR suppressant — plus corespotlightd
32.9 burst, loginwindow, GlobalProtect; ar arms flat 13.8-14.3, so
the speedup ratio is unusable). Contention only ever slows a run, and
identity is bitwise MATCH (AR tokens are speed-independent), so the
spec-arm numbers stand as LOWER BOUNDS: **bs8f 47.9** (vs 47.2), bs3f
34.2 (vs 32.2 — stacking finally moved the launch-bound short-round
arm, +2.0), bs8-adaptive 43.5 (vs 42.4). Vs the historical clean ar
20.0: >=2.40x. Warm same-thermal evidence (48.0/50.8 diag) says a
clean canonical lands ~49-51; rerun when the machine is idle. Earlier
attempts: r41 and r42 invalid (user-active + maintenance-burst
contamination; the displaysleepnow-triggers-maintenance trap is why
immediate-start benches sicken — the historical 20-min cooldown
absorbed it).

### R43 — census on the stacked build: QMM 256 confirmed; RMSNorm is the next class

Re-ran the op census (C11) on the stacked build: **QMM 256/pass
exactly** (496 -> 256, matching the 128-group arithmetic), Contiguous
UNCHANGED (the strided slice views added zero materialized copies),
profiled verify-seg 87.0 -> 79.3 ms. Largest remaining non-QMM launch
classes: RMSNorm 304/pass (96 = GDN q/k weightless norms + their
scale muls; 32 = attention q/k norms; the rest input/post norms),
compiled elementwise ~350, Contiguous+Concatenate ~180.

### R44 — fused GDN q/k norm in the scan kernel: REJECTED — the trajectory-sensitivity trap

Folded the weightless q/k RMS norms + scale factors into
`gated_delta_step`'s load path (4 kernel variants, `_qknorm` suffix;
capture/replay plumbed via `GDNCapture.qkNorm` incl. a compiled-replay
twin with the flag baked in, since a kernel choice cannot be a trace
input). Both decode arms share the kernel, so spec==AR held by
construction: **output-identity MATCH on every run.** The math is
exactly eager's formulas — but computed in f32 without eager's bf16
intermediate roundings (eager rounds the rmsNorm output AND the
bf16-cast scale constant). More precise, not bit-identical.

Result (screen, ar arms healthy 19.4-19.8): the greedy trajectory
forked and the new content drafts far worse — acceptance **45.7% ->
33.6%** (147/322 -> 134/399, bit-identical across 3 runs and 2
launches = deterministic), rounds 45 -> 58 for 192 tokens, bs8f
median 47.9-record -> **34.3**. Kill-switch A/B on the same binary
(DFLASH2_FUSED_QKNORM=0) restored 147/322 and 45.8 exactly ->
attribution clean, eager path untouched. Default flipped to OFF
(opt-in DFLASH2_FUSED_QKNORM=1); code kept as a banked variant.

**Trap (program-wide, binding):** any numerics change that perturbs
the AR token stream even by one ulp re-rolls the draft-acceptance dice
on the canonical prompt, and the swing (±12 points of acceptance,
~±13 tok/s) dwarfs any per-round launch saving (~1 ms here). The
record's 45.7% acceptance is a property of the exact bitwise
trajectory, not of the model. Every future lever must either preserve
the AR stream bit-for-bit (as QMM stacking did) or be priced against
an acceptance re-roll. This indicts the planned
layernorm-into-mma8-prologue fusion unless it emulates eager rounding
bitwise (knife-edge: even reduction-order ulps can flip a greedy
token over a 9k-token recurrence).

### R45 — clean canonical: 47.3; the ~49-51 projection REFUTED; plateau ~47.5

Clean cooled canonical on the R44 build's default path (bit-identical
to the record: stacking on, fusion off; binary pre-warmed, 20-min
cooldown): ar median 19.9 (runs 1-3 all 19.8-19.9 = healthy; run0
14.7 is the per-launch first-run ramp seen on every launch today),
**bs8f 47.3 (2.38x)**, bs3f 34.5 (1.74x), bs8-adaptive 43.7 (2.20x);
identity MATCH all arms; acceptance bit-stable (147/322, 114/154,
143/278). Sampler: Activity Monitor open with intermittent 28-30%
spikes but ar-health in band -> valid. Verdict: **the r42b 47.9 floor
stands as the record** (contention only slows, so it was real);
canonical-to-canonical spread is 47.2/47.9/47.3 -> the plateau is
~47.5 +/- 0.5. The warm-diag ~49-51 projection is REFUTED —
**measurement lesson: warm same-sequence A/B LEVELS do not transfer
to cooled canonicals (clock/thermal state differs); warm A/Bs rank
variants, only canonicals set records.**

### R46 — MLX_MAX_ACTIVE_TASKS sweep: a cliff between 40 and 48

Warm 4-leg sweep on the record path (order 64,24,48,32 to decorrelate
drift): cap 24 -> 46.8, 32 -> 46.8, 48 -> 44.1, 64 -> 44.7 (bs8f
median; acceptance bit-identical everywhere — pure scheduling).
Caps <= 32 beat >= 48 by ~2.3, past warm noise. The current default 40
sits at the cliff edge and was not in-sweep; 24-vs-40 ABBA follows
(r48).

### R47 — drafter same-input stacking: bitwise-exact, speed read pending

The R40 stacking treatment applied to the 5-layer drafter, which runs
an 8-step launch-bound sequential chain per round: DFlash2Attention
gains block-side q|k|v and context-side k|v stacks (k/v weights
duplicated across the two — negligible at 4-bit), all three call paths
(eager, draftContextKV trace, draftPreBody chain) routed through them;
DFlash2MLP gains gate|up; the walker + bench stack the drafter after
load (10 blocks). **Gate PASSED: accepted=147/322 bit-identical in
every run of every launch, identity MATCH.** The speed A/B leg was
contaminated (ar 13.2 mid-leg); stacking kept on the bitwise gate +
R40 priors (launch removal only), canonical to confirm.

**R46 addendum — 24-vs-40 ABBA (r48, stacked-drafter binary): no
difference.** Legs 40/24/24/40 -> 51.8, 48.7, 47.9, 47.3 (ar healthy
19.2-21.1 everywhere, acceptance bit-stable). cap40 mean 49.6 vs
cap24 48.3, inside the warm spread — the cliff is only above 40; the
default stays 40. REFUTED as a lever. Note: leg1 run1 hit **51.8**,
an all-time-high single run, with the drafter stacking in — all four
leg medians sit at/above the 47.5 canonical plateau; canonical r48
follows.

### R48 — canonical on the stacked-drafter build: 3f 35.0 NEW BEST; 8f within plateau

Cooled canonical (drafter stacked, cap 40): ar 19.7 healthy, **bs3f
35.0 (1.78x) — new best 3f** (34.2 r42b, 34.5 r45; both runs 33.9/
35.0 solid — the launch-bound short-round arm is where drafter launch
removal shows), bs8f 46.5 (run1 hit by a kernel_task 13.7% burst mid-
leg, fell to 30.0 — the run0 half is the clean read), bs8-adaptive
43.1. Identity MATCH all arms; fixed-width acceptance bit-stable
(adaptive's 143/278 vs 130/241 across runs is the bandit reacting to
the disturbed timings, not numerics). Verdict: drafter stacking
ACCEPTED (bitwise gate + 3f gain); bs8f record stands at 47.9,
plateau ~47.5.

### R49 — elementwise verify conv: bitwise-PASS but REFUTED for speed; concat reuse kept

The S>1 generalization of `decodeConv` for the verify path (in-trace
f32 tap chain, compile-fused into the segment): **bitwise gate PASSED
at S=3 and S=8** (114/154 + 147/322 exact, MATCH) — but same-binary
A/B says the specialized Convolution kernel is FASTER than the fused
elementwise chain: off 43.3/33.1 vs on 42.1/31.6 (8f/3f, ar 19.5
both legs). Default flipped OFF (opt-in DFLASH2_ELEMENTWISE_CONV=1).
KEPT from the same change: verifyForward now builds `convInput` ONCE
(generalConv used to build its own duplicate — the census's ~99
Concatenate/pass was 2x48 + boundary bits), removing ~48 concat
launches/pass, bitwise-neutral by construction. Also of note: the
whole evening warm band drifted down (42-43 vs 45-52 earlier on
identical configs, ar healthy in both) — cross-binary warm compares
remain meaningless; only same-binary pairs and canonicals count.

**R49b screen (concat-reuse baseline binary):** gate PASSED (both
arms bit-stable + MATCH), 42.8/31.9 in the evening's drifted band —
no regression. This binary is the new baseline (target + drafter
stacking, single convInput concat, fused-norm and elementwise-conv
banked default-off). Canonical record attempt deferred to a cool
machine phase.

**R49c — production stacking wired.** LLMActor's
`loadDFlash2DrafterIfPresent` now stacks BOTH sides of the speculative
pair at engine load (gated on a paired drafter; PARO/VLM classes
no-op via the QuantizedLinear casts; kill DFLASH2_STACK_GATEUP=0).
Verified live: app log "target=128 draft=10 blocks", bench-side calls
idempotent (0), gate 147/322 + MATCH, 46.3 at plateau. The real agent
decode now benefits from the round's wins, not just the bench.

### R51 — Metal System Trace attribution attempt: NEGATIVE (method, not target)

Tried per-kernel GPU-time attribution via `xctrace` Metal System
Trace attached to a live bench. Three blockers: (1) the shader
timeline table needs a launch-time profiler instrument — attach
leaves it empty; (2) MLX encoders are unlabeled, so encoder-level
intervals cannot be mapped to kernels; (3) tracing overhead starves
the GPU (37% busy vs the real 99-100%) AND inflates measured GPU
time ~2.7x (240 ms/round of "GPU busy" vs the real 89), biased
toward small encoders — glue quantification from this data would
overstate glue. The encoder-duration histogram is consistent with
known attribution (~3 large QMM-segment encoders/round at ~24 ms
inflated + ~106 mid encoders). The 578 MB trace is kept in the
scratchpad (r51-metal.trace) — Instruments GUI with a Shader
Timeline template on a fresh launch is the tool that would answer
the per-kernel question, if the glue pool ever needs exact pricing.

### R52 — post-ship gate screen (committed tree, rebuilt from pushed pins): PASS

The round-5 series shipped (both mlx forks + vendor pushed, tesseract
d11f9cc1/38823cf0), the app rebuilt through a full SwiftPM re-resolve
against the GitHub pins, and the gate screen re-run on the fresh
binary: bs8f accepted 147/322 on every run, bs3f 114/154, identity
MATCH both arms, warm 41.7/2.48x (3f 31.5/1.87x). The bench's own
target stacking now reports 0 blocks — expected: production stacking
(LLMActor, R49c) already stacked the target at engine load (app log
"target=128 draft=10 blocks"), and the walker is idempotent. The
checkout-loss risk is retired; trajectory bit-identity survived the
ship end-to-end.

### R53 — draft precision (4/8/bf16-bit drafter): axis CLOSED, 4-bit stands

The one lever the R44 trajectory trap does not bind: the draft only
proposes, the target verifies, so draft precision is identity-safe by
construction and can only move acceptance. The draft ships bf16 on
disk and is 4-bit quantized at load (reference parity) — so 8-bit and
bf16 are pure load-config probes. New env lever `DFLASH2_DRAFT_BITS`
(app DFlash2Support.loadDrafter; default 4 bit-identical — control leg
reproduced 147/322 + MATCH exactly).

Same-binary warm legs, repeat prompt, 8f (acceptance is deterministic
per config; one leg reads it exactly):
- 4-bit (control): 147/322 = 45.7%, warm 46.2
- 8-bit: 148/315 = 47.0%, warm 45.4
- bf16: 148/315 = 47.0% — IDENTICAL proposals to 8-bit — warm 40.3
  (drafter unstacked at bf16: plain Linear fails the QuantizedLinear
  casts, as designed)

Verdict: the entire draft-precision axis is worth +1.3 acceptance
points at its ceiling, and the extra draft weight stream eats it
(8-bit nets ~-0.8 warm). 8-bit is already proposal-equivalent to bf16
on this prompt — the whole quantization loss lives in 4→8, and it is
tiny. The drafter-quality gap vs paper tau is content/architecture,
not precision — R30's verdict (only *training* moves draft quality)
now confirmed at the measured ceiling. MATCH held on every leg, as
predicted. Default stays 4-bit; the lever stays as a probe.

60-verdict status: every axis — round time (R38 perfect-glue 56.2),
tokens/round via width/trees (R28/R37), drafter precision (R53),
drafter training (R30, priced), hardware (M5-class ~62 projected) —
is now measured, built, or priced. 60 on M3 Max under strict
losslessness needs new hardware or a trained deeper drafter; the
stack leads all known public numbers bandwidth-normalized (R28).

### R54 — keyed-path warm-start integration (ADR-0059): identity PASS, env re-roll documented

The prefix-cache × DFlash2 integration (app `PrefillExecutor` stays the
single prefill authority, vendor iterator gains `prefilledPrefixTokens`
warm start) gated two ways on 2026-08-24:

Cold gate, bs8f, same session A/B — parent binary (both repos stashed
to parent) vs integrated binary: **both** accepted=125/469, **both**
output-identity MATCH, each bit-stable across repeat runs. The vendor
change is bitwise-neutral at prefix 0, as designed. The banked 147/322
(R52/R53) did **not** reproduce on *either* binary: an environmental
acceptance re-roll of the R44 trajectory class (suspects: plist
speculationMode now `dflash2` vs `automatic` during R52/R53, which
loaded the MTP drafter beside it; machine contended, ar median 16.8).
Identity MATCH is the invariant that matters and it held everywhere;
125/469 + MATCH is the current-environment cold reference. A quiet
canonical re-bank is optional hygiene, not a blocker.

Live warm gate (the point of the exercise — server, tools + thinking,
temp 0): turn 1 `lookup=hit(branchPoint at 40/328)` → acceptance 62.3%
(43/69, 11 rounds), `directToolLeaf captured — offset=383`; turn 2
`cached_tokens 383/428`, `restoreMs=6.2`, acceptance 50.6%,
`canonicalLeaf captured — offset=490`, 2.35s total. Speculation and
the radix cache compound on the exact traffic ADR-0056's amendment had
parked — the #437 acceptance criterion.

### R55 — 2026-09-03/04 vendor re-pin (upstream e3d4a20): trajectory bit-identical, speed at parity

Re-pinned `Vendor/mlx-swift-lm` onto upstream main `e3d4a20` (44 commits:
#471 merged, #572 fused GDN in-projections, #573 direct expert reduction,
#589 `CompiledTrace` compile state, #569 generalized decode segments, …).
Our GDN in-proj stack was dropped for upstream's #572 fusion; every DFlash2
compiled function became a `CompiledTrace` (details:
`docs/mlx-swift-lm-fork.md`, ADR-0058 re-pin addendum).

Gate, Qwen3.8-27B 4-bit + DFlash2 draft, bs8f, direct-binary launches,
interleaved old-pin (`ddc1f66`) vs new-pin (`a7e5d9b`) binaries built from
the same tesseract tree:

| leg | ar median | bs8f run0 / run1 | accepted |
| --- | --- | --- | --- |
| old-1 | 20.6 | 24.6 / 30.2 | 115/532 |
| new-1 | 20.8 | 28.5 / 28.9 | 115/532 |
| old-2 | 20.6 | 28.9 / 29.6 | 115/532 |
| new-2 | 20.9 | 29.6 / 32.1 | 115/532 |

Better-of-two: old 30.2 / 29.6, new 28.9 / 32.1 — parity within the
run-to-run spread; output-identity MATCH on every leg; **acceptance
115/532 bit-identical on both binaries**, so the 44 upstream commits and the
two re-expressed carries leave the greedy trajectory untouched.

Why 115/532 and not the banked 125/469 (R54) or 147/322 (R52): the bench
prompt is built from `ARCHITECTURE.md` + `CONTEXT.md` + `AGENTS.md` + the
first 12 ADRs, and `ARCHITECTURE.md`/`CONTEXT.md` changed since R54 — a
different prompt, hence a different trajectory and acceptance. Same
R44 class as before, now with the cause pinned: **acceptance references
are only comparable across identical prompt bytes**; a future bank should
record the prompt hash. Under this prompt both binaries sit at ~30 tok/s
bs8f (1.4–1.5x over ar 20.6), so the 45 tok/s block line from the 47.9
record is not a valid absolute gate today; the old-vs-new A/B is.

Kill-switch probes on the new binary (single runs): `MLX_QWEN_FOUR_GDN=0`
→ 116/525, `DFLASH2_STACK_GATEUP=0` → 116/539 — each toggle perturbs the
trajectory by ~1 accepted token, i.e. the fused/stacked QMMs are NOT
bitwise against their unfused forms on this base (suspect: kernel choice by
N on the M=8 verify window; unverified). Neither toggle recovers anything.

Also found while gating: two of the real-model `DFlash2ParityTests`
running in one process (a `swift test --filter` substring match) deadlock
ABBA between mlx-swift's global `compiledSilu.lock` and its recursive
`evalLock`; run them one at a time (`--filter 'name\(\)'`).
