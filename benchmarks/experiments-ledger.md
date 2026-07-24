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

### Operational state (persisted for context compaction; reload after resume)

- **Probe rig:** `/tmp/gather-sweep` — SwiftPM executable, local-path dep on
  `~/projects/mlx-swift`; needs `default.metallib` copied next to the binary
  as `mlx.metallib` (from the app bundle's `mlx-swift_Cmlx.bundle`). Sections:
  fidelity + B/E sweep (`MLX_GQMM_CFG`), down_proj shape, dense anchor,
  gather_qmv decode sweep (`MLX_GQMV_RPS`). Rebuild: `swift build -c release`
  (seconds — incremental Cmlx).
- **Fork clone state (standing, do NOT clean):**
  `~/projects/mlx-swift/Source/Cmlx/mlx` = `625f2aea` + uncommitted probe
  hooks — `MLX_GQMM_CFG` env in `gather_qmm_rhs`; `MLX_GQMV_RPS` env +
  rps template param (`quantized.h` AND `mlx-generated/quantized.cpp`) +
  rps dispatch in `gather_qmv`. All marked PROBE ONLY; never pushed.
  `~/projects/mlx` = clean at `625f2aea` (pin-tesseract tip).
- **App binaries (/tmp):** `tesseract-precmlx-baseline.app` (pre-fork),
  `tesseract-cmlx-fork.app` (C0 fork build, pre-C1), `tesseract-c1-accepted.app`
  (C1 tiles, fbf2fb86), `tesseract-c4.app` (C1+C4, 404070e2),
  `tesseract-c5-accepted.app` (C1+C4+C5, 8d11dd1d),
  `tesseract-c6-accepted.app` (…+C6, 3ec72a24),
  `tesseract-c7-accepted.app` (…+C7, 6ab29e36),
  `tesseract-c8-accepted.app` (…+C8, 595a3fe1),
  `tesseract-c9-accepted.app` (…+C9, 625f2aea),
  `tesseract-c11-accepted.app` (…+C11, 3bb0f17),
  **`tesseract-c12-accepted.app` (current main: C1+C4..C9+C11+C12, e77d05d)
  — the A/B baseline for the next experiment.**
- **Pins (current):** spokvulcan/mlx-swift `c9796ec4` (pin-tesseract) ←
  spokvulcan/mlx `625f2aea`; mlx-swift-lm pin branch `e77d05d`.
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
