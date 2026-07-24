# mlx-core (Cmlx) optimization roadmap

What is left after the 2026-07-23 experiment-loop session (E1–E11). Every
entry is grounded in a measurement made that session — no priors. The
session's full record: `benchmarks/experiments-ledger.md`. The mlx-core
sources referenced live in the pinned checkout
(`DerivedData/.../SourcePackages/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/`;
the gitlink records upstream `ml-explore/mlx` @ `ce45c525` = tag `v0.31.1` —
an earlier version of this note said `dc43e62d`, which is an mlx-**swift**
revision, not mlx). Scope note: these need changes in **Cmlx / mlx-core** —
the app now consumes a writable fork (`spokvulcan/mlx` + `spokvulcan/mlx-swift`,
scheme in `docs/mlx-core-fork.md`, established 2026-07-23 for the Cmlx loop).
Gain estimates are the measured attribution × the measured headroom, not hopes.

Legend for evidence: E# = experiment in `benchmarks/experiments-ledger.md`.

## M1 — `gather_qmm_rhs` tile geometry at small rows-per-expert (the big one)

- **Evidence (E4):** on the real 35B shapes (E=256, N=512, K=2048, 4-bit,
  gs=128), the sorted-rhs fast path runs 1.37 / 5.14 / 7.69 TFLOP/s at
  B/E = 4 / 32 / 256 — **40.5% of peak at production's B/E=32** while a
  dense 4-bit qmm of the same FLOPs does 7.41 and the kernel itself
  converges to 7.69 at B/E=256. Weight-bandwidth floor is 0.22 ms vs
  3.34 ms actual at B/E=32 → occupancy loss, not bandwidth. Sources:
  `backend/metal/quantized.cpp` — `gather_qmm_rhs` (`bm=16, bn=32, bk=32`)
  and `gather_qmm_rhs_nax` (`bm=64`): at B/E=32 a 64-row tile spans/pads
  two experts.
- **Fix:** retune tile config for small B/E (smaller `bm`, expert-boundary-
  aware tiling, or a split-K path), upstreamable to `ml-explore/mlx`.
- **Estimated gain:** `gather_qmm` 5.14 → ~7.4 TFLOP/s ≈ 1.45× on the three
  expert matmuls ⇒ **~12–15% of 35B prefill** (attribution: MoE matmuls =
  42.8%×78% of 32K prefill per #254).
- **Risk:** high effort (kernel project; #251's precedent — measure the
  roofline question first, it's already answered here).

## M2 — decode command-buffer segmentation (~22% idle)

- **Evidence (E10):** xctrace Metal System Trace on a ctx=128 decode: GPU
  busy 78%, **~40 command buffers per token, ~60 µs inter-buffer gaps**
  ≈ 2.5 ms of the 12.5 ms step idle. (C2, 2026-07-23: the mechanism is
  real — raising `MLX_MAX_OPS_PER_BUFFER` 50→400 gave MoE 8K decode
  +3.5%, 6/6 pairs.)
- **Fix:** mlx-core eval scheduling: coalesce more kernels per command
  buffer, cheaper per-buffer setup, fewer sync boundaries. App/vendor can
  assist by shrinking per-step graph size (fewer ops → fewer segments).
- **C2 verdict (REJECTED as an env knob):** a global op-cap raise is
  blocked by three reproducible regressions — dense 8K decode −2.7%
  (6/6 at 400), dense 128 peak +7.25% (10/10 across 200/400; temporaries
  held per-buffer), MoE 128 prefill −4.5% (9/10, mechanism unidentified).
  Viable forms left: **graph-size-aware cap** or **mid-buffer temporary
  release** — eval.cpp/device.cpp internals.
- **Estimated gain:** up to ~20% decode if fully recovered; realistic
  **~10% decode** (both models).
- **Risk:** medium-high (eval internals; contract subtle).
- **C4 verdict (2026-07-23, ACCEPTED):** split the commit accounting —
  input leg relaxed 50→100 MB and a new output leg (unique temporary
  bytes > 10 MB commits). Decode mid-commits are weight-traffic-driven
  (unique input bytes); each cbuf boundary costs ~60-68 µs of GPU-side
  pipeline drain (E10's gaps, now tied end-to-end). The output leg keeps
  prefill temporaries on the stock cadence — without it a flat input-cap
  raise regresses peak +8-46% (measured: v1/v2/v3 variants). GPU-bound
  phases (dense 32K decode) need the stock small chunks (in200 regressed
  dense 32K decode −1.92%; in100 stays inside tolerance). No pure-Cmlx
  GPU-bound detector can separate the regimes (v4: MoE decode is
  boundary-limited, not throughput-limited — GPU busy either way).
  Measured: MoE decode +2.6/+4.5/+2.4% (128/8K/32K), dense 32K decode
  +4.2%, dense peak −2.2/−13.5/−9.2%, MoE peak −1.7/−1.4%, prefill
  flat, all gates token-identical. ~half of E10's boundary budget
  remains (mid/token 37→~27) — further recovery needs a phase-accurate
  signal (app/library-scoped, out of Cmlx scope) or a cheaper boundary.
- **M2 residual:** the remaining ~20-27 mid-commits/token in decode and
  the boundary cost itself (fence/encoder/commit ≈ 10 µs CPU + ~60 µs
  GPU drain each). Attack the *cost per boundary* (fence-map churn,
  encoder/fence recreation) rather than the count next.

## M3 — small-M qmv / gather_qmv latency floors

- **Evidence (E9/E10):** lm_head qmv [1,248320×2048] = 234 GB/s (good);
  gather_qmv decode shape (B=8) = 51 µs at **88 GB/s**; dense qmm same
  bytes = 41 µs at 108 GB/s. Decode is latency/occupancy-limited at M=1;
  E5 showed call-*count* reduction buys ~0 — per-call latency binds.
- **C3 verdict (2026-07-23, REJECTED):** in isolation (single-graph
  timing, disjoint expert sets) the decode-shape gather_qmv runs at
  **~306 GB/s — the DRAM envelope** — and rps geometry (4→32) moves it
  ≤3.5%, bitwise-identical. E10's 88 GB/s is an eval-environment
  artifact (overlap/contention within the token graph), not kernel
  geometry. **No kernel lever here; the decode gap is M2-class.**
- **Fix:** (dead — see C3) persistent vector GEMM scheduling, K-split
  (dead under the bitwise rule: changes accumulation order).
- **Estimated gain:** none via the kernel. Fold the remaining decode
  question into M2.
- **Risk:** (closed)

## M4 — fused rotate+dequant+GEMM (PARO without the rotated-x round trip)

- **Evidence:** post-E6b the rotation kernel runs ~100–200 GB/s vs a
  ~350 GB/s scale-only floor (probe `ref_scale`); it also materializes a
  rotated copy of every linear's input (extra write+read per projection).
- **Fix:** one kernel doing rotation (f32) → **round to f16** → dequant +
  dot with the SAME accumulation order as `quantizedMM` — the f16 rounding
  must be preserved explicitly or the parity gate fails.
- **Estimated gain:** rotation ≈ 5% of prefill eliminated ⇒ **~3–4%
  prefill**; one kernel less per linear at decode.
- **Risk:** high (fused custom kernel; the accumulation-order contract is
  the hard part; parity gate decides).

## M5 — attention fallback non-GEMM tail

- **Evidence:** #251 measured the two GEMMs = 93% of the fallback's time;
  the ~7% tail (mask/softmax/casts) × attention's 35.6% of prefill
  (#254) ≈ **~2.5% of prefill** sitting in elementwise chains.
- **Fix:** compile-fuse the mask+softmax chain (E2's proven-bitwise class),
  and/or cut score-matrix materialization traffic in the fallback path.
- **Estimated gain:** **~1–2% prefill** at long context.
- **Risk:** low-medium.

## M6 — tokenizer / chat-template render path (TTFT, not mlx-core but adjacent)

- **Evidence:** parity bench tokenize = 0.29 s at 32K (**~110K tok/s**);
  E11 measured ~100 ms per 10K-token render+encode (~206 ms for two).
- **Fix:** profile the Jinja-render vs BPE-encode split; optimize the
  encode loop (trie/batching) in swift-transformers or pre-render stable
  template segments (the E11 memo pattern generalizes).
- **Estimated gain:** **−0.2–0.3 s TTFT at 32K; seconds at 100K+**; E11's
  −205 ms/request already banked the render side of the stable prefix.
- **Risk:** low (CPU-only, cleanly measurable, zero numerics risk).

## M7 — GDN scan redesign (mlx-core only)

- **Evidence (E8/E8b):** 1.9 ms/layer/chunk, sequential-latency-bound,
  ~0.5 µs serial/step/CTA; software pipelining made it *slower* (0.80×);
  chunked/parallel scan changes f32 rounding order (dead under zero-loss).
- **Fix:** kernel redesign within the same sequential arithmetic (different
  thread/state mapping, K-vectorization, per-thread tiles) — the only
  remaining legal axis; expected value uncertain after E8b.
- **Estimated gain:** 2× on the scan ⇒ **~2.7% of 32K prefill**; scan is
  ~5.5% there. **Treat as a floor unless a redesign idea survives a probe.**
- **Risk:** high, uncertain.

## M8 — MoE decode expert-weight prefetch (novel, upstreamable)

- **Evidence (E10):** MoE decode is latency-bound; the 8 experts selected
  per layer per token are known only after that layer's router QMM.
- **Fix:** issue async prefetch reads of the *predicted* expert weights
  (prediction = previous token's routing — temporally correlated) before
  the router runs; the real gather still reads what it selects, so
  **numerics are untouched** — prefetching is pure memory warming.
- **Estimated gain:** hides part of the M3 gather latency ⇒ **~5–10% MoE
  decode** if routing locality holds (measure the overlap rate first).
- **Risk:** medium; needs an mlx-core prefetch op or a vendor-side dummy-
  read schedule; cheap to probe (measure expert-set overlap on real traces
  before building anything).

## Cross-cutting notes for the next loop

- **Bitwise budget:** the session proved `compile(shapeless:)` is
  bitwise-identical on fused elementwise chains (bf16 intermediates
  included) — the fusion family is safe territory; the parity gate
  (`--paro-parity-bench`, token IDs recorded per run) is the arbiter for
  anything touching the model graph.
- **Measurement traps (must re-read):** serialize GPU work; interleave
  A/B (ABBA) — absolute tok/s is not comparable across time; launch via
  `open` (nice 0); divide timings by FLOPs; eval-barrier attribution
  biases itself (coarse tier for absolute seconds).
- **Probe harness pattern:** scratch SwiftPM package on the vendor path
  (2 s rebuilds) with legacy-vs-new kernel copies, ABBA timing, bitwise
  gates — used by E4/E6b/E8b/E9/E10; reuse it before any app-level run.

## Kickoff prompt for the next session (Cmlx loop)

> Run the same endless inference-optimization experiment loop, now scoped
> to **mlx-core (Cmlx)**. Goal: make the Qwen3.5/3.6 PARO models (dense
> AND MoE) faster — prefill, decode, TTFT, CPU overhead, peak memory —
> with ZERO output-quality loss (no quantization changes, no KV-cache
> quantization, no accuracy-for-speed trades).
>
> Before the first experiment: read `benchmarks/experiments-ledger.md`
> (rules, measurement discipline, proven no-gos — never repeat a logged
> failure) and `docs/mlx-core-optimization-roadmap.md` (the measured
> opportunity list M1–M8). Start with **M1 — `gather_qmm_rhs` tile
> geometry at small rows-per-expert** (measured 40.5% of peak at
> production B/E=32, occupancy not bandwidth — already answered, don't
> re-measure; ~12–15% of 35B prefill). First task before any experiment:
> establish a buildable mlx-core fork/pin scheme (the app pins
> ml-explore/mlx-swift at an exact revision whose Cmlx tracks
> ml-explore/mlx @ `dc43e62d`) and record it in the ledger.
>
> Rules (unchanged): exactly one hypothesis per iteration; implement
> minimally; measure Release-only via `scripts/bench.sh` (interleaved ABBA
> against saved binaries, nice 0, serialized GPU — Debug MLX is ~20×
> slower; quit the running app first). Quality gate: anything touching
> numerics or the model graph must pass `--paro-parity-bench` (greedy)
> token-identical vs the unmodified baseline on both target models.
> Verdict: reproducible ≥1% win on any metric, no regression on the
> others → commit (Conventional Commits) + log ACCEPTED; otherwise revert
> completely + log REJECTED. Append to the ledger either way; tree clean
> between iterations. Never stop — keep generating and testing hypotheses
> until interrupted.
