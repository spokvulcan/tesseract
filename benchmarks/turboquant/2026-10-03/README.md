# TurboQuant optimization loop, 2026-10-03

Measurements behind the vendor commits after ADR-0083's integration
(`docs/mlx-swift-lm-fork.md`, "TurboQuant optimization loop"), plus two
experiments that were measured and not shipped. Apple M3 Max, 48 GB,
macOS 27.0. Qwen3.8-27B 4-bit unless a row says PARO (the fp16-activation
pack). Microbenches are `TurboQuantDecodeMicrobench` in
`Vendor/mlx-swift-lm` at the Qwen3.8-27B attention shape (24 query heads
over 4 KV heads, head dim 256), 16 layers; loaded-model runs are the app's
`--turboquant-bench` and `--dflash2-bench`.

## Shipped

**Verify partitions** (`testVerifyKernelPartitions`). The multi-query
verify kernel, 8 rows, alone: 16 key partitions were fastest from 2K to
32K rows and 32 at 64K. One verify pass's attention with each arm writing
its rows (in-place writes, as the app runs them), ms per 16 layers:

| rows | bf16 | turbo8v4 | turbo0v4 |
|---|---|---|---|
| 2K | 4.7 | 8.7 | 5.9 |
| 8K | 12.0 | 11.6 | 9.3 |
| 32K | 26.0 | 26.1 | 22.3 |
| 64K | 45.0 | 46.1 | 40.2 |

**Warm prefill chunks** (`testWarmChunk`). A causal chunk over a
compressed cache longer than 8 rows now dequantizes the visible rows once
and runs mlx's attention: 1.01–1.20x of bf16 SDPA for 16- to 1,024-row
chunks at 8K and 32K rows, where the 8-row MMA kernel took 1.4–2.7x.

**Decode graph build** (`testCacheStepCPU`). CPU time to build one decode
step's attention for 16 layers: bf16 0.7 ms; turbo8v4 3.1 → 1.8 ms;
turbo0v4 2.2 → 1.5 ms (dynamic slice updates for the append, lazily built
slices, cached scale and block-count arrays).

**Decode, loaded model** (`--turboquant-bench`, 8K prompt, 256 tokens,
four alternating rounds, mean tok/s):

| model | bf16 | turbo8v4 | turbo0v4 |
|---|---|---|---|
| 4-bit | 22.2 | 22.4 | 22.5 |
| PARO | 16.6 | 15.2 | 17.7 |

turbo8v4 on PARO stays 8% behind with either arm order. A per-kernel GPU
profile of the two schemes differs by 1% (4,362 vs 4,317 ms over 64
tokens), so the gap is in how the pipelined step overlaps, not kernel
time. Loop 2 (below) traced it to the host.

**DFlash2 on PARO** (`--dflash2-bench`, 8K prompt, 256 tokens, one run):
bf16 148 ms per round, turbo8v4 135, turbo0v4 131, same acceptance; every
arm's DFlash2 stream matched its AR stream.

## Loop 2: PARO decode is host-bound

On the PARO pack the CPU spends about as long building and encoding a
decode step as the GPU spends running it. Timed inside the vendor's token
loop (`--dflash2-bench` AR arm, 8K prompt, 256 tokens, command-buffer cap
lifted so encoding never waits on the GPU), steady state per step:

| arm | graph build | encode | waiting for the token | GPU busy |
|---|---|---|---|---|
| bf16 | 11.4 ms | 33.5 ms | 1.3 ms | 45.9 ms |
| turbo8v4 | 12.9 ms | 33.8 ms | 0.0 ms | 44.4 ms |
| turbo0v4 | 12.4 ms | 34.4 ms | 0.0 ms | 44.2 ms |

TurboQuant's GPU step is shorter than bf16's, but the host never waits, so
its extra work per layer sets the rate. The command-buffer trace agrees:
the GPU idles 1 to 15 ms at token boundaries in the slow runs. Two vendor
changes cut that work.

**Folded scale and casts** (`d0ed0be`). The decode kernels take the
activation dtype in and out and apply the softmax scale themselves: three
dispatches fewer per layer, bit-identical.

**One-dispatch row write** (`cba1803`, on mlx `3c6990d9`'s in-place custom
kernel outputs). Encode, key quantization and every buffer write of a
decode token or verify block run in one kernel, byte-identical to the
separate ops. CPU time per decode step for the 16 attention layers
(`testCacheStepCPU`, fp16, 8K rows, in-place writes on):

| arm | build | encode | step with GPU wait |
|---|---|---|---|
| bf16 | 0.80 ms | 1.20 ms | 4.04 ms |
| turbo8v4, separate ops | 1.75 ms | 2.89 ms | 6.27 ms |
| turbo8v4, one dispatch | 1.67 ms | 1.28 ms | 4.81 ms |
| turbo0v4, separate ops | 1.47 ms | 2.18 ms | 5.45 ms |
| turbo0v4, one dispatch | 1.37 ms | 1.22 ms | 3.95 ms |

A serialized per-kernel GPU profile (`MLX_KERNEL_PROFILE`, turbo0v4, 64
tokens) puts the write at 24 ms per 1,024 layer calls against about 80 ms
for the separate slice updates alone.

**Loaded model, PARO** (`--dflash2-bench --bench-check`, 8K prompt, 256
tokens, three interleaved rounds per build, median):

| build | arm | bf16 | turbo8v4 | turbo0v4 |
|---|---|---|---|---|
| before loop 2 | AR tok/s | 21.2 | 18.3 | 21.1 |
| loop 2 | AR tok/s | 21.1 | 20.9 | 21.5 |
| loop 2 | DFlash2 ms/round | 138.3 | 143.7 | 139.7 |

One loop-2 turbo8v4 run still stalled (18.8 tok/s, 1.6 s of GPU idle);
the other two ran at 20.9 and 21.1. Run-to-run spread is large on this
machine: two identical turbo0v4 runs measured 11.25 and 13.25 s of DFlash2
GPU time for the same rounds, with an animated wallpaper decoding video on
the GPU, so the DFlash2 column does not separate the arms.

## Measured, not shipped

**Recurrent state at 16 bits.** Each prefix-cache snapshot carries 154 MB
of float32 GatedDeltaNet state on Qwen3.8-27B whatever its length; stored
at 16 bits it would halve. Teacher-forced KL against the exact restore
(`--turboquant-bench` arms `+rbf16` / `+rf16`, 256 steps; noise floor = a
re-chunked prefill):

| prompt | arm | decode KL mean | first-step KL | top-1 |
|---|---|---|---|---|
| 8K | noise floor | 8.8e-4 | 2.7e-4 | 100% |
| 8K | bf16 state | 6.8e-4 | 1.0e-3 | 100% |
| 8K | fp16 state | 5.9e-4 | 3.8e-4 | 99.6% |
| 32K | noise floor | 4.7e-4 | 1.8e-3 | 98.8% |
| 32K | bf16 state | 3.9e-4 | 1.1e-2 | 99.2% |
| 32K | fp16 state | 4.2e-4 | 1.1e-2 | 99.6% |

Decode steps stay at the noise floor; the first step after a restore at
32K is about six times the floor's, for either 16-bit format. A
full-precision partition promises bitwise restores (the e2e and the
hybrid-cache correctness runner check it), so this could only become part
of a KV Scheme partition's Stored Form, and it needs a restore-dtype
marker in the snapshot format. Left for the owner's decision.

**Flash attention for head dim 256** (vendor branch `exp/flash-d256`). The
multi-query MMA pass over full-precision K/V never materializes mlx's
`[heads, rows, keys]` scores (1.7 GB per layer for a 1,024-row chunk at
32K rows, 6.7 GB for 2,048 rows at 64K), but run one layer at a time it
takes 1.5–1.8x SDPA's time (`testFlashPrefill`), latency-bound on four
barriers per 32-key block (`testFlashAblation`). Not routed until a
retiled kernel beats SDPA.
