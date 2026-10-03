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
time; not yet explained.

**DFlash2 on PARO** (`--dflash2-bench`, 8K prompt, 256 tokens, one run):
bf16 148 ms per round, turbo8v4 135, turbo0v4 131, same acceptance; every
arm's DFlash2 stream matched its AR stream.

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
