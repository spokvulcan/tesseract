# Small-M quantized matmul kernel study (2026-09-05)

Setup: `DFLASH2_QMM_MICROBENCH=1` mode of the bench runner (random 4-bit gs64
affine weights, M=8 bf16 activations), exact per-kernel GPU time from the mlx
fork's `MLX_KERNEL_PROFILE=1` probe, kernel variants loaded at runtime via
`MLX_QUANTIZED_KERNEL_FILE` (no rebuild). Times below are for the verify's
largest shape, gate_up `[8,5120] x [34816,5120]^T` (89 MB of weights), on the
M3 Max 48 GB. The microbench reproduces the in-model profile within 1%
(359.6 us here vs 360.0 us/launch in the decode profile).

| variant | us | note |
|---|---|---|
| shipped `affine_qmm_mma8n16` (16-wide tile) | 358-360 | 248 GB/s, 3.97 TMAC/s |
| `affine_qmm_mma8` (8-wide tile, `MLX_QMM_MMA8_N16=0`) | 358 | tile width is not the limiter |
| `affine_qmm_mma8n32` (`MLX_QMM_MMA8_N32=1`) | whole round 107 ms vs 68 | register cliff |
| qmv (M=1, same weights) | 266 | 335 GB/s |
| no dequant (raw shift only, wrong result) | 275 | dequant ALU costs ~85 us |
| no MMA (adds instead, wrong result) | 290 | MMA issue costs ~70 us |
| no weight loads (synthesized words) | 423 | slower: memory is NOT the limiter |
| 128+q mantissa dequant, no scale epilogue (wrong result) | 273 | the 2-op dequant is nearly free |
| v2: 128+q dequant + per-group scalar epilogue | 376 | the scalar epilogue eats the win |
| v2 + next-group register prefetch | whole round 84 ms | register cliff |
| v2 with diag-scale MMA epilogue (float MMAs) | 438 | float MMAs are expensive |
| v2 two-group software pipeline | 520 | code/register growth |
| two independent accumulator chains | 362 | MMA chain latency is hidden already |
| half fragments instead of bfloat | 380 | bf16 MMA is native; conversions cost |
| A loaded just in time inside the kt loop | 351 | within noise |
| +16 live registers probe | 392 | ~10% per 16 registers |
| per-group accumulator read probe | 359 | reading `thread_elements` is free |

Reading: at M=8 the 4-bit QMM sits on the compute/bandwidth ridge of this GPU
(1.43 GMAC per launch; ~7 TMAC/s FMA peak = 204 us; 89 MB at 400 GB/s = 223
us). The shipped tile is ALU-issue-bound: MMA issue + dequant + fragment
assembly. Cutting the dequant to two ops per weight (OR the nibble into the
mantissa of 128.0) is nearly free on its own but every way tried of applying
the per-group scale afterwards (scalar epilogue, diagonal MMA, pipelined
accumulators) cost more than it saved. Realistic remaining headroom in this
kernel is ~20% (275 us floor with a free epilogue), not the 35% the qmv rate
suggests. The bf16 rounding of dequantized weights in the shipped tile is
what keeps it bit-identical to the current acceptance trajectory; the 128+q
route changes numerics (acceptance re-rolls 144 -> 141 on travel).

Kernel sources for the variants: scratchpad `kvar/*.h` (not kept).

Addendum (02:05): E35 K-step 32 (half the packed-weight registers per step,
4 kt per step): 376.8 / 204.6 / 2659 us on 34816x5120 / 5120x17408 /
248320x5120 (base 356.5 / 188.0 / 2510). Slower: register footprint is not
the limiter either. The tile study is closed.

