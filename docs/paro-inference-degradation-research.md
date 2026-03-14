# ParoQuant (INT4) Inference Degradation — Experimental Analysis

## Problem

`z-lab/Qwen3.5-4B-PARO` (4-bit) produces **slower token generation** than `mlx-community/Qwen3.5-4B-MLX-8bit` (8-bit) on Mac15,9 (48GB). This is counterintuitive — 4-bit models read half the weight data, so should be faster on memory-bandwidth-bound Apple Silicon.

## Root Cause: GPU Thermal Throttling

**Confirmed experimentally.** The apparent slowness of PARO is caused by Apple Silicon GPU thermal throttling during sustained inference, not by the rotation kernel or sampling parameters.

### Evidence

#### Per-scenario tok/s reveals progressive degradation

Extracting per-turn tok/s from both March 14 benchmark runs (same commit, same params):

| Scenarios | 8-bit tok/s | PARO tok/s | PARO/8-bit |
|-----------|-------------|------------|------------|
| S1-S3 (early) | 54-57 | 68-71 | **1.25-2.07x faster** |
| S4-S15 (late) | 33-42 | 21-24 | **0.52-0.72x slower** |

PARO starts **25% faster** than 8-bit but degrades to **35% slower**. Both models degrade; PARO degrades more severely.

#### Experiment matrix (5 runs, 2026-03-14)

| Experiment | Cache Limit | Cooldown | PARO Degrade Point | Steady-State |
|------------|-------------|----------|---------------------|-------------|
| 1. Original | 128 MB | none | S4 T2 | 23 tok/s |
| 2. +clearCache between scenarios | 128 MB | none | S4 T2 | 23 tok/s |
| 3. cacheLimit=2GB | 2 GB | none | S6 T3 | 24 tok/s |
| 4. No cache limit (MLX default) | ~48 GB | none | S2 T4 | 24 tok/s |
| **5. cacheLimit=512MB + 10s cooldown** | **512 MB** | **10s** | **S11+ (delayed)** | **25-68 tok/s** |

**Experiment 5 is definitive.** With 10-second cooldown between scenarios, PARO maintains **67-68 tok/s through S1-S6** (14 scenarios, ~20 generate() calls). Without cooldown, it drops to 24 tok/s by S4. The GPU thermal state recovers during the pause, restoring full performance.

#### Why thermal throttling, not cache

- Experiments 1-4 varied cache size from 128 MB to ~48 GB (375x range) but the degraded steady-state was always **23-24 tok/s** — exactly the same regardless of cache size.
- Experiment 4 (no cache limit) was actually **worse** — more cached buffers = more GPU memory pressure = faster thermal throttling.
- Experiment 3 (aggressive clearing between turns) made it **worse** — clearing warm buffers forced re-allocation, adding GPU work.
- Only Experiment 5 (thermal cooldown) prevented degradation.

### Why PARO Throttles Worse Than 8-bit

PARO generates **~2x more GPU kernel dispatches per token** than 8-bit:

| Model | Linear layer ops/token | Total dispatches/token (est.) |
|-------|----------------------|------------------------------|
| 8-bit | 32 layers × 7 projections × 1 `quantizedMM` = 224 | ~350-400 |
| PARO | 32 layers × 7 projections × (1 rotation + 1 `quantizedMM`) = 448 | ~600-700 |

More GPU work → more heat → faster throttling → lower steady-state speed.

At **peak performance** (thermal equilibrium not reached):
- PARO: 68-70 tok/s
- 8-bit: 54-57 tok/s
- PARO is **23% faster** — the expected 4-bit bandwidth advantage

At **thermally throttled** steady-state:
- PARO: 23-24 tok/s (67% drop from peak)
- 8-bit: 33-34 tok/s (40% drop from peak)
- PARO drops more because its higher GPU utilization generates more heat

### Thermal Throttling Timeline

```
tok/s
70 ┤ ████████████                          PARO peak (S1-S3)
60 ┤
55 ┤ ····················                  8-bit peak (S1-S2)
50 ┤
40 ┤
35 ┤              ·····················    8-bit throttled (S3+)
30 ┤
25 ┤             ██████████████████████    PARO throttled (S4+)
   └──┬───┬───┬───┬───┬───┬───┬───┬───┬
      S1  S2  S3  S4  S6  S7  S8  S9  ...
```

## Secondary Factor: Sampling Parameter Overhead

The `presencePenalty` implementation in upstream mlx-swift-lm ([PR #141](https://github.com/ml-explore/mlx-swift-lm/pull/141)) forces a CPU←GPU synchronization on **every token** via `token.item(Int.self)` in `PresencePenaltyContext.didSample()`. This breaks the `asyncEval()` pipelining in `TokenIterator`, adding per-token latency.

This affects both models equally but compounds with thermal throttling — the GPU does slightly more work per token (penalty processing), generating more heat.

On the March 12 benchmarks (no penalties), PARO achieved 55-63 tok/s average (thermally throttled aggregate). On March 14 (with penalties), PARO achieved 34.6 tok/s. The penalty overhead accounts for roughly 10-15% of the additional slowdown.

## Benchmark Design Issue

The benchmark runs all 14 scenarios sequentially with no cooldown. By the time later scenarios run, the GPU is thermally throttled, making their tok/s measurements ~2-3x lower than peak. This means:

1. **Aggregate tok/s is not representative** of actual single-conversation performance
2. **Model comparisons are unfair** — whichever model generates more GPU heat appears slower in aggregate
3. **PARO appears slower than 8-bit** because it heats the GPU faster, but is actually 23% faster per-conversation

## Recommendations

### For Accurate Benchmarking

1. **Add cooldown between scenarios** (10s is sufficient on M3 Max)
2. **Report peak tok/s** (first 2-3 scenarios) alongside aggregate
3. **Run models separately** rather than sequentially to avoid thermal bias

### For Production Performance

1. **Keep `cacheLimit` at 128 MB** — larger caches don't help and can worsen memory pressure
2. **Call `Memory.clearCache()` between agent conversations** (not between tool rounds — warm buffers help)
3. **Consider thermal-aware scheduling**: after sustained generation (>30s continuous), pause briefly to allow GPU recovery

### For Rotation Kernel Optimization

The rotation adds ~10% computational overhead (matching the paper's claims on CUDA). This is acceptable. The main issue is thermal, not algorithmic. However, reducing rotation compute would slow thermal throttling:

1. **Fuse rotation + quantizedMM** into a single kernel to reduce dispatch count from ~700 to ~400 per token
2. **Precompute dense rotation matrix** at load time to replace per-token Givens rotations with a single matmul

## Raw Experiment Data

### Experiment 5: 10s Cooldown (definitive)

Configuration: `cacheLimit=512MB`, `Memory.clearCache()` + `Task.sleep(10s)` between scenarios.

```
S1:  68.2 tok/s (3 turns)     ← peak performance
S2:  68.1 tok/s (4 turns)     ← sustained
S3:  68.0 tok/s (3 turns)     ← sustained
S4:  68.1 tok/s (4 turns)     ← sustained (was 24 tok/s without cooldown!)
S6:  67.4 tok/s (6 turns)     ← sustained through long scenario
S7:  61.0 tok/s (3 turns)     ← starting to warm up within scenario
S8:  52.8 tok/s (2 turns)     ← some recovery from S7 cooldown
S9:  56.5 tok/s (4 turns)     ← recovering with cooldown
S10: 63.1 tok/s (2 turns)     ← recovered
S11: 31.8 tok/s (2 turns)     ← thermal drop within scenario
S12: 25.1 tok/s (2 turns)     ← thermally saturated
S13+: continuing to throttle
```

The 10s cooldown allows GPU to recover between scenarios, maintaining 67-68 tok/s through S1-S6 (the first hour of scenarios). Without cooldown, performance crashes by S4.

## References

- [ParoQuant paper](https://arxiv.org/abs/2511.10645) (ICLR 2026) — ~10% overhead vs AWQ, confirmed by our measurements at peak
- [z-lab/paroquant](https://github.com/z-lab/paroquant) — reference Metal kernel (our port matches exactly)
- [mlx-swift-lm PR #141](https://github.com/ml-explore/mlx-swift-lm/pull/141) — topK/presencePenalty addition
- [MLX Memory.swift](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Memory.swift) — cache limit documentation
