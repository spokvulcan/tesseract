# TTS Performance Optimization — Handoff Prompt

Copy everything below this line into a new Claude Code session to continue iteration.

---

## Context

I'm building a macOS TTS app using a vendored Swift port of mlx-audio's Qwen3-TTS model. Token generation currently runs at **10.6-11.1 tok/s** on an M3 Max (40 GPU cores). The Python mlx-audio reference achieves **~24 tok/s** on equivalent hardware. I need help closing this 2x gap.

## Architecture

The generation loop is in `Vendor/mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSModel.swift`, method `generateVoiceDesign()`. Each token step:

1. **Talker forward** (28-layer transformer): ~30ms — builds lazy graph, evaluated by `.item()` EOS check
2. **Main sampling** + **EOS check** (`.item()` forces GPU sync): ~1ms
3. **Code predictor** (5-layer transformer × 15 sequential passes): ~90ms with sync, dominates step time
4. **Embedding prep**: ~1ms
5. `eval(inputEmbeds)` — evaluates the code predictor lazy graph (one sync per step)
6. `GPU.clearCache()` — frees code predictor temporaries before next step

Total: ~94ms/step = 10.6 tok/s. Target: ~42ms/step = 24 tok/s.

## What's Already Been Investigated & Optimized

### Applied (in current code):
- **Profiling disabled by default** — was forcing 18 `eval()` GPU syncs/step instead of 2. Opt-in via `QWEN3TTS_PROFILE=1` env var.
- **sync `eval()` > `asyncEval()`** — sync eval + clearCache is ~15% faster. asyncEval causes GPU memory contention between code predictor and talker.
- **`GPU.clearCache()` placed AFTER `eval()`** — before eval it's a no-op (lazy graph still holds refs).
- **Code predictor KV cache: `step=16`** instead of default 256 (reduces 5MB→327KB GPU allocation per step).
- **Cache reuse via `trim()`** — pre-allocate 5 KVCacheSimple objects once, reset with `trim(offset)` each step instead of reallocating.

### Investigated but ruled out:
- **`mx.compile()` / `compile()`** — Neither Python nor Swift use it. Swift's `compile()` API exists but can't handle the code predictor because `KVCacheSimple.offset` is a Swift Int (not tracked by compile's MLXArray state tracking). Would need framework-level changes.
- **dtype** — Both implementations use BF16 from safetensors. No conversion needed.
- **Code structure differences** — Python and Swift generation loops are structurally identical. Same eval pattern, same cache handling, same sampling logic.

### Root cause of remaining gap:
- **GPU utilization only 65%** on M3 Max during generation → CPU-bound on graph construction
- **~2300 MLXArray graph nodes created per step** (5 layers × ~150 ops × 15 code predictor passes)
- Likely cause: per-operation overhead in Swift→C++ MLX bridging (each op involves Swift function call, C interop, node creation, ARC refcounting)
- Python's pybind11 bindings may have lower per-operation overhead

## Promising Optimization Directions to Explore

### 1. Reduce graph node count
- **Compile individual stateless ops** (e.g., RoPE application, MLP forward) with `compile(shapeless: true)`. These are called 75-150× per step. Compilation caches the graph, avoiding reconstruction.
- **Fused attention**: Check if `MLXFast` has a fused attention+RoPE op that replaces multiple separate ops.
- **Pre-compute code predictor RoPE**: Positions are always 0-16. Compute cos/sin once at init, slice per pass.

### 2. Reduce code predictor passes
- **Skip later code groups**: Predict only first N code groups (e.g., 8 instead of 16), fill rest with zeros. Later groups encode finer audio details — test quality/speed tradeoff.
- **Parallel code prediction**: The 15 passes are autoregressive (each depends on previous sample). But if temperature=0 (greedy), could predict all 15 greedily in one batch? Needs architectural change.

### 3. MLX Swift framework investigation
- **Profile with Instruments** (Metal System Trace): See actual kernel launch overhead, GPU idle gaps, memory transfer patterns.
- **Check mlx-swift issues/PRs** for known performance gaps vs Python bindings.
- **Test with `MLX_METAL_JIT=1`** or other MLX env vars that might enable optimizations.

### 4. Alternative approaches
- **Quantize code predictor to 4-bit**: Reduces memory bandwidth. Check if `MLX.quantize()` works on the code predictor weights.
- **Use Neural Engine (ANE)**: Currently 0% utilized. CoreML conversion of code predictor could offload from GPU.
- **Smaller model variant**: Check if mlx-community has a distilled/smaller Qwen3-TTS.

## Key Files

| File | What |
|------|------|
| `Vendor/.../Qwen3TTS/Qwen3TTSModel.swift` | Generation loop, profiling, eval pattern |
| `Vendor/.../Qwen3TTS/Qwen3TTSCodePredictor.swift` | 5-layer transformer, attention, MLP, KV cache |
| `Vendor/.../Qwen3TTS/Qwen3TTSConfig.swift` | Model config (numCodeGroups=16, layers, dims) |
| `docs/TTS_PERFORMANCE_INVESTIGATION.md` | Earlier profiling analysis |

## How to Test

```bash
scripts/dev.sh dev    # Build + launch
scripts/dev.sh log    # Tail logs, look for "Token generation:" line
# Enable profiling: QWEN3TTS_PROFILE=1 open /path/to/tesseract.app
```

## Profiling Data (with QWEN3TTS_PROFILE=1)

```
Profile (36 steps): talker=30.69ms (25.0%), sampling=1.00ms (0.8%),
  eos=0.06ms (0.0%), codePredictor=90.29ms (73.6%), perPass=6.19ms,
  embedPrep=1.34ms (1.1%), stepAvg=122.63ms
```

Note: Profiling mode is ~20% slower than normal (forces per-component eval syncs).

## Your Task

Continue optimizing token generation speed toward the ~24 tok/s Python target. Start by reading `Qwen3TTSModel.swift` and `Qwen3TTSCodePredictor.swift`, then try the most promising approaches from the directions listed above. Measure each change with `scripts/dev.sh dev` + `scripts/dev.sh log`.
