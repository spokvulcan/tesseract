# TTS Performance Investigation

## Current State (2026-02-07)

### Playback: SOLVED

Audio playback quality is perfect. `AudioPlaybackManager` accumulates all chunks, plays as one-shot via `AVAudioPlayerNode` with 5ms fade-in.

### Generation Speed

**Latest benchmarks (MLXFast.RoPE + code0Embed reuse):**

```
Token generation: 112 tokens in 5.78s (19.4 tok/s, RTF=0.62x)
Token generation: 312 tokens in 16.38s (19.0 tok/s, RTF=0.63x)
```

- **Current**: 18.2-19.4 tok/s → RTF 0.62-0.66x (well above real-time)
- **Target**: ~22-26 tok/s → RTF ~0.5x (Python mlx-audio)
- **Gap**: ~1.3x slower than Python (down from 2.2x original baseline)

**Decode is NOT the bottleneck** (0.29s). The entire problem is per-step token generation.

## Profiling Data (per-step averages, 36 steps)

Profiling uses sync `eval()` at each checkpoint to force GPU completion. Enable with `QWEN3TTS_PROFILE=1` env var.

```
Talker forward:    30.69ms (25.0%)
Main sampling:      1.00ms ( 0.8%)
EOS check:          0.06ms ( 0.0%)
Code predictor:    90.29ms (73.6%)   ← THE BOTTLENECK
  Per-pass avg:     6.19ms (× 15 passes)
Embedding prep:     1.34ms ( 1.1%)
Total step avg:   122.63ms
```

**Note**: Profiling data above is from session 1 baseline (10.6 tok/s). Should be re-profiled after optimizations.

## Optimizations Applied

### Session 1

| #   | Optimization                                                           | tok/s     | Improvement |
| --- | ---------------------------------------------------------------------- | --------- | ----------- |
| 0   | Baseline (before session)                                              | 9.3       | —           |
| 1   | Profiling off by default (was adding 18 eval() syncs/step)             | 10.7      | +15%        |
| 2   | sync `eval()` beats `asyncEval()` — GPU memory contention              | 10.7      | confirmed   |
| 3   | `GPU.clearCache()` AFTER eval (frees code predictor memory for talker) | 10.6      | confirmed   |
| 4   | KV cache: `step=16` (vs 256), pre-allocate once, reuse via `trim()`    | 10.6-11.1 | +0-5%       |

### Session 2

| #   | Optimization                                                                     | tok/s     | Improvement | Quality |
| --- | -------------------------------------------------------------------------------- | --------- | ----------- | ------- |
| 5   | Cache code0Embed (avoid duplicate embedding lookup per step)                     | 14.3-14.8 | +33%        | ✅ Good |
| 6   | Pre-computed RoPE table (slice per pass from pre-eval'd cos/sin)                 | 14.9      | +1%         | ✅ Good |
| 7   | **MLXFast.RoPE** in code predictor (fused Metal kernel, ~1000 fewer graph nodes) | 18.2-19.4 | +28%        | ✅ Good |

### What we tried that DIDN'T work:

- **Inline code predictor sampling** (`categorical(logits / temperature)` replacing `sampleToken()`): Produced noise/corrupted audio BOTH with `* invTemp` and `/ temperature`. Tested twice. The code is character-for-character equivalent to `sampleToken()` internals yet produces completely different (broken) output. Root cause unknown — possibly MLX Swift graph evaluation order or PRNG state interaction. Abandoned.
- **Pre-computed code predictor RoPE table** (session 2, optimization #6): Works correctly but superseded by `MLXFast.RoPE` which eliminates the need for manual RoPE entirely.

### What we tried that DIDN'T help:

- **`asyncEval(inputEmbeds)`**: 9.3 tok/s (worse than sync eval's 10.6). GPU memory contention — async leaves code predictor memory allocated while talker runs.
- **`GPU.clearCache()` before eval**: No-op — lazy graph still holds tensor references.
- **`compile()` for code predictor**: Not feasible. KV cache `offset` is a Swift `Int`, not tracked by MLX compilation. The compiled graph would use stale offsets on replay.
- **`compile()` for talker**: Same issue. `KVCacheSimple.offset` is `Int` (not `MLXArray`), used in control flow and slicing within `update()`. Compile cannot trace through Swift-level mutations of `offset`.

## MLXFast.RoPE Details

`MLXFast.RoPE` is a fused Metal kernel in mlx-swift that replaces manual RoPE computation (~14 graph nodes per application) with a single kernel call. For the code predictor (5 layers × 2 tensors × 15 passes = 150 RoPE applications per step), this saves ~1050 graph nodes per step.

**Critical shape requirement**: `MLXFast.RoPE` uses `x.shape[-2]` as the sequence dimension. Input must be `[..., seqLen, headDim]` or `[batch, numHeads, seqLen, headDim]` (after transpose). Applying to `[batch, seqLen, numHeads, headDim]` (before transpose) causes `numHeads` to be used as sequence positions → produces whisper output.

**Parameters**: `MLXFast.RoPE(x, dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1.0, offset: cache.offset)`

## Python Benchmark (verified on same hardware)

Benchmarked on M3 Max (40 GPU cores) using mlx-audio 0.3.1, mlx 0.30.6:

```
Short text (42 tokens): 22.8 tok/s, RTF=1.83x, progress bar peaks at 26 tok/s
Long text (113 tokens): 20.1 tok/s, RTF=1.60x, progress bar stabilizes at 22-25 tok/s
```

**Python steady-state: ~22-26 tok/s** on this exact machine.
**Swift current: ~18.2-19.4 tok/s** (pending quality fix confirmation).

Python mlx-audio does NOT use `mx.compile()`. Generation loop structure is identical to Swift.
Python creates fresh `code_cache` each step (vs Swift pre-allocate + trim — our approach should be faster).

## Root Cause Analysis

### Original gap: Swift 9.3 → Python 25 tok/s (2.7x)

### Current gap: Swift 18-19 → Python 22-26 tok/s (~1.3x)

### Confirmed: NOT the cause

- **Dtype**: Both BF16 (from safetensors). No conversion in either.
- **`mx.compile()`**: Neither Python nor Swift use it (confirmed by source inspection).
- **Code structure**: Generation loops are identical — same sampling, same eval pattern, same cache handling.
- **Model architecture**: Same 28-layer talker + 5-layer code predictor × 15 passes.

### Confirmed cause: MLX Swift framework overhead

- **~1,250 graph nodes per step** after MLXFast.RoPE optimization (down from ~2,300)
- **GPU utilization improved** but still CPU-bound on graph construction
- Each graph node involves Swift→C++ bridging overhead (function call, ARC refcount, type conversion)
- Python pybind11 wrappers are thinner than Swift's interop mechanism

### Remaining optimization paths

1. **MLXFast.RoPE for talker** — The talker uses MRoPE (3D rotary embeddings) which `MLXFast.RoPE` doesn't directly support. Would need custom approach.
2. **MLXFast.rmsNorm** — Check if `MLXNN.RMSNorm` already uses fused kernel. If not, switching could save nodes.
3. **Reduce remaining graph nodes** — Profile to identify other high-node-count operations.
4. **File mlx-swift issue** — The remaining 1.3x gap is framework overhead.

## Architecture

### Per-Step Operations

1. **Talker forward** (25%) — 28-layer transformer, 1024 hidden, 16 heads, 8 KV heads, 128 head_dim, MRoPE, SwiGLU MLP
2. **Main sampling** (<1%) — top-p with repetition penalty over vocab_size=3072
3. **`.item()` EOS check** (<1%) — unavoidable GPU→CPU sync, 1 per step
4. **Code predictor** (73.6%) — 15 sequential passes through 5-layer transformer, same hidden dims, **MLXFast.RoPE** (fused)
5. **Embedding prep** (~1%) — sum 16 codebook embeddings + text embedding, `eval(inputEmbeds)` + `GPU.clearCache()`

### Token Rate Math

- 12 Hz token rate: each token = 83.3ms of audio
- At 19 tok/s: each token takes 53ms → RTF 0.63x
- At target 24 tok/s: each token takes 42ms → RTF 0.5x

## Key Files

| File                                                   | Purpose                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| `Vendor/.../Qwen3TTS/Qwen3TTSModel.swift`              | Main generation loop, profiling, sampling, timing logs       |
| `Vendor/.../Qwen3TTS/Qwen3TTSCodePredictor.swift`      | Code predictor (5 layers), **MLXFast.RoPE**, called 15×/step |
| `Vendor/.../Qwen3TTS/Qwen3TTSTalker.swift`             | Talker transformer (28 layers), MRoPE, attention             |
| `Vendor/.../Qwen3TTS/Qwen3TTSConfig.swift`             | All config structs with defaults                             |
| `Vendor/.../Qwen3TTS/Qwen3TTSSpeechDecoder.swift`      | VQ decoder, speech tokenizer                                 |
| `tesseract/Features/Speech/SpeechEngine.swift`         | TTSActor, bridges coordinator ↔ model                        |
| `tesseract/Features/Speech/SpeechCoordinator.swift`    | State machine, calls generate()                              |
| `tesseract/Features/Speech/AudioPlaybackManager.swift` | Accumulate-then-play, AVAudioPlayerNode                      |

## How to Test

```bash
scripts/dev.sh dev     # Build + launch
scripts/dev.sh log     # Watch for [speech] timing logs
# Trigger TTS via the app, look for:
# Token generation: N tokens in X.XXs (XX.X tok/s, RTF=X.XXx)

# Enable profiling for per-component breakdown:
QWEN3TTS_PROFILE=1 scripts/dev.sh dev
```

## Branch

`feature/tts-with-vendored-mlx-audio`
