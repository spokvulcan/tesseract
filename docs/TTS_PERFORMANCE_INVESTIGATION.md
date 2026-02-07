# TTS Performance Investigation

## Current State (2026-02-07)

### Playback: SOLVED
Audio playback quality is perfect. `AudioPlaybackManager` accumulates all chunks, plays as one-shot via `AVAudioPlayerNode` with 5ms fade-in.

### Generation Speed: THE REMAINING PROBLEM

**Latest benchmarks:**
```
Token generation: 104 tokens in 9.79s (10.6 tok/s, RTF=1.13x)
Token generation: 387 tokens in 34.96s (11.1 tok/s, RTF=1.08x)
```

- **Current**: 10.6-11.1 tok/s → RTF 1.08-1.13x (barely real-time)
- **Target**: ~24 tok/s → RTF ~0.5x (Python mlx-audio)
- **Gap**: ~2.2x slower than Python

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

**Note**: Profiling mode is ~20% slower than normal due to the eval() sync points preventing lazy graph fusion. Normal mode: ~93ms/step (10.6 tok/s).

## Optimizations Applied (this session)

| # | Optimization | tok/s | Improvement |
|---|-------------|-------|-------------|
| 0 | Baseline (before session) | 9.3 | — |
| 1 | Profiling off by default (was adding 18 eval() syncs/step) | 10.7 | +15% |
| 2 | sync `eval()` beats `asyncEval()` — GPU memory contention | 10.7 | confirmed |
| 3 | `GPU.clearCache()` AFTER eval (frees code predictor memory for talker) | 10.6 | confirmed |
| 4 | KV cache: `step=16` (vs 256), pre-allocate once, reuse via `trim()` | 10.6-11.1 | +0-5% |

### What we tried that DIDN'T help:
- **`asyncEval(inputEmbeds)`**: 9.3 tok/s (worse than sync eval's 10.6). GPU memory contention — async leaves code predictor memory allocated while talker runs.
- **`GPU.clearCache()` before eval**: No-op — lazy graph still holds tensor references.
- **`compile()` for code predictor**: Not feasible. KV cache `offset` is a Swift `Int`, not tracked by MLX compilation. The compiled graph would use stale offsets on replay.

## Root Cause Analysis: Swift 11 tok/s vs Python 24 tok/s

### Confirmed: NOT the cause
- **Dtype**: Both BF16 (from safetensors). No conversion in either.
- **`mx.compile()`**: Neither Python nor Swift use it.
- **Code structure**: Generation loops are identical — same sampling, same eval pattern, same cache handling.
- **Model architecture**: Same 28-layer talker + 5-layer code predictor × 15 passes.

### Likely cause: MLX Swift framework overhead
- **~2,300 MLXArray graph nodes per step** (5 layers × ~150 ops × 15 code predictor passes)
- **GPU only 65% utilized** on M3 Max → CPU-bound on graph construction
- Each graph node involves Swift→C++ bridging overhead (function call, ARC refcount, type conversion)
- Python pybind11 wrappers are likely thinner than Swift's interop mechanism
- The per-node overhead compounds: even 20-30μs × 2,300 nodes = 46-69ms overhead/step

### Unexplored optimization paths
1. **Verify Python benchmark on same hardware** — is 24 tok/s real on M3 Max, or from different hardware?
2. **`compile()` for the talker forward pass** (28 layers, called once/step) — KV cache grows monotonically so may work with `inputs`/`outputs` tracking
3. **Reduce code groups** — skip later codebooks (quality/speed tradeoff). Later codes represent finer audio details.
4. **Pre-compute RoPE for code predictor** — positions 0-16 are always the same, saves ~105 graph nodes/step
5. **Inline code predictor sampling** — avoid `sampleToken()` call overhead for the simple temperature-only path
6. **Metal Performance Shaders** — bypass MLX for the code predictor's tight loop
7. **File an issue with mlx-swift** — the 2x gap vs Python bindings may be a known issue

## Architecture

### Per-Step Operations

1. **Talker forward** (25%) — 28-layer transformer, 1024 hidden, 16 heads, 8 KV heads, 128 head_dim, MRoPE, SwiGLU MLP
2. **Main sampling** (<1%) — top-p with repetition penalty over vocab_size=3072
3. **`.item()` EOS check** (<1%) — unavoidable GPU→CPU sync, 1 per step
4. **Code predictor** (73.6%) — 15 sequential passes through 5-layer transformer, same hidden dims, standard RoPE
5. **Embedding prep** (~1%) — sum 16 codebook embeddings + text embedding, `eval(inputEmbeds)` + `GPU.clearCache()`

### Token Rate Math
- 12 Hz token rate: each token = 83.3ms of audio
- At 10.6 tok/s: each token takes 94ms → RTF 1.13x
- At target 24 tok/s: each token takes 42ms → RTF 0.5x

## Key Files

| File | Purpose |
|------|---------|
| `Vendor/.../Qwen3TTS/Qwen3TTSModel.swift` | Main generation loop, profiling, sampling, timing logs |
| `Vendor/.../Qwen3TTS/Qwen3TTSCodePredictor.swift` | Code predictor (5 layers), called 15×/step |
| `Vendor/.../Qwen3TTS/Qwen3TTSTalker.swift` | Talker transformer (28 layers), MRoPE, attention |
| `Vendor/.../Qwen3TTS/Qwen3TTSConfig.swift` | All config structs with defaults |
| `Vendor/.../Qwen3TTS/Qwen3TTSSpeechDecoder.swift` | VQ decoder, speech tokenizer |
| `tesseract/Features/Speech/SpeechEngine.swift` | TTSActor, bridges coordinator ↔ model |
| `tesseract/Features/Speech/SpeechCoordinator.swift` | State machine, calls generate() |
| `tesseract/Features/Speech/AudioPlaybackManager.swift` | Accumulate-then-play, AVAudioPlayerNode |

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
