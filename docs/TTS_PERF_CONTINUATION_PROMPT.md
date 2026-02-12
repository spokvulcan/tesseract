# TTS Performance Optimization — Continuation Prompt

Copy this as a prompt for a fresh Claude Code session.

---

## Task

Continue optimizing TTS token generation speed. Current: 10.6-11.1 tok/s. Target: ~24 tok/s (Python mlx-audio speed).

## Context

Read `docs/TTS_PERFORMANCE_INVESTIGATION.md` for the full investigation, profiling data, what was tried, and what didn't work.

Key facts:
- **Bottleneck**: Code predictor (73.6% of step time) — 15 sequential 5-layer transformer passes per generation step
- **GPU utilization**: Only 65% on M3 Max → CPU-bound on MLXArray graph construction
- **Root cause**: ~2,300 graph nodes per step. Swift MLX per-op overhead (~20-30μs/node) compounds to 46-69ms
- **Python parity**: Both use identical code structure, BF16 dtype, no `mx.compile()`. The gap is at MLX Swift framework level.

## What's already optimized

1. Profiling off by default (was adding 18 eval syncs/step). Enable: `QWEN3TTS_PROFILE=1`
2. Sync `eval(inputEmbeds)` beats `asyncEval` by ~15% (GPU memory contention)
3. `GPU.clearCache()` after eval frees code predictor memory for talker
4. Code predictor KV cache: `step=16` (vs 256), pre-allocated once, reused via `trim()`

## Unexplored paths to investigate (in priority order)

### 1. Verify Python benchmark (FIRST!)
Install Python mlx-audio and benchmark on THIS machine (M3 Max, 40 GPU cores). The 24 tok/s claim may be from different hardware. If Python gets ~15 tok/s here, we're much closer than we think.

```bash
pip install mlx-audio
# Run the Qwen3-TTS benchmark with same model: mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
```

### 2. `compile()` for talker forward pass
The talker is 28 layers, called once/step, KV cache grows monotonically (never reset). This makes it a better candidate for `compile(inputs:outputs:)` than the code predictor (whose cache resets every step). This could reduce graph construction time for the 25% of step time spent in the talker.

Swift MLX compile API: `compile(inputs: [cache], outputs: [cache], shapeless: false) { ... }`
KVCacheSimple conforms to Updatable. The cache grows by 1 token each step — check if compile handles this shape change.

### 3. Pre-compute code predictor RoPE
Positions 0-16 are always the same for the code predictor. Pre-computing cos/sin values once and slicing per-pass saves ~105 graph nodes/step (~5% reduction).

### 4. Reduce code groups (quality/speed tradeoff)
Skip later codebooks (e.g., only predict 8 of 16). Later codes represent finer audio details. This would nearly halve code predictor time. Test quality degradation by setting `numCodeGroups` override.

### 5. Inline code predictor sampling
Replace `sampleToken()` call (which checks suppress/repetition penalty) with inline `categorical(logits[0, -1] / temperature).reshaped(1, 1)` for the 15 code predictor passes. Saves function call overhead and unnecessary conditional checks.

### 6. File mlx-swift issue
The 2x performance gap between Swift and Python MLX bindings (same model, same architecture, same eval pattern) may be a known issue. Check https://github.com/ml-explore/mlx-swift/issues.

## Key files
- `Vendor/mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSModel.swift` — generation loop (line ~180)
- `Vendor/mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSCodePredictor.swift` — code predictor
- `Vendor/mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSTalker.swift` — talker

## Build & test
```bash
scripts/dev.sh dev   # Build + kill + relaunch
scripts/dev.sh log   # Tail logs, look for [speech] Token generation: ... tok/s
```
