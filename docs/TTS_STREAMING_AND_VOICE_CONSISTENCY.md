# TTS Streaming & Voice Consistency

How Tesseract generates and plays back long-form speech with consistent voice across hundreds of segments.

## Architecture Overview

```
User text
    |
    v
TextSegmenter          Split into sentence-level segments
    |
    v
SpeechCoordinator      Orchestrates segment-by-segment generation
    |
    v
SpeechEngine           MainActor facade + TTSActor (actor isolation)
    |
    v
Qwen3TTSFullModel      Token generation (talker + code predictor)
    |
    v
SpeechTokenizer        Codec tokens -> PCM audio (24kHz)
    |
    v
AudioPlaybackManager   Push-based AVAudioPlayerNode streaming
```

## Streaming Pipeline

### Text Segmentation

`TextSegmenter` splits input text into segments at sentence boundaries. A text is considered "long-form" when it exceeds a threshold (multiple sentences). Each segment is generated independently as a streaming TTS call.

### Streaming Token Generation

Each segment runs through `generateStreamingVoiceDesign()`, which produces audio incrementally:

1. **Token generation loop** — the talker transformer produces one codec "step" per iteration (16 codebook tokens via the code predictor)
2. **Sliding-window decode** — every N steps, a window of recent codec tokens is decoded to PCM audio via the speech tokenizer
3. **Hann crossfade** — adjacent decoded chunks overlap and are crossfaded to eliminate boundary artifacts
4. **Chunk emission** — audio chunks are yielded through an `AsyncThrowingStream` to the coordinator

The two-phase strategy minimizes first-chunk latency:
- **Phase 1** (first ~48 tokens): emit every 5 tokens with a 48-token decode window
- **Phase 2** (steady state): emit every 8 tokens with an 80-token decode window

### Playback

`AudioPlaybackManager` operates in push-based streaming mode:
- `startStreaming(sampleRate:)` initializes an `AVAudioPlayerNode` with a ring of buffers
- `appendChunk(samples:)` fills buffers and schedules them on the audio engine
- Playback begins as soon as the first buffer is ready — no waiting for full generation

## Voice Consistency: The Voice Anchor

### The Problem

With VoiceDesign mode (text-based voice descriptions like "a calm, soothing male voice"), each segment is generated independently. The voice description KV cache captures the *intent* of the voice, but the actual voice characteristics (pitch, timbre, prosody) emerge from stochastic codec token sampling. Different text inputs + random sampling = slightly different voice each time.

Over hundreds of segments (e.g., a 527-segment audiobook), this causes noticeable voice drift.

### The Solution: Codec Prompt Pre-fill

After the first segment generates successfully, we save its codec tokens and use them to build an extended KV cache — the **voice anchor**. This cache contains both the instruct conditioning (voice intent) AND codec-level examples of the actual generated voice (voice realization).

```
Without anchor:  [instruct KV] → generate (voice varies per segment)
With anchor:     [instruct KV + codec prompt KV] → generate (voice stays consistent)
```

### How It Works

#### Segment 1: Normal Generation
```
Voice description → instruct KV cache → generate tokens → save codec codes
```

#### Building the Anchor (after segment 1)
```
1. Restore instruct-only KV cache into fresh cache
2. Take first 48 codec steps (~4s of audio at 12Hz)
3. For each step, embed all 16 codebooks:
   - code0: talker input embedding
   - codes 1-15: code predictor codebook embeddings (summed)
   - Add ttsPadEmbed (matches generation pattern when text is exhausted)
4. Concatenate all 48 step embeddings → forward through talker
5. Save extended KV state = instruct + codec prompt
```

#### Segments 2+: Anchored Generation
```
1. Restore voice anchor KV cache (instruct + codec prompt)
2. Generate normally — model sees concrete voice examples in its attention context
3. RoPE handles position offset gracefully
```

### Why This Works

During generation, the model produces embeddings of the form `textEmbed + codecEmbed` (or `ttsPadEmbed + codecEmbed` when text tokens are exhausted). The codec embeddings carry voice characteristics that the model's causal attention can leverage. By pre-filling the KV cache with codec embeddings from a real generation, subsequent segments have concrete voice examples to match — not just a text description of what the voice should sound like.

### Implementation Details

**Files involved:**

| File | Role |
|------|------|
| `Vendor/.../Generation.swift` | Protocol: `lastGeneratedCodes`, `buildVoiceAnchor()`, `clearVoiceAnchor()`, `generateStream(..., useVoiceAnchor:)` |
| `Vendor/.../Qwen3TTSModel.swift` | Core: anchor KV state management, codec embedding, cache save/restore |
| `tesseract/.../SpeechEngine.swift` | Actor bridge: delegates anchor operations to model |
| `tesseract/.../SpeechCoordinator.swift` | Orchestration: builds anchor after segment 1, passes flag for segments 2+ |

**State on `Qwen3TTSFullModel`:**
- `voiceAnchorKVState: [[MLXArray]]?` — saved KV cache (instruct + codec prompt)
- `voiceAnchorCodecCount: Int` — number of codec steps in the anchor
- `lastGeneratedCodes: [MLXArray]?` — codes from most recent generation

**Memory cost:** ~5.4 MB for the anchor KV cache (28 transformer layers, 48 extra token positions) — negligible.

### Lifecycle

| Event | Action |
|-------|--------|
| First segment completes | `buildVoiceAnchor(referenceCount: 48)` |
| Segments 2+ | `generateStream(..., useVoiceAnchor: true)` |
| User stops | `clearVoiceAnchor()` — frees anchor state |
| User pauses/resumes | Anchor persists; resumed segments use it |
| Short first segment (<48 tokens) | Anchor uses whatever codes are available |
| Single-segment text | No anchor built — existing behavior unchanged |

### Log Messages

```
[speech] Building voice anchor from segment 1
[speech] Voice anchor built: 48 codec steps (4.0s) from first segment
[speech] Voice anchor cache restored (48 codec steps)
[speech] Starting segment 3/527 (with voice anchor)
[speech] Voice anchor cleared
```

### Performance Impact

From production logs (527-segment audiobook, M3 Max):

| Metric | Without anchor (seg 1) | With anchor (seg 2+) |
|--------|----------------------|---------------------|
| Token rate | 16.2 tok/s | 18.8-21.3 tok/s |
| RTF | 0.74x | 0.56-0.64x |

The anchor adds 48 tokens to the KV cache context but does not measurably slow generation — in fact, segments 2+ are faster because the model has richer context and reaches EOS sooner (shorter sequences for the same content).

## Non-Breaking Design

All new parameters default to current behavior:
- `useVoiceAnchor: false` by default on all `generateStream` calls
- Protocol extensions provide default no-op implementations for `buildVoiceAnchor()` and `clearVoiceAnchor()`
- If `voiceAnchorKVState` is nil, generation falls through to the existing instruct-only path
- Single-segment and non-streaming generation are completely unchanged
- Other model conformers (LlamaTTS, MarvisTTS, etc.) are unaffected

## KV Cache Strategy Summary

The model maintains two levels of KV cache:

```
Level 1: Voice Prefix Cache (instruct-only)
  - Cached per voice description string
  - Reused across all segments with the same voice
  - Saves re-computing instruct tokens (~100-200 tokens)

Level 2: Voice Anchor Cache (instruct + codec prompt)
  - Built from first segment's generated codes
  - Extends Level 1 with 48 codec token positions
  - Gives the model concrete voice examples to match
  - Cleared on stop, persists across pause/resume
```

Both caches use the same save/restore pattern: snapshot `KVCacheSimple.state` (which stores keys/values per layer), `eval()` to materialize, then restore by setting `.state` back (which auto-updates the offset from key dimensions).
