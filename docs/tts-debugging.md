# TTS Streaming Debug Guide

How to capture, visualize, and diagnose audio artifacts in the streaming TTS pipeline.

## Overview

The TTS pipeline generates speech in chunks (one every ~640ms at `emitEvery=8`). Each chunk is crossfaded with its predecessor and written to a shared sample buffer that the audio engine pulls from in real time. The debug system captures every stage of this pipeline for offline analysis.

## Architecture

```
Qwen3-TTS Model
  │  generates speech tokens, emits audio every 8 frames
  ▼
SpeechEngine.streamSpeech()
  │  yields [Float] chunks via AsyncStream
  ▼
AudioPlaybackManager.appendChunk()
  │  prebuffers first 3 chunks, then:
  │  ├─ crossfades with previous chunk's tail (512-sample Hann window)
  │  ├─ applies fade-in on first chunk / after underruns
  │  └─ writes processed samples to AudioSampleBuffer
  ▼
AVAudioSourceNode render block
  │  pulls samples from AudioSampleBuffer at hardware rate (24kHz)
  ▼
speakers
```

## Enabling Debug Dumps

In `AudioPlaybackManager.swift`, the flag is:

```swift
private var debugDumpEnabled = true  // set to false in production
```

When enabled, each streaming session writes a timestamped directory to `/tmp/tesseract-debug/`. The app sandbox entitlement includes `/private/tmp/tesseract-debug/` as a writable path.

## Debug Output Structure

```
/tmp/tesseract-debug/2026-02-07_143827/
├── metadata.json           # Timing and chunk layout info
├── full_stream.wav         # Final audio as scheduled (float32 WAV)
├── raw_chunks/
│   ├── chunk_000.raw       # Raw float32 samples from the model
│   ├── chunk_001.raw
│   └── ...
└── processed_chunks/
    ├── chunk_000.raw       # After crossfade/fade-in processing
    ├── chunk_001.raw
    └── ...
```

### metadata.json

| Field | Description |
|-------|-------------|
| `sampleRate` | Audio sample rate (24000 Hz for Qwen3-TTS) |
| `blendSamples` | Crossfade window size in samples (512 = ~21ms) |
| `totalScheduledSamples` | Total samples written to the audio buffer |
| `chunks[]` | Per-chunk details (see below) |

Each chunk entry:

| Field | Description |
|-------|-------------|
| `index` | Chunk sequence number |
| `rawSamples` | Sample count as received from the model |
| `processedSamples` | Sample count after crossfade/fade-in |
| `scheduledOffset` | Byte offset in the final stream |
| `scheduledSize` | Samples actually written (processed minus held-back tail) |
| `arrivalTimeSec` | Wall-clock time since stream start |

### Raw chunk files

Binary `float32` arrays. Load in Python with:

```python
import numpy as np
chunk = np.fromfile("chunk_000.raw", dtype=np.float32)
```

### full_stream.wav

IEEE float32 mono WAV at 24kHz. This is the exact audio written to the sample buffer (after crossfade, before any hardware resampling). Open in any audio editor or load with scipy:

```python
from scipy.io import wavfile
sr, audio = wavfile.read("full_stream.wav")
```

## Visualization

### Setup (one time)

```bash
cd tesseract
scripts/setup-debug-venv.sh
```

This creates a Python virtual environment at `scripts/.venv/` with numpy, matplotlib, and scipy.

### Running

```bash
scripts/visualize-chunks.sh /tmp/tesseract-debug/<timestamp>/
```

Or directly:

```bash
scripts/.venv/bin/python3 scripts/visualize_chunks.py /tmp/tesseract-debug/<timestamp>/
```

### Output Plots

The script generates four PNGs in the dump directory:

**01_waveform_overview.png** -- Full waveform with red dashed lines at chunk boundaries and orange highlights over crossfade regions. Look for amplitude drops or silence gaps at boundaries.

**02_boundary_zooms.png** -- 100ms zoomed windows around each chunk boundary. Shows the waveform, boundary marker, crossfade region, and amplitude envelope. Discontinuities (clicks/pops) appear as sharp jumps at the red line.

**03_spectrogram.png** -- Frequency-domain view with boundary lines overlaid. Vertical energy smears at boundaries indicate transient artifacts. Gaps in spectral content indicate silence/underruns.

**04_discontinuity_metrics.png** -- Three bar charts per boundary:
- **Max Sample Delta**: amplitude jump at the exact boundary sample. Values above ~0.1 suggest audible clicks.
- **RMS**: energy level around the boundary. Dips indicate silence gaps.
- **ZCR Change**: zero-crossing rate difference before/after boundary. Large values suggest timbral discontinuity.

## Diagnosing Common Issues

### Clicks/pops at boundaries

Check `02_boundary_zooms.png` for sharp waveform jumps at the red boundary line. If the crossfade region (orange) shows smooth blending but the boundary itself has a spike, the crossfade window may be too small or the tail holdback isn't aligning properly.

### Silence gaps between chunks

In `01_waveform_overview.png`, look for flat regions between chunk boundaries. Check `metadata.json` timing: if `arrivalTimeSec[n+1] - arrivalTimeSec[n]` exceeds the chunk's audio duration (`scheduledSize / sampleRate`), the model is generating slower than real-time and the prebuffer is exhausted.

### Repeated fade-in artifacts

If logs show "Applied fade-in after buffer underrun" on every chunk, the sample buffer is being drained between chunks. The prebuffer count may need increasing, or `emitEvery` needs tuning for the hardware's generation speed.

### Timing analysis from metadata

```python
import json
meta = json.load(open("metadata.json"))
chunks = meta["chunks"]
sr = meta["sampleRate"]

for i in range(1, len(chunks)):
    gen_time = chunks[i]["arrivalTimeSec"] - chunks[i-1]["arrivalTimeSec"]
    audio_dur = chunks[i]["scheduledSize"] / sr
    deficit = gen_time - audio_dur
    print(f"Chunk {i}: gen={gen_time:.3f}s audio={audio_dur:.3f}s deficit={deficit:+.3f}s")
```

A positive deficit means the model is slower than real-time for that chunk. Sustained positive deficits will eventually drain the prebuffer.
