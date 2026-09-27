# ADR-0075: The codec's conv stack runs on the Neural Engine, built on device from the checkpoint

- Status: Accepted
- Date: 2026-09-27
- Relates to: ADR-0074 (exact streaming decoder, polyphase upsamplers),
  ADR-0039 (residency, the GPU lease), ADR-0073 (storage roots)

## Context

Everything in Qwen3-TTS ran on the GPU, one queue shared with the LLM. The
owner asked for Apple Neural Engine inference. A research pass measured the
options on the M3 Max (the notes are in the session's scratchpad `ane-research.md`):

- **Talker and code predictor: no.** Batch-1 decode is limited by how fast the
  weights stream. On this machine the ANE manages about 117 GB/s, the GPU about
  379 GB/s. Estimates put the talker at 10 to 14 ms a step on the ANE against
  5.6 on MLX, and the code predictor at 20 to 35 ms a frame against 10. Three
  public ports hit fp16 overflow or validation failures there. The two alternate
  strictly within a frame, so moving them adds no parallelism.
- **Codec decoder: yes.** It is convolutions at fixed shapes, with fp16
  headroom: the largest activation is about 342. The conv stack, everything
  after the decoder's transformer, is 98 % of its work. On the ANE it runs
  alongside the GPU. A kernel streaming 512 MiB per call (roughly a talker
  step) slowed the ANE codec by 7 %, and the same codec on Core ML's GPU path
  by 7.8×.
- **The route: public Core ML.** maderix/ANE, a set of reverse-engineered
  private APIs, is a training harness with no stability contract and a
  compile leak. It saves about 0.25 ms of dispatch, which doesn't matter for
  one call per chunk.
- **Nothing downloadable exists.** No converted Qwen3-TTS codec model fits our
  checkpoint and streaming state. The app is offline and downloads only what
  the Model Catalog lists.

## Decision

**The conv stack runs on the Neural Engine through Core ML.** The front end
stays in MLX: codebooks, the pre-conv, and the windowed transformer (2 % of the
work, and the most fp16-sensitive). Per chunk, it hands a `[frames, 1024]`
latent to `Qwen3TTSNeuralCodec`, which decodes it on a serial queue beside the
GPU loop.

**The Core ML model is built on this machine from the checkpoint's weights.**
The package writes an ML Program itself: the Model and MIL protobuf, a MILBlob
v2 weight file and the `.mlpackage` layout (`MLProgramBuilder`). It holds only
what this one fixed graph needs. The graph:
- fp16, in the Neural Engine's `[1, C, 1, W]` layout, 3 frames per call (at
  most 8, the M1–M3 width limit).
- Polyphase upsamplers, whose rewrite the ANE needs anyway: the stride-5
  transposed conv otherwise falls back to the CPU.
- Every causal conv's context passed in and out as explicit state: 20 small
  tensors. MLState failed on the ANE for another streaming codec.

`MLModel.compileModel` compiles it once, into Caches keyed by the graph
version, the chunk size, the decoder config and the checkpoint files' sizes
and dates. The builder writes each weight to the package's blob file as it
converts it, so the build holds one weight at a time, not the 140 MB file. Core ML's own cache keeps the ANE specialization per OS build.
Measured costs:
- Cold build: 4.8 s, in the background.
- Cached load: 0.1 s.
- On disk: 140 MB.

**It is used only where it is right.** At load:
- `MLComputePlan` must put every op on the Neural Engine. On M1/M2, `sin`
  falls to the CPU, so they stay on MLX.
- Its audio must be within 35 dB of the MLX conv stack's, on a probe that
  crosses a chunk boundary.

Until it is ready, and wherever it fails, MLX decodes. If a call fails
mid-stream, that segment fails and later ones use MLX.

**The GPU lease holds.** Building converts weights on MLX's CPU stream, and the
check's MLX half is made during warm-up, inside the lease. The Neural Engine
itself is outside the lease: it doesn't compete with the LLM's GPU work.

**Only one copy of the conv stack.** Once a generation runs on the Neural
Engine, MLX's conv-stack weights are released (−140 MB). The next MLX
synthesis reads them back from the checkpoint.

**Opt-in, with a cache directory.** `Qwen3Synthesizer` moves the conv stack
only when it is given a `neuralEngineCache`. The app passes one under
`StorageEnvironment.caches` (ADR-0073). Without one (tests, other callers) the
synthesizer stays on MLX and writes nothing.

## Consequences

Measured with the q6 checkpoint on the M3 Max:

| | MLX conv stack | Neural Engine conv stack |
|---|---|---|
| Against Qwen's fp32 decoder, golden frames | 55 to 59 dB | 50 to 53 dB |
| Per call (3 frames) | on the GPU queue | 5.6 ms, beside it |
| Real-time factor, engine longform | 0.153 | 0.148 |
| First audio, first segment | 54 ms | 60 ms |
| First audio, companion line | 53 to 79 ms | 59 to 81 ms |
| Peak footprint, engine longform | 2.72 GB | 2.66 GB |
| MLX memory with weights | 1,556 MB | 1,449 MB |

- The GPU no longer runs the conv stack, about 1.3 ms of work per 80 ms
  frame. With the TTS alone that is 3 % of the time, since the frame loop
  already overlaps most of the decoding. The bigger gain comes when the LLM
  is also on the GPU: the codec no longer queues behind it.
- First audio comes about 6 ms later: the first chunk waits for its Neural
  Engine call.
- The audio stays within 5 dB of the MLX path, and 20 to 30 dB closer to Qwen
  than the decoder it replaced (ADR-0074).
- Chunks on the Neural Engine are 3 frames each, the first included. The last
  chunk of a stream is zero-padded; being causal, padding changes only state
  nothing reads after it. Each stream carries its own conv contexts
  (`Qwen3TTSNeuralCodec.Stream`), so a cancelled stream's queued chunks never
  touch the next one's.
- A `Qwen3TTSTests` case compiles a tiny codec through Core ML (on the CPU, so
  CI needs no Neural Engine) and checks it against MLX. That pins the
  protobuf, the blob format, the polyphase shuffle and the carried state.
- The builder emits only the ops this graph uses. A new decoder shape means
  new emitter code, not a converter.

## Considered and rejected

- The talker or code predictor on the ANE: slower there (above). Revisit only
  if keeping the GPU free matters more than TTS speed.
- maderix/ANE's private APIs: see the context above.
- Converting with coremltools and hosting the `.mlpackage`: a Python pipeline
  and a second download to keep in step with the checkpoint, for a graph small
  enough to emit directly.
- One Core ML function per chunk size (multifunction): each is compiled and
  kept on the ANE separately. Padding the last chunk makes one enough.
