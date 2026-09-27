# ADR-0074: Qwen3-TTS decodes exactly as Qwen does, renders frames pipelined, and reads its text table from disk

- Status: Accepted
- Date: 2026-09-27
- Relates to: ADR-0071 (first-party Qwen3-TTS, upstream as reference),
  ADR-0072 (Reference Take, sampler), ADR-0039 (residency and memory),
  ADR-0075 (the codec's conv stack on the Neural Engine)

## Context

ADR-0071 made the Qwen3-TTS model code ours so it could be optimized and
checked against Qwen's own implementation. Both were overdue. Checked against
Qwen's official `qwen_tts` package (PyTorch, fp32), with teacher forcing on
recorded frames and the same weights, the talker and the code predictor
matched to bf16 noise, but the codec decoder did not.

- **The streaming decoder was 23 to 30 dB off Qwen's decoder** on the same
  frames (3 to 7 % waveform error), for two reasons both MLX ports share:
  - Qwen's decoder transformer attends over a sliding window of 72 frames
    (`sliding_window: 72`, "all layer in code2wav should be sliding
    attention"). Ours attended over the whole segment.
  - Streaming a transposed conv carried its overlap tail with the bias in it,
    so every chunk boundary added the bias twice. This is why the audio
    changed with the chunk size.
- **Qwen's generate suppresses EOS for the first two frames**
  (`min_new_tokens = 2`); ours didn't.
- **A CustomVoice dialect** replaced the language only for Chinese or auto in
  Qwen's code; ours replaced it always.
- **Top-k:** transformers keeps every logit at least the k-th largest, so ties
  at the cut stay. Ours kept exactly k.

Speed and memory had room too. A q6 frame (80 ms of audio) took 21.5 ms: 6.5
talker, 12.3 code predictor (15 short sequential passes, dispatch-bound),
2.7 decoder. The process peaked at a 5.6 GB footprint for 2.35 GB of weights.
A third of the talker file is the 151,936-row text embedding table, 622 MB of
bf16 that a prompt reads a few hundred rows of.

## Decision

**The decoder is exact.** Every layer of Qwen's decoder is causal, so a
stream decoded a chunk at a time, carrying the right state, equals the one-pass
decode.
- The transformer carries the last 71 keys and values per layer, so memory is
  bounded, and masks by window.
- Each transposed conv runs in polyphase form: a two-tap stride-1 conv (x[t-1],
  x[t]) to stride × out channels, whose channels-last output is already the
  upsampled sequence. This is exact for Qwen's kernels (s and 2s). It skips the
  zeros MLX's transposed conv multiplied and allocated, and its only state is
  the last input frame.
- Measured against Qwen's decoder on the golden frames:
  - fp32: 114 to 120 dB at every chunk size (1, 5, 25 and one pass).
  - fp16, which ships: 55 to 59 dB, at half the memory. bf16 was 46 dB.
- The chunk size no longer changes the audio, so it is free to choose. The
  first chunk is 3 frames, for time to first audio, and later ones 5.

**The sampler follows Qwen's processors.** EOS is suppressed for two frames,
top-k keeps ties, and the dialect rule is Qwen's. Two things differ, both on
purpose:
- The repetition window of 64 frames (ADR-0072).
- A frame cap of six per text token, as a safety bound.

The sampler's state (the repetition window) lives on the GPU. The loop never
waits for a token to decide the next graph.

**Frames run pipelined.** Each frame is two GPU dispatches:
- A samples the first code and runs the code predictor.
- B runs the talker step that appends to the KV cache.

B is encoded only after the previous B finishes, because MLX copies a buffer an
unfinished command buffer still reads, which here was the whole KV cache. A
keeps the GPU busy meanwhile. The end-of-speech check reads the frame before,
so the CPU builds frame t+1 while the GPU runs frame t.

**Fewer kernels per pass.** A frame ran about two thousand small kernels,
each costing a few microseconds however little it did.
- q/k/v and gate/up are stacked into single projections at load. This is
  exact, because quantization groups run along the input dimension; a
  mixed-precision checkpoint keeps them separate.
- RoPE is MLX's fused kernel. The talker's interleaved M-RoPE is plain RoPE
  when all three position axes are equal, which they are in TTS.
- Attention uses the `.causal` mode, so no mask is built.
- For one position (every talker step and code-predictor pass), three Metal
  kernels replace chains of MLX ops (`Qwen3TTSKernels`):
  - q/k RMSNorm with RoPE;
  - the residual add with the next RMSNorm;
  - the top-k categorical draw (about fourteen ops before).
- Each computes what the ops it replaces compute, bit for bit: the same
  reduction order, casts, math functions and random draw. They are compiled
  with fast math, as the package's MLX kernels are (the fork's `fastmath_`
  kernel names). Compiled without it, `exp2` in the rotary frequencies came
  out an ulp away now and then. One value in 100,000 to 200,000 landed a bf16
  step off, enough to change a render's codes within a few hundred frames.
  `qwen3-tts-bench --mode audit` checks every call against the ops on real
  data.

**Memory stays close to the weights.**
- The text embedding table stays on disk: rows are read with `pread` as a
  prompt needs them (`Qwen3TTSTextEmbedding`). −622 MB.
- The decoder is fp16. Its codebooks are folded through their output
  projections into one gather table, and the conv weights are made contiguous
  once: a strided weight was copied on every conv call.
- The talker's KV cache is kept across a segment's generations and rewound,
  in steps of 256 positions. MLX reuses a freed buffer only for a request of
  nearly its size, so a fresh cache per segment piled up in the pool. It is
  released at utterance end (`releaseWorkingMemory`).
- Loading converts the decoder weights one tensor at a time and stacks the
  talker's projections a layer at a time.
- A prompt of 32 positions or more is prefilled a layer at a time
  (`Qwen3TTSTalker.prefill`): each layer is queued as it is built, and the
  one before it waited for.
  - The GPU never idles, and command buffers in flight hold two layers'
    activations instead of about ten.
  - MLX's pool keeps those buffers, at sizes only that prompt length reuses.
    A new length left about 250 MB there; now about 75 MB.
  - Shorter prompts stay one graph: their activations are small, and the
    waits would cost first-audio time.

**One generation at a time.** The kept KV caches are shared state. A
generation holds a lock for as long as it uses them, and so does priming a
voice. A cancelled stream's generation still finishes its frame after its
consumer and the engine's GPU lease are gone. The next generation waits for
it, at most a frame, instead of rewinding caches it is still writing.

**Both of Qwen's text layouts exist; the engine's default is unchanged.** Qwen's
`generate_voice_design` defaults to `non_streaming_mode=True`, with the whole
text in the prompt. The engine, like both MLX ports, has always fed the text
one token per frame (Qwen's streaming mode). `Qwen3TTSTextLayout.upfront` is
Qwen's default. It needs a listening A/B before it becomes ours, so it isn't
the default yet. The Reference Take prompt was already Qwen's non-streaming
in-context layout (ADR-0072).

## Consequences

Measured on the M3 Max, q6, seed 42. Engine numbers come from v2-listen with
the MLX decoder: longform reads the 357-word story in six segments, and
pinned renders companion lines. Model numbers come from qwen3-tts-bench.

| | Before | After |
|---|---|---|
| Real-time factor, engine longform | 0.270 | 0.153 |
| First audio, first segment | 251 ms | 54 ms |
| First audio, segment continuing a take | about 200 ms | 119 to 136 ms |
| First audio, companion line | 141 to 154 ms | 53 to 79 ms |
| Time per frame, model bench | 20.7 ms | 11.3 ms |
| Talker step, model bench | 6.5 ms | 5.6 ms |
| Code-predictor frame, model bench | 12.3 ms | 9.8 ms |
| Decoder per frame, 5-frame chunks, model bench | 2.7 ms | 1.3 ms |
| MLX memory after load | 2,352 MB | 1,556 MB |
| Peak process footprint, longform | 5.57 GB | 2.78 GB |
| Peak resident memory, longform | 2.65 GB | 0.95 GB |
| Decoder against Qwen's | 23 to 30 dB | 55 to 59 dB (fp16) |

- Same seed, same code: the seeds of old renders produce new ones. Numerics
  changed (fused RoPE in fp32, stacked projections, top-k ties). Teacher
  forcing against Qwen's fp32 talker puts the new code where the old was:
  talker KL about 5e-4 and top-1 agreement of 96 to 100 %, code predictor KL
  about 7e-3 and 92 to 94 %. That is the bf16 floor.
- The fused kernels and the layer-at-a-time prefill change no value. With
  them on or off, the teacher-forced logits are byte-identical and a render's
  codes are the same.
- The decoder's audio changed, towards Qwen's. Listening: the owner should
  hear the fix most around chunk boundaries and in segments past about 6 s.
- The MLX buffer pool still grows when a segment brings a new prompt size,
  now by about 75 MB a segment instead of about 250. The engine's one clear
  per utterance end (ADR-0039) and the app's 2 GB cache limit bound it.
- `AudioGeneration.audio` carries samples (`[Float]`), not an `MLXArray`.
- The tests pin the fixes: chunk invariance, the window, the sliding mask,
  top-k ties, min frames, the dialect rule, both layouts, the file-backed
  table, and stacked against separate projections. They also pin each fused
  kernel bit for bit, the layer-at-a-time prefill, and overlapping
  generations. qwen3-tts-bench records golden frames and teacher-forced
  logits for the next change to check against.

## Considered and rejected

- Keeping the old decoder's behaviour for continuity: it was Qwen's model
  decoded wrong. The fix brings the audio closer to what the model was
  trained to produce.
- Quantizing the text table to 8 bits (−290 MB): reading it from disk saves
  all 622 MB and changes no number.
- bf16 for the decoder: 46 dB against fp16's 55 to 59, for the same memory.
- A per-segment `Memory.clearCache()`: it would drop the LLM's pooled buffers
  too. Rewinding the KV cache removed most of the growth at its source.
- Prefilling in fixed-size chunks, the last one padded: a new length would
  leave only about 5 MB in the pool. But each chunk streams the weights
  again, making prefill 20 to 50 % slower, and that lands on a
  reference-take segment's first audio.
- Waiting after every prefill layer: about 45 MB per new length, but 7 ms
  more per prefill. Two layers in flight cost nothing.
- The fused kernels compiled without fast math: close to MLX's ops but not
  bit-exact (above), so a kernel change couldn't be checked by comparing
  codes.
