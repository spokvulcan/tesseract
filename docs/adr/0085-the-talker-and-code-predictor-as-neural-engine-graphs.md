# ADR-0085: The talker and the code predictor run as Neural Engine graphs we write, int8 with measured fp16 outliers

- Status: Accepted on the Mac (2026-10-03); the iPhone 16 Pro Max check is
  slice 3's remaining half of #515
- Date: 2026-10-03
- Relates to: ADR-0084 (the iPhone voice runs on the Neural Engine), ADR-0075
  (the codec's conv stack there, and `MLProgramBuilder`), ADR-0074 (the parity
  measures), ADR-0077 (the Alignment Head), ADR-0072 (two temperatures)

## Context

ADR-0084 moves Qwen3-TTS 0.6B's talker and code predictor to the Neural
Engine on the phone, with graphs our own builder writes. ADR-0075 had
estimated them there at 10 to 14 ms a talker step and 20 to 35 ms a code
predictor frame on the M3 Max, and noted that public ports hit fp16
overflow. #515 asked for the code predictor's 15 passes in one call with
sampling inside, the talker's KV cache kept in place as Core ML state, the
Alignment Head's row as an output, 8-bit weights to start, and parity
checked the way ADR-0074 checked MLX against Qwen.

Measuring on the M3 Max (macOS 27.0.1, the 0.6B CustomVoice 8-bit
checkpoint, speaker ryan) settled the open details:

- **Weight format.** Of the formats Core ML 8 places on the Neural Engine,
  int8 with one fp16 scale per output row streams fastest. Finer blocks or
  zero points send the convs to the CPU. A 256-entry palette runs there too,
  but matches the checkpoint's weights 5 to 28 dB worse. fp16 weights cost 1.6 to 1.8×
  the time (talker step 9.5 ms against 6.0, code predictor frame 21.5 against
  12.1) and gained little parity.
- **Massive activations.** One MLP channel in the code predictor's third
  layer reaches about 150,000 and the residual stream about 50,000 (the
  talker's peaks are about 1,400 and 1,100). Through that channel, int8
  rounding of the down projection swamped everything else: the code
  predictor's logits came out at 10 dB against MLX's, its best code agreeing
  25 % of the time. That one matrix in fp16 restored 33 dB; every down
  projection in fp16 added a dB or two.
- **Scaling hurts here.** Running the residual stream scaled down (folded
  into the weights, so the norms don't see it) dropped parity to 13 dB even
  with fp16 weights: its small values fall below fp16's normal range, and
  the Neural Engine loses them. Unscaled, the Neural Engine carries the
  150,000 product through its fused ops (the CPU, computing the same program,
  overflows to NaN).
- **Ops that leave the Neural Engine.** `topk`, `argmax` and `gather` run on
  the CPU; a top-k found by bisection is a dozen dependent steps per pass.

## Decision

**Both graphs are written by `MLProgramBuilder` from the checkpoint, on the
device, as Core ML 8 programs.** They are compiled once and cached under a
key naming the graph version, the context length, the Alignment Head, the
measured precision and the checkpoint files. At load, Core ML's compute plan
must put every op on the Neural Engine (`.cpuAndNeuralEngine`).

- **The talker runs one position per call.** Its KV cache is two Core ML
  states, `[1, layers · kv · head_dim, 1, L]` with L = 512 positions per
  generation. The graph attends to the cache, masked to the positions written
  so far, and to its own key and value beside it, then returns them. The host
  writes them into the states at the position, so no call copies or rewrites
  the cache. The Alignment Head's scores come out with every step.
- **The code predictor runs a whole frame per call.** Its 15 passes are
  unrolled with their keys and values in the graph. Each pass draws its code
  in the graph: Gumbel-max over the top 50 on noise the host supplies, the
  cut read off two grids of 32 thresholds at once (ties at the cut stay, as in
  MLX), a tie for the best going to the lowest code. The code's embedding is
  a one-hot vector times the transposed table. The call returns the codes and
  the frame's embedding sum, the talker's next codec input.
- **Weights are int8 per output row; norms are the Neural Engine's layer norm
  over `[x, -x]`**, whose mean is zero, so its variance is x's mean square
  and x is never squared in fp16. The 1/√head_dim of attention rides in the
  query norm's weight.
- **Precision is measured, not assumed.** Before building, MLX runs a short
  probe on the CPU and records each layer's largest MLP product
  (`NeuralPrecision`). A layer above 8,192 keeps its down projection in fp16,
  with its product scaled by a power of two to stay inside fp16 (the up
  projection carries the scale, down undoes it). On the 0.6B that is the code
  predictor's layer 2, product ×1/16: one 3 MB matrix, about 0.4 ms a frame. The
  residual stream runs unscaled.
- **The host does what is cheap, with MLX on the CPU.** The talker's first
  code is drawn in Swift with MLX's rules (control codes suppressed, EOS held
  back two frames, the windowed repetition penalty, top-k with ties, top-p),
  from a seeded generator that also makes the code predictor's noise. MLX on
  the CPU builds the prompt and runs the codec's front end (ADR-0075), never
  the GPU. Its text projection is held as dense fp32 for the Neural Engine
  path: the CPU took about 33 ms per text token with the 8-bit weights, all
  before the first audio, and 0.2 ms with fp32 through Accelerate.
- **Preparation checks the voice against MLX** on its probe: the talker's
  logits within 20 dB; then one greedy code-predictor frame from MLX's hidden
  state, with MLX following the Neural Engine's codes, each code within 2
  logits of MLX's best and the embedding sum within 20 dB. A near tie can go
  either way in fp16; a broken graph lands far below both bars.
- **`Voice.preset(speaker:language:)`** is a CustomVoice checkpoint's own
  speaker, the Preset Voice. The engine refuses one the checkpoint doesn't
  have when the session opens, and never takes or continues a Reference Take
  for it.

## Measured on the M3 Max

| | |
| --- | --- |
| Placement | talker 1,834 ops, code predictor 4,484 ops, all on the Neural Engine |
| Talker step | 6.0 ms |
| Code predictor frame | 12.1 ms (an 80 ms frame needs 18 ms of both) |
| Whole renders, quiet machine | RTF 0.29 to 0.31, codec and first audio included |
| First audio | 120 to 200 ms: prompt 5 to 13, nine prefill steps 60 to 90, three frames 55 to 70, codec 15 to 40 |
| Preparing, models compiled before | about 37 s: the MLX probe 30, the check 7 (MLX on the CPU), Core ML's plans and loads under 1 |
| Building and compiling both, the first time | about a minute more; 425 MB talker, 140 MB code predictor |

Teacher-forced on 582 frames MLX rendered (three texts), against MLX:

| | Logits | KL of the draws | Best code agrees |
| --- | --- | --- | --- |
| Talker | 32 to 35 dB | 2 to 6 × 10⁻³ | 96 to 98 % |
| Code predictor | 33 to 35 dB | about 2 × 10⁻² | 88 to 90 % |
| For scale: MLX against Qwen's fp32 (ADR-0074) | | talker 5 × 10⁻⁴, code predictor 7 × 10⁻³ | 96 to 100 %, 92 to 94 % |

Whisper large-v3-turbo transcribed the renders (seeded, so the two paths
draw differently). Over three texts, ten seeds each, 3 of MLX's 30 renders
failed and 6 of the Neural Engine's: a stall at the frame cap, or a breath or
a cough instead of words, all on the two shorter texts. On the short line
alone, 30 seeds each, both failed 7 times (mean word error 17.3 % on MLX,
18.3 % on the Neural Engine), and half of MLX's renders and 11 of the Neural
Engine's ran to the frame cap. The stalls are the 0.6B's, not the Neural
Engine's.

## Considered and rejected

- **TTSKit's published Core ML models.** They run this checkpoint, but we
  could tune neither their layout, precision, state nor outputs (ADR-0084).
- **Updating the cache inside the graph**, by `slice_update` or by blending
  the new column in with masks: either rewrites the whole cache every step.
- **fp16 weights**: 1.6 to 1.8× slower for a dB or two.
- **Scaling the residual stream into range**: 13 dB, from the subnormals the
  Neural Engine drops.
- **A max-scaled RMSNorm** (x over its largest value, then the mean square):
  no better for the talker, worse for the code predictor.
- **int8 everywhere**: the code predictor's logits at 10 dB.

## Consequences

- The phone's half of slice 3 remains: speed, first audio, and compile time
  and memory on the A18 Pro.
- Preparation's MLX probe and check are slow on the CPU, whose 8-bit matmuls
  MLX doesn't optimize: about 37 s on the M3 Max. Slice 4 runs them once per
  checkpoint, as Voice Preparation, and keeps what they found with the
  compiled models, so a launch only loads them.
- MLX keeps the talker's and the code predictor's weights loaded beside the
  Core ML models, for the prompt and the preparation check. Slice 4 should
  release them once a voice is prepared.
- A prompt and its frames must fit 512 positions, about 80 text tokens at
  the six-frames-per-token cap. The phone's segments stay under that.
- ADR-0075's estimates are superseded: the talker runs at 6.0 ms against 10 to
  14 estimated, the code predictor at 12 ms against 20 to 35. Core ML state
  works on the Neural Engine for the talker's cache, where it had failed for
  a streaming codec.
- The 0.6B with the streaming text layout stalls on about one short line in
  four, on either path. The owner's listening decides what the phone does
  about it: Qwen's other layout (the whole text in the prompt, at the cost of
  a longer prefill), another speaker, or cutting a stall short.
