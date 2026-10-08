# ADR-0089: Vision and DFlash2 on one Qwen3.5 engine; DFlash2 speculates over images

- Status: Accepted (built; the loaded-model gates are under As built)
- Date: 2026-10-07
- Closes: map #457 and its children #458 (one engine) and #459 (pair the
  draft with the vision class, drop the Text-Only Override)
- Amends: ADR-0079 (the plan's text-only row now binds MTP alone),
  ADR-0057, ADR-0059 and ADR-0061 (the DFlash2 target protocol carries a rope
  delta; the iterator takes prompts with images)
- Relates to: ADR-0007 (Cache Key Space, Position Anchor), ADR-0008 and
  ADR-0013 (when the vision container loads), ADR-0067 (a Rotated Ternary
  Checkpoint still refuses the draft)

## Context

Images and DFlash2 speculation were exclusive on Qwen3.8-27B. The draft's
target side is the tuned MLXLLM Qwen3.5 engine: the compiled decode schedule,
the compiled verify segments, gated-delta captures and hidden-state capture.
`MLXVLM.Qwen35` carried its own copy of the language model with none of that,
so the draft paired only with the text class, and both 27B entries carried a
Text-Only Override: the HTTP server and the chat loaded them as text and
dropped every image (#457).

A target that paired would not have been enough. The Speculation Plan refused
any request that carried an image, and an agent conversation is sent whole on
every turn, so one screenshot turned speculation off for the rest of the
conversation.

The draft's authors speculate over image prompts in their own runtime
(incoai/splash). Its target places rows with M-RoPE: text rows advance one
counter on all three axes, an image's rows spread over temporal, height and
width from the counter at the image start, and the counter then moves on by
the image's largest merged side. Its draft "is a text model over logical
positions": it reads the target's hidden states for every row and places its
own context at token rows.

## Decision

1. **One engine.** `MLXVLM.Qwen35` hosts `MLXLLM.Qwen35TextModel`; MLXVLM
   depends on MLXLLM, which imports nothing from MLXVLM. The text model gains
   `Qwen35RotaryPositions`. `.shifted(delta)` puts every row at its cache row
   plus a delta and runs the text model's own rope at that offset: the fused
   norm and rope kernel, the compiled decode segments and the verify segments.
   `.multimodal` takes explicit `[3, batch, length]` positions; only a pass
   that holds image rows uses it, through an interleaved M-RoPE in float32.
   `inputEmbeddings` replace the token embedding for a pass whose image
   features are merged in. A zero delta runs exactly the text model's code, so
   text through the vision class is the text class, bitwise. The vision class
   keeps its contract: `prepare` (one shot or windowed) and decode carry the
   rope delta in `qwen35.ropeDeltas`, now built on the host, so a decode step
   reads it without waiting on the GPU.
2. **DFlash2 rotates past the images.** `DFlash2VerifyRequest` and
   `dflash2Prefill` take a `positionDelta`. A verify pass writes and masks at
   cache rows and rotates at row plus delta; the draft still places its
   context at cache rows, as splash's does. The iterator takes the delta at
   init and accepts `[1, L]` tokens. It also takes a prompt with images when
   the target is a `DFlash2MediaTargetModel`: the vision class prefills
   through the last image row with its own `prepare` and returns the delta,
   then the iterator speculates over the text after it.
3. **The plan.** DFlash2, when resident, engages with images or without; MTP
   stays text-only. On the keyed path an image-bearing plan's execution base
   is at or past the Minimum Warm Offset (the anchored vision prepare runs the
   image span) and the split never precedes the base, so the iterator takes
   the prompt's text and the Position Anchor's delta at the split. The Raw
   Generation Start hands over the whole prompt and the target prefills the
   images.
4. **The catalog.** Neither 27B entry withholds images any more, and the
   Text-Only Override is retired: a Vision-Capable Model is the checkpoint's
   own declaration.

## Considered options

- **Port the engine into the vision class.** Rejected in #457: a second copy
  of the tuned engine, every DFlash2 lever done twice.
- **Pair the draft but keep image-bearing turns on the ordinary path** (#459
  as filed). Text-only requests speculate on a vision load, but a conversation
  that ever carried an image decodes without the draft from then on.
- **Feed the image rows' hidden states to the draft**, as splash's context
  replay does. Not taken: a keyed split already sits past every capture, so
  the draft's context on a warm turn is the tail anyway. The acceptance
  measured without it is under As built.

## Consequences

- Every Qwen3.5 vision load (the PARO checkpoints, the 35B-A3B MoE, Bonsai 2,
  both 27B entries) now decodes through the tuned engine: compiled decode,
  projection stacking and the folded query scale apply to it too.
- Image rows rotate in float32; the old vision copy computed its cosines in
  the activation dtype.
- The VLM MTP drafter reads the engine through public accessors and keeps its
  contract (`mtpPositionDeltasKey` from the same rope delta).
- The DFlash2 target protocol change is a follow-up to upstream PR #607.

## As built (2026-10-08)

Measured on a MacBook Pro with an Apple M3 Max and 48 GB, Release builds,
Qwen3.8-27B 4-bit with the 4-bit draft at block 8, `--bench-check` (every
generated token compared):

- **The text class is unchanged.** `main` and this change on the travel
  fixture: full-stream identity on both arms, the same round accounting
  (acceptance 140/356, the banked reference), 51.71 against 51.75 tok/s.
- **The vision class is the text class.** The same fixture through the
  vision class: full-stream identity with the text class on both arms,
  acceptance 140/356. Four alternating runs, normalized by the
  autoregressive arm as the drift reference (the machine warmed and the
  autoregressive arm fell 12% across them): DFlash2 2.22x over
  autoregression on the text class and 2.27x on the vision class.
- **Images speculate.** A 1430x924 figure with a question, 1,318 prompt
  tokens: autoregressive 20.8-22.4 tok/s, DFlash2 46.0-49.8 tok/s (2.2x),
  acceptance 34.4% (180/524), 3.37 tokens a round. The description reads the
  figure's title, axis labels and ticks correctly. DFlash2 matches the
  autoregressive stream for 82 tokens. The streams part at near-ties that
  rounding alone decides: at token 56 the probe shows "Overall" at 29.25
  against "Top" at 29.125, one bf16 step, and an autoregressive decode over
  the iterator's own prefill split takes "Top" there while both other arms
  take "Overall". A different prefill split flips that token as surely as
  the verify pass does; every emitted token is still the target's own
  choice. The text `math` fixture parts from its autoregressive stream the
  same way at +8.
- **The PARO Checkpoint.** `qwen3.8-27b-paro` through the vision class
  against its text class on the travel fixture: full-stream identity on both
  arms, acceptance 136/382 on both, 31.84 against 31.75 tok/s. On the figure
  prompt the autoregressive, split-prefill and DFlash2 streams are identical
  for all 256 tokens (acceptance 31.8%, 177/557); its rounds take 117 ms
  against the 4-bit's 72, so the speedup there is 1.18x, the PARO pairing's
  known profile (#483).
- **The HTTP path.** `scripts/dev.sh prefix-cache-e2e --bench-model-id
  qwen3.8-27b`: 35 of 35 checks pass, the image scenario among them, which
  used to skip on this entry. The follow-up restores past the image (298
  cached tokens against the text-prefix baseline of 177), agent-shaped image
  history lands cache-aware (297), a different image of the same size reuses
  nothing past the text prefix, and warm image outputs are byte-equal to
  cold. Every image-bearing turn decodes with the draft
  (`image_turns_speculate`: 8 of 8, 29-41% acceptance).
- **Other vision loads on the new engine.** The catalog default,
  `qwen3.5-4b-paro`: the e2e runner passes 35 of 35, the image scenario among
  them. Bonsai 2 27B, whose rotated layers now sit in the hosted engine, reads
  the figure correctly with the autoregressive, split-prefill and DFlash2
  streams identical over 160 tokens; the app still refuses it the draft
  (0.72x in the bench, as ADR-0067 measured).
- **The app.** With the owner's settings (DFlash2, vision when available):
  the load logs `visionMode=true` with the draft resident. An HTTP request
  with the figure (1,302 prompt tokens, 1,200 generated) speculates at 33.9%
  acceptance over 357 rounds. The follow-up turn restores 2,502 of its 2,535
  prompt tokens past the image and speculates again (32.6%), 600 tokens in
  about 11 s.
