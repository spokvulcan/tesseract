# ADR-0077: Words are timed by the voice model's own attention

- Status: Accepted
- Date: 2026-09-27
- Amends: ADR-0076 (the Read-Along's placement of words inside a segment, and
  the Speech Overlay's pages)
- Replaces: the Segment Script's token offsets and the one-token-per-frame
  assumption behind them (ADR-0038)

## Context

The Read-Along (ADR-0076) places each segment on the audio timeline exactly,
but inside a segment it spread the characters evenly over the audio. The owner
reported the highlight going off within a paragraph, and the overlay jumping.
Measurements on 30 rendered passages
([research](../research/2026-09-27-word-timing-from-attention.md)) showed why:

- Passages that continue a Reference Take open with a median 0.53 s of silence
  (up to 1.55 s), and silence is 30% of their audio.
- The pace runs from 7 to 16 characters a second between passages, and numbers
  and symbols take much longer to say than their characters suggest.
- So the even spread started words a median 504 ms off; 21% landed within
  200 ms of when the word was heard.

The overlay showed two lines of the heard segment and swapped them for the
next two, and swapped all its text when the next segment started.

The owner ruled out running a transcription model beside the voice. Qwen3-TTS
has no duration model and outputs no timing. Its talker does have attention
heads that follow the text (Argmax found them in every variant,
[arXiv 2609.16989](https://arxiv.org/abs/2609.16989)), but nobody had
published which heads they are or how well they time words.

## Decision

1. **The talker's Alignment Head times the words.** Measured once per
   checkpoint family: layer 3, head 0 for the 1.7B (6- and 8-bit, any voice or
   language); layer 6, head 5 for the 0.6B. `TTSModelSpec` carries it, and
   `voiceDesign17B` sets it. A checkpoint without one gets no timing.
2. **A probe reads it.** While a generation asks for timing, that layer's
   attention multiplies the head's last query with the cached keys of the
   prompt's text track (a take's text, the new text, EOS). That is one small
   product per frame, in the step's own graph. Each kept frame's row goes out
   with the frame, ahead of its audio. The samples are unchanged.
3. **The Word Timer follows it** (`WordTimer`, pure), in three steps.
   - **Path.** A forward-only path over the text's tokens (0–2 tokens a frame,
     from before the text to EOS), decided 8 frames behind generation.
   - **Pause correction.** The model moves on to the next word as a pause
     begins, while the voice says it as the pause ends. So a start inside a
     pause, or up to 4 frames before one that begins before the next step,
     moves to where the sound resumes.
   - **Silence and frame counts.** A frame is silent when it is 25 dB under the
     recent peak for at least two frames; the silence cap already measures each
     frame. Frames the cap drops don't count.
4. **The stream carries word starts.** `SpeechEvent.words` gives each word's
   place in its segment and its start frame, counted over the utterance, after
   the audio it starts in. Words split on whitespace and newlines, as the Word
   Timeline splits them.
5. **The Read-Along looks words up.** Inside a timed segment, the heard word is
   the last one whose start is at or before the clock. An untimed segment still
   spreads its characters.
6. **The clock is the heard time.** `AudioPlayback.heardPlaybackTime()` is the
   playback head less the output's latency (the player node's
   `outputPresentationLatency`, times the playback rate). That is 14 ms on
   built-in speakers and 150 ms or more over Bluetooth. Pacing still reads the
   render head.
7. **The Speech Overlay is one continuous feed.**
   - **Layout.** It lays out each segment as it arrives, up to 8 s ahead, into
     numbered lines that break at paragraphs.
   - **Motion.** It moves the whole column up one line, on a 0.32 s critically
     damped spring, when the heard word reaches the next line; the finished
     line fades out.
   - **Drawing.** It never swaps text in place, and draws only the lines from
     one above the heard line to three below.
   - **Reduce Motion.** With Reduce Motion on, lines move without animation.
   - **Color.** Unread words are 60% white.

## Consequences

- **Accuracy.** Measured against Whisper on voices not used for tuning, word
  starts land a median 40 ms off, and 96% land within 200 ms (21% before). The
  8-bit checkpoint, the 0.6B checkpoint, Russian and German measured alike.
- **Cost.** In the Swift engine, a frame took a median 11.00 ms with the probe
  and 11.00 ms without. No second model loads.
- **Early frames.** Timing trails generation by about 8 frames plus the pause
  after a word. The engine runs about 7× faster than real time, so a start is
  known before it plays. The exception is the first half-second of a reading,
  which falls back to spreading characters.
- **New checkpoints.** A checkpoint family added later needs its head found:
  run the research harness against Whisper.
- **Chinese and Japanese** time well by eye, but the app splits words on spaces,
  so their highlight still needs word units.
- **Removed:** the Segment Script's `tokenCharOffsets`, the port's
  `alignmentOffsets(for:)` and `Qwen3TTSModel.tokenizeForAlignment`. They cost
  a tokenization per segment inside the GPU lease, for an assumption the model
  doesn't follow: the streaming layout feeds 12.5 tokens a second, and speech
  uses about 3.

## Alternatives considered

- **A forced aligner or recognizer on the audio.** Qwen3-ForcedAligner-0.6B,
  Whisper and wav2vec2 aligners all work, but each is a second model of up to
  0.9B parameters, more memory, and a wait for the audio. The owner ruled it
  out.
- **Audio alone: pauses pinned to punctuation, syllables between.** It
  measured 62% of words within 200 ms, better than the even spread but far
  from the attention's 96%.
- **Every head, or several.** Averaging the best heads with layer 3 head 0
  never beat it alone. Turning on attention weights model-wide is what slowed
  Chatterbox's analyzer.
- **A per-frame pick instead of a path.** It runs ahead to the next word and
  jumps; the forward-only path is what makes it usable.
- **Continuous scrolling at speech pace.** Moving text is harder to read, and
  caption practice advances one line at a time within 0.433 s (CEA-608
  roll-up).
