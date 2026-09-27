# ADR-0072: A pinned Reference Take keeps a designed voice the same person

- Status: Accepted
- Date: 2026-09-27
- Amends: ADR-0037 (the 48-step voice anchor), ADR-0038 (anchor policy,
  `PinnedVoice` schema, sampler defaults)
- Relates to: ADR-0071 (first-party Qwen3-TTS), the git-ignored study in
  `research/voice-consistency-2026-09-26/`

## Context

VoiceDesign samples a new speaker from the description on every generation,
so read-aloud segments sounded like different people. The 48-frame voice
anchor from ADR-0037 was meant to prevent that. It was checked mechanically
but never listened to (#337, #341), and at the temperatures the owner wants
for expressive reading (0.8 to 0.9) it held the voice only partly: in one run
of a deep male narrator, segment pitch went 81, 135, 92, 222 Hz. It also
re-rolled per utterance, so two read-aloud requests got two voices.

The anchor had two problems. It placed 3.8 seconds of codes between the user
turn and the assistant turn, with no text, where the model never saw audio in
training. And it was short and often started with silence.

A prototype on the shipped checkpoint (two voices, two seeds, temperature
0.9) compared the anchor with Qwen's own in-context layout: the reference
codes inside the assistant turn, after the text they speak, with the
VoiceDesign description in front as the user turn. Scored by speaker
similarity to the first segment and by pitch drift, the in-context reference
held the voice about as closely as two halves of one segment; the anchor did
not. Lowering only the code predictor's temperature to 0.5, while the talker
stayed at 0.9, tightened it further: the code predictor fills in the acoustic
detail that carries most of the timbre, and the talker carries the words and
the delivery. The owner listened and agreed: one person across segments,
expressive, clean.

## Decision

- Every segment of a session continues one **Reference Take**: a short
  rendering of the voice, kept as its codec frames plus the text they speak,
  placed with Qwen's in-context layout. It replaces the anchor. Nothing is
  encoded from audio; the frames come from our own generation.
- The take is pinned per voice, not per utterance. A session opened with a
  **Pinned Voice** continues its take; otherwise the first segment it renders
  becomes the take, and that segment is kept short (the first sentence or
  two). The app stores the Pinned Voice per checkpoint, language and
  description, so the voice survives relaunch. "Try another take" renders a
  new take from the opening of the composer text (or a built-in sample) with
  a fresh seed and keeps it once it finishes.
- Sampling has two temperatures: the talker's (expression, default 0.9) and
  the code predictor's (timbre, default 0.5). The sampler follows Qwen's
  order: temperature before top-k and top-p, and a repetition penalty (1.05)
  over the last 64 frames instead of 1.3 over the whole segment.
- Segments are about half as long as before (100 estimated tokens, about 75
  words). With the whole segment's text in the prompt, minute-long segments
  stalled for 10 to 13 seconds around the one-minute mark in two of four
  runs; at the new length none of four did.
- A silence cap keeps any silent run in a segment's audio to 1.2 seconds.
  Some segments still trailed off for 1.5 to 5 seconds before the model ended
  them; the cap drops only silent frames past the limit, so a reader's pauses
  and every spoken frame pass through.
- `PinnedVoice` moves to schema 2, which carries the take's text. Schema-1
  voices (anchor frames, no text) are rejected as incompatible.
- The sampler settings move to new storage keys. Values tuned for the old
  sampler meant something else, so they are left unread and the new defaults
  apply.

## Consequences

- Each segment re-reads its take before speaking: a few hundred prompt
  positions, of which only the description in front comes from the
  instruct-prefix cache. Measured on q6 with the v2-listen harness: first audio of a segment that continues a
  long-form take lands at about 200 ms (110 ms without one); a companion line
  that continues a short take lands at 141 to 154 ms (112 ms for the line
  that made the take). Both stay inside the 300 ms warm budget. Later
  segments render while the previous one plays, so a listener only waits on
  the first segment of a request. RTF stays at 0.26 to 0.33.
- Voice identity lives in the take, never in the seed; a seed still only makes
  a render reproducible.
- Readings run slower. The 357-word test story took 120 s at the old
  defaults (178 words a minute) and 126 to 161 s at the new ones (133 to 170).

## Considered and rejected

- A longer or better-placed anchor: the text-less codes between the turns are
  the problem, not their length.
- Design-then-clone through the Base checkpoint (#339): prosody-flat, and a
  second checkpoint.
- A rolling reference, each segment conditioned on the one before: drift
  compounds over a chapter.
- One temperature for both models at 0.9: expressive, but the timbre wanders
  more between segments.
