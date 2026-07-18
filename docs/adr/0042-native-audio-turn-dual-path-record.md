# Native Audio Turn: the model hears the take; transcription becomes the record

Status: accepted — experiment shipped behind a default-off toggle (PRD #358,
map #301)

Gemma 4 12B takes native audio input: the unified checkpoint carries an audio
tower (`audio_config`, 750-token / 30 s clip window at 40 ms per token)
alongside vision, so a spoken owner turn can reach the model as *sound* —
prosody, hesitation, emphasis — instead of a transcript. For Jarvis, whose
whole surface is the voice session, that is not a novelty: it is the first
time the assistant can hear the owner rather than read WhisperKit's summary
of him.

## Decision

A **Native Audio Turn** sends the closed take itself as the model input, and
demotes transcription to a parallel, record-only pass:

- **Model path (GPU):** the take resamples to 16 kHz mono, encodes to one
  canonical WAV blob (`VoiceTakeWAV`, 16-bit PCM — deterministic bytes), and
  attaches to the user message. The blob rides the same prefix-cache
  machinery as image attachments: its bytes are digested (`AudioDigest`,
  domain-separated from images), keyed length-preserving into the Cache Key
  Path, and restored warm on the next turn — a voice conversation stays
  cache-viable even though every turn appends a new clip.
- **Record path (ANE):** WhisperKit transcribes the same take concurrently,
  but the text goes only to the overlay feed, the flight recorder, and the
  conversation surfaces. Best-effort by design: the mic reopening for the
  next turn may supersede a slow transcription, costing the record and never
  the turn.

The turn's gates are per take, resolved by a pure function pinned in tests
(`resolveNativeTurn`): the experiment toggle (default off), an
**Audio-Capable Model** selected (read from the on-disk config, so the first
turn can go native while the model cold-loads), auto-send on (staging is
inherently text), and the take inside the 30 s clip window — anything longer
falls back whole to the ASR path rather than being truncated silently by the
processor.

**The empty-take gate is acoustic.** The ASR path judges emptiness by the
transcript (""); the native path cannot wait for one. The endpointer's
accumulated voiced time stands in: under 0.35 s of voicing (just above the
0.25 s listening start-debounce, beside the Soft Barge's 0.3 s
confirm-voiced) the take is no turn — a barged reply resumes, nothing sends.
This extends ADR-0041's owner decision that voice-session judgments are
purely acoustic, no word gates.

## Consequences

- The default path is byte-for-byte untouched: toggle off (or any gate
  failing toward `transcribe`) runs exactly the shipped ASR turn.
- The persisted conversation carries the take (canonical WAV in the message
  store, like image attachments), so history re-renders reproduce identical
  bytes — the radix prefix cache and the "reopen a conversation" contract
  both depend on that determinism, and `VoiceTakeWAV` is deliberately not a
  general WAV reader for the same reason.
- A native turn's user message has empty text at the model boundary; the
  memory system's recall cue for that turn is therefore empty, and episode
  capture leans on the record path's transcript. Accepted for the
  experiment; revisit if the toggle graduates.
- Model switching mid-conversation degrades honestly: replayed audio turns
  against a deaf model become an explicit text note ("cannot hear them"),
  mirroring how tool-result screenshots degrade on text-only sessions.

## Alternatives considered

- **Transcribe-then-send with the audio attached as garnish** — keeps the
  transcript as model input and loses the entire point (the model still
  reads, never hears) while paying both latencies in series.
- **Word-gated empty detection (wait for the transcript before sending)** —
  reintroduces the serialized ASR latency the native path removes, and
  re-litigates the word-gate removal ADR-0041 already settled.
- **Float32 WAV persistence** — double the bytes for fidelity below the
  audio tower's mel front end; 16-bit PCM is transparent here and halves
  every conversation-store write.
