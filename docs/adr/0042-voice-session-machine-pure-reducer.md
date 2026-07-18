# ADR-0042: The Voice Session loop is a pure reducer; the controller performs

- Status: Accepted
- Date: 2026-07-18
- Relates to: ADR-0041 (Echo Floor / Soft Barge / Dual-Path), ADR-0025 (the
  policy/performer split this copies), #310 (the voice session), #354 (the
  hardening list), map #301

## Context

`CompanionVoiceSessionController` held the whole auto-listen loop inline:
`tick()` interleaved decisions (which **Barge-In** stage, whether to commit a
turn, when the watchdog may exit `.speaking`) with effects (fade, pause,
capture reopen) inside `@MainActor` methods driven by a real 20 Hz ticker and
`Date()`. The judgment ADR-0041 chronicles — the Self-Echo defenses, the Soft
Barge confirm, the escalation ladder, the post-resume deafness, the capture
retry backoff — could only be exercised through CoreAudio.

The tell was in the tests: the only unit coverage drove `resolveSoftBarge`, a
three-line pure helper, while `VoiceBargeReplayTests` — the ADR-0041
calibration lock — carried a hand-copied mirror of the `.speaking` tick
("Mirrors `CompanionVoiceSessionController.tick()`…"). The constants were
pinned; the state machine that consumes them was not, and the mirror could
drift from the branch it guarded. Every field regression of 2026-07-17/18
(the nine-flap storm, the watchdog reopening the mic under live TTS, the
20 Hz capture-retry freeze) lived in exactly the sequencing no test reached.

## Decision

Split the module policy/performer, the same shape as the **Capture Engine
Lifecycle** (ADR-0025):

- **`VoiceSessionMachine`** (a `nonisolated struct`) owns every judgment as
  one fold: `handle(event, at: now) -> [Effect]`. Events are the session's
  inputs — enter/exit, ticks carrying meter + playback levels and the speech
  engine's activity, the overlay click, the reply, the speak-done callback,
  transcription outcomes, and capture-start results. Effects are ordered
  values (hold begin/end, overlay session, feed states, capture open/close,
  finish-take, speak/pause/resume/stop/fade, send/stage, flight-recorder
  records). The real `VoiceEndpointer` and `EchoResidualFloor` are machine
  sub-state.
- **`CompanionVoiceSessionController`** keeps the ticker, the injected
  dependency closures, the transcription task, session-ID stamping, and an
  observable `phase` mirror — and performs effects in order. Effects with
  results (`openCapture`, `finishTake`) feed back into the machine as events
  in the same dispatch loop, so even the capture retry backoff is machine
  judgment; the controller decides nothing.
- **Energy-record snapshots use the last tick's levels** on non-tick events
  (≤ 50 ms stale). Records are diagnostic; keeping live reads out of the
  machine is what keeps it pure.
- **The machine is total.** Any event in `.idle` except `.enter` is inert —
  which also fixes a latent zombie: a transcription outcome landing after
  `exit` could previously re-enter `beginListening` and reopen the mic in a
  dead session (the `.empty` outcome path had no `isActive` guard).

## Consequences

- The calibration lock now replays lab traces through the shipped path —
  floor ingest → threshold → endpointer → soft-barge reaction — instead of a
  mirror; the mirror is deleted. Zero-onsets-on-clean-traces and
  fire-≤600 ms survive unchanged as machine-level assertions.
- `VoiceSessionMachineTests` pins the loop as decision tables: soft-barge
  duck/confirm/fade-back, click hardening, the no-speech resume, the
  escalation ladder's both rungs (margin ×1.5 after 2 false fires, energy
  mute after 4), post-resume and post-utterance deafness, the sustained-only
  watchdog exit, mutual-silence timeout, capture backoff cadence, staged-vs-
  sent turns, and post-exit inertness — all with a virtual clock.
- The controller shrinks from 733 to ~340 lines of wiring with no branching
  judgment; the 19 injected closures remain, but only as effect performers.
- Retuning session behavior means editing one pure module and reading the
  diff of its decision tables.
