# ADR-0054: The push-based playback fold is a value machine

- Status: Accepted
- Date: 2026-07-21
- Relates to: ADR-0050 (Hold Wiring Arbiter — the policy/performer template
  this is the fourth application of), ADR-0041 (Dual-Path Playback — the two
  adapters the fold lives under)

## Context

The push-scheduling fold behind streaming TTS — `pendingBufferCount`,
`streamFinished`, `playerStarted`, `totalScheduledSamples`,
`streamingSampleRate`, the start gate, the finish detection — was declared
and mutated inline in three places: `AudioPlaybackManager` (the dedicated
engine adapter), `VoiceSessionPlayback` (the hosted/VPIO adapter), and
`InMemoryAudioPlayback`, the test peer, whose header admitted it "computes
`totalScheduledDuration` the same way the real manager does" — by hand.

The three copies had **already drifted on a race guard**.
`VoiceSessionPlayback` guarded stale buffer-completion callbacks with a
`streamEpoch` (a stopped node flushes its scheduled handlers, so a late
completion carrying an old epoch was dropped rather than allowed to
decrement a newer session's counter). `AudioPlaybackManager` had no such
guard: a stale buffer from a stopped or restarted stream could decrement
the *new* stream's `pendingBufferCount` and, if that hit zero under a
finished stream, fire `onPlaybackFinished` early — a latent early-finish
race. No test constructed either shipping adapter (`AudioPlaybackTests`
said so outright); the fold that ships had zero coverage.

## Decision

**`StreamingScheduler`** — a pure value machine over the fold's facts
(`pendingBufferCount`, `streamFinished`, `playerStarted`, `isPaused`,
`totalScheduledSamples`, `streamingSampleRate`, and the stream `epoch`),
one transition per driver, each returning a verdict the adapter performs:

- `beginStream(sampleRate:)` — resets the counters and bumps the epoch,
  staling any completion still in flight from a previous stream.
- `appendChunk(sampleCount:)` — records the scheduled chunk and returns an
  `AppendOutcome(epoch:, startPlayer:)`: the epoch to stamp this buffer's
  completion with, and whether this is the first unpaused chunk (the start
  gate `!playerStarted && !isPaused`).
- `bufferCompleted(epoch:)` — the landing verdict: `ignoreStale` when the
  epoch is old, else `pending`, else `finished` when the last buffer of a
  finished stream drains.
- `finishStream()` — `finishedNow` when nothing is pending, else
  `awaitingDrain`.
- `pause()` / `resume()` — return whether the pause state actually changed
  (resume also sets `playerStarted`, since resuming plays the node), so the
  adapter pauses/plays the node exactly once.
- `stop()` — resets and bumps the epoch, invalidating every in-flight
  completion.

Every counter mutation and every gate decision lives in the machine. The
adapters keep only their `AVAudioPlayerNode`/engine calls, the audio format
and loudness `PlaybackEnvelope` (a separate pure value), the dedicated
adapter's published `isPlaying`, and the clock position each holds while
paused (`pausedTime` — the machine has no notion of render time). Derived
reads (`totalScheduledDuration`, and the `hasUndrainedAudio` gate behind
`VoiceSessionPlayback.hostedPlaying`) come off the machine, not a
hand-copied formula. `pause` state is a machine field rather than an
adapter input, so the start gate stays a machine decision and the adapters
stay verdict-only. The `AudioPlayback` port is unchanged.

This is the fourth application of the policy/performer template — after the
capture engine's lifecycle and system-audio duck, and the hold-wiring
arbitration (ADR-0050). Like the arbiter it is stateful, so it is a value
machine rather than a stateless decision function; it stays a struct
(synchronous, MainActor-driven transitions; only the buffer callbacks are
async, and they carry the epoch that lets a stale one prove itself).

## Consequences

- **The drift is unrepresentable.** The epoch guard now has one home, and
  `AudioPlaybackManager` inherits it by construction — the latent
  early-finish race (a stale buffer from a stopped/restarted stream
  decrementing the new stream's counter and firing `onPlaybackFinished`
  early) is fixed. This is a deliberate behavior change for the dedicated
  adapter, not a silent one.
- **The scheduler that ships finally has tests** — `StreamingSchedulerTests`,
  a decision table needing no AVAudioEngine: the start gate firing exactly
  once per stream, stale completions ignored across an epoch bump, finish
  only when finished-and-drained, finish-on-finish with zero pending, stop
  mid-stream invalidating in-flight completions, plus the pause interplay
  and per-stream sample-rate/duration reset the real folds carry.
- The three sinks — two shipping adapters and the in-memory peer — drive one
  machine; a new field or gate is added once, and no adapter can fall behind.
