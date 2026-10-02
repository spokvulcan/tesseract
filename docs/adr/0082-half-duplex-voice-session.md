# ADR-0082: The voice session is half-duplex; a key or a click interrupts Jarvis

- Status: Accepted
- Date: 2026-10-01
- Supersedes: ADR-0041 (Echo Floor, Soft Barge and Dual-Path Playback over the
  voice hold), ADR-0050 (the Hold Wiring Arbiter)
- Amends: ADR-0042 (the Voice Session Machine loses barge-in by voice),
  ADR-0054 (the Streaming Scheduler has one adapter left)
- Relates to: ADR-0025 (Voice Processing armed for the app's lifetime, the
  System Audio Duck), #310 (the voice session), #354 (the hardening list)

## Context

The voice session kept the microphone open while Jarvis spoke so the owner
could interrupt him by voice: #310 §4 made voice barge-in a hard requirement,
and ADR-0041 rejected half-duplex in one line — "trivially loop-proof,
rejected — speak-to-interrupt is the product". Keeping that promise took a
**voice hold**: the capture engine ran for the whole session, a capture start
or stop was a gate flag read by a tap installed once, the reply rendered
through the voice-processing engine's own player node, and the wiring ran
detached (~2.3 s, lab E6) behind its own arbiter (ADR-0050) and its own
recovery path. On top of it sat the Echo Floor, the Soft Barge, an
escalation ladder and post-resume deafness, all there so Jarvis would not
interrupt himself. About 600 of the capture engine's 1,290 lines were the
hold. The evidence that it cost more than it bought:

- **The field log, 2026-10-01.** Three captures within 24 s delivered no
  samples under the hold ("Capture delivered no samples over 2.6 s / 5.9 s /
  5.1 s — discarding engine", the hold branch of `stopCapture`). The owner's
  speech in all three was lost: the hold only found out at the stop, then
  tore the engine down and re-wired a new one.
- **The wrong rate.** Captures under the hold were recorded at 24 kHz instead
  of the device's 48 kHz (the Capture Dump's WAVs) — every session take ran
  at half the rate dictation records at, on a graph dictation never uses.
- **The recovery is the wedge.** The hold recovered by tearing down and
  rebuilding a voice-processing engine on a background task (~2.3 s). The
  voice-hold lab's own note (quoted below) is that back-to-back VP engine
  create/destroy cycles are what wedge CoreAudio input. The hold answered a
  dead input with the move that kills inputs.
- **A latent crash.** When the held engine stopped on its own, the next
  `startCapture` re-wired the *same* engine: it detached only the player
  node, and the detached wiring called `installTap` again while the old tap
  was still on the bus (`inputTapInstalled` was never cleared) — outside
  `ObjCExceptions`, so AVFAudio's raise for a second tap on a bus would have
  taken the app down.
- **The wins were narrow.** ADR-0041's own research found that rendering on
  the capture unit is no echo-cancellation upgrade on macOS (the canceller's
  reference is the device loopback either way) and that it cost double-talk
  headroom — the hosted reply played at half gain only so the owner's voice
  could still compete with the residual-echo suppressor. The detector stack
  existed to make an open mic under the reply tolerable, and field tuning
  never reached the zero false barges the owner asked for (#354).

### The lab's wedge note

From `tools/voice-hold-lab/RUNBOOK.md` ("Lab-found gotchas"), deleted with
the lab:

> **Back-to-back VP engine create/destroy cycles wedge CoreAudio input** —
> later engines in the process (and for a while, in NEW processes) get zero
> input buffers. The same pattern the app's kept-engine design avoids. E2
> therefore uses ONE held rig with segment slicing; a too-short trace aborts
> the run rather than emitting garbage fixtures. If wedged: wait ~10 s and
> re-run; persistent wedges clear with a coreaudiod restart or device toggle.

## Decision

The voice session is **half-duplex**: the microphone is never open while
text-to-speech plays.

1. **Interrupting Jarvis is a key press or a click.** While a session is
   active, the Talk to Tesseract and Speak Selected Text hotkeys (⌃Space and
   fn+Space by default) and a click on the speaking line stop the reply at
   once and open the mic for the owner's turn. `voice.barge-in` records the
   source (`key`, `click`) and how far into the reply it landed. Nothing
   resumes an interrupted reply. **This
   reverses ADR-0041's "Half-duplex (mic closed while speaking): rejected —
   speak-to-interrupt is the product" and #310 §4's requirement of voice
   barge-in.** Speaking over Jarvis no longer stops him.
2. **Replies play on the normal speech path** — `AudioPlaybackManager`
   through `SpeechCoordinator`'s one sink — and follow the read-aloud speed.
   The coordinator loses its playback route, the voice-session sink, the
   fades and the level read; the playback port loses its loudness envelope
   and volume.
3. **Every capture is a per-take tap** on the kept engine at the device
   rate, session takes included: a start installs the tap and starts the
   engine, a stop stops it and removes the tap — the lifecycle dictation has
   run since ADR-0025. A device change always takes the idle rebuild.
4. **A live-input check** watches every open capture (the settings meter
   excepted) every 0.5 s through a heartbeat the tap bumps once per buffer;
   a live input delivers buffers through silence too. An input that never
   delivered a buffer gets 1.5 s (a Bluetooth headset switching to its
   microphone profile takes about a second), then is rebuilt and restarted
   once under the same open capture — lossless, there is nothing to lose yet —
   unless its engine was built less than 5 s ago: back-to-back
   voice-processing engines are what wedge input, so a silent young engine is
   reported dead instead. One that went quiet, or that the restart did not
   revive, is marked dead: the meter drops to zero,
   the engine is marked for a rebuild, and the capture stays open for its
   owner. Its stop discards the engine, schedules the idle rebuild, and
   returns what arrived before the input died. The verdict is a pure
   function of the **Capture Engine Lifecycle**; the engine performs it.
5. **The Voice Session Machine keeps its shape** (ADR-0042) and loses
   barge-in by voice. The mic closes before a reply speaks. `speechDone`, the
   watchdog and `bargeIn(source)` all stop speech first, then open the mic
   behind the 0.3 s deaf grace. A dead input while listening closes the
   capture, records `voice.capture-dead` and reopens on the 1 s backoff; two
   in a row with the owner not heard in between end the session
   (`capture-dead`). An input that dies mid-turn needs no rule: the zeroed
   meter lets trailing silence close the turn on what was captured.
6. **Half-duplex covers all speech, not only the session's reply.** Speech
   the session didn't start — the chat reading a reply aloud, a Companion
   line — holds the mic closed while the session listens (the session
   timeout waits for it), and the interrupt key stops it the same way;
   speech that starts over the owner's turn closes his take on what he had
   said. And while his take is still transcribing the mic stays closed: a
   reply that lands meanwhile can't supersede the take by reopening a
   capture, so his words always reach the chat.

## Considered and rejected

- **Keep the hold and fix its recovery.** Any recovery that rebuilds
  voice-processing engines under load runs into the lab's wedge, and the
  hold's acoustic case was already thin.
- **Keep voice barge-in over a per-take capture during playback.** An open
  mic under the reply is the Self-Echo problem ADR-0041 spent three layers
  on without reaching zero false barges.
- **Push-to-talk only, no auto-listen.** The auto-listen loop works; only
  interrupting by voice failed.

## Consequences

- The owner cannot talk over Jarvis; he presses a key or clicks. A reply
  plays to its end unless he does.
- Self-Echo is unreachable by construction: the mic is closed whenever
  speech plays, and the 0.3 s grace covers the room tail after it stops.
- A session take is recorded at the device rate, like dictation.
- The live-input check serves dictation too: a dead mic is noticed while the
  key is held, and the release returns the audio captured before the input
  died instead of nothing.
- Deleted: the voice hold (about 600 lines of `AudioCaptureEngine`), the
  Hold Wiring Arbiter, `VoiceSessionPlayback`, the Echo Floor and its ticker
  gap credit, the Soft Barge and the escalation ladder, the barge-in
  sensitivity Setting, the playback envelope, and `tools/voice-hold-lab`
  with its fixtures and replay tests. Its wedge note is kept above.
- The trace vocabulary loses `voice.barge-soft-onset`,
  `voice.barge-false-resume`, `voice.energy-sample` and
  `voice.barge-suppressed`; `voice.barge-in` now carries `source` and
  `offsetSeconds`; `voice.capture-dead` is new.
- While a session is active, the Speak Selected Text hotkey no longer reads
  a selection aloud and the Talk to Tesseract hotkey no longer starts
  composer voice input: both are the interrupt key, and do nothing when
  nothing is speaking.
- A mutual-silence timeout no longer leaves a capture running: the timeout
  is decided before a capture retry, and a capture that opens after the
  session ended is closed at once (an old leak the review of this change
  found).

## As built

- `Core/Audio/AudioCaptureEngine.swift` — per-take capture only; the
  `InputHeartbeat`, the live-input check (`startLiveInputCheck`,
  `restartSilentInput`, `markInputDead`) and `isInputDead`; `stopCapture`
  returns partial audio from a dead input.
- `Core/Audio/CaptureEngineLifecycle.swift` — `liveInputInterval`,
  `LiveInputVerdict`, `liveInputVerdict(buffersSinceLastCheck:
  buffersThisCapture:rebuiltThisCapture:)`; the hold decisions are gone.
- `Features/Companion/Voice/VoiceSessionMachine.swift` — `Tick.inputDead`,
  `Event.bargeIn(source:)`, the dead-capture count, the hold for other
  speech, the take-in-flight guard; no hold, pause, resume or fade effects.
- `Features/Companion/Voice/CompanionVoiceSessionController.swift` — the
  `inputDead` read, wired to `AudioCaptureEngine.isInputDead`.
- `Features/Speech/SpeechCoordinator.swift`, `AudioPlayback.swift`,
  `AudioPlaybackManager.swift` — one sink, the speed setting always applied.
- `App/DependencyContainer.swift` — the Talk to Tesseract and Speak Selected
  Text hotkeys call `bargeIn(source: "key")` while a session is active.
- Tests: `VoiceSessionMachineTests` (the half-duplex loop, barge-in, other
  speech, takes in flight, the dead-capture recovery),
  `CaptureEngineLifecycleTests` (the live-input table),
  `SpeechCoordinatorTests` (replies follow the read-aloud speed).
