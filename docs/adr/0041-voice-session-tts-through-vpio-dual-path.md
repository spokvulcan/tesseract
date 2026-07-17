# Voice-session self-echo: Echo Floor + Soft Barge over Dual-Path Playback

Status: accepted — detector + Soft Barge landed (PR A); acoustic voice hold v2
redo pending (PR B, continues PR #355)

The Voice Session keeps the microphone open while the assistant speaks (full
duplex — voice barge-in is a hard requirement, #310 §4), which makes Self-Echo
the signature failure. The 2026-07-17 flight recorder measured it: 21 energy
barges, 20 false — including nine pause/resume flaps inside one 438-char reply
— each false fire costing a 2–3 s dead pause. The owner's verdict: the session
is unusable, and false positives must not happen at all.

## The corrected acoustic model (researched 2026-07-17)

The first version of this ADR assumed the fix was acoustic: render the reply
through the VPIO capture engine so the canceller gets a sample-aligned far-end
reference. Research against primary sources corrected the model:

- **macOS VPIO's AEC reference is the output *device* signal (loopback).**
  `AVAudioIONode.h` says VP removes "audio that is played from the device";
  Chromium ships its macOS native AEC as a capture-only VPIO unit with a
  *silent* output element (playout through a separate AUHAL) — cancellation of
  other-engine audio is how it works at all. Same-unit rendering is not a raw
  AEC upgrade (confirmed on this hardware: voice-hold-lab E2 measured hosted ≈
  dedicated residual in steady state; PR #355's E2 found the same).
- **What same-unit rendering does buy:** the reply is "voice audio" instead of
  "other audio", so it escapes the recording duck (WWDC23: VP ducks all
  non-voice-path audio) — it plays undipped; and the *voice hold* (engine kept
  running across the session) keeps AEC/AGC converged between turns with ~0 ms
  turn transitions (an engine start costs ~50 ms and AGC re-ramps from
  near-zero).
- **Residual leaks in every path.** Lab E2, real hardware: residual peaks
  0.6–0.7 normalized against a static barge gate of 0.25, in hosted and
  dedicated paths alike — onset transients, not steady state. No static
  threshold survives that; an undipped (louder) reply makes it strictly worse.
  **This is why the first redo attempt (PR #355 as committed) made
  self-interruption worse in the field: it shipped the louder reply with the
  reference-blind detector.** The detector layers below are therefore not
  hardening — they are the fix, and must land before or with the acoustic
  swap.

## Decision — three layers

**1. Echo Floor (reference-aware energy gate).** During playback the barge
threshold is `max(static, floor + margin)`, where the floor tracks the mic
level attributable to the reply: fast attack (converges inside the 0.45 s
debounce), slow decay while playback is loud, fast decay once it has been
quiet past a trailing hold — and, the load-bearing discrimination, the floor
may never believe more than `playbackEnvelope − echoPathLoss`: residual
physically cannot out-shout the reply minus the room's echo-path loss, while
the owner can. The playback envelope comes from the sink itself
(`PlaybackEnvelope`, 50 ms bins over scheduled samples — every `AudioPlayback`
reports `playbackLevel()`), so the gate is route-independent. Constants are
calibrated, not guessed: the voice-hold lab records real traces, and
`VoiceBargeReplayTests` replays them through the real detector — **zero
onsets on clean-reply traces is a committed regression test**, alongside
fire-≤600 ms on owner-level speech.

**2. Soft Barge (two-stage barge-in).** An energy onset no longer pauses: it
ducks the reply to 25 % within ~100 ms (instant acoustic acknowledgment,
capture opens immediately — no owner words lost) and opens a 0.8 s confirm
window. Sustained voicing (≥ 0.3 s accumulated — never `isInSpeech`, which
holds through trailing silence) commits the hard pause and the take proceeds
exactly as before (Substance Gate, Session Directives, resume-on-false).
Without voicing the volume fades back: a residual false fire costs a ~1 s
murmur instead of a 2–3 s dead pause. The overlay click stays an immediate
hard pause — a click is deliberate, and had zero false positives in the field
data. #310 §4's "threshold + debounce, not ASR" holds: both stages are pure
energy.

**3. Anti-flap guards.** A 1.0 s post-resume deafness absorbs the fade-up and
re-settling transient (the field flap cycle re-barged ~0.85 s after each
resume); an escalation ladder per utterance (#354 item 2) widens the floor
margin ×1.5 after 2 false barges and mutes the energy detector after 4 (click
and directives keep working); and every barge event now records
level/threshold/floor/playbackLevel plus a 1 Hz `voice.energy-sample` — the
07-17 storms shipped no numbers at all, so tuning was blind.

**Acoustic substrate (the redo of the redo, PR B).** Voice hold v2
(PR #355) is kept — its crash-safe discipline is verified against the header
contracts: the tap installed once per hold on a stopped engine (capture
start/stop = a gate flag read by the tap block; `installTapOnBus` with a
non-matching format is a format-*set*, and VP IO formats are stopped-state
only — the 2026-07-17 SIGABRT), the render side wired stopped-only with the
format pin verified by read-back, detached wiring committing by generation.
Two changes on top: a **persistent player node** attached during hold wiring
(Apple's own VP sample wires all players before start; no per-utterance graph
churn), and wiring latency treated as measured — **~2.3 s on this hardware
(lab E6)**, so captures keep fast-failing into the session's 1 s backoff and
the session-enter UX covers the gap. The dedicated-engine fallback remains
mandatory: the reply always plays.

## Considered and rejected

- **Transcript-level self-echo filter** (drop turns fuzzy-matching the spoken
  text): owner-rejected — the fix must be real at the acoustic and
  state-machine layers.
- **Half-duplex** (mic closed while speaking): trivially loop-proof, rejected
  — speak-to-interrupt is the product.
- **ASR-confirmed barge** (pause only after words transcribe): violates #310
  §4's mechanical mandate and adds ~1 s to every real interruption.
- **Device-level VAD** (`kAudioDevicePropertyVoiceActivityDetectionEnable`,
  "with echo cancellation"): designed for process-muted mic scenarios;
  unproven for an open capture. Future work, not adopted.
- **Static threshold raise while TTS plays**: the naive form of the Echo
  Floor; a fixed raise either misses quiet real barges or admits loud
  residual — the floor + path-loss cap adapts to both.

## Consequences

- Missing a real barge is now the cheaper failure (the reply keeps playing;
  the owner repeats louder, clicks, or says a directive once capture opens) —
  the zero-false-positive goal deliberately biases the gate. Field telemetry
  (energy samples on every barge event) is the tuning loop for sensitivity.
- The confirm window means a real interruption hears the reply murmur under
  it for up to ~0.8 s before the pause — accepted for the instant duck
  acknowledgment.
- The runtime harness is **committed** at `tools/voice-hold-lab` (v1 lived in
  gitignored `research/` and evaporated with its evidence — unverifiable
  claims gate nothing). Its RUNBOOK carries the measured results; its
  fixtures + `VoiceBargeReplayTests` are the calibration lock. The acoustic
  redo (PR B) does not land without them green.
- VP render-path coloration of hosted replies remains gated by an owner ear
  test (lab `e5`/PR checklist); the engine-rebuild-mid-reply behavior is
  unchanged (invalidation reports end-of-utterance; the session recovers).
