# Voice-session TTS renders through the VPIO capture engine (Dual-Path Playback)

Status: accepted — first implementation REVERTED 2026-07-17 (crash); redo landed 2026-07-17 (voice hold v2)

**Status note (2026-07-17):** the first implementation (d30412c7) shipped a
*voice hold* that kept the engine running between captures and installed the
capture tap on the running engine. On macOS 26.5.2 that raises an uncatchable
NSException (`installTapOnBus` → `CreateRecordingTap` → `SetOutputFormat` →
`SetFormat`) — an IO node's format cannot change on a running VP engine — and
it took down live dictation (owner crash report, SIGABRT on the main thread).
The retry loop around the failing capture also froze the UI (20 Hz re-attempts,
each costing up to rebuild+re-arm on the main thread; a backoff now guards
that). The acoustic layer is reverted to the dedicated-engine playback path;
every other layer of the self-echo fix (synchronous speak state, hardened
watchdog, pause-on-barge, Substance Gate, Session Directives) stands.

**Redo (2026-07-17, same day):** researched against Apple's own docs and
field evidence, then rebuilt the hold behind a runtime harness
(`research/voice-hold-lab`, results in its RUNBOOK). The crash-safe
discipline, all four invariants proven by the harness and pinned by
`AVAudioIONode.h` ("the output format of the input node and the input format
of the output node have to be the same and they can only be changed when the
engine is in a stopped state"):

1. The capture tap is installed **once per hold, on a stopped engine**;
   capture start/stop is a *capture gate* flag the tap block reads (buffer
   discipline) — never tap install/remove on a running engine.
2. The render side (mainMixer→output) is connected only on a stopped engine,
   with the format pin verified by read-back before start.
3. While running, only Apple-documented dynamic reconfiguration: player
   nodes attach/connect/schedule/detach upstream of the mixer
   (`AVAudioEngine.h` class discussion) and gate flips.
4. VP arm/disarm only while stopped (unchanged).

The wiring (tap + render + start) measures ~860–900 ms on this machine, so it
runs **detached** and commits onto the main actor by generation; captures
fast-fail into the session's existing 1 s backoff until it lands, and a voice
reply that beats the wiring falls back to the dedicated engine. Harness E7
verified the background-thread wiring shape.

**Measured expectation-setting (harness E2):** on the owner's hardware,
macOS VPIO's *device-wide* loopback AEC already cancels other-process
playback to the same steady-state residual as same-unit rendering (chirp
correlation ≈ 0.0007 both ways; a common >2 kHz nonlinear tail remains). The
dual path is therefore not a 20 dB AEC upgrade — its distinct wins are: the
reply **plays undipped** under the open mic (it stops being "other audio" to
the recording duck), cancellation that is the canceller's own render
reference *by construction* (independent of duck/loopback policy), and the
hold lifecycle itself (turn transitions ~0 ms vs ~48 ms engine starts;
AEC/AGC stay converged between turns — AGC re-ramps from near-zero on every
engine start). The state-machine layers shipped in d30412c7/29edc92f remain
the primary self-echo fix; this is the acoustic belt-and-suspenders plus the
UX upgrades. Render-path coloration is gated by the owner ear test
(harness E5).

The Voice Session keeps the microphone open while the assistant speaks (full
duplex — voice barge-in is a hard requirement, #310 §4), which makes Self-Echo
the signature failure: TTS residual leaking past echo cancellation and coming
back as a committed "owner turn". Live flight-recorder traces (2026-07-16)
showed both classes — a false energy barge-in that killed an 808-char reply and
auto-sent a 4-char scrap, and a mid-playback turn committed with no barge-in at
all.

**Decision:** a voice session's replies play through the *capture engine's*
`AVAudioEngine` — the VPIO-armed unit — so the AEC hears the reply as its own
sample-aligned far-end reference instead of "other audio" reconstructed from
the device loopback (`AVAudioIONode.h`: VP removes "audio that is played from
the device"; same-unit rendering is the strongest form of that). A welcome
side effect: the reply stops being "other audio" to the recording duck, so it
no longer plays dipped while the mic is open underneath it. Every other TTS
surface (notch reading sessions, long-form) keeps the dedicated
`AudioPlaybackManager` engine and its unprocessed fidelity — hence
**Dual-Path Playback**. To host this, the engine gains a *voice hold*: for the
session's lifetime it keeps running between captures (start/stop degrade to
tap install/remove) and hosts the session's player nodes.

Acoustics alone is not the fix; it lands with two non-acoustic layers decided
in the same session: pause-on-barge (a barge-in pauses the reply; a take
without substance resumes it; the Substance Gate + Session Directive allowlist
decides) and state-machine hardening (the settled-engine watchdog force-stops
TTS, requires consecutive settled reads, and records its exits; `speakText`
sets its state synchronously so an in-flight utterance can never read `.idle`).

Considered and rejected:
- **Transcript-level self-echo filter** (drop turns that fuzzy-match the text
  just spoken): owner-rejected — the fix must be real at the acoustic and
  state-machine layers, not papered over after transcription.
- **Half-duplex** (mic closed while speaking, click-only barge-in): trivially
  loop-proof, rejected — speak-to-interrupt is the product.
- **All TTS through the VPIO engine** (single path): rejected to keep VP output
  coloration away from the long-form quality pillar (ADR-0036/0039) and TTS
  free of capture-engine lifecycle coupling outside voice sessions.

Consequences:
- VP's render path is a voice-chat path; audible coloration of voice replies is
  the accepted risk, gated by an owner ear test. The adapter falls back to the
  dedicated-engine path (logged) if the held engine cannot host playback, so
  voice keeps working — with weaker cancellation — rather than failing.
- An engine rebuild mid-reply (device change, wedge teardown) now kills that
  reply's audio; the playback adapter reports it as end-of-utterance and the
  session recovers to listening. Rare, logged, and preferable to coupling
  rebuild policy to playback.
- The IO-format match constraint (`AVAudioIONode.h`: input node's output format
  == output node's input format, changeable only while stopped) is why the
  render side is wired at hold begin, not at first play.
