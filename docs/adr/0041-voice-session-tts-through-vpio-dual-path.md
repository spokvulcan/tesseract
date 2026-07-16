# Voice-session TTS renders through the VPIO capture engine (Dual-Path Playback)

Status: accepted

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
