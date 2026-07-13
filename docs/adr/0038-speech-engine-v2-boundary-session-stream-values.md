# ADR-0038: Speech engine v2 boundary — sessions own voice values, the stream carries everything else

- Status: Accepted (under overnight delegation, 2026-07-13 — owner review pending)
- Date: 2026-07-13
- Relates to: map #334, ticket #343; ADR-0036 (substrate), ADR-0037 (one checkpoint, two roles); autopsy `research/v1-autopsy-342/autopsy.md` §4 (the ten constraints); ADR-0003 (speech seams), ADR-0004 (word-highlight port)

## Context

Ticket #343 asked for engine v2's public surface: session model, streaming contract, word-timing feed, cancellation semantics, voice/tier selection — UI-free, shaped so a `/v1/audio/speech` endpoint can sit on top later. The v1 autopsy supplied ten "boundary must…" constraints; ADR-0036 fixed the substrate; ADR-0037 fixed one checkpoint serving two role configs.

Per Design-It-Twice, three interfaces were designed in parallel under opposing constraints (minimal surface / maximal flexibility / trivial common caller). **All three converged, independently, on the same deep structure** — strong evidence the shape is right:

- Segmentation, the 48-step anchor dance, prefix prefill, pacing, GPU-lease choreography, and memory policy all move **behind** the seam (v1's 432-line coordinator proves callers otherwise re-grow them).
- Timing is **ground truth riding the stream**: per-segment token→char offsets + start frame, per-chunk frame ranges. The UI never calls back into the engine mid-session; the K1 "pacing is an estimator" caveat dies.
- **The stream is the cancellation token.** No `cancelGeneration`, no `buildVoiceAnchor`/`clearVoiceAnchor`, no mid-session verbs at all — v1's stale-cancel race (T4) and anchor race (T3) become unrepresentable.
- **Seed is a reproducibility knob, never voice identity** (decided): #339 measured that a seed does not pin a voice. Seed moves out of `TTSParameters` into per-utterance options; default fresh entropy. `.fixed` reproduces a waveform only for identical (checkpoint, precision, text, options, anchor state) — tests and benchmarks.
- Voice conditioning flows as **request data (opaque KV values)**, never mutable model fields.
- GPU lease **per generation burst** (one segment + prefix/anchor builds); every wait happens structurally outside the lease.
- Typed configuration replaces all seven env vars; logger and diagnostics injected; typed progress events, no `@Observable` engine.

The three genuine divergences, adjudicated:

1. **Pacing signal** — a playhead-clock closure crossing the seam (common-case design) vs demand on the stream as the only pacing signal (minimal + flexible). **Demand wins**: no caller-honesty coupling, no clock contract to violate, and the server endpoint gets pacing for free from socket backpressure. Pause falls out: a paused player stops pulling; the engine finishes its burst, releases the lease, and suspends. v1's fake pause (T8) becomes real with zero API.
2. **Session object vs stream-only** — a `SpeechSession` object owning voice state (flexible + common-case) vs a round-tripped `PinnedVoice` value (minimal). **Hybrid**: the session object stays (callers don't thread values; prefix KV and anchor live with a session lifetime — constraint 6), and the minimal design's key product insight is kept as an **export**: a pinned voice serializes to a few KB — voice spec + the ≤ 48 anchor *code frames* + a {model, precision, schema} fingerprint — and is rebuilt into KV on restore. The companion's voice survives app relaunch; `.pinned` restore validates the fingerprint and throws `voiceIncompatible` rather than silently re-rolling timbre.
3. **Admission** — engine-wide FIFO interleaving of concurrent sessions (flexible) vs one active utterance with deterministic supersession (common-case). **Supersession wins for v2**: one GPU, one product voice path; a new `speak` cancels the active utterance and returns only after the old stream has terminated — T4 dies by serialization in the engine actor. Multi-utterance queueing (server) is an additive future mode; sessions already isolate voice state, so the surface won't break.

## Decision

Engine v2's public surface (full contracts in the v2 spec, `docs/voice-engine-v2-spec.md`):

```swift
actor SpeechEngine {
    init(model: TTSModelSpec, synthesizer: any SpeechSynthesizer,
         gpu: any GPULeasing, memory: MemoryPolicy,
         diagnostics: (any SpeechDiagnosticsTap)? = nil)   // default off

    func prepare(_ target: Readiness,                       // .unloaded / .loaded / .warm(priming:[VoiceSpec])
                 onProgress: (@Sendable (EnginePhase) -> Void)?) async throws
    var readiness: Readiness { get }
    func unload() async                                     // deterministic teardown, GPU-synced

    func session(_ profile: SessionProfile, voice: Voice) async throws -> SpeechSession
}

final class SpeechSession: Sendable {
    func speak(_ text: String, options: SpeechOptions = .default) async throws -> Utterance
    func exportPinnedVoice() async -> PinnedVoice?          // serializable; nil until first anchor forms
    func close() async
}

struct Utterance: Sendable {                                // admission-time facts + one stream
    let sampleRate: Int; let framesPerSecond: Double; let segmentCount: Int
    let events: AsyncThrowingStream<SpeechEvent, Error>     // the session's audio+timing+lifetime channel
    var audio: some AsyncSequence<AudioChunk>               // projection for non-highlighting callers
}

enum SpeechEvent: Sendable {
    case segment(SegmentScript)      // text slice + tokenCharOffsets + startFrame, before its audio
    case audio(AudioChunk)           // PCM + utterance-global frame range, gapless
    case segmentDone(index: Int)
    case finished(SessionSummary)    // exactly once, only on full render
}

enum Voice: Sendable {  case standard(language: String?)
                        case designed(description: String, language: String?)
                        case pinned(PinnedVoice) }
struct PinnedVoice: Sendable { /* opaque: VoiceSpec + anchor code frames + fingerprint; Codable envelope */ }

struct SessionProfile: Sendable {    // ADR-0037's roles as configs
    var anchor: AnchorPolicy         // .none / .perUtterance(steps: 48) / .pinned(steps: 48)
    var defaults: TTSParameters
    var pacing: PacingPolicy         // .eager / .lookahead(segments: Int)  — demand-based
    static let readAloud: SessionProfile   // .perUtterance, .lookahead(segments: 1)
    static let companion: SessionProfile   // .pinned, .eager
}

struct SpeechOptions: Sendable { var seed: Seed = .entropy; var parameters: TTSParameters? = nil }
```

Binding contracts (normative, tested at the interface):

1. **Event grammar** per utterance: `segment(k) → audio(k)* → segmentDone(k)` in text order, frame ranges gapless and monotonic from 0, `finished` exactly once iff the full text rendered. Admission-time facts (`sampleRate`, `framesPerSecond`, `segmentCount`) are correct before any event.
2. **Cancellation**: cancel the consuming task → `CancellationError` out of the iterator (never translated — `finished` must mean "fully rendered"); drop the utterance → generation halts within one decoder step, silently. A superseded utterance's stream terminates with `CancellationError` **before** the superseding `speak` returns.
3. **Pacing**: under `.lookahead(segments: n)`, at most the in-flight segment plus `n` undelivered segments exist; the engine then suspends **without the GPU lease** until pulled. `.eager` generates flat-out (short utterances, server, export).
4. **GPU**: lease per burst (one segment's generation, or a prefix/anchor build); never held across a demand-wait or wall-clock playback — assertable via the recording lease adapter.
5. **Memory**: cache limits set and **restored** strictly inside lease scope; clear cadence internal; `unload()`/`prepare(.unloaded)` releases weights, KV, caches, restores the process cache limit, and synchronizes the GPU stream. (#344 owns the policy content behind `MemoryPolicy`.)
6. **Voice**: identity is `Voice` + session anchor state; `PinnedVoice` fingerprints {model, precision, schema} and restore throws `voiceIncompatible` on mismatch — the #339 "voices don't survive precision changes" physics is type-enforced.
7. **Errors**: `modelUnavailable` (prepare/load), `voiceIncompatible` (before any event), `generationFailed` (mid-stream, terminal), `CancellationError` (benign). After any throw the engine state is fully released — no poisoned state.

The **Speech Synthesizer** port (CONTEXT.md name kept) narrows one layer down: `load/warmUp/unload`, `alignmentOffsets(for:) ` (O(n) incremental), `generateSegment(SegmentRequest) -> stream of RawChunk`, `buildAnchor(from codes:...)` — conditioning as values in the request, no mutable model fields. Two adapters each: `Qwen3SpeechSynthesizer` over the ADR-0036 vendor / scripted synthetic; `ArbiterGPULease` over `InferenceArbiter.withExclusiveGPU(.tts)` / recording lease.

Above the seam: `SpeechCoordinator` shrinks to presentation + state machine (~70 lines); **Segment Playback retires** (drain → `for try await`, boundary poll → `startFrame`, pause poll → demand); Word Timeline / TTS Word Tracker / Word Highlight Surface survive, upgraded to ground-truth frames.

## Considered options

Three full designs (transcripts in the #343 resolution): minimal (2 entry points, stream-as-session, serializable PinnedVoice), flexible (session values + epochs, multi-session FIFO, dialogue-export scenario), common-case (whole-text speak, playhead-clock pacing, supersession). The decision above is the convergent core plus the three adjudications; the rejected variants are noted per-divergence in Context.

## Consequences

- All ten autopsy constraints are satisfied structurally (scorecard in each design; the spec carries the merged one), most by making the defect unrepresentable rather than forbidden.
- The engine lives in its own UI-free module; the arbiter's `.tts` `ensureLoaded` path is superseded — the engine self-loads under its own lease bursts and `loadedSlots` reads `engine.readiness`.
- The `/v1/audio/speech` endpoint later needs only: a name→`Voice` registry, WAV/format encoding transport-side, and (if concurrent requests must not supersede) an additive queued-admission mode.
- New CONTEXT.md terms to register: Speech Session, Utterance, Segment Script, Pinned Voice, Readiness, Pacing Policy; Segment Playback retirement note.

## Accepted costs

- Consumer pull discipline is a documented contract, not type-enforced: a paced caller that pulls flat-out buffers audio in AVAudioEngine (engine-side bound still holds). Lives in exactly one place per caller.
- Supersession-only admission means a future concurrent-request server needs an additive mode.
- One event enum: every consumer switches over cases it may ignore; the ignore is a `break`.
- An LLM turn can wait up to one segment burst (~4–6 s GPU at measured RTF); the mirror image of v1's ten-minute starvation, and tunable via internal burst size.
