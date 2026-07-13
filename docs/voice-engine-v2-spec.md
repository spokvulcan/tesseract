# Voice Engine v2 — Specification

- Status: **Locked** (under overnight delegation 2026-07-13; owner review pending — two listen-gated contingencies in §9)
- Map: [#334](https://github.com/spokvulcan/tesseract/issues/334) · Decisions: ADR-0036 (substrate), ADR-0037 (model), ADR-0038 (boundary), ADR-0039 (residency)
- Evidence: #335 (field survey), #336 (model family), #337 (re-port spike), #338 (budgets), #342 (v1 autopsy), #339 (benchmarks + ear verdicts)

This document is the single implementation handoff for engine v2. Everything here is decided; nothing below requires a new decision before or during implementation. Where a runtime measurement picks between two pre-decided outcomes, the gate and both outcomes are stated.

## 1. What v2 is

The streaming-first, resource-frugal TTS engine beneath the Speech UI and the upcoming companion assistant: a UI-free `SpeechEngine` actor in its own module, serving **one quantized checkpoint in two role configurations**, on a **re-vendored upstream `mlx-audio-swift`** substrate, with word-timing as ground truth, cancellation as stream lifetime, GPU leased only while computing, and memory lifecycle owned end-to-end. Full v1 feature parity (word timing, voice design, TTS parameters, long-form voice consistency) is required; the four v1 failure classes (CPU/GPU waste, tangled architecture, slow first audio, ad-hoc memory) are fixed **by design**, each with a structural mechanism, not a patch.

Out of scope (per map): Speech UI changes, the companion feature itself, STT, building the server endpoint, adopting speech-swift.

## 2. Substrate (ADR-0036)

- `Vendor/mlx-audio-swift` is replaced wholesale by **upstream v0.1.3 (`d302a5c`)** — the exact revision the #337 spike built and smoke-tested — plus the spike's feature port (+283/−5 LOC: seed, `tokenizeForAlignment`, two-phase first-chunk cadence, instant-cancel parity shim, voice anchor across three seams including the **offset-aware causal mask fix** for warm-cache multi-token forwards).
- Full 1:1 tree; the app links the **`MLXAudioTTS` product** (today it links legacy `MLXAudio`, building unused STT/STS/UI targets — `project.pbxproj` product ref change).
- Manifest: `.package(path: "../mlx-swift-lm")` (the vendored fork), mlx-swift pinned at `dc43e62d` (= tag 0.31.4), `swift-huggingface` 0.6.0 → 0.8.1.
- **`TESSERACT-PATCHES.md`** at the vendor root: every divergence from the tag — what, why, which seam. A divergence not in the ledger is a bug.
- Maintenance: tags-only pin; sync on need, never cadence; upstream the portable patches post-v2 (mask fix first); every load-bearing patch pinning-tested (§8).

## 3. Model (ADR-0037)

- **One checkpoint serves both roles**: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, quantized. Design-then-Clone is retired; the 0.6B fast tier is retired (measured speed edge only 1.22×; failed the ear test).
- **Precision gate** (mechanical, post-implementation): measure 8-bit long-form generation peak RSS in the TTS-only bench per ADR-0039's metric. ≤ 3 GB → ship **8-bit** (owner's ear-preferred); else ship **6-bit** (2.84–2.86 GB, fits today). One precision ships for everyone; both checkpoints staged in the app container.
- Roles are `SessionProfile` configs: `.readAloud` (per-utterance anchor, `.lookahead(segments: 1)` pacing) and `.companion` (pinned anchor, `.eager` pacing). Parameter defaults per role; `bf16` remains a reference/debug load option, never shipped.

## 4. Public boundary (ADR-0038 — normative surface)

The engine lives in a new UI-free module (`AgentPackages/SpeechEngineV2` or `tesseract/Features/Speech/EngineV2` per synchronized-group convention; implementation picks by build ergonomics, both satisfy "own module, no app imports").

```swift
actor SpeechEngine {
    init(model: TTSModelSpec, synthesizer: any SpeechSynthesizer,
         gpu: any GPULeasing, memory: MemoryPolicy,
         diagnostics: (any SpeechDiagnosticsTap)? = nil)
    func prepare(_ target: Readiness, onProgress: (@Sendable (EnginePhase) -> Void)?) async throws
    var readiness: Readiness { get }
    func unload() async
    func session(_ profile: SessionProfile, voice: Voice) async throws -> SpeechSession
}
final class SpeechSession: Sendable {
    func speak(_ text: String, options: SpeechOptions = .default) async throws -> Utterance
    func exportPinnedVoice() async -> PinnedVoice?
    func close() async
}
struct Utterance: Sendable {
    let sampleRate: Int; let framesPerSecond: Double; let segmentCount: Int
    var events: some AsyncSequence<SpeechEvent>   // single-consumer; see amendment 2026-07-13
    var audio: some AsyncSequence<AudioChunk>
}
enum SpeechEvent { case segment(SegmentScript), audio(AudioChunk), segmentDone(index: Int), finished(SessionSummary) }
```

> **Amendment (2026-07-13, implementation):** the sketch above originally
> spelled `events` as `AsyncThrowingStream<SpeechEvent, Error>`. That type
> buffers without exposing consumer demand — and demand IS the pacing signal
> (contract 3). The shipped `events` is a custom single-consumer
> `AsyncSequence` over a pull-signaled channel (`UtteranceChannel`): the
> producer parks lease-free at segment boundaries when
> `produced − delivered ≥ lookahead`, and consumer-side task cancellation or
> dropping the `Utterance` cancels the driver (contract 2). Same surface for
> callers (`for try await`), same contracts — only the concrete type changed.

Values: `TTSModelSpec{repo, precision}`, `Voice{.standard|.designed|.pinned}`, `PinnedVoice` (opaque: voice spec + ≤48 anchor code frames + {model, precision, schema} fingerprint; Codable envelope; restore validates or throws `voiceIncompatible`), `SessionProfile{anchor, defaults, pacing}`, `AnchorPolicy{.none|.perUtterance(48)|.pinned(48)}`, `PacingPolicy{.eager|.lookahead(segments:)}`, `SpeechOptions{seed: .entropy|.fixed, parameters?}`, `SegmentScript{index, text, tokenCharOffsets, startFrame}`, `AudioChunk{samples, frames: Range<Int>, segmentIndex}`, `Readiness{.unloaded|.loaded|.warm(priming:)}`, `EnginePhase`, `SpeechEngineError{modelUnavailable, voiceIncompatible, generationFailed}`.

**Binding contracts** (each is a test in §8):

1. Event grammar: `segment(k) → audio(k)* → segmentDone(k)` in text order; frame ranges gapless, monotonic from 0; `finished` exactly once iff fully rendered; admission-time facts correct at `speak` return.
2. Cancellation: task-cancel → `CancellationError` untranslated; drop → silent halt ≤ 1 decoder step; supersession — old stream terminated before the new `speak` returns.
3. Pacing: `.lookahead(n)` bounds buffering to in-flight + n segments; all engine waits happen **outside** the GPU lease.
4. GPU: lease per burst (segment generation / prefix / anchor / load+warmup); never across demand-waits or playback.
5. Memory: per ADR-0039 — **no cache-limit writes**, one clear per utterance end, deterministic unload (release + clear + GPU sync).
6. Voice: identity = `Voice` + session anchor; fingerprint mismatch throws; anchor/prefix survive barge-in; sessions survive engine unload as ingredients.
7. Seed: reproducibility only; identical (checkpoint, precision, text, options, anchor state) → bit-identical tokens.

**Speech Synthesizer port** (below the engine; production = `Qwen3SpeechSynthesizer` over the vendor, test = scripted synthetic): `load/warmUp/unload`, `alignmentOffsets(for:)` (O(n) — §6), `generateSegment(SegmentRequest) -> AsyncThrowingStream<RawChunk, Error>` with conditioning as opaque KV **values** in the request (`VoicePrefix`, `VoiceAnchor`) and anchor capture declared in the request / delivered with the result — no mutable model fields. **GPULeasing port**: production wraps `InferenceArbiter`'s lease queue; the arbiter's `.tts ensureLoaded` path retires (engine self-loads; `loadedSlots` reads `engine.readiness`).

**Above the seam**: `SpeechCoordinator` becomes presentation + state machine (~70 lines); `SegmentPlayback` and `TextSegmenter` retire (absorbed); Word Timeline / TTS Word Tracker / Word Highlight Surface stay, upgraded to ground-truth frames (`SegmentScript.startFrame`, `AudioChunk.frames` replace the smoothed estimator; the estimator path is deleted, the snap is no longer a no-op). `AudioPlayback` gains real `pause()/resume()`; its `diagnostics:` parameter is removed (policy moves to its constructor). `SpeechEngine` v1's `@Observable` surface moves to a thin presenter owned by the app.

## 5. Residency & memory (ADR-0039)

Lazy load on first use → warmup in the same lease (kernels, fused weights, tokenizer, configured voice prefix) → keep warm, no TTL → one `Memory.clearCache()` per utterance end → deterministic app-triggered unload (settings-off / pressure-critical / termination). The engine never writes `Memory.cacheLimit` (LLM-owned; pinning-tested). Envelope metric: TTS-attributable peak RSS in the TTS-only bench. Idle-warm measured post-clear; target ≤ 2.5 GB.

## 6. Performance workstream (owner mandate: squeeze, without quality loss)

Baseline to beat (#339, upstream+port, 8-bit long): **46.1 steps/s, RTF 0.271, warm TTFA 123–177 ms**. Quality gates for every optimization: the §8 machine sanity gates, plus bit-identical output at fixed seed where the change claims neutrality, plus the morning A/B for anything that touches sampling/decode math.

Ranked, evidence-cited targets:

- **P1 — O(n) alignment offsets** (autopsy D5; ported verbatim, still quadratic at `Qwen3TTS.swift:177-189`): incremental detokenizer with an equality pinning test against the quadratic reference over a unicode-heavy corpus; falls back to off-hot-path quadratic for any corpus divergence (offsets already ride the stream per §4, so F3 is fixed regardless).
- **P2 — sampler hot loop** (autopsy D7 class): audit upstream's per-step path for residual per-step tensor rebuilds (suppress mask, repetition-penalty set, full-vocab top-p); hoist/reuse. Bit-identical-at-fixed-seed required.
- **P3 — copy discipline on the chunk path** (D6): one stream end-to-end, no `[Float]→MLXArray→[Float]` round-trips, no re-pump task — by construction in the new engine; verify with an allocation trace.
- **P4 — warmup completeness** (F2): after warmup, first `speak` must pay generation only — assert warm TTFA ≤ 300 ms in the bench, no lazy fused-QKV eval on the hot path.
- **P5 — prefill/alignment concurrency** (F3): offsets computed concurrently with voice-prefix prefill inside the burst; first chunk never waits on alignment.
- **Opportunistic**: profile one long-form run (os_signpost via the diagnostics tap); if the codec decode dominates unexpectedly, document — do not redesign the vendor decode tonight.

Each landed optimization records before/after numbers in the morning report; anything not landing tonight goes to the report's "measured, not taken" list.

## 7. Implementation plan (tonight's order)

1. Re-vendor v0.1.3 + spike patches (+ smoke tool, listed in ledger) + manifest adaptation; link `MLXAudioTTS`; write `TESSERACT-PATCHES.md`; build green. *(Scratch-verified already: patches apply clean, full release build 258 s.)*
2. New engine module: values + event grammar + session/utterance machinery against the scripted synthesizer (tests first — the grammar, pacing, supersession, and cancellation contracts are pure-logic testable).
3. `Qwen3SpeechSynthesizer` adapter over the vendor: KV-value snapshots (spike-verified `KVCache.state` get/set), anchor-in-request, injected logger, typed knobs.
4. Wire: coordinator rewrite to the new surface; arbiter `.tts` path retirement; playback pause/resume; word-timing ground truth; presenter for engine state; DependencyContainer wiring; delete `SegmentPlayback`, v1 engine surface, env-var reads, diagnostics-dump default-on path.
5. Perf pass per §6, bench before/after.
6. §8 gates + full suite green; runtime smoke + WAV artifacts; push.

## 8. Validation

- **Pinning tests** (vendor-adjacent, survive re-ports): offset-aware causal mask (anchor path produces the mask fix's output shape — the #337 regression that upstream could silently reintroduce); alignment O(n) ≡ quadratic reference on corpus; cache-limit bit-identical before/after bursts; seed reproducibility.
- **Engine contract tests** (scripted synthesizer, no weights): event grammar, gapless frames, supersession ordering, cancellation latency (scripted), lease-never-spans-wait (recording lease), pacing bound, unload-under-active-stream, fingerprint rejection, pinned-voice export/restore round-trip.
- **Machine sanity gates** (real weights, per #339 method): duration = steps × 80 ms ± 5 %; RMS ∈ [0.01, 0.5]; clipping < 0.1 %; silent-window < 25 %; longest silence < 1.2 s flag-and-listen.
- **Budget scorecard** (bench, both precisions): warm/cold TTFA, sustained RTF, peak + idle-warm RSS → feeds the §3 gate.
- **Morning listening checklist** (owner, decisive): (a) anchored-voice consistency across ≥ 5 companion-style utterances in one pinned session — *the ADR-0037 contingency listen*; (b) same-seed A/B new engine vs #339 reference WAVs at shipped precision; (c) one ≥ 5-min long-form read — boundaries, drift, pauses; (d) if the gate forced 6-bit: extended 6-bit fatigue listen (ADR-0037 contingency 2).

## 9. Nothing left to decide — except

Two owner-gated contingencies, both **listen-triggered with pre-decided outcomes** (ADR-0037): anchored-consistency failure → fast role falls back to 0.6B-CustomVoice-8bit (fixed timbre accepted); forced-6-bit fatigue → invoke #338's "ear outranks memory," escalate 8-bit's ~8 % envelope overage as a budget amendment. Everything else in this spec is final pending the owner's morning review of the delegated decisions (ADR-0037/0038/0039 and this lock).
