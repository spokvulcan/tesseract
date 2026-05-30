---
status: accepted
---

# The speech seams sit below the engines and coordinator, not at the module interface

The speech features now have three seams **below** their `@Observable @MainActor`
facades, so each facade's orchestration is exercisable without models, a microphone,
or `AVAudioEngine`:

- **`SpeechRecognizer`** — the ASR model port below `TranscriptionEngine`.
- **`SpeechSynthesizer`** — the TTS model port below `SpeechEngine`.
- **`AudioPlayback`** — a playback port below `SpeechCoordinator`.

Each seam has two adapters — the framework-backed one in the app target
(`WhisperKitSpeechRecognizer`, `Qwen3SpeechSynthesizer`, `AudioPlaybackManager`; the
former `WhisperActor`/`TTSActor` plus the existing AVFoundation player, now the only
production code touching WhisperKit / MLX / AVFoundation for these features) and an
in-memory peer in `tesseractTests` (`InMemorySpeechRecognizer`,
`InMemorySpeechSynthesizer`, `InMemoryAudioPlayback`). This is the same
**facade-above / port-below** shape as the Settings Store (ADR-0002, the direct
precedent), and a second adapter is what makes each seam real rather than hypothetical
(ADR-0001's supporting rule). Vocabulary for this area is in `CONTEXT.md` →
**Language → Speech model ports and playback**.

The seams are deliberately **not** the modules' public interface. The
coordinator-facing port the rest of the app already depends on is `Transcribing` (and
the engines/coordinator themselves) — that seam swaps the *whole* engine. These new
ports sit a layer lower and swap the *model* under the **real** engine, so the engine's
own orchestration — the timeout race, lazy load, `.mlmodelc` verification, the
`@Observable` lifecycle state read by views and `InferenceArbiter`, and the mapping of
model failures onto `DictationError` — is finally on a test surface. Those concerns
stay *above* the port; the port is model-only and never learns about leases or timeouts.

## AudioPlayback is a `@MainActor` sibling, not an actor-backed model port

The two model ports are `Sendable nonisolated protocol`s satisfied by **actors**: the
engine races `transcribe`/`generate` across a `@Sendable` boundary inside a task group,
so the model is `await`-ed off-main and `Sendable` comes for free (no `@unchecked`).

`AudioPlayback` is shaped differently *on purpose*: it is
`@MainActor protocol AudioPlayback: AnyObject`. Playback wraps `AVAudioEngine` on the
main actor, and `SpeechCoordinator` calls it **synchronously** inside the long-form
loop (`appendChunk`, and `currentPlaybackTime()` polled at each segment boundary).
Making it an actor like the model ports was rejected: it would force an `await` on
every `currentPlaybackTime()` poll and `appendChunk`, turn the synchronous boundary
wait into cross-actor hops, and buy nothing — the work is already main-actor-bound.
It is therefore class-bound and main-actor-isolated, mirroring the existing collaborator
ports (`TextExtracting`, `AudioCapturing`, `TextInjecting`). `onPlaybackFinished` is a
`@MainActor @Sendable` callback so the audio layer's completion can never smuggle in a
cross-actor playback call.

Two behavior-neutral refinements ride on this seam:

- **Diagnostics is a value, not a mutable toggle.** The coordinator used to flip a
  `debugDumpDisabled` flag across `stop()` and the long-form path. That is replaced by a
  domain-neutral `PlaybackDiagnosticsPolicy` (`.default` / `.disabled`) passed once at
  `startStreaming`; long-form passes `.disabled`. The adapter owns the actual dump
  behavior; the in-memory adapter merely records the policy.
- **The in-memory adapter's playback clock is non-wall-clock and test-driven.** It reads
  `0` until a test calls `advance(by:)`, never tracks elapsed time, and is untouched by
  the lifecycle methods — so the long-form segment-boundary wait loop (which polls
  `currentPlaybackTime()`) is deterministic and fast rather than gated on real audio.

## Consequences

An architecture review that re-suggests any of the following should treat them as
already-decided:

- **"Make `AudioPlayback` an actor for consistency with the model ports"** — no. It is
  called synchronously on the main actor inside the long-form loop; an actor forces
  `await`/cross-actor hops on a hot polling path for no isolation gain. Keep it
  `@MainActor` / `AnyObject`.
- **"Expose the model port as the speech module's interface"** — no. The engine-facing
  seam is the engine (and `Transcribing`); the model port is *below* it. They are
  different layers — collapsing them loses the engine-orchestration test surface that
  was the whole point.
- **"Split `SpeechSynthesizer` into voice-anchoring / alignment ports"** — no. Each has
  exactly one real adapter today, so a split would be a hypothetical seam. It stays one
  wide port faithful to the model surface.
- **"Re-introduce a mutable debug toggle on playback"** — no. Diagnostics intent is a
  value passed at `startStreaming`; the coordinator no longer mutates playback state to
  express it.
- **"Drop the constructor production default and wire playback explicitly in the
  container"** — no. `SpeechCoordinator(playback: any AudioPlayback = AudioPlaybackManager())`
  mirrors `SettingsManager(store: … = UserDefaultsSettingsStore())`. The AVFoundation
  adapter is needed by nothing else in the graph, so `DependencyContainer` relies on the
  default (as it already does for `SettingsManager()`); tests inject the in-memory peer.

The in-memory adapters are **peer implementations, not mocks** — they return canned
results and trivial defaults for the surface a given test does not drive. They live in
`tesseractTests` (like `InMemorySettingsStore`) unless previews or dev tooling later
need them in the app target.
