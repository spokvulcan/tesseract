---
status: accepted
---

# The speech seams sit below the engines and coordinator, not at the module interface

The speech features expose three seams **below** their `@Observable @MainActor`
facades, so each facade's orchestration is testable without models, a microphone, or
`AVAudioEngine`:

- **`SpeechRecognizer`** — the ASR model port below `TranscriptionEngine`.
- **`SpeechSynthesizer`** — the TTS model port below `SpeechEngine`.
- **`AudioPlayback`** — the playback port below `SpeechCoordinator`.

Each has two adapters: a framework-backed one in the app target
(`WhisperKitSpeechRecognizer`, `Qwen3SpeechSynthesizer`, `AudioPlaybackManager` — the only
production code touching WhisperKit / MLX / AVFoundation here) and an in-memory peer in
`tesseractTests` (`InMemory*`). This is the **facade-above / port-below** shape of the
Settings Store (ADR-0002, the direct precedent); the second adapter is what makes the seam
real (ADR-0001's rule). Vocabulary: `CONTEXT.md` → **Language → Speech model ports and
playback**.

The seam is deliberately **not** the module's interface. The app already depends on the
engine and `Transcribing`, which swap the whole engine; these ports sit a layer lower and
swap the *model* under the real engine. That keeps the engine's orchestration — timeout
race, lazy load, `.mlmodelc` verification, the `@Observable` lifecycle read by views and
`InferenceArbiter`, model-failure mapping onto `DictationError` — on a test surface.
Exposing the model port *as* the interface would forfeit exactly that, which was the
point.

The two model ports are `Sendable nonisolated protocol`s satisfied by **actors**: the
engine races `transcribe`/`generate` across a `@Sendable` boundary in a task group, so the
model runs off-main and `Sendable` is free (no `@unchecked`). `AudioPlayback` is shaped
differently on purpose — `@MainActor protocol AudioPlayback: AnyObject`. It wraps
`AVAudioEngine` and is driven synchronously inside the long-form loop (`appendChunk` via
`SegmentPlayback`, `currentPlaybackTime()` polled per segment boundary); an actor would
force `await`/cross-actor hops on that hot path for no gain. So it stays class-bound and
main-actor-isolated like the other collaborator ports (`TextExtracting`, `AudioCapturing`,
`TextInjecting`), with `onPlaybackFinished` a `@MainActor @Sendable` callback that rules
out cross-actor playback calls from the audio layer.

One behavior-neutral refinement: playback diagnostics is a **value, not a mutable
toggle**. A `PlaybackDiagnosticsPolicy` (`.default` / `.disabled`) passed once at
`startStreaming` replaces the old `debugDumpDisabled` flag the coordinator flipped across
`stop()`/long-form — the coordinator no longer mutates playback state to express it.

## Consequences

- The in-memory adapters are **peer implementations, not mocks** — canned results and
  trivial defaults for the surface a test doesn't drive. They live in `tesseractTests`
  (like `InMemorySettingsStore`); the playback peer's clock is test-driven (reads `0`
  until `advance(by:)`, untouched by lifecycle methods), keeping the segment-boundary wait
  loop deterministic rather than gated on real audio.
- `DependencyContainer` relies on the constructor default
  `SpeechCoordinator(playback: … = AudioPlaybackManager())` (mirroring `SettingsManager(store:
  … = UserDefaultsSettingsStore())`) — nothing else in the graph needs the AVFoundation
  adapter, so it is not wired explicitly; tests inject the peer.
- `SpeechSynthesizer` stays one wide port faithful to the model surface, **not** split
  into voice-anchor / alignment ports: it has one real adapter today, so a split would be
  a hypothetical seam.
