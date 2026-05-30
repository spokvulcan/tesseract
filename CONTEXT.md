# Context

Single source of truth for Tesseract's domain language and the *why* behind the
system. Keep this narrative current; record discrete decisions as ADRs in
`docs/adr/`.

> Stub — populate as the domain model stabilizes. See `ARCHITECTURE.md` for
> structure and `docs/adr/` for decisions.

## Domain

_TODO: the ubiquitous language — Agent, Session, Prefix Cache, TriAttention,
Package, Tool — defined in one place._

## Language

### Prefix cache snapshot lifecycle

**Snapshot State**:
The value attached to each radix-tree node encoding which tier(s) hold its KV-cache
snapshot and the snapshot's write phase. A six-case enum (`empty`, `ramOnly`,
`pendingWrite`, `pendingDropped`, `committed`, `ssdOnly`) that owns *both* the RAM
body and the **Snapshot Ref**; transitions return a **State Effect**.
_Avoid_: storage-ref lifecycle, slot, residency, the "five-state table".

**Snapshot Ref**:
The immutable on-disk identity of a snapshot — `snapshotID`, `partitionDigest`,
`tokenOffset`, `checkpointType`, `bytesOnDisk`. Carried as the payload of the
ref-bearing **Snapshot State** cases. Knows *what and where on disk*, never the
write phase (that is the enum case).
_Avoid_: SnapshotStorageRef, storage ref, descriptor.

**State Effect**:
The topology-only outcome a **Snapshot State** transition reports to its caller:
`settled`, `becameEmpty`, or `ignored(reason)`. `becameEmpty` carries **no payload**
and is the only signal that triggers the tree's self-heal (which then removes the node
only if topology allows). The dropped snapshot ID is **not** on `becameEmpty` — it
rides on `DropBodyResult` instead, because the common body drops (states 2/4) are
`settled`, not `becameEmpty`. `ignored(reason)` is propagated (not precondition'd) only
on the two forgiving SSD-writer callback edges (`commit`, `dropRef`).
_Avoid_: transition result, mutation outcome.

**dropRef**:
The forgiving SSD-writer callback edge for a pending **Snapshot Ref**. It is distinct
from committed-ref cleanup after a hydration failure; pending refs are dropped by the
writer callback, committed refs are cleared only after a failed SSD hydration.
_Avoid_: clear ref, remove ref.

**Committed Ref Cleanup**:
The strict cleanup edge after a failed SSD hydration of a committed **Snapshot Ref**.
It applies only to committed-ref states; pending refs are handled by `dropRef`, not
by hydration cleanup.
_Avoid_: generic ref clear, storage-ref cleanup.

**Explicit Ref Discard**:
The strict cleanup edge after the SSD backing has already been explicitly deleted or
cancelled, such as during leaf supersession. It may discard any ref-bearing
**Snapshot State** because the caller has already removed the backing snapshot.
_Avoid_: hydration cleanup, generic ref clear.

**canEvictNode**:
The load-bearing invariant query on **Snapshot State**: true iff the node holds no
live **Snapshot Ref**, i.e. removing the node structure cannot orphan an
SSD-resident snapshot. Distinct from `hasResidentBody` (the RAM-budget concept) —
a node may be node-removable yet still hold a useful RAM body.
_Avoid_: canRemove, isOrphanable.

**hasResidentBody**:
The RAM-budget query on **Snapshot State**: true iff the node holds a RAM body
that can be dropped to free memory. Distinct from `canEvictNode` — body eviction
may leave a node in place when a **Snapshot Ref** still pins an SSD-resident
snapshot.
_Avoid_: body-removable, resident snapshot.

> **Flagged ambiguity — "State".** `SnapshotState` is the prefix-cache lifecycle
> enum. It is unrelated to `HybridCacheSnapshot.LayerState` (MLX layer tensors) and
> to `@Observable` view/app state. When unqualified "state" is ambiguous, say
> "snapshot state" or "layer state".

**Example dialogue:**

> **Dev:** When the SSD writer's drop callback fires, who removes the node?
> **Expert:** Nobody removes it directly. The callback routes through the tree,
> the tree applies `dropRef`, and the *state* decides. If the drop leaves a RAM
> body, the state settles to `ramOnly` — node stays. If there's nothing left, the
> transition returns `becameEmpty` and the tree self-heals: it removes the node.
> **Dev:** So the eviction loop also checks `becameEmpty`?
> **Expert:** No — that loop is proactive, it picks LRU victims with
> `hasResidentBody`, then drops the RAM body. `becameEmpty` is reactive cleanup
> after a drop, and `canEvictNode` is the structural-removal invariant. Different
> predicates: a `ramOnly` node has both `hasResidentBody` and `canEvictNode`, while
> a committed node with RAM has `hasResidentBody` but not `canEvictNode`.

### Settings persistence

**Settings Store**:
The seam between *what a setting means* and *where its bytes live* — a typed
key-value persistence port with default-on-read semantics. Exposes typed getters
that carry the default (`bool(for:default:)`, `int(for:default:)`, …), typed
setters, and `setOptional` (writing `nil` removes the key). Has no
`register(defaults:)` step: the default travels with every read. Satisfied by two
**Settings Store Adapters**.
_Avoid_: SettingsManager (that is the **Settings Facade** above it), UserDefaults
(that is one adapter), preferences store.

**Settings Store Adapter**:
A concrete **Settings Store**. Exactly two exist: `UserDefaultsSettingsStore` (the
app — the only production Swift code that calls `UserDefaults`; the privacy
manifest still declares the API) and `InMemorySettingsStore` (tests — a
dictionary; hermetic and parallel-safe). Two adapters are what make the seam
real rather than indirection. The UserDefaults adapter owns **default-on-read**
(there is no `register(defaults:)`): a missing key returns the passed default, so
it must check `object(forKey:) == nil` rather than trust `bool`/`integer`, which
coerce a missing key to `false`/`0`.
_Avoid_: backend, provider, mock (the in-memory one is a peer implementation, not
a mock).

**Setting**:
The single immutable declaration of one persisted setting — its key, its one
canonical default, and its codec to a stored primitive. The sole source of truth
for that setting's default, consumed by both initial load and reset.
_Avoid_: preference, key, default (a **Setting** *has* a key and a default; it is
neither).

**Settings Catalogue**:
The table of all **Setting** declarations. Replaces the former triplication
(stored-property literal + `register(defaults:)` + `resetToDefaults`) so each
default has exactly one home — the drift that left `prefixCacheSSDBudgetBytes` at
50 GiB in one place and 20 GiB in two others becomes unrepresentable.
_Avoid_: defaults dictionary, schema, registry.

**Settings Facade**:
The `@Observable @MainActor SettingsManager`. Keeps one stored property per
setting — so SwiftUI `$settings.foo` bindings and per-property Observation survive
— and forwards each `didSet` to the **Settings Store**. Non-persistence side
effects (launch-at-login via `SMAppService`, dock visibility via `NSApp`) live in
the facade's `didSet`, *above* the store; the store moves bytes and never learns
what a setting means.
_Avoid_: settings service, settings model.

> **Flagged ambiguity — "store".** The **Settings Store** is the settings
> persistence seam. It is unrelated to `SnapshotStore`/`SSDSnapshotStore` (the
> prefix-cache tiers). When unqualified, say "settings store".

**Example dialogue:**

> **Dev:** Where does the SSD-budget default live now?
> **Expert:** In its **Setting** in the **Settings Catalogue** — once. Both the
> initial load and `resetToDefaults` read it from there, so the 50-vs-20 GiB drift
> can't recur.
> **Dev:** And when a view flips `$settings.playSounds`?
> **Expert:** That writes the facade's stored property — Observation invalidates
> only the views that read it, exactly as before — and the `didSet` forwards the
> value to the **Settings Store**. In the app that's the UserDefaults adapter; in a
> test it's the in-memory adapter, so you assert persistence without touching
> `UserDefaults.standard`.
> **Dev:** Is launch-at-login in the store?
> **Expert:** No. That side effect stays in the facade's `didSet`, above the store.
> The store only persists; the **Settings Facade** owns the side effect.

### Speech model ports and playback

**Speech Recognizer**:
The model port (seam) for ASR, sitting *below* the `TranscriptionEngine` facade.
Surface: `load(modelPath:)` (a local `.mlmodelc` folder URL),
`transcribe(_:language:)`, `cancel()`. The port is itself `Sendable`: the engine
races `transcribe` against the timeout inside a `withThrowingTaskGroup`, so the
recognizer crosses a `@Sendable` boundary. Above the seam: the engine owns the
timeout race, lazy `ensureModelLoaded`, model-file (`.mlmodelc`) verification, the
`@Observable` lifecycle state (`isModelLoaded`/`isTranscribing`) read by views and
`InferenceArbiter`, and the mapping of model failures onto `DictationError`. Below
the seam: the model only. What crosses: `AudioData` in, `TranscriptionResult` out
(both `Sendable`). The engine holds an injected `@Sendable` factory; on load it
builds a *fresh* adapter and on unload it drops it to `nil` to release model
memory. **Cancellation must reach the adapter**: the engine retains its in-flight
work so `cancelTranscription()` actually cancels it and calls the port's `cancel()`
(today `cancelTranscription` clears a `transcriptionTask` that is never assigned —
a latent no-op the seam fixes). Satisfied by two **Speech Model Adapters**.
_Avoid_: WhisperActor (that is one adapter), Transcribing (that is the
**engine-facing port** the coordinator depends on), transcriber, ASR backend.

**Speech Synthesizer**:
The model port (seam) for TTS, below the `SpeechEngine` facade. `Sendable`, and
deliberately *faithful* to the model surface — `load(modelRepo:)` (a model-repo
identifier string), `generate`, `generateStreaming` (yields an
`AsyncThrowingStream<[Float]>` plus the model `sampleRate`), `buildVoiceAnchor` /
`clearVoiceAnchor`, `cancelGeneration`, `computeTokenCharOffsets`. The
synthesizer's lifecycle state (`isModelLoaded`/`isLoading`/`loadingStatus`) stays
on the `SpeechEngine` facade. It stays one wide port rather than splitting
voice-anchoring or alignment into their own ports, because each of those has only
one real adapter today (one adapter ⇒ hypothetical seam). Satisfied by two
**Speech Model Adapters**.
_Avoid_: TTSActor (one adapter), SpeechEngine (the facade above it), TTS backend.

**Speech Model Adapter**:
A concrete **Speech Recognizer** or **Speech Synthesizer**. Exactly two of each:
the framework-backed adapter (`WhisperKitSpeechRecognizer`,
`Qwen3SpeechSynthesizer` — today's `WhisperActor`/`TTSActor`, the only production
code that touches WhisperKit/MLX for these features) and the in-memory adapter
(`InMemorySpeechRecognizer`, `InMemorySpeechSynthesizer` — hermetic, no model
files). The in-memory adapters live in `tesseractTests` (like
`InMemorySettingsStore`), not the app target, unless previews or dev tooling later
need them. They are **actors** (like the production `WhisperActor`/`TTSActor`), so
`Sendable` is free and the call-recording / cancellation / latency / canned-output
state is actor-isolated — tests `await` it; no `@unchecked Sendable`. Two adapters
are what make the seam real rather than indirection; the in-memory one is a peer
implementation, not a mock — it returns canned results and trivial defaults for the
surface a given test does not drive.
_Avoid_: mock, stub, fake-only (it is a real peer adapter); model wrapper.

**Audio Playback**:
A `@MainActor` *sibling* seam to the model ports — not a model port — that
`SpeechCoordinator` depends on for turning generated samples into sound. It is
`@MainActor protocol AudioPlayback: AnyObject` because it wraps `AVAudioEngine` on
the main actor and the `@MainActor` coordinator calls it *synchronously* (e.g.
`appendChunk`, `currentPlaybackTime()` inside the long-form loop) — unlike the
model ports, which are actor-backed and `await`-ed off-main. Surface:
`play(samples:sampleRate:)`, `startStreaming(sampleRate:diagnostics:)`,
`appendChunk(samples:)`, `finishStreaming()`, `stop()`, `currentPlaybackTime()`,
`totalScheduledDuration`, and `onPlaybackFinished` (a `@MainActor @Sendable`
callback). It carries **no
mutable debug toggle**: instead of the coordinator flipping `debugDumpDisabled`
across `stop()`/long-form, diagnostics intent is a *value passed at*
`startStreaming` — a domain-neutral `PlaybackDiagnosticsPolicy` (`.default` /
`.disabled`). Long-form passes `.disabled`; the adapter owns the actual dump
behavior; the in-memory adapter ignores it. Two adapters: the AVFoundation adapter
(today's `AudioPlaybackManager`, which news-up a real `AVAudioEngine`; a
`@MainActor final class`) and an in-memory adapter (in `tesseractTests`, also a
`@MainActor final class` — `Sendable` via main-actor isolation, no
`@unchecked`) that records scheduled samples and advances a *virtual playback
clock* — without it, the long-form loop's `currentPlaybackTime()` polling never
terminates. The clock is **non-wall-clock and test-driven**: it advances only when
the test calls `advance(by:)` (and `currentPlaybackTime()` reports that scheduled
position), so the long-form segment-boundary wait loop is deterministic and fast,
never gated on real elapsed time. Introduced because the model seam alone does not make
`SpeechCoordinator`'s long-form loop hermetic: playback is the other concrete
dependency. The coordinator gains this as a constructor seam; its logic stays
behavior-neutral.
_Avoid_: AudioPlaybackManager (that is one adapter), AVAudioEngine (inside it),
CoreAudio, audio output, player.

> **Flagged ambiguity — "Transcribing" vs "Speech Recognizer".** `Transcribing`
> is the *engine-facing* gerund port that `DictationCoordinator` depends on — it
> lets a test swap the whole engine. **Speech Recognizer** is the *model-facing*
> noun port *below* the engine — it lets a test swap the model under the **real**
> engine, so the engine's own orchestration (timeout, lazy load, file check) is
> finally on a test surface. Different seams, different layers. The same split
> holds for `SpeechEngine` (facade) vs **Speech Synthesizer** (model port). GPU
> arbitration (`InferenceArbiter`) and model-file verification stay *above* the
> port; the port is model-only and never learns about leases or timeouts. This is
> the same facade-above / port-below shape as the **Settings Store** (ADR-0002),
> the direct precedent. (ADR-0001 is about a *different* seam — SSD hydration —
> but supplies the supporting rule that a second adapter is what makes a seam real
> rather than hypothetical.)

**Example dialogue:**

> **Dev:** Where does the transcription timeout live now?
> **Expert:** In the `TranscriptionEngine` — the facade. It still races the call
> against the budget; the **Speech Recognizer** below it just transcribes. That's
> the point: inject an `InMemorySpeechRecognizer` that sleeps and you can assert
> the timeout fires without a gigabyte of WhisperKit on disk.
> **Dev:** And unloading to free memory?
> **Expert:** The engine drops the adapter to `nil` — its `@Sendable` factory
> builds a fresh one on the next load. Same memory behavior as today's
> `whisperActor = nil`, now behind the seam.
> **Dev:** Does the **Speech Recognizer** know about the GPU lease?
> **Expert:** No. The `InferenceArbiter` drives `loadModel`/`unloadModel` on the
> engine, above the port. The port is model-only.

## Why

_TODO: the constraints that shape the system (fully offline, on-device MLX,
privacy-first, Apple Silicon) and the trade-offs they force._
