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

**Snapshot Admission**:
The write-side decision to place already-captured KV-cache snapshots into the
Prefix Cache. It pairs each `HybridCacheSnapshot` with the optional
`SnapshotPayload` extracted inside the Metal-affine `ModelContainer.perform`
scope, makes RAM-only versus RAM+SSD admission explicit, and carries the
partition/token context needed by the cache. It is the unified write shape for both
mid-prefill checkpoints and leaf snapshots; leaf supersession is a cache-manager
effect of applying a leaf admission, not a separate caller-facing write path. The
cache manager still owns radix-tree insertion, **Snapshot Ref** creation, SSD
partition registration, eviction, and leaf supersession; **Snapshot Admission**
owns the caller-facing shape so positional payload alignment and "empty array means
SSD disabled" do not leak to call sites. Invalid write shapes should be
unrepresentable: a caller should not be able to express "payload missing for this
SSD admission", "payload count differs from snapshot count", or "leaf tokens do not
match the leaf snapshot offset" and rely on the cache manager to recover after
partial mutation. Public Snapshot Admission construction should be total once the
caller has valid inputs; the only throwing/failable factory belongs at the MLX
extraction edge where `SnapshotPayload` bytes are produced inside
`ModelContainer.perform`; that factory also performs **Snapshot Admission Path**
validation and absorbs unsupported-cache `nil` capture results. The cache manager
should receive already-valid admissions, not validate caller mistakes. The cache-facing
interface is one synchronous `admit` operation returning store diagnostics over
**Snapshot Admission**; checkpoint versus leaf behaviour is encoded in the admission
value and handled inside the cache manager, where leaf supersession already belongs.
For mid-prefill checkpoints, build the admission at this extraction edge during
prefill and carry it as one optional value on the generation handle until the
post-stream `admit`; do not carry parallel snapshots/payloads or reconstruct the
admission late. Sandbox replay writers such as **Alpha Tuner** should also use
RAM-only **Snapshot Admission** rather than keeping old write entry points, while
their scoring behaviour remains out of scope.
The request ID used for diagnostics and eviction correlation is part of the admission
value, not a side-channel argument to `admit`. Token-path and offset validation belong
in a small **Snapshot Admission Path** value so "this snapshot can be stored at this
token path" has a name and a focused test surface. RAM-only versus RAM+SSD is encoded
per admitted snapshot, not as an admission-wide mode; mid-prefill admission may mix
storage cases if needed, and leaf admission remains the same single-entry shape. To
preserve current edge behavior, the single extraction-edge factory validates checkpoint
entries, drops invalid entries internally, builds from the surviving non-empty set, and
returns no admission when none survive; an invalid leaf path produces no leaf admission
at all. Snapshot Admission values cross from the
model execution scope to the MainActor cache manager, so they must be `Sendable` and
nonisolated under the app's default MainActor isolation.
_Avoid_: capturedPayloads plumbing, payload alignment, storeSnapshots payloads,
empty payload array, SSD write call-site wiring.

**Snapshot Admission Path**:
The token path carried by a **Snapshot Admission**, validated against the
snapshot offset before cache mutation begins. For mid-prefill checkpoints it names
one shared full prompt token sequence plus per-checkpoint validated prefix views; for
leaf snapshots it names the full stored token sequence and proves the leaf snapshot
offset equals the token count. Checkpoint admission should not duplicate token
arrays per entry: store full prompt tokens once, then give each admitted checkpoint
its own **Snapshot Admission Path** into that shared sequence. It exists to keep
offset checks out of cache-manager mutation code and out of call sites. Empty
checkpoint admissions are not representable; callers skip `admit` when no snapshots
were captured. Prefer a small non-empty collection value for checkpoint entries if
the Swift ergonomics stay simple; otherwise a precondition at the public constructor
is acceptable, but the cache manager should never receive an empty checkpoint
admission.
_Avoid_: promptTokens, storedTokens, path prefix slicing, offset guard.

**Snapshot Resolution**:
The read-side counterpart to **Snapshot Admission**: the operation that resolves a
token path to the best usable `HybridCacheSnapshot` for restore. It looks up the
deepest reachable radix node for the path under a `CachePartitionKey`, and when that
node carries only a committed **Snapshot Ref** with no RAM body (the `.ssdHit` lookup
reason), it hydrates the body from disk via the off-MainActor `loadSync` (ADR-0001),
then either *promotes* the node back to a resident state on success or applies
**Committed Ref Cleanup** on failure and downgrades to a miss. On the main prefill path
it is the home for the lookup-then-hydrate-if-SSD composition, replacing the dance that
was previously hand-written there. The canonical-leaf fallback still hydrates inline
through `LLMActor.hydrateSSDLookupIfNeeded` — a near-duplicate of the same composition,
slated to converge onto resolution next (see **Leaf Admission Builder**) — so it does not
yet acquire its snapshot through here. `.ssdHit` is an internal intermediate the resolution
consumes and never surfaces: callers receive a resolved `LookupResult` whose reason is
only `.hit` or a miss. Checkpoint planning is **not** part of resolution — planning runs
*after* resolution against the settled tree, so a post-hydration-failure replan is the
ordinary single plan, not a special "replan after clear" branch. Resolution crosses
isolation the same way **Snapshot Admission** does: the lookup and the promote/clear
tree mutations are MainActor, while the `loadSync` disk read stays off-MainActor inside
`container.perform` per ADR-0001; it is therefore a `nonisolated` operation driven from
inside the Metal-affine scope that hops to MainActor for the tree work.
_Avoid_: hydrator / SSD hydration as a standalone unit (the dance is only half of it —
resolution owns lookup *and* hydrate as one composition), folding planning into resolution
(planning is a separate `planCheckpoints` step run *after* resolution now), restore / restoreCache (that is the model-affine step
that loads the *resolved* snapshot — see the ambiguity note), boundary resolution (it
resolves a snapshot for any read, not only leaf boundaries).

> **Flagged ambiguity — "resolve" vs "restore".** **Snapshot Resolution** is the
> read-side *find + hydrate* that produces a usable `HybridCacheSnapshot` from a token
> path. `restoreCache` is the later model-affine step that loads that snapshot's layers
> into a live `[any KVCache]` for suffix prefill. Resolution decides *which snapshot*;
> restore *applies* it. When unqualified, say "resolve the snapshot" vs "restore the
> cache".

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

### Prefill orchestration

**Prefill Plan**:
The set of pre-prefill decisions for one HTTP prefix-cache generation, produced by the
**Prefill Planner** and read by `LLMActor` to drive the model-affine prefill. It names: a
`Restore` decision (`.cold` for a miss → full prefill, or `.restore(cacheOffset:)` for a hit
→ suffix prefill from the offset the resolved snapshot covers); the
checkpoint offsets to capture, already filtered to the suffix; the transient boundary
*offsets* (last-message and last-user) to capture during this prefill; and the stable-prefix
offset. It carries **offsets, not snapshots** — the boundary and checkpoint snapshots are
post-prefill artifacts the actor lifts off the `TokenIterator` onto the generation handle;
the plan is the pre-prefill intent computed *from* the flat token sequence, which it never
re-carries. `skippedTokens` and `checkpointBaseOffset` are the same number (the restore
offset, `0` on a miss) and exist as one derived `prefillBaseOffset`, not two stored fields.
The **Prefill Planner** is tokenizer-affine, not model-affine: it runs the boundary probes
(`StablePrefixDetector`, the generation-prompt suffix subtraction, the last-user re-render)
through an injected `any Tokenizer`, and consumes an already-resolved `LookupResult` (see
**Snapshot Resolution**) plus the checkpoint plan as input *values* — it never touches the
GPU model, so it tests against a fake tokenizer with no model files. The actor keeps every
model-affine step (processor `prepare`, the `LMInput` slice, `restoreCache`, the iterator,
snapshot extraction).
_Avoid_: prefill config, generation params (that is `GenerateParameters`, which the plan
configures), the planner owning lookup/hydration (resolution runs *before* the planner and
feeds it a value), carrying the full token array on the plan (it lives on the handle).

**Leaf Admission Builder**:
The GPU-free routing core for storing one leaf snapshot. Today it owns the token-path
**reusable-prefix probes**: given a stored conversation it renders the turn in isolation
and again with one synthetic continuation appended — a user turn for the *canonical* mode
(`.userTurn`), a tool result for the *directTool* mode (`.toolResult`) — and returns the
shared token prefix the immediate continuation can hydrate, or `nil` when the probe
diverges. It is tokenizer-affine (an injected `any Tokenizer`, no model), so it tests
against a fake tokenizer with no model files. Every model-affine and live-cache step stays
in `LLMActor`: leaf-mode selection (`selectHTTPLeafStoreMode`), boundary acquisition, the
Metal capture, **Snapshot Admission** construction at the edge, and `admit`.

_Planned (deferred from the prefill carve, not yet built):_ promoting the builder's output
to a three-case **Leaf Capture Plan** value — `.liveCache(storedTokens:)` (the *directLeaf*
mode — snapshot the live final KV cache at `storedTokens.count`), `.fromBoundary(boundary:storedTokens:)`
(the *directTool* and *canonical* modes — restore a boundary snapshot, reprefill the
residual `storedTokens[boundary.tokenOffset...]`, capture a leaf), or `.skip(reason:)` — so
the builder also owns mode selection and boundary acquisition (the transient boundary
snapshot when usable, else falling back through **Snapshot Resolution** on the canonical
path). It would emit `.skip(reason:)` only for tokenizer/resolver-decidable failures
(tokenization failed, no common prefix, missing transient boundary, no usable resolved
snapshot); the live-`finalCache` skips (`no-final-cache`, `no-reusable-cache-state`,
`normalization-trim`, `unsupported-cache-type`, `invalid-path`, `capturedThenEvicted`) and
the `intervention` guard stay actor-side. Until that lands, the *canonical* fallback
resolves its boundary through `LLMActor.hydrateSSDLookupIfNeeded`, not through the builder
or **Snapshot Resolution**.
_Avoid_: leaf store mode as the whole story (mode is one of its inputs), the builder owning
capture or `admit` (model-affine, actor-side), a capture port (the builder returns a
decision; the actor executes Metal — no behaviour-less fake seam).

> **Flagged ambiguity — "plan": prefill plan vs checkpoint plan.** The **Prefill Plan** is
> the whole pre-prefill decision value. The *checkpoint plan* is one field inside it — the
> `[(offset, type)]` list from `PrefixCacheManager.planCheckpoints`, filtered to the suffix.
> When unqualified, say "prefill plan" vs "checkpoint plan".

**Example dialogue:**

> **Dev:** The planner needs the tokenizer but not the model — how does it stay model-free
> if tokenization happens inside `container.perform`?
> **Expert:** It takes `any Tokenizer`, not the `ModelContext`. The actor still calls
> `processor.prepare` and slices the `LMInput` — those build the MLX tensor the GPU eats.
> The planner only runs the `[Int]`-returning probes and the boundary arithmetic, so a fake
> tokenizer drives it with no model files.
> **Dev:** And the leaf builder — why not give it a capture port so `build()` returns the
> finished admission?
> **Expert:** Because that fake would exercise no behaviour — a mock, not a peer. The design
> has the builder return a **Leaf Capture Plan**; the actor runs the Metal capture and builds
> the **Snapshot Admission** at the edge, exactly where it does today. The routing — mode,
> probe, boundary source, skip — is what's worth testing, and that needs only the tokenizer
> and a **Snapshot Resolution** peer. (Today only the probe half is carved out — see the
> _Planned_ note above; the full plan value is the next step.)

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

### Generation accumulation

**Generation Accumulator**:
The single home for folding an `AgentGeneration` event stream into the accumulated
content of one assistant turn — `text`, optional `thinking`, `[ToolCall]`, the raw
malformed-tool-call buffer, and the safeguard safe-prefix length. A
`nonisolated Sendable` value type with one `mutating func ingest(_:)`; it owns the
subtle transitions that were previously hand-copied into five consumers
(`LLMActor.handle`, both `CompletionHandler` paths, `InternalInferenceRouting`, the
agent double-loop) and had already drifted into a bug: `.thinkReclassify` **appends**
buffered thinking onto text, `.thinkTruncate` resets thinking to the safe prefix,
`.thinkStart`/`.thinking` lazily open the optional thinking buffer, `.thinkEnd` is a
no-op, `.malformedToolCall` accumulates raw, `.toolCall` appends the event's raw
`ToolCall`. It holds **no** side effects, control flow, or output type: each caller
keeps its own `for await` loop, its per-event side effects (SSE chunking,
`emit(.messageUpdate)`, `continuation.yield`), its control flow (the safeguard
`break` + continuation swap; cancellation emits), and its **Generation Projection**.
The deletion test passes: remove it and the reclassify/truncate/thinkStart rules
reappear across five call sites.
_Avoid_: StreamResult (one caller's output type, not the accumulator), event handler,
GenerationFold (names the operation; we name the value), parser/`ToolCallParser` (that
is *upstream* — it produces the events the accumulator folds).

**Generation Projection**:
The per-caller mapping from a **Generation Accumulator**'s state to that caller's
output shape — `AssistantMessage` (agent loop), `StreamResult` plus the HTTP replay
message (server), the leaf-store `HTTPPrefixCacheMessage` (`LLMActor`), or just `text`
(the summarizer). The projection is where a caller's *intent* lives: the summarizer
ignores `thinking`/`toolCalls`; the server's malformed→text fallback fires only when
the turn is otherwise empty; the agent loop assigns stable tool-call identities.
Projections stay out of the accumulator so adding a consumer never edits the shared
fold.
_Avoid_: conversion, adapter (it is not a seam adapter), output builder.

> **Flagged ambiguity — `thinking` nil vs "".** `thinking == nil` means no `<think>`
> block was ever opened (so no thinking row renders); `thinking == ""` means a block
> opened but has produced no content yet. Consumers that do not need the distinction
> read `thinking ?? ""`. Do not collapse the optionality.

> **Flagged invariant — reclassify appends.** On `.thinkReclassify` (generation ended
> inside an unclosed `<think>`), the one rule is `text += (thinking ?? "")` — buffered
> thinking goes *after* any text the model emitted before opening the block. The agent
> double-loop previously prepended (`thinking + text`), reversing output whenever text
> preceded the block; the **Generation Accumulator** makes append canonical.

**Example dialogue:**

> **Dev:** The server streams SSE while the agent loop builds an `AssistantMessage`.
> Do they share the accumulator?
> **Expert:** They share the *fold*, not the loop. Each keeps its own `for await`: the
> server sends an SSE chunk per event, the loop emits a `messageUpdate`. Both call
> `accumulator.ingest(event)`, so the text/thinking/tool-call state — and the reclassify
> and truncate rules — come from one place.
> **Dev:** Where does the summarizer's "text only, drop the reasoning" go?
> **Expert:** That is a **Generation Projection**, not an accumulation rule. The
> accumulator faithfully keeps `thinking`; the summarizer returns `accumulator.text` and
> ignores it. Nothing about "I don't want reasoning" leaks into the fold.
> **Dev:** And if a `<think>` block never closes?
> **Expert:** `.thinkReclassify` fires and the accumulator appends the buffered thinking
> onto the text — append, never prepend. That used to be wrong in the agent loop; now
> there is one rule and one regression test.

### Generation stream loop

**Generation Stream Loop**:
The single home for consuming one raw model `AsyncStream<Generation>` and producing the
agent's `AgentGeneration` event stream under the thinking-loop safeguard. A `nonisolated`
value (`GenerationStreamLoop`) constructed from the initial handle and driven by one
`run(continuation:sink:) async throws -> Outcome`; it consumes the vendor `Generation` stream
directly (no mirror type). It owns the `while` loop over the four raw cases
(`chunk`/`info`/`toolCall`/`toolCallBufferDelta`), the `ToolCallParser` lifecycle
(`processChunk`/`finalize`, the `libraryParsedToolCalls` suppression, and the
`toolCallBufferDelta` accumulation that surfaces a silently-dropped `<tool_call>` as
`.malformedToolCall` — now for both callers, previously server-only), the `ThinkingSafeguardObserver` step with its intervention triple
(`.thinkTruncate` → `.thinking(injection)` → `.thinkEnd`), and the **continuation swap**:
cancel the current raw handle, await it, call `continuationStarter(safePrefix)`, swap in the
new stream, and re-init the parser out-of-think so post-`</think>` output classifies as text.
It owns **`cancelCurrent`** — the one place an external cancel reaches whichever raw handle is
live *across* swaps, the invariant the two hand-copied loops kept re-deriving. It folds the
terminal `.info` into `Outcome.completionInfo` and never pushes it through the `sink`; the
`sink` sees content events only. It holds **no** per-caller side effects, output stream, or
projection: each caller passes its own inline `sink` (the server path's `handle` folds a
**Generation Accumulator**, projects `.toolCall → HTTPPrefixCacheToolCall`, then yields; the
agent path just yields, folding later in `AgentLoop`) and does its post-loop work from the
`Outcome` (`intervened`, `completionInfo`, `cancelled`, plus an `Outcome.diagnostics` group —
joined raw chunks, the post-`finalize()` parser snapshot, and `libraryParsedToolCalls` — that
feeds the server's silent-close warning and the agent ignores).
It sits one layer *above* the
**Generation Accumulator** (which a `sink` may call) and *below* each caller's **Generation
Projection**. It is driven from both an actor (`LLMActor`) and the MainActor (`AgentEngine`),
so it is `nonisolated`, the `sink` is called inline (not `@Sendable`), and only
the continuation-starter port is `@Sendable` (it hops to the actor). The deletion test passes: remove
it and the four-case switch, the parser-suppression rules, and the
cancel→wait→continuation→swap→re-init protocol reappear in both
`AgentEngine.wrapManagedGeneration` and `LLMActor.generateServerTextCompletion`.
_Avoid_: managed generation (that is `AgentEngine`'s `@MainActor` cancellable wrapper), stream
consumer / generation pump, GenerationFold (names the operation; the fold is the **Generation
Accumulator**), ToolCallParser (that is *upstream* — it produces the parser events the loop
drives).

> **Flagged ambiguity — "loop": stream loop vs agent loop.** The **Generation Stream Loop**
> consumes one raw model stream — with safeguard continuation swaps — for a single assistant
> turn. The agent double-loop (`AgentLoop`) is the outer/inner orchestration across many turns
> and tool calls. When unqualified, say "stream loop" vs "agent loop".

**Example dialogue:**

> **Dev:** The thinking-loop safeguard fires mid-stream. Who restarts generation?
> **Expert:** The **Generation Stream Loop**. It emits the truncate / injection / `</think>`
> triple to the `sink`, cancels the current handle and awaits it, then calls
> `continuationStarter` with the safe prefix and swaps in the new stream — re-initialising the
> parser out-of-think. The agent path and the server path each used to hand-write that dance;
> now there is one.
> **Dev:** And if the client cancels right after the swap?
> **Expert:** `cancelCurrent` targets the post-swap handle, not the original `mlxStart`. That
> cross-swap reachability is the invariant the two copies kept re-deriving; it lives in one
> place now.
> **Dev:** Does it fold the stream into an `AssistantMessage`?
> **Expert:** No — it *produces* the `AgentGeneration` events; it folds nothing. The **Generation
> Accumulator** the server `sink` calls does the fold, and the agent path folds later in
> `AgentLoop`. The loop owns control flow and the safeguard; the fold and the **Generation
> Projection** stay with the caller.

### Chat transcript projection

**Chat Transcript**:
The pure projection of the agent **message log** (`agent.state.messages`, the single
source of truth) into the flat `[ChatRow]` the chat list renders, grouped into
**Turn**s. A stateless namespace of two pure functions: `turns(from:)` applies the
grouping rule (a **Turn** boundary is a user message or a compaction marker), and
`rows(for:_:)` folds one **Turn** plus a `Context` of inputs into that Turn's rows.
It reads **no** coordinator state and has **no** side effects: expansion state, the
live `streamMessage`, and a timestamp formatter are passed *in* as `Context`. The
coordinator maps `rows(for:_:)` over every **Turn** for a full rebuild and over just
the last **Turn** for the streaming tail-patch, then splices onto the stable prefix —
so the fast path projects only the active Turn (avoiding `ToolDisplayHelpers.displayProps`
for all history) while sharing one fold. Pruning stale expansion entries and the
"auto-expand the streaming header unless the user collapsed it" decision are explicit
coordinator steps computed *before* the call, never side effects of projecting (they
used to mutate `expandedTurns` from inside a row emitter). Distinct from **Generation
Projection**: that maps one turn's **Generation Accumulator** state to a caller's
output (e.g. `AssistantMessage`); the **Chat Transcript** projects the *whole* committed
log plus the live stream into the rendered row sequence — a different layer.
_Avoid_: rebuildRows / patchStreamingTail (the two duplicated method bodies it
replaces), row builder, ChatRowBuilder, view model, render model.

**Turn**:
The **Chat Transcript**'s grouping unit — a contiguous run of messages from one user
message (or compaction marker) through the assistant's complete response, up to the
next user/compaction message. A single **Turn** may contain several assistant messages
— one per agent-loop `turnEnd` — when a tool-calling loop runs; the Turn projects them
into thinking / tool-call / intermediate-text rows plus one final answer row.
_Avoid_: round, exchange, conversation turn, message group.

**Chat Row**:
The flat, pre-computed, `Equatable & Sendable` atom of the **Chat Transcript** — one
displayable line (`user`, `assistantText`, `thinking`, `toolCall`, `toolText`,
`system`, `turnHeader`, `streamingText`, `streamingIndicator`). Every string is
render-ready: no JSON, no `Date` formatting, no protocol conversion in the render path.
It is the unit SwiftUI diffs, so its `id` is stable across rebuilds.
_Avoid_: cell, item, list element, view model.

> **Flagged ambiguity — "Transcript" vs ASR transcription.** The **Chat Transcript**
> is the rendered chat conversation (message log → rows). It is unrelated to
> `TranscriptionEngine` / `Transcribing` / `TranscriptionResult`, which are
> speech-to-text. When unqualified, say "chat transcript".

> **Flagged ambiguity — "Turn": transcript turn vs loop turn.** A **Chat Transcript**
> **Turn** spans a user prompt through the assistant's full response and can contain
> several agent-loop turns. The agent loop's `turnEnd` event marks one finer-grained
> turn (a single assistant message plus its tool results). When ambiguous, say
> "transcript turn" vs "loop turn".

**Example dialogue:**

> **Dev:** Streaming updates the rows ~20×/second. Does it re-group the whole
> conversation every tick?
> **Expert:** No. `turns(from:)` is cheap, but `rows(for:_:)` runs `displayProps` per
> tool call. So the coordinator projects every **Turn** only on a full rebuild; for the
> streaming tail it projects just the last **Turn** and splices it onto the stable
> prefix. One fold, two call shapes.
> **Dev:** Where did "expand the streaming header unless the user collapsed it" go?
> **Expert:** Out of the projection. It used to mutate `expandedTurns` from inside a row
> emitter. Now the coordinator decides it and passes `isExpanded` in the `Context`; the
> **Chat Transcript** is pure — same inputs, same rows, no coordinator state touched.
> That purity is what put the grouping rules on a test surface.
> **Dev:** A tool-calling loop — one **Turn** or many?
> **Expert:** One transcript **Turn**, many loop turns. The Turn runs from the user
> message through every assistant message and tool result until the next user message;
> each assistant message inside it was its own `turnEnd`.

## Why

_TODO: the constraints that shape the system (fully offline, on-device MLX,
privacy-first, Apple Silicon) and the trade-offs they force._
