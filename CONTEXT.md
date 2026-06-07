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
**Committed Ref Cleanup** on failure and downgrades to a miss. It is the home for the
lookup-then-hydrate-if-SSD composition for **both** callers — the main prefill path and the
canonical-leaf fallback, which the **Leaf Admission Builder** reaches through an injected
`resolveBoundary` closure-**peer** the actor wires to drive resolution inside `container.perform`
(ADR-0001), exactly as the main path does. `.ssdHit` is an internal intermediate the resolution
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

### SSD snapshot ledger

**Snapshot Ledger**:
The in-memory authority over the SSD prefix-cache tier — which snapshots are
resident, their byte budget, their recency, and the durability of that record —
carved out of `SSDSnapshotStore` (#48). A `nonisolated final class` (`@unchecked Sendable`, one `NSLock`)
owning the `manifest: SnapshotManifest` value (partition registry + descriptor
set), `currentSSDBytes`, the debounced `manifest.json` persist, the corrupt-manifest
directory-walk rebuild, and the **type-protected LRU** eviction policy. It is the
single home for the on-disk schema's whole life: load `manifest.json`,
rebuild-on-corruption, fingerprint/schema/checkpoint-type restore filtering, commit,
recency bump, debounced persist — including descriptor construction
(`SnapshotLedger.makeDescriptor`), moved off `PrefixCacheManager` behind the
`TieredSnapshotStore` enqueue front door, which hands domain inputs and returns a
**Snapshot Ref**.

**The split from the store is along the lock.** `SSDSnapshotStore` keeps the *queue
lock* (`pending`, `pendingBytes`, `drainWaiters`) and the `.safetensors` body I/O
(`writePayload`, `loadSync`, the placeholder-container codec); the Ledger keeps the
*ledger lock* (`manifest`, `currentSSDBytes`, the persist dirty-flag/task, and the
in-flight-delete tombstones). This is safe because the writer is single-threaded and
already releases the lock between every step, so the two locks never nest. The one
method spanning both — `deleteSnapshot` — runs as *sequential* locked steps
(queue-lock to check `pending`; if absent, `ledger.removeOrTombstone(id)`) and is
correct under every writer interleaving because `removeOrTombstone` atomically does
"resident → remove+return, else tombstone." The whole tombstone protocol is the
Ledger's: `commit(descriptor) -> Bool` (vetoed by a prior tombstone),
`consumeTombstone(id) -> Bool` (the writer's pre-write skip),
`removeOrTombstone(id) -> EvictedResident?`.

**The Ledger returns what changed; the store performs the effects.**
`admit(descriptor) -> (decision, evicted: [EvictedResident])`,
`retryAfterDiskFull -> EvictedResident?`, and `remove -> EvictedResident?` return the
residents that left the manifest; the store deletes their files and fires
`onCommit`/`onDrop` + diagnostics *outside* any lock. The type-protected LRU policy
(asymmetric system/non-system pass-1/pass-2) lives **inside** `admit`, run atomically
under the ledger lock — a multi-candidate cut must not tear. The SSD tier's α=0
type-protected LRU is local and does not collide with the RAM tier's **Eviction
Configuration** / **AlphaTuner**.

It is reached from the off-MainActor `loadSync` (the ADR-0001 path) for the
`partitionFingerprint(digest:)` gate and the failure-path `remove`, so it stays
`nonisolated` and lock-based — never an actor. This does **not** reopen ADR-0001: a
lock is not a MainActor hop, the same reason the store satisfied the ADR. The Ledger
owns `manifest.json` + `_meta.json` durability; the store owns `.safetensors` bodies.
`SSDSnapshotStore` composes the Ledger in its `init`, so the external
`SSDSnapshotStore(config:)` interface and the `TieredSnapshotStore` wiring are
unchanged, while Ledger unit tests construct it standalone — corrupt→rebuild,
schema-mismatch→wipe, LRU ordering, byte accounting, and the tombstone protocol no
longer stand up the detached `writerLoop`/`wakeupStream`.
_Avoid_: Snapshot Manifest Store (the working title — "store" overloads
`SnapshotStore`/`SSDSnapshotStore`, "manifest" undersells the live budget/recency
authority), manifest-as-cold-path-only (`loadSync` reads and writes it on the hot
path), descriptor schema as the manager's concern (construction moved onto the
Ledger), eviction *effects* (file delete + `onDrop`) as ledger work (those are the
store's, outside the lock).

> **Flagged ambiguity — Snapshot Ledger vs SSDSnapshotStore.** The **Snapshot
> Ledger** is the in-memory authority over the SSD-resident set plus its
> `manifest.json`/`_meta.json` durability. `SSDSnapshotStore` is the writer queue +
> `.safetensors` body I/O that composes the Ledger and performs the effects of its
> decisions. When unqualified, say "snapshot ledger" vs "SSD store".

**Example dialogue:**

> **Dev:** If the manifest moves into the Ledger, doesn't splitting the one `NSLock`
> race the writer against a delete?
> **Expert:** No, because the writer never holds the lock across steps — every helper
> locks and unlocks on its own, and there's exactly one writer. The only method
> touching both locks is `deleteSnapshot`, and it goes queue-lock-then-ledger
> sequentially. If the item's still queued, it's removed from `pending` and never
> written. If it's in flight, the Ledger tombstones it and `commit` later vetoes
> itself. If it already committed, `removeOrTombstone` finds the resident and returns
> it for deletion. Every ordering resolves.
> **Dev:** Why is the LRU cut inside the Ledger instead of the store driving it?
> **Expert:** Because the cut walks many candidates and frees bytes until the incoming
> fits — that has to be one atomic step under the ledger lock, or a concurrent
> `recordHit` or hydration-failure drop could tear it. The store gets back the list of
> evicted residents and does the file deletes and `onDrop` callbacks outside the lock.

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
The GPU-free routing core for storing one leaf snapshot. It owns the token-path
**reusable-prefix probes** — given a stored conversation it renders the turn in isolation
and again with one synthetic continuation appended (a user turn for the *canonical* mode,
`.userTurn`; a tool result for the *directTool* mode, `.toolResult`) and returns the shared
token prefix the immediate continuation can hydrate, or `nil` when the probe diverges — and
emits the whole decidable routing decision as a two-case **Leaf Capture Plan**:
`.fromBoundary(boundary:storedTokens:)` (restore a boundary snapshot, reprefill the residual
`storedTokens[boundary.tokenOffset...]`, capture a leaf) or `.skip(reason:)`. It takes a
`BoundaryLeafMode` (`.directTool` / `.canonical`) — the two modes that capture from a restored
boundary — and owns boundary acquisition and the offset-guard arithmetic. It acquires the
boundary from the transient boundary snapshot when usable, else falls back through **Snapshot
Resolution** on the canonical path — injected as a closure-**peer**
`resolveBoundary: ([Int]) async -> HybridCacheSnapshot?` so the builder's own code stays
GPU-free: the actor wires the production closure that drives `SnapshotResolution.resolve`
inside `container.perform` (ADR-0001), while a test drives the same routing with a pure closure.
(One production resolver, so a closure — not a one-adapter resolver protocol, which would be a
hypothetical seam.) It is tokenizer-affine (an injected `any Tokenizer`, no model), so the whole
routing tests against a fake tokenizer and a pure resolver with no model files.

The *directLeaf* mode never enters the builder: it snapshots the **live** final KV cache at
`storedTokens.count` and needs no probe, boundary, or tokenizer. The actor maps the selected
`HTTPLeafStoreMode` (from `selectHTTPLeafStoreMode`, still the separately-tested pure function)
down to `BoundaryLeafMode?` — returning `nil` for *directLeaf*, the one place that knows
*directLeaf* is the live-cache path — so `plan`, `leafStages`, and the skip-log mapping stay
*total* over exactly the two boundary cases (no dead *directLeaf* arm kept only for
exhaustiveness) and *directLeaf* pays none of the boundary machinery's tokenizer/probe cost.

`.skip(reason:)` carries a **typed** `LeafSkipReason` whose cases hold their own diagnostic
payload (offsets, lengths). The actor maps each reason to its `logSkip` stage/reason/level/fields
through the pure `leafSkipLog`, pinned byte-for-byte by `LLMActorLeafSkipLogTests` (the same
"pure wire-string, test-pinned" discipline as `ssdDropReasonString`), so a renamed stage or a
flipped level fails a test rather than silently shifting dashboards. It emits `.skip` only for
tokenizer/resolver-decidable failures (tokenization failed, no common prefix, missing transient
boundary, no usable resolved snapshot, the residual offset guard); the live-`finalCache` skips
(`no-final-cache`, `no-reusable-cache-state`, `normalization-trim`, `unsupported-cache-type`,
`invalid-path`, `capturedThenEvicted`) and the `intervention` guard stay actor-side. The actor's
post-generation spine branches *directLeaf* to the live capture and routes the two boundary modes
through one exhaustive `switch` over the plan — `.fromBoundary` executes Metal
(`captureStructuredLeafFromBoundary`, now purely the executor), `.skip` logs the mapped reason and
breaks — so leaf-mode selection, the Metal capture, **Snapshot Admission** construction at the
edge, and `admit` are the only leaf steps left in `LLMActor`. The probe-only predecessor's two
capture helpers (`captureDirectToolLeaf` / `captureCanonicalTemplateLeaf`) dissolved into the
builder.
_Avoid_: leaf store mode as the whole story (mode is one of its inputs), the builder owning
capture or `admit` (model-affine, actor-side), the builder calling `loadSync` directly (it takes
a `resolveBoundary` closure-peer; the Metal-affine hydration stays the actor's to wire), a
capture port (the builder returns a decision; the actor executes Metal — no behaviour-less fake
seam).

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

### Speech word timeline

**Word Timeline**:
The pure projection of one segment's spoken text plus the current playback position into
the highlighted character count and the active word — the single home for the
token→char→word model and the elapsed→position pacing the TTS notch overlay renders. An
immutable `nonisolated` value type, built once per segment from the text plus token offsets:
it holds the per-word character ranges (the `offset += word.count + 1` model, computed
**once** at construction, not re-derived per call) and exposes three entry points — `advance`
(the **single** pacing fold: token character offsets stretched across the smoothed effective
duration, absent offsets degrading to uniform proportional — one path, not the former
`tickTokenTimeline`/`tickProportional` pair), `activeWordIndex` (the char→word query the
view's scroll-centering uses) and `litFraction(wordIndex:)` (the per-word value the view
renders, reading the lengths cached on each `Word` so it stops re-deriving boundaries or
re-walking strings each frame). It owns **no** timer, clock,
`@Observable` state, or UI: elapsed time, total/estimated duration, the `* 0.08` smoothing
carry-over, and the **Segment Window** are passed in and returned, never stored. It is the
**internal seam** inside `TTSWordTracker` — reached by its own unit tests with no timer and no
MainActor — and is driven by the **TTS Word Tracker**, exactly as the **Chat Transcript** is
driven by the **Chat Transcript Controller**. The deletion test passes: remove it and the
`offset += word.count + 1` arithmetic reappears across the tracker's `start`/`updateText`
and the view's word rendering, and the smoothing across two.
_Avoid_: WordPacing (names the operation; we name the value), TTSWordTracker (that is the
**TTS Word Tracker** driver above it), word highlighter, marquee, pacing model.

**TTS Word Tracker**:
The `@Observable @MainActor` stateful driver of the pure **Word Timeline**
(`TTSWordTracker`). It owns the 60fps `Timer`, the injected `playbackTimeProvider` clock seam
(the ADR-0003 virtual clock under test), the monotonic `recognizedCharCount` and the other
published view state the notch overlay reads, and the per-segment carry-over (smoothed
duration, the static learned chars/sec, the **Segment Window**). It delegates the per-tick
pacing fold to **Word Timeline**, keeping only the cross-segment estimate model (the duration
seed, the learned chars/sec, the smoothing carry). Distinct from **Word Timeline** (the pure
fold it calls): the tracker decides *when* to re-fold and publishes the result, and exposes
the word model so the view stops re-deriving word boundaries.
_Avoid_: word timeline (that is the pure core it drives), word highlighter, word state machine.

**Segment Window**:
The single playback-time base a **TTS Word Tracker** measures one long-form segment's pacing
against — replacing the coupled `segmentTimeBase` / `segmentDurationBase` pair that were
always assigned the same value (the previous segment's cumulative scheduled duration). One
value, so "the time base and the duration base disagree" is unrepresentable.
_Avoid_: segmentTimeBase, segmentDurationBase, segment offset, time base.

**Segment Playback**:
The deep module owning the consume-one-TTS-stream-into-playback loop shared by every speech
path — the first long-form segment, each subsequent segment, and single-shot streaming. Given
a generated-sample stream and a small `Segment` value (its optional **segment boundary** plus
the `SpeechState` to assume on the first chunk), it drains the stream into
`AudioPlayback.appendChunk`, drives the **Word Highlight Surface** — pushing total duration
**for the segment the surface is currently displaying**, performing the boundary switch and
drain-wait when a boundary is present — and returns `false` on cancellation so each caller keeps its own cleanup
(`cleanupLongForm` versus the streaming stop/dismiss). The per-segment difference is the
`Segment` *value*, not a bag of flags — the same "the only injected difference is a value"
move as **Overlay Placement** — so it stays deep rather than a config-flag loop. The
duration-update timing and the boundary switch are not separate knobs: both are *derived* from
whether the `Segment` carries a boundary. The deletion test passes: remove it and the
cancel-check / `appendChunk` / duration-update loop reappears in three places and the boundary
switch-and-wait in one.
_Avoid_: chunk loop, stream pump, playback driver, SegmentConsumer, a config-flag loop (the
difference is a `Segment` value, never toggles).

**Word Highlight Surface**:
The `@MainActor` port the **Segment Playback** loop and `SpeechCoordinator`'s session-level
calls drive to render spoken-word highlighting — `show` a fresh segment, `switchText` to the
next segment at a crossed **Segment Window**, push the running `updateTotalDuration`,
`markSegmentComplete` / `markGenerationComplete`, and `dismiss`. The methods are exactly the
surface the real call sites use, nothing more. The production adapter is `TTSNotchPanelController`
(the `NSPanel` plus the **TTS Word Tracker** it hosts); the second adapter is a test-only
`RecordingHighlightSurface` peer that records the call sequence — which is what finally makes
the **segment boundary** switch, and its ordering against the ADR-0003 virtual clock,
*assertable* (every `SpeechCoordinator` test previously passed `notchOverlay: nil`, so the
switch was untested). A **real** seam, not hypothetical: two behaviorally-distinct adapters,
justified exactly as ADR-0003 justifies `InMemoryAudioPlayback` ("peer implementations, not
mocks"). Promoting it also collapses the overlay's old
`updateText(segmentTimeBase:segmentDurationBase:)` — always passed the same value twice — into
one `switchText(…, segmentBase:)` over the **Segment Window**. Not the **Overlay Panel**: the
notch stays its own surface (ADR-0003).
_Avoid_: notch overlay / TTSNotchPanelController (that is *one* adapter, not the seam), overlay
protocol, highlight view, Overlay Panel (a different surface).

> **Flagged parallel — Word Timeline mirrors Chat Transcript.** Both are pure, stateless
> projections driven by an `@Observable @MainActor` controller that decides *when* to re-fold:
> **Word Timeline** / **TTS Word Tracker** for spoken-word highlight pacing, **Chat
> Transcript** / **Chat Transcript Controller** for the chat row list. Different layers and
> inputs; the same pure-fold-plus-driver shape.

> **Flagged ambiguity — Word Timeline vs the dictation Overlay.** The **Word Timeline** paces
> the TTS notch overlay's spoken-word highlight. It is unrelated to the **Overlay Panel** (the
> dictation HUD + full-screen border) and to ASR/dictation. When unqualified, say "word
> timeline" vs "overlay panel".

**Example dialogue:**

> **Dev:** If the word pacing stays inside `TTSWordTracker`, how is it suddenly testable?
> **Expert:** The math moves into **Word Timeline** — a pure `nonisolated` namespace the
> tracker calls. A test feeds it elapsed values and asserts the highlighted char and active
> word, with no `Timer`, no clock, no MainActor. The tracker keeps the timer and the
> `@Observable` output; it just folds the timeline now.
> **Dev:** Three playback loops, one **Segment Playback** — doesn't the first segment differ
> from the later ones?
> **Expert:** Only by its `Segment` value. The first/only segment carries no boundary, so the
> overlay is already showing it and duration updates from the first chunk. A later segment
> carries a boundary, so **Segment Playback** waits for the previous segment's audio to drain,
> switches the overlay, then updates duration — same code, different value. Cancellation
> returns `false` and the caller cleans up its own way.

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
output shape — `AssistantMessage` (agent loop), the server's **CompletionProjection**
plus the shared HTTP replay message (both completion paths), the leaf-store
`HTTPPrefixCacheMessage` (`LLMActor`), or just `text` (the summarizer). The projection
is where a caller's *intent* lives: the summarizer ignores `thinking`/`toolCalls`; the
server's malformed→text fallback fires only when the turn is otherwise empty; the agent
loop assigns stable tool-call identities. Projections stay out of the accumulator so
adding a consumer never edits the shared fold.
_Avoid_: conversion, adapter (it is not a seam adapter), output builder.

**CompletionProjection**:
The server's concrete **Generation Projection** — one `nonisolated Sendable` value that
both HTTP completion paths (streaming SSE, non-streaming JSON) build from a terminal
**Generation Accumulator** plus the turn's completion `info`, effective max-tokens, and
completion id. It is the one home for the rules the two paths used to hand-roll in
parallel: the `finish_reason` rule (previously computed three times), the
malformed-tool-call→text fallback, the empty-payload finish-reason diagnostic, the
thinking-safeguard sidecar. Each path then keeps only its framing (per-event SSE
chunking vs one JSON encode). It stays **pure** — no logging, I/O, or transport — and
returns a `FinishReasonDiagnostic` *value* the caller logs, so the classification (which
had drifted: the non-streaming copy classified *post*-fallback and lacked the malformed
branch and `malformedLen` field) has one unit-tested home while emission stays each
path's side effect. The deletion test passes: remove it and the finish_reason rule, the
fallback, the diagnostic classification, and the safeguard sidecar reappear across both
paths.
_Avoid_: StreamResult (the dissolved per-path capsule — both paths now build one
CompletionProjection), response builder, envelope (that is the per-path framing the
projection feeds, not the projection).

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

**Chat Transcript Controller**:
The `@Observable @MainActor` stateful driver of the pure **Chat Transcript** fold,
`ChatTranscriptController` (carved out of `AgentCoordinator`). It owns the transcript's view-interaction state —
`expandedTurns`, `expandedDetails`, the streaming-throttle clock,
`autoExpandedTurnID`/`streamingManuallyCollapsed`, the `activeTurnRowIndex` splice
point, the timestamp formatter — and publishes the `rows` and `streamingRowVersion`
the chat list renders. It is **publisher-agnostic** (ARCHITECTURE.md's panel-controller
rule): the coordinator's event dispatcher feeds it `messages`, the live `stream`, and
`isGenerating` per call, so it holds no `Agent` reference. It owns the rebuild-vs-patch
decision — a full `rebuild` over every **Turn** versus the throttled tail-patch that
re-projects only the active **Turn** and splices onto the stable prefix — plus the
explicit pre-projection steps (streaming-header auto-expand, stale-expansion pruning)
that used to be hidden side effects in the coordinator. Its interface is deep: feed it a
scripted `(messages, stream, isGenerating)` sequence and assert `rows` — no Agent,
arbiter, or conversation store, the scaffolding the `AgentCoordinator*Tests` files stand
up today. The `isGenerating`-before-rebuild ordering invariant lives in the dispatcher
that drives it, not here. Distinct from **Chat Transcript** (the pure, stateless fold it
calls) and **Generation Projection** (one turn's accumulator → a caller's output): the
Chat Transcript Controller is the stateful orchestrator that decides *when* and *how much* to
re-fold.
_Avoid_: view model, render model, ChatViewModel (same Avoid as **Chat Transcript**), row
store, rebuildRows/patchStreamingTail as the whole story (those are two of its methods).

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

### Agent run lifecycle

**Agent Run**:
The lifecycle of one *foreground* LLM invocation — a `sendMessage` turn or a `/compact`
— serialized behind the `InferenceArbiter`'s exclusive `.llm` lease. The
`@Observable @MainActor` module that owns it, `AgentRunController` (carved out of `AgentCoordinator`), is the
**single writer** of `isGenerating`, and also owns the lease-bearing `sendTask` and the
cancellation contract (`cancel`, `cancelAndWait`). `isGenerating` is set *eagerly* at
`send` — before `agent.prompt` runs — because the run may sit **queued** behind the lease
while `agent.state.phase` is still `.idle`; the flag therefore means "a foreground run is
queued **or** active," a fact only this module knows, which is why it lives here and not
on the coordinator spine. The command side (`send`, `cancel`, `cancelAndWait`, and a
shared `runUnderLease` entry that `/compact` reuses so the lease/flag/cancel contract is
written once) is driven by the view and command execution; the event-side transitions
(`markStarted`/`finish`) are *fed* by the coordinator's event dispatcher on
`.agentStart`/`.agentEnd`. Failures inside the lease task are reported to the
coordinator's shared `error` banner via an injected closure; the only view surface it
owns is `isGenerating`, which the coordinator re-exposes as a computed passthrough so
existing call sites are unchanged. Both `send` and `runUnderLease` preserve the
arbiter-less fallback — with no arbiter wired they drive `agent.prompt`/`agent.forceCompact`
directly (the back-compat branch locked by `compactCommandFallsBackToDirectPathWhenArbiterMissing`).
The deletion test passes: remove it and the
eager-flag-while-queued semantics plus the lease/cancel contract reappear in the
coordinator. Distinct from the **Generation** family (Accumulator / Projection / Stream
Loop), which folds the token stream *inside* a turn — an **Agent Run** is the outer
lease+busy+cancel envelope around the whole invocation.
_Avoid_: generation lifecycle (collides with the Generation* stream family), send
coordinator, run as a transcript/loop turn, busy flag as standalone spine state (the run
owns it).

### Agent state reduction

**Agent State Reducer**:
The single home for folding the `AgentEvent` stream into the `@Observable @MainActor
AgentState` — the run-level view state (`messages`, `streamMessage`, `phase`,
`pendingToolCalls`, `error`) the chat UI and the coordinator's dispatcher read. A pure
fold, `reduce(_ event:, into: state)`, **total** over `AgentEvent`, that mutates the
`@Observable` class **in place** — not a value-type swap, per ADR-0002: a whole-value copy
looks like every property changed and coarsens Observation's per-property invalidation
(the same reason the **Settings Facade** keeps stored properties). It restores pi-mono's
`processEvents` shape — *reduce all state, then notify listeners* — replacing the prior
Swift `Agent.handleEvent`, which split the fold across **two switches with the subscriber
notify wedged between them** (messages/streamMessage before, phase/pendingToolCalls after).
The rules it concentrates: `messageEnd` clears `streamMessage` then appends under a
`hasContent` guard (empty assistant turns from cancel/error paths are dropped); `turnEnd`
does the authoritative full-replace of `messages` from the loop's context snapshot (which
carries tool results `messageEnd` never saw) — pi-mono additionally folds a turn-level
assistant error into `error`, but Swift's `AssistantMessage`/`turnEnd` shape carries no error
field, so that fold has nothing to act on here and is omitted;
`messageUpdate` sets `streamMessage`; `toolExecutionStart`/`End` maintain `pendingToolCalls`
and the `.executingTool`/.streaming phase; `agentStart` → `.streaming`;
`contextTransformStart`/`End` drive the **Swift-only** `.transformingContext`/.streaming
phase. It owns the **event fold only**; the run-lifecycle envelope — the `.idle` transition
and the end-of-run clear of `streamMessage`/`pendingToolCalls` — stays in
`beginRun`/`finishRun`, exactly as pi-mono splits `processEvents` from `finishRun`. Because
the reduce settles before any notify, the coordinator's dispatcher reads fully-current
`AgentState`; it needs no rewrite to read event payloads.

The `AgentPhase` apparatus (`.transformingContext`, `.executingTool`, the `contextTransform*`
events) has **no pi-mono analog** — pi-mono applies the transform inside the loop and tracks
only an `isStreaming` boolean — so it is the genuinely Swift-specific part of the fold and
is where the residual subtlety lived. The retired standalone-`/compact` gate is part of it:
the two-switch split let the coordinator catch a transient `phase == .idle` to fire
`markStandaloneCompactionFinished`; reduce-then-notify removes that transient, so `/compact`
instead **awaits its body under the lease** and clears `isGenerating` on lease completion
like `send` (see **Agent Run**), and both `markStandaloneCompactionFinished` and the
`phase == .idle` read are deleted. The deletion test passes: remove the reducer and the
hasContent guard, the `turnEnd` replace, and the phase/pending transitions reappear inside a
private `handleEvent` with no test surface — today no test constructs an `AgentState` and
feeds it events.
_Avoid_: **Generation Accumulator** (that folds within-turn `Generation` *token* events into
one `AssistantMessage`'s content — a different layer and source; the reducer folds
agent-lifecycle `AgentEvent`s into run-level `AgentState`), event handler / `handleEvent`
(the two-switch predecessor it replaces), dispatcher (that is the coordinator's
`handleAgentEvent`, which routes events to sub-controllers *after* the reduce), state
machine, store.

> **Flagged ambiguity — "fold": accumulator vs reducer.** The **Generation Accumulator**
> folds one turn's token stream into `AssistantMessage` content; the **Agent State Reducer**
> folds the agent's lifecycle events into the run-level `AgentState`. Both are folds;
> different layers, different inputs. When unqualified, say "generation accumulator" vs
> "agent state reducer".

**Example dialogue:**

> **Dev:** If the reducer settles phase before notifying, doesn't the coordinator's
> `phase == .idle` compaction check break?
> **Expert:** It does — and that's the point. The check only ever worked off a transient the
> two-switch split exposed. So we delete it. `/compact` now awaits its body under the lease,
> so `isGenerating` clears when the lease finishes, exactly like a `send` turn. No
> event-derived gate, no race.
> **Dev:** Why not fold a value-type `AgentState` and copy it back — wouldn't that be purer?
> **Expert:** Observation. `AgentState` is `@Observable`; a whole-value copy looks like every
> property changed and coarsens invalidation — the ADR-0002 lesson. The reducer mutates the
> class in place, so only the properties an event actually touches invalidate, and
> `$state.foo` bindings survive.

### Agent coordinator leaves

The publisher-agnostic sub-controllers carved off `AgentCoordinator` that own their state
and deps and touch the event dispatcher *not at all* — the *leaves*, as opposed to the
spine (**Agent Run**, **Chat Transcript Controller**). Each becomes testable on its own; the
coordinator re-exposes the hot view reads (`rows`, `isGenerating`, `streamingRowVersion`)
as computed passthroughs, and the view reaches the rest via nested access
(`coordinator.voiceInput.voiceState`).

**Voice Input**:
The `@Observable @MainActor` module (`AgentVoiceInputController`) owning the agent
composer's push-to-talk capture→transcribe→emit flow. Interface: `voiceState`, `start()`,
`finishCapture()`, `cancel()`, and the `onVoiceTranscription` output callback —
`finishCapture()` stops recording, transcribes, and *emits the text to the composer via
the callback; it does not send* (the pre-carve name `stopVoiceInputAndSend` is a
misnomer). Behind that small interface: the **stale-task guard** (a monotonic
`currentVoiceOperationID` so a transcription completing after a cancel-and-restart sees it
is stale and leaves the newer operation untouched), the minimum-duration check,
post-processing, and the auto-resetting `.error` display. Deps: `AudioCapturing`,
`Transcribing`, `TranscriptionPostProcessor`, `settings.language` — no `Agent`, no arbiter.
Zero event-spine coupling: transcribed text leaves via the callback, which the input bar
feeds into **Agent Run**'s `send`; its errors live in `voiceState`, never the shared
`error` banner. The deletion test passes: remove it and the stale-task guard reappears in
the coordinator — exactly what `AgentCoordinatorVoiceCancellationTests` guards.
_Avoid_: dictation (the separate global `DictationCoordinator`/overlay path), mic
controller, voice state machine.

**System Prompt Inspector**:
The `@Observable @MainActor` module (`AgentSystemPromptInspector`) owning the
system-prompt transparency panel.
Interface: published `assembledSystemPrompt`, `rawChatMLPrompt`, `systemPromptTokenCount`,
and `fetchRawSystemPrompt()`. Behind it: the cancellable async fetch that renders the
assembled prompt through the injected `formatRawPrompt` closure into raw ChatML plus a
token count, superseding any in-flight fetch. Deps: the `formatRawPrompt` closure and a
read-only source of the current prompt + tools (an injected closure, so it tests with no
`Agent`). View-triggered, never event-driven — zero event-spine coupling.
_Avoid_: prompt builder (that assembles the prompt; this inspects it), token counter.

**Command Palette**:
The `@Observable @MainActor` module (`SlashCommandPaletteController`) owning the
slash-command popup *presentation*.
Interface: `commandRegistry`, `showCommandPopup`, `commandSelectedIndex`,
`commandFilteredResults`, plus `updateCommandPopup(for:)`/`dismissCommandPopup()`/
`autocompleteCommand(_:)`. It owns the registry rebuild (discovered skills + extensions)
and the filter/selection/autocomplete interaction over the already-pure
`SlashCommandParser`/`SlashCommandRegistry`. Deps: `extensionHost`, `packageRegistry` — no
spine coupling; lifts clean.
_Avoid_: command executor / command router as a module (execution stays on the spine),
slash command registry (that is the pure `SlashCommandRegistry` the palette drives).

> **Note — deliberately left on the spine.** Two coordinator concerns are *not* carved,
> by the deletion test. **Command execution** (`executeCommand` routes `/compact` into
> **Agent Run**'s lease, `/new`·`/clear` into conversation orchestration, and skills back
> into `AgentRun.send`) is a router that only forwards into three spine concerns; a module
> would be shallow. **Voice output** (`speakMessage`/`stopSpeaking`/auto-speak) is thin,
> stateless calls already sitting over the seamed `SpeechCoordinator`. Both stay as
> coordinator methods; carving them would move complexity, not concentrate it.

> **Flagged ambiguity — "Voice Input" vs dictation.** **Voice Input** is the agent chat
> composer's push-to-talk (mic → transcript → composer text). It is unrelated to
> `DictationCoordinator`, the global system-wide dictation overlay. When unqualified, say
> "agent voice input".

### Operation staleness

**Operation Guard**:
The single home for the monotonic-epoch *stale-result* protocol shared by the capture→
transcribe→commit coordinators — today `DictationCoordinator` and **Voice Input**
(`AgentVoiceInputController`). A `@MainActor final class OperationGuard` holding one
private `epoch` counter behind a three-part interface: `invalidate()` advances the epoch;
`capture() -> OperationTicket` snapshots the current epoch at the start of async work; and
the vended **Operation Ticket** answers `isCurrent`. It exists because the underlying
recognizer may *ignore cancellation and return success anyway*, so a `Task.cancel()` alone
cannot stop a stale transcription from committing — only a post-`await` epoch comparison
can. The load-bearing rule it concentrates: the epoch is advanced at **two** distinct
moments — at `cancel()` (drop the in-flight op) **and** at *operation start*
(`startRecording`/`start`) — while the snapshot is taken at a **third** moment, the
async-work entry (`processAudio`/`finishCapture`), and compared after **every** `await`
resume. The operation-start bump is load-bearing in `DictationCoordinator`, which can begin
a new recording while a prior transcription is still in flight without cancelling it (its
restart paths, plus a second `await` for text injection); in `AgentVoiceInputController` it
is uniformity/defense-in-depth only, since `start()` is `.idle`-gated and so cannot overlap
a live op. A naive `begin()` that bumped-and-snapshotted only at the async entry would drop
the operation-start bump and silently reintroduce stale-result commits in the dictation case. It owns **only** the
epoch protocol: `Task` cancellation, `audioCapture.stopCapture()`, and
`transcriptionEngine.cancelTranscription()` stay caller-side (those are domain-specific and
already served by Swift's `Task`). It needs no `Sendable`: the guard is a stored property of
the `@MainActor` coordinator and every **Operation Ticket** is captured inside a
coordinator-owned non-detached `Task`, so the guard outlives every ticket it vends (the
ticket holds an `unowned` back-reference and reads the live epoch on the MainActor). The
deletion test passes: remove it and the bump/capture/compare protocol reappears across both
coordinators — eight hand-written `guard operationID == currentOperationID` sites today.
_Avoid_: operation ID / `currentOperationID` (that is the bare counter it replaces),
cancellation token (it does not own `Task` cancellation), debounce, sequence number.

**Operation Ticket**:
The value vended by `OperationGuard.capture()` — an `unowned` reference to its owning
**Operation Guard** plus the snapshot epoch. Its sole interface is `isCurrent`
(`owner.epoch == snapshot`), read on the MainActor after each `await` resume to decide
whether the still-running async work may commit. Inert: it carries no behaviour beyond the
comparison and is never persisted past the operation that captured it.
_Avoid_: operation ID, token (unqualified — say "operation ticket"), snapshot (that is the
prefix-cache concept).

> **Flagged ambiguity — "Operation Guard" vs `Task` cancellation.** The **Operation Guard**
> is the epoch protocol that catches a stale *success* (a recognizer that ignored
> cancellation). `Task.cancel()` / `CancellationError` is the complementary mechanism for a
> cancellation-*aware* side effect (text injection suspended mid-flight). Both coordinators
> use both: cancel the task **and** advance the epoch. When unqualified, say "operation
> guard" vs "task cancellation".

> **Flagged ambiguity — "guard".** The **Operation Guard** is the staleness module. It is
> unrelated to Swift's `guard` statement (which is, confusingly, how its `isCurrent` check is
> written at call sites). When ambiguous, say "operation guard".

### Model loading

**Model Identity**:
The value computed **once** from a model directory at load that answers "what model is this,
and what does that imply downstream." A `nonisolated Sendable Equatable` value with a
**total, non-throwing** `init(directory:)` that reads `config.json` and `chat_template.jinja`
exactly once — replacing the loose `LLMActor` statics (`detectToolCallFormat`,
`isQwen35Model`, `isQwen35MoEModel`, `isTriAttentionEligibleModel`, `detectModelFlopProfile`,
`detectPromptStartsThinking`) that each re-read the directory at their own call site (four
`config.json` parses per load today). It carries: `toolCallFormat` (optional — `nil` means
"no override, use the vendor JSON default"), `isQwen35` and `isMoE` (the `model_type`
family/variant facts), `promptStartsThinking` (from the generation-prompt block of the chat
template), and `flopProfile`. `isTriAttentionEligible` is a **computed view** of `isQwen35` —
eligibility is architecture-coupled to Qwen3.5 today, but the property names the *caller's
intent* so it can diverge from the raw family check later.

`flopProfile` is **total**: a non-Qwen3.5 or unparseable config yields `ModelFlopProfile.fallback`,
not `nil`, so the single consumer (`EvictionPolicy`) never handles an absent profile. That
`.fallback` is the **one home** for the "unknown ⇒ Qwen3.5-4B-PARO" default — the identity's
parse path and `LLMActor`'s pre-load cache both resolve to it, so the `?? .qwen35_4B_PARO`
literals that used to sit at call sites are gone. Quant-format routing (`isParoQuantModel`) is **not** identity — it
stays in `ParoQuantLoader`, a container-load concern, not a capability fact. The weight
**fingerprint** (`ModelFingerprint`) is also separate: it throws, and it is
identity-*for-cache-invalidation*, not capability.

Model Identity is computed once at the top of `loadModel` and threaded as a **local** — through
every gate and into `verifyAndStore` as a parameter — so all reads come from that one value
rather than a re-parse. The same value is also installed as load-time actor state through the
existing `installLoadTimeState` single-site (beside the fingerprint, SSD config, and TriAttention
selection), populated even on a failed container load and cleared on unload; that installed
snapshot is the load/unload lifecycle the unit suite pins across the actor boundary (via
`currentModelIdentityForTesting`), not a second source the gates read from. The flop profile is no
longer published into a `@MainActor EvictionPolicy.modelProfile` knob: that mutable static
(with its sibling `alpha`) has been **retired** in favour of an injected **Eviction
Configuration** (see **Eviction tuning** below) — a **separate** deepening from naming the
identity. That retirement does *not* ride on the `LoadTimeState` bundle: `flopProfile` is
load-time and could live there, but `alpha` is runtime-tuned and belongs to the cache/tuner, so
the global's home is the prefix-cache `EvictionConfiguration`, not load-time state.

The public interface is `init(directory:)` — the directory is the seam (production model
dirs and test fixture dirs are its two real inhabitants, so there is no filesystem *port*:
a lone production adapter would be a hypothetical seam). No-disk interpretation tests use an
**internal `init(configJSON:chatTemplate:)`** the directory init delegates to — an *internal*
seam for the test surface, reached via `@testable` and kept off the directory-based interface
(`@testable` cannot see a `private` init, so the seam is `internal`, not `private`). Two alternatives were
designed and **deferred**: a `ModelFamilyDescriptor` registry (open-for-extension, but
speculative generality while only the Qwen3.5 family exists — revisit when a second family
lands), and a `LoadTimeState` bundle subsuming the fingerprint/SSD/TriAttention selection
(a locality pass on the scattered load-time fields, complicated by the two-moment install —
directory facts pre-container-load, sizing post-load; **not** a prerequisite for retiring the
eviction global, which is its own **Eviction tuning** deepening, now implemented).
_Avoid_: ModelProfile (collides with the `ModelFlopProfile` it contains), model config /
the `config.json` dict (one of its two sources), ModelFingerprint (separate, throwing),
model capabilities as a runtime grab-bag (it is the *load-time, directory-derived* facts
only — never lease/timeout/lifecycle state).

> **Flagged ambiguity — "profile".** **Model Identity** is the directory-derived capability
> value; `ModelFlopProfile` is the eviction cost model it carries in its `flopProfile` field.
> When unqualified, say "model identity" vs "flop profile".

**Example dialogue:**

> **Dev:** Why is `flopProfile` non-optional when `detectModelFlopProfile` returned `nil` for
> non-Qwen3.5 models?
> **Expert:** Because the only consumer is `EvictionPolicy`, and it always needs *some*
> architecture to score with — the old call site just wrote `?? .qwen35_4B_PARO` right there.
> Making the field total moves that fallback into the identity's construction, so the default
> lives once and eviction never branches on a missing profile.
> **Dev:** A non-Qwen model reporting a Qwen flop shape isn't wrong?
> **Expert:** It matches today's behaviour, and `alpha` defaults to `0` so the flop term is
> usually skipped. The identity is honest that eviction has one cost model; it does not
> pretend to know a bespoke profile it can't parse.

### Eviction tuning

**Eviction Configuration**:
The `(flopProfile, alpha)` pair the prefix cache scores eviction against — the single mutable
home replacing the two `@MainActor` statics that used to sit on `EvictionPolicy`
(`modelProfile`, `alpha`). A `nonisolated Sendable` value with `let flopProfile: ModelFlopProfile`
and `var alpha: Double`, owned by `PrefixCacheManager` as its **one mutable cell**
(`var evictionConfig`). `flopProfile` is set once when the cache is built, read straight from
**Model Identity**'s `flopProfile` in `ensurePrefixCache` (no separate publish — the redundant
`EvictionPolicy.modelProfile = identity.flopProfile` at load is gone), falling back to the
shared `ModelFlopProfile.fallback` before a model loads (equal to the value the static carried). `alpha` starts at
`0` (LRU within the eligible set) and is adapted at runtime by the **AlphaTuner**. The
configuration dies with the cache on unload — no separate clear.

`EvictionPolicy` stays a pure-function namespace — `computeScores(candidates:now:config:)`,
`selectVictim(…:config:)`, `parentRelativeFlops(…:profile:)` — taking the configuration **by
value**, so every scorer (the manager's two eviction calls, the radix tree's telemetry score,
the tuner's sandbox replays) gets a snapshot it cannot alias. There is exactly one mutable cell;
an instance-shaped policy was rejected because a shared mutable `alpha` is just a cache-scoped
global. `EvictionPolicy` stays MainActor-isolated either way — its scorers read MainActor-isolated
`RadixTreeNode` state (the build defaults actor isolation to MainActor), so the isolation was
never only about the statics — but it sheds all *mutable* state, which is the point.
_Avoid_: `EvictionPolicy.modelProfile` / `EvictionPolicy.alpha` (the retired statics), eviction
settings (it is not a user **Setting**), model profile as a global, the `LoadTimeState` bundle
as its home (the bundle is a separate deepening for the scattered load-time fields; `alpha` is
runtime state and never belonged there).

**AlphaTuner inversion**:
The **AlphaTuner** is constructed with the production `flopProfile` (`AlphaTuner(flopProfile:)`)
so its sandbox-replay caches and its direct `parentRelativeFlops` flops tally score against the
real architecture without reaching a global. Each grid-search replay builds a sandbox
`PrefixCacheManager(evictionConfig:…, alphaTuner: nil)` carrying its **own** candidate `alpha`,
so the search no longer mutates a shared register — the old "leaves `alpha` set to the last
candidate; no `defer` restore" hazard is unrepresentable. The tuned winner returns through
`recordRequest(...) -> Double?` on the call that completes bootstrapping; the manager assigns it
to `evictionConfig.alpha`. The tuner never holds a back-reference to the manager.
_Avoid_: writing `EvictionPolicy.alpha` (the retired global write), a tuner→manager callback or
weak back-reference (the winner rides the return value), continuous retuning (still out of scope).

> **Flagged ambiguity — "profile" vs "config".** The **flop profile** (`ModelFlopProfile`) is the
> immutable per-architecture cost model; the **Eviction Configuration** is the `(flopProfile,
> alpha)` pair the cache holds, whose `alpha` half is runtime-mutable. When unqualified, say
> "flop profile" vs "eviction configuration".

**Example dialogue:**

> **Dev:** If we're retiring the eviction global, why doesn't `alpha` go into the `LoadTimeState`
> bundle with the flop profile?
> **Expert:** Different clocks. `flopProfile` is decided once at load — it could live in the
> bundle. `alpha` is tuned *after* the first eviction by the **AlphaTuner** and keeps adapting;
> it's runtime state, not load-time. So the home for both is the cache's **Eviction
> Configuration**, and the bundle stays a separate concern.
> **Dev:** The grid search used to set the global per candidate. Where does that go?
> **Expert:** Each replay builds its own sandbox cache with its own `evictionConfig`. No shared
> register, so no "left it set to the last candidate" footgun. The winner comes back as the
> `recordRequest` return value and the manager writes its one cell.

### Overlay presentation

**Overlay Panel**:
The deep module owning the lifecycle of a transparent, click-through global
`NSPanel` that floats above all apps (including full-screen) and reacts to
`DictationState`. It owns *everything the dictation HUD and the full-screen border
share line-for-line today*: panel creation (the borderless / `.nonactivatingPanel`
style mask, `.statusBar` level, `canJoinAllSpaces`/`fullScreenAuxiliary`/`ignoresCycle`
collection behaviour, clear-background transparency), the show/hide alpha fade with the
`hideRequestID` cancellation so a stale fade-out can't hide a re-shown panel, the
four-notification screen observation (`didChangeScreenParameters`, `activeSpaceDidChange`,
`didWake`, `screensDidWake`) driving `refreshPanelLayout`, the post-show
visibility re-assertion (`scheduleVisibilityCheck` → `ensureVisibleIfNeeded`: re-`orderFront`
on occlusion, correct alpha drift), and the `DictationState`→visible rule. Generic over its
hosted SwiftUI `Content`. Visibility is driven through one side-effecting entry,
`handleStateChange(_:)`; pure view data (`audioLevel`, `glowTheme`) is set directly on
the injected `OverlayState` the content view reads, so those carry no panel-side behaviour
and need no method. The only injected difference between overlays is an **Overlay Placement**
plus the content view; both former controllers (`OverlayPanelController`,
`FullScreenBorderPanelController`) collapse into two configured instances wired in
`DependencyContainer`. The single screen seam stays `OverlayScreenLocator.preferredScreen()`,
from which it lifts a **Screen Geometry** to feed the **Overlay Placement**.
_Avoid_: overlay controller (the per-overlay class is what dissolved *into* this), overlay
manager, HUD window, generic NSPanel wrapper (it is specifically the dictation-reactive
floating overlay, not any panel — see the TTS notch ambiguity), config-flag panel (the
behaviour is uniform; differences are an **Overlay Placement**, not toggles).

**Overlay Placement**:
The whole injected difference between one **Overlay Panel** and another, as a small
value: a pure `frame(ScreenGeometry, DictationState) -> NSRect` (the dictation HUD: a
state-sized rect centred at the bottom inset of the visible frame; the border: the full
screen frame) and `animatesResizeOnShow` (the HUD animates its resize when a visible state is
applied — *including* transitions between visible states while already on screen, e.g.
`recording → processing → error`; the border, whose frame never changes size, snaps).
`animatesResizeOnShow` governs only the show / visible-state path; screen-change relayout
(the four observed notifications) is instant for both overlays. The per-state pill sizes are
a single non-isolated `PillMetrics` value shared by the placement (which sizes the panel) and
the HUD (which sizes the pill it hosts), so the frame and its content can't drift; being
non-isolated is also what lets the frame closure (and its tests) run off the main actor.
Because the frame computation takes a **Screen Geometry** value
rather than an `NSScreen`, and is a pure closure rather than the former private
`updatePanelFrame`, it has a focused test surface — assert the rect math for a given geometry
and `DictationState` with no panel, no app, no `NSScreen`. Two presets exist: `.pill` and
`.fullScreenBorder`.
_Avoid_: layout strategy, frame provider (it carries the resize-animation bit too, not
only the frame), overlay style (`OverlayStyle` is the user **Setting** that selects *which*
placement is live, not the placement itself), `NSScreen` in the signature (the placement
takes a **Screen Geometry** so it stays unit-testable).

**Screen Geometry**:
The plain-rect screen value an **Overlay Placement** consumes — `{ frame, visibleFrame }`,
both `NSRect`. The **Overlay Panel** lifts it from `OverlayScreenLocator.preferredScreen()`
(`.frame` / `.visibleFrame`) and hands it to the placement, so the frame math depends on
CoreGraphics rects, not on a live `NSScreen` that cannot be constructed under test. This is
the move that turns the buried `updatePanelFrame` into the carve's one real unit-test surface.
_Avoid_: passing `NSScreen` to the placement, screen frame as a bare `NSRect` (the placement
needs both the full frame and the visible frame — the HUD insets the visible frame, the
border fills the full frame).

> **Flagged ambiguity — Overlay Panel vs the TTS notch.** The **Overlay Panel** is the
> shared lifecycle behind the dictation HUD and the full-screen border — both non-interactive,
> faded, screen-observed, `.statusBar`-level, created once. `TTSNotchPanelController` is a
> deliberately separate beast: interactive (`ignoresMouseEvents = false`), `.screenSaver` level,
> no fade, no screen observation, no occlusion re-assertion, created per `show()` with a fresh
> view, and self-resizing via `NotchFrameTracker`. It shares almost no behaviour, so it stays
> its own controller rather than paying an **Overlay Panel**'s interface for behaviour it opts
> out of. When unqualified, "overlay panel" means the dictation-reactive one, not the notch.

**Example dialogue:**

> **Dev:** The dictation HUD and the border are basically the same file — why not one panel
> class for all three overlays, notch included?
> **Expert:** Because the HUD and border *are* the same behaviour — same fade, same screen
> observation, same re-assertion — differing only in frame math and which view they host. That
> difference is an **Overlay Placement**, so they collapse cleanly into one **Overlay Panel**.
> The notch shares none of it: it's interactive, never fades, ignores screen changes, and
> rebuilds per show. Folding it in would mean `fade: false, observe: false, reassert: false,
> interactive: true` — a config-flag soup that re-shallows the module. It earns staying separate.
> **Dev:** If the lifecycle is all `NSPanel`/`NSScreen`, what's actually testable after the carve?
> **Expert:** The placement. The frame math moves out of a private method into a pure
> `frame(geometry, state)` closure over a **Screen Geometry** value, so you assert "recording
> state on this geometry → this rect" with no panel and no `NSScreen`. The fade and
> re-assertion stay AppKit-bound and are exercised through the app —
> the win there is locality and the ~400 deduplicated lines, not a unit test.

## Why

_TODO: the constraints that shape the system (fully offline, on-device MLX,
privacy-first, Apple Silicon) and the trade-offs they force._
