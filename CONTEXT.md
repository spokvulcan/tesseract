# Tesseract

Tesseract Agent is a privacy-first, fully offline AI assistant for macOS —
dictation, text-to-speech, and a tool-calling LLM agent, all running on-device on
Apple Silicon. Those constraints (no cloud, one GPU, sandboxed) shape most of the
language below.

This file is the domain glossary — terms only, no implementation detail. Structure
lives in `ARCHITECTURE.md`; discrete decisions in `docs/adr/`; the detailed
rationale behind each carve lives in git history.

## Language

### Prefix cache snapshot lifecycle

**Snapshot State**:
The per-radix-node value encoding which tier(s) hold a KV-cache snapshot and its
write phase — a six-case enum (`empty`, `ramOnly`, `pendingWrite`, `pendingDropped`,
`committed`, `ssdOnly`) owning both the RAM body and the **Snapshot Ref**; every
transition returns a **State Effect**.
_Avoid_: storage-ref lifecycle, slot, residency.

**Snapshot Ref**:
The immutable on-disk identity of a snapshot (`snapshotID`, `partitionDigest`,
`tokenOffset`, `checkpointType`, `bytesOnDisk`). Knows *what and where on disk*,
never the write phase — that is the **Snapshot State** case carrying it.
_Avoid_: SnapshotStorageRef, storage ref, descriptor.

**Snapshot Admission**:
The write-side operation and value shape for placing captured KV-cache snapshots
into the prefix cache: one synchronous `admit` over a value that pairs snapshots
with payloads and encodes RAM-only vs RAM+SSD per snapshot. Admissions are built
only at the MLX extraction edge, so invalid write shapes are unrepresentable and
the cache manager receives already-valid values; checkpoint-vs-leaf behaviour rides
the value, not the call site.
_Avoid_: capturedPayloads plumbing, payload alignment, storeSnapshots payloads.

**Snapshot Admission Path**:
The validated token path carried by a **Snapshot Admission** — the proof that a
snapshot can be stored at a token path, checked before any cache mutation.
Checkpoint admissions share one full prompt token sequence with per-checkpoint
prefix views.
_Avoid_: promptTokens, storedTokens, offset guard.

**Snapshot Resolution**:
The read-side counterpart to **Snapshot Admission**: resolve a token path to the
best usable snapshot — radix lookup, then SSD hydration via the off-MainActor
`loadSync` (ADR-0001) when the node is `ssdOnly`, promoting on success or applying
**Committed Ref Cleanup** on failure. Both the main prefill path and the
canonical-leaf fallback go through it; callers only ever see a hit or a miss, and
checkpoint planning runs *after* resolution.
_Avoid_: a standalone "hydrator" (resolution owns lookup *and* hydrate), folding
planning into resolution, restore/restoreCache (see the ambiguity below).

> **Flagged ambiguity — "resolve" vs "restore".** **Snapshot Resolution** finds and
> hydrates the snapshot; `restoreCache` is the later model-affine step that loads it
> into a live `[any KVCache]`. Resolution decides *which snapshot*; restore
> *applies* it.

**State Effect**:
The topology-only outcome of a **Snapshot State** transition: `settled`,
`becameEmpty` (the only trigger for the tree's self-heal node removal), or
`ignored(reason)` (propagated only on the two forgiving SSD-writer callback edges).
_Avoid_: transition result, mutation outcome.

**dropRef**:
The forgiving SSD-writer callback edge that drops a *pending* **Snapshot Ref**.
_Avoid_: clear ref, remove ref.

**Committed Ref Cleanup**:
The strict cleanup edge for a *committed* **Snapshot Ref** after a failed SSD
hydration.
_Avoid_: generic ref clear, storage-ref cleanup.

**Explicit Ref Discard**:
The strict cleanup edge used when the SSD backing was already explicitly deleted or
cancelled (e.g. leaf supersession); may discard any ref-bearing state.
_Avoid_: hydration cleanup, generic ref clear.

**canEvictNode**:
The structural invariant query on **Snapshot State**: true iff the node holds no
live **Snapshot Ref**, so removing the node cannot orphan an SSD-resident snapshot.
_Avoid_: canRemove, isOrphanable.

**hasResidentBody**:
The RAM-budget query on **Snapshot State**: true iff the node holds a droppable RAM
body. A node can be node-removable yet still hold a useful RAM body, and vice versa.
_Avoid_: body-removable, resident snapshot.

> **Flagged ambiguity — "State".** `SnapshotState` is the prefix-cache lifecycle
> enum. It is unrelated to `HybridCacheSnapshot.LayerState` (MLX layer tensors) and
> to `@Observable` view state. Say "snapshot state" or "layer state".

**Example dialogue:**

> **Dev:** When the SSD writer's drop callback fires, who removes the node?
> **Expert:** Nobody directly. The tree applies `dropRef` and the *state* decides:
> a remaining RAM body settles to `ramOnly`; nothing left returns `becameEmpty` and
> the tree self-heals.
> **Dev:** So the eviction loop also checks `becameEmpty`?
> **Expert:** No — that loop is proactive: it picks LRU victims by
> `hasResidentBody` and drops bodies. `canEvictNode` is the separate
> structural-removal invariant.

### SSD snapshot ledger

**Snapshot Ledger**:
The in-memory authority over the SSD prefix-cache tier — which snapshots are
resident, the byte budget, recency, and the durability of that record
(`manifest.json` + `_meta.json`, including the corrupt-manifest rebuild and the
type-protected utility cut — terminal-loss **Recovery Cost** scoring under the
shared `alpha`, `.system` chains hard-protected — which runs atomically inside
`admit`). Lock-based and
`nonisolated`, never an actor — it is reached from the off-MainActor `loadSync`
(ADR-0001). It returns *what changed*; `SSDSnapshotStore` (the writer queue plus
`.safetensors` body I/O) performs the effects outside any lock.
_Avoid_: Snapshot Manifest Store, manifest-as-cold-path-only, eviction effects as
ledger work (those are the store's).

**Survival Gate**:
The SSD admission pre-check derived from the cut itself: an incoming chain is
written only if its terminal-loss utility would survive the eviction its own
admission triggers — otherwise the write is skipped (a demotion terminal-drops;
a leaf stays RAM-only with supersession *preserve*). Bites only under budget
contention; an unfilled ledger admits everything. End-of-turn leaf admissions
bypass the gate — the just-finished leaf is the highest-reuse object in the
system and its extension write is suffix-sized.
_Avoid_: judicious admission (the paper mechanism this derives, not copies),
admission policy (vague), write filter.

> **Flagged ambiguity — "gate".** The **Survival Gate** decides whether an SSD
> write happens at all; the **Leaf Extension Admission** worth-it gate decides
> the *shape* of a write that is already happening (suffix vs full). Say which.

> **Flagged ambiguity — Snapshot Ledger vs SSDSnapshotStore.** The Ledger is the
> in-memory authority plus manifest durability; `SSDSnapshotStore` is the writer
> queue and body I/O that composes it and executes its decisions. Say "snapshot
> ledger" vs "SSD store".

**Example dialogue:**

> **Dev:** Why is the LRU cut inside the Ledger instead of the store driving it?
> **Expert:** The cut walks many candidates until the incoming snapshot fits — that
> must be one atomic step under the ledger lock or a concurrent hit or drop could
> tear it. The store gets back the evicted residents and does the file deletes and
> callbacks outside the lock.

### SSD leaf extension

**Snapshot Segment**:
One on-disk file holding a token range of a persisted leaf — either a full
snapshot from offset zero, or the suffix a later leaf added past its base.
Per layer: sliceable attention state stores only the suffix range;
non-sliceable state (recurrent, rotating, chunked) rides whole in every
segment, last-segment-wins on hydration. Every segment has exactly one owner.
_Avoid_: delta file, diff, chunk (collides with prefill chunks), partial
snapshot.

**Segment Chain**:
The ordered **Snapshot Segment**s that together materialize one committed leaf
snapshot — the unit the **Snapshot Ledger** admits, evicts, deletes, and
hydrates. Bytes and budget are chain totals; hydration composes the chain back
into one snapshot, and any broken link condemns the whole chain. One manifest
entry owns the whole chain — there are no cross-entry references.
_Avoid_: parent/child snapshots, snapshot lineage, delta chain.

**Leaf Extension Admission**:
A leaf **Snapshot Admission** whose SSD payload carries only the suffix past
its base — the deepest SSD-backed ancestor leaf it supersedes. When accepted,
supersession *transfers* the base's **Segment Chain** to the new leaf instead
of deleting it, so a turn's SSD write scales with new tokens, not conversation
length (#78). Worth-it-gated: a near-full suffix admits full instead. When the
base disappears mid-flight the leaf degrades to RAM-only and the next turn
self-heals with a full write.
_Avoid_: delta admission (the design-phase working name), incremental write,
suffix write-through.

> **Flagged distinction — three supersession modes.** Superseding an ancestor
> leaf now does one of three things to its SSD backing: **transfer** (a
> **Leaf Extension Admission** takes ownership of the chain — completed at
> the writer's commit, with the base staying fully reachable until then; a
> dropped suffix degrades the transfer to preserve), **delete** (a full SSD
> write replaced it — the pre-extension behavior), or **preserve** (a
> RAM-only leaf admission keeps the ancestor's SSD backing alive as the
> best on-disk approximation for warm start). RAM bodies are dropped in all
> three.

**Chain-Prefix Restore**:
Hydrating only the leading **Snapshot Segment**s of a **Segment Chain**, up to
a historical leaf boundary, recreating a superseded ancestor's snapshot without
its identity. The chain keeps exactly one owner — the restore point is a
tree-side reference resolving through the owning entry, never a second manifest
entry — so every offset where a leaf was once extended stays a restore point
for divergent futures (a **Think-Strip Rewind**, a client-side history edit) at
zero extra bytes on disk.
_Avoid_: partial hydration (suggests arbitrary offsets; restore points are
segment boundaries only), chain split, sub-snapshot, cross-entry reference
(the invariant above still holds).

**Example dialogue:**

> **Dev:** Turn N+1 stores its leaf — what happens to turn N's gigabyte on
> disk?
> **Expert:** Nothing moves. The new leaf admits one suffix segment and takes
> ownership of the old chain; the old leaf's *identity* dies (its node ref is
> discarded when the suffix commits) but its bytes live on as the new leaf's
> prefix.
> **Dev:** And if the suffix write is dropped before it commits?
> **Expert:** Then nothing was transferred. The base keeps its manifest entry
> *and* its node ref — it stays hittable, and the next turn just extends it
> again.

### Image-aware prefix caching

**Image Digest**:
The content identity of one image for prefix-cache keying — a hash over the raw
encoded bytes exactly as received, computed per image. Identity is exact-byte: a
re-encoded or resized variant of the same picture is a *different* image — always
a miss, never a wrong hit.
_Avoid_: perceptual hash, pixel hash, attachment ID (the UI-side UUID is diffing
equality, not content identity), image fingerprint (collides with
`ModelFingerprint`).

**Cache Key Path**:
The sequence the radix tree is keyed on for one request: the prepared prompt
tokens with each image's placeholder run replaced, length-preserving, by
pseudo-tokens deterministically expanded from that image's **Image Digest**, drawn
from a range no vocabulary can occupy. Same length as the KV sequence — key index
== KV offset stays the tree's invariant. The model never sees pseudo-tokens; only
the cache does. Produced by the **Cache Key Space**.
_Avoid_: prompt tokens (the model-facing sequence), token path (unqualified near
images), virtual/hash tokens, key tokens.

**Cache Key Space**:
The per-request authority over the two token spaces — built once after prepare
from the prepared tokens, the per-image **Image Digest**s, and the family's
placeholder identity (a **Model Identity** fact). It owns the image table
(digest, run length, span), produces the **Cache Key Path**, and translates any
render-space token sequence (the planner's last-user re-render, the leaf
probes) into key space — so every token path that touches the radix tree lives
in one space. Construction failure ⇒ **Unkeyed Completion**; a later
translation failure degrades only the consuming feature (a typed skip), never
the request.
_Avoid_: key path splice (the shallow predecessor), space converter, token
mapper, render fixup.

**Conversation Render**:
The single token-only render step the planner re-renders and the leaf probes
share: family message-forming plus chat-template application — exactly the
message shape prepare uses, with no pixel work. One render home means a probe
render cannot drift from prepare's render.
_Avoid_: re-render (unqualified), probe tokenization, per-call-site
applyChatTemplate.

**Position Anchor**:
The M-RoPE continuation state a restored conversation resumes generation at:
the restore offset plus the rope delta accumulated by the cached prefix.
Reconstructed per request by the **Cache Key Space** from its image table
(per cached image: position span minus run length) — zero for an image-free
prefix, never persisted, seeded into the model state ahead of the first
warm forward. Phase 2 (ADR-0007) extends the anchor to also position a *new*
image that lands in the restored remainder: the continuation computes that
image's diverging t/h/w positions *from* the anchor instead of resetting to
zero, so an image-add turn restores warm rather than serving cold.
Vision-container-only; the text container needs no anchor.
_Avoid_: rope delta (the vendor-internal ingredient, not the concept), position
offset (collides with cache/token offsets), mrope state.

> **Flagged ambiguity — "tokens" near the prefix cache.** Radix paths, admission
> paths, and prefill offsets are **Cache Key Path** positions; the model consumes
> the prepared prompt tokens. For text-only requests the two are identical; with
> images they differ in *values*, never in length or offsets.

**Unkeyed Completion**:
A **Server Completion** served with zero cache participation — no lookup, no
admission — because no valid **Cache Key Path** could be built (typed reason:
unrecognized placeholder strategy, run count ≠ image count). Discovered
actor-side after prepare; never a bounce back to the standard path, never an
error. Cache participation is best-effort; serving is not.
_Avoid_: prefix-cache bypass (a route decision, not this), fallback completion,
in-actor `nil` return (the retired pattern this is not), degraded mode
(unqualified).

> **Flagged ambiguity — "cannot serve" vs "cannot key".** The **Completion
> Route** guarantees **Server Completion** never sees a request it cannot
> *serve*. Image guards can make a request impossible to *key* — that yields an
> **Unkeyed Completion**, not a route bounce.

### Vision capability and mode

**Vision-Capable Model**:
A model whose on-disk directory declares image input — its **Model Identity** carries
an image-placeholder identity (the `qwen3_5`-family `vision_config`); text-only
checkpoints do not. A fixed property of the model as downloaded — the same fact the
`/v1/models` snapshot advertises to clients. The PARO family is vision-capable.
_Avoid_: "vision model" (ambiguous with the loaded container), "multimodal" (there is
no audio/video input path), "supports images" as a per-request flag.

**Vision Mode**:
Whether a **Vision-Capable Model** is currently loaded as its image-able VLM container
rather than its text-only container — a load-state of the `.llm` slot, never a property
of the model itself. The two containers wrap the same language-model weights; the VLM
adds the vision tower. Text prefill is equivalent across the two containers (the
historical "slower VLM prefill" is retired), so the only standing cost of vision mode
is the resident vision tower — which is why a vision-capable model can hold it on by
default. The HTTP server forces it on unconditionally (ADR-0008); the chat composer's
former per-turn toggle is retired in favour of a soft global opt-out.
_Avoid_: "vision enabled" as a per-message attribute, conflating it with
**Vision-Capable Model**, "the toggle loads the image" (the toggle selects the
container, not an attachment).

### Prefill orchestration

**Prefill Plan**:
The pre-prefill decision value for one HTTP prefix-cache generation, produced by
the tokenizer-affine **Prefill Planner** and read by the **Server Completion**
module: the restore
decision (cold vs suffix-prefill from the resolved offset), the checkpoint offsets
filtered to the suffix, the transient boundary offsets, and the stable-prefix
offset. It carries offsets — never snapshots or the token array — and the planner
consumes an already-resolved **Snapshot Resolution** result as a value, so it tests
with a fake tokenizer and no model files. Every model-affine step stays in the
actor.
_Avoid_: prefill config, generation params (that is `GenerateParameters`), the
planner owning lookup/hydration, carrying tokens on the plan.

**Leaf Admission Builder**:
The GPU-free routing core for storing one leaf snapshot. It owns the
reusable-prefix probes and boundary acquisition (the transient boundary snapshot,
else **Snapshot Resolution** through an injected `resolveBoundary` closure-peer)
and returns a two-case **Leaf Capture Plan** — `.fromBoundary(...)` or
`.skip(reason:)` with a typed `LeafSkipReason` — for the two boundary modes
(`.directTool`, `.canonical`). The *directLeaf* mode never enters the builder: it
snapshots the live final KV cache. The **Server Completion** module executes the
plan: Metal capture,
**Snapshot Admission** at the edge, `admit`, and the test-pinned skip-log mapping.
_Avoid_: leaf store mode as the whole story (mode is one input), the builder owning
capture or `admit` (model-affine, actor-side), a capture port (the builder returns
a decision; the actor executes Metal).

**Think-Strip Rewind**:
The prefix invalidation a thinking template causes when a new real user message
arrives: the template strips the `<think>` blocks it had kept in every assistant
turn since the previous user query, so the next request's token path diverges at
that span's first assistant turn and everything past it must re-prefill. The
canonical-leaf probe bounds the rewind to that divergence point; it does not
remove it.
_Avoid_: cache miss after tools (the felt symptom, not the mechanism), template
drift (the render is deterministic, not drifting), client mutation.

**Tool Stretch**:
The span of assistant turns and tool results since the last real user message —
the region a thinking template renders with its `<think>` blocks kept, and
exactly the span a **Think-Strip Rewind** re-renders when the next user message
arrives. The longer the stretch, the larger the rewind it is exposed to.
_Avoid_: agentic loop (client-side vocabulary), tool session, turn (a stretch
spans many turns).

**Stretch Abandonment**:
The event that a **Tool Stretch**'s continuation never arrives: the client
aborts the in-flight stream, or no follow-up request lands within a short idle
window after a tool-calls finish. It signals that the next request is likely a
real user message — an incoming **Think-Strip Rewind** — and seeds
**Speculative Canonical Prefill** so the rewind's re-prefill runs before that
message arrives. Only completed messages enter the speculated path; a
half-generated turn never does.
_Avoid_: cancellation invalidating the cache (the felt symptom; nothing is
invalidated), interrupt handling (UI vocabulary), abandoned request (the
stretch is abandoned, not one request).

**Speculative Canonical Prefill**:
The post-turn countermeasure to the **Think-Strip Rewind**: after a final
(non-tool) answer or a **Stretch Abandonment**, re-prefill the full
think-stripped render of the completed span in the background and store that
leaf, so the next user turn restores at full depth instead of the rewind point.
_Avoid_: cache warming (this targets one known future path, not general
pre-population), background generation (it prefills, never decodes).

**Rewind Telemetry**:
The three numbers that make a **Think-Strip Rewind** observable without
reproducing it: the *divergence offset* (how far the request shared the
deepest cached path before forking), the *restore floor* (where the restore
actually landed — a **Chain-Prefix Restore** point at-or-below the divergence),
and their gap, the *rewind size* (the re-prefill the rewind forced). Carried
per request in the **Completion Trace Log** and rolled up on the prompt-cache
dashboard, so a regression shows as a rising rewind count or size.
_Avoid_: cache miss (a rewind is a partial hit at a deeper-than-zero floor),
latency spike (the symptom, not the measured cause).

**Preserve-Thinking Render**:
An opt-in render mode for templates that natively declare it (a template
context flag) keeping `<think>` blocks in every assistant turn, so the render
is append-stable across new user messages and the **Think-Strip Rewind**
cannot occur. The flag is part of the template context and therefore of the
cache partition — toggling it moves the conversation to a fresh partition —
and retained reasoning permanently occupies context. With it on, **Speculative
Canonical Prefill** has nothing to speculate and self-skips.
_Avoid_: think retention hack (vendor-sanctioned where the template declares
it), template patching (vendor templates are never edited), global setting
(per-model).

> **Flagged ambiguity — "plan": prefill plan vs checkpoint plan.** The **Prefill
> Plan** is the whole pre-prefill value; the *checkpoint plan* is one field inside
> it. Say "prefill plan" vs "checkpoint plan".

**Example dialogue:**

> **Dev:** How does the planner stay model-free if tokenization happens inside
> `container.perform`?
> **Expert:** It takes `any Tokenizer`, not the `ModelContext` — it runs the
> `[Int]`-returning probes and boundary arithmetic only; the actor builds the MLX
> tensors.
> **Dev:** Why doesn't the leaf builder get a capture port and return the finished
> admission?
> **Expert:** That fake would exercise no behaviour. The routing — mode, probe,
> boundary source, skip — is what's worth testing, and it needs only a tokenizer
> and a resolver closure; the actor keeps the Metal.

### Server completion

**Server Completion**:
The deep module owning one cache-aware HTTP completion on `LLMActor`'s isolation —
an actor-confined module installed at model load (with the prefix cache, the SSD
config snapshot, and the load-time identity facts) and drained-then-cleared at
unload. It executes what the decision modules produce — **Snapshot Resolution**,
restore, suffix prefill from the **Prefill Plan**, the **Generation Stream Loop**
drive, **Snapshot Admission** at the MLX extraction edge, and the **Leaf Capture
Plan** — and keeps the former cross-step generation bundle as a private value, not
an interface. It registers its active completion so unload cancels-and-awaits
before the container is released; the GPU lease, held across the whole HTTP
request, stays the primary guard.
_Avoid_: CompletionHandler (the HTTP framing edge), CompletionProjection (the
output rules), managed HTTP generation (the deleted MainActor re-pump), server
engine, HTTP generation pipeline.

**Completion Route**:
The dispatcher's pure decision for one server inference request — cache-aware
versus standard-with-reason — computed from request shape alone (empty
conversation, last message from the assistant, no usable prefix-cache
conversation), never from model state. Owned by `ServerInferenceService`, whose
dispatch between the **Server Completion** module and the engine's managed path is
what makes it a real module rather than a pass-through. Image-bearing requests
route cache-aware — key-path failures degrade actor-side to an **Unkeyed
Completion**, never back through the route; video/audio content remains a
no-usable-conversation reason.
_Avoid_: prefix-cache bypass (the retired in-actor `nil` returns), fallback flag,
route checks inside `CompletionHandler` or the actor, image bypass (images are
keyed, not bypassed).

> **Flagged ambiguity — "completion".** `CompletionHandler` is the HTTP framing
> edge; **CompletionProjection** is the terminal output rules; **Server
> Completion** is the model-affine execution module. Say which.

**Example dialogue:**

> **Dev:** A request arrives whose last message is an assistant turn — who bails
> to the standard path?
> **Expert:** The **Completion Route**, in the dispatcher, from request shape
> alone. The **Server Completion** module never sees a request it cannot serve.
> **Dev:** And a model unload mid-stream?
> **Expert:** Can't happen through the arbiter — the GPU lease spans the whole
> request. For non-lease teardown, `LLMActor.unloadModel` drains the module's
> active-completion registry before releasing the container.

### Client integrations

**Integration**:
A supported external client (OpenCode is the first) together with Tesseract's
recipe for configuring it to talk to the server. Each Integration is one adapter;
the set is open-ended.
_Avoid_: connector, plugin, client config (the artifact, not the concept).

**Setup One-liner**:
The single copyable terminal command, served by the live server itself, that
configures an Integration end-to-end — the user-facing unit of setup. It reflects
the server's state at the moment it runs; re-running it is how a setup is
refreshed.
_Avoid_: install command, onboarding script.

**Config Merge**:
The server-side operation that regenerates Tesseract's own block inside a
client's config while preserving everything else in the file untouched.
Tesseract's block is generated output — owned by the merge, replaced wholesale on
every run; a backup of the prior file is the escape hatch.
_Avoid_: config write, config sync, deep merge (explicitly not the policy).

**Example dialogue:**

> **Dev:** The user hand-tuned a model entry inside the tesseract provider block
> and re-ran the **Setup One-liner** — what survives?
> **Expert:** Nothing inside that block: the **Config Merge** owns it and rewrote
> it from current server state. The rest of the file — other providers, MCP
> servers, keybinds — is untouched, and the previous file is in the backup.

### Settings persistence

**Settings Store**:
The seam between what a setting *means* and where its bytes live — a typed
key-value persistence port with default-on-read semantics: the default travels with
every read; there is no `register(defaults:)`. Satisfied by two **Settings Store
Adapters**.
_Avoid_: SettingsManager (the **Settings Facade** above it), UserDefaults (one
adapter), preferences store.

**Settings Store Adapter**:
A concrete **Settings Store**. Exactly two: `UserDefaultsSettingsStore` (the only
production code that calls `UserDefaults`; owns default-on-read via
`object(forKey:) == nil`) and `InMemorySettingsStore` (tests — hermetic,
parallel-safe). Two adapters are what make the seam real.
_Avoid_: backend, provider, mock (the in-memory one is a peer, not a mock).

**Setting**:
The single immutable declaration of one persisted setting — its key, its one
canonical default, and its codec to a stored primitive.
_Avoid_: preference, key, default (a **Setting** *has* those; it is neither).

**Settings Catalogue**:
The table of all **Setting** declarations. Each default has exactly one home, so
default drift between load and reset is unrepresentable.
_Avoid_: defaults dictionary, schema, registry.

**Settings Facade**:
The `@Observable @MainActor` `SettingsManager`: one bindable stored property per
setting (so SwiftUI bindings and per-property Observation survive), each `didSet`
forwarding to the **Settings Store**. Non-persistence side effects (launch-at-login,
dock visibility) live here, above the store. See ADR-0002.
_Avoid_: settings service, settings model.

> **Flagged ambiguity — "store".** The **Settings Store** is the settings
> persistence seam — unrelated to `SnapshotStore`/`SSDSnapshotStore` (the
> prefix-cache tiers). Say "settings store".

**Example dialogue:**

> **Dev:** Where does the SSD-budget default live?
> **Expert:** In its **Setting** in the **Settings Catalogue** — once. Initial load
> and `resetToDefaults` both read it from there.
> **Dev:** Is launch-at-login in the store?
> **Expert:** No — the store only moves bytes. Side effects stay in the facade's
> `didSet`, above it.

### Speech model ports and playback

**Speech Recognizer**:
The `Sendable` ASR model port below the `TranscriptionEngine` facade
(`load`/`transcribe`/`cancel`). The engine keeps everything above it — the timeout
race, lazy load, `.mlmodelc` verification, lifecycle state, `DictationError`
mapping. The port is model-only and never learns about leases or timeouts.
_Avoid_: WhisperKitSpeechRecognizer (one adapter), Transcribing (the engine-facing
port the coordinator depends on), transcriber, ASR backend.

**Speech Synthesizer**:
The `Sendable` TTS model port below the `SpeechEngine` facade, deliberately
faithful to the model surface (`generate`/`generateStreaming`, voice anchoring,
token offsets). One wide port — its sub-surfaces each have only one real adapter,
and one adapter means a hypothetical seam.
_Avoid_: Qwen3SpeechSynthesizer (one adapter), SpeechEngine (the facade above it),
TTS backend.

**Speech Model Adapter**:
A concrete **Speech Recognizer** or **Speech Synthesizer**. Exactly two of each:
the framework-backed actor in the app target (`WhisperKitSpeechRecognizer`,
`Qwen3SpeechSynthesizer` — the only production code touching WhisperKit/MLX for
these features) and the in-memory actor peer in `tesseractTests`
(`InMemorySpeechRecognizer`, `InMemorySpeechSynthesizer`).
_Avoid_: mock, stub, model wrapper, WhisperActor/TTSActor (the pre-seam names).

**Audio Playback**:
The `@MainActor` *sibling* seam (not a model port) below `SpeechCoordinator`,
turning generated samples into sound; the coordinator calls it synchronously inside
the long-form loop. Diagnostics intent is a value passed at `startStreaming`
(`PlaybackDiagnosticsPolicy`), never a mutable toggle. Two adapters:
`AudioPlaybackManager` (AVFoundation) and an in-memory peer whose non-wall-clock
virtual clock (`advance(by:)`) makes the segment-boundary wait loop deterministic.
_Avoid_: AudioPlaybackManager (one adapter), AVAudioEngine (inside it), player.

> **Flagged ambiguity — "Transcribing" vs "Speech Recognizer".** `Transcribing` is
> the engine-facing port `DictationCoordinator` depends on — it swaps the whole
> engine. **Speech Recognizer** is the model-facing port *below* the engine — it
> swaps the model under the **real** engine, putting the engine's own orchestration
> on a test surface. Same split for `SpeechEngine` vs **Speech Synthesizer**. Same
> facade-above / port-below shape as the **Settings Store**. See ADR-0003.

**Example dialogue:**

> **Dev:** Where does the transcription timeout live?
> **Expert:** In the `TranscriptionEngine`. Inject an `InMemorySpeechRecognizer`
> that sleeps and you assert the timeout fires without a gigabyte of WhisperKit on
> disk.
> **Dev:** Does the **Speech Recognizer** know about the GPU lease?
> **Expert:** No. The `InferenceArbiter` drives load/unload on the engine, above
> the port. The port is model-only.

### Speech word timeline

**Word Timeline**:
The pure, immutable projection of one segment's spoken text plus the current
playback position into the highlighted character count and active word — the single
home for the token→char→word model and the pacing fold (`advance`,
`activeWordIndex`, `litFraction(wordIndex:)`). It owns no timer, clock,
`@Observable` state, or UI: elapsed time, durations, smoothing carry-over, and the
**Segment Window** are passed in and returned. Driven by the **TTS Word Tracker** —
the same pure-fold-plus-driver shape as **Chat Transcript** / **Chat Transcript
Controller**.
_Avoid_: WordPacing (names the operation; we name the value), TTSWordTracker (the
driver above it), word highlighter, pacing model.

**TTS Word Tracker**:
The `@Observable @MainActor` stateful driver of the pure **Word Timeline**
(`TTSWordTracker`): the 60fps timer, the injected playback clock seam, the
published view state the notch overlay reads, and the cross-segment estimate model.
It decides *when* to re-fold and publishes the result.
_Avoid_: word timeline (the pure core it drives), word state machine.

**Segment Window**:
The single playback-time base a **TTS Word Tracker** measures one long-form
segment's pacing against — one value, so "the time base and the duration base
disagree" is unrepresentable.
_Avoid_: segmentTimeBase, segmentDurationBase, segment offset, time base.

**Segment Playback**:
The deep module owning the consume-one-TTS-stream-into-playback loop shared by
every speech path. Given a sample stream and a small `Segment` value (optional
boundary plus the `SpeechState` to assume on the first chunk), it drains into
`AudioPlayback`, drives the **Word Highlight Surface** (including the boundary
switch-and-wait), and returns `false` on cancellation so each caller keeps its own
cleanup. The only per-segment difference is the `Segment` value — never flags.
_Avoid_: chunk loop, stream pump, playback driver, a config-flag loop.

**Word Highlight Surface**:
The `@MainActor` port that **Segment Playback** and `SpeechCoordinator` drive to
render spoken-word highlighting (`show`, `switchText`, `updateTotalDuration`,
`markSegmentComplete`, `markGenerationComplete`, `dismiss`). Two adapters:
`TTSNotchPanelController` (production — the `NSPanel` hosting the **TTS Word
Tracker**) and the test peer `RecordingHighlightSurface`, which makes the
segment-boundary switch assertable. See ADR-0004. Not the **Overlay Panel** — the
notch stays its own surface.
_Avoid_: notch overlay / TTSNotchPanelController (one adapter, not the seam),
highlight view, Overlay Panel (a different surface).

> **Flagged ambiguity — Word Timeline vs the dictation Overlay.** The **Word
> Timeline** paces the TTS notch overlay's highlight. It is unrelated to the
> **Overlay Panel** (the dictation HUD + border) and to ASR. Say "word timeline" vs
> "overlay panel".

**Example dialogue:**

> **Dev:** Three playback loops, one **Segment Playback** — doesn't the first
> segment differ from the later ones?
> **Expert:** Only by its `Segment` value. The first carries no boundary, so
> duration updates from the first chunk. A later segment carries a boundary, so the
> loop waits for the previous audio to drain, switches the overlay, then updates —
> same code, different value.

### Generation accumulation

**Generation Accumulator**:
The single home for folding an `AgentGeneration` event stream into one assistant
turn's accumulated content — text, optional thinking, tool calls, the raw
malformed-tool-call buffer, the safeguard safe-prefix length. A `Sendable` value
with one `mutating ingest(_:)` owning the subtle transitions (reclassify appends,
truncate resets to the safe prefix, lazy thinking buffer). It holds no side
effects, control flow, or output type: each caller keeps its own loop and its
**Generation Projection**.
_Avoid_: StreamResult, event handler, GenerationFold (names the operation; we name
the value), ToolCallParser (upstream — it produces the events).

**Generation Projection**:
The per-caller mapping from a **Generation Accumulator**'s state to that caller's
output shape (`AssistantMessage`, **CompletionProjection**, the leaf-store message,
bare text). Caller intent lives in the projection, never in the shared fold.
_Avoid_: conversion, adapter (not a seam adapter), output builder.

**CompletionProjection**:
The server's concrete **Generation Projection** — the one pure home for the rules
both HTTP completion paths (streaming SSE, non-streaming JSON) build from a
terminal accumulator: the finish_reason rule, the malformed-tool-call→text
fallback, the finish-reason diagnostic, the thinking-safeguard sidecar. Each path
keeps only its framing.
_Avoid_: StreamResult (the dissolved per-path capsule), response builder, envelope
(the per-path framing it feeds).

> **Flagged ambiguity — `thinking` nil vs "".** `thinking == nil` means no
> `<think>` block ever opened; `""` means a block opened but is empty so far. Do
> not collapse the optionality.

> **Flagged invariant — reclassify appends.** On `.thinkReclassify` the one rule is
> `text += (thinking ?? "")` — buffered thinking goes *after* any text emitted
> before the block. Append, never prepend.

**Example dialogue:**

> **Dev:** The server streams SSE while the agent loop builds an
> `AssistantMessage`. Do they share the accumulator?
> **Expert:** They share the *fold*, not the loop. Each keeps its own `for await`
> and side effects; both call `ingest(event)`, so the reclassify and truncate rules
> come from one place.

### Generation stream loop

**Generation Stream Loop**:
The single home for consuming one raw model `AsyncStream<Generation>` into the
agent's `AgentGeneration` event stream under the thinking-loop safeguard
(`GenerationStreamLoop`). It owns the loop over the four raw cases, the
`ToolCallParser` lifecycle, the safeguard intervention triple, the continuation
swap (cancel → await → restart from the safe prefix → re-init the parser
out-of-think), and `cancelCurrent` — the one place an external cancel reaches
whichever raw handle is live *across* swaps. Per-caller side effects and
projections stay with the callers (`AgentEngine`, the **Server Completion**
module) via an inline `sink`;
terminal info and diagnostics return on the `Outcome`.
_Avoid_: managed generation (`AgentEngine`'s wrapper), stream consumer / generation
pump, GenerationFold (the fold is the **Generation Accumulator**), ToolCallParser
(upstream).

> **Flagged ambiguity — "loop": stream loop vs agent loop.** The stream loop
> consumes one raw model stream for a single assistant turn; the agent double-loop
> (`AgentLoop`) orchestrates turns and tool calls above it. Say "stream loop" vs
> "agent loop".

**Example dialogue:**

> **Dev:** The thinking-loop safeguard fires mid-stream. Who restarts generation?
> **Expert:** The stream loop: it emits the truncate triple to the sink, cancels
> and awaits the handle, restarts from the safe prefix, swaps in the new stream,
> and re-inits the parser out-of-think.
> **Dev:** And a client cancel right after the swap?
> **Expert:** `cancelCurrent` targets the post-swap handle — cross-swap
> reachability is the invariant it owns.

### Chat transcript projection

**Chat Transcript**:
The pure projection of the agent message log (`agent.state.messages`) into the flat
`[ChatRow]` the chat list renders, grouped into **Turn**s — a stateless namespace
of two pure functions: `turns(from:)` applies the grouping rule, `rows(for:_:)`
folds one **Turn** plus a `Context` of inputs into rows. Expansion state, the live
stream, and formatting are passed *in*; it reads no coordinator state and has no
side effects.
_Avoid_: rebuildRows / patchStreamingTail (the duplicated bodies it replaced), row
builder, ChatRowBuilder, view model, render model.

**Turn**:
The **Chat Transcript**'s grouping unit — a contiguous run from one user message
(or compaction marker) through the assistant's complete response. One **Turn** may
contain several assistant messages when a tool-calling loop runs.
_Avoid_: round, exchange, conversation turn, message group.

**Chat Row**:
The flat, render-ready, `Equatable & Sendable` atom of the **Chat Transcript** —
every string pre-computed, its `id` stable across rebuilds; the unit SwiftUI diffs.
_Avoid_: cell, item, list element, view model.

**Chat Transcript Controller**:
The `@Observable @MainActor` stateful driver of the pure **Chat Transcript** fold
(`ChatTranscriptController`, carved out of `AgentCoordinator`). It owns the
view-interaction state (expansion, streaming throttle, splice point) and the
rebuild-vs-tail-patch decision — a full rebuild over every **Turn** versus
re-projecting only the active **Turn** and splicing onto the stable prefix. The
pre-projection steps (streaming-header auto-expand, stale-expansion pruning) are
explicit controller steps, never projection side effects. Publisher-agnostic: fed
`(messages, stream, isGenerating)` per call.
_Avoid_: view model, render model, ChatViewModel, row store.

> **Flagged ambiguity — "Transcript" vs ASR transcription.** The **Chat
> Transcript** is the rendered chat conversation. It is unrelated to
> `TranscriptionEngine` / `Transcribing` / `TranscriptionResult` (speech-to-text).
> Say "chat transcript".

> **Flagged ambiguity — "Turn": transcript turn vs loop turn.** A transcript
> **Turn** spans user prompt through full response and can contain several
> agent-loop turns (one per `turnEnd`). Say "transcript turn" vs "loop turn".

**Example dialogue:**

> **Dev:** Streaming updates the rows ~20×/second. Full re-group every tick?
> **Expert:** No. Every **Turn** is projected only on a full rebuild; the streaming
> tail re-projects just the last **Turn** and splices onto the stable prefix. One
> fold, two call shapes.

### Agent run lifecycle

**Agent Run**:
The lifecycle of one *foreground* LLM invocation — a `sendMessage` turn or
`/compact` — serialized behind the GPU lease. Its module (`AgentRunController`,
carved out of `AgentCoordinator`) is the single writer of `isGenerating` and owns
the lease-bearing task and the cancellation contract (`cancel`, `cancelAndWait`).
`isGenerating` is set eagerly at `send` because a run may sit queued behind the
lease while the agent is still `.idle` — the flag means "queued **or** active", a
fact only this module knows. It depends on a non-optional `any
InferenceArbitrating`; tests inject the in-memory peer. Distinct from the
Generation family, which folds the token stream *inside* a turn — an **Agent Run**
is the outer lease+busy+cancel envelope.
_Avoid_: generation lifecycle (collides with the Generation* family), send
coordinator, busy flag as standalone spine state.

### Agent state reduction

**Agent State Reducer**:
The single home for folding the `AgentEvent` stream into the `@Observable`
`AgentState` — a pure fold (`reduce(_:into:)`), total over `AgentEvent`, that
mutates the observable class **in place** (a whole-value swap would look like every
property changed and coarsen Observation's per-property invalidation — the
ADR-0002 lesson). Reduce all state, then notify, mirroring pi-mono's
`processEvents`; the run-lifecycle envelope (the `.idle` transition, end-of-run
clears) stays in `beginRun`/`finishRun`.
_Avoid_: **Generation Accumulator** (a different fold — token events within a
turn), event handler / handleEvent (the predecessor it replaced), dispatcher, state
machine, store.

> **Flagged ambiguity — "fold": accumulator vs reducer.** The **Generation
> Accumulator** folds one turn's token stream into message content; the **Agent
> State Reducer** folds lifecycle events into run-level `AgentState`. Say which.

**Example dialogue:**

> **Dev:** Why not fold a value-type `AgentState` and copy it back — purer?
> **Expert:** Observation. A whole-value copy looks like every property changed.
> The reducer mutates in place, so only the properties an event touches invalidate.

### Agent coordinator leaves

The publisher-agnostic sub-controllers carved off `AgentCoordinator` that never
touch the event dispatcher — the *leaves*, versus the spine (**Agent Run**, **Chat
Transcript Controller**). The coordinator re-exposes hot view reads as computed
passthroughs; the view reaches the rest via nested access
(`coordinator.voiceInput.voiceState`).

**Voice Input**:
The `@Observable @MainActor` module (`AgentVoiceInputController`) owning the agent
composer's push-to-talk capture→transcribe→emit flow. `finishCapture()` emits text
to the composer via the `onVoiceTranscription` callback — it does not send.
Staleness is handled by an **Operation Guard**; errors live in `voiceState`, never
the shared banner. No `Agent`, no arbiter.
_Avoid_: dictation (the separate global `DictationCoordinator` path), mic
controller, voice state machine.

**System Prompt Inspector**:
The `@Observable @MainActor` module (`AgentSystemPromptInspector`) owning the
system-prompt transparency panel — the cancellable fetch that renders the assembled
prompt into raw ChatML plus a token count. View-triggered, never event-driven.
_Avoid_: prompt builder (that assembles; this inspects), token counter.

**Command Palette**:
The `@Observable @MainActor` module (`SlashCommandPaletteController`) owning the
slash-command popup *presentation* — registry rebuild, filter/selection/
autocomplete — over the pure `SlashCommandParser`/`SlashCommandRegistry`.
_Avoid_: command executor / router (execution stays on the spine), slash command
registry (the pure type it drives).

> **Note — deliberately left on the spine.** Command execution (a thin router into
> three spine concerns) and voice output (thin stateless calls over the seamed
> `SpeechCoordinator`) stay coordinator methods — carving them would move
> complexity, not concentrate it.

> **Flagged ambiguity — "Voice Input" vs dictation.** **Voice Input** is the agent
> chat composer's push-to-talk. It is unrelated to `DictationCoordinator`, the
> global system-wide dictation overlay. Say "agent voice input".

### Operation staleness

**Operation Guard**:
The single home for the monotonic-epoch *stale-result* protocol shared by the
capture→transcribe→commit coordinators (`DictationCoordinator`, **Voice Input**).
It exists because the recognizer may ignore cancellation and return success anyway
— only a post-`await` epoch comparison can stop a stale commit. The protocol: the
epoch advances at `cancel()` **and** at operation start; an **Operation Ticket**
snapshots it at async-work entry and is compared after every `await` resume. `Task`
cancellation and the domain-specific stops stay caller-side.
_Avoid_: operation ID / currentOperationID (the bare counter it replaced),
cancellation token (it does not own `Task` cancellation), debounce, sequence
number.

**Operation Ticket**:
The value vended by `OperationGuard.capture()` — its sole interface is `isCurrent`,
read on the MainActor after each `await` resume to decide whether still-running
work may commit.
_Avoid_: operation ID, token (unqualified), snapshot (the prefix-cache concept).

> **Flagged ambiguity — Operation Guard vs `Task` cancellation.** The guard catches
> a stale *success* (a recognizer that ignored cancellation); `Task.cancel()`
> handles a cancellation-*aware* side effect. Both coordinators use both.

> **Flagged ambiguity — "guard".** The **Operation Guard** is the staleness module,
> unrelated to Swift's `guard` statement. Say "operation guard".

### GPU lease arbitration

**GPU Lease Queue**:
The pure mutual-exclusion lease for the GPU (`GPULeaseQueue`) — one scoped
operation, `withExclusive { body }`, owning the FIFO waiter queue, the atomic
handoff (no instant where a third caller can barge), and the cancellation protocol
(a waiter cancelled during handoff throws and passes the lease *onward* rather than
orphaning it). Slot-agnostic: it knows nothing of `ModelSlot`, engines, or models.
_Avoid_: arbiter (that composes this), GPU mutex/semaphore, scheduler (no policy
beyond FIFO).

**Inference Arbiter**:
The model-affine facade (`InferenceArbiter`, `@Observable @MainActor`) composing a
**GPU Lease Queue** with model ownership: the lease is held across both the load
and the body, so model identity can never change under a running consumer. It keeps
the `.llm`/`.tts` slot model, load/unload, reload-on-mismatch, the pending-unload
drain, and read-only model state.
_Avoid_: lease queue (the layer below), model manager (collides with
`ModelDownloadManager`), GPU manager.

**Inference Arbitrating**:
The narrow, deliberately single-member `@MainActor` seam (`InferenceArbitrating`:
`withExclusiveGPU(_:llmModelIDOverride:body:)`) the lease-acquiring consumers
depend on, non-optional. Two adapters: the production **Inference Arbiter** and the
in-memory peer `InMemoryInferenceArbiter` in `tesseractTests`. A consumer that
needs `reloadLLMIfNeeded` or model state holds the concrete arbiter instead.
_Avoid_: arbiter protocol / arbitering, lease provider, widening it speculatively
(add a member only when a peer-consuming caller needs it).

> **Flagged ambiguity — "lease".** Unqualified, "lease" means the GPU lease. The
> HTTP server's "inference lease" *is* this lease. It is **not** the prefix-cache
> snapshot-pinning protocol. Near the cache, say "GPU lease" vs "snapshot pin".

> **Flagged ambiguity — queue vs arbiter vs seam.** The *queue* hands off the
> lease; the *arbiter* composes it with model ownership; "the seam" / "the peer"
> belong to the consumer interface. "The arbiter hands off the lease" is wrong.

**Example dialogue:**

> **Dev:** Can a cancelled waiter wedge the queue during handoff?
> **Expert:** No — it throws at the pre-claim check and the queue passes the lease
> onward in that catch, rather than orphaning it.
> **Dev:** Why doesn't the protocol carry `reloadLLMIfNeeded`?
> **Expert:** Its only caller already holds the concrete arbiter. A seam member
> nobody reaches through the seam is interface without a consumer.

### Model loading

**Model Identity**:
The value computed **once** from a model directory at load that answers "what model
is this, and what does that imply downstream": `toolCallFormat` (`nil` means use
the vendor default), the `model_type` family facts (`isQwen35`, `isMoE`),
`promptStartsThinking`, the image-placeholder identity the **Cache Key Space**
consumes (`nil` means the family is not recognized for image keying), and a
**total** `flopProfile` — unknown architectures yield
`ModelFlopProfile.fallback`, the one home for that default. Total, non-throwing
`init(directory:)`; computed at the top of `loadModel` and threaded as a local.
Quant-format routing stays in `ParoQuantLoader` (a container-load concern), and the
throwing `ModelFingerprint` is identity-*for-cache-invalidation*, not capability.
_Avoid_: ModelProfile (collides with `ModelFlopProfile`), model config /
`config.json` dict (one of its sources), ModelFingerprint (separate), runtime
capability grab-bag (it is the load-time, directory-derived facts only).

> **Flagged ambiguity — "profile".** **Model Identity** is the directory-derived
> capability value; `ModelFlopProfile` is the eviction cost model carried in its
> `flopProfile` field. Say "model identity" vs "flop profile".

**Example dialogue:**

> **Dev:** Why is `flopProfile` non-optional when detection can fail?
> **Expert:** Its only consumer, eviction scoring, always needs *some* architecture
> to score with. The fallback lives once, in the identity's construction, instead
> of `??` defaults at call sites.

### Cache memory budget

**Pressure-Reactive Budget**:
The RAM-tier byte budget as a band, not a constant: a load-time auto-sized
ceiling, a current value pushed down by OS memory-pressure events and regrown
with hysteresis when pressure clears, never below the **Budget Floor**. The
cache is greedy when RAM is idle, polite when it is contested.
_Avoid_: static budget, memoryBudgetBytes-as-constant, cache size limit.

**Budget Floor**:
The lower bound of the **Pressure-Reactive Budget**: enough RAM to keep the
`.system` chains and the *single* most-recently-extended leaf resident — the
snapshots that buy the next turn's near-instant TTFT. Content-defined, not a
fixed byte count, and deliberately minimal and dumb: a last-resort survival
set at critical pressure, never the protection mechanism (protecting the tall
main-agent leaf against subagent churn is the eviction score's job).
_Avoid_: minimum cache size, reserved bytes, fixed floor, per-partition floor,
workload heuristics in the floor.

**Snapshot Demotion**:
Moving a snapshot's body out of RAM while keeping it recoverable: persist to
SSD if not already backed, then drop the RAM body — trading a future
re-prefill for a far cheaper hydration. The first response to *any* RAM-tier
shrink — pressure events and ordinary evict-to-fit alike; outright dropping
is the fallback when SSD backing is unavailable.
_Avoid_: spill, flush, evict-to-SSD, eviction (terminal — a demotion is
recoverable).

> **Flagged invariant — demotion never refreshes recency.** A demotion write
> must not touch the ledger's `lastAccessAt`: demoted bodies are the *least*
> valuable, and refreshing them would make every pressure event invert the
> SSD tier's recency signal. Only hydrations and extensions refresh.

> **Flagged ambiguity — "budget".** The **Pressure-Reactive Budget** is the
> RAM-tier band; the **Snapshot Ledger** keeps its own SSD byte budget (a user
> setting, static). Unqualified near the cache, say which tier.

> **Flagged distinction — demotion vs eviction vs preserve.** **Snapshot
> Demotion** removes a RAM body but keeps the snapshot hittable via SSD;
> *eviction* is terminal loss (RAM drop without backing, or the SSD-tier cut);
> supersession *preserve* keeps an ancestor's SSD backing after a RAM-only
> leaf admission. All three drop RAM bodies; they differ in what survives.

### Eviction tuning

**Recovery Cost**:
What the next hit pays if a snapshot leaves a tier — the tier-aware F in
eviction scoring. Hydration cost for an SSD-backed RAM body (a **Snapshot
Demotion** is recoverable); re-prefill FLOPs where loss is terminal — an
unbacked RAM drop or the SSD-tier cut. Denominated in seconds via rolling
measured device estimates (prefill FLOPs/s, hydration bytes/s) — never guessed
constants — so hydration and re-prefill compare in one unit. Replaces the
single-tier Marconi reading of F as the FLOPs a snapshot embodies.
_Avoid_: FLOP savings (the embodied-FLOPs reading), flops-per-byte
(unqualified — density flattens for backed bodies), parentRelativeFlops (one
ingredient, not the concept).

> **Flagged consequence — where α earns its keep.** Among SSD-backed RAM
> bodies, recovery cost per byte is a constant, so the density term goes flat
> and demotion ordering degenerates to recency (LRU — correctly). The α-blend
> only changes outcomes where loss is terminal: the SSD-tier cut and unbacked
> RAM eviction.

**Eviction Configuration**:
The `(flopProfile, alpha)` pair the prefix cache scores eviction against — the
single mutable cell, owned by `PrefixCacheManager`. `flopProfile` is set once from
**Model Identity** when the cache is built; `alpha` starts from the persisted
per-model-fingerprint value (an offline-trace-seeded default on first run) and
keeps adapting at runtime via the **AlphaTuner**. `EvictionPolicy`
stays a pure-function namespace taking the configuration **by value**, so every
scorer gets a snapshot it cannot alias.
_Avoid_: `EvictionPolicy.modelProfile` / `.alpha` (retired statics), eviction
settings (not a user **Setting**), model profile as a global, alpha-starts-at-zero
(the retired cold-start).

**AlphaTuner inversion**:
The **AlphaTuner** is constructed with the production `flopProfile`; each
grid-search replay builds a sandbox carrying its own candidate `alpha` (no
shared mutable register), and the tuned winner returns through
`recordRequest(...) -> Double?` for the manager to assign — damped, never a
jump. The tuner never holds a back-reference to the manager. Tuning is
continuous (a sliding window retuned on terminal-eviction pressure, not a
one-shot bootstrap) and the result is persisted per model fingerprint, so
sessions inherit instead of relearning — on a single-user Mac, persistence
does the work that traffic volume does in the cloud.
_Avoid_: writing a global alpha, tuner→manager callbacks or weak back-references,
one-shot bootstrap / first-eviction phase machine (the retired lifecycle).

> **Flagged ambiguity — "profile" vs "config".** The **flop profile** is the
> immutable per-architecture cost model; the **Eviction Configuration** is the
> pair whose `alpha` half is runtime-mutable.

**Example dialogue:**

> **Dev:** Why doesn't `alpha` live with the flop profile in load-time state?
> **Expert:** Different clocks. The profile is decided once at load; `alpha` keeps
> adapting after evictions. So both live in the cache's configuration, not
> load-time state.

### Overlay presentation

**Overlay Panel**:
The deep module owning the lifecycle of a transparent, click-through global
`NSPanel` that floats above all apps and reacts to `DictationState` — everything
the dictation HUD and the full-screen border share: panel creation, the show/hide
fade with stale-fade cancellation, the four-notification screen observation, the
post-show visibility re-assertion. Visibility is driven through one entry,
`handleStateChange(_:)`; pure view data (`audioLevel`, `glowTheme`) is set directly
on the injected `OverlayState`. The only difference between overlays is an
**Overlay Placement** plus the hosted content view — the two former per-overlay
controllers are now two configured instances wired in `DependencyContainer`.
_Avoid_: overlay controller / manager, HUD window, generic NSPanel wrapper,
config-flag panel.

**Overlay Placement**:
The whole injected difference between one **Overlay Panel** and another, as a small
value: a pure `frame(ScreenGeometry, DictationState) -> NSRect` plus
`animatesResizeOnShow`. Per-state pill sizes live in the shared `PillMetrics` value
so the frame and its content can't drift. Two presets: `.pill`,
`.fullScreenBorder`.
_Avoid_: layout strategy, frame provider, overlay style (`OverlayStyle` is the user
**Setting** that selects *which* placement is live), `NSScreen` in the signature.

**Screen Geometry**:
The plain-rect screen value an **Overlay Placement** consumes — `{ frame,
visibleFrame }` — lifted from `OverlayScreenLocator.preferredScreen()`, so the
frame math is unit-testable without a live `NSScreen`.
_Avoid_: passing `NSScreen` to the placement, a bare `NSRect` (placements need both
frames).

> **Flagged ambiguity — Overlay Panel vs the TTS notch.** The **Overlay Panel** is
> the shared dictation HUD/border lifecycle. `TTSNotchPanelController` is
> deliberately separate — interactive, no fade, no screen observation, rebuilt per
> show — and folding it in would re-shallow the module into config flags.
> Unqualified, "overlay panel" means the dictation-reactive one.

**Example dialogue:**

> **Dev:** Why not one panel class for all three overlays, notch included?
> **Expert:** The HUD and border are the same behaviour differing only in placement
> and content; the notch shares none of it. Folding it in means `fade: false,
> observe: false, interactive: true` — config-flag soup.
> **Dev:** What's actually testable after the carve?
> **Expert:** The placement: pure rect math over a **Screen Geometry** — no panel,
> no app, no `NSScreen`.

### App composition

**App Bindings**:
The deep module owning the app's launch sequence and every long-lived runtime
subscription *with a rule* — carved out of `DependencyContainer.setup()`: the
Whisper auto-load gate, the lazy LLM reload guard (the initial settings emission
never forces a model load), the server enable/port reaction, the overlay style
switch, the glow-theme seed-before-panel-setup ordering, hotkey re-binding, and
the single dictation-state subscription fanning out to both **Overlay Panel**
instances and the menu bar — one subscription path, so the overlays and the menu
bar always see the same emission. The **Settings Facade** comes in concrete;
effects leave through a closure-struct the container wires — the launch mirror of
`AppTerminationCoordinator`'s teardown steps. Subscriptions install before the
initial Whisper load, which runs as an owned child task, so the HTTP server never
waits on a model load. The container itself stays pure wiring: lazy properties,
callback forwarding, codec and route registration.
_Avoid_: app glue (the pre-carve working name), setup() behaviour, launch
coordinator, app services, SwiftUI binding (unrelated).

> **Flagged ambiguity — "binding".** **App Bindings** is the launch +
> subscription-rules module; a SwiftUI `Binding` is view data flow. Say "app
> bindings" vs "a SwiftUI binding".

**Example dialogue:**

> **Dev:** Selecting an agent model in Settings right after launch — does that
> force a model load?
> **Expert:** No. The reload guard is an **App Bindings** rule: the emission is
> dropped unless an `.llm` slot is already loaded. Lazy loading is pinned by a
> test, not a comment.
> **Dev:** And the border flashing the default glow theme on first frame?
> **Expert:** The launch ordering — seed before panel setup — is recorded effect
> order in the same tests.
