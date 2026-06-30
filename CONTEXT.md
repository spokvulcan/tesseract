# Tesseract

Tesseract is a privacy-first, fully offline AI assistant for macOS — dictation,
text-to-speech, and a tool-calling LLM agent, all running on-device on Apple
Silicon. Those constraints (no cloud, one GPU, sandboxed) shape most of the
language below.

This file is the domain glossary — terms only, no implementation detail.
Structure lives in `ARCHITECTURE.md`; decisions in `docs/adr/`; rationale in git
history.

## Language

### Prefix cache snapshot lifecycle

**Snapshot State**:
The per-radix-node lifecycle value: a six-case enum (`empty`, `ramOnly`,
`pendingWrite`, `pendingDropped`, `committed`, `ssdOnly`) encoding which tier(s)
hold a node's KV-cache snapshot and its write phase, owning both the RAM body and
the **Snapshot Ref**. Distinct from the on-disk **Snapshot Ref**, which carries no
phase.
_Avoid_: storage-ref lifecycle, slot, residency; state (unrelated to MLX layer
state and `@Observable` view state).

**Snapshot Ref**:
The immutable on-disk identity of a snapshot — what it is and where on disk it
lives, never the write phase (that phase is the **Snapshot State** case carrying
it).
_Avoid_: SnapshotStorageRef, storage ref, descriptor.

**Snapshot Admission**:
The write side of the prefix cache: an already-validated value pairing captured
snapshots with payloads and per-snapshot RAM-only vs RAM+SSD storage intent.
Built only at the MLX extraction edge so invalid write shapes are
unrepresentable; the read-side counterpart is **Snapshot Resolution**.
_Avoid_: capturedPayloads plumbing, payload alignment, storeSnapshots payloads.

**Snapshot Admission Path**:
The validated token path carried by a **Snapshot Admission** — the proof that a
snapshot may be stored at a given token offset, checked before any cache mutation.
_Avoid_: promptTokens, storedTokens (both name unrelated token fields), offset
guard.

**Snapshot Resolution**:
The read side of the prefix cache: resolve a token path to the best usable
snapshot, returning a hit or a miss. The read-side counterpart to **Snapshot
Admission**; distinct from `restoreCache`, the later model-affine step that applies
a resolved snapshot into a live cache (resolution picks *which*, restore
*applies* it). Owned by the **Prefix Cache Manager** as its single read-side
entry, never a free-standing module reaching back across the manager's mutation
seam.
_Avoid_: a standalone "hydrator" module (the retired shape that reached back into
the manager — resolution is manager-owned); **Snapshot Hydrating** (a different
concept: the off-main handle, not the choose step); restore/restoreCache (the
apply step, not the choose step).

**Snapshot Hydrating**:
The narrow off-main handle that **Snapshot Resolution** depends on to materialize
a body-absent snapshot from SSD, satisfied by the concrete `SSDSnapshotStore` and
an in-memory test peer — the second adapter that made the seam real. Carries only
`loadSync`, `loadSyncPrefix`, and `recordHit`; a consumer needing broader SSD
access reaches for the concrete store instead. The ADR-0001 off-MainActor
hydration discipline lives at this seam.
_Avoid_: widening it before a second caller needs a member; the concrete store's
full surface; mock/stub (the in-memory peer is a real adapter).

**State Effect**:
The topology-only outcome of a **Snapshot State** transition: `settled`,
`becameEmpty` (the sole trigger for the tree's self-heal node removal), or
`ignored(reason)`. Carries no telemetry payload.
_Avoid_: transition result, mutation outcome.

**dropRef**:
The forgiving SSD-writer callback edge that drops a *pending* **Snapshot Ref**.
_Avoid_: clear ref, remove ref.

**Committed Ref Cleanup**:
The strict cleanup edge for a *committed* **Snapshot Ref** after a failed SSD
hydration. Distinct from **dropRef** (pending refs) and **Explicit Ref Discard**
(already-deleted backing).
_Avoid_: generic ref clear, storage-ref cleanup.

**Explicit Ref Discard**:
The strict cleanup edge used when the SSD backing was already explicitly deleted or
cancelled (e.g. leaf supersession); unlike **Committed Ref Cleanup**, may discard
any ref-bearing state, not just a committed one.
_Avoid_: hydration cleanup, generic ref clear.

**canEvictNode**:
The structural invariant query on **Snapshot State**: true iff the node holds no
live **Snapshot Ref**, so removing it cannot orphan an SSD-resident snapshot.
Distinct from **hasResidentBody** — a node can be removable yet hold a useful RAM
body, and vice versa.
_Avoid_: canRemove, isOrphanable.

**hasResidentBody**:
The RAM-budget query on **Snapshot State**: true iff the node holds a droppable RAM
body. Distinct from **canEvictNode** (the SSD-ref orphan invariant).
_Avoid_: body-removable, resident snapshot.

### SSD snapshot ledger

**Snapshot Ledger**:
The in-memory authority over the SSD prefix-cache tier — which snapshots are
resident, the byte budget, recency, and the durability of that record — and the
type-protected, terminal-loss-scored eviction cut that decides what stays. It
only decides and records *what changed*; the separate SSD store performs the
file effects.
_Avoid_: Snapshot Manifest Store; SSD store (the writer/body-I/O that composes the
ledger — say "snapshot ledger" vs "SSD store"); eviction effects as ledger work
(those are the store's).

**Survival Gate**:
The SSD admission pre-check that admits an incoming chain only if it would survive
the eviction its own write triggers, skipping the write otherwise. It decides
*whether* a write happens at all (unlike the **Leaf Extension Admission** worth-it
gate, which decides the *shape* of a write already happening); it bites only under
budget contention, and end-of-turn leaf writes bypass it.
_Avoid_: judicious admission (the Marconi-paper mechanism this derives, not
copies); admission policy (vague); write filter.

### SSD leaf extension

**Snapshot Segment**:
One on-disk file holding a token range of a persisted leaf — either a full
snapshot from offset zero or the suffix a later leaf added past its base — with
exactly one owner. Sliceable attention state stores only the suffix range;
non-sliceable state rides whole in every segment.
_Avoid_: delta file; diff; chunk (collides with prefill chunks); partial snapshot.

**Segment Chain**:
The ordered **Snapshot Segment**s that together materialize one committed leaf
snapshot — the single unit the **Snapshot Ledger** admits, evicts, deletes, and
hydrates. Bytes and budget are chain totals; one manifest entry owns the whole
chain, with no cross-entry references, and any broken link condemns it all.
_Avoid_: parent/child snapshots; snapshot lineage; delta chain.

**Leaf Extension Admission**:
A leaf **Snapshot Admission** whose SSD payload carries only the suffix past its
base — the deepest SSD-backed ancestor leaf it supersedes — so a turn's write
scales with new tokens, not conversation length. On acceptance the base's
**Segment Chain** *transfers* to the new leaf rather than being deleted; a
near-full suffix admits full instead, and a lost base degrades the leaf to
RAM-only until the next turn self-heals with a full write.
_Avoid_: delta admission (the design-phase working name); incremental write;
suffix write-through.

**Chain-Prefix Restore**:
Hydrating only the leading **Snapshot Segment**s of a **Segment Chain**, up to a
historical leaf boundary, to recreate a superseded ancestor's snapshot without its
identity. The restore point is a tree-side reference resolving through the chain's
single owning entry — never a second manifest entry — so every former extension
boundary stays a zero-extra-bytes restore point for divergent futures.
_Avoid_: partial hydration (suggests arbitrary offsets; restore points are segment
boundaries only); chain split; sub-snapshot; cross-entry reference (the one-owner
invariant still holds).

### Image-aware prefix caching

**Image Digest**:
The exact-byte content identity of one image for prefix-cache keying — a hash over
its raw encoded bytes as received. A re-encoded or resized variant of the same
picture is a *different* digest: always a miss, never a wrong hit.
_Avoid_: perceptual/pixel hash (digest is exact-byte, not visual), attachment ID
(the UI's diffing UUID, not content identity), image fingerprint (collides with
`ModelFingerprint`).

**Cache Key Path**:
The token sequence the radix tree is keyed on for one request — the prepared prompt
tokens with each image's placeholder run swapped, length-for-length, for
pseudo-tokens derived from that image's **Image Digest** (drawn from a range no
vocabulary occupies). Same length and offsets as the model-facing prompt tokens but
different values at image runs; the model never sees it.
_Avoid_: prompt tokens (the model-facing sequence; identical only for text-only
requests), token path (ambiguous near images), virtual/hash/key tokens.

**Cache Key Space**:
The per-request authority that holds the request's image table and reconciles the
two token spaces — it produces the **Cache Key Path** and translates any
render-space token sequence into key space, so everything that touches the radix
tree shares one space. Failing to build it yields an **Unkeyed Completion**; a
later translation failure degrades only the consuming feature, not the request.
_Avoid_: key path splice (a shallower predecessor), space converter, token mapper,
render fixup.

**Conversation Render**:
The token-only rendering contract — family message-forming plus chat-template
application, no pixel work — shared by the planner's re-render and the leaf probes,
identical to the shape prepare uses, so a probe render cannot drift from prepare's.
_Avoid_: re-render (unqualified), probe tokenization, per-call-site
`applyChatTemplate` (the rendering it standardizes, not a synonym for it).

**Render–Cache Offset Contract**:
The length-agreement a write-side cache manipulation assumes between a canonical
**Conversation Render** of a completed turn and the live decode cache of that same
turn: the render may exceed the cache only by template scaffolding around the stop
condition (an end-of-message marker, then a trailing separator), because the cache
stops at the decoded stop token while the render re-tokenizes the stored text
through the template. The canonical leaf store relies on it; **Asymmetric-State
Restore**'s bearing capture reuses the same contract to locate its excision
offsets. A drift beyond scaffolding (vision expansion, prior-think re-render, a BPE
seam) means the re-tokenized path no longer matches the cached path, and the
operation declines rather than mis-align.
_Avoid_: token-count match (implementation-level), offset guard (one check that
enforces it, not the concept); re-tokenization contract (the code comment's working
name — say the contract).

**Position Anchor**:
The M-RoPE continuation state a warm-restored conversation resumes generation at —
the restore offset plus the rope delta the cached prefix accumulated —
reconstructed per request from the image table, never persisted. Vision-container
only; a text prefix has no anchor (delta zero), and an image landing in the
restored remainder is positioned *from* the anchor rather than recomputed cold.
_Avoid_: rope delta (the vendor-internal ingredient, not the concept), position
offset (collides with cache/token offsets), mrope state.

**Unkeyed Completion**:
A **Server Completion** served with zero cache participation — no lookup, no
admission — because no valid **Cache Key Path** could be built (typed reasons:
unrecognized placeholder family, placeholder-run count ≠ image count). It is correct
serving, discovered after prepare, never a route bounce and never an error.
_Avoid_: prefix-cache bypass (a route decision, not this), fallback completion,
in-actor `nil` return (a retired pattern), degraded mode (unqualified).

### Vision capability and mode

**Vision-Capable Model**:
A model whose on-disk config declares image input (the Qwen3.5-family
`vision_config`); text-only checkpoints do not. A fixed property of the model as
downloaded — distinct from **Vision Mode**, which is whether that capability is
currently loaded. The PARO family is vision-capable.
_Avoid_: "vision model" (ambiguous with the loaded container), "multimodal" (there
is no audio/video input path), "supports images" as a per-request flag.

**Vision Mode**:
Whether a **Vision-Capable Model** is currently loaded as its image-able VLM
container rather than its text-only one — a load-state of the loaded model, never a
property of the model itself. The HTTP server always loads the VLM container; the
chat loads it unless the user opts out globally.
_Avoid_: "vision enabled" as a per-message attribute, conflating it with
**Vision-Capable Model**, per-turn vision toggle (retired).

**Image Input Availability**:
The chat-composer verdict on whether image affordances appear: the selected model
is a **Vision-Capable Model** *and* the global "use vision when available" setting
is on. A UI/input decision, not a load request and not a property of an attachment.
_Avoid_: per-turn vision toggle (retired), disabled-but-accepted images, attachment
capability.

**Vision Token Budget**:
The per-image ceiling on vision tokens (hence image patches) a processed image may
contribute, bounding the vision tower's quadratic global attention; a default the
app sets, raisable per request. It governs only the processed grid the tower sees,
so the same picture keeps the same **Image Digest** while its placeholder run
shrinks. Distinct from the request-wide *patch guard*, which prices a turn's
combined patches and rejects an over-budget many-image turn before the tower runs.
_Avoid_: "max pixels" (the processor knob it rides, not the concept), "image
downscaling" (the mechanism), "resolution cap" (names the input, not the
vision-token unit), conflating the per-image budget with the request-wide patch
guard.

### Model catalog

**Model Catalog**:
The read-model answering which models exist and which are usable — the join of the
static model-definition table with live download state, answering
downloaded-in-a-category, is-this-downloaded, and the **Vision-Capable Model**
check. The one home for the category-filter × downloaded join callers used to
re-derive inline.
_Avoid_: "model registry" (registry collides; this is a read-model), the raw
statuses dict (that is download state, not the catalog), download facade, model
list.

**Catalogue vs download state**:
What models *exist* is the static definition table — category and identity, no
runtime input; what is *on disk* is per-id download status. The **Model Catalog**
is their join; raw download status (downloading, verifying, error, progress) stays
directly readable for download UI and is not a catalog question.

**Vision Capability Memo**:
The per-model-id cache of the **Vision-Capable Model** disk probe, held once and
shared by every caller; a known answer is cached permanently while an undownloaded
model answers `false` uncached so a later download re-probes. Distinct from
**Vision Mode**: this memoizes intrinsic capability, not load-state.
_Avoid_: vision toggle (a **Vision Mode** concept), per-view capability flag.

### Prefill orchestration

**Prefill Plan**:
The pre-prefill decision value for one HTTP prefix-cache generation — the restore
decision (cold vs suffix-prefill), the suffix-filtered checkpoint offsets, the
transient boundary offsets, and the stable-prefix offset. It carries offsets only,
never snapshots or the token array.
_Avoid_: prefill config; generation params (a separate notion); checkpoint plan
(one field inside the Prefill Plan, not the whole value).

**Leaf Admission Builder**:
The GPU-free routing decision for storing one leaf snapshot — given a boundary
mode, it returns either a capture-from-boundary plan or a typed skip reason. It
decides; the actor-side execution does the Metal capture and admit.
_Avoid_: leaf store mode (one input, not the whole story); capture port (it returns
a decision, not a capture).

**Think-Strip Rewind**:
The prefix invalidation a thinking template causes when a new real user message
arrives and the template strips the `<think>` blocks it had kept in the assistant
turns since the previous user query, forcing everything past that divergence point
to re-prefill. It is bounded — not removed — by the canonical-leaf probe.
_Avoid_: cache miss after tools (the felt symptom, not the mechanism); template
drift (the render is deterministic); client mutation.

**Tool Stretch**:
The span of assistant turns and tool results since the last real user message — the
region a thinking template renders with `<think>` kept, and the exact span a
**Think-Strip Rewind** re-renders. A longer stretch means a larger rewind.
_Avoid_: agentic loop, tool session (client-side vocabulary); turn (a stretch spans
many turns).

**Stretch Abandonment**:
The event that a **Tool Stretch**'s continuation never arrives — the client aborts
the in-flight stream, or no follow-up lands within a short idle window after a
tool-calls finish — signaling the next request is likely a real user message and
seeding **Speculative Canonical Prefill**.
_Avoid_: cancellation invalidating the cache (nothing is invalidated); interrupt
handling (UI vocabulary); abandoned request (the stretch is abandoned, not one
request).

**Speculative Canonical Prefill**:
The post-turn countermeasure to the **Think-Strip Rewind**: after a final answer or
a **Stretch Abandonment**, re-prefilling the think-stripped render of the completed
span in the background so the next user turn restores at full depth instead of the
rewind point.
_Avoid_: cache warming (this targets one known future path, not general
pre-population); background generation (it prefills, never decodes).

**Rewind Telemetry**:
The three numbers that make a **Think-Strip Rewind** observable without reproducing
it — the divergence offset, the restore floor, and their gap (the rewind size).
_Avoid_: cache miss (a rewind is a partial hit at a deeper-than-zero floor); latency
spike (the symptom, not the measured cause).

**Preserve-Thinking Render**:
An opt-in render mode, declared by a template that natively supports it, that keeps
`<think>` blocks in every assistant turn so the render is append-stable and the
**Think-Strip Rewind** cannot occur. Being part of the template context, the flag
is part of the cache partition, and retained reasoning permanently occupies
context.
_Avoid_: think retention hack (vendor-sanctioned where the template declares it);
template patching (vendor templates are never edited); global setting (per-model).

**Asymmetric-State Restore**:
The experimental single-prefill counter to the **Think-Strip Rewind**: rather than
re-prefilling the think-stripped **Tool Stretch** (the **Speculative Canonical
Prefill** path), derive a snapshot for the stripped token path from the
think-bearing snapshot by excising each `<think>` span from the sliceable
(attention) layers, re-rotating the retained keys to their shifted positions, and
leaving the non-sliceable recurrent (**MambaCache**) state as-is — advanced
through the stretch (thinks included), since recurrent state is irreversible.
The two layer kinds then serve different renders — attention aligned to the
stripped path, recurrent state still carrying the bearing render — hence
*asymmetric*.
Correctness is unproven (the recurrent state is stale by construction, and
recurrent state is irreversible — ADR-0009); it is a shipped experimental
mechanism only when the user opts in, off by default, with serving fidelity
measured rather than guaranteed.
_Avoid_: stale-state hit / think-stripped hit / trimmed-think hit (the rejected
and colliding earlier names); conflating it with the sliceable/non-sliceable
layer distinction it exploits; **ASR** unqualified — it also means Automatic
Speech Recognition (the **Speech Recognizer** domain; see that entry's _Avoid_),
so say "Asymmetric-State Restore" in full wherever the speech sense could apply.

### Server completion

**Server Completion**:
The actor-confined module that owns one cache-aware HTTP completion on the LLM
actor's isolation, executing what the prefill-orchestration decisions produce
(resolution, restore, suffix prefill, the stream drive, admission, and the leaf
capture). It is the model-affine execution stage, distinct from the HTTP framing
edge and the output-projection stage.
_Avoid_: CompletionHandler (the HTTP framing edge); CompletionProjection (the
terminal output rules) — both are real types, say which; server engine; HTTP
generation pipeline.

**Completion Route**:
The dispatcher's pure decision for one server inference request — cache-aware versus
standard-with-named-reason — computed from request shape alone, never from model
state. Image-bearing requests route cache-aware; only video/audio (or undecodable
images) yield a no-usable-conversation reason.
_Avoid_: prefix-cache bypass (the retired in-actor `nil` returns); fallback flag;
image bypass (decodable images are keyed, not bypassed).

### Client integrations

**Integration**:
A supported external client (OpenCode is the first) paired with Tesseract's recipe
for pointing it at the local server; each Integration is one adapter, and the set
is open-ended.
_Avoid_: connector, plugin, client config (the generated artifact, not the concept).

**Setup One-liner**:
The single copyable terminal command, served by the running server itself, that
configures an Integration end-to-end and reflects live server state each time it
runs.
_Avoid_: install command, onboarding script.

**Config Merge**:
The server-side operation that regenerates Tesseract's own block in a client's
config file while leaving everything else byte-for-byte intact — distinct from a
deep merge, which would interleave fields.
_Avoid_: config write, config sync, deep merge (explicitly not the policy).

**Request Model Selection**:
The contract governing which model a `/v1/chat/completions` request runs on: an
absent `request.model` uses the selected agent model, a downloaded in-catalogue ID
overrides it, and anything else is rejected as `model_not_found` — so `/v1/models`
advertises exactly the downloaded agent models a client may pick.
_Avoid_: ignoring `request.model`, advertising undownloaded catalogue models,
treating the selected agent model as the server's only routable model.

### Settings persistence

**Settings Store**:
The persistence seam between what a setting *means* and where its bytes live: a
typed key-value port with default-on-read semantics, sitting *below* the
**Settings Facade**, never as the module's public interface.
_Avoid_: SettingsManager (that is the **Settings Facade** above it), UserDefaults
(one adapter), preferences store; not the prefix-cache `SnapshotStore` — say
"settings store".

**Settings Store Adapter**:
A concrete **Settings Store**. Exactly two — a `UserDefaults`-backed production one
and an in-memory test one — and having two genuine implementations is what keeps
the seam real.
_Avoid_: backend, provider, mock (the in-memory one is a peer, not a mock).

**Setting**:
The single immutable declaration of one persisted setting: its key, its one
canonical default, and its codec to a stored primitive.
_Avoid_: preference, key, default (a **Setting** *has* those; it is none of them).

**Settings Catalogue**:
The table of all **Setting** declarations — the one home for every default, so
default drift between initial load and reset is unrepresentable.
_Avoid_: defaults dictionary, schema, registry.

**Settings Facade**:
The bindable, observable surface that SwiftUI reads and writes — one property per
setting, forwarding persistence down to the **Settings Store** and hosting the few
non-persistence side effects (launch-at-login, dock visibility) that have nowhere
lower to live.
_Avoid_: settings service, settings model, Settings Store (the seam beneath it).

### Speech model ports and playback

**Speech Recognizer**:
The model-only ASR port below the `TranscriptionEngine` facade — load, transcribe,
cancel a model, nothing more. Everything orchestral (timeout race, lazy load,
lifecycle, error mapping) lives above it in the engine.
_Avoid_: Transcribing (the engine-facing port the dictation coordinator swaps — a
different seam, one layer up), WhisperKitSpeechRecognizer (one adapter),
transcriber, ASR backend.

**Speech Synthesizer**:
The model-only TTS port below the `SpeechEngine` facade, faithful to the model
surface (one-shot/streaming generate, voice anchoring, token offsets). The
synthesis counterpart of **Speech Recognizer**.
_Avoid_: SpeechEngine (the facade above it, not the port), Qwen3SpeechSynthesizer
(one adapter), TTS backend.

**Speech Model Adapter**:
A concrete **Speech Recognizer** or **Speech Synthesizer** — the framework-backed
production actor and its in-memory test peer. Exactly two of each.
_Avoid_: mock, stub, model wrapper, WhisperActor/TTSActor (pre-seam names).

**Audio Playback**:
The main-actor sibling port below `SpeechCoordinator` that turns generated samples
into sound — a collaborator seam, not a model port, and the distinction from
**Speech Synthesizer** (which makes the samples) is the whole point.
_Avoid_: AudioPlaybackManager (one adapter), AVAudioEngine (used inside it), player.

### Speech word timeline

**Word Timeline**:
The pure, immutable per-segment projection of spoken text plus playback position
into a highlighted character count and active word — the single home for the
token→char→word model and the pacing fold. Holds no timer, clock, observable state,
or UI; the **TTS Word Tracker** supplies all of those and drives it.
_Avoid_: WordPacing (names the operation; this names the value), TTSWordTracker (the
driver above it, not the value), word highlighter, pacing model, overlay panel (a
different, dictation surface unrelated to the TTS timeline).

**TTS Word Tracker**:
The observable, main-actor stateful driver of the pure **Word Timeline** — owns the
frame timer, the playback-clock seam, the cross-segment estimate, and the published
state the notch overlay reads. It decides *when* to re-fold; the timeline decides
*what* the fold yields.
_Avoid_: word timeline (the pure value it drives), word state machine.

**Segment Window**:
The single playback-time base a **TTS Word Tracker** measures one long-form
segment's pacing against — one value, so a time-base/duration-base disagreement is
unrepresentable.
_Avoid_: segmentTimeBase / segmentDurationBase (the old coupled pair this replaced),
segment offset, time base.

**Segment Playback**:
The deep module owning the consume-one-TTS-stream-into-playback loop shared by every
speech path; the only per-segment difference is a small `Segment` value (optional
boundary plus initial state), never flags. It drains samples into **Audio
Playback** and drives the **Word Highlight Surface**, leaving cleanup to each
caller.
_Avoid_: chunk loop, stream pump, playback driver, a config-flag loop.

**Word Highlight Surface**:
The main-actor port that **Segment Playback** and `SpeechCoordinator` drive to
render spoken-word highlighting (show, switch, mark complete, dismiss). The
production adapter is the notch panel; a recording test peer makes the
segment-boundary switch assertable.
_Avoid_: notch overlay / TTSNotchPanelController (one adapter, not the seam),
highlight view, **Overlay Panel** (the separate dictation HUD surface).

### Generation accumulation

**Generation Accumulator**:
The one value that folds an `AgentGeneration` event stream into a single assistant
turn's accumulated state — text, optional thinking, finalized tool calls, the raw
malformed-tool-call buffer, the safeguard's safe-prefix length. A pure value with no
side effects and no output type; each caller supplies its own loop and its own
**Generation Projection**. (`thinking == nil` means no `<think>` block ever opened;
`""` means one opened but is empty so far — never collapse the optionality.) Its
`surfacesMalformedBuffer` query is the single home of the malformed→text fallback
predicate (empty text, no successful tool calls, non-empty malformed buffer), a
derived `Bool` — not an output shape — consumed by both **Generation Projection**s.
_Avoid_: StreamResult, event handler, GenerationFold (names the fold's *operation*,
not the value — and "fold" also means a reducer); ToolCallParser (the upstream
source of the events, not the accumulator).

**Generation Projection**:
The per-caller step that maps **Generation Accumulator** state to one caller's output
shape (`AssistantMessage`, **CompletionProjection**, the leaf-store message, bare
text). It covers both *terminal* projection (the committed message / final response)
and *intermediate* per-event projection for streaming callers (a snapshot + delta per
event, as the agent path's **AssistantMessageProjection** emits). It is where caller
intent lives, kept out of the shared fold; it is a concept, not a single type.
_Avoid_: conversion, adapter (not a seam adapter), output builder.

**CompletionProjection**:
The server's concrete **Generation Projection** — the one home for the rules both
HTTP completion paths (streaming SSE, non-streaming JSON) share when building a
response from a terminal accumulator, so the two paths differ only in framing. It
applies the malformed→text fallback whose predicate lives once on the accumulator
(`surfacesMalformedBuffer`), shared with **AssistantMessageProjection**.
_Avoid_: StreamResult (the dissolved server per-path capsule; a same-named *private*
agent-loop type still exists, so do not reuse it for the server), response builder,
envelope (the per-path framing it feeds).

**AssistantMessageProjection**:
The agent path's concrete **Generation Projection** — the sibling to
**CompletionProjection** that the agent stream driver composes directly (no port).
A stateful value owning the turn's tool-call *identity* (`ToolCallInfo` with stable
ids, which the accumulator's raw `[ToolCall]` lacks): `step` maps each folded event
to the driver's next action (a per-event snapshot + delta, the malformed event, or
nothing), `snapshot` is the raw partial turn (per-event / cancel / error), and
`finalize` applies the shared `surfacesMalformedBuffer` fallback for the terminal
message. Pure — every emit and log stays in the driver.
_Avoid_: StreamResult, message builder, AssistantMessageFactory; CompletionProjection
(the server sibling — say which path).

### Generation stream loop

**Generation Stream Loop**:
The one home that consumes a single raw model generation stream into the agent's
`AgentGeneration` event stream for one assistant turn, under the thinking-loop
safeguard — owning the parser lifecycle and the safeguard's truncate-and-restart
across stream swaps. Caller side effects and projections stay with the callers;
terminal info and diagnostics come back on its outcome.
_Avoid_: managed generation (an `AgentEngine` method, not this loop); stream consumer
/ generation pump; GenerationFold (the fold is the **Generation Accumulator**);
ToolCallParser (upstream); agent loop — this consumes one turn's raw stream, whereas
the agent double-loop orchestrates turns and tool calls above it (say "stream loop"
vs "agent loop").

### Chat transcript projection

**Chat Transcript**:
The pure, stateless projection of the agent message log into the flat `[ChatRow]`
the chat list renders, grouped into **Turn**s. All inputs — expansion state, the
live stream, formatting — are passed in; it reads no coordinator state and has no
side effects.
_Avoid_: rebuildRows (the duplicated body it replaced), row builder, ChatRowBuilder,
view model, render model; ASR transcription — unrelated to `TranscriptionEngine` /
`TranscriptionResult` (speech-to-text), so say "chat transcript".

**Turn**:
The **Chat Transcript**'s grouping unit — a contiguous run from one user message (or
compaction marker) through the assistant's complete response, possibly spanning
several assistant messages when a tool-calling loop runs.
_Avoid_: round, exchange, conversation turn, message group; loop turn — a transcript
**Turn** can contain several agent-loop turns (say "transcript turn" vs "loop turn").

**Chat Row**:
The flat, render-ready `Equatable & Sendable` atom of the **Chat Transcript**, with
every string pre-computed and a stable `id` — the unit SwiftUI diffs.
_Avoid_: cell, item, list element, view model.

**Chat Transcript Controller**:
The stateful driver that feeds the pure **Chat Transcript** fold: it holds the
view-interaction state (expansion, streaming throttle, splice point) and makes the
full-rebuild-vs-tail-patch decision the stateless projection cannot.
_Avoid_: view model, render model, ChatViewModel, row store.

### Agent run lifecycle

**Agent Run**:
The lifecycle of one *foreground* LLM invocation — a `sendMessage` turn or a
`/compact` — serialized behind the GPU lease, from queued through active to
cancelled or done. Distinct from the Generation* family, which is the token stream
*inside* a turn; an Agent Run is the outer lease + busy + cancel envelope, and its
`isGenerating` means "queued **or** active," not just running.
_Avoid_: generation lifecycle (collides with the Generation* family), send
coordinator, busy flag as standalone spine state.

### Agent state reduction

**Agent State Reducer**:
The single fold of the `AgentEvent` stream into run-level `AgentState`, total over
every event. Distinct from the **Generation Accumulator**, which folds one turn's
*token* stream into message content — the reducer folds *lifecycle* events into the
observable run state.
_Avoid_: Generation Accumulator (the token-stream fold — say which "fold"), event
handler / `handleEvent` (this is the fold, not the notify wrapper that hosts it),
dispatcher, state machine, store.

### Agent coordinator leaves

The publisher-agnostic sub-controllers carved off `AgentCoordinator` that own their
own state but never touch the event dispatcher — the *leaves*, as opposed to the
dispatcher-coupled *spine* (**Agent Run**, **Chat Transcript Controller**).

**Voice Input**:
The agent chat composer's push-to-talk capture→transcribe→emit module: it composes
the shared **Voice Capture Session**, hands transcribed text to the composer rather
than sending, and keeps its errors local instead of on the shared banner. Distinct
from the spine — it touches no `Agent` and no arbiter.
_Avoid_: dictation (the separate global system-wide overlay — say "agent voice
input"), mic controller, voice state machine.

**Image Draft**:
The agent chat composer's image queue and preview module: it owns pending
attachments, full-window image drops, model-capability hinting, and the Quick Look
request projection. It receives committed conversation images through an injected
read closure, so it stays a leaf rather than reaching into `Agent` or the
conversation store.
_Avoid_: image input (the broader UI affordance/capability concept), image cache
(the server-side **Image Digest** cache path), Quick Look host (the AppKit bridge it
feeds).

**System Prompt Inspector**:
The system-prompt transparency module: it renders the *already-assembled* prompt
into raw ChatML plus a token count, on demand. Distinct from the prompt builder,
which assembles the prompt; this only inspects it.
_Avoid_: prompt builder (assembles; this inspects), token counter.

**Command Palette**:
The slash-command popup module: registry, filtering, selection, and autocomplete for
the *presentation* of slash commands. Distinct from command execution (which stays
on the spine) and from the pure parser/registry types it merely drives.
_Avoid_: command executor / router (execution stays on the spine), slash command
registry (the pure type it drives).

### Operation staleness

**Operation Guard**:
The shared stale-result protocol for the capture→transcribe→commit coordinators: a
monotonic epoch that advances on cancel and on each new operation, so a post-`await`
epoch check can reject a result from a superseded operation. Distinct from `Task`
cancellation — it catches a recognizer that ignores cancellation and returns success
anyway.
_Avoid_: operation ID / `currentOperationID` (the bare counter it replaced),
cancellation token (it does not own `Task` cancellation), debounce, sequence number;
not Swift's `guard` statement — say "operation guard".

**Operation Ticket**:
The epoch snapshot a coordinator captures when it enters async work; its `isCurrent`
check, after each `await` resume, decides whether still-running work may commit.
_Avoid_: operation ID, token (unqualified), snapshot (the prefix-cache concept).

**Voice Capture Session**:
The one concrete module that owns the push-to-talk capture→transcribe→commit
lifecycle — the **Operation Guard** ticket discipline, the microphone-busy guard, the
minimum-duration and empty-text guards, post-processing, the in-flight transcription
`Task`, and cancellation — behind a small value-returning interface
(`start`/`stop`/`transcribeAndCommit`/`cancel`), delivering clean text to a
caller-injected commit closure. Composed *directly* by both `DictationCoordinator` and
**Voice Input**, which keep only their own state, errors, sounds, and commit. Distinct
from **Voice Input** (one caller, agent-composer presentation) and from the **Operation
Guard** it composes (the epoch protocol alone).
_Avoid_: coordinator (it is composed by the coordinators, not one), capture engine
(`AudioCaptureEngine`, the mic port below it), voice controller, session (unqualified).

### GPU lease arbitration

**GPU Lease Queue**:
The pure mutual-exclusion lease for the GPU: a single scoped operation grants one
caller exclusive use at a time, FIFO. Slot-agnostic — it knows nothing of models,
engines, or slots; the policy and ownership layer above it is the **Inference
Arbiter**.
_Avoid_: arbiter (composes this), GPU mutex/semaphore, scheduler (no policy beyond
FIFO).

**Inference Arbiter**:
The model-affine layer that composes a **GPU Lease Queue** with model ownership,
holding the lease across both load and body so the loaded model cannot change under
a running consumer. Distinct from the lease queue below it (no model awareness) and
from `ModelDownloadManager` (acquisition, not arbitration).
_Avoid_: lease queue (the layer below), model manager (collides with
`ModelDownloadManager`), GPU manager.

**Inference Arbitrating**:
The narrow single-member seam that lease-acquiring consumers depend on, satisfied by
the production **Inference Arbiter** and an in-memory test peer. A consumer needing
reload or model-state access reaches for the concrete arbiter instead, so the seam
stays minimal rather than widened speculatively.
_Avoid_: arbiter protocol / arbitering, lease provider; widening it before a
peer-consuming caller needs the member. ("Lease" unqualified = GPU lease, not the
prefix-cache snapshot pin.)

### Model loading

**Model Identity**:
The value computed once from a model directory at load that answers "what model is
this, and what does that imply downstream" — tool-call format, family facts,
thinking-prompt and image-keying behavior, and a total `flopProfile`. The load-time,
directory-derived capability value; distinct from `ModelFingerprint` (a throwing
hash for cache invalidation) and from the runtime engine container.
_Avoid_: ModelProfile (would collide with `ModelFlopProfile`, its eviction-cost
field), model config / `config.json` dict (a source, not the value), ModelFingerprint
(separate). "Model identity" vs "flop profile" — the latter is one field of the
former.

### Cache memory budget

**Pressure-Reactive Budget**:
The RAM-tier byte budget expressed as a band rather than a constant — an auto-sized
ceiling and a current value that OS memory-pressure events push down and hysteresis
regrows, never below the **Budget Floor**. The cache is greedy when RAM is idle,
polite when it is contested.
_Avoid_: static budget, memoryBudgetBytes-as-constant, cache size limit. (This is the
RAM tier; the SSD tier has its own separate, static byte budget.)

**Budget Floor**:
The content-defined lower bound of the **Pressure-Reactive Budget**: the minimal
survival set — the `.system` chains plus the single most-recently-extended leaf —
kept resident even at critical pressure. A last-resort floor, not the protection
mechanism (defending the main-agent leaf against subagent churn is the eviction
score's job).
_Avoid_: minimum cache size, reserved bytes, fixed floor, per-partition floor,
workload heuristics in the floor.

**Snapshot Demotion**:
Moving a snapshot's body out of RAM while keeping it recoverable — backing it to SSD
first, then dropping the RAM body — so the next hit pays a cheap hydration instead of
a re-prefill. The first response to any RAM-tier shrink; outright dropping (eviction)
is the fallback when SSD backing is unavailable.
_Avoid_: spill, flush, evict-to-SSD; eviction (terminal — a demotion is recoverable,
and supersession *preserve* differs again in keeping an ancestor's SSD backing).

### Eviction tuning

**Recovery Cost**:
What the next hit pays if a snapshot leaves a tier — the tier-aware numerator in
eviction scoring: hydration cost for an SSD-backed RAM body, re-prefill cost where
loss is terminal. Denominated in seconds from rolling measured device rates (never
guessed constants) so hydration and re-prefill compare in one unit; distinct from the
single-tier reading of the term as the FLOPs a snapshot embodies.
_Avoid_: FLOP savings (the embodied-FLOPs reading), flops-per-byte (unqualified —
density flattens for backed bodies), parentRelativeFlops (one ingredient, not the
concept).

**Eviction Configuration**:
The `(flopProfile, alpha)` pair the prefix cache scores eviction against — the single
mutable cell owned by `PrefixCacheManager`, passed to the pure-function scorers by
value. `flopProfile` is fixed from **Model Identity** at cache build; `alpha` starts
at the LRU default and adapts at runtime via the **AlphaTuner**.
_Avoid_: `EvictionPolicy.modelProfile` / `.alpha` (retired statics), eviction settings
(not a user **Setting**), model profile as a global. ("Flop profile" = the immutable
per-architecture cost model; the config is the pair whose `alpha` half is mutable.)

**AlphaTuner inversion**:
The dependency direction between tuner and cache: the **AlphaTuner** is constructed
with the production `flopProfile`, replays each grid-search candidate in its own
sandbox, and *returns* the winning `alpha` for the manager to assign — holding no
back-reference to the manager and writing no global. The inversion is that the manager
pulls the result, not that the tuner pushes it.
_Avoid_: writing a global alpha, tuner→manager callbacks or weak back-references.

### Overlay presentation

**Overlay Panel**:
The transparent, click-through global `NSPanel` that floats above all apps and
shows/hides in reaction to dictation state — the shared behaviour behind both the
dictation pill HUD and the full-screen border. Two overlays differ only in their
**Overlay Placement** and hosted content view; the interactive TTS notch is a
separate panel, not an Overlay Panel.
_Avoid_: overlay controller / manager, HUD window, generic NSPanel wrapper,
config-flag panel, the TTS notch panel (a separate, interactive surface).

**Overlay Placement**:
The whole injected difference between one **Overlay Panel** and another, expressed as
a pure value: where the panel sits for a given screen and dictation state, plus
whether it animates its resize. Two presets exist — pill and full-screen border.
_Avoid_: layout strategy, frame provider, overlay style (the user **Setting** that
selects which placement is live, not the placement itself).

**Screen Geometry**:
The plain screen rectangles — full frame and visible frame — that an **Overlay
Placement** consumes, decoupled from any live `NSScreen` so the frame math stays
unit-testable.
_Avoid_: an `NSScreen` (deliberately not passed to placements), a bare single rect
(placements need both frames).

### App composition

**App Bindings**:
The module owning the app's launch sequence and every long-lived runtime subscription
that carries a rule (model auto-load and hot-swap, lazy-reload guards, server and
overlay-style reactions, hotkey rebinding, the single dictation-state fan-out) — the
launch-time mirror of the teardown-owning termination coordinator. Distinct from the
composition root, which stays pure wiring with no behaviour.
_Avoid_: app glue (pre-carve working name), setup() behaviour, launch coordinator,
app services, a SwiftUI `Binding` (view data flow, unrelated).
