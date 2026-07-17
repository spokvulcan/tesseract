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

**SSD Residency**:
One coherent read of the SSD tier's observable state — resident descriptors,
the accounted byte total, extension-transfer shields, partition metas —
captured under a single ledger lock hold. The typed observation surface for
diagnostics and tests; the only sanctioned way to *read* the tier from
outside (driving hooks for tests remain separate and effect-only).
_Avoid_: ForTesting reads (the retired ad-hoc accessors this replaces);
manifest dump (residency is the in-memory authority's view, not the disk
file's).

**Survival Gate**:
The SSD admission pre-check that admits an incoming chain only if it would survive
the eviction its own write triggers, skipping the write otherwise. It decides
*whether* a write happens at all (unlike the **Leaf Extension Admission** worth-it
gate, which decides the *shape* of a write already happening); it bites only under
budget contention, and end-of-turn leaf writes bypass it.
_Avoid_: judicious admission (the Marconi-paper mechanism this derives, not
copies); admission policy (vague); write filter.

**Adaptive Write Eagerness**:
The admission-time skip of a non-guarantee SSD write while RAM comfortably holds
the body and the node has not proven reuse — the copy would be pure redundancy,
and **Recoverable Eviction**'s demote-before-drop persists it later if RAM ever
needs the bytes back. Reuse re-earns the write: crossing the hit-count threshold
issues a one-shot deferred-class promotion write off the lookup path. The
end-of-turn leaf (guarantee class) is never deferred.
_Avoid_: write throttling (a rate limiter — explicitly not built); lazy writes
(the guarantee class is never lazy); HiCache write_through_selective (the
precedent it extends with RAM-tier health, not the mechanism itself).

**Storage Activity Gate**:
The shared busy signal between the inference path and the SSD writer: while a
hydration read or a prefill is in flight, deferred-class writes wait (bounded by
a holdup ceiling; flushes force-drain), because concurrent large-block reads and
writes on one NVMe device collapse total bandwidth. Guarantee- and
write-through-class writes ignore it — durability outranks bandwidth.
_Avoid_: I/O scheduler (it delays one write class, it does not schedule I/O);
write lock (nothing blocks the inference side).

**Endurance Ledger**:
The persistent bytes-written / bytes-deleted counters for the SSD tier, keyed by
write class and delete reason, bucketed hourly and daily, accumulated from the
same diagnostics events the JSONL file sink records — so the counters reconcile
with `ssdAdmit accepted` sums by construction. "Measure before throttle": it
exists so a future write limiter would be justified by field data. Also buffers
the panel's sparing notable events (partition invalidations).
_Avoid_: write throttle, SSD protection (no such knob ships); telemetry store
(the window-scoped event buffer — this ledger is eager and survives restarts).

**Stale-Partition GC**:
The warm-start reclaim of partitions unused past a fixed gap measured *relative
to the tier's freshest partition's use stamp* — never the wall clock, so an idle
tier ages together and survives any break, while an abandoned kv-config or
template-digest variant of a still-active model ages out. "Use" is admission or
an SSD hit; warm start itself never refreshes a stamp, and legacy stamp-less
partitions are grace-stamped without inflating the anchor.
_Avoid_: TTL, expiry (absolute-clock framings); cache eviction (that is the
byte-budget LRU cut — GC is staleness, not space).

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

**Appshot**:
A hotkey-invoked capture of the frontmost window — whatever window that is,
Tesseract's own included — staged as a pending composer image identified by its
source app name and window title. Image-only: no accessibility text is read from
the captured window.
_Avoid_: snapshot (owned by the prefix cache lifecycle), "capture" unqualified
(dictation vocabulary for voice), screenshot (generic any-pixels; an Appshot is
the frontmost window specifically).

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

**Model Fetching**:
The narrow hub port below the model download lifecycle — list a repo's files,
fetch one file, resolve-or-download a snapshot — satisfied by the
HuggingFace-backed production adapter and a scripted in-memory test peer. Disk
stays outside the seam: file checks and status computation run against the real
file system.
_Avoid_: hub client (one adapter, not the seam), download client/backend, model
fetcher; widening it past the three verbs before a second consumer needs a
member.

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

**Client Prefix Divergence**:
A deep prefix-cache loss caused by the client changing early tokens of its own
prompt mid-session (e.g. OpenCode re-injecting the live content of an AGENTS.md
the session itself is editing) — the tokens genuinely differ, so the re-prefill
is correct, and the loss is attributed at lookup, never "fixed" with fuzzy
matching. See `docs/prompt-cache-client-divergence.md`.
_Avoid_: cache bug, server-side loss (it is neither); **Think-Strip Rewind** (the
tail-local cousin — a template artifact, not a client prefix change).

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

**Completion Phase Map**:
The six named phases of one cache-aware **Server Completion** (ADR-0033):
**Request Keying** (conversation → the identities later phases key on, or the
**Unkeyed Completion** degrade), resolution + plan (the Prefill Planner),
plan application (inline by decision — the deletion test fails), the stream
drive (the Managed Generation Driver, shared with the agent), the leaf store,
and trace accumulation. Phases are implementation structure inside the
module's seam — the dispatcher's interface is unchanged — and each phase
returns values; the completion module owns effects.
_Avoid_: pipeline stages (the Generation* family owns "stream" vocabulary);
new entry points (ADR-0015's seam is untouched); extracting plan application
(recorded shallow — see ADR-0033).

**Completion Trace Accumulator**:
The fold of one cache-aware **Server Completion**'s trace facts into the terminal
trace record — the terminal-vs-recovered eviction tally paired with its correlated
diagnostics emission (so tally and log lines cannot drift), the restored-offset
rule, and the admitted-snapshot projections. The drive feeds facts; the value
decides what the record contains.
_Avoid_: telemetry store (the window-scoped event buffer); trace logger (it
derives, the diagnostics context logs).

**Completion Route**:
The dispatcher's pure decision for one server inference request — cache-aware versus
standard-with-named-reason — computed from request shape alone, never from model
state. Image-bearing requests route cache-aware; only video/audio (or undecodable
images) yield a no-usable-conversation reason.
_Avoid_: prefix-cache bypass (the retired in-actor `nil` returns); fallback flag;
image bypass (decodable images are keyed, not bypassed).

**Model Session**:
The scoped, Metal-affine model handle **Server Completion** enters for one batch of
model verbs (prepare, cache creation, restore, prefill, decode iteration, snapshot
capture) — one session is one Metal-affine batch, so the ADR-0015 affinity
discipline lives at this seam. Two adapters make it real: the container-backed
production adapter and the toy-model-backed test peer that runs the module's
sequencing without a downloaded model.
_Avoid_: session (unqualified); Generation Session (the Generation* family is the
token-stream vocabulary, not the model handle); Inference Session (collides with
**Inference Arbiter**); model surface / perform wrapper (the mechanism, not the
concept); widening it before a second consumer needs a member.

### Streaming tool calls

**Argument Transcoder**:
The server-side component that converts model-native in-flight tool-call text
(Qwen `<function=…>` XML, or the JSON wrapper body) into OpenAI
`function.arguments` **Argument Fragment**s, incrementally. It engages only after
the function name is locked and only for formats it understands (per-format
strategies behind one seam); every other format keeps the atomic
name-then-full-arguments emission. When engaged, its fragments are authoritative
for the streamed wire — the parser's final tool call is a diagnostic
cross-check, never re-sent.
_Avoid_: tool-call streamer (the deltas already stream internally; this
transcodes), converter (ToolCallConverter is the non-streaming id/JSON adapter),
parser (upstream — it produces the deltas the transcoder consumes).

**Argument Fragment**:
One streamed piece of a tool call's `function.arguments` on the OpenAI wire. The
concatenation of a call's fragments is the canonical arguments JSON: it must
parse, and no strict prefix of it may parse (clients finalize on the first
parseable accumulation). Schema-typed non-string parameter values are emitted
whole at parameter close; string values stream progressively.
_Avoid_: chunk (prefill vocabulary), delta (the internal parser event —
`.toolCallDelta` carries raw model text, a fragment carries transcoded JSON).

**Wire-Valid Close**:
The closure rule for a tool call already on the streamed wire: there is no
retraction, so any termination — malformation, cancel, safeguard intervention,
max-tokens — synthesizes closers so the accumulated **Argument Fragment**s still
parse as JSON, then the stream finishes with the appropriate finish reason. The
malformed→text fallback survives only where nothing was streamed yet.
_Avoid_: error recovery (the client's tool-error loop handles semantics; this
only guarantees wire validity), abort (the stream ends validly, not abruptly).

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

### Agent browser and MCP

**Agent Browser**:
The app-owned WebKit browser that agents drive — the single web-access path for
every Tesseract web capability, always rendered in visible windows so the user
can watch and intervene.
_Avoid_: headless browser (visibility is the point), webview, computer use.

**Agent Profile**:
The persistent credential silo behind the **Agent Browser** — only the logins the
user deliberately performs inside it, never imported from and never shared with
the user's personal browsers. Curating what it is logged into *is* the security
model.
_Avoid_: cookie import, session sync, real/default profile.

**Browser Session**:
One MCP client's private set of tabs over the shared **Agent Profile**; sessions
never see each other's tabs while login state is common to all.
_Avoid_: shared tab set, global browser state.

**Ephemeral Page**:
A cookieless page outside the **Agent Profile** — the anonymous mode backing
plain fetch and search, so casual reads don't carry the agent's identity.
_Avoid_: incognito, private browsing.

**Page Read**:
The default read path — a readability-distilled markdown extraction of the
current page, paginated under a hard token cap.
_Avoid_: page source, raw HTML dump.

**Page Map**:
The interaction representation — a pruned accessibility-tree outline with stable
element refs, requested only when the agent must act on the page rather than
read it.
_Avoid_: snapshot (collides with the prefix-cache Snapshot vocabulary), a11y
dump, DOM tree.

**Browser MCP Server**:
The MCP endpoint the running app serves so external agents can drive the
**Agent Browser**; Tesseract's own agent consumes it through its **MCP Client**,
speaking the same protocol over an in-process transport (not the loopback
socket, so browser-use in chat never depends on the inference server running).
It is the _sole_ web-access surface — search, fetch, and interactive browsing all
live here, with no standalone web tools beside it — and is governed by two
independently-default-on switches: **Web Access** (its tools reach the in-app
agent over the in-process transport, no port opened) and **HTTP exposure** (its
loopback listener admits outside clients).
_Avoid_: standalone server, stdio server, plugin API, web\_search/web\_fetch (the
retired standalone tools).

**MCP Client**:
The in-app agent's client for Model Context Protocol servers (#190): it connects
to configured HTTP servers — and to the app's own **Browser MCP Server**
in-process — so their tools materialize in the agent's registry alongside
built-ins, namespaced by server. Dogfooding the browser server through it
(ADR-0027) keeps one honest tool surface.
_Avoid_: plugin loader, tool proxy, RPC bridge.

**Connected Server**:
One configured MCP server the **MCP Client** talks to — URL, display name,
enabled flag, optional headers. The **Browser MCP Server** is the pre-registered
first entry (default-on; governed by the Web Access and HTTP-exposure switches
above); user servers are added deliberately and persist through the
**Settings Catalogue**.
_Avoid_: integration, plugin, AgentExtension (that is the in-app tool-source type).

**Tool Consent**:
The explicit user approval a user-added **Connected Server** requires before its
tools reach the agent — no third-party tool surfaces silently. Adding a server
is the consent gate; disabling one instantly withdraws its tools.
_Avoid_: allowlist, sandbox policy, capability grant.

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

**Playback Diagnostics Dump**:
The pure value that turns one TTS playback's captured samples plus conditions into
the on-disk diagnostic artifact bytes (WAV + metadata) — the playback-side sibling
of the dictation **Capture Dump**. The playback adapter feeds it and writes what it
returns; encoding is byte-testable without an audio engine.
_Avoid_: capture dump (the dictation ring buffer), WAV encoder (one part of it),
debug recording.

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
Retired in engine v2 (ADR-0038): its stream-drain became the caller's
`for try await` over an **Utterance**, its boundary poll became **Segment
Script**'s `startFrame`, its pause poll became demand-based pacing. Kept here
until the last v1 reference dies.
_Avoid_: chunk loop, stream pump, playback driver, a config-flag loop.

**Word Highlight Surface**:
The main-actor port that `SpeechCoordinator` drives to render spoken-word
highlighting (show, switch, mark complete, dismiss). The production adapter is
the notch panel; a recording test peer makes the segment-boundary switch
assertable. In engine v2 its switch timing comes from **Segment Script**
ground truth, not playback bookkeeping.
_Avoid_: notch overlay / TTSNotchPanelController (one adapter, not the seam),
highlight view, **Overlay Panel** (the separate dictation HUD surface).

### Speech engine v2 (ADR-0038)

**Speech Session**:
The voice-identity owner at the engine boundary: opened from a `SessionProfile`
and a **Voice**, it holds the voice-prefix KV and the anchor per policy, admits
utterances one at a time, and dies by `close()`. Sessions survive engine unload
as ingredient values and rebuild KV transparently.
_Avoid_: generation session (an LLM concept), voice handle, session manager.

**Utterance**:
One admitted text→speech run: admission-time facts (sample rate, frames/s,
segment count) plus the single event stream that is simultaneously the audio
transport, the timing channel, the backpressure channel, and the cancellation
token. Dropping it stops generation; cancelling the consuming task surfaces
`CancellationError` untranslated.
_Avoid_: speech stream (one field of it), generation handle, request.

**Segment Script**:
The per-segment value announced on the stream before its audio: text slice,
token→char offsets, and the utterance-global `startFrame` — the **Segment
Window** as ground truth from the generation loop, ending the estimator era.
_Avoid_: segment metadata, offsets payload.

**Pinned Voice**:
Voice identity as a serializable value: voice spec + the ≤48 anchor code frames
+ a {model, precision, schema} fingerprint — a few KB, rebuildable into KV, the
thing that makes the companion's voice survive relaunch. Restore validates the
fingerprint or throws; a seed is never voice identity (#339).
_Avoid_: voice anchor (the KV realization inside a session), seed, voice id.

**Readiness**:
The engine lifecycle ladder — `.unloaded` / `.loaded` / `.warm(priming:)` —
driven by one idempotent `prepare` verb; warmup and voice-prefix priming are
rungs, not separate APIs.
_Avoid_: engine state (the observable presenter concept above the seam), load
status.

**Pacing Policy**:
The demand-based backpressure contract: `.eager` or `.lookahead(segments: n)` —
at most n undelivered segments beyond the in-flight one, every engine wait
happening outside the GPU lease. Pause is its consequence, not an API.
_Avoid_: throttling, playhead clock (rejected design), lookahead buffer.

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
_Avoid_: managed generation (the **Managed Generation Driver** above it, not this
loop); stream consumer / generation pump; GenerationFold (the fold is the
**Generation Accumulator**); ToolCallParser (upstream); agent loop — this consumes
one turn's raw stream, whereas the agent double-loop orchestrates turns and tool
calls above it (say "stream loop" vs "agent loop").

**Managed Generation Driver**:
The one module that drives a raw model stream through the **Generation Stream
Loop** for both consumers — the agent turn and the cache-aware **Server
Completion**: safeguard derivation from the request's parameters, loop
construction, late-bound cancel bridging, stream-termination cancel wiring, and
the terminal-info re-yield into the caller's sink. Callers keep their own task
envelopes, diagnostics, and projections.
_Avoid_: wrapManagedGeneration (the `AgentEngine` call site); stream driver (the
agent's projection-composing layer above); generation manager.

### Chat rewrite vocabulary (ADR-0024)

The canonical terms of the rewritten chat; they replaced the Chat Transcript
projection family.

**Content Part**:
The typed, ordered unit of assistant message content — text, thinking, or tool
call — a verbatim Swift mirror of pi-ai's content model. Stream events address a
part by its content index within the message.
_Avoid_: block, segment, chunk; Chat Row (a render atom, not a model unit);
part (unqualified) when ambiguity with tool-result content is possible.

**Live Part**:
The single observable box holding the one **Content Part** currently streaming —
the only mutable render state during generation; a token delta invalidates only
its view, and at part end it commits into immutable value rows.
_Avoid_: streaming bubble, stream message; partial message (in the pi-mono event
protocol "partial" is the whole-message snapshot carried by every event, not the
live box).

**Chat Session**:
The single store holding the event fold for the active **Conversation** —
messages, the **Live Part**, and run phase — and the sole agent-event subscriber.
Leaf controllers (composer draft, voice, pills) live outside it, owned by views.
_Avoid_: coordinator, dispatcher, view model; session (unqualified); Conversation
(the persisted document a Chat Session folds live).

**Workspace**:
The user-facing name of the agent tool sandbox root — the directory file tools
resolve paths against. Tool rows render their targets workspace-relative and say
"workspace" for the root itself.
_Avoid_: sandbox (the enforcement mechanism, not the place); root / home
directory; "." as a user-facing label.

**Tool Row Title**:
The verb + target grammar of a tool-call row — an imperative verb (Read, Write,
Edit, List, Load skill, Search, Fetch) plus a **Workspace**-relative target;
unknown tools fall back to the raw tool name.
_Avoid_: progressive verbs ("Reading…" — implies still running, which the
spinner owns); bare tool names for known tools; filename-only targets.

**Open Tool Call**:
The chat-side partial tool-call **Content Part** — born with a stable id and
real name the moment the streaming parser locks the tool name, its arguments
accumulating raw fragments until the call parses (the normalized-JSON guarantee
applies only to committed parts). Lives only in the live message; if the turn
ends without a parsed call (malformed, truncated, aborted) it vanishes without
trace.
_Avoid_: fragment (the server wire term); pending tool call (the
executing-phase set); partial part (ambiguous with the whole-message "partial"
snapshot).

**Tool Clock**:
The single per-call wall clock: starts when the **Open Tool Call** is born,
ticks live in whole seconds (appearing once elapsed reaches 1s), runs
continuously through writing, waiting, and execution, and freezes into the
row's duration badge at execution end. The badge therefore reads "time from
first visible to result."
_Avoid_: execution duration (the pre-2026-07 badge semantics — execution-only);
generation time / write time (phases of the one clock, never shown separately).

**Row Rhythm**:
The single vertical spacing between every pair of transcript rows — between
messages and within them alike, one shared constant, deliberately without
clustering exceptions (a run of consecutive tool rows is *not* grouped tighter).
_Avoid_: separate item-spacing / part-spacing knobs; section gap.

**Prose Accent Palette**:
The named color roles applied to assistant markdown syntax — heading, strong,
emphasis, inline code, link, list marker, blockquote bar — so document
structure reads at a glance; colors taken exactly from OpenCode's default
theme, dark and light variants alike, plus the neutral inline-code chip
behind the code accent. With the **Code Accent Palette**, one of the two
sanctioned exceptions to the otherwise monochrome chat content layer.
_Avoid_: coloring body prose, quoted blockquote prose, or user messages;
per-view ad-hoc colors outside the named roles.

**Code Accent Palette**:
The named color roles for code and diffs in the transcript — syntax-highlight
token roles (mapped from syntect scopes in Tool Panels, from the markdown
renderer's tokenizer in assistant-prose code blocks) plus the semantic diff
tints (added-green, removed-red) — with dark and light variants derived from
system semantic colors. The second sanctioned color exception to the
monochrome chat content layer; diff tints are semantic like error red, never
decoration.
_Avoid_: stock .tmTheme colors; using diff green/red outside added/removed
semantics.

**Markdown Gallery**:
The always-shipped instrument window for the chat's markdown rendering: an
editable source pane, pre-filled with the canonical all-construct document,
rendered live through the chat's own markdown stack in light, dark, or both
appearances side by side. Markdown chrome is judged and tuned here, against
the canonical document, in the exact render path the transcript uses.
_Avoid_: debug-only gating (the dev loop runs Release builds); a second
preview-only render path (the gallery renders through the chat's, or it
proves nothing).

**Tool Panel**:
The specialized expanded body of a tool-call row: a per-tool derived rendering
(diff, file slice with real line offsets, listing, search-result list, rendered
page markdown, image thumbnails) that replaces raw arguments/result JSON in the
UI. Unknown (external MCP) tools fall back to the generic panel — pretty-printed
arguments plus result text. The exact wire payloads stay persisted in the
transcript model; the panel is only a projection of them.
_Avoid_: expanded view / detail section (the pre-2026-07 raw-JSON body);
raw-mode toggle per panel (rendered/raw switches exist only where a rendered
mode does, e.g. fetched markdown).

**Panel Cap**:
The length policy of a **Tool Panel**: the first ~40 lines, then a quiet
"Show N more lines" row that expands the rest fully inline. Panels never scroll
internally — the transcript remains the only scroller.
_Avoid_: nested scroll views; fixed panel heights; truncation without a path to
the full content.

**Pending Row**:
The user message rendered from send until the event spine commits the same
message — ephemeral derived view-state like the **Live Part**, never agent
state, so the transcript shows the message instantly while the run is still
queued behind the GPU lease or a model load. If the run dies before the commit
(cancel while queued, load failure) it vanishes and its content is restored to
the composer.
_Avoid_: optimistic update / eager append (the rejected agent-state mutation);
local echo; queued message (the lease-queue concept, not the row).

**Waiting Row**:
The placeholder row shown while a turn is waiting on the model — no live
message yet, no pending tools — with a spinner in the marker slot and a
stage label: "Loading model…" while the model is not yet loaded, "Reading
context…" during turn prefill. Hands off to the live thinking row at the first
delta with no geometry change; shown only for models whose template starts
generation inside a think block, so the handoff is always seamless. Never
appears between parts mid-stream or during tool execution.
_Avoid_: prefill spinner (the removed floating `ProgressView`); thinking
placeholder (it never says "Thinking"); progress row.

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
The single fold of the `AgentEvent` stream into the agent's committed message log,
total over every event. Run-presentation detail (live stream, phase, pending tool
calls) is the **Chat Session** fold's alone (ADR-0024), and the busy bit belongs to
the run envelope — the reducer owns only the message log. Distinct from the
**Generation Accumulator**, which folds one turn's *token* stream into message
content.
_Avoid_: Generation Accumulator (the token-stream fold — say which "fold"), event
handler / `handleEvent` (this is the fold, not the notify wrapper that hosts it),
dispatcher, state machine, store.

### Chat leaves

The sub-controllers that own their own state but never subscribe to agent events
— the *leaves*, as opposed to the event-subscribing **Chat Session** and its
**Agent Run**.

**Voice Input**:
The agent chat composer's push-to-talk capture→transcribe→emit module: it composes
the shared **Voice Capture Session**, hands transcribed text to the composer rather
than sending, and keeps its errors local instead of on the shared banner. Distinct
from the spine — it touches no `Agent` and no arbiter.
_Avoid_: dictation (the separate global system-wide overlay — say "agent voice
input"), mic controller, voice state machine.

**Composer Draft**:
The agent chat composer's unsent staging area taken as one unit — the typed text
together with the pending images (**Appshot**s, pastes, drops, picker adds) the user
has staged but not yet sent — owned by the `ComposerDraftController` leaf (which also
holds the pending-image previews, model-capability hinting, full-window drops, and the
Quick Look request projection; committed conversation images arrive through an
injected read closure, so it never reaches into `Agent` or the conversation store).
The draft lives *above* any single **Conversation**: starting a new chat, loading
another conversation, or deleting the current one resets the transcript but carries
the whole draft across intact. It is consumed by **send** (and a **Skill Pill** fire),
replaced wholesale by **Edit & resend**, and discarded (thrown away unsent) only by
the explicit `/clear` hard reset — never dropped by mere navigation. Text and images
share one lifetime: any path that clears one clears the other.
_Avoid_: splitting the text and image halves into separate lifetimes; "Image Draft"
(retired — the leaf owns the text too now, so it is the Composer Draft);
per-conversation draft (one shared draft, not one per thread); **Image Input
Availability** (the broader affordance/capability verdict); image cache (the
server-side **Image Digest** path).

**Vision Availability**:
The leaf that owns the lifecycle around the pure **Image Input Availability**
verdict: the cached is-the-selected-model-vision-capable probe (refreshed on
selection/status/setting changes, never per keystroke), the effects of an
availability flip on the **Composer Draft** (mirror the verdict, lower a moot
switch hint, clear pending images the text-only container would silently drop),
and the switch-hint *remedy* ladder — turn the setting on, switch to a downloaded
vision model, or nothing to offer. Catalog reads arrive as injected closures; it
touches no `Agent`.
_Avoid_: **Image Input Availability** (the pure verdict this leaf wraps), vision
switch (say "remedy"), capability prober.

**Image Gesture**:
An inbound paste or drop into the agent chat whose payload carries image content —
one concept regardless of delivery (⌘V, drag onto the composer, drag onto the
window) or source form (copied file, raw bytes, decoded image, file promise).
Keyed on *content, not outcome*: once the payload holds an image, the gesture
resolves as an image action — attaching what it can and voicing failures through
composer feedback — and never falls back to inserting the payload's textual
sidecar (file name, path, or source URL). A payload with no image content is not
an Image Gesture and takes the ordinary text path.
_Avoid_: image paste (one delivery of several), mixed text+image paste (retired —
the sidecar text of an Image Gesture is never inserted), drop handling (delivery
mechanism, not the concept).

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

### Agent skills

**Skill**:
A named markdown instruction unit (frontmatter name/description plus body) the
agent loads on demand — user-invocable as a slash command or **Skill Pill**,
model-invocable via the prompt's skills listing. Discovered from user directories
and packages.
_Avoid_: command (the palette concept), prompt template, tool (a callable
capability, not an instruction), persona/mode.

**Skill Pill**:
The tappable capsule inside the **Skill Cluster** that runs one **Skill** instantly
on tap — the composer's current text and pending images ride along as the skill's
arguments and attachments, and a bare tap with an empty composer still fires.
Presentation only: a surface over skills, never a second invocation mechanism.
_Avoid_: quick action, suggestion chip, shortcut button, mode/toggle (a pill arms
nothing).

**Skill Cluster**:
The floating glass surface for **Skill Pill**s: a collapsed bubble above the
composer's trailing corner that morphs open into the fanned pills — hover opens,
click pins, a composer draft gaining content auto-opens it, firing/Esc/the draft
emptying collapse it, and a manual close is final for the current draft. Dimmed
and inert while a run is generating; its visibility is the "show skill pills"
Setting. The pills fan leftward from the bubble, most-used nearest, wrapping
upward when out of width.
_Avoid_: FAB / floating action button, quick actions, toolbar, menu (nothing arms
or navigates), palette (the slash popup concept).

**Skill Invocation Row**:
The chat's compact rendering of a fired **Skill** — the skill name
plus the user's argument text and attachments, expandable to the full injected
skill block. One rendering for every invocation surface (pill or slash command).
_Avoid_: raw `<skill>` text as the user bubble, skill message (it is a rendering,
not a message kind).

**Skill Usage Ranking**:
The order of **Skill Pill**s — most-used nearest the **Skill Cluster**'s collapsed
bubble: user-initiated invocations (pill tap or slash command — never
model-initiated) accumulate a per-skill count; zero-count skills follow the curated
default order. Recomputed at conversation start and held stable within a
conversation.
_Avoid_: frecency (not the V1 mechanism), MRU/recently-used (counts, not recency),
live re-sort (explicitly rejected — the order never shifts mid-conversation).

### Living memory read paths

The three distinct ways memory reaches the model (ADR-0035); they are not
synonyms, and the 2026-07-12 recall defect (#332) hid partly in the gap.

**Memory Injection**:
The automatic read path — the lifecycle-scored working set (core tier, top-ranked
beliefs, relevant episodes) put in front of the model on a turn without being
asked, riding a user message. Deliberately excludes the cold tier apart from the
ε-exploration draw.
_Avoid_: auto-recall, context stuffing; **Recall** (the deliberate tool, not this).

**Memory Search**:
The deliberate relevance-only lookup over *every* belief — retired and superseded
included, tiers and lifecycle ignored — that marks what it surfaces as seen.
The engine verb beneath **Recall**; sleep's internal reconcile lookups use it
without marking.
_Avoid_: retrieval (unqualified — say injection or search), semantic search (it
is hybrid dense ⊕ keyword).

**Recall**:
The agent's tool over memory: **Memory Search** across beliefs *plus* the raw
episodic record, so a fact told the same morning — an episode not yet distilled
into a belief — is still findable. Superseded beliefs come back plainly marked;
episodes come back dated and quoted.
_Avoid_: memory_search (the ADR's design-phase name; the tool shipped as
`recall`), beliefs-only search (the pre-#332 blind spot).

**Agent Voice**:
The deictic convention every stored belief obeys: memories are the assistant's
own, so every pronoun must resolve with no conversational context — "he" is
always the owner, "I" is always the assistant ("He gave me the nickname
Pelican"). A directive stays anchored on its ordainer: "He wants me to answer
briefly when he is debugging". Holds at both write doors (`remember` and sleep
consolidation); the owner's words as spoken are episodes, quoted verbatim,
never beliefs.
_Avoid_: first person (unqualified — ADR-0035's "first-person layer" means the
*assistant's* first person, and the unqualified phrase inverted the referent
and caused #333), user voice (the owner's "I" stored as if it could survive
recall).

**One Door Per Testimony**:
The episodic-capture rule: one utterance becomes exactly one episode, entering
through the door that knows the most about it. Words sent to the agent enter
through chat capture, which attaches the reply; dictation capture takes only
speech aimed at *other* apps and skips when the frontmost app is Tesseract
itself.
_Avoid_: episode dedup(lication) — the rule turns the twin away at the door
rather than deleting it later; episode merging.

**Owner's Veto**:
The revision path that answers "that's wrong": the memory flips to *contested*
— a status change, never an edit or a delete — and the next sleep re-reads its
source episodes with the owner's rejection note beside them, minting a
corrected successor (which inherits none of the rejected belief's strength) or
retiring it cold. Two doors relay the veto: the Memory window, and the agent's
`contest` tool, which names a memory by the short handle on its `recall` line
and records what he said when he rejected it.
_Avoid_: forget/delete (deletion is the owner's hand alone, in the window),
belief editing (the agent never rewrites a memory's text), negation belief (the
pre-#333 workaround — `remember`ing the correction beside the wrong memory left
both live, contradicting each other).

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

**Proofread Pass**:
The optional LLM polish stage between transcription and commit (ADR-0034): a second,
small co-resident MLX model — its own, never the agent's — that fixes punctuation,
capitalization, and misheard words, or rejects an unintelligible take outright.
Strictly fail-open: disabled, model not downloaded, GPU lease held (skip-when-busy —
it *reads* the lease, never queues on it), budget overrun, or any error all commit
the raw text unchanged. Runs inside the **Voice Capture Session**, so dictation and
**Voice Input** both gain it; its word-level edits ride the commit for overlay
narration, and a rejected take's raw text stays available for "insert raw anyway".
_Avoid_: post-processing (the regex cleanup that always runs, pass or no pass),
autocorrect, grammar check, second agent (it is a fixed-prompt pass, not an agent).

**Correction Pair**:
One dictation take's full text lineage — raw ASR, regex-cleaned, **Proofread
Pass** output + verdict, committed text, the owner's correction — plus capture
conditions and a Capture Dump audio reference; the local, bounded, exportable
training-pair collection the flywheel feeds from day one. Every take is a
*candidate*; an owner signal (a correction edit in the history, or a one-click
wrong-flag from the overlay's lingering beat) makes it *gold* — evicted last,
its audio exempt from the dump's ring eviction. Full editing lives in the
history window; the overlay stays keyboard-free.
_Avoid_: training data (unqualified — pairs are candidates until gold),
feedback log, transcription history (the sibling store it links to by id),
fine-tune corpus (the export's *consumer*, out of scope — see the map).

### Tracking (Companion)

The measurement grain beside memory's two (map #301, ticket #308): episodes are
*testimony*, observations are *measurement*, beliefs are *conclusion*. Flow into
memory is one-way, in sleep — pattern distillation reads the observation stream
and mints pattern beliefs.

**Observation**:
One dated, typed fact about the owner's day — an elicited sample (sleep, mood,
energy, movement), a step event, a habit check-off, or a sensed span (presence,
app session, power). Append-only and kept forever: facts don't decay; a
correction is a newer row (recency wins at read), never an edit. Elicited
observations carry the episode that produced them, so the verbatim words stay
recoverable behind the structured fact.
_Avoid_: sample (one species — the elicited kind), metric/log line,
observation-as-belief (facts are never superseded and have no lifecycle).

**Observation Source**:
Which of the three doors wrote an observation: *sensed* (sensor code, no LLM
between sensor and disk), *elicited* (conversation, through a typed tool),
*imported* (a bridge from an external store). One door per fact: each
observation kind has exactly one writing door — the tracking-side analogue of
**One Door Per Testimony**.
_Avoid_: provenance (that's the episode link), mixed-source kinds.

**Contract Chain**:
The day contract's shape: an ordered chain of hard steps with at most one
*active*; finishing a step arms the next immediately, the same day. Generalizes
the anchor day's "ONE hard step" — the preserved invariant is one step *at a
time*, not one per day; the push (the midday pulse) always aims at the single
active step.
_Avoid_: task list (the chain is chosen and pushed, never dumped), parallel
steps, ONE hard step (the superseded per-day cap — see #302's amendment).

**Keystone**:
Step one of the **Contract Chain**: the single step that makes the day a win.
Streaks count keystones alone; nothing past the keystone can fail a day.
_Avoid_: hard step (any chain step is hard; only one is the keystone),
most-important-task.

**Chain Depth**:
How far past the **Keystone** a day's chain actually went — surplus recorded,
never failure ("kept, depth 2/3"). The signal that eventually tells the owner
his realistic daily ambition.
_Avoid_: completion rate (depth is not a percentage of plan), failure count.

**Stream**:
A named area of the owner's life that work flows in — Tesseract, the employer's
work, health. Tags work items and observations so cross-stream balance becomes
visible to pattern distillation; streams are conversational names, never a
configuration surface.
_Avoid_: project (a stream outlives any project), area/category (say stream),
audio/transport streams (Generation Stream Loop and friends are unrelated).

**Work Item**:
One backlog entry the Companion may draw a chain step from: a title, its
**Stream**, and a cadence — one-shot, or a recurring habit that re-arms daily.
The evening close-out checks due habits; check-offs land as **Observation**s.
The successor to the retired `tasks.md`.
_Avoid_: task (the retired file's word; too broad here), reminder (EventKit's,
not ours), habit tracker (a habit is a cadence value, not a product surface).

**Read-Through Source**:
An external store the Companion consults live at compose time and never
mirrors — the calendar is queried when the evening's tomorrow-preview or the
morning placement needs it. Observations record only what has no other home.
_Avoid_: import (that's a copy, with `imported` as its **Observation Source**),
sync (nothing is mirrored, so nothing can drift).

### Proactive loop (Companion)

The entity/harness split (map #301, ticket #307, ADR-0040): the model — Jarvis,
the *entity* — decides everything with judgment in it; the *harness* (code)
contributes turns, continuity, and the record, never judgment.

**Wake**:
One persisted row granting the entity a future turn — a promise, a rhythm beat,
a follow-up, or a re-summons, all one table. Booked by the entity through a
typed tool; state transitions written only by app code. The loop's one
correctness invariant: a wake is consumed only by a completed turn.
_Avoid_: trigger/timer (implementation words for what fires it), reminder (a
wake wakes Jarvis, not the owner), notification (one possible *outcome* of the
turn a wake grants).

**Turn**:
One full agent run granted to the entity by a **Wake**, a transition (day
start, Mac-wake, launch catch-up), or ambient eligibility. Every turn persists
as an origin-tagged conversation in the one chat list — full observability.
Silence is a decision a turn records, never a branch code took.
_Avoid_: check/tick (the evaluator's clock, which decides nothing), beat (the
anchor rhythm's word for the *content* of some turns), heartbeat (the retired
skeleton's fixed-time pings).

**Ambient Turn**:
An unoccasioned **Turn** — time to think, research, notice, book — granted when
the eligibility gate passes (AC power, `.llm` slot free, owner not using the
agent, spacing). The waking analogue of a sleep pass, and the seed of the North
Star's continuous loop.
_Avoid_: background job (it's cognition, not maintenance), idle task, cron.

**Situation Briefing**:
The code-gathered context handed to the entity at the start of every turn:
time, presence span, frontmost app, calendar (read-through), contract state,
its own due and upcoming **Wake**s, recency of last interaction. Gathering is
mechanical; interpreting it is the turn's job.
_Avoid_: prompt (it's one input to the turn, not the instructions), snapshot
(the flight recorder's word for the verbatim copy a trace keeps).

**Standing Instructions**:
The entity's self-authored policy document — versioned, injected into the
system prompt beside memory, edited by the entity through a typed tool,
reviewed in sleep passes, always owner-readable and owner-editable. Escalation
ladders, interruption ethics, quiet hours, and rhythm defaults live here, as
seeds the entity rewrites with wear — never as code.
_Avoid_: system prompt (the instructions ride inside it), settings/config (not
a UI surface; a document the entity owns), rules (they're his practice, not his
cage).

### Voice session (Companion)

**Voice Session**:
Voice as a mode of the one conversation, never a separate surface: an
auto-listen loop — listen, capture, transcribe, send; the reply spoken with the
microphone open underneath — whose spoken and typed turns share one persisted
message stream. Entered from the overlay or the chat toggle; left by dismissal,
staging to the composer, or mutual silence.
_Avoid_: voice mode (UI shorthand), speech session (the TTS reading concept),
voice chat.

**Barge-In**:
The owner interrupting a speaking reply — by voice energy (a **Soft Barge**
first) or a click (immediate pause; a click is deliberate). A hard barge
*pauses* playback while the take resolves: a take that transcribed to
anything stops the reply for good and becomes the turn; an empty take — a
false barge — resumes it where it paused. Purely acoustic: no word gates.
_Avoid_: interruption (unqualified), stop-on-barge (the discarded kill-first
semantics), Substance Gate / Session Directive (the removed word gates).

**Soft Barge**:
The energy barge's first stage: an onset ducks the reply instantly and opens
a short confirm window — only sustained voicing inside it commits the hard
pause; without it the volume fades back. A false fire costs a ~1 s murmur,
never a dead pause.
_Avoid_: two-phase commit, barge preview, duck-on-barge (the duck is the
acknowledgment, not the barge).

**Echo Floor**:
The tracked level of the reply's own residual at the open microphone: while
playback is audibly emitting, the energy barge threshold rides floor + margin,
never the static level alone. The floor chases the mic fast but may never
believe more than the playback envelope minus the calibrated echo-path loss —
residual can't out-shout the reply; the owner can. Calibrated by the
voice-hold lab's traces; pinned by the replay tests' zero-false-onset lock.
_Avoid_: noise gate (room noise is not the subject), VAD, adaptive threshold
(unqualified).

**Self-Echo**:
The voice session's signature failure: the assistant's own TTS re-captured by
the microphone and treated as the owner's speech — a false barge-in, or worse,
a committed turn feeding the conversation its own words back. The session's
acoustic and gating defenses exist to make this unreachable.
_Avoid_: feedback loop (the mechanism, not the name), echo (the raw acoustic
signal cancellation removes).

**Dual-Path Playback**:
Where TTS renders: a **Voice Session**'s replies play through the capture
engine under its voice hold — the reply escapes the recording duck and the
engine stays running between turns, so echo cancellation and gain control
stay converged; every other TTS surface keeps its dedicated playback path and
unprocessed fidelity. The hosted reply renders at a fixed master gain: the VP
unit's residual-echo suppressor clamps the owner's microphone harder the
louder its own voice stream plays (field 2026-07-18), so reply loudness is
traded for double-talk headroom. On macOS the canceller's reference is the
output *device* signal, so both paths are echo-cancelled.
_Avoid_: single playback engine, VPIO routing (the implementation).

**Voice Hold**:
The capture engine's state for a **Voice Session**'s lifetime: the engine
keeps running between captures — capture start/stop degrade to a capture-gate
flip, never tap install/remove or render rewiring on a running engine (the
2026-07-17 crash class) — and hosts the session's persistent TTS player node
upstream of its mixer. Wired asynchronously at session enter (~2.3 s of
stopped-engine work: tap once, render side verified, node attached, start);
a reply that beats the wiring falls back to the dedicated playback path.
_Avoid_: warm engine (the prewarm concept), engine reuse, always-on mic.

### Text injection

**Clipboard Loan**:
How dictated text reaches the frontmost app: the system pasteboard is borrowed as
the transport for a synthetic Cmd+V and returned — the pre-dictation contents
restored, or cleared when there was nothing to save (empty before, or over the
snapshot cap) — so a transcript never lingers for a later Cmd+V to re-paste. One
loan is out at a time: a new dictation waits through the prior app-read window and
return before taking the pasteboard, and the return outlives a cancelled dictation.
Transient return-write failures retry; a persistently refused snapshot is retained
for recovery before the next clipboard use. Two deliberate exceptions: the return
only lands if the pasteboard generation is still ours (a mid-window copy wins), and
a pasteboard that could not be read aborts before mutation — never destroy what
could not be seen. Restore mode off is not a loan at all: dictate-to-clipboard
keeps the transcript deliberately.
_Avoid_: clipboard restore (half the contract — the return also clears),
clipboard backup, paste injection (the Cmd+V is one step of the loan).

### Hotkey handling

**Hotkey Matcher**:
The pure fire-or-not decision for global hotkeys — normalized event + bindings +
pressed-set in, fires plus suppression verdict out — implemented once and fed by
two thin event adapters (the CGEvent tap and the NSEvent fallback monitor), which
differ only in delivery timing. Follows the DoubleCommandDetector template.
_Avoid_: HotkeyManager (the adapter host above it); event tap (one adapter);
duplicating the decision per event source (the pre-extraction shape).

### Microphone capture

**Voice Processing**:
Apple's capture-time processing bundle — echo cancellation, automatic gain control,
noise suppression — applied by the OS before the app ever sees samples; in Tesseract
the standard mode for all microphone capture (dictation, **Voice Input**, and the
settings level meter alike), with raw capture only as the fallback when the platform
refuses it. Capture-time: it changes what gets recorded, so its effect can never be
replayed offline against the same utterance — unlike post-capture DSP.
_Avoid_: Voice Isolation (the user-only Control Center mic mode — not programmatically
settable), noise cancellation, VPIO / `setVoiceProcessingEnabled` (the implementation).

**System Audio Duck**:
What happens to all *other* system audio while **Voice Processing** is armed. Two
treatments: *idle* — full volume, by ear indistinguishable from the app not running —
and *recording* — the standard dip exactly while a dictation capture runs. Armed
otherwise means audibly ducked; the idle treatment is what makes staying armed free
(ADR-0025).
_Avoid_: ducking level (one lever inside a treatment), mute/suppression (it attenuates,
never silences).

**Capture Engine Lifecycle**:
The pure policy deciding the capture engine's lifecycle moves — rebuild-vs-reuse
on press, prewarm arming, external-config-change detection, the empty-capture
verdict, disarm-after-grace, idle rebuild and arm retry — with the AVFoundation
engine as the performer. The ADR-0025 policy/performer split, applied to the
engine itself; the arm mode (always-armed vs disarm-after-grace) is one input.
_Avoid_: engine defaults (capture mechanics stay on the engine); VPIO lifecycle
(the arm mode is an input, not the policy); duck policy (the sibling policy for
system audio).

**Capture Dump**:
The on-disk ring buffer of recent dictation capture audio — what the microphone tap
delivered (post–**Voice Processing** when enabled, pre-resample) — tagged with its
capture conditions and kept for diagnosing bad transcriptions; bounded by count/size,
oldest evicted first.
_Avoid_: recording archive (it is bounded and diagnostic, not an archive), audio log.

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

### Batch inference

**Batch Engine**:
The single generation engine that holds the GPU lease whenever any **Lane** is
live, driving every lane's prefill and decode on the model-affine actor;
completions submit to it rather than acquiring the lease themselves. Distinct
from the **GPU Lease Queue** (the mutex it holds) and the **Inference Arbiter**
(model ownership it sits under, as one long-running lease consumer).
_Avoid_: scheduler (one policy inside it, not the module), continuous batching
(the technique family), server engine, engine (unqualified).

**Lane**:
One admitted request's live generation inside the **Batch Engine** — its
execution identity from admission to drain. One request is one lane for its
whole completion; admitted lanes are FIFO-fair, never descheduled for a
sibling.
_Avoid_: slot (the arbiter's `.llm`/`.tts` co-residency unit, and oMLX's static
per-slot KV reservation — both different things), worker, stream (the wire
concept), request (the HTTP envelope; a lane is its execution).

**Lane Admission**:
The gate that turns the waiting queue's head into a **Lane** — headroom-priced
by the per-lane reserve and hard-capped; the queue it draws from is ordered by
longest radix prefix match, aged to strict FIFO so no request starves. Distinct
from **Snapshot Admission** (the cache write side — always say which).
_Avoid_: admission (unqualified — collides with **Snapshot Admission**),
scheduling (the step-loop share, not the gate), request start.

**Boundary Yield**:
The **Batch Engine**'s release of the GPU lease at a decode-step or
prefill-chunk boundary to a waiting slot-preserving consumer (TTS), lanes
pausing as plain data until the engine re-acquires. Never for a consumer that
would change the loaded model — that is an **Admission Freeze**.
_Avoid_: preemption (nothing is descheduled or lost), GPU handoff, lease steal.

**Admission Freeze**:
The drain mode where **Lane Admission** stops so the pool can empty for a
consumer that needs the pool gone — a model switch, or an image-bearing request
running solo. The freeze is the cause; the drain is the emptying that follows.
_Avoid_: drain (the effect), pool pause (a **Boundary Yield** pauses lanes; a
freeze retires them by attrition), admission stop.

**KV Page**:
The refcounted fixed-size block of KV cache the RAM tier stores and **Lane**s
reference — a shared prefix is held by reference, restore is a refcount bump
rather than a copy, and a page with a live reference is structurally
unevictable.
_Avoid_: block (vLLM vocabulary; generic), snapshot body (the deep-copied
predecessor it replaces), slot reservation (the oMLX static shape, explicitly
not built).

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

**Prepared Checkpoint**:
The once-converted MLX-native form of a PARO checkpoint, stored beside the
original so later loads skip the AutoAWQ conversion; rotation parameters remain
runtime state loaded verbatim — nothing semantic is baked into the artifact.
Stale or unreadable artifacts self-heal by re-conversion.
_Avoid_: prerotated cache (rotations are not pre-applied), weights cache / cache
(collides with the prefix cache), converted weights (holds the stacked MoE
layout too, not just per-tensor conversion).

### Cache memory budget

**Pressure-Reactive Budget**:
The RAM-tier byte budget expressed as a band rather than a constant — a ceiling
derived from measured machine headroom and a current value that OS memory-pressure
events push down and hysteresis regrows, never below the **Budget Floor**. The cache
is greedy when RAM is idle, polite when it is contested.
_Avoid_: static budget, memoryBudgetBytes-as-constant, cache size limit. (This is the
RAM tier; the SSD tier's budget is separate — dynamic by default, user-cappable.)

**Budget Floor**:
The content-defined lower bound of the **Pressure-Reactive Budget**: the minimal
survival set — the in-flight requests' restore paths plus the single
most-recently-extended leaf — kept resident at critical pressure and honored by
*every* eviction drain, admission included. A last-resort floor, not the protection
mechanism (defending the main-agent leaf against subagent churn is the eviction
score's job).
_Avoid_: minimum cache size, reserved bytes, fixed floor, per-partition floor,
workload heuristics in the floor; `.system` chains as floor members (they are
SSD-protected, not RAM-pinned — ADR-0019).

**Snapshot Demotion**:
Moving a snapshot's body out of RAM while keeping it recoverable — backing it to SSD
first, then dropping the RAM body — so the next hit pays a cheap hydration instead of
a re-prefill. The required response to any RAM-tier shrink under **Recoverable
Eviction**; a terminal drop in its place is a defect, not a fallback.
_Avoid_: spill, flush, evict-to-SSD; eviction (terminal — a demotion is recoverable,
and supersession *preserve* differs again in keeping an ancestor's SSD backing).

**Leaf Home Guarantee**:
The cross-tier invariant that the newest end-of-turn leaf always has a home on some
tier: RAM when the budget holds it, otherwise a mandatory SSD write that no
incidental cap, gate, or ordering may reject — and whose enqueue precedes any
deletion of the backing it supersedes.
_Avoid_: leaf pinning (it may leave RAM freely), floor membership (the floor is
RAM-only; the guarantee spans tiers), best-effort persistence.

**Recoverable Eviction**:
The rule that a RAM body may be dropped only if its bytes are SSD-recoverable —
demotion succeeds or a backing already exists. A terminal drop is a bug class, legal
only for explicit invalidation (model change, user clear), disk-full, or I/O error;
never a silent policy outcome.
_Avoid_: demote-if-possible (the old best-effort reading), terminal drop as an
eviction strategy, survival-gate veto of a demotion.

**Restore Pin**:
An in-flight request's claim on the node it restored from: pinned into the **Budget
Floor** at resolve, released at the completion drive's all-exit-paths tail, so no
drain may evict the body a running generation depends on. Weak by reference — a pin
protects, it does not own.
_Avoid_: node lock, refcount, lease; leaf pinning (pins hold restore *paths*, not the
newest leaf — that floor member is recency-defined).

**Guarantee-Class Write**:
The mandatory SSD write of the newest end-of-turn leaf — the write the **Leaf Home
Guarantee** promises. Exempt from the pending-queue size cap and never a back-pressure
victim; its remaining rejection paths are hard errors surfaced to telemetry, never
silent.
_Avoid_: mandatory flag (wire detail, not the concept), priority write (no queue-jump
— FIFO order holds), best-effort write.

**Condemned Resident**:
A superseded ancestor's SSD backing that a pending replacement write has marked as the
first victim of its own admission cut — evicted before any innocent resident, but only
once the cut actually needs the room, and never for a write that cannot fit at all.
_Avoid_: doomed/stale backing (it is still live and hittable until evicted),
enqueue-before-delete (the ordering invariant; condemnation is the budget-efficiency
half).

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
hosts the live **Overlay Variant**'s view — a dumb, fixed-frame canvas that is
created once, stays permanently ordered front, and never resizes, fades, or
reacts to dictation state itself: all visibility and motion belong to the
hosted SwiftUI content. Takes its **Overlay Placement** at construction and
swaps hosted content on demand; the interactive TTS notch is a separate panel,
not an Overlay Panel.
_Avoid_: overlay controller / manager, HUD window, generic NSPanel wrapper,
show/hide or animated-frame panel APIs (retired — SwiftUI owns all motion),
config-flag panel, the TTS notch panel (a separate, interactive surface), the
full-screen border overlay (retired — a legacy MVP exploration).

**Overlay Placement**:
Where an **Overlay Panel**'s fixed canvas sits for a given **Screen Geometry**,
expressed as a state-free pure value an **Overlay Variant** brings along with
its hosted view. One preset exists — pill.
_Avoid_: layout strategy, frame provider, per-state frames or resize-animation
flags (retired — the canvas is fixed), overlay style (the retired
pill-vs-border user **Setting**).

**Overlay Feed**:
The one variant-agnostic surface of dictation signals every **Overlay Variant**
renders from: typed lifecycle phases, typed errors, terminal outcome beats
carrying the committed text, and the audio meter (level + spectrum). The
dictation coordinator is its sole phase/beat writer; the capture engine's meter
stream drives its meter. Variants consume the feed and nothing else, so the
dictation pipeline never learns which variant is live.
_Avoid_: overlay state / OverlayState (retired push-model object), view model,
pre-flattened error strings, per-variant state surfaces.

**Overlay Variant**:
One live dictation-overlay design exploration (map #283): a hosted view over
the shared **Overlay Feed** plus the **Overlay Placement** of the canvas it
draws in, selected at runtime by a **Setting** and switchable live. Exploration
scaffolding — the registry and its Setting are deleted when the redesign
program prunes to one winner.
_Avoid_: theme, skin, style, prototype window (variants live in the real
panel), a permanent plugin surface (it is scaffolding).

**Screen Geometry**:
The plain screen rectangles — full frame and visible frame — that an **Overlay
Placement** consumes, decoupled from any live `NSScreen` so the frame math stays
unit-testable.
_Avoid_: an `NSScreen` (deliberately not passed to placements), a bare single rect
(placements need both frames).

### Onboarding tour

**Onboarding Tour**:
The first-launch welcome experience: a chaptered, user-paced cinematic tour of the
app's features that also carries first-run setup — the model download runs in the
background from its first screen and permission requests live inside the chapter
that motivates them. Optional and skippable, never re-shown after a skip, and
relaunchable from Settings, where it replays the same state-aware flow rather than
a separate "tour mode".
_Avoid_: setup wizard, tutorial, walkthrough, splash screen, Setup One-liner (a
server/Integration concept, unrelated).

**Chapter**:
One user-paced beat of the **Onboarding Tour**, owning a single feature story plus
whatever setup belongs to that story. Chapters are navigable in both directions and
never block on downloads or denied permissions.
_Avoid_: step (the retired flow's term), page, slide, screen.

**Try-it**:
A live, real-functionality demo slot inside a **Chapter** that activates when its
preconditions (model on disk, permission granted) are met, and otherwise shows the
chapter's scripted animation with a soft note — the tour proving the app rather
than describing it.
_Avoid_: interactive tutorial, demo video, sandbox.

**Welcome Window**:
The dedicated window the **Onboarding Tour** runs in — the only window shown on a
first launch, an ordinary additional window when relaunched from Settings. Closing
it mid-tour is a permanent skip, equivalent to finishing, never a nag deferred to
next launch.
_Avoid_: onboarding sheet (the retired presentation), modal, popup.

**Handoff**:
The finish transition out of the tour: the main window appears first, then the
Welcome Window dissolves — there is never a moment with zero windows on screen, and
the landing surface must state download progress honestly if setup is still
running.
_Avoid_: dismissal, close animation.

### App composition

**App Bindings**:
The module owning the app's launch sequence and every long-lived runtime subscription
that carries a rule (model auto-load and hot-swap, lazy-reload guards, server and
overlay-style reactions, hotkey rebinding, the single dictation-state fan-out) — the
launch-time mirror of the teardown-owning termination coordinator. Distinct from the
composition root, which stays pure wiring with no behaviour.
_Avoid_: app glue (pre-carve working name), setup() behaviour, launch coordinator,
app services, a SwiftUI `Binding` (view data flow, unrelated).

### Release and distribution

**Release PR**:
The rolling pull request that automation keeps open against `main`, holding the
next semantic version and its accumulated changelog. Merging it *is* the release
decision — the tag, the GitHub Release, and the signed build all follow
mechanically from that one merge.
_Avoid_: version-bump PR, release branch (no such branch exists), draft release.

**Release Pipeline**:
The automated path from a merged **Release PR** to a downloadable, notarized
disk image attached to the GitHub Release — gated on the released commit's CI
being green, with no human step inside it.
_Avoid_: deploy (nothing is deployed to a server), publish flow, the CI workflow
that builds pull requests (a gate the pipeline consumes, not the pipeline).
