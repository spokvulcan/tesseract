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

**Prefix-View Checkpoint**:
An interior snapshot owning only its whole-state layers, their metadata and its
token offset; its attention rows come from a resident descendant leaf. It owns
no attention bytes and is never an eviction victim. A transient boundary is a
request-local Prefix-View Checkpoint: it is never admitted as a tree body and
resolves its Backing Leaf only when consumed after the turn. The live leaf a
think-stripping turn checks in to back it is released once the canonical leaf
is admitted, so a boundary turn leaves one resident leaf (ADR-0068 amendment).
_Avoid_: shared KV, copy-on-write checkpoint, partial leaf.

**Backing Leaf**:
The resident, unleased, full-body descendant chosen at **Snapshot Resolution**
for one **Prefix-View Checkpoint** restore, including a **Warm Body**. It is never
recorded on the view node. A hit served through the view credits the Backing
Leaf's recency and hit count as a direct hit would, and while it alone keeps
the view alive its terminal **Recovery Cost** spans the view's prefix as well
(ADR-0068 amendment).
_Avoid_: parent body, permanent backer, shared owner.

**View Materialization**:
Producing live cache objects from a **Prefix-View Checkpoint** by copying its
whole-state layers and the **Backing Leaf**'s leading attention rows, with warm
attention dequantized after slicing. The result
belongs to the request or an SSD payload, never to a new tree body.
_Avoid_: view promotion, alias restore, materialization on backer departure.

**Hot Leaf**:
A movable fp16 leaf body eligible for **Leaf Handoff**: a leased leaf or the
most recently checked-in leaf of a path in the **Hot Leaf Set**. It is exempt from compression.
_Avoid_: warm leaf, pinned leaf.

**Hot Leaf Set**:
The bounded set of paths whose most recently checked-in leaves stay exempt
from compression, ordered by check-in rather than lookup recency. Leased
leaves are exempt independently of this set and its path limit.
_Avoid_: Budget Floor, hottest snapshots.

**Warm Body**:
A RAM-tier body with quantized attention layers, restored by copy through
dequantization and never checked out by move.
_Avoid_: quantized live cache, compressed leaf lease.

**Stored Form**:
The per-partition dtype policy for bodies at rest, warm and on SSD: fp16 or
quantized at a given bit width and group size. It is part of cache partition identity.
A partition with a **KV Scheme** stores the scheme's TurboQuant layers.
_Avoid_: live KV dtype, per-segment compression choice.

**KV Scheme**:
How a request holds its full-attention KV once its prompt is prefilled: full
precision, or TurboQuant (`turbo8v4`: 8-bit affine keys, `turbo0v4`: bf16 keys,
both with 4-bit values). The KV Cache Compression setting picks it for a model
that supports it (Qwen3.8-27B). It is a request fact and part of cache partition
identity (ADR-0083): the prompt prefills unquantized, the cache converts after
prefill (inside the iterator on a DFlash2 turn and on an **Unkeyed
Completion**, where the iterator runs the prefill), and the turn's leaf
stores compressed layers that only a request of the same scheme restores.
_Avoid_: kvBits (the unrelated affine quantization the product never sets), KV
quantization (ambiguous between the two).

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

**Deferred Payload Extraction**:
Building a snapshot's SSD payload without copying its bytes: the extraction edge
fixes the byte total (slicing and evaluating a **Leaf Extension Admission**'s
suffix on the Metal-affine caller), and the SSD writer borrows no-copy host views
of evaluated contiguous arrays on its own task right before the file write. Each
view retains its array until its layer's bytes have been written, then releases
it. Header and blobs go directly to the file in bounded chunks, without a full
output buffer. **Snapshot Admission**, eviction and **Snapshot Demotion** on the
MainActor and a **Leaf Admission** on the inference thread never pay a full-KV
host copy. A full payload's arrays are the body's own and exclude checkout until
the write releases them, except for a Prefix-View Checkpoint's full-format
payload. View and extension payloads retain no body array: their attention
prefix or suffix and their whole-state layers are independent evaluated copies
the extraction edge *detaches* there.
_Avoid_: lazy payload, async extraction, background asData, payload streaming.

**Snapshot Payload**:
The SSD tier's byte form of one snapshot: the whole body, a **Leaf Extension
Admission**'s suffix past its base, or a **Prefix-View Checkpoint**'s
materialized prefix, each array under a stable dtype name that is part of the
on-disk contract. **Deferred Payload Extraction** builds it, and whether an
extension is worth writing instead of the whole body is its rule too.
_Avoid_: SSD payload (unqualified); serialized snapshot, wire format (the
container framing around the payload is separate).

**Layer Kind**:
The one fact a snapshot layer carries about how its arrays relate to the token
axis — *sliceable attention* or *whole-state* — derived once, from the vendor
cache class and the array shapes, when the layer state is built (captured,
moved, deserialized, or hydrated from a **Segment Chain**), and read by every
consumer that slices, composes, trims or rebuilds. Sliceable attention
(`KVCacheSimple`, `QuantizedKVCache`) has arrays that cover the snapshot's
offset along the token axis, so a token-range slice is exact; whole-state is
recurrent, rotating and chunked state, plus any attention layer whose arrays
fail that shape guard, which then rides whole. The extraction edge slices the
former into a **Leaf Extension Admission** and detaches the latter whole;
check-out eligibility hands off only trimmable sliceable attention beside
recurrent whole-state; **Leaf Rewind** trims the former and rebuilds the latter.
**Capacity Compaction** (#534) rebuilds sliceable attention at the offset's
rows plus one growth step, on **Leaf Rewind** and at check-in, when the rows a
generation grew into but no longer addresses exceed the smaller of a quarter
of the body and 64 MB (the 2026-09-21 profile: a cancelled long generation
retains 50–117 MB, an ordinary check-in 5–17 MB); whole-state and quantized
layers are untouched. _Avoid_: trim (trim moves the offset, the arrays stay).
_Avoid_: demoted (a mis-shaped attention layer *is* whole-state — **Snapshot
Demotion** is the RAM-to-SSD move); layer type; class-name matching (a consumer
reads the kind, never re-derives it); sliceable class (the class alone is not
the kind — the shape guard is part of it).

**Snapshot Admission Path**:
The validated token path carried by a **Snapshot Admission** — the proof that a
snapshot may be stored at a given token offset, checked before any cache mutation.
_Avoid_: promptTokens, storedTokens (both name unrelated token fields), offset
guard.

**Leaf Admission**:
The one way a leaf enters the prefix cache, whichever producer made it: a
finished turn, a boundary re-prefill, a **Speculative Canonical Prefill**, or a
salvaged cancelled prefill. The producer brings the cache to its final state and
says whose it is. The admission decides move or copy, checks a leased leaf in
through its **Cache Claim** before anything is extracted, builds the **Snapshot
Payload** and the **Snapshot Admission**, and classifies what the admission
evicted and superseded. Its extension base is settled before the **Model
Session** is entered; every other step runs inside it.
_Avoid_: leaf write ("write" is the SSD tier's word, as in **Guarantee-Class
Write**); Leaf Store (the phase that decides whether a turn stores a leaf, and
which); **Leaf Admission Builder** (the boundary route's plan, which a Leaf
Admission carries out); admission tail (the retired hand-sequenced shape).

**Snapshot Resolution**:
The read side of the prefix cache: resolve a token path to the best usable
snapshot, returning a hit or a miss. The read-side counterpart to **Snapshot
Admission**; distinct from `restoreCache`, the later model-affine step that applies
a resolved snapshot into a live cache (resolution picks *which*, restore
*applies* it). Owned by the **Prefix Cache Manager** as its single read-side
entry, never a free-standing module reaching back across the manager's mutation
seam. The lookup-then-hydrate *choosing* — the Hydration Gate outcome, the
promote-on-success vs typed-cleanup-on-failure fork, the gate-fallback shapes —
now lives in a pure decision ladder (`SnapshotResolutionLadder`) the manager
consults; ownership is unmoved, the manager still performs every effect (the
awaits, the tree/ledger mutations, the hydration call, pin placement, telemetry)
and the ladder holds no references at all.
_Avoid_: a standalone "hydrator" module (the retired shape that reached back into
the manager — resolution is manager-owned); a ladder holding references to the
manager/tree/ledger/hydrating handle (the retired hydrator's shape again — the
ladder takes plain facts, returns decision values); **Snapshot Hydrating** (a
different concept: the off-main handle, not the choose step); restore/restoreCache
(the apply step, not the choose step).

**Snapshot Hydrating**:
The narrow off-main handle that **Snapshot Resolution** depends on to materialize
a body-absent snapshot from SSD, satisfied by the concrete `SSDSnapshotStore` and
an in-memory test peer — the second adapter that made the seam real. Carries only
`loadSync`, `loadSyncPrefix`, and `recordHit`; a consumer needing broader SSD
access reaches for the concrete store instead. The ADR-0001 off-MainActor
hydration discipline lives at this seam.
_Avoid_: widening it before a second caller needs a member; the concrete store's
full surface; mock/stub (the in-memory peer is a real adapter).

**Leaf Handoff**:
Transferring a conversation's KV cache objects between their only two possible
owners — the radix tree and a running generation — by move, never by copy and
never by alias: the finished turn's live cache becomes the leaf as it is, and
the request that extends that leaf takes the objects back as its live cache.
A boundary turn's re-prefilled cache moves in the same way: it is the
request's own by construction, so the capture takes it rather than copying it.
Only at quiescent points; only for a leaf hit at its full offset that every
layer can return from (**Leaf Rewind**); anything else restores by copy as
before — except a check-out refused only for a pending full payload, which is
worth the **Pending-Payload Wait** first.
_Avoid_: copy-on-write restore (the alias ADR-0023 rejected — a handoff shares
nothing between two owners); zero-copy (the SSD write still copies); cache
sharing (one owner at a time, never two).

**Leaf Lease**:
The part of a **Cache Claim** that owns the leaf the request took by **Leaf
Handoff**, from check-out to check-in: while it holds, the tree may not drop the
body, demote it, clear the RAM tier of it, promote it into an SSD write, or let
the SSD writer read from it. Ends only at check-in or **Leaf Rewind**; nothing
else ends it, not even the claim's tripwire. The leased bytes stay counted even
while the tree holds no body.
_Avoid_: **Restore Pin** (the weak hold of a copy restore: it protects a path,
it does not own a body); **Cache Claim** for the lease alone (the lease is one
part of a claim); LLM gate (the inference arbiter's turn-taking, a different
resource); lock, refcount (one owner needs neither).

**Leaf Rewind**:
Returning a leaf taken by **Leaf Handoff** to its exact pre-check-out state when
the turn is cancelled, fails, or is intervened: the attention layers are trimmed
back to the leaf offset and the recurrent layers' state is restored from the
independent copy saved at check-out, including lengths, padding and state-slot
metadata, then the leaf is checked back in unchanged.
_Avoid_: **Think-Strip Rewind** (a render-caused prefix invalidation,
unrelated); rollback (the vendor's speculative-decoding checkpoint, which aliases
and is not used here); discard (the pre-handoff cancel outcome, which loses the
conversation's cache).

**Cache Claim**:
One request's whole hold on the prefix cache: its lane in the **Active-Inference
Reserve**, its **Restore Pins**, and, when it takes the leaf by **Leaf Handoff**, its
**Leaf Lease**. **Snapshot Resolution** opens it, and it concludes exactly once,
inside the request's **LLM Gate** turn. The claim is where the request learns whether it
takes the leaf or restores by copy, and why; a claim that took the leaf ends only
by check-in or **Leaf Rewind**, and the leaf is back before the pins and the lane
are let go. Every keyed request holds one, a miss included; a **Speculative
Canonical Prefill** pass holds its own, which never takes a leaf. A claim dropped
without concluding, or concluded twice, is a bug its tripwire reports, not a
leak an age-out trims.
_Avoid_: Cache Turn (a turn is the conversation's generation turn); **Leaf Lease**
or **Restore Pin** for the whole (each is one part of a claim); restore (the claim
decides handoff or copy, and plan application performs the copy); lock, refcount.

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

**Warm-Start Plan**:
The pure decision the **Snapshot Ledger** makes from a decoded manifest before
touching disk: which partitions and descriptors to keep, the grace-stamp
mutations, the **Stale-Partition GC** anchor cut with its typed invalidation
reasons, the seed byte total, the persist-needed flag, and the file deletions as
root-relative paths. The ledger performs it — install under the lock, persist per
the flag, delete off the hot path — but decides none of it (ADR-0055; sibling to
the **Eviction Candidate Policy**). The reused **Warm-Start Outcome** is the
partitioned view it returns to the manager.
_Avoid_: warm-start outcome as the whole decision (that value is only the
manager-facing partition view the plan wraps); manifest rebuild (the corrupt-file
directory walk that *feeds* a plan, not the plan itself); TTL/expiry framings
(staleness is anchor-relative, see the Stale-Partition GC avoids).

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
tree shares one space, and collapses its key path back to render space (one pad
per image) for the **Emitted Path Index**, which never holds an image's identity
or run. Failing to build it yields an **Unkeyed Completion**; a later
translation failure degrades only the consuming feature, not the request.
_Avoid_: key path splice (a shallower predecessor), space converter, token mapper,
render fixup.

**Conversation Render**:
The token-only rendering contract — family message-forming plus chat-template
application, no pixel work — shared by the request edge, the planner's
re-render, the leaf probes, the speculative future-shared-prefix probe pair and
the stable-prefix detector's probe pair, identical to the shape prepare uses, so
a probe render cannot drift from prepare's. Since 2026-09 a module, not a
convention: one per-request value (`ConversationRender`) built at **Request
Keying** where instance truth lives — image-agnostic, every render one pad per
image — whose render verbs own the whole choreography — cache eligibility, the
Render+Token Cache resolve, the template fallback, the no-generation-prompt
merged context — so a call site can no longer wire an ingredient wrong (the
issue #439 defect class). The two probe pairs use its cache-free probe verbs
(cancellable, no cache telemetry). The render cache stays the implementation
below it, and no server source outside the two applies the chat template
(source-shape-tested); the processor `prepare` a bypassing request falls back
to and the agent hand-off suffix, a plain-text encode past the last end-of-turn
marker, stay outside by design. The **Generation Prompt** the planner once
spelled by hand is measured inside it (ADR-0070).
_Avoid_: re-render (unqualified), probe tokenization, per-call-site
`applyChatTemplate` (the rendering it standardizes, not a synonym for it);
RenderTokenSource (the dissolved predecessor — eligibility as a free-standing
value each call site wired by hand); per-site resolve-or-fallback ladders (the
retired shape).

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
`vision_config`); text-only checkpoints do not. Fixed for a downloaded model —
distinct from **Vision Mode**, which is whether that capability is currently
loaded. Every **PARO Checkpoint** in the catalog declares image input.
_Avoid_: "vision model" (ambiguous with the loaded container), "multimodal" (there
is no audio/video input path), "supports images" as a per-request flag, Text-Only
Override (retired by ADR-0089: no catalog entry withholds the image input its
checkpoint declares).

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
fetch one file — satisfied by the HuggingFace-backed production adapter and a
scripted in-memory test peer. Disk stays outside the seam: file checks and
status computation run against the real file system.
_Avoid_: hub client (one adapter, not the seam), download client/backend, model
fetcher; widening it past the two verbs before a second consumer needs a
member.

**Vision Capability Memo**:
The per-model-id cache of the **Vision-Capable Model** disk probe, held once and
shared by every caller; a known answer is cached permanently while an undownloaded
model answers `false` uncached so a later download re-probes. Distinct from
**Vision Mode**: this memoizes intrinsic capability, not load-state.
_Avoid_: vision toggle (a **Vision Mode** concept), per-view capability flag.

**Model Completeness**:
When a catalog entry's folder counts as downloaded. The default is any file
with the entry's required extension, at any depth. The Voice Engine uses the
speech engine's own checkpoint rule instead (`Qwen3Checkpoint`: talker weights,
text tokenizer, speech tokenizer), so the catalog can't show a folder as
downloaded that the engine refuses to load.
_Avoid_: "has weights" (one nested `.safetensors` is not a checkpoint), verify
(that compares sizes against the hub listing, a separate step).

**Retired Checkpoint**:
A directory in the model store that an earlier catalog entry downloaded and
that no entry lists or loads anymore. Removed once at launch, because the
Models page shows only catalog entries and offers no way to delete it. The one
today is the Voice Engine's bf16: the entry downloaded it while `SpeechEngine`
loaded q6. The entry and the engine now read one spec,
`ModelDefinition.textToSpeechModelSpec`.
_Avoid_: orphaned model, legacy download (both suggest the user chose it).

### Prefill orchestration

_The Prefill Plan entry and the terms after it name the design ADR-0087
proposes. Until it is built, plan application still derives the shape, the
Capture Schedule, the split and the Maximum Advance inline._

**Prefill Plan**:
Everything one keyed request's prefill does that is decidable before the
**Cache Claim** check-out, decided by the Prefill Planner as one value: its
**Cache Opening**, the **Image Span** and **Text Tail** it forwards, where the
**Position Anchor** is seeded, its **Capture Schedule**, its **Decode
Handover** and its **Maximum Advance** (ADR-0087). It carries offsets, ranges
and enums only, never snapshots, token arrays or the Speculation Plan's
drafters; plan application carries it out inline and re-derives none of it.
_Avoid_: prefill config; generation params (a separate notion); checkpoint plan
(an input the plan filters, not the plan); the stable-prefix offset as one of
its fields (it reaches the plan only through the checkpoint plan); restore
point (a **Chain-Prefix Restore** term).

**Cache Opening**:
How a keyed request's cache opens before its prefill: cold, from a fresh
cache; restore, where the **Cache Claim** is asked for the resolved snapshot
and a restore that yields no cache falls back to the planner's cold prefill
(ADR-0069 amendment); or whole prompt, where the Speculation Plan's iterator
prefills the whole prompt into a fresh cache.
_Avoid_: restore mode (what the restore turned out to be, an execution
report); opening, unqualified (the Companion's **Day Opening**); restore
shape (the five shapes are products of the opening and the forward).

**Minimum Warm Offset**:
The end of a request's last image run in its **Cache Key Space**, or zero for
a text-only request. Below it a prefill must forward an **Image Span**; at or
past it the remainder is text, and no checkpoint below it is captured.
_Avoid_: image prefix end (true only of a cold span); warm offset clamp; warm
as in **Warm Body** (here it means the first offset a restore can continue
from as text, not a compressed tier).

**Image Span**:
The key-space range an image-bearing prefill forwards through the anchored
vision continuation: from the restore offset, or zero, to the **Minimum Warm
Offset**. It carries only the images whose runs fall inside it and skips the
pixel rows of the images already cached.
_Avoid_: image prefix (only the cold span); vision prefix; restore point (a
**Chain-Prefix Restore** term).

**Text Tail**:
The text a prefill forwards after any **Image Span**: from the restore offset
(zero when cold), or from the **Minimum Warm Offset** after a span, to the end
of the prompt. The app's chunked prefill runs all of it, or only up to the
split under a speculative **Decode Handover**. Its start is the one offset
checkpoints are based at, the split counts from and salvage measures its
progress from.
_Avoid_: execution base offset (plan application's local, which ADR-0087
replaces); prefill base offset (the cached-token count, which differs after a
span); suffix, unqualified; the tail (in ADR-0059 and ADR-0079, the
speculative iterator's own prefill from the split).

**Capture Schedule**:
The checkpoints one prefill captures: the planned checkpoints past what is
cached and at or past the **Minimum Warm Offset**, plus the prefill's
**Transient Boundaries**, a planned type winning at a shared offset. The
chunked prefill also cuts its chunks at these offsets.
_Avoid_: checkpoint plan (the resolution-side input it filters); capture map
(its executor form).

**Transient Boundary**:
A **Prefix-View Checkpoint** captured at the end of the last message or the
last user message, to synthesize this turn's leaf or seed a **Speculative
Canonical Prefill**. Of the two boundary offsets, a prefill captures the ones
past what is cached, at or past the **Minimum Warm Offset**, inside the
prompt and off every planned checkpoint; a text-only request under a
**Preserve-Thinking Render** captures none.
_Avoid_: boundary checkpoint (a telemetry field); boundary helper, helper
checkpoint (the older name in ADR-0019, ADR-0064 and ADR-0068, and today's
local); planned checkpoint.

**Decode Handover**:
Where a prefill hands the prompt to the decode iterator: autoregressive, where
the app prefills the whole **Text Tail** and the standard iterator decodes from
its last token; or speculative at a split, where the app's capturing prefill
runs to the split and the **Speculation Plan**'s iterator prefills the rest. A
keyed split never precedes the **Minimum Warm Offset**, so the iterator takes
text only.
Any generation path can name its own.
_Avoid_: decode route, speculation route (route belongs to the **Prefill
Strategy** and the **Completion Route**); decode handoff (handoff is the
**Leaf Handoff**'s word for moving a leaf between owners).

**Maximum Advance**:
How far one turn may grow its cache past what it restored: its new prompt
tokens, plus the output ceiling, plus the speculative allowance; unbounded
without a ceiling. The **Cache Claim** judges check-out eligibility against
it, and the **Active-Inference Reserve** prices the turn's growth at it.
_Avoid_: max tokens (output only); LeafCheckout.maximumAdvance (retired).

**Prefill Strategy**:
The chunked-vs-single-shot route for one raw-generation prompt (the agent chat
arm), decided once from the prompt's shape —
token dimensionality, sequence length, media presence, step size. 2D text-only
prompts longer than one step chunk through the app driver; everything else goes
single-shot to the token iterator (ADR-0044).
_Avoid_: **Prefill Plan** (the server path's richer pre-prefill value); chunking
flag; VLM path (the model class is one input, not the route).

**Leaf Admission Builder**:
The GPU-free routing decision for storing one leaf snapshot on the boundary path,
in two steps: the reusable-prefix probe that finds the token path a future
continuation will share, then a capture-from-boundary plan or a typed skip
reason. A turn the fast path takes (Live Leaf Capture) never enters it. It
decides; the boundary executor re-prefills and a **Leaf Admission** does the
capture and admit.
_Avoid_: leaf store mode (one input, not the whole story); capture port (it returns
a decision, not a capture).

**Emitted Path**:
The token path a conversation prefix actually took through the model on this
server — the prompt ids as fed plus the generated ids, ending with the canonical
end-of-turn id; kept in render space, each image's fed run collapsed to the
template's single pad, so the path names no image. The truth for every
assistant turn the server generated; a canonical re-encoding of such a turn is
not it.
_Avoid_: **Cache Key Path** (the prompt as keyed, which the Emitted Path begins
with); generated tokens (only the tail); canonical path, stored path (the
re-render's encoding, which the Emitted Path replaces for server-generated
turns).

**Emitted Path Index**:
The per-model-fingerprint map from a rendered prefix — the template's bytes from
the start of the render through a server-generated assistant message's
end-of-turn marker — to that prefix's **Emitted Path**. Keyed on the whole
prefix so the same text after a different history is a different key; bounded in
bytes, evicted least-recently-used, cleared on model unload; last writer wins on
a same key. Image-agnostic: the render bytes carry one placeholder per image and
the path one pad, and an image's identity and run enter through the request's
own **Cache Key Space** and grids after the resolve.
_Avoid_: token cache (the Render+Token Cache holds canonical encodes and never an
emitted id); session table, lineage store (nothing is keyed by client or
session); the tree (KV lives there — the index is provenance and outlives the
leaf it points at).

**Emitted Path Resolve**:
The one way a request's tokens are built: render to bytes, find the deepest
end-of-turn marker the **Emitted Path Index** knows, take that entry's path,
canonically encode only the bytes after the marker, concatenate. Its output is
both the model input and the **Cache Key Space** input; anything unindexed
encodes canonically and is a safe miss. For an image-bearing request the
**Request Keying** edge then expands each pad into the run the processor placed
for that image, so the model input is the resolve's list at prepared length.
_Avoid_: tokenize (the resolve may not tokenize an indexed prefix at all);
render-and-encode (the per-call-site spellings it replaces); lookup (the tree's
read side, which consumes the resolve's output).

**Live Leaf Capture**:
Storing a finished turn's leaf straight from the live decode cache at the
cache's own offset, under the **Emitted Path**, with no re-prefill. By
construction for a turn the **Emitted Path Index** registers: the path the leaf
is keyed on is the path the model fed, so nothing is compared against a
re-render; only structural eligibility (an intervened turn, no fed ids, an
offset outside the live path) sends a turn to the boundary path.
_Avoid_: cache reuse (too broad — the prompt hit is also reuse); proven
append-stability (the ADR-0062 shape — a per-turn comparison the index made
unnecessary); preserve-thinking fast path (the render mode makes a turn
indexable; the index makes it exact).

**Append-Stable Render**:
The property of a chat-template render under which a finished turn's canonical
re-render equals the token path the model was fed — prompt plus emitted ids —
followed only by template glue. A property some templates have and the cache no
longer depends on: with the **Emitted Path Index** a server-generated turn is
keyed on what the model fed whether or not the re-render would have matched.
_Avoid_: preserve-thinking (a render mode that usually has the property, not the
property); canonical render (the re-render itself, which may or may not be
append-stable); a cache invariant (it was one under ADR-0062; it is a template
observation now).

**Think-Strip Rewind**:
The prefix invalidation a thinking template causes when a new real user message
arrives and the template strips the `<think>` blocks it had kept in the assistant
turns since the previous user query, forcing everything past that divergence point
to re-prefill. It is bounded — not removed — by the canonical-leaf probe.
_Avoid_: cache miss after tools (the felt symptom, not the mechanism); template
drift (the render is deterministic); client mutation; **Leaf Rewind** (returning
a checked-out leaf to its prior state — an ownership mechanism, not a template
artifact).

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
A render mode, declared by a template that natively supports it, that keeps
`<think>` blocks in every assistant turn so the render is append-stable and the
**Think-Strip Rewind** cannot occur. Whether it is the template's default or a
per-model choice is the template's business; being part of the template context,
the flag is part of the cache partition, and retained reasoning permanently
occupies context.
_Avoid_: think retention hack (vendor-sanctioned where the template declares it);
template patching (vendor templates are never edited); global setting (per-model).

**Generation Prompt**:
What the chat template appends after the last message to open the assistant
turn under one render context — on the Qwen3.5/3.8 templates an open `<think>`
block by default, a closed empty one when the request turns thinking off.
Measured from the template, never spelled by hand; whether generation starts
inside a think block, whether the turn carries one the template will later
strip, and where the last-message boundary sits are all read from it
(ADR-0070).
_Avoid_: prompt-starts-thinking (the retired load-time guess, blind to a
request's kwargs); generation-prompt string (a hand-spelled copy of it);
assistant header (only its family-specific first part).

### Reasoning effort

**Reasoning Effort**:
The native thinking-depth level of an effort-declaring chat template
(`low`/`medium`/`xhigh`; Qwen3.8 is the first): a template kwarg whose value is
prose injected into the request's *first system block* — so each level is a
distinct prefix from token 0 and its own cache partition. The wire accepts the
union of the OpenAI and Qwen vocabularies mapped to native levels; the kwarg
is emitted only when it differs from the template's own default (`xhigh` for
Qwen3.8), so omitted-or-default keeps the canonical render. Capability is
template introspection, never model name (ADR-0060).
_Avoid_: thinking level (Pi's client-side vocabulary); a sampling parameter (it
changes the prompt, not the sampler); erroring on effort for a
non-declaring model (it is ignored with a log line).

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
**Unkeyed Completion** degrade), resolution + plan (the Prefill Planner;
under ADR-0087, proposed, it decides the whole **Prefill Plan**), plan
application (inline by decision; under ADR-0087 it only carries the plan
out), the stream
drive (the Managed Generation Driver, shared with the agent), the leaf store,
and trace accumulation. Phases are implementation structure inside the
module's seam — the dispatcher's interface is unchanged — and each phase
returns values; the completion module owns effects, except a leaf's: the leaf
store decides which leaf, and its **Leaf Admission** stores it (ADR-0078).
_Avoid_: pipeline stages (the Generation* family owns "stream" vocabulary);
new entry points (ADR-0015's seam is untouched); extracting plan application
(recorded shallow — see ADR-0033); shape logic in plan application (the
inline derivation ADR-0087 removes).

**Keyed Request**:
What **Request Keying** yields for a request it can key: its identities (the
partition, the **Cache Key Space**, the **Conversation Render**) together with
every per-request fact later phases read, each derived once there — the
**Generation Prompt**, whether it is text-only by instance truth (no image
reached the model, whatever the request carried), the prefill step. Its
counterpart is the **Unkeyed Completion**, which carries the same facts and no
keys (ADR-0070).
_Avoid_: request context, keyed turn; a fact re-derived at a call site (the
defect class the value exists to end); the request's own image list as
"text-only" (request intent, not instance truth).

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

**Raw Generation Start**:
The one script that starts a whole-prompt-from-zero generation over a **Model
Session** for an agent chat turn: tokenize through the session's agent-edge verb (the **Conversation
Render**'s agent edge when eligible, the processor otherwise), emit the lookup and prefill progress
events, decode through the session's **Speculation Plan** when it has one, else run
the **Prefill Strategy** route, start the token-event loop, wrap
the handles. It is the **Model Session**'s second consumer (ADR-0016 amendment);
`LLMActor` keeps only the lifecycle around it. Never consults the prefix cache.
_Avoid_: raw arm (the three retired `LLMActor` copies); standard path (the
pre-cache name).

**Stream Lifecycle Driver**:
The module owning one streaming completion's transport-lifecycle race — the
disconnect watch, the idle keepalive, and the drive as first-finisher-wins —
behind injected transport closures, sitting below the ADR-0015 dispatcher seam.
It is the reason a client abort cancels a long prefill promptly (and an idle
prefill keeps proxies from timing out): `CompletionHandler` builds the SSE
envelope and hands the race to the driver rather than constructing a task group
inline; since **Completion Delivery** both transports run under it.
_Avoid_: SSE framing (`SSEWriter`'s); the event pump (`CompletionDelivery.pump`);
keepalive timer (one task inside it, not the module).

**Completion Delivery**:
The one script that carries a started HTTP completion from its first
generation event to the client's last byte — register the dashboard cancel,
record the cache lookup, open the response, drive the stream under the
**Stream Lifecycle Driver**, project the terminal accumulator, emit the
diagnostic, surface the malformed→text fallback, record the session replay,
log and complete. It runs once for both transports; the streaming and
non-streaming arms of `CompletionHandler` were two hand copies of it. Two
rules it fixes: every drive runs under the driver (a dropped client cancels
generation on either transport), and the replay record and `complete` follow
only a fully delivered response.
_Avoid_: completion pipeline; response writer (the sink's job); the two arms
(retired).

**Delivery Sink**:
The transport adapter one **Completion Delivery** writes to — open, one
per-event side effect, the Wire-Valid Close, the terminal envelope — behind
which the script sees only "still connected". Two production adapters make
the seam real (the single-JSON-body sink and the SSE sink, which owns the
**Argument Transcoder** and the `[DONE]` sentinel); a recording peer makes it
the test surface.
_Avoid_: writer (`HTTPResponseWriter` is the socket, not the seam); output
channel; transport (the driver's probe closures, which the sink exposes).

### Streaming detokenization

**Live Delivery**:
The delivery of generated text and tool-call source deltas while generation is
still in progress, preserving the reference decoder's chunk bytes and token-step
timing, including its withholding of incomplete Unicode.
_Avoid_: **Verified Replay** (reconstruction after generation); immediate delivery
(which obscures incomplete-Unicode withholding).

**Verified Replay**:
The reconstruction of a completed turn's emitted chunks for **Emitted Path**
fidelity, with each segment checked before its reconstructed chunks are accepted.
_Avoid_: live audit (already-delivered text cannot be repaired); **Live Delivery**.

**ByteLevel Decoding Capability**:
Established compatibility of a tokenizer with incremental byte decoding that
preserves reference chunks and their release steps, including literal added-token
boundaries.
_Avoid_: model-family support; sample-probe match (neither establishes the
compatibility contract).

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
retraction, so any termination — malformation, cancel,
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
surface (one-shot/streaming generate, **Reference Take** capture and
conditioning, token offsets). The synthesis counterpart of **Speech
Recognizer**.
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

**Streaming Scheduler**:
The pure push-scheduling decider (ADR-0054) that the **Audio Playback** adapter
and the in-memory peer both drive: the buffer counters, the start gate, finish
detection, and the stream epoch that makes a stale buffer completion ignorable
after a stop/restart. Verdicts out, every AVAudioEngine effect stays in the
adapter — a policy/performer value machine like the capture engine's lifecycle
and duck. Voice-session replies play through the same adapter (ADR-0082).
_Avoid_: playback scheduler (AVAudioPlayerNode's own scheduling), stream pump
(retired **Segment Playback** vocabulary), duplicating the fold per adapter (the
pre-#395 shape).

**Playback Diagnostics Dump**:
The pure value that turns one TTS playback's captured samples plus conditions into
the on-disk diagnostic artifact bytes (WAV + metadata) — the playback-side sibling
of the dictation **Capture Dump**. The playback adapter feeds it and writes what it
returns; encoding is byte-testable without an audio engine.
_Avoid_: capture dump (the dictation ring buffer), WAV encoder (one part of it),
debug recording.

### Speech word timeline

**Word Timeline**:
The pure, immutable words of one spoken segment on a character line: where each
word starts and ends when the words are joined by single spaces, and which word a
character count falls in. For a segment without **Word Timing** the
**Read-Along** places the heard character and the timeline names the word.
Words split the way the engine, the **Word Timer** and the **Reader** split
them: on whitespace and newlines.
_Avoid_: WordPacing, pacing model (the retired fold the notch drove), word
highlighter, overlay panel (a different, dictation surface).

**Read-Along**:
The one clock that says which word of the speech now playing is being heard,
followed by both the **Reader** and the **Speech Overlay** (ADR-0076). It is the
production **Word Highlight Surface**: segment scripts arrive seconds ahead under
lookahead pacing, but a segment starts only when the heard time reaches its
**Segment Window**. Inside a segment, the heard word is the last one whose
**Word Timing** start has passed (ADR-0077); a segment without timing spreads
its characters evenly over its audio. The word never moves back within an
utterance. It samples the heard time (the playback head less the output's
latency) 30 times a second and publishes only when the heard word changes.
_Avoid_: TTS Word Tracker (retired with the notch: it switched text on arrival
and ran ahead of the voice), karaoke, word state machine.

**Word Timing**:
When each spoken word's sound starts, found by the engine from the voice model's
own attention (ADR-0077): the **Word Timer** follows the talker's **Alignment
Head** through the segment's text and moves starts past the pauses the audio
shows. It rides the **Utterance** stream as word starts, each after the audio
it starts in. No transcription model runs.
_Avoid_: forced alignment (a second model run on the audio), token offsets and
the K1 invariant (the retired one-text-token-per-frame assumption), word
tracker.

**Alignment Head**:
The talker attention head whose attention sits on the text token being
spoken, frame by frame: layer 3, head 0 in the 1.7B checkpoints; layer 6,
head 5 in the 0.6B. A property of a checkpoint family, measured once
(`docs/research/2026-09-27-word-timing-from-attention.md`) and carried by the
model spec.
_Avoid_: alignment layer, the probe (the mechanism that reads the head).

**Word Timer**:
The engine's pure follower of the **Alignment Head** for one segment: a
forward-only path over the text's tokens, decided 8 frames behind generation,
with each start moved to where the sound resumes after a pause. It counts frames
as the silence cap kept them.
_Avoid_: aligner (suggests a separate model), DTW (it decides as it goes).

**Segment Window**:
The single playback-time base a **Read-Along** places one long-form segment
against — one value, so a time-base/duration-base disagreement is
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
highlighting (show, switch, time words, mark complete, dismiss). The production adapter is
the **Read-Along**; a recording test peer makes the segment-boundary switch
assertable. In engine v2 its switch timing comes from **Segment Script**
ground truth, not playback bookkeeping.
_Avoid_: notch overlay / TTSNotchPanelController (retired adapter, not the seam),
highlight view, the **Lens** (the separate dictation overlay).

### Speech page (ADR-0076)

**Reader**:
The Speech page: the owner's text of any length, its **Bookmark**, and the
reading in progress. It starts speech from the bookmark or a selection, follows
the **Read-Along**, and maps each heard word to a range of the text by counting
words forward from where the previous segment ended. Nothing walks the whole
document, so a book costs what a page costs.
_Avoid_: composer (retired with the old page), player, speech editor.

**Bookmark**:
Where reading resumes: an offset into the **Reader**'s text, saved beside it.
It follows the sentence being heard, moves with edits before it, and returns to
the start once the text has been read to the end.
_Avoid_: cursor (the text view's insertion point), playhead (audio time),
progress.

**Speech Overlay**:
The floating panel that shows the words being read over every app, as an island
at the top of the screen or captions at the bottom, following the **Read-Along**.
It shows one continuous feed of the reading: the heard line on top, the next
below, the column moving up a line as the voice reaches it; text is never
swapped in place (ADR-0077). In its automatic scope it hides while the Speech
page is in front. Replaced the TTS notch.
_Avoid_: notch, TTS notch panel (retired), the **Lens** (the dictation overlay),
pages (its retired two-line pages).

**Voice Source**:
Which voices the speech checkpoint offers: designed from a description
(VoiceDesign) or a fixed list of **Preset Voices** (CustomVoice). The Speech
page offers designing a voice only for a designed source.
_Avoid_: voice mode, voice type, voice provider.

### Speech engine v2 (ADR-0038)

**Speech Session**:
The voice-identity owner at the engine boundary: opened from a `SessionProfile`
and a **Voice**, it holds the voice's **Reference Take** (per its policy),
admits utterances one at a time, and dies by `close()`. Sessions survive engine
unload as ingredient values.
_Avoid_: generation session (an LLM concept), voice handle, session manager.

**Utterance**:
One admitted text→speech run: admission-time facts (sample rate, frames/s,
segment count) plus the single event stream that is simultaneously the audio
transport, the timing channel, the backpressure channel, and the cancellation
token. Dropping it stops generation; cancelling the consuming task surfaces
`CancellationError` untranslated.
_Avoid_: speech stream (one field of it), generation handle, request.

**Segment Script**:
The per-segment value announced on the stream before its audio: text slice and
the utterance-global `startFrame` — the **Segment Window** as ground truth from
the generation loop, ending the estimator era. Its words' starts follow as
**Word Timing** (ADR-0077).
_Avoid_: segment metadata, offsets payload, token offsets (retired).

**Reference Take**:
One short rendering of a voice, kept as its codec frames plus the text they
speak, that every later segment of the voice continues so it stays the same
person (ADR-0072). It is the first segment a session renders, or the one
"Try another take" renders; it is never encoded from audio.
_Avoid_: voice anchor (the retired 48-frame KV prefix), reference audio, voice
sample, seed.

**Pinned Voice**:
Voice identity as a serializable value: the description, its **Reference
Take**, and a {model, precision, schema} fingerprint. A few KB, stored per
designed voice so the voice survives relaunch. Restore validates the
fingerprint and schema or throws; a seed is never voice identity (#339).
_Avoid_: voice anchor, seed, voice id.

**Preset Voice**:
A fixed-timbre named speaker of a CustomVoice checkpoint: chosen from the
checkpoint's own list, never designed, needing no **Reference Take**; the
phone's voice identity. Distinct from **Pinned Voice**, which is designed from
a description and holds a take.
_Avoid_: speaker id (the wire field), custom voice (the checkpoint family, not
the identity), voice preset, fallback voice (the system synthesizer that plays
while a Preset Voice loads is not a voice identity at all).

**Readiness**:
The engine lifecycle ladder — `.unloaded` / `.loaded` / `.warm(priming:)` —
driven by one idempotent `prepare` verb; warmup and voice-prefix priming are
rungs, not separate APIs.
_Avoid_: engine state (the observable presenter concept above the seam), load
status.

**Pacing Policy**:
The demand-based backpressure contract: `.eager` or `.lookahead(segments: n)` —
at most n undelivered segments beyond the in-flight one; the engine parks until
the consumer pulls. Pause is its consequence, not an API.
_Avoid_: throttling, playhead clock (rejected design), lookahead buffer.

### Generation accumulation

**Generation Accumulator**:
The one value that folds an `AgentGeneration` event stream into a single assistant
turn's accumulated state — text, optional thinking, finalized tool calls, the raw
malformed-tool-call buffer. A pure value with no
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
`AgentGeneration` event stream for one assistant turn, preserving the model’s
reasoning and owning the parser lifecycle and external cancellation. Caller side effects and projections stay with the callers;
terminal info and diagnostics come back on its outcome.
_Avoid_: managed generation (the **Managed Generation Driver** above it, not this
loop); stream consumer / generation pump; GenerationFold (the fold is the
**Generation Accumulator**); ToolCallParser (upstream); agent loop — this consumes
one turn's raw stream, whereas the agent double-loop orchestrates turns and tool
calls above it (say "stream loop" vs "agent loop").

**Managed Generation Driver**:
The one module that drives a raw model stream through the **Generation Stream
Loop** for both consumers — the agent turn and the cache-aware **Server
Completion**: loop construction, late-bound cancel bridging, stream-termination cancel wiring, and
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
queued at the **LLM Gate** or behind a model load. If the run dies before the commit
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
`/compact` — taking its turn at the **LLM Gate**, from queued through active to
cancelled or done. Distinct from the Generation* family, which is the token stream
*inside* a turn; an Agent Run is the outer gate + busy + cancel envelope, and its
`isGenerating` means "queued **or** active," not just running.
_Avoid_: generation lifecycle (collides with the Generation* family), send
coordinator, busy flag as standalone spine state.

### Active tool resolution

**Active Tool Set**:
The one resolve answering "which tools does the agent see right now": the
registry-ordered tools plus a **Tool Gating** in, the live set and its
**Prompt Tool Facts** out (ADR-0048). Pure and stateless; both the callable
set and the prompt's orientation sections derive from one resolve, so they
cannot diverge. Distinct from `ToolRegistry`, which answers what tools
*exist* (identity + precedence order) and stays beneath it.
_Avoid_: tool filter (three of the old eight homes), tool registry (the
identity layer below), per-consumer filter chains (the retired shape),
toolset (unqualified).

**Tool Gating**:
The context one Active Tool Set resolve runs under: the Web Access switch.
There are no tool audiences: every conversation and every Companion moment
resolves the same way, so they share one cached system-and-tools prefix
(ADR-0080).
_Avoid_: audience filter (retired with the old Companion), web gate
(unqualified — the switch is one field), consumer flag.

**Prompt Tool Facts**:
The prompt-facing facts of a *resolved* Active Tool Set — has the skill
tool, carries browser tools — the only input the system-prompt assembler's
orientation sections may condition on. A facts change (Web Access flip,
browser tools landing after a late MCP connect) re-derives the prompt
through the factory-wired reassembler; unchanged facts never touch it, so
the prompt head's prefix-cache entry dies only for a real orientation
change.
_Avoid_: prompt flags, membership predicates (the retired inline checks),
deriving facts from the raw registry (the drift that shipped).

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

**Skill Execution**:
The chat's collaborator for firing a **Skill** (ADR-0045 continuation, #408): it owns
what a fire *is* — assembling the argument text from the drained composer draft,
rendering the **Skill Envelope** injection block around the skill body, and recording
the user-initiated invocation for the **Skill Usage Ranking** — and returns the
injection for the **Chat Session** spine to send. Touches no `Agent`, no arbiter, no
error banner (a load failure is a nil render the spine surfaces). Held by the session;
a no-op default leaf in tests, the container-wired one (assembly + recording on the
**Skill Pill** controller) in production.
_Avoid_: skill tool (the model-invoked **SkillTool** `use_skill`), command execution
(the palette/spine that routes a slash command *to* this), executeSkill-the-method
(the retired inline home this replaced).

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

**Skill Envelope**:
The one home for how injected **Skill** content is framed (ADR review Strong #6):
three renderers — the `<skill name=… location=…>…</skill>` user-message
*injection*, the `use_skill` *tool result*, the *linked file* result — and one
parser. Only the injection format is persisted (transcripts store it, the **Skill
Invocation Row** re-parses it), so only it earns an inverse, governed by the
round-trip law `parse(injection(x)) == x` for every realistic skill; the
tool-result formats are model-facing prose and parser-less by design. Round-trip
boundary (name with `"`/`>`, body with a literal `</skill>`) is documented-lossy,
not escaped — escaping would change the bytes the model reads or the persisted
block.
_Avoid_: skill block (vague), per-call-site literals (the pre-#401 shape — a
contract living only as matching producer/fixture literals drifts green), parsing
the tool results (deliberately parser-less).

### Companion: the day (ADR-0080)

Jarvis thinks at moments; code keeps the promises. Every decision becomes an
artifact — a reminder, a calendar event, a Nudge, an owner rule, a Profile fact
or a card — that code and the OS keep without the model.

**Today**:
The app's home and first main-window page: the date and "N of M done" over a
progress bar, the Now Card, then the day as steps: the Timeline (events and
tasks in time order, free gaps, a Now line, then the Anytime tasks) on one
**Day Line**, walked up to now and ahead after it. The line runs on past the
**day break** into tomorrow (its events, all-day events and the tasks due
then), so what comes next is always in sight. Today is the owner's day: until
04:00 (the Day Key) it keeps the day that is ending. A wide page draws the
steps as a table (time, task, Area, length) with the Inbox and "Jarvis
noticed" (Fact Proposals) beside them; a narrower one puts those below, and a
phone-width one draws the steps as a list. The **Today composer** at the bottom (the agent composer
simplified) asks Jarvis, adds a task through Capture or takes a held-to-talk
question, and its notice slot shows the latest agenda change with an undo.
Its Chat mode shows the Day Thread.
_Avoid_: dashboard, Mission Control (retired), home screen (unqualified).

**Now Card**:
The top of Today: the step the day is on (the meeting under way, the task
whose slot is now — with how much of it is left, in words and a short bar
that drains like a visual timer — a task that slid, free time and what fits
in it, the next step, what's left, or a done day and how tomorrow starts; in
the evening it closes the day — what is still ahead tonight, else what got
done and what is still open, never a slid task at the top) with one-click
offers that move it on (Done, Start now, "Do it at 16:30", Tomorrow, Plan my
day, Wrap up the day). Code builds it from the Timeline, so it never waits on
the model; Jarvis's latest card line rides along while fresh (a plan for four
hours, a welcome back for one, the evening's cards until the day ends) — a
Breakpoint with nothing for the owner says nothing, so the plan's word stands —
and everything Waiting on You and the evening's leftovers sit on it, each said
once.
_Avoid_: hero, focus card, "the card" (unqualified: each Moment makes a card).

**Agenda**:
Apple Reminders and Calendar as Tesseract reads and writes them: one port with
an EventKit store and an in-memory store (the test host's, ADR-0073), and a
facade that keeps today's snapshot and turns every write into a one-line
confirmation with an undo. The single source of truth for tasks and plans.
_Avoid_: tasks file (retired `tasks.md`), task store (Reminders is the store).

**Area**:
A part of the owner's life a day spans — Work, a side project, Health, Life —
stored as a Reminders list and mapped once in Settings.
_Avoid_: project, category, domain.

**Inbox**:
The Reminders list where captures without a home land, undated. On Today each
item carries an offer: a free half hour today, clear of the Now Card's offer
and the items above it ("At 16:30"), or Tomorrow (due tomorrow, for the next
Morning Plan to place) once today has no room or it is evening.

**Capture**:
Typed or spoken words turned into a reminder without a model: the capture
hotkey's panel, the + in the Jarvis Panel's field and the Today composer's
Add task (⌘↩) all go through one door and the
deterministic capture parser ("call the dentist tomorrow at 10", "after the
1:1", "#health"). The hotkey is one key — Right ⌥ alone by default: tap to type,
hold to speak; another key pressed with it cancels, so ⌥-typing works as usual.

**One-key hotkey**:
A modifier pressed and released on its own (Right ⌥, Right ⌘, Right ⌃, Right ⇧
or fn) used as a hotkey, stored as the modifier's own key code with no
modifiers. Down when the key goes down with nothing else held, up when it is
released, cancelled when any other key or modifier joins.
_Avoid_: double-tap (the ⌘⌘ Appshot chord is both Command keys held together),
modifier-only combo.

**Must-do**:
The day's one optional thing that matters most. It can sit anywhere in the day,
even late; the owner moves or clears it with one click.
_Avoid_: focus (the day has several goals, not one focus), priority.

**Nudge**:
A notification scheduled with the OS a few minutes before a calendar event, so
it fires even with Tesseract closed. Reminders with a due time need none: they
carry their own alarm, delivered by Reminders on every device.
_Avoid_: wake (retired), alert (unqualified).

**Step Cue**:
A planned step put on the Jarvis Panel when its slot starts and the owner is at
the Mac: the task, until when and what follows, with Start (the slot starts
this minute), In 15 min (it moves on and is cued again then), Tomorrow and
Done. A Morning Plan's slots and the owner's own "Do it at 16:30" live only in
the day's state, so without it nothing marks their start. A step the owner
started (Start on a cue, or Start now on Today) checks in when its time is up:
Done, 15 more min (it runs longer and checks in again), Tomorrow; while it
runs, no other start interrupts it. Built by code, no model; each start and
end once, within ten minutes; never while away, in quiet hours, a call, a
game or a meeting, nor over a panel that is up; a slot whose reminder rings
at the same minute is left to Reminders. Closing it changes nothing.
_Avoid_: reminder (Reminders' own alarm), nudge (an event's OS notification).

**Day Engine**:
The Companion's pure decider: signals (the clock, presence, a meeting ending,
notifications, an app coming forward, coding agents, the agenda, a moment's
result, a card action, power) and a snapshot in; the day's next state and an
ordered list of effects out. It performs no I/O; the Companion runtime gathers
and performs.
_Avoid_: Wake Evaluator, fold reducer (both retired), scheduler.

**Moment**:
One model call at a point in the day when there is something to judge — the
Morning Plan, a Breakpoint, a Triage, the Evening Wrap-up, the Night
Reflection — asking for a JSON card. A reply is validated against the facts it
was shown; an invalid or failed one gets one retry, then a deterministic card.
Every moment runs at the owner's own reasoning effort (one cached prefix) and
under its own output cap. No moment runs without new input.
_Avoid_: turn (a chat word), beat, wake.

**Morning Plan**:
The moment at the day's first sit-down (after an overnight gap of four hours or
more, within the morning window), or when Today is first opened that day: small
tasks into the free time before the first meeting, the Must-do placed where it
fits. A card built by code goes up at once — the day's shape and where it
starts — and Jarvis's version replaces it in place when he has thought it
through; when the Mac is awake and on power before the owner sits down, the plan
is made ahead and comes forward at the sit-down. The first sit-down is the
owner starting the day, so the plan meets them on the panel even before quiet
hours end. A plan the app quit in the middle of runs again once when the
Companion next comes on (not once the owner closed it, nor in the evening).
Skipping it costs nothing and nothing nags.

**Breakpoint**:
Coming back after the away threshold (ten minutes by default), or a meeting
ending while the owner is present. A code-built **Breakpoint Card** goes up at
once; the model refines it in place only when there are notifications to judge.
Nothing waiting means no card at all, and a card with nothing that needs the
owner stays in Today — it never pops up over their work.
_Avoid_: summons (retired), welcome-back notification.

**Breakpoint Card**:
What needs the owner (people, agents, missed notifications — one action each:
Open, Later, Done), the next event and what fits before it, where the owner
was, and how many other things can wait (expandable). Titled "While you were
away"; Jarvis's line does the greeting.

**Jarvis Panel**:
The floating card rung: a Siri-style Liquid Glass panel near the top-right,
over any app — close top-left, expand to Today top-right, the card or a Step
Cue, and an "Ask Jarvis" field between + (capture) and a mic. A card on it is
whole: a Morning Plan lists its steps, an Evening Wrap-up its leftovers with
their choices, and what the owner handles there leaves it. As tall as what it
says. It never steals typing: it turns key only when its field is clicked.
Replies go to the Day Thread.

**Triage**:
The moment that judges new unresolved notifications from people while the
owner works — at most every ten minutes, in a batch, never one by one, never
while a game is in front — and raises only what can't wait for the next
Breakpoint.

**Notification Source**:
Who a banner is from, decided by code on arrival: a person (a chat, a mail, a
call — messaging apps, or a messaging site in a browser), an app's own news (an
image is ready, a download finished, a bot such as Jira or CI posting through a
chat app — paging tools stay people), or noise (the system's own banners such
as Game Mode, and games). Only people reach Triage; an app's news waits for the
next Breakpoint; noise is never shown. Owner rules apply first.
_Avoid_: priority, category (the App Store's).

**Evening Wrap-up**:
The moment at the evening time (or the next presence before 03:00): what got
done, and each leftover rolled to tomorrow or another day, kept for later, or
let go. Nothing is ever labelled missed or failed.

**Night Reflection**:
Once a night, after the Evening Wrap-up — on power, or on a battery at least
half full with a cool Mac: tomorrow's carry-over note, a first draft of
tomorrow, and zero to three Fact Proposals. The note and the draft open the
next day's thread, where the Morning Plan reads them.

**Day Thread**:
One append-only conversation per day, on its own agent with the same system
prompt and tools as every chat. It opens with the **Day Opening** (the Profile,
the Areas, today's agenda, last night's carry-over note and first draft of the
day); moments append a
request and a card; the owner's Today chat appends too. A new day (at 04:00)
starts a new thread; within a day compaction runs only past its ceiling.
_Avoid_: Mission Control, standing conversation (both retired).

**Now Tag**:
The one line of time every user message carries into the model — local date,
weekday, time and time zone — stamped when the message is created and stored
with it. The system prompt carries no time, so it is byte-identical everywhere.

**Seen Ledger**:
Which of other apps' banners the owner has already seen: seen when the app comes
forward (within 15 minutes while present, or on return if it arrived while
away). Only unresolved banners reach Jarvis, never twice; they expire after a
day. **Owner rules** (app, sender, keywords → ignore, hold or raise), set by
talking to Jarvis, apply first.
_Avoid_: Notification Hub, Event (both retired).

**Waiting on You**:
People, coding agents and missed notifications that need the owner, under
"Needs you" on Today's Now Card; a waiting agent also shows on the glyph.

**Delivery Ladder**:
How a card reaches the owner, decided by code: the glyph for anything waiting,
the Jarvis Panel for Breakpoints and urgent items when present, a banner when
away or locked, a spoken line for urgent items when voice is on. Quiet hours
silence Jarvis's own deliveries — except the Morning Plan at the day's first
sit-down, which is the owner starting their day; the owner's reminders and
Nudges still fire. Nothing unanswered is re-summoned; it stays in Today.

**Wind-down**:
The night's one banner as quiet hours begin with the owner still at the Mac:
"Time to wind down — Tomorrow starts with All Hands at 07:30 — 8 h 30 min from
now." Built by code, within the first hour of quiet hours, never in a game or
a call; the owner can switch it off beside quiet hours in Settings.
_Avoid_: bedtime (Health's own), reminder.

**Governor**:
No daily budget, usefulness first — but Triage and the Night Reflection wait
while the Mac is hot or low on battery; the Night Reflection runs on power, or
on a battery at least half full with a nominal thermal state.

**Profile**:
The small set of facts about the owner that they approved — readable, editable
and deletable on the Profile page. `remember` saves one (an explicit ask is
approval), `forget` removes one. Facts ride only the Day Opening; ordinary chats
get none.
_Avoid_: memory (unqualified), beliefs, episodes (all retired).

**Fact Proposal**:
A "Should I remember this?" from the Night Reflection, waiting in Today and on
the Profile page until the owner chooses Remember, Edit or Not true.

**Recall**:
The `recall` tool: a search, only when asked, over the Profile and past
conversations (a full-text index, reranked by the embedder), answering with
dated snippets.

**Companion Trace**:
The append-only JSONL record of every Jarvis decision, card, reaction and
agenda change, in a closed vocabulary, one file per day, each record stamped
with its Day Thread; model calls carry tokens (with the cache's share),
latency, the model, and the thermal and power state.
_Avoid_: flight recorder (the retired one it replaced).

### Voice capture

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
minimum-duration and empty-text guards, the silent-capture skip (a capture whose level
never rose above silence is not transcribed), post-processing, the **Learned Words**
(applied after the regex cleanup and before the **Proofread Pass**), the in-flight
transcription `Task`, and cancellation — behind a small value-returning interface
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
Off by default and opt-in (ADR-0085): **Learned Words** are the corrector, and when
on, the pass reads the text after them. Strictly fail-open: disabled, model not
downloaded, the LLM generating (skip-when-busy — it *reads* whether the **LLM Gate**
is held, never waits on it), budget overrun, or any error all commit the raw text
unchanged. Runs inside the **Voice Capture Session**, so dictation and
**Voice Input** both gain it; its word-level edits ride the commit, and a rejected
take's raw text stays available for the **Lens**'s "Insert anyway".
_Avoid_: post-processing (the regex cleanup that always runs, pass or no pass),
autocorrect, grammar check, second agent (it is a fixed-prompt pass, not an agent).

**Correction Pair**:
One dictation take's full text lineage — raw ASR, regex-cleaned, after the
**Learned Words**, **Proofread Pass** output + verdict, committed text, the owner's
correction — plus capture conditions and a Capture Dump audio reference; the local,
bounded, exportable training-pair collection the flywheel feeds from day one. Every
take is a *candidate*; an owner signal makes it *gold*: evicted last, its audio
exempt from the dump's ring eviction. The signals are a word fixed in the **Lens**
(in a **Held Take**, after the paste, or from the **Catch Record**), recorded with
the heard and meant words, how the take was reached and the app; "Insert anyway" on
a take the **Proofread Pass** rejected; and, on older pairs, a correction or
wrong-flag saved in the history before its editor was retired. Fixing a take
happens only in the **Lens**; nothing edits a pair's text by hand.
_Avoid_: training data (unqualified — pairs are candidates until gold),
feedback log, transcription history (the sibling store it links to by id),
fine-tune corpus (the export's *consumer*, out of scope — see the map).

**Learned Word**:
One "heard → meant" replacement learned from the owner's fix in the **Lens**: the
owner's spelling, every way it was heard (a heard form may be several words), the
apps it is left alone in, its **Catch** count per day, and the **Correction Pair**
it came from. The **Voice Capture Session** applies it to every take after the regex
cleanup: whole words, any case, case fitted at a sentence start, skipped in an app
it is left alone in. Learned on the first fix, unless the fix touches only ordinary
words, only changes a word's ending, or does not sound like what was heard (a
rewrite): those fix that take only. Fixing it back in an app leaves it alone there.
Forget (on its tile in the **Catch Record**) keeps the word, so Forget can be
undone, and only a new fix learns it again (ADR-0085).
_Avoid_: dictionary entry, custom vocabulary (a list the recognizer is biased
toward, which is out of scope), replacement rule (unqualified), autocorrect,
snippet.

**Catch**:
One application of a **Learned Word** to a take: the misheard words it replaced in
that take's text. Counted per day once the take commits (a rejected, failed or
superseded take caught nothing, and a catch shown in the **Live Preview** is not
counted); a catch the owner fixes back in the **Lens** is taken back. The **Catch
Record** counts them per day.
_Avoid_: correction or fix (the owner's act; a catch is the app's), hit, match.

**Lens**:
The one dictation overlay: a glass card at the bottom center of the screen that
shows a take while it is recorded (the **Live Preview**), while it finishes, as it
lands (the words the preview had wrong settle into place), while it waits as a
**Held Take**, and while it is fixed. The fix hotkey (⌃⌥Space) reopens the last
take, and the **Catch Record** opens one of today's takes, or any take in its
history; the owner types the word they meant, and the Lens picks the words that
sound like it (← → or a click pick by hand). A fix makes the take's **Correction
Pair** gold, teaches a **Learned Word** when it is a mishearing, and goes back into
the app while the pasted text is still the last thing typed there (a take opened
from the Catch Record is not pasted back). It never takes focus while listening:
it takes the keyboard only while a take waits or is being fixed, and gives focus
back when done (ADR-0085, ADR-0086).
_Avoid_: overlay (unqualified), pill, HUD, Overlay Variant and Overlay Panel (both
retired: the Lens replaced the variant registry, its Setting and the pill's
fixed-frame panel), editor, fix window, popup, history editor (the history's
full-text pair editor, retired with the **Catch Record**: the Lens is the one
place a take is fixed).

**Live Preview**:
What the **Lens** shows while a take is recorded: the same Whisper model's decodes
of the take so far, read the way the take will be (the regex cleanup and the
**Learned Words**, so a learned word flips as it is heard), as confirmed words and
a provisional tail that the next decode rewrites. Only ever shown: what pastes is
the full pass over the whole take after release (ADR-0086). Succeeds the Live
Partial signal (#291).
_Avoid_: Live Partial (the retired trailing-window signal), partial, caption,
streamed transcript (streamed text is never pasted), interim result.

**Held Take**:
A take that waits in the **Lens** instead of pasting: committed (history,
**Correction Pair**, **Catches**) but pasted only when the owner presses ↩, with
any fixes. Esc or clicking away keeps it unpasted, and the fix hotkey brings it
back. The Check Before Pasting setting decides which takes are held: one where the
owner tapped ⇧ while talking (the default), every take, or none (ADR-0086).
_Avoid_: waiting take (the Lens's state, not the take), pending paste, draft,
queued take.

**Catch Record**:
The Dictation page, showing the **Learned Words** at work: one sentence and a
seven-day chart of this week's **Catches** and the owner's fixes; one tile per
Learned Word (every way it was heard, the fix that taught it as before and after,
the apps it is left alone in), each with Forget and Undo; and today's takes,
each one click from a fix in the **Lens**. A forgotten word leaves the tiles, the
chart and the sentence; today's takes still show what it caught. The full
transcription history stays one toolbar button away, beside
recording and the **Correction Pair** export (PRD #612).
_Avoid_: dashboard, stats, Dictation history (the history is behind the toolbar,
not the page), word list.

### Voice session (Companion)

**Voice Session**:
Voice as a mode of the one conversation, never a separate surface: an
auto-listen loop — listen, capture, transcribe, send, speak the reply — whose
spoken and typed turns share one persisted message stream. Half-duplex: the
microphone is never open while speech plays — the reply, or anything else
read aloud — and it opens again once it stops (ADR-0082). Entered from the
overlay or the chat toggle; left by dismissal, staging to the composer,
mutual silence, or a microphone that dies twice in a row.
_Avoid_: voice mode (UI shorthand), speech session (the TTS reading concept),
voice chat, full duplex (the retired open-mic design).

**Voice Session Machine**:
The pure reducer owning every judgment of the **Voice Session**'s auto-listen
loop — phases, the half-duplex turn order, **Barge-In**, the dead-capture
recovery, the deaf window after speech, the speaking watchdog, and the
capture retry backoff. It folds events (ticks, the barge-in, the reply,
transcription outcomes) into ordered effect values the controller performs;
time, the mic level and the capture engine's dead-input flag are inputs, and
capture start results return as events, so even the retry backoff is machine
judgment. The endpointer is its sub-state (ADR-0042, ADR-0082).
_Avoid_: controller (the performer above it), state machine (unqualified),
tick handler, **Voice Session** (the product concept, not this module).

**Barge-In**:
The owner interrupting speech during a **Voice Session** — the reply, or other
speech holding the mic closed — by a key press (the Talk to Tesseract or
Speak Selected Text hotkey while a session is active) or a click on the
speaking line, never by voice: the mic is closed while anything speaks. The
speech stops at once and for good, then the mic opens for the owner's turn
behind the short deaf grace that follows all speech.
_Avoid_: interruption (unqualified), voice barge-in / Soft Barge / pause-on-barge
(the retired open-mic stages), Substance Gate / Session Directive (the removed
word gates).

**Self-Echo**:
The assistant's own TTS re-captured by the microphone and treated as the
owner's speech — a committed turn feeding the conversation its own words back.
The open-mic design's signature failure; half-duplex makes it unreachable: the
mic is closed while speech plays, and a 0.3 s deaf grace covers the room tail
after it stops.
_Avoid_: feedback loop (the mechanism, not the name), echo (the raw acoustic
signal cancellation removes).

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
verdict, the live-input check, disarm-after-grace, idle rebuild and arm retry —
with the AVFoundation engine as the performer. The ADR-0025 policy/performer
split, applied to the engine itself; the arm mode (always-armed vs
disarm-after-grace) is one input. Every capture is a per-take tap at the device
rate. The *live-input check* watches an open capture (not the settings meter)
every 0.5 s through a heartbeat the tap bumps per buffer — a live input delivers
buffers through silence too: an input that never delivered gets 1.5 s, then is
rebuilt and restarted once under the same open capture (nothing recorded,
nothing lost) — unless its engine is under 5 s old, when it is marked dead
rather than rebuilt back to back; one that went quiet, or that the restart
didn't revive, is marked dead — the
meter drops to zero and the capture stays open for its owner, whose stop
discards the engine and keeps what arrived before the input died (ADR-0082).
_Avoid_: engine defaults (capture mechanics stay on the engine); VPIO lifecycle
(the arm mode is an input, not the policy); duck policy (the sibling policy for
system audio); voice hold (the retired session-long engine, ADR-0082).

**Capture Dump**:
The on-disk ring buffer of recent dictation capture audio — what the microphone tap
delivered (post–**Voice Processing** when enabled, pre-resample) — tagged with its
capture conditions and kept for diagnosing bad transcriptions; bounded by count/size,
oldest evicted first.
_Avoid_: recording archive (it is bounded and diagnostic, not an archive), audio log.

### LLM gate

**LLM Gate**:
One language-model generation at a time: a single scoped operation grants one
caller the loaded LLM, FIFO, with an atomic handoff and cancellation while queued.
Only LLM work takes it — chat turns, the Companion's moments, HTTP requests,
`/compact`, reloads, offload — because the model is one container with one prefix
cache. Speech, dictation, the proofreader and the embedder never take it: they run
beside the LLM, and MLX's own lock keeps concurrent evaluation safe (ADR-0081). It
was the **GPU Lease Queue**, which also made speech wait out whole generations.
_Avoid_: GPU lease (retired: the GPU is not serialized), GPU mutex/semaphore,
scheduler (no policy beyond FIFO).

**Inference Arbiter**:
The model-affine layer that composes the **LLM Gate** with the LLM's identity,
holding the gate across both load and body so the loaded model cannot change under
a running consumer, and the residency mirror Offload Model reads (LLM and voice).
Distinct from the gate below it (no model awareness) and from
`ModelDownloadManager` (acquisition, not arbitration).
_Avoid_: gate (the layer below), model manager (collides with
`ModelDownloadManager`), GPU manager.

**Inference Arbitrating**:
The narrow single-member seam LLM consumers depend on (`withLLM`), satisfied by
the production **Inference Arbiter** and an in-memory test peer. A consumer needing
reload or model-state access reaches for the concrete arbiter instead, so the seam
stays minimal rather than widened speculatively.
_Avoid_: arbiter protocol / arbitering, lease provider; widening it before a
peer-consuming caller needs the member.

**Models section**:
The menu bar's always-visible list of every model Tesseract can hold — language
model, its speed-up draft, voice, dictation, proofreader, memory search — each
with whether it is loaded and what it is doing right now ("Jarvis · Triage",
"Speaking", "Listening"), its size, and the app's memory footprint with the
system's swap. Memory, not the GPU, is what the models share; this is where the
owner sees it.
_Avoid_: model manager (the Models page in Settings), activity monitor.

**Foreground Gate**:
The phone's one switch between "GPU work may start" and "the app is leaving the
foreground". It is owned by the **Inference Arbiter**, because iOS refuses GPU
work from a backgrounded app. While it is closed, no LLM turn starts, and every
GPU consumer stops at its next safe point: a reply pauses after its current token
(it is never cancelled), speech redoes its in-flight segment, and consolidation
yields.
_Avoid_: background mode (the audio entitlement that keeps playback alive),
suspension (the OS's act, not the app's), GPU lock, pause (unqualified).

### Batch inference

_Not pursued (2026-09-22): the Batch Engine was reverted in 72d61ed3 and PRD #173
closed as not planned, so these terms name the ADR-0022 design, not running code.
"Lanes" in the **Active-Inference Reserve** are in-flight generations, not these._

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

**PARO Checkpoint**:
A checkpoint whose weights are quantized with pairwise-rotation INT4 (the
`paroquant` quantization method), whatever its base architecture — dense or
sparse-MoE, any Qwen generation. Recognized by its quantization method, never by
name or size.
_Avoid_: "PARO model" / "PARO family" as an architecture (it is a weight format
spanning architectures), "z-lab model" (the publisher, not the format), "AWQ
model" (PARO extends AWQ's layout with rotations; the two are not interchangeable).

**Prepared Checkpoint**:
The once-converted MLX-native form of a **PARO Checkpoint**, stored beside the
original so later loads skip the AutoAWQ conversion; rotation parameters remain
runtime state loaded verbatim — nothing semantic is baked into the artifact.
Stale or unreadable artifacts self-heal by re-conversion.
_Avoid_: prerotated cache (rotations are not pre-applied), weights cache / cache
(collides with the prefix cache), converted weights (holds the stacked MoE
layout too, not just per-tensor conversion).

**Rotated Ternary Checkpoint**:
A checkpoint whose language-model weights are ternary values carried in MLX affine
2-bit form in a Hadamard-rotated input basis, whatever its base architecture; the
runtime rotates activations before every packed matmul and un-rotates embedding
rows, so a loader that skips the rotation yields plausible garbage, not an error.
Recognized by its config's module manifest, never by name.
_Avoid_: "Bonsai model" as an architecture (Bonsai 2 27B is Qwen3.8-27B in this
format), "2-bit model" (the container width, not the weight format), "Hadamard
model", "ternary model" (the values, not the checkpoint).

**Base Architecture**:
The architecture whose forward pass a checkpoint runs — `base_model_type` when a
pack declares its own `model_type`, else `model_type` itself. The **Model
Identity** family facts key on it, so a **Rotated Ternary Checkpoint** of Qwen3.8
is a Qwen3.5-family model to everything downstream.
_Avoid_: "model type" for the family (the pack's type name is the loader key, not
the architecture), "base model" (the checkpoint it derives from, e.g. Qwen3.8-27B,
not the architecture).

### Speculative decoding

**Speculation**:
The drafters resident beside one model load, and the rules for engaging them: the
MTP head a Qwen3.5-family checkpoint ships, and the separate DFlash2 draft beside
Qwen3.8-27B. A load attaches what the Speculative Decoding setting allows and the
loaded target pairs with; the draft pairs with the text and the vision class alike,
which run one engine (ADR-0089). Residency (the Models page's draft row) is read
from it, and every request asks it for a **Speculation Plan** (ADR-0079).
_Avoid_: drafter support (the per-family facts it loads through); speculation
mode (the load-time setting); **Speculative Canonical Prefill** (an unrelated
background prefill).

**Speculation Plan**:
What one request runs speculatively, decided once from the request's facts (text-only
input, KV quantization and **KV Scheme**, temperature, prompt length, whether a prefix is restored,
which leaf the turn stores): the arm, the advance allowance its rounds add to the
turn's **Maximum Advance**, and where the app's prefill hands over to the iterator. The
**Server Completion** and the **Raw Generation Start** read the same plan; no plan
means ordinary decoding. DFlash2 speculates with images or without (ADR-0089): the
images are prefilled before the hand-over and the iterator rotates the text after
them by the **Position Anchor**'s rope delta; MTP needs text-only input. Under ADR-0087 (proposed) the Prefill Planner asks for it
once on the keyed path and the **Prefill Plan** carries its answer, the arm and split
in the **Decode Handover** and the allowance in the **Maximum Advance**; plan
application reads neither.
_Avoid_: engagement policy or predicate (the retired per-arm rules); speculative arm
(the plan's arm, not the plan).

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

**Working-Set Bound**:
The cap on measured memory headroom set by the process itself: the per-process
working set the GPU driver recommends, minus what the process already holds. The
headroom feeding the **Pressure-Reactive Budget**'s ceiling is the smaller of the
kernel's reclaimable pages and this bound, so the cache's own growth always
shrinks its ceiling — the kernel's buckets alone rise with the cache, because its
cold pages become "inactive".
_Avoid_: memory limit (nothing is refused — the bound caps the ceiling, and
eviction does the rest); footprint cap; static tax (it is measured every time).

**Active-Inference Reserve**:
The bytes withheld from the **Pressure-Reactive Budget**'s ceiling for in-flight
generations' KV working sets — the named replacement for the retired `/2`
divisor (ADR-0018), count-aware over active lanes and never below one lane. Priced
from observation, not constants: per lane, the largest leaf admitted, doubled only
while the most recent leaf store — on the partition the next lane runs on — was a
capture by copy (the `leafStore` source `live` or `boundary`; a `handoff` moved the
objects, a `copy` restore's source body is already counted in the tree), plus a
growth allowance — that leaf's
bytes per token times the turn's **Maximum Advance**, the quantity **Leaf Handoff**'s
check-out eligibility judges `isTrimmable(after:)` against (ADR-0064). The bootstrap constant stands until the first
observation and stands in for the growth of an unbounded turn. A pure value the
leaf admission feeds; the `budgetMeasure` event reports its inputs and per-lane
result.
_Avoid_: capture-copy factor (unqualified — it applies only after a capture by
copy, ADR-0064); `/2` divisor; working-set reserve; the bootstrap as a floor (it
retires at the first observation — #238).

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
The part of a request's **Cache Claim** that protects the path it restored from:
pinned into the **Budget Floor** at resolve and released when the claim concludes,
so no drain may evict the body a running generation depends on. Weak by reference:
a pin protects, it does not own. Nothing ages a pin out while its request runs.
_Avoid_: node lock, refcount; lease unqualified (the **Leaf Lease** is the strong
hold of a **Leaf Handoff**, which owns the body for the turn, and the **LLM Gate** is
arbitration; a pin is neither); **Cache Claim** for the pin alone (a pin is one part
of a claim); leaf pinning (pins hold restore *paths*, not the newest leaf; that
floor member is recency-defined).

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
The load-bearing prefix-cache tuning the manager owns as one mutable cell and
passes to the pure-function policies by value: `flopProfile` and `alpha` (what
eviction scores against), the measured `estimates` that denominate **Recovery
Cost** in seconds, and `pendingFullPayloadWait` (the bound on the
pending-full-payload wait below). `flopProfile` is fixed from **Model Identity**
at cache build; production `alpha` stays at the static LRU default (`0`), with
**AlphaTuner** disabled pending
[#504](https://github.com/spokvulcan/tesseract/issues/504);
`pendingFullPayloadWait` defaults to 500 ms.
_Avoid_: `EvictionPolicy.modelProfile` / `.alpha` (retired statics), eviction settings
(not a user **Setting**), model profile as a global, "the `(flopProfile, alpha)`
pair" (it has outgrown the pair). ("Flop profile" = the immutable
per-architecture cost model; `alpha` and the wait bound are the mutable halves.)

**Pending-Payload Wait**:
The bounded pause a request takes instead of copying a leaf it is about to own:
when its **Cache Claim**'s check-out is refused only because the leaf's full
payload still aliases the body, and the SSD writer reports that payload *in
progress*, the request waits for the writer to finish using the borrowed arrays
and then re-attempts the check-out. Bounded by the **Eviction Configuration**'s
`pendingFullPayloadWait` (500 ms); an `await`, never a blocking sleep, and never
across a model verb. A payload still *queued* behind other writes has no bounded
completion time and is not waited for — that request copies at once, and a
refusal the writer has already let go of is re-attempted rather than copied
on. The copy
reason keeps its name (`pendingFullPayload`) and gains the waited time
(`copyWaitMs` on `lookup` and `leafStore`, `restoreCopyWaitMs` on
`requestMemory`).
_Avoid_: retry/backoff (one bounded wait, then one re-attempt — not a retry
loop); blocking the writer (the wait observes the writer, it never holds it);
**Leaf Lease** deferral (the writer-side 500 ms recheck of ADR-0019, a different
timer on the other side of the same exclusion).

**Eviction Candidate Policy**:
The pure selection of "who is evicted next", shared shape on both tiers: the RAM
ladder (preferred-utility → global-utility → the residual oldest-first fallbacks,
**Budget Floor** members never victims) and the SSD terminal-loss ordering, each a
value-in/decision-out function of tier contents plus the **Eviction Configuration**.
The tier managers keep every effect — drain loop, demotion, `dropBody`, the ledger
lock; only the naming of the victim lives here.
_Avoid_: `findEvictionCandidate` / `terminalLossOrder`-on-the-ledger (the retired
private homes), eviction policy (unqualified — `EvictionPolicy` is the scorer it
composes, not the selection).

**AlphaTuner inversion**:
The retained, production-disabled dependency direction between tuner and cache:
an explicitly attached **AlphaTuner** takes a `flopProfile`, replays each grid-search candidate in its own
sandbox, and *returns* the winning `alpha` for the manager to assign — holding no
back-reference to the manager and writing no global. The inversion is that the manager
pulls the result, not that the tuner pushes it.
_Avoid_: writing a global alpha, tuner→manager callbacks or weak back-references.

### Overlay presentation

**Overlay Placement**:
Where a floating panel's fixed canvas sits for a given **Screen Geometry**: a
state-free pure value, so the canvas never moves or resizes with what it shows.
The Companion's voice overlay concepts bring one along; the **Lens** places its own
glass panel, and the pill's placement went with the pill.
_Avoid_: layout strategy, frame provider, per-state frames or resize-animation
flags (retired — the canvas is fixed), overlay style (the retired
pill-vs-border user **Setting**).

**Overlay Feed**:
The one surface of dictation signals the **Lens** renders from: typed lifecycle
phases, typed errors, terminal outcome beats carrying the committed text, the
**Live Preview** while recording, the take's app, whether ⇧ held the take, and the
audio meter (level + spectrum). The dictation coordinator writes everything but the
meter, which the capture engine's meter stream drives. The Lens follows the feed
itself, so the dictation pipeline never learns what draws it.
_Avoid_: overlay state / OverlayState (retired push-model object), view model,
pre-flattened error strings, per-variant state surfaces (retired with the Overlay
Variants).

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

### Phone (ADR-0066, ADR-0084)

**Library**:
Every text the owner has added on the phone, newest first, each with its own
**Bookmark** and the language it is read in. The Mac's **Reader** holds one
text; the phone's Reader opens one text of the Library at a time, and keeps
reading it while the owner browses the others.
_Avoid_: documents, reading list, queue (nothing plays one text after another).

**Library Inbox**:
Where the share sheet's "Read in Tesseract" leaves a text for the **Library**:
a folder the app and its extension share. The extension only drops texts
there; the app takes them in, and opens the newest, when it next comes to the
front.
_Avoid_: share queue, pending imports.

**Thermal Policy**:
The rule from the phone's thermal state to what the voice does: nominal or
fair, the neural voice reads; serious, the **System Voice** reads from the
next segment until the phone is back to fair; critical, reading stops. The
Reader says why each time. It changes the voice, not the pacing: reading
ahead in bursts or just in time costs the same per second of audio.
_Avoid_: heat throttling, cooldown mode.

**System Voice**:
The phone's own speech synthesizer, reading where the neural voice can't:
while that voice downloads or is prepared, on a phone too slow for it, and
while the phone cools down. It reads in the text's language and is never a
voice identity: the chosen **Preset Voice** stays chosen while it reads.
_Avoid_: fallback voice, backup voice, Apple voice.

**Voice Preparation**:
Turning the downloaded checkpoint into the Neural Engine graphs this phone
runs: building and compiling them, checking them against the checkpoint, and
the **Speed Check**. The first one takes minutes; it runs again only when the
graphs or the checkpoint change, and every later launch only loads what it
built, in seconds. The **System Voice** reads meanwhile.
_Avoid_: installation, compilation, setup.

**Speed Check**:
The timed render at the end of **Voice Preparation** that says whether this
phone's neural voice keeps up. A phone too slow for normal speed reads with the
**System Voice** and says why; one that keeps up at some speeds but not all
offers only those in the speed menu.
_Avoid_: benchmark, performance test.

**Device Tier**:
The phone's sizing class, read once at launch from physical memory. It is the
one answer to every "how much fits on this phone" question: context window,
output cap, compaction preset, prefix-cache budgets, buffer-cache limit, the
catalog entries offered, and the free memory the neural voice needs before it
loads. Below the floor the tier is *unsupported*. It arrives with the chat model
in release 2; read-aloud alone has nothing to size (ADR-0084).
_Avoid_: RAM tier (the prefix cache's in-memory tier), device profile, memory
class; per-setting device checks (the tier answers them all, once).

### App composition

**App Bindings**:
The module owning the app's launch sequence and every long-lived runtime subscription
that carries a rule (model auto-load and hot-swap, lazy-reload guards, server
reactions, hotkey rebinding, the single dictation-state fan-out) — the
launch-time mirror of the teardown-owning termination coordinator. Distinct from the
composition root, which stays pure wiring with no behaviour.
_Avoid_: app glue (pre-carve working name), setup() behaviour, launch coordinator,
app services, a SwiftUI `Binding` (view data flow, unrelated).

**Settled Width**:
The width a page lays out against — the detail column's width once the column
has stopped moving. While the sidebar slides, every page keeps its previous
settled width and reflows once when the column settles, so heavy content is
never re-measured per animation frame. A window resize is not column motion:
pages follow it live.
_Avoid_: frozen width (the pin is the mechanism, not the concept); debounced
width (the release is the slide ending, not a delay); snapshot / rasterized
transition (nothing is captured — the page is laid out, just not per frame).

### Release and distribution

**Release PR**:
The rolling pull request that automation keeps open against `main`, holding the
next semantic version and its accumulated changelog. Merging it *is* the release
decision — the tag, the GitHub Release, and the signed build all follow
mechanically from that one merge.
_Avoid_: version-bump PR, release branch (no such branch exists), draft release
(that is the GitHub Release before the **Release Pipeline** publishes it).

**Release Pipeline**:
The automated path from a merged **Release PR** to a downloadable, notarized
disk image attached to the GitHub Release — gated on the released commit's CI
being green, with no human step inside it. The GitHub Release stays a draft
until the image is attached; publishing it is the pipeline's last step.
_Avoid_: deploy (nothing is deployed to a server), publish flow, the CI workflow
that builds pull requests (a gate the pipeline consumes, not the pipeline).
