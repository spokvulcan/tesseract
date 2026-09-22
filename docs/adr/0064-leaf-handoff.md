# ADR-0064: Leaf Handoff — the finished turn's live cache is the leaf, moved between exactly one owner and the next

- Status: Accepted (as built through #480; large-model memory and latency
  acceptance remain open, as recorded below)
- Date: 2026-09-06
- Relates to: ADR-0023 (its rejection of copy-on-write restore stands; this
  ADR explains why a move is not that alias), ADR-0019 (the Restore Pin is
  amended by the Leaf Lease; its Deferred Payload Extraction amendment is
  narrowed), ADR-0018 (the working-set bound the freed memory feeds),
  ADR-0015 (Metal affinity), ADR-0016 (the Model Session seam every transfer
  happens inside), ADR-0006 (the app-owned live cache), ADR-0009 (the
  speculative seed keeps restoring by copy), ADR-0062 (superseded together
  with ADR-0063), ADR-0063 (the Emitted Path the moved leaf is keyed under)

## Context

Every turn copies the conversation's KV state twice: once when a hit is
restored into a live cache, once when the finished turn is captured as the
new leaf; and the previous leaf stays resident until the new one is admitted.
A turn therefore holds the conversation's KV up to three times. On a 48 GB
machine the 93k-token Pi session of 2026-09-06 at 16:32 had 15.1 GB of
weights, a 6.2 GB leaf, 35.5 GB of active MLX memory and a 50.2 GB peak; the
RAM-tier budget had been zero since 16:15. Capture of a live leaf, 50 to
100 ms at 46k tokens when unpressured, took 2 to 3 s per turn from 16:26,
and the warm-turn prefill of about 30 new tokens went from 0.6 s to 2 to
3 s. The machine was swapping on its own cache.

The hard constraint is the Metal invalid-resource precedent behind ADR-0023:
aliasing one KV buffer between two owners, one of them a running generation,
produced the `InvalidResource` abort that the deep-copy capture and restore
were introduced to end. ADR-0023 rejected copy-on-write restore for that
reason, and the rejection stands. The vendor's shallow `copy()` helpers on
the cache classes alias arrays and are not usable for any of this.

Two facts about the current code shape the design. Today's Deferred Payload
Extraction (ADR-0019 amendment) deep-copies and evaluates only the
suffix-sliceable attention layers of a **Leaf Extension Admission** and keeps
references to every other layer's arrays, so on a hybrid model an extension
payload references the body's recurrent state until the writer materializes
it; harmless while the body is an immutable snapshot, a hazard once the body
is the next turn's live cache. And the vendor's `isTrimmable` is not enough
to promise a rewind: a rotating attention cache is trimmable now and still
overwrites old entries once generation crosses its window, and trimming its
offset afterwards cannot bring them back; the vendor exposes
`isTrimmable(after:)` for exactly this distinction.

## Decision

1. **Move, never alias.** At any instant the cache objects of a conversation
   have exactly one owner, the radix tree or the running generation. The
   finished turn's live cache array becomes the leaf body as it is, at the
   quiescent point after the completion drive has awaited the generation,
   inside the **Model Session**; no capture copy. A request whose **Cache Key
   Path** extends a resident leaf's full path takes the leaf's cache objects
   as its live cache; no restore copy. The objects themselves move, not
   their arrays, so neither side rebuilds a snapshot. Every transfer happens
   at a provable quiescent point; nothing is shared between two owners at
   any time, which is what separates a move from the alias ADR-0023
   rejected.

2. **Check-out eligibility.** A hit is taken by move only when all hold:
   the snapshot is a leaf hit at its full offset (never a `.system`
   checkpoint, a branch point or a chain-prefix restore point, which are
   shared restore points by design and keep restoring by copy); no pending
   SSD payload of any kind retains an array of its body (after decision 5
   only a full payload can); every attention layer is a full-attention cache
   whose retained prefix later writes cannot overwrite, and reports
   `isTrimmable(after:)` for the turn's maximum advance (the new prompt
   tokens plus the request's output ceiling plus the speculative
   draft-and-bonus allowance); every recurrent layer's state can be saved as
   an independent copy; the KV is unquantized; the key space is the identity.
   A rotating or sliding-window cache never hands off. This app never
   configures a bounded KV plan, so rotating layers arise only from
   sliding-window architectures; they restore by copy. Otherwise the hit
   restores by copy exactly as today, with the reason recorded:
   `checkpoint`, `immutableBody`, `pendingFullPayload`, `untrimmable`, `rotating`,
   `quantized`, `imageKeySpace`. (Since #523, `pendingFullPayload` is the one
   refusal that clears itself, so it earns a bounded wait before the copy —
   see the 2026-09-19 as-built note.)

3. **Leaf Lease.** From check-out to check-in the tree may not drop the body,
   demote it, clear the RAM tier of it, promote it into an SSD write, or let
   the SSD writer materialize from it. The lease ends only at check-in or
   rewind, never by the pin table's age-out backstop; the backstop stays for
   Restore Pins and is exempt for leases. If memory pressure cannot be
   relieved because the only candidate is leased, the outcome is what the
   **Budget Floor** gives today.

4. **Leaf Rewind.** At check-out the server saves each recurrent layer's
   state as an independent deep copy, evaluated into private backings (never
   the array handles the vendor's own speculative checkpoint aliases),
   together with the layer's metadata (offset, lengths, padding) and the
   leaf offset; the state is small per layer, so the copy is cheap. If the
   turn is cancelled, fails, or is intervened, attention layers are trimmed
   back to the leaf offset, the saved recurrent state and metadata are
   restored exactly, and the leaf is checked back in unchanged. The next
   request continues from it.

5. **Check-in.** The extended cache becomes the new leaf body; the superseded
   entry has no body to drop because it was moved out. SSD admission changes
   in one place: the extractor detaches every array an extension payload
   retains, the attention suffix slices as today and the recurrent layers'
   state as well, all deep-copied and evaluated on the Metal thread before
   the tail returns, so a suffix payload never references the body and
   decision 1 holds literally. A full payload (the first leaf of a
   conversation, or a degraded extension) still aliases the attention body,
   because copying it is what Deferred Payload Extraction exists to avoid.
   A full-format view payload is the exception: #526 detaches its leading
   attention rows and whole-state layers before enqueue, exactly like an
   extension payload, so it cannot block the Backing Leaf's next check-out.
   For an ordinary full payload, decision 2 makes the next check-out copy until
   the writer has materialized it — after a bounded wait for that materialization
   (#523; see the 2026-09-19 as-built note).

6. **Accounting.** A leased leaf's bytes stay counted in the tree total;
   check-in reconciles the growth. The `ActiveInferenceReserve` capture-copy
   factor is left as it is in this change and re-priced from measurements
   afterwards; the real memory drops regardless. (Re-priced in #522 — see
   ADR-0018's 2026-09-19 amendment: one leaf plus growth, doubled only after
   a capture by copy.)

7. **Scope.** The handoff applies to text-only identity key spaces with
   unquantized KV, the regular configuration and the only one speculative
   decoding runs on. Quantized-KV partitions, image-bearing requests, the MTP
   path and the Batch Engine keep today's copy paths.

8. **Telemetry.** The `leafStore` event gains the leaf source (`live`,
   `handoff`, `copy`, `rewind`) and the copy reason when a check-out fell
   back; the `lookup` event gains the restore mode; new events for lease
   begin and end and for rewind.

### As built — 2026-09-12

The implementation landed in stages: detached extension payloads (#474),
capture by move (#478), tree-side leases (#479), then checkout and rewind
(#480). `LeafCheckout` runs in the Model Session. It transfers the movable
resident leaf body out of `HybridCacheSnapshot` and the tree under the lease;
all retained snapshot views lose their arrays. The tree keeps the original
logical byte charge and floor membership until return. Immutable captured or
hydrated leaf bodies continue to restore by copy and report `immutableBody`;
structural checkpoint/branch/chain-prefix hits report `checkpoint`. A completed
eligible turn produces a movable leaf for its next extension.

Eligibility checks topology, full-offset extension, identity key space,
unquantized full-attention caches, and `isTrimmable(after:)` with the suffix,
output ceiling and DFlash2 allowance. The currently accepted plain
`KVCacheSimple` always permits that advance; this is a conservative capability
contract, not a measured bound or support for rotating/windowed caches. Those
remain excluded. Weak full-payload materialization probes
reject checkout without retaining payloads or host data; detached extensions
do not block it. Recurrent rewind copies include state-slot metadata,
lengths and padding. Rewind trims the same attention objects and reconstructs
recurrent layers from that saved independent state after the generation has
quiesced. The request relinquishes all cache references on return.

Startup cancellation, decode cancellation/failure, and boundary intervention
return the original leaf before releasing request pins. Session providers
retain their container and expose `async rethrows` entry: nonthrowing cache
cleanup cannot independently fail on entry, even after cancellation. Unload
drains starts/completions before dropping its container reference. A checked-out
direct (non-thinking) turn rejected by a structural live-path guard concludes with
the original leaf rewound; it does not stamp uncertain current state with a
canonical path or try to capture the already-returned cache. Successful check-in
is the ownership commit point; normal RAM/SSD admission follows the return.
Cancellation observed before that point rewinds. Cancellation after the return
cannot undo the committed leaf and follows ordinary admission/cleanup rules.
Quantized copy paths retain the post-quantization cache array, because that
step replaces attention objects before decode.

Preserve-thinking, text-only identity requests no longer capture unused
last-message/last-user transient boundary helpers. Planned checkpoints,
image-bearing boundaries, think-stripping paths and ADR-0009 seeds remain.
`lookup`, `leafStore` and `requestMemory` report actual handoff/copy mode and
fallback reasons; rewind emits both `leafRewind` and `leafStore source=rewind`.
Request memory includes recurrent backup bytes and lease lifecycle facts.
Rewind trimming executes independently of assertion settings; zero-layer
caches cannot be captured or admitted as completed leaves. Recurrent lengths
and absent/present padding are also restored correctly on the copy path.

Small-cache identity, growth, recurrent replacement/metadata, pending-payload,
fallback, pressure, cancellation/resend and release tests pass. In 24 bounded
success/cancel/failure cycles, checkout allocated only the 64-byte recurrent
backup for a 2 MiB attention body; copied restore allocated 2,097,232 bytes.
This proves the ownership mechanism, not production footprint. Trimming may
retain attention allocation capacity above the logical offset; #501 tracks
that lifetime. Large-model parity, peak/retained footprint and tail latency
remain unmeasured here. The owner constraint prohibits automatically repeating
the prior crash workload on this 48 GiB Mac. The [evidence report](../../benchmarks/leaf-checkout/2026-09-12/README.md)
records baseline comparisons and a bounded plan awaiting approval.
`ActiveInferenceReserve` remains unchanged.

### As built — 2026-09-19: the bounded pending-payload wait (#523)

Decision 5's copy is now a last resort rather than a first response. When
`claimLeaf` refuses only because the leaf's full payload still aliases the
body, the request asks the SSD writer where that payload stands. The writer
answers from its own queue lock — `inProgress` for the item it has popped and
is materializing or writing, `queued` for one still behind others, `absent`
otherwise — and the **Prefix Cache Manager** exposes that answer to the
completion path, gated on the node's own probe so a `absent` answer means the
next check-out is not refused on the writer's account at all.

Only `inProgress` is waited for: the writer's materialize step releases the
body arrays, and on a conversation's second turn that is milliseconds away. A
`queued` payload has no bounded completion time, so that request copies at
once, exactly as it did before. An `absent` answer means the refusal was
already stale — the writer let go between the refusal and the question — and
the check-out is re-attempted instead of copied on. The bound is an **Eviction Configuration**
value, `pendingFullPayloadWait`, defaulting to 500 ms; zero restores the
pre-#523 behavior. The wait is an `await` that polls the writer's answer every
5 ms — no Metal work runs, no thread is held, and no Model Session verb has
touched the cache yet — and cancellation settles it as a copy immediately.
After the wait the check-out is re-attempted once.

The copy reason keeps its name. What the wait cost rides beside it:
`copyWaitMs` on the `lookup` and `leafStore` events, `restoreCopyWaitMs` on the
`restored` phase of `requestMemory`. A handoff that only happened because the
request waited reports the same field, so the wait's payoff and its cost read
off one trace.

The wait sits at the check-out decision inside `makeHTTPPrefixCacheGeneration`,
which the request has already entered the Model Session to reach: the decision
depends on the keying phase's non-`Sendable` products, so it cannot be hoisted
above `withSession` without a second session entry and a re-key. It therefore
holds the session's serial-access mutex across the suspension. Under
`arbiter.withExclusiveGPU(.llm)` the request already owns the GPU for the whole
turn, so nothing else could enter the session meanwhile; the SSD writer the
wait is watching runs on its own detached task and is not blocked by either.

### Amendments this ADR makes

- **ADR-0019, Restore Pin.** A pin remains the weak claim of a request that
  restored *by copy*; it protects a restore path, it does not own it. The
  Leaf Lease is the strong claim of a request that took the leaf's objects
  by move: it owns the body for the turn, ends only at check-in or rewind,
  and is exempt from the pin table's age-out. Budget Floor membership is
  unchanged in meaning; a leased body is a floor member for the turn.
- **ADR-0019, Deferred Payload Extraction amendment.** Its sentence "a leaf
  shares them with its RAM body" narrows to full payloads. An extension
  payload retains no body array.
- **ADR-0023.** The rejection of copy-on-write restore stands as written.
  That design aliased one buffer between two owners and copied on write; a
  move has one owner, transfers only at quiescent points, and the lease
  blocks every tree-side release for the duration. It is also not the
  refcounted KV Pages of ADR-0023's own decision, which remain the answer for
  batch lanes sharing a prefix; a sequential coding-agent session has one
  reader of the leaf at a time, and a move gives it the single-owner memory
  shape without a page kernel.

## Consequences

- One KV instance lives in memory per conversation instead of two or three.
  The capture stage of the post-EOS tail goes to near zero, and the wait
  after the last token is bounded by the suffix written to SSD.
- The cancel path changes shape: today a cancelled generation's live cache is
  discarded and no leaf is stored; under the handoff it is rewound and
  checked back in, so stop-and-resend keeps the conversation's cache.
- Every tree-side release path (eviction to fit the budget, demotion before
  drop, write-eagerness promotion, RAM-tier clear, the body-drop chokepoint,
  the SSD writer) gains a refusal for a leased body, tested by physical
  address the way the ADR-0062-era snapshot tests already work.
- The vendor's cache `copy()` helpers and its speculative-checkpoint aliasing
  are never used for ownership transfer or rewind.
- A rotating or sliding-window attention model never hands off; it keeps the
  copy paths and reports why.
- The `ActiveInferenceReserve` factor was over-priced until re-priced on one
  leaf plus growth in #522 (follow-up from #471; ADR-0018 amendment).

## Rejected alternatives

- **Copy-on-write restore.** ADR-0023's rejection; the alias it introduces is
  the abort this ADR is built to never reintroduce.
- **Refcounted KV Pages for the sequential path.** Kernels, page alignment
  to quantization groups and a refcount discipline, to solve a problem the
  coding-agent workload does not have: one reader at a time. Pages stay the
  batch-lane answer.
- **Removing only one of the two copies.** Each copy is a full KV memcpy and
  a doubled residency; keeping either keeps the swap on the 48 GB machine.
- **Rewind by restoring a saved copy of the whole leaf.** That copy is the
  restore copy being removed. Saving only the recurrent state, small per
  layer, plus trimming the attention layers is exact and cheap.
- **Trusting `isTrimmable` at check-out.** Trimmable now is not rewindable
  after the turn for a rotating cache; eligibility uses
  `isTrimmable(after:)` with the turn's maximum advance and excludes rotating
  caches outright.
- **Relying on MLX value semantics for pending payloads.** A pending payload
  holding an array handle might be memory-safe under copy-on-write
  allocation, but the one-owner rule must hold literally and be testable by
  address; detaching the recurrent state costs kilobytes.
- **Ending a lease by age-out.** The backstop exists for leaked pins of
  copy restores; ending an owner's claim while it decodes would free a buffer
  under a running generation.

## Amendment — 2026-09-19: Warm Body restore (#527)

A Warm Body is an explicit restore-by-copy path with `copyReason=warmBody`.
Vendor dequantization produces fresh live attention arrays in the Model Session;
whole-state arrays are independently copied. A Warm Body never enters Leaf
Handoff. Leaf Lease, Leaf Rewind and the single-owner rule are unchanged.

## Amendment 2026-09-19 — Borrowed SSD bytes (#469)

A prepared full payload can still alias the tree body: the no-copy writer holds
array-backed Data until the layer has been written. The pending-payload probe
tracks borrowed array ownership, not merely whether Data views exist. The
writer's body read claim spans the write and every error exit. Pending-Payload
Wait keeps its same bound and queued/in-progress rules; an in-progress wait now
covers writing the borrowed bytes as well as preparing their views.

## Amendment 2026-09-20 — the boundary leaf is moved, not copied (#501)

Leaf Handoff was introduced for the live path: the finished turn's own cache
becomes the leaf by a move. The boundary path kept a deep copy, and the
2026-09-20 session logs showed that copy to be the process's memory peak on
every think-stripping turn.

The boundary executor restores its boundary snapshot, re-prefills the
canonical residual, and captures. That restored cache is request-private by
construction: `HybridCacheSnapshot.restore` deep-copies and evaluates every
layer out of the tree — a Prefix-View Checkpoint's slices of its Backing Leaf
included — and only the residual prefill writes into it. No tree body, no
other request and no pending payload can reach it. The one-owner rule
therefore permits the move, and the executor now takes it, emptying its own
reference so the `FinalGenerationCache` it hands to the admission is the only
one.

`HybridCacheSnapshot.canCaptureMoving` asks `captureMoving`'s own guards
before the executor commits. A quantized partition answers `false` and keeps
the deep copy, which supports the `QuantizedKVCache` layers a move cannot;
without the pre-check a refused move would return `nil` and drop the leaf.

Three costs go with the copy, each one leaf at the turn's context:

- the capture's transient, which was the peak sample of every boundary
  request in the session logs;
- the next turn's restore, which fell back to `copy` with reason
  `immutableBody` because the boundary capture left a `.copied` body where
  check-out needs a `.moved` one;
- the Active-Inference Reserve's lane doubling, which applies while the most
  recent leaf store captured by copy.

A moved boundary capture reports `source=handoff`, so the reserve reads one
leaf plus growth. The `leafStore` event's `path` still reads `boundary`: the
two fields together say where the leaf came from and how it was taken. The
`Source` enum's `.boundary` case now means specifically a boundary capture
that copied.

Leaf Lease, Leaf Rewind, the Restore Pin and the boundary path's own
sequencing are unchanged. The backing leaf the transient views resolve
through is still released once the canonical leaf is admitted (#551).

Measured on one growing conversation under a think-stripping render: the
boundary capture's allocation goes to zero, peak active MLX at 25.7k prompt
tokens falls 1.4 GB, and the restores that were refused for `immutableBody`
stop being refused. The saving is one leaf per boundary turn and scales with
the leaf — the session logs measured the same copy at 4.45 GB at 69k tokens.
`hybrid-cache-correctness` and `prefix-cache-e2e` both pass on the same
model. The session audit, the A/B, its driver and its limits are in
[`benchmarks/boundary-leaf-move/2026-09-20/`](../../benchmarks/boundary-leaf-move/2026-09-20/README.md).

## Amendment 2026-09-22: a leaf loaded from SSD is handed off (#554)

Before this amendment only a resident moved body could be checked out. An SSD
hit decoded the leaf's chain into fresh MLX arrays and admitted them as a
copied body; the check-out refused it with `immutableBody`, and the restore
deep-copied it into the live cache. For the whole turn the process held two
leaves: the tree's body, pinned, and the request's copy.

SSD hydration now gives the tree a moved body. `loadSync` decodes the chain as
before, then builds the leaf's cache objects directly on the arrays it just
loaded, evaluates any chain layer it joined lazily (once, inside the Model
Session, where hydration already runs), and boxes the objects as the node's
moved body. The request that hydrated the leaf takes it by Leaf Handoff under
the same guards a captured leaf faces.

The one-owner rule holds for the loaded arrays:

- every array is allocated fresh and copied out of its file mapping, and the
  mappings are released when the load returns, so no file view or read cache
  shares them;
- a node that is already on SSD enqueues no SSD write, so no pending full
  payload can alias the body;
- the moved box empties every value copy of the snapshot when the check-out
  takes the objects, so a retained lookup result cannot reach them afterwards.

Only a full leaf loaded by `loadSync` becomes a moved body. A Chain-Prefix
Restore, system and branch-point checkpoints, warm bodies, quantized
partitions, and any layer a move cannot take keep the copied body and copy as
before, with their reasons.

Leaf Rewind and check-in treat an SSD-loaded leaf like any other leased leaf.
Taking the body leaves the node with its SSD ref, and returning the leaf
restores the body beside that ref, so the next extension still chains from it.
An SSD-loaded handoff reports `restoreMode=handoff` and `source=handoff`; the
`lookup` event's existing `hydratedFromSSD` flag tells it apart from a RAM
handoff.

The saving is the restore copy and the second resident leaf: one leaf per SSD
hit, about 4.1 GB at 61k tokens on the 27B. The loaded attention arrays have
exactly the offset's capacity, so the first suffix prefill still grows the body
once, as a copied restore did. The model-free evidence in
`CacheClaimMemoryEvidenceTests` measures the check-out of a loaded 8 MiB hybrid
leaf at the 4 MiB recurrent rewind backup and nothing else, where restoring the
same leaf by copy costs the whole 8 MiB.
