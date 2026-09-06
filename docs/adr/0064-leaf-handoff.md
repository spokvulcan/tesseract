# ADR-0064: Leaf Handoff — the finished turn's live cache is the leaf, moved between exactly one owner and the next

- Status: Proposed (documents-first, ticket #472 of
  [issue #471](https://github.com/spokvulcan/tesseract/issues/471); flips
  to Accepted with as-built notes at ticket #480)
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
   `checkpoint`, `pendingFullPayload`, `untrimmable`, `rotating`,
   `quantized`, `imageKeySpace`.

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
   because copying it is what Deferred Payload Extraction exists to avoid;
   that is why decision 2 makes the next check-out copy until the writer has
   materialized it.

6. **Accounting.** A leased leaf's bytes stay counted in the tree total;
   check-in reconciles the growth. The `ActiveInferenceReserve` capture-copy
   factor is left as it is in this change and re-priced from measurements
   afterwards; the real memory drops regardless.

7. **Scope.** The handoff applies to text-only identity key spaces with
   unquantized KV, the regular configuration and the only one speculative
   decoding runs on. Quantized-KV partitions, image-bearing requests, the MTP
   path and the Batch Engine keep today's copy paths.

8. **Telemetry.** The `leafStore` event gains the leaf source (`live`,
   `handoff`, `copy`, `rewind`) and the copy reason when a check-out fell
   back; the `lookup` event gains the restore mode; new events for lease
   begin and end and for rewind.

The change lands in four steps so each ownership boundary is reviewed on its
own: the extractor detaches every retained array (safe on today's copy
paths); capture by move while the next request still restores by copy; the
Leaf Lease as a tree-side state every release path honours, proven by tests
before anything checks a leaf out; then check-out by move with the rewind.

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
- The `ActiveInferenceReserve` factor is over-priced until it is re-measured
  (follow-up from #471).

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
