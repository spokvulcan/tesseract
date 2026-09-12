---
status: accepted
---

# Recoverable eviction and the leaf home guarantee — terminal drops become a bug class

A long OpenCode session against a 35B model surfaced the failure this ADR
kills: with the budget collapsed by memory pressure, the just-finished turn's
leaf was evicted *by its own admission* while older `.system` bodies survived
above budget, and the SSD copy was silently rejected too — after the previous
turn's backing had already been deleted. The conversation's newest state ended
up on neither tier. Code audit (2026-07-04) traced four compounding causes:
admission drains ran with the Budget Floor disabled; `.system` and multi-child
bodies were excluded from eviction eligibility regardless of budget; the
end-of-turn write was subject to the incidental `min(4 GiB, physRAM/16)`
pending-queue cap; and supersession deleted the old backing *before* the new
write was enqueued. Decided in the 2026-07-04 grilling.

## Decision

- **Leaf Home Guarantee.** The newest end-of-turn leaf always has a home: RAM
  if the current budget holds it, otherwise a mandatory SSD write. No
  incidental cap, survival gate, or eviction pass may reject that write; the
  enqueue always precedes deletion of the backing it supersedes
  (enqueue-before-delete). Oversized payloads stream to disk rather than being
  rejected for exceeding a RAM-sized pending buffer.

- **Recoverable Eviction.** A RAM body may be dropped only if its bytes are
  SSD-recoverable (a backing exists, or demotion succeeds first). Terminal
  drops are legal only for explicit invalidation (model change, user clear),
  disk-full, or I/O error — anything else is a defect, surfaced by
  diagnostics, never a silent policy outcome. The survival-gate veto of a
  demotion is removed for this class.

- **No type-based RAM shielding.** `.system` and multi-child bodies lose their
  eviction immunity; one recovery-cost policy (ADR-0011) prices every body.
  `.system` protection moves to where loss is actually expensive — the SSD
  ledger's type-protected cut, which already has it. A demoted system body
  costs ~0.2–1.5 s of hydration on the next cold conversation; a shielded one
  cost the newest leaf its life. The Budget Floor shrinks to the in-flight
  requests' restore paths plus the single most-recently-extended leaf, and is
  honored on *every* drain — admission included.

- **SSD write eagerness is adaptive; the guarantee write is not.** When RAM
  comfortably holds a snapshot, its SSD copy is redundancy and may be
  deferred or coalesced; when RAM cannot, the write is mandatory (above).
  Write-rate and bytes-written counters persist from day one and feed the
  user-facing cache panel; no hard write-rate throttle ships until field
  counters justify one (measured arithmetic: ~100–150 GB/day of suffix writes
  ≈ a decade of consumer-SSD endurance), and no user-facing "SSD protection"
  knob ever does.

## Consequences

- Partially supersedes ADR-0011's Budget Floor definition (`.system` chains
  leave the floor) and its "terminal drop is the fallback" demotion language;
  everything else in ADR-0011 stands.
- Eviction is never data loss, only a latency tax — which is what makes a
  near-zero RAM budget (35B models on 48 GiB machines, ADR-0018) survivable:
  the cache degrades to SSD-served (measured hydration 0.65–0.87 GB/s vs
  ~370 tok/s re-prefill ≈ 50× cheaper) instead of failing.

## Amendment 2026-09-06 — Deferred Payload Extraction

A **Snapshot Demotion** used to copy the victim's arrays to host memory on
the MainActor, inside the admission that triggered the eviction (the "full
KV copy" of `extractSnapshotPayload`). On 2026-09-06 a 27B model with 3 GB
leaves demoted two or three victims per admission while the machine was
swapping: four admissions held the MainActor 24–50 s each, and the leaf
path's own extraction added 2–3.5 s to the client-visible tail of every
large turn.

A `SnapshotPayload` now owes its bytes instead of carrying them. The
extraction edge fixes the byte total (and, for a **Leaf Extension
Admission**, slices and evaluates the suffix on the Metal-affine caller); the
SSD writer's task materializes the host copy right before the file write and
logs it as `event=ssdPayloadMaterialize`. Nothing on the MainActor or the
inference thread copies KV bytes any more. Until the writer gets to it, the
payload keeps the arrays alive — a leaf shares them with its RAM body, a
demotion victim's live on only there — and the materializer releases each
layer as it copies it, so a demotion never doubles in RAM. The front door's
`maxPendingBytes` accounting is unchanged: it always counted the byte total,
which the payload still knows up front.

## Amendment 2026-09-06 (proposed with ADR-0064) — Leaf Lease, and extension payloads that retain no body array

Takes effect when ADR-0064 is Accepted
([issue #471](https://github.com/spokvulcan/tesseract/issues/471)).

Implementation staging: #479 establishes the tree-side protection below,
with small real-cache ownership and lifecycle tests. Production check-out
and the recurrent-state rewind remain #480 work; ADR-0064 stays Proposed
until that integration and its acceptance gates are complete.

The **Restore Pin** above is the weak claim of a request that restored *by
copy*: it protects a restore path in the Budget Floor, it owns nothing, and
the pin table's age-out backstop may end it. ADR-0064 adds a second, strong
claim beside it. A request that takes a leaf's cache objects by **Leaf
Handoff** holds a **Leaf Lease** on that body from check-out to check-in:
while it holds, no eviction drain, **Snapshot Demotion**, RAM-tier clear,
write-eagerness promotion, replacement admission or SSD-writer materialization
may touch the body. This ownership exclusion also applies to mandatory SSD
admission: the caller must first check in or rewind the lease, then submit
the finished leaf through normal admission. An out-of-order admission is
refused with diagnostics (`StoreDiagnostics.leaseRefusals` identifies the
blocking leases) and must be retried after return; it is not a
completed end-of-turn admission. The Leaf Home Guarantee and its cap bypass
apply once ownership has returned, without weakening enqueue-before-delete.
The lease ends only at check-in or **Leaf Rewind**, never by age-out; the
backstop stays for pins and is exempt for leases. Leased bytes stay counted
in the tree total, and check-in reconciles the growth. Budget Floor
membership keeps its meaning: a leased body is a floor member for the turn,
and if pressure cannot be relieved because the only candidate is leased the
outcome is what the floor gives today.

The Deferred Payload Extraction amendment above says a deferred payload
"keeps the arrays alive" and that "a leaf shares them with its RAM body".
That sharing narrows to full payloads. An extension payload detaches every
array it retains, the recurrent layers' state as well as the attention
suffix slices, deep-copied and evaluated on the Metal-affine caller, so a
pending suffix payload never references a body a generation may own. A full
payload still aliases the attention body. Under ADR-0064, a pending full
payload makes move checkout ineligible: #480 must fall back to copy restore
until materialization releases the body arrays. The tree-side gate below
is a second exclusion boundary, not the production checkout eligibility
decision: it can protect a queued payload with a lease but refuses lease
acquisition while the writer is already reading.

### Tree-side implementation (#479)

The lease token and the writer's shared access state contain only scalars;
acquisition keeps the existing body in place and adds no cache copy. Tree
byte/count accounting includes that body once throughout the lease. An
explicit quiescent return reconciles the supplied body's bytes and, for an
extended path, moves the tree entry to the returned offset. Invalid or stale
returns leave the current lease intact. `completeRequest` releases Restore
Pins only; it cannot establish that a mutable body has been rewound.

A full deferred payload and its node share an exclusion state. Before
removing the payload from the queue, the writer atomically claims read
access. If a lease holds, the item stays queued and charged; flush and the
write-eagerness timeout cannot override it. If the writer wins, lease
acquisition fails until materialization releases the body arrays. File I/O
then uses independent host bytes. Unrelated writes can proceed, but an
extension never overtakes a queued base. Detached extensions and already
materialized payloads need no body read claim.

A lease-blocked flush waits for the existing 500 ms writer recheck; return
does not retain or invoke a writer callback. Ordinary drains keep their
flush wake/resume behavior. The bounded delay is a scheduling cost, not an
extension of the lease after return.

When check-in advances the leaf to another node, its exclusion state follows
the returned body. A pending full payload may retain prefix views of backing
reused by the grown body; a later lease must still exclude that writer. The
vacated node receives a fresh exclusion state. Returns also validate tree
membership before changing topology or accounting.

`leafLeaseBegin`, `leafLeaseEnd`, `leafLeaseRefused`, and
`leafLeaseDeferred` carry request and lease identity, leaf offset and bytes;
end events record release reason and reconciled growth. `requestMemory`
tree facts also carry lease bytes/count. These are logical ownership
counters, not additional physical allocations. The #480 checkout must
still refuse move checkout for pending full payloads (falling back to copy)
and perform the actual object transfer and recurrent-state rewind inside the
Model Session. It must return the lease before end-of-turn admission.
