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
