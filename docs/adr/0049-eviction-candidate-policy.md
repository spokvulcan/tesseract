# ADR-0049: Eviction candidate selection is a pure policy — on both tiers

- Status: Accepted
- Date: 2026-07-19
- Relates to: ADR-0011 (recovery-cost scoring, the shared α), ADR-0018
  (pressure-reactive budget), ADR-0019 (Budget Floor, uniform eligibility,
  Recoverable Eviction), ADR-0048 (the same one-decision move on the tool
  set)

## Context

Both cache tiers had the same sandwich: a tested pure scorer at the bottom
(`EvictionPolicy` — Marconi utility blend, 15 direct test references), an
effectful drain at the top (body drops, demotion, the ledger lock), and in
between an untested private ladder deciding *who* is evicted next.

- RAM tier: `PrefixCacheManager.findEvictionCandidate` — the four-strategy
  ladder (preferred-utility → global-utility → preferred-fallback →
  global-fallback) plus the **Budget Floor** filter — private inside a
  2,600-line manager, reachable only by replaying whole-manager admission
  scenarios and asserting counters.
- SSD tier: `SnapshotLedger.terminalLossOrder` — the admission cut's
  worst-victim-first ordering, whose own doc comment calls it a pure
  derivation — private, lock-held, and additionally impure in one detail:
  it read `Date()` inside, so the "pure" ordering depended on when it ran.

The decisions ADR-0011/0019 constrain (writing-partition-first,
floor-members-never-victims, the α-blend ordering, the LRU tiebreak) were
pinned by scenario replays, not by tests that name them.

## Decision

One pure module, **`EvictionCandidatePolicy`**, holds candidate selection
for both tiers as sibling statics; the managers keep every effect.

- **`candidate(now:orderedPartitions:preferred:protected:config:)`** — the
  RAM ladder, moved whole. It reads the passed trees live (the drain loop
  re-derives eligible sets after every drop) and returns a `Candidate`
  value: partition, tree, node, which strategy fired, and the score when
  utility (not the fallback) decided. The **Eviction Configuration** comes
  in by value — the manager's mutable cell stays with the manager.
- **`terminalLossOrder(_:config:now:)`** — the SSD ordering, moved whole,
  with `now` promoted to a parameter so the derivation is actually pure
  over its inputs. `nonisolated`: the **Snapshot Ledger** calls it under
  its own lock on the writer thread, for both the admission cut and the
  **Survival Gate** simulation.

`evictToFitBudget` keeps the drain loop, demote-before-drop, the `dropBody`
chokepoint, and the counters; the ledger keeps its lock, eligibility
filtering (`sortedEligibleResidentsLocked`), and manifest mutation. Only
the *naming of the victim* moved.

## Consequences

- The ladder is a decision table (`EvictionCandidatePolicyTests`): tests
  hand the policy a fixed candidate set and assert which branch fired and
  which victim it named — writing-partition-first over colder neighbors,
  spill-over on a drained partition, floor members never victims, the α
  blend reordering recency ties, the stable `snapshotID` tiebreak.
- The ladder's two fallback arms are unreachable today (uniform
  eligibility makes the eligible set equal the snapshot set) and stay as
  the documented residual safety net for the hard budget invariant — the
  tests say so rather than pretending to reach them.
- Whole-manager replay suites (`PrefixCacheManagerTests`,
  `SurvivalGateTests`, `SnapshotDemotionTests`) keep their role: they pin
  the *drain* — loop, demotion, accounting — no longer doubling as the
  only proof of the selection order.
- The two tiers' selection logic now sits side by side in one file, which
  is where their symmetry (and any future divergence, e.g. a tier-specific
  protection rule) becomes visible and reviewable.
