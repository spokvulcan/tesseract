# ADR-0046: The Companion loop is an Event Fold over one standing conversation

- Status: Accepted
- Date: 2026-07-18
- Relates to: ADR-0040 (the entity/harness split this refines), ADR-0043 (the
  pure decider that absorbs the fold's clock), ADR-0035 (the sleep pass the
  Digest rides), #353 (the North Star tiebreaker), #366 (the PRD), map #301
- Supersedes: the conversation-model core of #357 (which cites "ADR-0042" for
  it — that number was since taken by the Voice Session Machine; this is the
  record)

## Context

Two days of wearing returned a verdict: the Companion's thinking is scattered
across throwaway conversations (one minted per turn), his rhythm is partly
code-owned (a 30-minute ambient cadence, an attention gate granting turns),
and his tracking practice is a five-tool ceremony that derails him in the
field. The owner's requirement, re-grilled 2026-07-18 from first principles:
less hard-coded behavior, a mathematically correct harness implementing the
simple loop that makes a human human, every judgment left to the entity.

## Decision

The loop's algorithm is the **Event Fold**: every digital input becomes
exactly one **Event**; events queue in total order; each granted **Turn**
folds everything pending into **Mission Control** — one standing conversation
that is the entity's whole cognitive state. `state' = turn(state, events)`.
The harness owns exactly the fold's invariants — exactly-once, total order,
batch drain, one turn at a time, effects executed-or-recorded — and zero
judgment.

- **Purist clock, no net.** A turn runs iff (pending Events or a due Wake)
  and mechanically eligible. The ambient cadence, the Ambient Turn, and the
  attention gate's granting role are deleted; there is no liveness fallback.
  Time to think is a Wake the entity books itself.
- **Dialogue out, Report-Back in.** Summoned dialogues stay their own chats,
  run as sub-agents owing Mission Control a **Report-Back** that lands as an
  Event.
- **Lean toolset, 14 → 10.** The five tracking ceremony tools collapse into
  one `track(kind, payload)` (the Observation schema survives as data);
  `flight_log` dies (the conversation is the record); wakes gain
  revise/cancel; `report_back` is new, dialogue-only.
- **Sleep-authored Digest under an 80k-token ceiling.** Nightly, after
  ADR-0035 consolidation, the entity authors a **Digest** spliced in as the
  conversation's head with the recent tail verbatim; an intraday ceiling hit
  runs the same fold early, on the record.

## Considered options

- **A harness safety tick / auto-booked liveness wake** — rejected by the
  owner knowingly: the entity fully owns its clock. Day-start and Mac-wake
  being Events restores de facto morning liveness; the only deadlock is a Mac
  never opened.
- **One context with dialogue inside** — rejected: dialogues stay snappy
  separate chats; the fold keeps cognition, not chit-chat.
- **Harness-scheduled rolling compaction** — rejected: the digest is the
  entity's own memory practice, and nightly folding costs exactly one
  prefix-cache invalidation per day instead of continuous thrash.

## Consequences

- The append-only standing conversation is the radix prefix cache's best
  case; the design and the server reinforce each other.
- Silence is now purely entity judgment: a quiet Mac with nothing booked
  grants no turns, indefinitely, by design — auditable on the record, not
  recoverable by code.
- Until the purist-clock ticket lands, ADR-0043's text and the Wake Evaluator
  glossary entry describe the previous tick-and-gate clock; they are amended
  with that ticket. Deliberate, not drift.
