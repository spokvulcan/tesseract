# ADR-0051: The Event Fold's write side is a reducer

- Status: Accepted
- Date: 2026-07-19
- Relates to: ADR-0040 (§13 failure semantics, the harness/entity split),
  ADR-0043 (Wake Evaluator — the fold's read side), ADR-0046 (the Event
  Fold; the store-enforced consume invariant), ADR-0042/0050 (the same
  decider/performer move on the voice loop and the hold wiring)

## Context

ADR-0046 left the fold's decisions in two unequal states. The read side —
"should a turn run now" — is the Wake Evaluator, a pure decider with its
own decision-table suite. The storage math (nothing lost, nothing
duplicated, order kept) is enforced by `MemoryStore` and tested there. But
the write-side sequencing that binds them lived inline in `CompanionLoop`:

- fired-before-the-turn (a crash between fire and completion must be
  visible as fired-but-unconsumed),
- consume only on a completed turn — the correctness invariant, *enforced*
  at the store but *decided* in loop code,
- the retry ladder (ADR-0040 §13): re-present everything while retries
  remain, then wakes fall back to plain banners (never-silent-give-up)
  while Events deliberately stay presented for launch recovery,
- the reaction writes (#309): heard stamped first for every outcome, the
  engage upgrade, the reply-becomes-followup-wake composition,
- wake-state transitions scattered across three methods.

`CompanionLoop` is wired with ~19 closures at the container and no test
constructs it — so none of these decisions had a test, and the report's
review found them only readable by tracing the loop end to end.

## Decision

**`CompanionFoldReducer`** — the Wake Evaluator's write-side sibling: a
pure decider whose only state is the failed-attempt ledger, returning
ordered **effect values** naming every store/notifier write. The loop
shrinks to gather → decide → perform.

- `begin(batch:dueWakes:carriesBeat:now:)` → a `TurnPlan`: `skip` when
  nothing drained and nothing due, else `present` with `fireWake` values
  (the reducer is the home of wake-state flips — the fired copy rides the
  effect) and the beat's `runResurfacingPass`.
- `settle(batch:wakes:outcome:now:)` — the invariant's home. A completed
  turn yields `consumeEvents` + guarded `deliverFiredWake` (the performer
  re-reads so a `revise_wake` flip back to booked wins over delivery) and
  clears the ledger whole; a failure counts an attempt under the batch's
  earliest wake (or event), re-presents everything while retries remain,
  then yields `fallbackBanner` values — Events deliberately absent.
- `reaction(outcome:wakeID:conversationID:note:)` → the reaction writes,
  heard first for every outcome; the reply's followup-wake *content* is
  composed here, while the performer mints ids and dates.

`CompanionLoop` keeps gathering (drain, due wakes, signals), the opening
composition, the recorder, `postedPings`, delivery plumbing, and one
`perform(_:now:)` switch executing effects in order. `CompanionTurnRunner`
stays the turn's performer, untouched. Effect lists omit empty-set writes
(`consumeEvents`/`representEvents` with no ids) that were previously
no-op store calls.

## Consequences

- The write side is a decision table (`CompanionFoldReducerTests`):
  presentation (fire order, `firedAt` stamped once, event-only turns
  present rather than skip), the settlement invariant in both halves, the
  ladder's exhaustion and ledger reset, and every reaction shape —
  scenarios that previously required a constructed loop, store, runner,
  and notifier to express at all.
- Wake-state transitions have one home. The remaining flips outside it
  are the store-owned recovery selections (`runLaunchRecoveryOnce`) and
  the performer's guarded delivered-flip — each named by an effect's
  contract or the recovery's own comment.
- A future write-side rule (a new retry class, a new reaction outcome)
  is a reducer case plus a decision-table row, not a fourth inline home.
- The loop remains unconstructed by tests — deliberately: what it still
  owns is wiring and performance. The decisions left it.
