# ADR-0043: The Wake Evaluator is a pure decider; the loop gathers and performs

- Status: Accepted
- Date: 2026-07-18
- Relates to: ADR-0040 (the promise this delivers — §2's "a pure function,
  replayable over a recorder snapshot"), ADR-0042 (the sibling split in the
  voice session), map #301

## Context

ADR-0040 §2 specified the loop's spine as `(now, persisted loop state,
signals) → due wakes / eligibility` — a pure function. What shipped braided
the decisions into `CompanionLoop.evaluate()`: the attention-gate check, the
overdue-batching rule, the origin pick (`wake`/`beat`/`catchup`), the
day-start gate, and ambient eligibility were interleaved with store reads,
day-state writes, recorder events, and `runner.run` in one effectful
`@MainActor` method.

The tell was in the tests: the 568-line loop suite covered the wake table,
resurfacing, `book_wake`, the wake-time grammar, instructions, and the
briefing's render — and never constructed `CompanionLoop`. No test anywhere
exercised the origin pick, the rule that overdue wakes preempt merely-due
ones, the 04:00 day-start boundary, or the ambient gate. A second smell:
"owner present" (`!isIdle && !isScreenLocked`) was derived independently in
the loop's day-start gate and in `CompanionBriefing.gather`.

## Decision

Split the module gather/decide/execute:

- **`CompanionEvaluator`** (a `nonisolated struct`) owns every per-tick
  decision as `mutating func decide(Signals) -> Decision`. `Signals` is the
  gathered snapshot ADR-0040 named — now, local hour, due wakes, day state,
  gate verdict, presence, power, GPU. `Decision` is at most one grant:
  `wait`, `recordDeferral`, `wakeTurn(batch:origin:carriesBeat:)`,
  `dayStart(updated:)`, or `ambient(updated:)`. The decision constants
  (catch-up grace, ambient spacing, the 04:00 day-start floor) move in with
  it. The one bit of cross-tick state — the per-closed-gate-episode deferral
  dedup — lives inside the struct; replaying the same signals reproduces the
  same decisions.
- **`CompanionLoop.evaluate()`** shrinks to gather → decide → execute: it
  reads the stores, asks the evaluator, and performs the single decision —
  turns, day-state writes, recorder events. It decides nothing.
- **`carriesBeat` rides the decision** because origin cannot encode it: an
  overdue rhythm wake fires as `.catchup` and must still run the resurfacing
  ladder. Previously `runWakeTurn` re-derived this; now it is the evaluator's
  call, made once.
- **Day-state updates are decision payloads.** `dayStart`/`ambient` carry the
  updated `CompanionLoopDayState` as a value; the loop persists it before
  running the turn (unchanged ordering — a crash mid-turn must not re-fire
  the day start).
- **"Owner present" gets one home**: `IdleMonitor.isOwnerPresent`, consumed
  by both the evaluator's signals and the Situation Briefing.

## Consequences

- `CompanionEvaluatorTests` pins the rules as decision tables — the origin
  pick, overdue-preempts-due batching, the exact-grace boundary, the
  overdue-beat-still-resurfaces case, deferral dedup across gate episodes,
  the 04:00 rule, presence-gated day start, and the full ambient eligibility
  gate — with no store, no clock, no loop.
- The evaluator is replayable over a flight-recorder snapshot, as ADR-0040
  promised: gather the same signals, get the same decision.
- Retuning due-ness or eligibility means editing one pure module and reading
  the diff of its decision tables; `CompanionLoop` keeps only mechanics
  (ticker, serialization, one-time recovery, delivery plumbing, reactions).
