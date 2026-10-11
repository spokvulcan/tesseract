# ADR-0091: The day has one read model, and a step is one value

- Status: Accepted (built on `refactor/companion-deepening`)
- Date: 2026-10-11
- Amends: ADR-0080 (the day's state and how Today reads it; the mechanism
  only, every promise it makes stands)

## Context

The Companion grew one rule at a time (ADR-0080 and its amendments), and the
day's state grew with it.

**A step's life was spread over seven fields.** The plan was a list of
placements (task, start, minutes); what happened to each slot lived beside
it in `DayState`: `cuedSteps` (when a start was cued, and an end checked in),
`startedSteps`, `smallStarts`, `startedMinutes`, `putOff`, `cutShort` and
`cueOnPanel`. A start's marks were keyed `id@epoch`, an end's
`id@epoch+minutes` ("a slot made longer has a new end"), put-off counts and
cut minutes by task. Every new per-step fact had to be threaded through the
coding keys and the decoder, `relaunched`, `rolledOver`, `follow`, the
Companion switching off and a card taking the panel. The repeating must-do
fix (2026-10-10) rewrote the maps by string-prefix surgery; Today rebuilt the
engine's key from a Timeline row to learn whether a step was started; the
tests filled three or four maps to set up "a started step" (34 references in
six files).

## Decision

### A step is one value

1. **The plan holds Steps** (`Step`, `DayState.plan`): one per task, as the
   plan already was — every writer replaced or skipped a placement by task
   id. A step carries its slot (`Placement`) and its life.
2. **Its life is bound three ways**, read off the keys it had:
   - **the start's** (`id@epoch`): cued, started, a five-minute start, the
     minutes a whole start is timed against — a step moved to a new start
     begins anew there;
   - **the end's** (`id@epoch+minutes`): its end checked in — a new end (a
     new start, made longer, cut short) checks in again;
   - **the task's** (by id): how often it was put off, the minutes a meeting
     cut — carried wherever the step moves that day, a re-plan that keeps
     the task included.

   Moving is one operation (`Step.reslot`, and `move(to:)`, `resize(to:)`)
   that applies those rules, so no caller resets a life by hand.
3. **The cue on the panel names a step and a phase** (`CueOnPanel`), not a
   key. Taking it down unanswered — the app quitting, the Companion going
   off, a card taking the panel — is one operation (`DayState.uncuePanel`):
   that cue is cued again by its rules.
4. **day-state.json keeps loading.** A state saved with the maps is read
   once into its steps (`DayState.init(from:)`, under the old keys); only
   steps are written. A step's JSON is a placement's with its life added, so
   an older build reading a newer file keeps the plan's slots.

## Considered options

- **Keep the maps, behind accessors.** The keys and their prefix surgery
  stay; every reader still has to know which key a fact lives under.
- **A step that outlives the plan** (taken off, its put-off count kept for
  the day). It would keep the old count for a task put back after being
  taken off, at the price of steps with no slot in every reader of the plan.
  A task taken off the plan and put back is the owner's own fresh decision.

## Consequences

- A new per-step fact is one field on `Step` and its coding key; tests build
  a started step in one line (`Step("letter", at:, minutes:).started(at:)`).
- **Behaviour change:** a task taken off today's plan and put back starts a
  fresh step, its put-off count begun again (`putOff` kept it by task for
  the rest of the day) — `aStepTakenOffThePlanAndPutBackStartsAfresh`.
- The marks of a slot a step has left are no longer kept. Only the cue
  reaction's `secondsToReact` read them; it now measures from the step's
  latest cue at its current slot, so a re-plan that moves a step while its
  cue is still on the panel leaves that one reaction without a time.
- The Companion Trace's vocabulary and values are otherwise unchanged; no
  notification id depends on a step.

## As built

- `Features/Companion/Engine/Step.swift`: `Step` and `CueOnPanel`.
- `Features/Companion/Engine/DayTypes.swift`: `DayState.plan`, the earlier
  maps read into steps, `uncuePanel`.
- `DayStateStoreTests`: a day saved with the maps loads into its steps and
  saves as steps.
