# ADR-0091: The day has one read model, and a step is one value

- Status: Accepted (built on `refactor/companion-deepening`)
- Date: 2026-10-11
- Amends: ADR-0080 (the day's state and how every surface reads the day; the
  mechanism only, every promise it makes stands)

## Context

The Companion grew one rule at a time (ADR-0080 and its amendments), and the
day's state and the ways of reading it grew with it.

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

**The day was read three ways.** The engine projected the Agenda's snapshot
and its state into `DayFacts` (`DaySnapshot.facts(state:)`); Today built its
own `DayFacts` and Timeline inside a SwiftUI body; the Jarvis Panel built a
third from the plan card's placements. On top of them:

- **"The step under way" had four rules that disagreed.** The engine's focus
  (and the menu bar's clock) was a step the owner started; the Now Card
  showed any event first, then a task; the Day Line tinted every row that
  spanned now. A step started inside the owner's own block ("Work",
  09:00–13:00) was the step for the engine and the menu bar, the block on
  the Now Card, and both on the Day Line.
- **Free and busy time was computed three ways**: the Day Line (every event
  and the way to one, a quarter-hour grid, gaps of half an hour), the engine
  (meetings with other people or a place, a five-minute grid) and the card
  parser (no placement over any event). "A time to leave still applies"
  (its event still on the calendar, same id and start) was written eight
  times.
- **The evening window, the Area name and Waiting on You were each written
  twice**, the last with a deliberate difference (the glyph doesn't count a
  card the owner took in) that only the comments carried.

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
   that applies those rules, so no caller resets a life by hand. Starting is
   one too (`Step.started(at:timed:)`): a step whose slot is now is cut
   clear of the next meeting first, then started (`DayEngine.startNow`),
   whether the owner starts it from Today or from its cue.
3. **The cue on the panel names a step and a phase** (`CueOnPanel`), not a
   key. Taking it down unanswered — the app quitting, the Companion going
   off, a card taking the panel — is one operation (`DayState.uncuePanel`):
   that cue is cued again by its rules.
4. **day-state.json keeps loading.** A state saved with the maps is read
   once into its steps (`DayState.init(from:)`, under the old keys); only
   steps are written. A step's JSON is a placement's with its life added, so
   an older build reading a newer file keeps the plan's slots.

### The day is read once

5. **One projection, one read model.** `DayFacts(snapshot:state:)` is the
   only projection of the Agenda's snapshot and the day's state; the
   **Timeline** (`Timeline(snapshot:state:)`, `Features/Companion/Day/`) is
   built on it and answers what every surface asks: the Day Line's rows and
   tomorrow, the step under way, the next step after a slot, when to leave
   for an event, free time and the Inbox's offers in it, Waiting on You, the
   menu bar's clock, whether the day is planned, wrapped up, in its evening.
   Today reads it in one call (`CompanionRuntime.timeline(at:)`), the Now
   Card is a decision table on it (`NowCardBuilder.build(_:companionOn:)`),
   the Jarvis Panel lists a plan's steps from it, the runtime pushes its
   clock to the menu bar, and the engine reads its focus, next step and
   tomorrow from it. The Timeline moved out of `Today/`: the engine depends
   on it, and it is no view's.
6. **One rule for the step under way** (`UnderWay`), the first that matches:
   1. a step the owner started whose slot holds now — over a block of their
      own, or a meeting it runs into (the engine's rule);
   2. a meeting under way: other people, or a place to be;
   3. a block of the owner's own under way;
   4. a task whose slot or time holds now, not started (the Now Card offers
      Start now).

   The Now Card leads with it (a time to leave still comes before a task,
   and after a meeting or block, as it did), the Day Line tints its row
   alone, the menu bar counts a started one down, and the engine's focus is
   a started one.
7. **One classification of time** (`DayFacts+Time`): an event is a
   **meeting** (other people, or a place to be: `AgendaEvent.isMeeting`) or
   a **block** of the owner's own; a time to leave stands while its event is
   on the calendar under the same id at the same start
   (`Departure.event(in:)`). Three readers compose it, each keeping its own
   grid and strictness: the Day Line's free time (every event and open
   task, and the way to an event in person; a quarter-hour grid, gaps of
   half an hour), a step's boundaries in the engine (today's meetings and
   the times to leave for them; five minutes before, a five-minute grid),
   and a plan's placements in the card parser (`isClearOfEvents`: off every
   event, a block included).
8. **One home each** for the evening window (`DaySettings.isEvening`), the
   Area name (`DayFacts.areaName`) and Waiting on You (`WaitingOnYou`:
   everything Today shows, and the glyph's count, which leaves out a card
   the owner took in).

## Considered options

- **Keep the maps, behind accessors.** The keys and their prefix surgery
  stay; every reader still has to know which key a fact lives under.
- **A step that outlives the plan** (taken off, its put-off count kept for
  the day). It would keep the old count for a task put back after being
  taken off, at the price of steps with no slot in every reader of the plan.
  A task taken off the plan and put back is the owner's own fresh decision.
- **A new read model beside the Timeline.** The Timeline was already the
  domain's word for the day as Today shows it; deepening it gives one noun,
  not two.
- **One busy-time rule for every reader.** The Day Line counts a block as
  busy (no free time to offer inside "Work"), a step does not end at one,
  and a plan never lands on one: their differences are the product's, so
  they stay, named, over one classification.

## Consequences

- A new per-step fact is one field on `Step` and its coding key; tests build
  a started step in one line (`Step("letter", at:, minutes:).started(at:)`)
  and read the day through `Timeline(snapshot:state:)`.
- A surface that needs the day asks the Timeline; a rule about the day's time
  lives in `DayFacts+Time` or the Timeline, not in a view.
- **Behaviour changes**, each pinned by a test:
  - A step the owner started inside a block of their own is the step on the
    Now Card, as it already was for the engine and the menu bar
    (`aStepStartedInsideABlockReadsTheSameEverywhere`); the Now Card showed
    the block.
  - The Day Line tints one row, the step under way; it tinted every row that
    spanned now (a block and the task inside it), a row of tomorrow's in the
    small hours too.
  - A meeting under way inside a block is the step on the Now Card, whichever
    started first (`aMeetingInsideABlockIsTheStepTheDayIsOn`); it showed the
    event that started first.
  - A step the owner started is the step on the Now Card over a meeting it
    runs into, and over an earlier task, not started, whose slot holds now
    too (`aStartedStepLeadsTheNowCardOverAMeetingAndAnEarlierTask`); it
    showed the meeting, and else offered the earlier task's Start now.
  - The Jarvis Panel lists a Morning Plan's steps from the day's plan as it
    stands, not the card's copy of it: while Jarvis rethinks a plan in the
    afternoon, the panel lists the plan the day still keeps, and a step the
    owner started is listed and not offered to start again
    (`thePlansStepsAreTheTasksPlannedAmongTheEvents`).
  - A task taken off today's plan and put back starts a fresh step, its
    put-off count begun again (`putOff` kept it by task for the rest of the
    day) — `aStepTakenOffThePlanAndPutBackStartsAfresh`.
  - Started from Today just before a meeting, a step is timed against the
    minutes the meeting leaves it, as a cue's Start is
    (`startNowOnTodayBeforeAMeetingEndsBeforeIt`). It was timed against the
    minutes offered, so the plan's pace read the meeting's cut as the owner
    being quick (60 planned, 25 taken).
- The marks of a slot a step has left are no longer kept. Only the cue
  reaction's `secondsToReact` read them; it now measures from the step's
  latest cue at its current slot, so a re-plan that moves a step while its
  cue is still on the panel leaves that one reaction without a time.
- The Companion Trace's vocabulary and values are otherwise unchanged; no
  notification id depends on a step.

## As built

- `Features/Companion/Engine/Step.swift`: `Step` and `CueOnPanel`;
  `Engine/DayTypes.swift`: `DayState.plan`, the earlier maps read into
  steps, `uncuePanel`, `DaySettings.isEvening`.
- `Features/Companion/Day/`: `Timeline.swift` (the Timeline, `UnderWay`,
  `InboxSlot`), `DayFacts.swift` (moved from `Moments/`, with the one
  projection), `DayFacts+Time.swift` (meetings, blocks, times to leave and
  their compositions), `WaitingOnYou.swift`.
- `TimelineTests` (with the `DayFixture` day): the Timeline from snapshots
  and states, the step under way's decision table, the started step inside
  a block read the same on the Now Card, the Day Line, the menu bar and the
  engine. `NowCardTests` builds every card from a snapshot and a state.
  `DayStateStoreTests`: a day saved with the maps loads into its steps and
  saves as steps.
