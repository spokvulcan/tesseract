# ADR-0092: The Agenda answers for today's occurrence

- Status: Accepted (built on `refactor/companion-deepening`)
- Date: 2026-10-11
- Amends: ADR-0080 (supersedes the mechanism of its 2026-10-10 amendment, "the
  day follows the done occurrence"; what that amendment promises stands)
- Relates to: ADR-0073 (test runs never reach the owner's data: the in-memory
  store is the EventKit store's peer, and the EventKit adapter stays thin)

## Context

**Reminders keeps a repeating reminder as one item for its series.** Apple's
EventKit guide says so ("Creating a recurring event"): recurrence applies to
reminders as to events, but only the first incomplete reminder of a
recurring set can be fetched, in EventKit as in the Reminders app, and
completing it makes the next one available. What a tick leaves behind is not
documented. On the owner's Mac (9 October, ADR-0080 2026-10-10) it was a done
copy of the occurrence with an id of its own and no recurrence, while the
series kept its id, open, on its next date. EventKit links the copy to
nothing. Its header adds that a reminder another client completed may have
no completion date (`EKReminder.h`).

**The Agenda passed that model straight through**, and every reader made up
for it:

- The engine matched the copy by list, title and time of day
  (`followDoneOccurrences`) and moved what the day held of the task to the
  copy's id (`DayState.follow`): the must-do, the plan's step with its life,
  the cue on the panel. Every reader of the day's state had to allow for a
  held id that had changed.
- Tomorrow on the Day Line took today's star off the series, which had moved
  on to tomorrow under the must-do's id.
- **Taking a tick back went wrong.** The tick's Undo reopened the series,
  which was already open, and the occurrence stayed done. A cue's Undo
  (unticking the task) did the same. Unticking the done row on Today, the
  copy after the follow, reopened the copy beside the series, leaving two
  open reminders for one task.
- Today's Chat's Morning Plan looked its tasks up by id. After a tick it
  showed the series, open and due tomorrow, and its ring ticked off
  tomorrow's occurrence.

**The two stores disagreed on events.** The EventKit store names an
occurrence `identifier|start`, because a recurring event's calendar item
identifier is the same for every occurrence and the header says to tell them
apart by start date (`EKCalendarItem.h`). So an event moved to another time
gets a new id. The in-memory store's ids survived a move, and tests on it
could not see what the Mac sees.

## Decision

1. **The day holds a repeating task by its series' id**, all day. The Agenda
   answers for the task's **occurrence** on a day
   (`AgendaSnapshot.occurrence(of:on:)`, `doneOccurrence(of:on:before:)`):
   open, or done and when. Done means the reminder itself ticked off or, for
   a repeating reminder, the copy Reminders kept of that day's occurrence.
   The must-do, a step seen done, the pace a step's run adds to, the Day
   Line's row and Today's Chat all ask it. The Timeline shows the copy in
   the task's own slot, under the task's id (`TimelineTask.id`; `reminder`
   is what it shows).
2. **One rule says what a copy of a series looks like**, and the Agenda
   holds it (`AgendaReminder.isDoneOccurrence(of:calendar:)`): done and not
   repeating, in the series' list under its title, at the series' time of
   day when it has one, and due before the series' next date. Reading the
   day goes by it, from the snapshot, so a tick on the phone counts too. A
   copy counts for the day it was done once nothing of its series is open
   that day: an occurrence overdue from yesterday, ticked off late, moves a
   daily series on only to today, where today's occurrence still waits.
3. **Only what the Agenda ticked off itself is taken back as an
   occurrence.** Right after its own tick of a repeating reminder, the
   Agenda reads what the tick left (`Agenda.tickedOccurrence(of:since:)`):
   the series moved on, and the copy done since the tick and due on the
   occurrence's day. It keeps that in memory (`occurrencesTicked`) and in
   the tick's Undo (`AgendaUndo.reopenOccurrence`). Taking it back
   (`Agenda.takeBack`) deletes the copy, then puts the series back on the
   occurrence's date — in that order, so a failure on the way leaves at
   most one open reminder — and only while both are as the tick left them:
   the series open where the tick moved it, the copy done or gone. The
   tick's Undo, unticking the series that day (a cue's Undo, Today's ring)
   and unticking the copy (an agent's tool) all take it back. **Every other
   untick opens again just the reminder it names**: a one-off of the
   series' name, a copy ticked off on the phone or before the app last
   started, an occurrence whose series moved since. A copy may then be
   open beside its series; data safety comes before tidiness, and nothing
   the Agenda can't be sure it made is deleted.
4. **The in-memory store behaves as the Mac's stores do, where the day can
   tell**: a repeating reminder ticked off leaves a done copy with an id of
   its own and the series moved on (as it already did), and an event's id
   names its start (`identifier|epoch`), so a moved event gets a new id.
5. **The logic is the facade's.** It sits in the Agenda, above the
   `AgendaStore` seam, so both adapters share it, and the EventKit adapter
   still only translates (ADR-0073).

## Considered options

- **Keep following the copy** (the 2026-10-10 mechanism). Every reader of
  the day's state would still have to allow for a held id that moved to
  another reminder, and the Undo and an untick would still reach the wrong
  reminder.
- **Have completion return the copy.** Each store would search for the copy
  after its save, which puts the matching rule in both adapters. It still
  could not link a copy ticked off on the phone, which the day has to see as
  well.
- **Find the copy when the tick is taken back**, by what it is (as first
  built). It also took back a copy made on another device, but it could
  take a one-off of the series' name for a copy: unticking "Call mom" done
  today, beside a weekly "Call mom", deleted it and moved the series, and
  the untick's Undo ticked off the series. Matching by completion day also
  missed an overdue occurrence ticked off late. Reading may guess; deleting
  may not.
- **Detach or edit one occurrence in Reminders.** EventKit has no API for a
  reminder's occurrences: only the next one can be fetched.

## Consequences

- One id per task, all day: nothing in the day's state moves to another
  reminder. Anything about a repeating reminder's occurrences is decided in
  `Features/Companion/Agenda/Agenda+Occurrences.swift`.
- **Behaviour changes**, each pinned by a test:
  - The Undo of a tick on a repeating task leaves Reminders as it was before
    the tick: the series back on the occurrence's date, the copy gone, one
    open reminder (`takenBackTheOccurrenceLeavesRemindersAsItWas`, "undo").
    Before, it reopened the series, which was already open, and the
    occurrence stayed done.
  - Unticking a repeating task the Agenda ticked off today, by the task (a
    cue's Undo, Today's ring) or by its copy, does the same (the same test,
    "the task" and "the copy"), an occurrence overdue and ticked off late
    included (`overdue: true`). Before, the series was left as it was, or
    the copy was reopened beside it.
  - Any other untick opens again only the reminder it names, as before: a
    one-off of the series' name (`aOneOffOfTheSeriesNameIsOnlyEverOpenedAgain`,
    its Undo ticking the one-off off again), a copy ticked off elsewhere
    (`aCopyTickedOffElsewhereIsOnlyOpenedAgain`), an occurrence whose series
    moved since (`anOccurrenceChangedSinceItsTickIsOnlyOpenedAgain`).
  - So a must-do taken back is no longer done
    (`aMustDoTakenBackIsNotDoneAfterAll`).
  - Today's Chat's Morning Plan shows a repeating task ticked off as done,
    and its ring takes it back (`theMorningPlanShowsTodaysOccurrenceDone`).
    Before, it showed the series open, and the ring ticked off tomorrow's
    occurrence.
  - In the in-memory store, an event moved to another time gets a new id,
    as it does in EventKit (`AgendaToolsTests`). This is for the tests only:
    the Mac's behaviour doesn't change.
  - Looking a reminder up for a change (`AgendaStore.reminder(id:now:)`)
    also finds one completed this second: it searched done reminders up to
    `now`, so one ticked off at the same instant, on a test's fixed clock,
    was missed (`aTickOnTodayGoesThroughTheEngineToReminders`).
- What the 2026-10-10 amendment promised still holds, pinned through the
  Agenda and from snapshots (`DayEngineRepeatingTaskTests`). The must-do
  counts as done and its step is done in its slot, and the minutes it took
  join the plan's pace. Tomorrow's occurrence doesn't wear today's star,
  because the must-do is today's. A task moved on undone, one that doesn't
  repeat, and an occurrence done another day are left as they are.
- **Open risks.**
  - Reading still goes by what a copy looks like: two repeating reminders
    with one title in one list and no time of day can be taken for each
    other, and a one-off of a series' name done today can make that series,
    held by the day, read done. Nothing is changed by it.
  - A copy the Agenda didn't make — ticked off on the phone, or before the
    app last started, since the record of its own ticks lives in memory —
    opens again beside its series when unticked: two open reminders, which
    the owner tidies in Reminders.
  - If Reminders hasn't made the copy readable by the time the Agenda looks,
    right after its tick, the Agenda keeps the occurrence without it: taking
    it back moves the series back and leaves the copy done.
  - A copy with no completion date is not among the done reminders (they
    are read by completion date), so the day can't see it.
  - An account might complete the item itself and make a new reminder for
    the next occurrence (a third-party report against an older EventKit,
    not what the owner's Mac does). The day would still see the task done,
    but taking the tick back would reopen the item beside the new one.

## As built

- `Features/Companion/Agenda/Agenda+Occurrences.swift`: what a copy looks
  like, the day's occurrence, what the Agenda's own tick left
  (`TickedOccurrence`) and taking it back. `Agenda.swift`: the record of its
  ticks, the tick's Undo, and an untick of an occurrence it ticked off.
  `AgendaStore.reminder(id:now:)` also finds a reminder completed this
  second.
- `InMemoryAgendaStore.swift`: event ids name their start.
- Deleted from the engine: `followDoneOccurrences`, `DayState.follow`,
  `Step.follow(to:)`. The must-do's done mark and a step's run read the
  day's occurrence. `DayFacts.doneOccurrences` holds the copies of the tasks
  the day holds.
- `DayEngineRepeatingTaskTests` runs the engine and the Timeline from
  snapshots, and the Agenda over the in-memory store: a tick, a tick taken
  back three ways (today's occurrence and an overdue one), a must-do taken
  back, Today's Chat's plan, and the unticks that only open again.
