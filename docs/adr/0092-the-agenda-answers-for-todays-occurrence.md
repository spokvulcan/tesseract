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
2. **One rule matches a copy to its series**, and the Agenda holds it
   (`AgendaReminder.isDoneOccurrence(of:calendar:)`). The copy is done and
   doesn't repeat, is in the series' list under its title, at the series'
   time of day when it has one, and was done on a day the series has since
   moved past. The copy is read from the snapshot, not caught where a click
   completed it, so a tick on the phone counts too.
3. **Taking an occurrence back leaves Reminders as it was**
   (`Agenda.reopenOccurrence`). The copy is deleted, and then the series is
   put back on the occurrence's date. In that order, a failure on the way
   leaves at most one open reminder, never two. The tick's Undo carries the
   occurrence (`AgendaUndo.reopenOccurrence`: the series, and its date
   before the tick). Unticking the task or its copy (a cue's Undo, Today's
   ring, an agent's tool) does the same (`doneOccurrence(touching:)`).
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
- **Keep the copy's id in the tick's Undo.** The save doesn't return the
  copy, so the Agenda would search for it either way. Searching when the
  tick is taken back is the same search an untick needs, and it also finds a
  copy made on another device.
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
  - Unticking a repeating task done today, by the task (a cue's Undo,
    Today's ring) or by its copy, does the same (the same test, "the task"
    and "the copy"). Before, the series was left as it was, or the copy was
    reopened beside it.
  - So a must-do taken back is no longer done
    (`aMustDoTakenBackIsNotDoneAfterAll`).
  - Today's Chat's Morning Plan shows a repeating task ticked off as done,
    and its ring takes it back (`theMorningPlanShowsTodaysOccurrenceDone`).
    Before, it showed the series open, and the ring ticked off tomorrow's
    occurrence.
  - In the in-memory store, an event moved to another time gets a new id,
    as it does in EventKit (`AgendaToolsTests`). This is for the tests only:
    the Mac's behaviour doesn't change.
- What the 2026-10-10 amendment promised still holds, pinned through the
  Agenda and from snapshots (`DayEngineRepeatingTaskTests`). The must-do
  counts as done and its step is done in its slot, and the minutes it took
  join the plan's pace. Tomorrow's occurrence doesn't wear today's star,
  because the must-do is today's. A task moved on undone, one that doesn't
  repeat, and an occurrence done another day are left as they are.
- **Open risks.**
  - Two repeating reminders with one title in one list and no time of day
    can be taken for each other, as before.
  - A copy with no completion date is not among the done reminders (they
    are read by completion date), so the day can't see it.
  - An account might complete the item itself and make a new reminder for
    the next occurrence (a third-party report against an older EventKit,
    not what the owner's Mac does). The day would still see the task done,
    but taking the tick back would reopen the item beside the new one.

## As built

- `Features/Companion/Agenda/Agenda+Occurrences.swift`: the matching rule,
  the day's occurrence, and taking an occurrence back. `Agenda.swift`: a
  tick's Undo and an untick of a repeating task go through the occurrence.
  `AgendaStore.reminder(id:now:)` also finds a reminder completed this
  second.
- `InMemoryAgendaStore.swift`: event ids name their start.
- Deleted from the engine: `followDoneOccurrences`, `DayState.follow`,
  `Step.follow(to:)`. The must-do's done mark and a step's run read the
  day's occurrence. `DayFacts.doneOccurrences` holds the copies of the tasks
  the day holds.
- `DayEngineRepeatingTaskTests` runs the engine and the Timeline from
  snapshots, and the Agenda over the in-memory store: a tick, a tick taken
  back three ways, a must-do taken back, and Today's Chat's plan.
