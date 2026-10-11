//
//  Agenda+Occurrences.swift
//  tesseract
//
//  A repeating reminder's occurrences, as the Agenda answers for them
//  (ADR-0092). Reminders keeps a repeating reminder as one item for its
//  whole series: only its next occurrence is open, and ticking it off makes
//  the next one so (Apple's EventKit guide, "Creating a recurring event").
//  What it keeps of the occurrence done is a copy with an id of its own and
//  no repeat, while the series, under its own id, moves on to its next date
//  — what the owner's Reminders did on 9 October (ADR-0080, 2026-10-10) and
//  what the in-memory store does. EventKit links the copy to nothing.
//
//  So the day holds a repeating task by its series' id, and asks the
//  Agenda for that day's occurrence: open, or done, and when. That is read
//  from what a copy is — same list, same title, the series' time of day
//  when it has one, due before the series' next date — so a tick on the
//  phone counts too; reading may take a one-off of the same name for one.
//  Changing Reminders never goes by that. Only an occurrence the Agenda
//  ticked off itself is taken back as one — its copy deleted, the series
//  back on the occurrence's date — and only while both are as the tick
//  left them. Any other untick opens again just the reminder it names,
//  even if that leaves a copy open beside its series: data safety over
//  tidiness.
//

import Foundation

/// An occurrence of a repeating reminder the Agenda ticked off itself, read
/// right after the tick: what taking it back restores.
nonisolated struct TickedOccurrence: Sendable, Equatable {
    /// The repeating reminder ticked off.
    let seriesID: String
    /// The copy Reminders kept of the occurrence, when the Agenda found it.
    let copyID: String?
    /// The occurrence's date: the series' before the tick.
    let due: Date
    let dueHasTime: Bool
    /// The date the tick moved the series on to.
    let movedTo: Date
    let at: Date
}

nonisolated extension AgendaReminder {

    /// This reminder looks like a copy Reminders kept of an occurrence of
    /// `series` ticked off: done and not repeating, in the series' list
    /// under its title, at its time of day when it has one, and due before
    /// the series' next date. A one-off of the same name, or a copy of
    /// another series of that name with no time of day, looks the same.
    func isDoneOccurrence(of series: AgendaReminder, calendar: Calendar) -> Bool {
        guard isCompleted, !repeats, series.repeats, !series.isCompleted, id != series.id,
            listID == series.listID, title == series.title,
            timeOfDay(calendar) == series.timeOfDay(calendar),
            let due, let next = series.due
        else { return false }
        return next > due
    }

    private func timeOfDay(_ calendar: Calendar) -> DateComponents? {
        guard dueHasTime, let due else { return nil }
        return calendar.dateComponents([.hour, .minute], from: due)
    }
}

nonisolated extension AgendaSnapshot {

    /// The task `id` as the owner's `day` sees it: its occurrence that day
    /// done (`doneOccurrence`), or else the task open.
    func occurrence(of id: String, on day: DayKey, calendar: Calendar = .current)
        -> AgendaReminder?
    {
        doneOccurrence(of: id, on: day, calendar: calendar) ?? open.first { $0.id == id }
    }

    /// The task `id` done (before `before`, when given): the reminder
    /// itself, ticked off — or, for a repeating reminder, a copy of its
    /// occurrence done on the owner's `day`, the latest, once nothing of
    /// the series is open that day any more (an overdue one ticked off
    /// late moves on to today, where its next occurrence waits).
    func doneOccurrence(
        of id: String, on day: DayKey, before: Date? = nil, calendar: Calendar = .current
    ) -> AgendaReminder? {
        let done = doneToday + doneThisWeek
        let limit = before ?? .distantFuture
        if let itself = done.first(where: {
            $0.id == id && ($0.completedAt ?? .distantPast) < limit
        }) {
            return itself
        }
        guard let series = open.first(where: { $0.id == id && $0.repeats }), let next = series.due,
            let start = day.date(calendar: calendar), let end = day.end(calendar: calendar)
        else { return nil }
        return done.filter { copy in
            guard let completed = copy.completedAt, completed >= start, completed < min(end, limit),
                copy.isDoneOccurrence(of: series, calendar: calendar),
                let doneOn = DayKey(for: completed, calendar: calendar).date(calendar: calendar),
                let after = calendar.date(byAdding: .day, value: 1, to: doneOn)
            else { return false }
            return next >= after
        }
        .max { ($0.completedAt ?? .distantPast) < ($1.completedAt ?? .distantPast) }
    }
}

extension Agenda {

    /// What ticking off the repeating `series` left in Reminders, read right
    /// after the tick: the series moved on, open, and the copy of the
    /// occurrence — done since `since`, due on the occurrence's day. Nil
    /// when the series didn't move on (the store ticked off the item
    /// itself).
    func tickedOccurrence(of series: AgendaReminder, since: Date) async -> TickedOccurrence? {
        guard let due = series.due,
            let moved = await store.reminder(id: series.id, now: now()), moved.repeats,
            !moved.isCompleted, let movedTo = moved.due, movedTo > due
        else { return nil }
        let noted = Set(occurrencesTicked.compactMap(\.copyID))
        let copy = await store.completedReminders(
            from: since.addingTimeInterval(-1), to: now().addingTimeInterval(1)
        )
        .filter { copy in
            guard let copyDue = copy.due, !noted.contains(copy.id) else { return false }
            return calendar.isDate(copyDue, inSameDayAs: due)
                && copy.isDoneOccurrence(of: moved, calendar: calendar)
        }
        .max { ($0.completedAt ?? .distantPast) < ($1.completedAt ?? .distantPast) }
        return TickedOccurrence(
            seriesID: series.id, copyID: copy?.id, due: due, dueHasTime: series.dueHasTime,
            movedTo: movedTo, at: now())
    }

    /// The occurrence the Agenda ticked off that unticking `reminder` takes
    /// back: the one whose copy it is — or, for the series, the latest
    /// ticked off on the owner's day.
    func tickedOccurrence(touching reminder: AgendaReminder) -> TickedOccurrence? {
        if let byCopy = occurrencesTicked.last(where: { $0.copyID == reminder.id }) {
            return byCopy
        }
        guard reminder.repeats, !reminder.isCompleted else { return nil }
        let today = DayKey(for: now(), calendar: calendar)
        return occurrencesTicked.last {
            $0.seriesID == reminder.id && DayKey(for: $0.at, calendar: calendar) == today
        }
    }

    /// Take back an occurrence the Agenda ticked off: its copy goes, then
    /// the series is back on the occurrence's date — in that order, so a
    /// failure on the way leaves at most one open reminder. Only while both
    /// are as the tick left them: the series open where it moved on to, the
    /// copy done (or gone). Otherwise nothing is touched, and it says so.
    func takeBack(_ occurrence: TickedOccurrence) async throws -> Bool {
        guard let series = await store.reminder(id: occurrence.seriesID, now: now()),
            series.repeats, !series.isCompleted, series.due == occurrence.movedTo
        else { return false }
        if let copyID = occurrence.copyID,
            let copy = await store.reminder(id: copyID, now: now())
        {
            // Opened again since, it is the occurrence open already.
            guard copy.isCompleted else { return false }
            try store.deleteReminder(id: copyID)
        }
        occurrencesTicked.removeAll { $0 == occurrence }
        _ = try store.updateReminder(
            id: occurrence.seriesID,
            ReminderChange(due: .set(occurrence.due, hasTime: occurrence.dueHasTime)))
        return true
    }
}
