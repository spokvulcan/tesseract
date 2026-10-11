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
//  what the in-memory store does. EventKit links the copy to nothing, so
//  the Agenda knows it by what it is: same list, same title, the series'
//  time of day when it has one, done on a day the series has since moved
//  past.
//
//  So the day holds a repeating task by its series' id, and asks the
//  Agenda for that day's occurrence: open, or done, and when. Ticking an
//  occurrence off and taking it back leave Reminders as they were — the
//  series back on that occurrence's date, the copy gone, never two open
//  reminders — whether the owner takes it back by Undo, by unticking the
//  task, or by unticking the copy.
//

import Foundation

nonisolated extension AgendaReminder {

    /// This reminder is the copy Reminders kept of an occurrence of `series`
    /// ticked off: done and not repeating, in the series' list under its
    /// title, at its time of day when it has one, done on a day the series
    /// has since moved past. Two repeating reminders of one title in one
    /// list, with no time of day, can be taken for each other.
    func isDoneOccurrence(of series: AgendaReminder, calendar: Calendar) -> Bool {
        guard isCompleted, !repeats, series.repeats, !series.isCompleted, id != series.id,
            listID == series.listID, title == series.title,
            timeOfDay(calendar) == series.timeOfDay(calendar),
            let completed = completedAt, let next = series.due,
            let day = DayKey(for: completed, calendar: calendar).date(calendar: calendar),
            let after = calendar.date(byAdding: .day, value: 1, to: day)
        else { return false }
        return next >= after
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
    /// itself, ticked off — or, for a repeating reminder, the copy of its
    /// occurrence on the owner's `day`, the latest.
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
        guard let series = open.first(where: { $0.id == id && $0.repeats }),
            let start = day.date(calendar: calendar), let end = day.end(calendar: calendar)
        else { return nil }
        return done.filter { copy in
            guard let completed = copy.completedAt, completed >= start, completed < min(end, limit)
            else { return false }
            return copy.isDoneOccurrence(of: series, calendar: calendar)
        }
        .max { ($0.completedAt ?? .distantPast) < ($1.completedAt ?? .distantPast) }
    }
}

extension Agenda {

    /// The occurrence of a repeating reminder ticked off, taken back: the
    /// copy Reminders kept of it goes, then the series is back on its date
    /// (`due`; the copy's when not known). In that order, so a failure on
    /// the way leaves at most one open reminder.
    func reopenOccurrence(
        series seriesID: String, due: Date?, dueHasTime: Bool, copy: AgendaReminder? = nil
    ) async throws {
        guard let series = await store.reminder(id: seriesID, now: now()) else {
            throw AgendaError.notFound("reminder \(seriesID)")
        }
        // Ticked off as itself (no copy made): open it again.
        if series.isCompleted {
            var change = ReminderChange(completed: false)
            if let due { change.due = .set(due, hasTime: dueHasTime) }
            _ = try store.updateReminder(id: seriesID, change)
            return
        }
        var copy = copy
        if copy == nil { copy = await latestDoneCopy(of: series) }
        if let copy { try store.deleteReminder(id: copy.id) }
        guard let date = due ?? copy?.due else { return }
        let hasTime = due == nil ? (copy?.dueHasTime ?? series.dueHasTime) : dueHasTime
        if series.due != date || series.dueHasTime != hasTime {
            _ = try store.updateReminder(
                id: seriesID, ReminderChange(due: .set(date, hasTime: hasTime)))
        }
    }

    /// What unticking `reminder` means for a repeating task: the series and
    /// the copy of its occurrence done today when `reminder` is either, nil
    /// for any other reminder (opened again as itself).
    func doneOccurrence(touching reminder: AgendaReminder) async
        -> (series: AgendaReminder, copy: AgendaReminder)?
    {
        if reminder.repeats, !reminder.isCompleted {
            guard
                let copy = await latestDoneCopy(
                    of: reminder, on: DayKey(for: now(), calendar: calendar))
            else { return nil }
            return (reminder, copy)
        }
        guard reminder.isCompleted, !reminder.repeats else { return nil }
        let open = await store.openReminders()
        guard
            let series = open.first(where: { reminder.isDoneOccurrence(of: $0, calendar: calendar) }
            )
        else { return nil }
        return (series, reminder)
    }

    /// The latest copy Reminders kept of an occurrence of `series` done in
    /// the last week — or, with `day`, done that day.
    private func latestDoneCopy(of series: AgendaReminder, on day: DayKey? = nil) async
        -> AgendaReminder?
    {
        let end = now().addingTimeInterval(1)
        let start = day?.date(calendar: calendar) ?? now().addingTimeInterval(-7 * 86_400)
        return await store.completedReminders(from: start, to: end)
            .filter { $0.isDoneOccurrence(of: series, calendar: calendar) }
            .max { ($0.completedAt ?? .distantPast) < ($1.completedAt ?? .distantPast) }
    }
}
