//
//  Agenda+Changes.swift
//  tesseract
//
//  The changes the day makes to the owner's Reminders — done or open again,
//  to tomorrow, to today, to another day or a time, kept for later, let go,
//  a follow-up or a proposed task added — applied in one place, with
//  Reminders' rules for dates and series. The owner's today and tomorrow
//  follow the Day Key (until 04:00, the day that is ending), by the Agenda's
//  clock. A move to another day keeps a task's time of day. A repeating
//  reminder is one item for its whole series, so a new date or time is every
//  occurrence's: it never loses its date (kept for later, or let go, it
//  skips to tomorrow at its time instead — EventKit refuses the one, and the
//  other would delete every occurrence), and it is never re-dated to today
//  (done early, it is ticked off where it is). The Day Engine says what the
//  owner chose; the runtime hands it here.
//

import Foundation

/// A change the day makes to the owner's Reminders.
nonisolated enum AgendaMutation: Sendable, Equatable {
    /// Due tomorrow, keeping a time of day if it had one.
    case dueTomorrow(reminderID: String)
    /// Due on the owner's today, at no set time: one of today's Anytime
    /// tasks. Never a repeating reminder.
    case dueToday(reminderID: String)
    /// No date: it waits in its Area or the Inbox. A repeating reminder
    /// skips to tomorrow instead.
    case clearDue(reminderID: String)
    /// Let it go. A repeating reminder skips to tomorrow instead.
    case delete(reminderID: String)
    /// Done, or not done after all: a card's item, a Step Cue's Did it or
    /// Done, its Undo — or, `onToday`, a tick on Today.
    case setDone(reminderID: String, done: Bool, onToday: Bool = false)
    /// Due on this day, keeping a time of day if it had one.
    case dueOn(reminderID: String, day: Date)
    /// Due at this moment.
    case dueAt(reminderID: String, at: Date)
    /// A new reminder: a follow-up for something that can't be handled now.
    case followUp(title: String, at: Date)
    /// A new reminder the owner accepted from a proposal: due that day, or
    /// in the Inbox.
    case add(title: String, due: Date?)
}

extension Agenda {

    /// Make `mutation` in Reminders, by its rules for dates and series. Each
    /// change is confirmed, undoable and traced, as every Agenda write is,
    /// with where it came from: Today, a card, the wrap-up, a proposal.
    func apply(_ mutation: AgendaMutation) async throws {
        switch mutation {
        case .dueTomorrow(let id):
            try await move(id, toDay: ownersDay().next(calendar: calendar))
        case .dueToday(let id):
            // A repeating reminder's date is its whole series'.
            guard await !repeats(id) else { return }
            let today = ownersDay().date(calendar: calendar) ?? now()
            try await updateReminder(id: id, due: .set(today, hasTime: false), source: "today")
        case .clearDue(let id):
            // Kept for later: a repeating reminder can't lose its date.
            if await repeats(id) {
                try await move(id, toDay: ownersDay().next(calendar: calendar))
            } else {
                try await updateReminder(id: id, due: .clear, source: "wrapUp")
            }
        case .delete(let id):
            // Let go: deleting a repeating reminder deletes every occurrence.
            if await repeats(id) {
                try await move(id, toDay: ownersDay().next(calendar: calendar))
            } else {
                try await deleteReminder(id: id, source: "wrapUp")
            }
        case .setDone(let id, let done, let onToday):
            try await updateReminder(id: id, completed: done, source: onToday ? "today" : "card")
        case .dueOn(let id, let day):
            try await move(id, to: calendar.startOfDay(for: day))
        case .dueAt(let id, let at):
            try await updateReminder(id: id, due: .set(at, hasTime: true), source: "card")
        case .followUp(let title, let at):
            try addReminder(title: title, due: at, dueHasTime: true, source: "card")
        case .add(let title, let due):
            try addReminder(title: title, due: due, source: "proposal")
        }
    }

    /// The owner's day by the Agenda's clock: until 04:00, the one ending.
    private func ownersDay() -> DayKey {
        DayKey(for: now(), calendar: calendar)
    }

    private func repeats(_ id: String) async -> Bool {
        await store.reminder(id: id, now: now())?.repeats == true
    }

    private func move(_ id: String, toDay day: DayKey) async throws {
        try await move(id, to: day.date(calendar: calendar) ?? now())
    }

    /// Re-date a reminder to `day`, keeping its time of day if it had one (a
    /// repeating reminder's is its whole series').
    private func move(_ id: String, to day: Date) async throws {
        let reminder = await store.reminder(id: id, now: now())
        var target = day
        if let due = reminder?.due, reminder?.dueHasTime == true {
            let time = calendar.dateComponents([.hour, .minute], from: due)
            target =
                calendar.date(
                    bySettingHour: time.hour ?? 9, minute: time.minute ?? 0, second: 0, of: day)
                ?? day
        }
        try await updateReminder(
            id: id, due: .set(target, hasTime: reminder?.dueHasTime ?? false), source: "wrapUp")
    }
}
