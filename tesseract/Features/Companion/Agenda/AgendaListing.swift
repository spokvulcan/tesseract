//
//  AgendaListing.swift
//  tesseract
//
//  The agenda tool's text: a compact, dated listing the model can read at a
//  glance and act on by id. With today in range it adds overdue reminders,
//  the Inbox, and what is already done today.
//

import Foundation

nonisolated enum AgendaListing {

    struct Input: Sendable {
        var from: Date
        var days: Int
        var now: Date
        var events: [AgendaEvent]
        var open: [AgendaReminder]
        var done: [AgendaReminder]
        var areas: [Area]
        var inbox: AgendaList?
        var access: AgendaAccess
    }

    /// Read the store and render.
    @MainActor
    static func render(agenda: Agenda, from: Date, days: Int, now: Date) async -> String {
        let calendar = Calendar.current
        let end = calendar.date(byAdding: .day, value: days, to: from) ?? from
        let input = Input(
            from: from, days: days, now: now,
            events: agenda.store.events(from: from, to: end),
            open: await agenda.store.openReminders(),
            done: await agenda.store.completedReminders(from: from, to: end),
            areas: agenda.areas, inbox: agenda.inbox, access: agenda.access)
        return render(input, calendar: calendar)
    }

    static func render(_ input: Input, calendar: Calendar = .current) -> String {
        var lines: [String] = []
        if !input.access.canUseCalendar {
            lines.append("(No access to Calendar — events are not shown.)")
        }
        if !input.access.canUseReminders {
            lines.append("(No access to Reminders — reminders are not shown.)")
        }
        let startOfToday = calendar.startOfDay(for: input.now)
        let dayFormatter = DateFormatter()
        dayFormatter.locale = Locale(identifier: "en_US_POSIX")
        dayFormatter.calendar = calendar
        dayFormatter.timeZone = calendar.timeZone
        dayFormatter.dateFormat = "EEEE d MMMM yyyy"

        for offset in 0..<input.days {
            guard let day = calendar.date(byAdding: .day, value: offset, to: input.from),
                let next = calendar.date(byAdding: .day, value: 1, to: day)
            else { continue }
            var header = dayFormatter.string(from: day)
            if calendar.isDate(day, inSameDayAs: startOfToday) {
                header += " (today)"
            } else if let tomorrow = calendar.date(byAdding: .day, value: 1, to: startOfToday),
                calendar.isDate(day, inSameDayAs: tomorrow)
            {
                header += " (tomorrow)"
            }
            if !lines.isEmpty { lines.append("") }
            lines.append(header)

            let events = input.events.filter { $0.start < next && $0.end > day }
            if events.isEmpty {
                lines.append("Events: none")
            } else {
                lines.append("Events:")
                for event in events {
                    let when =
                        event.isAllDay
                        ? "all day"
                        : "\(AgendaTime.clock(event.start, calendar: calendar))–\(AgendaTime.clock(event.end, calendar: calendar))"
                    var line = "- \(when) \(event.title)"
                    if let place = AgendaPlace.label(event.location, withLink: true) {
                        line += " @ \(place)"
                    }
                    line += " [\(event.calendarTitle); id \(event.id)]"
                    lines.append(line)
                }
            }

            let due = input.open.filter { reminder in
                guard let date = reminder.due else { return false }
                return date >= day && date < next
            }
            if !due.isEmpty {
                lines.append("Reminders due:")
                for reminder in due {
                    lines.append(line(for: reminder, input: input, calendar: calendar))
                }
            }

            let done = input.done.filter { reminder in
                guard let at = reminder.completedAt else { return false }
                return at >= day && at < next
            }
            if !done.isEmpty {
                lines.append("Done:")
                for reminder in done { lines.append("- \(reminder.title)") }
            }
        }

        let rangeEnd =
            calendar.date(byAdding: .day, value: input.days, to: input.from) ?? input.from
        let includesToday = input.from <= startOfToday && startOfToday < rangeEnd
        if includesToday {
            let overdue = input.open.filter { ($0.due ?? .distantFuture) < startOfToday }
            if !overdue.isEmpty {
                lines.append("")
                lines.append("Overdue:")
                for reminder in overdue.prefix(20) {
                    lines.append(
                        line(for: reminder, input: input, calendar: calendar, withDay: true))
                }
                if overdue.count > 20 { lines.append("- …and \(overdue.count - 20) more") }
            }
            let undated = input.open.filter { $0.due == nil }
            if !undated.isEmpty {
                lines.append("")
                lines.append("No date:")
                for reminder in undated.prefix(30) {
                    lines.append(line(for: reminder, input: input, calendar: calendar))
                }
                if undated.count > 30 { lines.append("- …and \(undated.count - 30) more") }
            }
        }

        if input.access.canUseReminders {
            lines.append("")
            let areaNames = input.areas.map(\.name).joined(separator: ", ")
            lines.append(
                "Areas: \(areaNames.isEmpty ? "none" : areaNames) · Inbox: \(input.inbox?.title ?? "none")"
            )
        }
        return lines.joined(separator: "\n")
    }

    private static func line(
        for reminder: AgendaReminder, input: Input, calendar: Calendar, withDay: Bool = false
    ) -> String {
        var line = "- "
        if let due = reminder.due {
            if withDay {
                let formatter = DateFormatter()
                formatter.locale = Locale(identifier: "en_US_POSIX")
                formatter.calendar = calendar
                formatter.timeZone = calendar.timeZone
                formatter.dateFormat = "EEE d MMM"
                line += "(\(formatter.string(from: due))) "
            }
            if reminder.dueHasTime { line += "\(AgendaTime.clock(due, calendar: calendar)) " }
        }
        line += reminder.title
        let area = input.areas.first { $0.id == reminder.listID }?.name ?? reminder.listTitle
        if reminder.listID == input.inbox?.id {
            line += " · Inbox"
        } else if !area.isEmpty {
            line += " · \(area)"
        }
        line += " [id \(reminder.id)]"
        return line
    }
}
