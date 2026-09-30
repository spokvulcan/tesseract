//
//  MomentPrompts.swift
//  tesseract
//
//  The request text each moment appends to the Day Thread. The system prompt
//  and the Day Opening never change, so a request carries only what this
//  moment needs: the relevant slice of the day, and the card to reply with.
//  Pure: facts in, text out.
//

import Foundation

nonisolated enum MomentPrompts {

    // MARK: Day Opening

    /// The Day Thread's first message: the owner's Profile, their Areas,
    /// today's agenda and last night's carry-over note.
    static func dayOpening(
        facts: DayFacts, profile: [String], carryOver: String?
    ) -> String {
        var lines = ["[Day Opening — \(dayName(facts.now, calendar: facts.calendar))]"]
        lines.append(
            "This is today's thread. Jarvis's moments (Morning Plan, Breakpoints, Evening Wrap-up) and the owner's own messages on the Today page are added here as the day goes on."
        )
        if !profile.isEmpty {
            lines.append("")
            lines.append("What the owner has told you about themselves:")
            lines += profile.map { "- \($0)" }
        }
        if !facts.areas.isEmpty {
            lines.append("")
            lines.append("Areas: " + facts.areas.map(\.name).joined(separator: ", "))
        }
        if let carryOver, !carryOver.isEmpty {
            lines.append("")
            lines.append("Carried over from last night: \(carryOver)")
        }
        lines.append("")
        lines += agendaLines(facts)
        return lines.joined(separator: "\n")
    }

    // MARK: Morning Plan

    static func morningPlan(facts: DayFacts) -> String {
        var lines = ["[Morning Plan]"]
        lines.append(
            "Help the owner start the day. They have several goals across their Areas, not one focus. Put small tasks into the free time before the first meeting, and pick at most one must-do — the one thing that matters most today; it can sit anywhere in the day, even late."
        )
        lines.append("")
        lines += agendaLines(facts)
        if let free = facts.freeBeforeFirstEvent {
            lines.append(
                "Free before the first meeting: \(clock(free.start, facts))–\(clock(free.end, facts)) (\(minutesText(Int(free.duration / 60))))."
            )
        }
        lines.append("")
        lines.append("Reply with only this JSON, nothing before or after it:")
        lines.append(
            #"{"line": "<one warm sentence about the shape of the day>", "must_do": "<task id or null>", "plan": [{"id": "<task id>", "at": "HH:MM", "minutes": <number>}], "suggestions": ["<short tip>"]}"#
        )
        lines.append(
            "Use only task ids listed above. Times are today, local, 24-hour, from now on, never over an event. Leave breathing room. At most 3 suggestions; none is fine."
        )
        return lines.joined(separator: "\n")
    }

    // MARK: Evening Wrap-up

    static func eveningWrapUp(facts: DayFacts, leftovers: [AgendaReminder]) -> String {
        var lines = ["[Evening Wrap-up]"]
        if facts.doneToday.isEmpty {
            lines.append("Done today: nothing checked off in Reminders.")
        } else {
            lines.append("Done today: " + facts.doneToday.map(\.title).joined(separator: "; "))
        }
        if leftovers.isEmpty {
            lines.append("Nothing left over from today.")
        } else {
            lines.append("Still open from today (id — title):")
            lines += leftovers.map { "- \($0.id) — \($0.title)" }
        }
        if let first = facts.tomorrowEvents.first {
            lines.append("Tomorrow starts with: \(clock(first.start, facts)) \(first.title)")
        }
        lines.append("")
        lines.append("Reply with only this JSON, nothing before or after it:")
        lines.append(
            #"{"line": "<one warm sentence that notices what got done>", "leftovers": [{"id": "<id>", "suggest": "tomorrow" | "later" | "drop"}]}"#
        )
        lines.append(
            "Never call anything missed or failed. Suggest \"tomorrow\" for what still matters soon, \"later\" for what can wait undated, \"drop\" only for what no longer matters."
        )
        return lines.joined(separator: "\n")
    }

    // MARK: Night Reflection

    static func nightReflection(facts: DayFacts, profile: [String]) -> String {
        var lines = ["[Night Reflection]"]
        lines.append(
            "Look back over today's thread. Write tomorrow's carry-over note, a first draft of tomorrow, and at most three facts worth asking the owner to remember."
        )
        if !facts.doneToday.isEmpty {
            lines.append("Done today: " + facts.doneToday.map(\.title).joined(separator: "; "))
        }
        let tomorrow = facts.tomorrowEvents
        if !tomorrow.isEmpty {
            lines.append(
                "Tomorrow's calendar: "
                    + tomorrow.map { "\(clock($0.start, facts)) \($0.title)" }.joined(
                        separator: "; "))
        }
        if !profile.isEmpty {
            lines.append("Already in their Profile (don't propose these again):")
            lines += profile.map { "- \($0)" }
        }
        lines.append("")
        lines.append("Reply with only this JSON, nothing before or after it:")
        lines.append(
            #"{"carry_over": "<2–4 warm sentences for tomorrow morning: where things stand, what matters first>", "tomorrow": ["<one short line each, at most 5>"], "proposals": [{"text": "<a lasting fact about the owner, third person>", "reason": "<what today showed>"}]}"#
        )
        lines.append(
            "Propose only lasting facts the owner showed today — a preference, a routine, a person who matters — never guesses, never anything sensitive they didn't volunteer. An empty list is fine."
        )
        return lines.joined(separator: "\n")
    }

    // MARK: Shared

    static func agendaLines(_ facts: DayFacts) -> [String] {
        var lines: [String] = []
        let today = facts.eventsToday
        if today.isEmpty {
            lines.append("Today's calendar: nothing scheduled.")
        } else {
            lines.append("Today's calendar:")
            lines += today.map { "- \(clock($0.start, facts))–\(clock($0.end, facts)) \($0.title)" }
        }
        let allDay = facts.events.filter {
            $0.isAllDay && $0.start < facts.endOfToday && $0.end > facts.startOfToday
        }
        if !allDay.isEmpty {
            lines.append("All day: " + allDay.map(\.title).joined(separator: "; "))
        }
        let due = facts.dueOrOverdue
        if !due.isEmpty {
            lines.append("Tasks due today or earlier (id · Area · when — title):")
            lines += due.prefix(25).map { taskLine($0, facts) }
        }
        let undated = facts.undated
        if !undated.isEmpty {
            lines.append("Undated tasks (id · Area — title):")
            lines += undated.prefix(25).map { taskLine($0, facts) }
            if undated.count > 25 { lines.append("- …and \(undated.count - 25) more") }
        }
        if !facts.doneToday.isEmpty {
            lines.append("Done today: " + facts.doneToday.map(\.title).joined(separator: "; "))
        }
        return lines
    }

    static func taskLine(_ reminder: AgendaReminder, _ facts: DayFacts) -> String {
        var line = "- \(reminder.id) · \(facts.areaName(of: reminder))"
        if let due = reminder.due {
            if due < facts.startOfToday {
                line += " · overdue"
            } else if reminder.dueHasTime {
                line += " · \(clock(due, facts))"
            } else {
                line += " · today"
            }
        }
        return line + " — \(reminder.title)"
    }

    static func clock(_ date: Date, _ facts: DayFacts) -> String {
        AgendaTime.clock(date, calendar: facts.calendar)
    }

    static func minutesText(_ minutes: Int) -> String {
        let hours = minutes / 60
        let rest = minutes % 60
        switch (hours, rest) {
        case (0, _): return "\(rest) min"
        case (_, 0): return "\(hours) h"
        default: return "\(hours) h \(rest) min"
        }
    }

    static func dayName(_ date: Date, calendar: Calendar) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "EEEE d MMMM"
        return formatter.string(from: date)
    }
}
