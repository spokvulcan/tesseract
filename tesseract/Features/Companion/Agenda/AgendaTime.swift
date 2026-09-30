//
//  AgendaTime.swift
//  tesseract
//
//  Reading and writing times for the agenda tools. The model reads "now" from
//  the Now Tag and passes absolute local times (`2026-10-01T10:00`, or a bare
//  `2026-10-01` for a whole day); `today`/`tomorrow` and a bare `10:00` are
//  accepted too. Relative anchors ("after the 1:1") resolve to the end of
//  the matching calendar event.
//

import Foundation

nonisolated enum AgendaTime {

    /// A parsed time: a moment, or a whole day.
    struct Parsed: Equatable, Sendable {
        var date: Date
        var hasTime: Bool
    }

    /// Parse a tool's time argument against `now`. nil when unreadable.
    static func parse(_ raw: String, now: Date, calendar: Calendar = .current) -> Parsed? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !text.isEmpty else { return nil }
        let startOfToday = calendar.startOfDay(for: now)
        var day: Date?
        var rest = Substring(text)
        if rest.hasPrefix("today") {
            day = startOfToday
            rest = rest.dropFirst("today".count)
        } else if rest.hasPrefix("tomorrow") {
            day = calendar.date(byAdding: .day, value: 1, to: startOfToday)
            rest = rest.dropFirst("tomorrow".count)
        } else if let match = rest.prefixMatch(of: /(\d{4})-(\d{1,2})-(\d{1,2})/) {
            let wanted = DateComponents(year: Int(match.1), month: Int(match.2), day: Int(match.3))
            // Calendar would roll "2026-13-40" into 2027; an impossible date is unreadable.
            guard let date = calendar.date(from: wanted),
                calendar.dateComponents([.year, .month, .day], from: date) == wanted
            else { return nil }
            day = date
            rest = rest[match.range.upperBound...]
        }
        // The ISO `T`, a space, a comma, or "at" separate the day from the clock.
        rest = rest.drop { $0 == "t" || $0 == " " || $0 == "," }
        if rest.hasPrefix("at ") { rest = rest.dropFirst(3) }
        if rest.isEmpty {
            guard let day else { return nil }
            return Parsed(date: day, hasTime: false)
        }
        guard let clock = rest.wholeMatch(of: /(\d{1,2}):(\d{2})(?::\d{2})?/),
            let hour = Int(clock.1), let minute = Int(clock.2), hour < 24, minute < 60,
            let date = calendar.date(
                bySettingHour: hour, minute: minute, second: 0, of: day ?? startOfToday)
        else { return nil }
        return Parsed(date: date, hasTime: true)
    }

    /// The event an anchor phrase names ("the 1:1", "standup", an event id):
    /// an exact id first, then the soonest event from now whose title contains
    /// the phrase, then the most recent one that already ended today.
    static func anchorEvent(
        _ phrase: String, in events: [AgendaEvent], now: Date
    ) -> AgendaEvent? {
        if let byID = events.first(where: { $0.id == phrase }) { return byID }
        var words = phrase.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        for article in ["the ", "my ", "our "] where words.hasPrefix(article) {
            words = String(words.dropFirst(article.count))
        }
        guard !words.isEmpty else { return nil }
        let matching = events.filter { !$0.isAllDay && $0.title.lowercased().contains(words) }
        if let upcoming = matching.filter({ $0.end > now }).min(by: { $0.start < $1.start }) {
            return upcoming
        }
        return matching.max { $0.end < $1.end }
    }

    // MARK: - Describing

    /// "10:00" today, "tomorrow 10:00", "Friday 3 October 10:00", or the day
    /// alone for a whole-day time.
    static func describe(
        _ date: Date, hasTime: Bool, now: Date, calendar: Calendar = .current
    ) -> String {
        let time = clock(date, calendar: calendar)
        if calendar.isDate(date, inSameDayAs: now) {
            return hasTime ? "today \(time)" : "today"
        }
        if let tomorrow = calendar.date(byAdding: .day, value: 1, to: now),
            calendar.isDate(date, inSameDayAs: tomorrow)
        {
            return hasTime ? "tomorrow \(time)" : "tomorrow"
        }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "EEEE d MMMM"
        let day = formatter.string(from: date)
        return hasTime ? "\(day) \(time)" : day
    }

    /// 24-hour `HH:mm`, the one clock the tools and cards speak.
    static func clock(_ date: Date, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.hour, .minute], from: date)
        return String(format: "%02d:%02d", parts.hour ?? 0, parts.minute ?? 0)
    }

    /// `2026-10-01T10:00`, the form the tools read back.
    static func iso(_ date: Date, hasTime: Bool, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.year, .month, .day, .hour, .minute], from: date)
        let day = String(
            format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
        return hasTime ? day + "T" + clock(date, calendar: calendar) : day
    }
}
