//
//  NowTag.swift
//  tesseract
//
//  The Now Tag: the one line of time every user message carries into the
//  model — local date, weekday, time and time zone. The system prompt carries
//  no time at all, so it stays byte-identical and every conversation shares
//  one cached system-and-tools prefix; the model reads "now" from the newest
//  message instead, which is always current.
//
//  The tag is stamped when the message is created and stored with it, so a
//  message renders the same bytes every time it is replayed, which keeps the
//  prefix cache valid for the rest of the conversation.
//

import Foundation

nonisolated enum NowTag {

    /// `<now>Wednesday, 30 September 2026, 14:05, Europe/Berlin (UTC+02:00)</now>`
    static func render(_ date: Date, timeZone: TimeZone = .current) -> String {
        "<now>\(describe(date, timeZone: timeZone))</now>"
    }

    /// The tag's text without the markup — also used where a moment request
    /// names a time in prose.
    static func describe(_ date: Date, timeZone: TimeZone = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = timeZone
        formatter.dateFormat = "EEEE, d MMMM yyyy, HH:mm"
        return
            "\(formatter.string(from: date)), \(timeZone.identifier) (\(utcOffset(timeZone, at: date)))"
    }

    /// `UTC+02:00`, `UTC-05:30`, or `UTC` at zero offset.
    static func utcOffset(_ timeZone: TimeZone, at date: Date) -> String {
        let seconds = timeZone.secondsFromGMT(for: date)
        guard seconds != 0 else { return "UTC" }
        let sign = seconds > 0 ? "+" : "-"
        let minutes = abs(seconds) / 60
        return String(format: "UTC%@%02d:%02d", sign, minutes / 60, minutes % 60)
    }
}
