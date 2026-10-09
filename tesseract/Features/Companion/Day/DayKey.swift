//
//  DayKey.swift
//  tesseract
//
//  Which day a moment belongs to. The owner's day does not end at midnight:
//  a wrap-up at 01:00 and a reflection at 02:30 still belong to the day that
//  is ending. So the day rolls over at 04:00 local time, the same boundary the
//  Morning Plan waits for. The key names the Day Thread, the trace file a
//  moment is read against, and every per-day ledger.
//

import Foundation

nonisolated struct DayKey: Hashable, Comparable, Sendable, Codable, CustomStringConvertible {

    /// The local hour at which one day hands over to the next.
    static let rolloverHour = 4

    /// `yyyy-MM-dd`, the local date the day started on.
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The day `date` belongs to: before 04:00 it is still the previous day —
    /// by the wall clock, not four hours back: on the night the clocks go
    /// forward, 04:00 is three hours after midnight, and four hours back
    /// landed on the day before (so "tomorrow" was today).
    init(for date: Date, calendar: Calendar = .current) {
        let early = calendar.component(.hour, from: date) < Self.rolloverHour
        let day = early ? calendar.date(byAdding: .day, value: -1, to: date) ?? date : date
        let parts = calendar.dateComponents([.year, .month, .day], from: day)
        self.rawValue = String(
            format: "%04d-%02d-%02d", parts.year ?? 1970, parts.month ?? 1, parts.day ?? 1)
    }

    var description: String { rawValue }

    /// Encoded as the plain `yyyy-MM-dd` string.
    init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    static func < (lhs: DayKey, rhs: DayKey) -> Bool { lhs.rawValue < rhs.rawValue }

    /// The local calendar date the key names, at midnight.
    func date(calendar: Calendar = .current) -> Date? {
        let parts = rawValue.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return calendar.date(
            from: DateComponents(year: parts[0], month: parts[1], day: parts[2]))
    }

    /// When this day begins: 04:00 local on its date, by the wall clock (four
    /// hours after midnight is 05:00 the day the clocks go forward).
    func start(calendar: Calendar = .current) -> Date? {
        date(calendar: calendar).flatMap {
            calendar.date(bySettingHour: Self.rolloverHour, minute: 0, second: 0, of: $0)
        }
    }

    /// When this day ends: 04:00 local the next calendar day.
    func end(calendar: Calendar = .current) -> Date? {
        start(calendar: calendar).flatMap {
            calendar.date(byAdding: .day, value: 1, to: $0)
        }
    }

    /// The day after this one.
    func next(calendar: Calendar = .current) -> DayKey {
        guard let start = start(calendar: calendar),
            let tomorrow = calendar.date(byAdding: .day, value: 1, to: start)
        else { return self }
        return DayKey(for: tomorrow, calendar: calendar)
    }
}
