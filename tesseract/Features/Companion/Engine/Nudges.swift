//
//  Nudges.swift
//  tesseract
//
//  Nudges are notifications scheduled with the OS a few minutes before each
//  calendar event, so they fire even when Tesseract is closed, busy or its
//  model is failing. The planner is pure: the agenda in, the nudges that
//  should be scheduled out. Reminders with a due time need no nudge from
//  Tesseract: they carry their own alarm, which Reminders delivers on every
//  device.
//

import Foundation

nonisolated struct Nudge: Sendable, Equatable, Hashable, Codable, Identifiable {
    /// `nudge.event.<event id>.<content hash>`: a changed title, place or
    /// lead time gives a new id, so the stale one is withdrawn.
    let id: String
    let eventID: String
    let fireAt: Date
    let title: String
    let body: String
}

nonisolated enum NudgePlanner {

    static let idPrefix = "nudge.event."

    /// The nudges that should be scheduled at `now`: one per timed event that
    /// starts within the horizon and whose nudge time is still ahead.
    static func plan(
        events: [AgendaEvent], now: Date, leadMinutes: Int, horizon: TimeInterval = 36 * 3600,
        calendar: Calendar = .current
    ) -> [Nudge] {
        let lead = TimeInterval(max(leadMinutes, 0) * 60)
        return events.compactMap { event in
            guard !event.isAllDay, event.start > now, event.start <= now.addingTimeInterval(horizon)
            else { return nil }
            let fireAt = event.start.addingTimeInterval(-lead)
            guard fireAt > now else { return nil }
            let when =
                "\(AgendaTime.clock(event.start, calendar: calendar))–\(AgendaTime.clock(event.end, calendar: calendar))"
            var body = leadMinutes > 0 ? "In \(leadMinutes) min · \(when)" : "Now · \(when)"
            if let location = event.location, !location.isEmpty { body += " · \(location)" }
            let digest = stableHash("\(event.title)|\(event.location ?? "")|\(leadMinutes)")
            return Nudge(
                id: "\(idPrefix)\(event.id).\(digest)", eventID: event.id, fireAt: fireAt,
                title: event.title, body: body)
        }
        .sorted { $0.fireAt < $1.fireAt }
    }

    /// What to change so the scheduled set matches `desired`.
    static func diff(desired: [Nudge], scheduled: Set<String>) -> (add: [Nudge], remove: [String]) {
        let wanted = Set(desired.map(\.id))
        return (
            desired.filter { !scheduled.contains($0.id) },
            scheduled.filter { !wanted.contains($0) }.sorted()
        )
    }

    /// FNV-1a, 32-bit, hex: stable across launches, unlike `hashValue`.
    static func stableHash(_ text: String) -> String {
        var hash: UInt32 = 2_166_136_261
        for byte in text.utf8 {
            hash ^= UInt32(byte)
            hash = hash &* 16_777_619
        }
        return String(hash, radix: 16)
    }
}
