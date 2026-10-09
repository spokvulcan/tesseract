//
//  Nudges.swift
//  tesseract
//
//  Nudges are notifications scheduled with the OS a few minutes before each
//  calendar event — and at the time the Morning Plan set to leave for one in
//  person — so they fire even when Tesseract is closed, busy or its model is
//  failing. The planner is pure: the agenda in, the nudges that
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
    /// A call's link: the nudge offers to join it.
    var link: URL? = nil
}

nonisolated enum NudgePlanner {

    static let idPrefix = "nudge.event."
    /// A departure's: `nudge.leave.<event id>.<content hash>`.
    static let leavePrefix = "nudge.leave."
    /// Every nudge Tesseract schedules, whichever kind.
    static let familyPrefix = "nudge."

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
            if let place = event.place { body += " · \(place)" }
            let link = event.meetingLink
            // Its end and link too: a meeting made longer, or a new link, is
            // nudged again with what is true now.
            let digest = stableHash(
                "\(event.title)|\(event.location ?? "")|\(leadMinutes)|\(Int(event.end.timeIntervalSince1970))|\(link?.absoluteString ?? "")"
            )
            return Nudge(
                id: "\(idPrefix)\(event.id).\(digest)", eventID: event.id, fireAt: fireAt,
                title: event.title, body: body, link: link)
        }
        .sorted { $0.fireAt < $1.fireAt }
    }

    /// The departures still ahead, for events still on the calendar: "Time
    /// to leave for Class" at the time the plan set.
    static func plan(
        departures: [Departure], events: [AgendaEvent], now: Date, calendar: Calendar = .current
    ) -> [Nudge] {
        departures.compactMap { departure in
            guard departure.at > now,
                let event = events.first(where: { $0.id == departure.eventID }),
                event.start == departure.eventStart
            else { return nil }
            var body = "It starts at \(AgendaTime.clock(event.start, calendar: calendar))"
            if let place = event.place { body += " · \(place)" }
            let digest = stableHash(
                "\(event.title)|\(event.location ?? "")|\(departure.at.timeIntervalSince1970)")
            return Nudge(
                id: "\(leavePrefix)\(event.id).\(digest)", eventID: event.id,
                fireAt: departure.at, title: "Time to leave for \(event.title)", body: body)
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
