//
//  CompanionCalendarReader.swift
//  tesseract
//
//  The Companion's read-through into the owner's calendar (map #301 stage G):
//  strictly read-only, strictly briefing material. EventKit access is asked
//  for when the Companion is switched on — never on launch — and a denial
//  degrades to "no calendar in the briefing", never to an error the owner
//  has to deal with.
//

import EventKit
import Foundation

@MainActor
final class CompanionCalendarReader {

    private let store = EKEventStore()
    private var accessAsked = false

    /// Ask once per launch, only when the Companion is actually on. Returns
    /// whether events are readable.
    func requestAccessIfNeeded() async -> Bool {
        switch EKEventStore.authorizationStatus(for: .event) {
        case .fullAccess:
            return true
        case .notDetermined:
            guard !accessAsked else { return false }
            accessAsked = true
            let granted = (try? await store.requestFullAccessToEvents()) ?? false
            Log.companion.info("Calendar access asked: \(granted ? "granted" : "denied")")
            return granted
        default:
            return false
        }
    }

    var hasAccess: Bool {
        EKEventStore.authorizationStatus(for: .event) == .fullAccess
    }

    /// The briefing's calendar block: what remains of today, and tomorrow
    /// morning when the day is winding down. Plain lines, local wall time —
    /// interpretation is the turn's job.
    func briefingLines(now: Date = Date()) -> [String] {
        guard hasAccess else { return [] }
        let calendar = Calendar.current
        guard let endOfDay = calendar.date(bySettingHour: 23, minute: 59, second: 59, of: now)
        else { return [] }

        var lines: [String] = []
        let today = events(from: now, to: endOfDay)
        if !today.isEmpty {
            lines.append("Calendar — the rest of his day:")
            lines.append(contentsOf: today)
        }

        // After 17:00 the evening journal wants tomorrow's shape too.
        if calendar.component(.hour, from: now) >= 17,
            let tomorrowStart = calendar.date(
                byAdding: .day, value: 1, to: calendar.startOfDay(for: now)),
            let tomorrowNoon = calendar.date(
                bySettingHour: 13, minute: 0, second: 0, of: tomorrowStart)
        {
            let tomorrow = events(from: tomorrowStart, to: tomorrowNoon)
            if !tomorrow.isEmpty {
                lines.append("Calendar — tomorrow morning:")
                lines.append(contentsOf: tomorrow)
            }
        }
        return lines
    }

    private func events(from start: Date, to end: Date) -> [String] {
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        return store.events(matching: predicate)
            .filter { !$0.isAllDay }
            .sorted { $0.startDate < $1.startDate }
            .prefix(8)
            .map { event in
                let time = event.startDate.formatted(date: .omitted, time: .shortened)
                return "- \(time) \(event.title ?? "(untitled)")"
            }
    }
}
