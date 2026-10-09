//
//  Delivery.swift
//  tesseract
//
//  The Delivery Ladder and the compute governor, both pure.
//
//  The ladder: the model decides what matters; code decides how it reaches
//  the owner — from importance, presence, the app in front, and quiet hours.
//  Glyph for anything waiting; the Jarvis panel for Breakpoints and urgent
//  items when the owner is at the Mac; a banner when away or locked; voice
//  for urgent items when present and voice is on. Quiet hours silence
//  Jarvis's own deliveries (the owner's reminders and nudges still fire).
//  Nothing unanswered is ever re-summoned; it stays in Today.
//
//  The governor: no fixed budget, usefulness first — but non-urgent model
//  work waits while the Mac is hot or low on battery. The Night Reflection
//  runs on power, or on a battery at least half full with a cool Mac.
//

import Foundation

nonisolated enum Importance: String, Sendable, Equatable, Codable {
    /// A Breakpoint card, a wrap-up: worth a look when convenient.
    case normal
    /// Can't wait for the next break: a person waiting now, an agent stuck.
    case urgent
}

nonisolated enum DeliveryLadder {

    /// Apps in which a panel or a spoken line would interrupt the owner
    /// mid-call or mid-presentation.
    static let interruptionFreeApps: Set<String> = [
        "us.zoom.xos", "com.apple.FaceTime", "com.microsoft.teams2", "com.microsoft.teams",
        "com.cisco.webexmeetingsapp", "com.apple.iWork.Keynote",
    ]

    /// - Parameter sittingDown: the owner just sat down to start the day (the
    ///   first sit-down after the night): for them quiet hours are over.
    static func rungs(for importance: Importance, snapshot: DaySnapshot, sittingDown: Bool = false)
        -> [DeliveryRung]
    {
        if isQuietHours(snapshot), !sittingDown { return [.today] }
        guard snapshot.ownerPresent else {
            // Away or locked: a banner waits on the lock screen; normal cards
            // wait in Today.
            return importance == .urgent ? [.banner] : [.today]
        }
        if snapshot.frontmostIsGame
            || interruptionFreeApps.contains(snapshot.frontmostBundleID ?? "")
        {
            return importance == .urgent ? [.banner] : [.today]
        }
        switch importance {
        case .normal:
            return [.panel]
        case .urgent:
            return snapshot.settings.speaks ? [.panel, .voice] : [.panel, .banner]
        }
    }

    /// The owner sat down to start the day and it is still the morning part
    /// of quiet hours: for them, quiet hours are over.
    static func dayStarted(_ satDownAt: Date?, snapshot: DaySnapshot) -> Bool {
        guard let satDownAt, satDownAt <= snapshot.now,
            snapshot.now.timeIntervalSince(satDownAt) < 12 * 3600
        else { return false }
        return snapshot.minuteOfDay < snapshot.settings.quietEndMinutes
    }

    static func isQuietHours(_ snapshot: DaySnapshot) -> Bool {
        let start = snapshot.settings.quietStartMinutes
        let end = snapshot.settings.quietEndMinutes
        let minute = snapshot.minuteOfDay
        if start == end { return false }
        return start < end ? (minute >= start && minute < end) : (minute >= start || minute < end)
    }
}

// MARK: - Governor

/// The Mac's power and heat, as the governor reads them.
nonisolated struct PowerState: Sendable, Equatable {
    enum Thermal: Int, Sendable, Comparable {
        case nominal, fair, serious, critical
        static func < (lhs: Thermal, rhs: Thermal) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    var onACPower: Bool
    /// 0–100, nil on a desktop.
    var batteryPercent: Int?
    var thermal: Thermal

    static let nominal = PowerState(onACPower: true, batteryPercent: nil, thermal: .nominal)
}

nonisolated enum Governor {

    /// The Night Reflection runs on a battery at least this full.
    static let nightReflectionMinimumBattery = 50

    /// Why a moment must wait, or nil when it may run now.
    static func deferral(for kind: MomentKind, power: PowerState) -> String? {
        switch kind {
        case .nightReflection:
            if power.onACPower {
                return power.thermal > .fair ? "thermal \(power.thermal)" : nil
            }
            guard let battery = power.batteryPercent, battery >= nightReflectionMinimumBattery
            else { return "battery \(power.batteryPercent.map { "\($0)%" } ?? "unknown")" }
            return power.thermal > .nominal ? "thermal \(power.thermal)" : nil
        case .triage:
            if power.thermal >= .serious { return "thermal \(power.thermal)" }
            if !power.onACPower, let battery = power.batteryPercent, battery < 20 {
                return "battery \(battery)%"
            }
            return nil
        case .morningPlan, .breakpoint, .eveningWrapUp:
            return nil
        }
    }
}
