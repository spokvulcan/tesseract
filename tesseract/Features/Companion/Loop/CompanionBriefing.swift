//
//  CompanionBriefing.swift
//  tesseract
//
//  The situation briefing (ADR-0040): the code-gathered context handed to the
//  entity at the start of every turn — time, presence, frontmost app, power,
//  contract state, its own due and upcoming wakes. Gathering is mechanical;
//  interpreting it is the turn's job. No judgment lives here.
//

import AppKit
import Foundation

@MainActor
enum CompanionBriefing {

    struct Inputs {
        let now: Date
        let ownerPresent: Bool
        let screenLocked: Bool
        let frontmostApp: String?
        let onACPower: Bool
        let today: DayRecord?
        let yesterday: DayRecord?
        let dueWakes: [CompanionWake]
        let upcomingWakes: [CompanionWake]
        /// Sunday-evening turns carry the deterministic weekly numbers (#313).
        let weeklyNumbers: String?
    }

    static func gather(
        store: MemoryStore,
        idleMonitor: IdleMonitor,
        sensed: SensedObservationRecorder,
        dueWakes: [CompanionWake],
        recorder: CompanionFlightRecorder,
        now: Date = Date()
    ) async -> Inputs {
        let todayKey = TrackingDay.key(for: now)
        let yesterdayKey = TrackingDay.yesterdayKey(from: now)
        let today = try? await store.day(todayKey)
        let yesterday = try? await store.day(yesterdayKey)
        let upcoming = (try? await store.upcomingWakes(after: now)) ?? []

        // The Sunday evening journal extends into the weekly review (#313):
        // code computes the numbers, the turn narrates them.
        var weekly: String?
        let weekday = Calendar.current.component(.weekday, from: now)
        let hour = Calendar.current.component(.hour, from: now)
        if weekday == 1, hour >= 17 {
            let records = recorder.records(since: now.addingTimeInterval(-7 * 86_400), until: now)
            if !records.isEmpty {
                weekly = CompanionWeeklyAggregator.formatted(
                    CompanionWeeklyAggregator.aggregate(records))
            }
        }

        return Inputs(
            now: now,
            ownerPresent: !idleMonitor.isIdle && !idleMonitor.isScreenLocked,
            screenLocked: idleMonitor.isScreenLocked,
            frontmostApp: NSWorkspace.shared.frontmostApplication?.localizedName,
            onACPower: sensed.isOnACPower,
            today: today ?? nil,
            yesterday: yesterday ?? nil,
            dueWakes: dueWakes,
            upcomingWakes: upcoming,
            weeklyNumbers: weekly
        )
    }

    /// The `<situation>` block — plain, factual, timestamped. The entity reads
    /// this the way a pilot reads instruments.
    static func render(_ inputs: Inputs) -> String {
        var lines: [String] = []
        lines.append(
            "Time: \(inputs.now.formatted(date: .complete, time: .shortened))")
        if inputs.screenLocked {
            lines.append("He is away — the screen is locked.")
        } else if inputs.ownerPresent {
            var presence = "He is at the Mac"
            if let app = inputs.frontmostApp { presence += ", in \(app)" }
            lines.append(presence + ".")
        } else {
            lines.append("He is idle — no input for a while.")
        }
        lines.append("Power: \(inputs.onACPower ? "AC" : "battery").")

        if let yesterday = inputs.yesterday {
            if yesterday.closedAt == nil {
                lines.append("Yesterday was never closed — no journal, no close-out.")
            }
            if let keystone = yesterday.keystone {
                lines.append(
                    "Yesterday's keystone: \"\(keystone.title)\" — \(keystone.status.rawValue).")
            }
            if let seed = yesterday.seed {
                lines.append("Seed left for today: \(seed)")
            }
        }
        if let today = inputs.today, !today.chain.isEmpty {
            lines.append(today.chainSummary)
        } else {
            lines.append("No contract for today yet.")
        }

        if !inputs.dueWakes.isEmpty {
            lines.append("DUE NOW — the wakes that triggered this turn:")
            for wake in inputs.dueWakes {
                let overdue = Int(inputs.now.timeIntervalSince(wake.due) / 60)
                let lateness = overdue > 5 ? " (overdue by \(overdue) min)" : ""
                lines.append(
                    "- [\(wake.wakeClass.rawValue)] \(wake.content)\(lateness)")
            }
        }
        if !inputs.upcomingWakes.isEmpty {
            lines.append("Your booked future:")
            for wake in inputs.upcomingWakes {
                lines.append(
                    "- \(wake.due.formatted(date: .abbreviated, time: .shortened)) "
                        + "[\(wake.wakeClass.rawValue)] \(wake.content)")
            }
        } else {
            lines.append("You have NOTHING booked ahead — establish your rhythm.")
        }
        if let weekly = inputs.weeklyNumbers {
            lines.append("")
            lines.append(weekly)
        }
        return "<situation>\n\(lines.joined(separator: "\n"))\n</situation>"
    }
}
