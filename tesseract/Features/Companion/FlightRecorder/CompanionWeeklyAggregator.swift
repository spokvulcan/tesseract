//
//  CompanionWeeklyAggregator.swift
//  tesseract
//
//  The slow loop's deterministic half (#326, #313): code computes the weekly
//  numbers over the flight-recorder traces and hands them to the model —
//  the model narrates, it never tallies. Powers the Sunday weekly-review
//  edition of the evening journal.
//

import Foundation

nonisolated enum CompanionWeeklyAggregator {

    struct Report: Sendable {
        var wakesBooked = 0
        var wakesFired = 0
        var wakesConsumed = 0
        var promisesBooked = 0
        var promisesDeliveredUnheard = 0
        /// The zero-silent-drops defect signal (#313): any `wake.dropped`
        /// presence is a defect, not a quality issue.
        var promisesDropped = 0
        var deliveriesByRung: [String: Int] = [:]
        var reactions: [String: Int] = [:]
        var callbackVerdicts: [String: Int] = [:]
        var turnFailures = 0
        var toggleOffIncidents = 0
        /// Verbatim feedback lines, dated — the qualitative half rides along
        /// untallied.
        var feedbackLines: [String] = []
    }

    static func aggregate(_ records: [CompanionTraceRecord]) -> Report {
        var report = Report()
        for record in records {
            switch record.event {
            case "wake.booked":
                report.wakesBooked += 1
                if record.snapshot?["class"] == "promise" { report.promisesBooked += 1 }
            case "wake.fired":
                report.wakesFired += 1
            case "wake.consumed":
                report.wakesConsumed += 1
            case "wake.delivered-unheard":
                report.promisesDeliveredUnheard += 1
            case "wake.dropped":
                report.promisesDropped += 1
            case "turn.failed":
                report.turnFailures += 1
            case "feedback.toggle-off":
                report.toggleOffIncidents += 1
            case "callback.delivered":
                let verdict = record.snapshot?["verdict"] ?? "unknown"
                report.callbackVerdicts[verdict, default: 0] += 1
            default:
                break
            }
            if record.event.hasPrefix("delivery.") {
                let rung = String(record.event.dropFirst("delivery.".count))
                report.deliveriesByRung[rung, default: 0] += 1
            }
            if record.event.hasPrefix("reaction.") {
                let kind = String(record.event.dropFirst("reaction.".count))
                report.reactions[kind, default: 0] += 1
            }
            if record.event.hasPrefix("feedback."), let note = record.note {
                let day = Date(timeIntervalSince1970: record.ts)
                    .formatted(date: .abbreviated, time: .omitted)
                report.feedbackLines.append("(\(day)) \(note)")
            }
        }
        return report
    }

    /// The text the Sunday review's turn receives — numbers computed here,
    /// narration left entirely to the model.
    static func formatted(_ report: Report) -> String {
        var lines = ["WEEKLY NUMBERS (computed by code from the flight recorder):"]
        lines.append(
            "Wakes: \(report.wakesBooked) booked, \(report.wakesFired) fired, "
                + "\(report.wakesConsumed) consumed by a completed turn.")
        lines.append(
            "Promises: \(report.promisesBooked) booked, "
                + "\(report.promisesDeliveredUnheard) delivered-unheard, "
                + "\(report.promisesDropped) dropped"
                + (report.promisesDropped > 0 ? "  ← DEFECT: a dropped promise is never OK" : ".")
        )
        if !report.deliveriesByRung.isEmpty {
            let rungs = report.deliveriesByRung.sorted { $0.key < $1.key }
                .map { "\($0.key) \($0.value)" }.joined(separator: ", ")
            lines.append("Deliveries by rung: \(rungs).")
        }
        if !report.reactions.isEmpty {
            let reactions = report.reactions.sorted { $0.key < $1.key }
                .map { "\($0.key) \($0.value)" }.joined(separator: ", ")
            lines.append("Owner reactions: \(reactions).")
        }
        if !report.callbackVerdicts.isEmpty {
            let verdicts = report.callbackVerdicts.sorted { $0.key < $1.key }
                .map { "\($0.key) \($0.value)" }.joined(separator: ", ")
            lines.append("Callback verdicts: \(verdicts).")
        }
        if report.turnFailures > 0 {
            lines.append("Turn failures: \(report.turnFailures).")
        }
        if report.toggleOffIncidents > 0 {
            lines.append("Toggle-off incidents: \(report.toggleOffIncidents) ← incident review.")
        }
        if !report.feedbackLines.isEmpty {
            lines.append("His feedback, verbatim:")
            lines.append(contentsOf: report.feedbackLines.map { "- \($0)" })
        }
        return lines.joined(separator: "\n")
    }
}
