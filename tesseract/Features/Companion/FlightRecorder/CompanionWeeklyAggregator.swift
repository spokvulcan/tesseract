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
        var turnFailures = 0
        var toggleOffIncidents = 0
        /// The Notification Hub's evidence (#380), counts only — the model
        /// weighs escalation precision, code never does.
        var notificationsAdmitted = 0
        /// `track(kind: hold)` verdicts — important-but-held, the one triage
        /// call the free record can't otherwise see.
        var trackedHolds = 0
        /// An unescalated notification followed, within the window, by a
        /// sustained switch to its source app — a plausible missed call.
        var inferredMissCandidates = 0
        /// Verbatim feedback lines, dated — the qualitative half rides along
        /// untallied.
        var feedbackLines: [String] = []
    }

    /// The window in which a switch to a notification's own source app, with no
    /// escalation between, reads as a plausibly-missed call (#380). A proxy the
    /// aggregator counts, never a verdict it renders.
    static let inferredMissWindow: TimeInterval = 600

    static func aggregate(_ records: [CompanionTraceRecord]) -> Report {
        var report = Report()
        // Correlation material for the inferred-miss tally, gathered in one
        // pass and paired after: notifications admitted, sustained app
        // switches, and every escalation's time.
        var notifications: [(app: String, ts: Double)] = []
        var appSwitches: [(app: String, ts: Double)] = []
        var deliveryTimes: [Double] = []
        for record in records {
            // Names not in the vocabulary (legacy v0 `beat.*`, a future
            // version's events) are skipped by name — forward-compat (#393).
            guard let event = CompanionTraceEvent(rawValue: record.event) else { continue }
            // Exhaustive on purpose: a new event family forces a decision here,
            // producer and reader sharing one symbol (#393). The `.rawValue`
            // strings the report emits (rung, reaction kind) are derived here so
            // the wire text is unchanged.
            switch event {
            case .wakeBooked:
                report.wakesBooked += 1
                if record.snapshot?["class"] == "promise" { report.promisesBooked += 1 }
            case .wakeFired:
                report.wakesFired += 1
            case .wakeConsumed:
                report.wakesConsumed += 1
            case .wakeDeliveredUnheard:
                report.promisesDeliveredUnheard += 1
            case .wakeDropped:
                report.promisesDropped += 1
            case .turnFailed:
                report.turnFailures += 1
            case .feedbackToggleOff:
                report.toggleOffIncidents += 1
                appendFeedbackLine(record, to: &report)
            case .eventAdmitted:
                switch record.snapshot?["kind"] {
                case "notification-arrived":
                    report.notificationsAdmitted += 1
                    if let app = record.snapshot?["app"] {
                        notifications.append((app: app, ts: record.ts))
                    }
                case "app-switch":
                    if let app = record.snapshot?["app"] {
                        // Key off when he switched *to* the app (the session's
                        // start, `at`), not the record's stamp — that lands at
                        // the session's close, which would score the app he was
                        // already in when the ping arrived as a miss (#380).
                        let switchedTo = record.snapshot?["at"].flatMap(Double.init) ?? record.ts
                        appSwitches.append((app: app, ts: switchedTo))
                    }
                default:
                    break
                }
            case .holdTracked:
                report.trackedHolds += 1
            // Deliveries — the rung is the case; the ladder shares the tally,
            // and every rung feeds the inferred-miss suppression window (#380).
            case .deliveryFallback:
                recordDelivery("fallback", ts: record.ts, into: &report, times: &deliveryTimes)
            case .deliveryNotification:
                recordDelivery("notification", ts: record.ts, into: &report, times: &deliveryTimes)
            case .deliverySpoken:
                recordDelivery("spoken", ts: record.ts, into: &report, times: &deliveryTimes)
            case .deliveryGlyph:
                recordDelivery("glyph", ts: record.ts, into: &report, times: &deliveryTimes)
            case .deliverySummons:
                recordDelivery("summons", ts: record.ts, into: &report, times: &deliveryTimes)
            // Reactions — the kind is the case.
            case .reactionEngaged:
                report.reactions["engaged", default: 0] += 1
            case .reactionReplied:
                report.reactions["replied", default: 0] += 1
            case .reactionDismissed:
                report.reactions["dismissed", default: 0] += 1
            case .reactionUnheard:
                report.reactions["unheard", default: 0] += 1
            case .reactionBargeIn:
                report.reactions["barge-in", default: 0] += 1
            // Feedback testimony — the verbatim note rides along, untallied.
            case .feedbackSolicited, .feedbackSpontaneous, .feedbackFabricationFlag,
                .feedbackAnnoyance, .feedbackDialChange:
                appendFeedbackLine(record, to: &report)
            // Recorded for replay, not the weekly numbers.
            case .wakeRevised, .wakeCancelled, .wakeRepresented, .wakeResurfaced,
                .eventRepresented,
                .instructionsSeeded, .instructionsRevised, .instructionsSleepReview,
                .instructionsOwnerEdited,
                .turnStarted, .turnDeferred, .turnCompleted,
                .dialogueBegan, .dialogueSuperseded, .dialogueNudged, .dialogueNudgeMissed,
                .digestRejected, .digestFolded, .digestFailed,
                .glyphChanged, .glyphNoticeCleared,
                .loopAuthDenied, .loopCalendarDenied,
                .reportBackDeposited,
                .voiceSessionEntered, .voiceSessionExited, .voiceReplySpoken, .voiceOwnerTurn,
                .voiceWatchdogExit, .voiceBargeSoftOnset, .voiceBargeFalseResume,
                .voiceEnergySample, .voiceBargeSuppressed:
                break
            }
        }
        report.inferredMissCandidates = inferredMisses(
            notifications: notifications, appSwitches: appSwitches,
            deliveryTimes: deliveryTimes)
        return report
    }

    /// One delivery rung: the by-rung tally plus the shared inferred-miss
    /// suppression window (every delivery, whatever the rung, counts).
    private static func recordDelivery(
        _ rung: String, ts: Double, into report: inout Report, times: inout [Double]
    ) {
        report.deliveriesByRung[rung, default: 0] += 1
        times.append(ts)
    }

    /// A dated verbatim feedback line, when the record carried a note.
    private static func appendFeedbackLine(_ record: CompanionTraceRecord, to report: inout Report)
    {
        guard let note = record.note else { return }
        let day = Date(timeIntervalSince1970: record.ts)
            .formatted(date: .abbreviated, time: .omitted)
        report.feedbackLines.append("(\(day)) \(note)")
    }

    /// A notification is a plausible miss when he switched to its source app
    /// within the window and nothing escalated between the two — counted once
    /// per notification, however many switches followed. Case-folded app match
    /// (the tree gives a display name, never a bundle id). Deliveries aren't
    /// notification-tagged (the ladder is shared, by design — zero new tools),
    /// so *any* escalation in the gap suppresses the count: the tally
    /// deliberately under-reports rather than claim a miss the entity may have
    /// caught. A candidate, never a verdict.
    private static func inferredMisses(
        notifications: [(app: String, ts: Double)],
        appSwitches: [(app: String, ts: Double)],
        deliveryTimes: [Double]
    ) -> Int {
        notifications.reduce(into: 0) { count, notification in
            let app = notification.app.lowercased()
            let missed = appSwitches.contains { appSwitch in
                appSwitch.app.lowercased() == app
                    && appSwitch.ts > notification.ts
                    && appSwitch.ts <= notification.ts + inferredMissWindow
                    && !deliveryTimes.contains {
                        $0 >= notification.ts && $0 <= appSwitch.ts
                    }
            }
            if missed { count += 1 }
        }
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
        if report.notificationsAdmitted > 0 || report.trackedHolds > 0
            || report.inferredMissCandidates > 0
        {
            lines.append(
                "Notification Hub: \(report.notificationsAdmitted) admitted, "
                    + "\(report.trackedHolds) held (tracked), "
                    + "\(report.inferredMissCandidates) inferred-miss candidate"
                    + (report.inferredMissCandidates == 1 ? "" : "s")
                    + " (unescalated, then he switched to that app). "
                    + "Escalation reactions are in the deliveries/reactions lines above.")
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
