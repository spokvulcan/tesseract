//
//  CompanionFlightRecorderTests.swift
//  tesseractTests
//
//  The flight recorder (#326): write/read round-trip, the v0 heartbeat
//  import, the weekly aggregator's deterministic numbers, and the
//  `log_feedback` stamping contract.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

private func scratchDirectory(_ label: String) -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("\(label)-\(UUID().uuidString)", isDirectory: true)
}

@Suite struct CompanionFlightRecorderTests {

    @Test func recordsRoundTripWithWindowFilter() throws {
        let recorder = CompanionFlightRecorder(directory: scratchDirectory("flight"))
        let wakeID = UUID()
        recorder.record(
            .wakeBooked, wakeID: wakeID, snapshot: ["class": "promise"],
            note: "ask about the dentist")
        recorder.record(.wakeFired, wakeID: wakeID)
        recorder.record(
            .deliveryNotification, source: .appObserved, wakeID: wakeID)

        let records = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(records.count == 3)
        #expect(records[0].event == "wake.booked")
        #expect(records[0].snapshot?["class"] == "promise")
        #expect(records[0].wakeID == wakeID.uuidString)
        #expect(records.allSatisfy { $0.source == "app-observed" })

        let none = recorder.records(
            since: Date().addingTimeInterval(-7200),
            until: Date().addingTimeInterval(-3600))
        #expect(none.isEmpty)
    }

    @Test func v0ImportConvertsAndRetiresTheFile() throws {
        let dir = scratchDirectory("v0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let v0 = dir.appendingPathComponent("heartbeat.jsonl")
        let ping = UUID().uuidString
        let lines = [
            #"{"ts":"2026-07-13T09:00:03+03:00","event":"fired","beat":"morning","ping":"\#(ping)","trigger":"fixed-time 09:00 daily (skeleton v0)"}"#,
            #"{"ts":"2026-07-13T09:00:09+03:00","event":"spoken","beat":"morning","ping":"\#(ping)"}"#,
            #"{"ts":"2026-07-13T13:35:00+03:00","event":"dismissed","beat":"midday","note":"busy"}"#,
        ]
        try lines.joined(separator: "\n").data(using: .utf8)!.write(to: v0)

        let recorder = CompanionFlightRecorder(directory: scratchDirectory("flight-import"))
        recorder.importV0IfNeeded(from: v0)

        let records = recorder.records(since: Date(timeIntervalSince1970: 0))
        #expect(records.count == 3)
        #expect(records[0].event == "beat.fired")
        #expect(records[0].wakeID == ping)
        #expect(records[0].snapshot?["beat"] == "morning")
        #expect(records[2].event == "beat.dismissed")
        #expect(records[2].note == "busy")

        // The source file retired; a second call is a no-op.
        #expect(!FileManager.default.fileExists(atPath: v0.path))
        #expect(
            FileManager.default.fileExists(atPath: v0.appendingPathExtension("imported").path))
        recorder.importV0IfNeeded(from: v0)
        #expect(recorder.records(since: Date(timeIntervalSince1970: 0)).count == 3)
    }

    @Test func aggregatorComputesDeterministicNumbers() {
        var records: [CompanionTraceRecord] = []
        func add(
            _ event: CompanionTraceEvent, snapshot: [String: String]? = nil, note: String? = nil
        ) {
            records.append(
                CompanionTraceRecord(
                    ts: Date().timeIntervalSince1970, event: event.rawValue, source: .appObserved,
                    snapshot: snapshot, note: note))
        }
        add(.wakeBooked, snapshot: ["class": "promise"])
        add(.wakeBooked, snapshot: ["class": "rhythm"])
        add(.wakeFired)
        add(.wakeFired)
        add(.wakeConsumed)
        add(.wakeDropped)
        add(.deliveryNotification)
        add(.deliverySpoken)
        add(.deliverySpoken)
        add(.reactionEngaged)
        add(.reactionDismissed)
        add(.turnFailed)
        add(.feedbackSolicited, note: "less pinging before noon")

        let report = CompanionWeeklyAggregator.aggregate(records)
        #expect(report.wakesBooked == 2)
        #expect(report.promisesBooked == 1)
        #expect(report.wakesFired == 2)
        #expect(report.wakesConsumed == 1)
        #expect(report.promisesDropped == 1)
        #expect(report.deliveriesByRung == ["notification": 1, "spoken": 2])
        #expect(report.reactions == ["engaged": 1, "dismissed": 1])
        #expect(report.turnFailures == 1)
        #expect(report.feedbackLines.count == 1)

        let text = CompanionWeeklyAggregator.formatted(report)
        #expect(text.contains("1 dropped"))
        #expect(text.contains("DEFECT"))
        #expect(text.contains("less pinging before noon"))
    }

    @Test func aggregatorTalliesTheNotificationHub() {
        let base = Date().timeIntervalSince1970
        var records: [CompanionTraceRecord] = []
        func add(
            _ event: CompanionTraceEvent, at offset: TimeInterval,
            snapshot: [String: String]? = nil
        ) {
            records.append(
                CompanionTraceRecord(
                    ts: base + offset, event: event.rawValue, source: .appObserved,
                    snapshot: snapshot))
        }
        // Two notifications admitted.
        add(.eventAdmitted, at: 0, snapshot: ["kind": "notification-arrived", "app": "Mail"])
        add(.eventAdmitted, at: 300, snapshot: ["kind": "notification-arrived", "app": "Slack"])
        // Mail: he switched to Mail three minutes later, nothing escalated
        // between — a plausible miss.
        add(.eventAdmitted, at: 180, snapshot: ["kind": "app-switch", "app": "Mail"])
        // Slack: escalated before he switched to it — NOT a miss.
        add(.deliverySpoken, at: 320)
        add(.eventAdmitted, at: 380, snapshot: ["kind": "app-switch", "app": "Slack"])
        // One held notification, and one escalation reaction.
        add(.holdTracked, at: 60, snapshot: ["app": "Calendar"])
        add(.reactionEngaged, at: 130)

        let report = CompanionWeeklyAggregator.aggregate(records)
        #expect(report.notificationsAdmitted == 2)
        #expect(report.trackedHolds == 1)
        #expect(report.inferredMissCandidates == 1)  // Mail only
        #expect(report.reactions == ["engaged": 1])

        let text = CompanionWeeklyAggregator.formatted(report)
        #expect(text.contains("Notification Hub: 2 admitted"))
        #expect(text.contains("1 held (tracked)"))
        #expect(text.contains("1 inferred-miss candidate "))
    }

    @Test func inferredMissCountsEachNotificationOnceAndSkipsLateSwitches() {
        let base = Date().timeIntervalSince1970
        var records: [CompanionTraceRecord] = []
        func add(
            _ event: CompanionTraceEvent, at offset: TimeInterval,
            snapshot: [String: String]? = nil
        ) {
            records.append(
                CompanionTraceRecord(
                    ts: base + offset, event: event.rawValue, source: .appObserved,
                    snapshot: snapshot))
        }
        add(.eventAdmitted, at: 0, snapshot: ["kind": "notification-arrived", "app": "Mail"])
        // Two later switches to Mail — the notification is one candidate, not two.
        add(.eventAdmitted, at: 120, snapshot: ["kind": "app-switch", "app": "Mail"])
        add(.eventAdmitted, at: 200, snapshot: ["kind": "app-switch", "app": "Mail"])
        // A switch past the window doesn't count.
        add(.eventAdmitted, at: 1000, snapshot: ["kind": "notification-arrived", "app": "News"])
        add(
            .eventAdmitted, at: 1000 + 900,
            snapshot: ["kind": "app-switch", "app": "News"])

        let report = CompanionWeeklyAggregator.aggregate(records)
        #expect(report.notificationsAdmitted == 2)
        #expect(report.inferredMissCandidates == 1)  // Mail once; News too late
    }

    @Test func traceEventRawValuesAreUnique() {
        // The rawValue is the wire string every producer and the aggregator
        // share; a collision is silent data corruption (#393).
        let rawValues = CompanionTraceEvent.allCases.map(\.rawValue)
        #expect(Set(rawValues).count == rawValues.count)
    }

    @Test func inferredMissKeysOffTheSwitchStartNotTheSessionClose() {
        let base = Date().timeIntervalSince1970
        var records: [CompanionTraceRecord] = []
        func add(
            _ event: CompanionTraceEvent, at offset: TimeInterval,
            snapshot: [String: String]? = nil
        ) {
            records.append(
                CompanionTraceRecord(
                    ts: base + offset, event: event.rawValue, source: .appObserved,
                    snapshot: snapshot))
        }
        // He was already in Slack when the ping arrived: the session started
        // before the notification (`at` = base − 200) and closed after it
        // (record stamp = base + 300, inside the window). Keying off the close
        // would wrongly score it a miss; keying off the start must not.
        add(.eventAdmitted, at: 0, snapshot: ["kind": "notification-arrived", "app": "Slack"])
        add(
            .eventAdmitted, at: 300,
            snapshot: ["kind": "app-switch", "app": "Slack", "at": String(Int(base - 200))])
        // Mail he genuinely switched *to* after its ping — start is after it.
        add(.eventAdmitted, at: 0, snapshot: ["kind": "notification-arrived", "app": "Mail"])
        add(
            .eventAdmitted, at: 400,
            snapshot: ["kind": "app-switch", "app": "Mail", "at": String(Int(base + 90))])

        let report = CompanionWeeklyAggregator.aggregate(records)
        #expect(report.inferredMissCandidates == 1)  // Mail (switched to), not Slack (already in)
    }
}

@Suite struct CompanionFlightRecorderToolTests {

    private func text(_ result: AgentToolResult) -> String {
        result.content.compactMap { block -> String? in
            if case .text(let string) = block { return string }
            return nil
        }.joined(separator: "\n")
    }

    @Test func logFeedbackStampsModelReportedAndConversation() async throws {
        let recorder = CompanionFlightRecorder(directory: scratchDirectory("flight-tool"))
        let conversationID = UUID()
        let tool = createLogFeedbackTool(
            recorder: recorder, currentConversationID: { conversationID })

        let result = try await tool.execute(
            "call",
            [
                "kind": .string("solicited"),
                "verbatim": .string("the midday pulse was noise today"),
            ], nil, nil)
        #expect(text(result).contains("noise"))

        let records = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(records.count == 1)
        #expect(records[0].event == "feedback.solicited")
        #expect(records[0].source == "model-reported")
        #expect(records[0].conversationID == conversationID.uuidString)
        #expect(records[0].note == "the midday pulse was noise today")
    }

}
