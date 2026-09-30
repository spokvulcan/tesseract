//
//  CompanionTraceTests.swift
//  tesseractTests
//
//  The Companion Trace over a scratch directory: a closed vocabulary, every
//  record stamped with its Day Thread, and field values kept as JSON scalars
//  so later analysis reads numbers as numbers.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct CompanionTraceTests {

    private var berlin: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Berlin")!
        return calendar
    }

    private func date(_ y: Int, _ m: Int, _ d: Int, _ h: Int, _ min: Int) -> Date {
        berlin.date(from: DateComponents(year: y, month: m, day: d, hour: h, minute: min))!
    }

    @Test func vocabularyIsClosedAndDotNamespaced() {
        let names = CompanionTraceEvent.allCases.map(\.rawValue)
        #expect(Set(names).count == names.count)
        for name in names {
            #expect(name.split(separator: ".").count == 2, "\(name) is not family.event")
        }
        for required in [
            "moment.started", "moment.finished", "moment.failed", "card.presented",
            "card.reaction", "nudge.scheduled", "nudge.fired", "notification.arrived",
            "notification.seen", "notification.triaged", "agent.signal", "agenda.changed",
            "fact.proposed", "fact.decided", "profile.changed", "thread.opened",
            "thread.compacted", "governor.deferred", "migration.wiped",
        ] {
            #expect(CompanionTraceEvent(rawValue: required) != nil, "\(required) missing")
        }
    }

    @Test func recordsRoundTripWithTheirDayThreadAndTypedFields() throws {
        let trace = CompanionTrace(directory: makeTempDir("trace"), calendar: berlin)
        let at = date(2026, 9, 30, 9, 15)
        let conversation = UUID()
        trace.record(
            .momentFinished, at: at, conversationID: conversation,
            fields: [
                "moment": "morningPlan", "promptTokens": 1834, "latencySeconds": 4.5,
                "cached": true,
            ])

        let records = trace.records(
            since: at.addingTimeInterval(-60), until: at.addingTimeInterval(60))
        let record = try #require(records.first)
        #expect(records.count == 1)
        #expect(record.traceEvent == .momentFinished)
        #expect(record.thread == "2026-09-30")
        #expect(record.conversationID == conversation.uuidString)
        #expect(record.fields?["promptTokens"] == .int(1834))
        #expect(record.fields?["latencySeconds"] == .double(4.5))
        #expect(record.fields?["cached"] == .bool(true))
        #expect(record.fields?["moment"] == .string("morningPlan"))
    }

    /// The day rolls over at 04:00: a wrap-up at 01:30 belongs to the day
    /// that is ending.
    @Test func eventsBeforeFourBelongToThePreviousDay() {
        let trace = CompanionTrace(directory: makeTempDir("trace"), calendar: berlin)
        let late = date(2026, 10, 1, 1, 30)
        trace.record(.cardPresented, at: late)
        let record = trace.records(
            since: late.addingTimeInterval(-1), until: late.addingTimeInterval(1))
        #expect(record.first?.thread == "2026-09-30")
    }
}

struct DayKeyTests {

    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Berlin")!
        return calendar
    }

    private func at(_ d: Int, _ h: Int, _ m: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: 9, day: d, hour: h, minute: m))!
    }

    @Test(arguments: [
        (30, 3, 59, "2026-09-29"),
        (30, 4, 0, "2026-09-30"),
        (30, 23, 59, "2026-09-30"),
    ])
    func rollsOverAtFour(day: Int, hour: Int, minute: Int, expected: String) {
        #expect(DayKey(for: at(day, hour, minute), calendar: calendar).rawValue == expected)
    }

    @Test func startEndAndNext() throws {
        let key = DayKey(rawValue: "2026-09-30")
        #expect(key.start(calendar: calendar) == at(30, 4))
        #expect(
            key.end(calendar: calendar) == calendar.date(byAdding: .day, value: 1, to: at(30, 4)))
        #expect(key.next(calendar: calendar).rawValue == "2026-10-01")
        #expect(DayKey(rawValue: "2026-09-29") < key)
    }
}
