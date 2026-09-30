//
//  AgendaTimeTests.swift
//  tesseractTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct AgendaTimeTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0, month: Int = 9) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: month, day: day, hour: hour, minute: minute))!
    }

    static let now = local(30, 9)

    @Test(arguments: [
        ("2026-10-01T10:00", local(1, 10, month: 10), true),
        ("2026-10-01 10:00", local(1, 10, month: 10), true),
        ("2026-10-01", local(1, 0, month: 10), false),
        ("today 20:00", local(30, 20), true),
        ("tomorrow at 7:30", local(1, 7, 30, month: 10), true),
        ("tomorrow", local(1, 0, month: 10), false),
        ("18:15", local(30, 18, 15), true),
    ])
    func parses(_ text: String, _ date: Date, _ hasTime: Bool) {
        #expect(AgendaTime.parse(text, now: Self.now) == .init(date: date, hasTime: hasTime))
    }

    @Test(arguments: ["", "soon", "2026-13-40", "25:00", "next week"])
    func rejects(_ text: String) {
        #expect(AgendaTime.parse(text, now: Self.now) == nil)
    }

    @Test func describesRelativeToNow() {
        #expect(
            AgendaTime.describe(Self.local(30, 20), hasTime: true, now: Self.now) == "today 20:00")
        #expect(
            AgendaTime.describe(Self.local(1, 9, month: 10), hasTime: false, now: Self.now)
                == "tomorrow")
        #expect(
            AgendaTime.describe(Self.local(3, 11, month: 10), hasTime: true, now: Self.now)
                == "Saturday 3 October 11:00")
    }
}
