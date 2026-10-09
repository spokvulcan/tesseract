//
//  CaptureParserTests.swift
//  tesseractTests
//
//  Capture as a decision table: one row per phrasing the owner might say,
//  with the reminder it must become. No model, no EventKit.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct CaptureParserTests {

    /// Wednesday 30 September 2026, 14:00 local.
    static let now = local(30, 14, 0)

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let areas = [
        Area(id: "work", name: "Work"), Area(id: "health", name: "Health"),
        Area(id: "tess", name: "Tesseract"),
    ]

    static let events = [
        AgendaEvent(
            id: "e1", title: "1:1 with Anna", start: local(30, 15, 0), end: local(30, 15, 30),
            calendarID: "c", calendarTitle: "Work"),
        AgendaEvent(
            id: "e2", title: "Design review", start: local(30, 16, 0), end: local(30, 17, 0),
            calendarID: "c", calendarTitle: "Work"),
    ]

    private func parse(_ text: String) -> CaptureIntent? {
        CaptureParser.parse(text, now: Self.now, events: Self.events, areas: Self.areas)
    }

    struct Row: CustomTestStringConvertible, Sendable {
        let said: String
        let title: String
        let due: Date?
        let hasTime: Bool
        var area: String? = nil
        var testDescription: String { said }
    }

    static let rows: [Row] = [
        Row(
            said: "remind me to call the dentist tomorrow at 10", title: "Call the dentist",
            due: local(31, 10), hasTime: true),
        Row(said: "Buy stamps", title: "Buy stamps", due: nil, hasTime: false),
        Row(
            said: "remind me to send the notes after the 1:1", title: "Send the notes",
            due: local(30, 15, 30), hasTime: true),
        Row(
            said: "after design review ping Maria", title: "Ping Maria",
            due: local(30, 17, 0), hasTime: true),
        Row(
            said: "stretch in 20 minutes", title: "Stretch",
            due: local(30, 14, 20), hasTime: true),
        Row(
            said: "take out the trash tonight", title: "Take out the trash",
            due: local(30, 20), hasTime: true),
        Row(
            said: "tomorrow morning water the plants", title: "Water the plants",
            due: local(31, 9), hasTime: true),
        Row(said: "call mom at 9", title: "Call mom", due: local(30, 21), hasTime: true),
        Row(
            said: "pay the invoice tomorrow", title: "Pay the invoice",
            due: Calendar.current.startOfDay(for: local(31, 12)), hasTime: false),
        Row(
            said: "call Anna the day after tomorrow at 3pm", title: "Call Anna",
            due: local(32, 15), hasTime: true),
        Row(
            said: "book physio #health", title: "Book physio", due: nil, hasTime: false,
            area: "Health"),
        Row(
            said: "Tesseract: ship the Today page", title: "Ship the Today page", due: nil,
            hasTime: false, area: "Tesseract"),
        Row(
            said: "don't forget to renew the passport in 3 days", title: "Renew the passport",
            due: Calendar.current.startOfDay(for: local(33, 12)), hasTime: false),
        // A bare 1 to 7 o'clock is the afternoon, not an alarm before dawn.
        Row(
            said: "call Anna tomorrow at 3", title: "Call Anna", due: local(31, 15),
            hasTime: true),
        Row(said: "ring the bank at 4", title: "Ring the bank", due: local(30, 16), hasTime: true),
        // A clock said apart from its day keeps both.
        Row(said: "call mom tonight at 9", title: "Call mom", due: local(30, 21), hasTime: true),
        Row(
            said: "call Anna at 3pm tomorrow", title: "Call Anna", due: local(31, 15),
            hasTime: true),
        // "after I …" is a clause, not an event: no due, the words kept.
        Row(
            said: "call mom after I get home", title: "Call mom after I get home", due: nil,
            hasTime: false),
        // A phrasal verb keeps its preposition; a possessive day word goes whole.
        Row(said: "log in to the portal", title: "Log in to the portal", due: nil, hasTime: false),
        Row(
            said: "tomorrow's meeting prep", title: "Meeting prep",
            due: Calendar.current.startOfDay(for: local(31, 12)), hasTime: false),
    ]

    @Test(arguments: rows)
    func phrasing(_ row: Row) throws {
        let intent = try #require(parse(row.said))
        #expect(intent.title == row.title)
        #expect(intent.due == row.due)
        #expect(intent.dueHasTime == row.hasTime)
        #expect(intent.area?.name == row.area)
    }

    @Test func fillerAloneIsNothingToCapture() {
        #expect(parse("   ") == nil)
        #expect(parse("remind me to") == nil)
    }

    @Test func pastMidnightTomorrowIsTheComingDay() throws {
        // 01:00 on Thursday 1 October is still Wednesday's night: tomorrow
        // morning is Thursday 09:00, not Friday's.
        let intent = try #require(
            CaptureParser.parse(
                "call the bank tomorrow morning", now: Self.local(31, 1), events: [],
                areas: Self.areas))
        #expect(intent.title == "Call the bank")
        #expect(intent.due == Self.local(31, 9))
    }

    @Test func anAnchorRemembersItsEvent() throws {
        let intent = try #require(parse("send the deck after the design review"))
        #expect(intent.anchorEventID == "e2")
    }
}
