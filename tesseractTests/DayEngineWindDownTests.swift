//
//  DayEngineWindDownTests.swift
//  tesseractTests
//
//  The wind-down as a decision table: as quiet hours begin with the owner at
//  the Mac, one banner a night says when tomorrow starts — within the first
//  hour of quiet hours, midnight or not; never while away, in a game or a
//  call, with quiet hours off, or with the owner's setting off.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineWindDownTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let allHands = AgendaEvent(
        id: "all-hands", title: "All Hands", start: local(31, 7, 30), end: local(31, 8, 30),
        calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)

    static func snapshot(
        at now: Date, present: Bool = true, events: [AgendaEvent] = [allHands],
        frontmost: String? = "com.apple.Safari", game: Bool = false,
        quiet: (Int, Int) = (23 * 60, 8 * 60), windDown: Bool = true
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.events = events
        var settings = DaySettings()
        settings.quietStartMinutes = quiet.0
        settings.quietEndMinutes = quiet.1
        settings.windDown = windDown
        return DaySnapshot(
            now: now, settings: settings, agenda: agenda, ownerPresent: present,
            frontmostAppName: "Safari", frontmostBundleID: frontmost, frontmostIsGame: game)
    }

    static func state() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = local(30, 8)
        state.eveningWrapUpAt = local(30, 21)
        state.nightReflectionAt = local(30, 21, 40)
        return state
    }

    static func banners(_ effects: [DayEffect]) -> [(title: String, body: String)] {
        effects.compactMap {
            if case .postBanner(let title, let body) = $0 { (title, body) } else { nil }
        }
    }

    @Test func asQuietHoursBeginItSaysWhenTomorrowStarts() throws {
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23)), state: Self.state())
        let banner = try #require(Self.banners(decision.effects).first)
        #expect(banner.title == "Time to wind down")
        #expect(banner.body == "Tomorrow starts with All Hands at 07:30 — 8 h 30 min from now.")
        #expect(decision.state.windDownAt == Self.local(30, 23))
        #expect(
            decision.effects.contains(.trace(.windDown, ["minutesUntil": .int(510)])))
    }

    @Test func itIsSaidOnceANight() {
        let first = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23)), state: Self.state())
        let second = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23, 1)), state: first.state)
        #expect(Self.banners(second.effects).isEmpty)
    }

    @Test func comingBackInTheFirstHourStillHearsIt() {
        let away = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23), present: false),
            state: Self.state())
        #expect(Self.banners(away.effects).isEmpty)
        let back = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23, 40)), state: away.state)
        #expect(Self.banners(back.effects).count == 1)
    }

    struct QuietRow: Sendable, CustomTestStringConvertible {
        let name: String
        let snapshot: DaySnapshot
        var testDescription: String { name }
    }

    static let quietRows: [QuietRow] = [
        QuietRow(name: "before quiet hours", snapshot: snapshot(at: local(30, 22, 50))),
        QuietRow(name: "past their first hour", snapshot: snapshot(at: local(31, 0, 5))),
        QuietRow(name: "a game in front", snapshot: snapshot(at: local(30, 23), game: true)),
        QuietRow(
            name: "a call in front", snapshot: snapshot(at: local(30, 23), frontmost: "us.zoom.xos")
        ),
        QuietRow(
            name: "quiet hours off",
            snapshot: snapshot(at: local(30, 23), quiet: (23 * 60, 23 * 60))),
        QuietRow(
            name: "the owner's setting off", snapshot: snapshot(at: local(30, 23), windDown: false)),
    ]

    @Test(arguments: quietRows)
    func noWindDown(_ row: QuietRow) {
        let decision = DayEngine.decide(.tick, snapshot: row.snapshot, state: Self.state())
        #expect(Self.banners(decision.effects).isEmpty)
    }

    @Test func quietHoursThatBeginAfterMidnightLookAtTheDayAhead() throws {
        // Quiet from 01:00: still the owner's 30 September until 04:00, so
        // "tomorrow" is the 1st, its first event at 09:00.
        let standup = AgendaEvent(
            id: "standup", title: "Standup", start: Self.local(31, 9), end: Self.local(31, 9, 15),
            calendarID: "c", calendarTitle: "Work")
        let decision = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(
                at: Self.local(31, 1), events: [standup], quiet: (60, 9 * 60)),
            state: Self.state())
        let banner = try #require(Self.banners(decision.effects).first)
        #expect(banner.body == "Tomorrow starts with Standup at 09:00 — 8 h from now.")
    }

    @Test func anOpenTomorrowSaysSo() throws {
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23), events: []),
            state: Self.state())
        let banner = try #require(Self.banners(decision.effects).first)
        #expect(banner.body == "Nothing is set for tomorrow yet. Rest well.")
    }
}
