//
//  DayWalkTests.swift
//  tesseractTests
//
//  One day walked through the Day Engine signal by signal, the way the loop
//  feeds it: the first sit-down and its plan, Looks Good, a Step Cue started,
//  an absence over its end and the late check-in, the must-do put off twice
//  and started small, kept going and done, a break after two hours at the
//  Mac, the menu bar's clock along the way, and the evening. Each rule has
//  its own tests; this walk is where they meet.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayWalkTests {

    static func at(_ hour: Int, _ minute: Int = 0, day: Int = 30) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let letter = AgendaReminder(
        id: "letter", title: "Write to the case worker", listID: "inbox",
        listTitle: "Reminders", due: at(0))
    static let deck = AgendaReminder(
        id: "deck", title: "Finish the deck", listID: "work", listTitle: "Work", due: at(0))
    static let standup = AgendaEvent(
        id: "standup", title: "Standup", start: at(11), end: at(11, 15), calendarID: "c",
        calendarTitle: "Work", hasOtherAttendees: true)
    static let measure = MomentMeasure(
        promptTokens: 6000, cachedTokens: 5000, outputTokens: 300, prefillSeconds: 2,
        generateSeconds: 9, latencySeconds: 11, hitCap: false, modelID: "m")

    /// The day as the loop sees it. The Agenda performs what the engine asks
    /// (a task done moves to done), and the panel stays up until answered.
    struct World {
        var state = DayState(day: DayKey(rawValue: "2026-09-29"))
        var open = [DayWalkTests.letter, DayWalkTests.deck]
        var done: [AgendaReminder] = []
        var present = true
        var panelUp = false

        func snapshot(_ now: Date) -> DaySnapshot {
            var agenda = AgendaSnapshot.empty
            agenda.access = .full
            agenda.open = open
            agenda.doneToday = done
            agenda.events = [DayWalkTests.standup]
            return DaySnapshot(
                now: now, settings: DaySettings(), agenda: agenda,
                areas: [Area(id: "work", name: "Work")], inboxListID: "inbox",
                ownerPresent: present, frontmostAppName: "Safari",
                frontmostBundleID: "com.apple.Safari", panelUp: panelUp)
        }

        @discardableResult
        mutating func send(_ signal: DaySignal, at now: Date) -> [DayEffect] {
            let decision = DayEngine.decide(signal, snapshot: snapshot(now), state: state)
            state = decision.state
            for effect in decision.effects {
                switch effect {
                case .mutateAgenda(.complete(let id)):
                    guard let index = open.firstIndex(where: { $0.id == id }) else { continue }
                    var reminder = open.remove(at: index)
                    reminder.isCompleted = true
                    reminder.completedAt = now
                    done.append(reminder)
                case .presentStep, .presentBreak, .presentCard(_, .panel):
                    panelUp = true
                case .retractBreak:
                    panelUp = false
                default:
                    continue
                }
            }
            return decision.effects
        }

        /// The owner answers what is on the panel, and it closes.
        mutating func answer(_ action: CardAction, at now: Date) {
            send(.cardAction(action), at: now)
            panelUp = false
        }

        func clock(at now: Date) -> MenuBarClock? {
            DayEngine.clock(snapshot: snapshot(now), state: state)
        }
    }

    static func cue(_ effects: [DayEffect]) -> StepCue? {
        effects.lazy.compactMap { if case .presentStep(let cue) = $0 { cue } else { nil } }.first
    }

    static func rest(_ effects: [DayEffect]) -> BreakCue? {
        effects.lazy.compactMap { if case .presentBreak(let cue) = $0 { cue } else { nil } }.first
    }

    static func moment(_ effects: [DayEffect]) -> MomentRequest? {
        effects.lazy.compactMap { if case .runMoment(let request) = $0 { request } else { nil } }
            .first
    }

    @Test func aDayWithJarvis() throws {
        var world = World()
        world.state.syncedNudgeIDs = []
        world.state.lastPresentAt = Self.at(23, 40, day: 29)

        // 07:50, the first sit-down after the night: the plan's code card on
        // the panel at once, and Jarvis thinking it through.
        let sitDown = world.send(
            .presenceReturned(awayFrom: Self.at(23, 40, day: 29)), at: Self.at(7, 50))
        let request = try #require(Self.moment(sitDown))
        #expect(request.kind == .morningPlan)
        #expect(
            sitDown.contains {
                if case .presentCard(let card, .panel) = $0 {
                    card.kind == .morningPlan
                } else {
                    false
                }
            })
        let reply = """
            {"line": "A focused morning.", "must_do": "deck",
             "plan": [{"id": "letter", "at": "09:00", "minutes": 20},
                      {"id": "deck", "at": "10:00", "minutes": 45}]}
            """
        world.send(.momentOutcome(request, .reply(reply, Self.measure)), at: Self.at(7, 51))
        #expect(world.state.plan.map(\.reminderID) == ["letter", "deck"])
        #expect(world.state.mustDoID == "deck")
        // Looks Good: the plan leaves the panel and stays in Today.
        let plan = try #require(world.state.cards.last)
        world.answer(.keep(cardID: plan.id), at: Self.at(7, 52))
        #expect(world.state.cards.last?.kept == true)

        // 09:00: the letter's slot starts; Start, and the menu bar counts it.
        let nine = try #require(Self.cue(world.send(.tick, at: Self.at(9))))
        #expect(nine.reminderID == "letter")
        #expect(!nine.late)
        world.answer(.step(reminderID: "letter", .start), at: Self.at(9, 1))
        #expect(
            world.clock(at: Self.at(9, 5))
                == MenuBarClock(
                    kind: .focus, title: "Write to the case worker", until: Self.at(9, 21)))

        // Away from 09:08 to 09:40, over the letter's end: nothing while
        // away, then the late check-in.
        world.send(.presenceLeft, at: Self.at(9, 8))
        world.present = false
        #expect(Self.cue(world.send(.tick, at: Self.at(9, 21))) == nil)
        world.present = true
        world.send(.presenceReturned(awayFrom: Self.at(9, 8)), at: Self.at(9, 40))
        let back = try #require(Self.cue(world.send(.tick, at: Self.at(9, 40))))
        #expect(back.reminderID == "letter")
        #expect(back.phase == .end)
        #expect(back.late)
        world.answer(.step(reminderID: "letter", .done), at: Self.at(9, 41))
        #expect(world.done.map(\.id) == ["letter"])

        // 10:00: the must-do, put off twice, then offered as five minutes.
        let ten = try #require(Self.cue(world.send(.tick, at: Self.at(10))))
        #expect(ten.reminderID == "deck")
        #expect(ten.isMustDo)
        world.answer(.step(reminderID: "deck", .later), at: Self.at(10, 1))
        let again = try #require(Self.cue(world.send(.tick, at: Self.at(10, 16))))
        #expect(again.putOff == 1)
        #expect(!again.offersSmallStart)
        world.answer(.step(reminderID: "deck", .later), at: Self.at(10, 16))
        let third = try #require(Self.cue(world.send(.tick, at: Self.at(10, 31))))
        #expect(third.offersSmallStart)
        world.answer(.step(reminderID: "deck", .startSmall), at: Self.at(10, 32))

        // Five minutes in, it asks to keep going: a quarter of an hour more.
        let five = try #require(Self.cue(world.send(.tick, at: Self.at(10, 37))))
        #expect(five.phase == .end)
        #expect(five.small)
        world.answer(.step(reminderID: "deck", .extend), at: Self.at(10, 37))
        // The standup comes after the step's new end: the clock counts the step.
        #expect(world.clock(at: Self.at(10, 45))?.kind == .focus)
        let up = try #require(Self.cue(world.send(.tick, at: Self.at(10, 52))))
        #expect(up.phase == .end)
        #expect(!up.small)
        world.answer(.step(reminderID: "deck", .done), at: Self.at(10, 52))

        // The must-do is seen done; the standup is next on the clock.
        world.send(.tick, at: Self.at(10, 53))
        #expect(world.state.mustDoDoneAt != nil)
        #expect(
            world.clock(at: Self.at(10, 53))
                == MenuBarClock(kind: .event, title: "Standup", until: Self.at(11)))
        // Nothing more is cued: both steps are done.
        #expect(Self.cue(world.send(.tick, at: Self.at(11, 30))) == nil)

        // 11:40: two hours at the Mac since the owner came back at 09:40 — a
        // break. Taking 5; a walk to the kitchen counts as one, and the two
        // hours start again from the return.
        let rest = try #require(Self.rest(world.send(.tick, at: Self.at(11, 40))))
        #expect(rest.since == Self.at(9, 40))
        #expect(rest.minutes == 120)
        world.answer(.breakCue(.taking), at: Self.at(11, 41))
        world.send(.presenceLeft, at: Self.at(11, 45))
        world.send(.presenceReturned(awayFrom: Self.at(11, 42)), at: Self.at(11, 49))
        #expect(world.state.sittingSince == Self.at(11, 49))
        #expect(Self.rest(world.send(.tick, at: Self.at(13, 48))) == nil)

        // 21:00: the Evening Wrap-up, with nothing left over.
        let evening = try #require(Self.moment(world.send(.tick, at: Self.at(21))))
        #expect(evening.kind == .eveningWrapUp)
        #expect(evening.text.contains("Nothing left over from today."))
        #expect(world.state.mustDoDays.isEmpty)
        #expect(
            world.state.rolledOver(to: DayKey(rawValue: "2026-10-01")).mustDoDays
                == ["2026-09-30": true])
    }
}
