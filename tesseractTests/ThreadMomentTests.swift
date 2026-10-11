//
//  ThreadMomentTests.swift
//  tesseractTests
//
//  A moment's turn read back for Today's Chat: every request the moments
//  write names its kind (the builders and the reader can't drift), the line
//  comes from the card in the reply, a turn finds the card it made in the
//  day's state, the Day Opening reads back into what Jarvis started the day
//  with, and a card's summary says what waits on the owner first.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ThreadMomentTests {

    static let facts = MomentPromptsTests.facts

    private func kind(of request: String) -> ThreadMoment.Kind {
        ThreadMoment.header(request).kind
    }

    @Test func everyRequestNamesItsMoment() {
        #expect(kind(of: MomentPrompts.morningPlan(facts: Self.facts)) == .moment(.morningPlan))
        #expect(
            kind(of: MomentPrompts.eveningWrapUp(facts: Self.facts, leftovers: []))
                == .moment(.eveningWrapUp))
        #expect(
            kind(of: MomentPrompts.nightReflection(facts: Self.facts, profile: []))
                == .moment(.nightReflection))
        #expect(kind(of: BreakpointMoment.triageRequest([])) == .moment(.triage))

        let away = BreakpointInputs(
            awayFrom: DayFixture.local(30, 13, 20),
            now: DayFixture.local(30, 14, 5), toJudge: [], alreadyWaiting: [],
            raisedByRule: [], agents: [], dueWhileAway: [], nextEvent: nil, fitsBefore: [],
            whereYouWere: nil)
        let welcome = ThreadMoment.header(BreakpointMoment.request(away))
        #expect(welcome.kind == .moment(.breakpoint))
        #expect(welcome.detail == "away 45 min, 13:20–14:05")

        let opening = ThreadMoment.header(
            MomentPrompts.dayOpening(facts: Self.facts, profile: [], carryOver: nil))
        #expect(opening.kind == .dayOpening)
        #expect(opening.detail == "Wednesday 30 September")
    }

    @Test func aRetryAndAnUnknownMomentReadAsSuch() {
        #expect(
            kind(of: "That wasn't the card. Reply again with only the JSON object described above.")
                == .retry)
        #expect(kind(of: "[Weekly Review]\nLook back.") == .other("Weekly Review"))
    }

    @Test func theLineIsTheCardsLineOrTheNightsNote() {
        #expect(ThreadMoment.line(in: #"{"line": "A calm start."}"#) == "A calm start.")
        #expect(
            ThreadMoment.line(in: "Here it is:\n```json\n{\"line\": \" Fenced. \"}\n```")
                == "Fenced.")
        #expect(
            ThreadMoment.line(in: #"{"carry_over": "Finish the spec first.", "tomorrow": []}"#)
                == "Finish the spec first.")
        // A Triage that raised nothing says nothing.
        #expect(ThreadMoment.line(in: #"{"line": "", "raise": []}"#) == nil)
        #expect(ThreadMoment.line(in: "No card at all.") == nil)

        let request = UserMessage(content: "[Evening Wrap-up]", turnOrigin: .moment)
        let reply = AssistantMessage(
            content: #"{"line": "A good day."}"#, thinking: #"{"line": "not this"}"#)
        #expect(ThreadMoment(request: request, reply: reply).line == "A good day.")
        #expect(ThreadMoment(request: request, reply: nil).line == nil)
    }

    // MARK: Its card

    private static func at(_ hour: Int, _ minute: Int = 0, _ second: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(
                year: 2026, month: 10, day: 4, hour: hour, minute: minute, second: second))!
    }

    private static func turn(_ header: String, _ at: Date) -> ThreadMoment {
        ThreadMoment(
            request: UserMessage(content: header, timestamp: at, turnOrigin: .moment), reply: nil)
    }

    private static func card(_ id: String, _ kind: MomentKind, _ at: Date) -> DayCard {
        let body: DayCard.Body =
            switch kind {
            case .morningPlan:
                .morningPlan(
                    MorningPlanCard(line: id, mustDoID: nil, placements: [], suggestions: []))
            case .triage: .triage(TriageCard(line: id, raise: []))
            case .eveningWrapUp:
                .eveningWrapUp(
                    EveningWrapUpCard(line: id, done: [], leftovers: [], tomorrowFirst: nil))
            case .breakpoint:
                .breakpoint(
                    BreakpointCard(
                        awayFrom: at, awayUntil: at, line: id, needsYou: [], next: [],
                        whereYouWere: nil, canWait: []))
            case .nightReflection:
                .reflection(ReflectionCard(carryOver: id, tomorrow: [], proposals: []))
            }
        return DayCard(id: id, kind: kind, createdAt: at, isFallback: false, body: body)
    }

    @Test func aTurnFindsTheCardItMade() {
        let plan = Self.turn("[Morning Plan]", Self.at(8, 18, 1))
        let firstTriage = Self.turn("[Triage] New notifications", Self.at(15, 40))
        let secondTriage = Self.turn("[Triage] New notifications", Self.at(15, 52))
        let wrapUp = Self.turn("[Evening Wrap-up]", Self.at(21))
        let turns = [plan, firstTriage, secondTriage, wrapUp]
        let cards = [
            // The plan code puts up a beat before its request.
            Self.card("plan", .morningPlan, Self.at(8, 18)),
            // Only the second Triage raised something.
            Self.card("raised", .triage, Self.at(15, 52, 30)),
            // Two tries and the card code builds, well within the window.
            Self.card("wrap", .eveningWrapUp, Self.at(21, 31)),
        ]
        #expect(plan.card(in: cards, turns: turns)?.id == "plan")
        #expect(firstTriage.card(in: cards, turns: turns) == nil)
        #expect(secondTriage.card(in: cards, turns: turns)?.id == "raised")
        #expect(wrapUp.card(in: cards, turns: turns)?.id == "wrap")

        // A card of its kind long after it is another's, not its own.
        let late = [Self.card("later", .eveningWrapUp, Self.at(22, 30))]
        #expect(wrapUp.card(in: late, turns: turns) == nil)
        // The opening and a retry make no card.
        let opening = Self.turn("[Day Opening — Sunday 4 October]", Self.at(7, 42))
        #expect(opening.card(in: cards, turns: turns) == nil)
    }

    // MARK: Day Opening

    @Test func theDayOpeningReadsBackWhatJarvisStartedWith() {
        var facts = Self.facts
        facts.weekFocus = "Ship the Companion"
        let opening = MomentPrompts.dayOpening(
            facts: facts, profile: ["Works on Tesseract in the evenings."],
            carryOver: "Finish the spec first.\nThen rest.",
            draft: ["Send Anna the request early", "Documents ready for the ID check"])
        let digest = DayOpeningDigest(opening)
        #expect(digest.day == "Wednesday 30 September")
        #expect(digest.carryOver == "Finish the spec first.\nThen rest.")
        #expect(
            digest.draft == ["Send Anna the request early", "Documents ready for the ID check"])
        #expect(digest.profile == ["Works on Tesseract in the evenings."])
        #expect(digest.areas == ["Health", "Life", "Work"])
        #expect(digest.weekFocus == "Ship the Companion")
        #expect(digest.calendar == ["09:30–10:00 Standup", "13:00–13:45 1:1"])
        #expect(digest.due == ["Call the dentist", "Pay rent", "Renew passport"])
        #expect(digest.doneToday == ["Answer mail"])
    }

    @Test func aBareDayOpeningReadsBackEmpty() {
        let opening = MomentPrompts.dayOpening(
            facts: DayFacts(now: DayFixture.local(30, 7, 30)), profile: [],
            carryOver: nil)
        let digest = DayOpeningDigest(opening)
        #expect(digest.day == "Wednesday 30 September")
        #expect(digest.carryOver == nil)
        #expect(digest.calendar.isEmpty)
        #expect(digest.due.isEmpty)
        #expect(digest.profile.isEmpty)
    }

    // MARK: Summary

    @Test func aCardsSummarySaysWhatWaitsFirst() {
        let wrapUp = DayCard.Body.eveningWrapUp(
            EveningWrapUpCard(
                line: "", done: ["Answer mail", "Review PR #612"],
                leftovers: [
                    Leftover(reminderID: "adr", title: "Write the ADR", suggestion: .tomorrow)
                ], tomorrowFirst: "09:00 Work"))
        let live = MomentCardSummary(wrapUp, live: true)
        #expect(live.waiting == ["1 to settle"])
        #expect(live.parts == ["2 done", "tomorrow starts 09:00 Work"])
        let gone = MomentCardSummary(wrapUp, live: false)
        #expect(gone.waiting.isEmpty)
        #expect(gone.parts == ["1 left", "2 done", "tomorrow starts 09:00 Work"])

        let night = DayCard.Body.reflection(
            ReflectionCard(carryOver: "", tomorrow: ["Work at nine"], proposals: []))
        let deciding = MomentCardSummary(night, live: true, tasksWaiting: 2, factsWaiting: 1)
        #expect(deciding.waiting == ["2 tasks to add", "1 to remember"])
        #expect(deciding.parts == ["1 for tomorrow"])
        #expect(
            MomentCardSummary(night, live: true, remembered: 1).parts
                == ["1 for tomorrow", "1 remembered"])

        let quiet = DayCard.Body.triage(TriageCard(line: "", raise: []))
        #expect(MomentCardSummary(quiet, live: true).isEmpty)
    }
}
