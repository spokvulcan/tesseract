//
//  DayEngineMorningPlanTests.swift
//  tesseractTests
//
//  The Morning Plan never makes the owner wait on the model: a card built by
//  code goes up at once and Jarvis's version replaces it in place; when the
//  Mac is awake and on power before the owner sits down, the plan is ready
//  for them. Also: each nudge macOS delivered is recorded once, and cards
//  saved by an earlier build still load.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineMorningPlanTests {

    private typealias Day = DayEngineMomentTests

    private let measure = MomentMeasure(
        promptTokens: 7800, cachedTokens: 4300, outputTokens: 900, prefillSeconds: 17,
        generateSeconds: 30, latencySeconds: 49, waitSeconds: 0.2, hitCap: false,
        modelID: "test-model")

    private func panelCards(_ effects: [DayEffect]) -> [DayCard] {
        effects.compactMap { if case .presentCard(let card, .panel) = $0 { card } else { nil } }
    }

    @Test func theCodeCardGoesUpAtOnceAndJarvisRefinesItInPlace() throws {
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(29, 23)),
            snapshot: Day.snapshot(at: Day.local(30, 9, 3)), state: Day.state())
        let shown = try #require(panelCards(sitDown.effects).first)
        #expect(shown.kind == .morningPlan)
        #expect(shown.isRefining)
        #expect(!shown.isFallback)
        #expect(shown.line == "Here's your day: 1 event ahead — first Standup at 09:30.")
        let request = try #require(Day.moments(sitDown.effects).first)
        #expect(request.context.cardID == shown.id)

        let reply = """
            {"line": "A calm start before the standup.", "must_do": "R1",
             "plan": [{"id": "R2", "at": "09:05", "minutes": 20}]}
            """
        let refined = DayEngine.decide(
            .momentOutcome(request, .reply(reply, measure)),
            snapshot: Day.snapshot(at: Day.local(30, 9, 4)), state: sitDown.state)
        let card = try #require(refined.state.cards.first { $0.id == shown.id })
        #expect(!card.isRefining)
        #expect(card.line == "A calm start before the standup.")
        #expect(refined.state.cards.filter { !$0.dismissed }.count == 1)
        #expect(refined.state.mustDoID == "R1")
        #expect(refined.effects.contains(.presentCard(card, .panel)))
        #expect(
            refined.effects.contains {
                if case .trace(.momentFinished, let fields) = $0 {
                    fields["prefillSeconds"] == .double(17) && fields["waitSeconds"] == .double(0.2)
                } else {
                    false
                }
            })
    }

    @Test func whenJarvisCantThinkItThroughTheCodeCardStands() throws {
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(29, 23)),
            snapshot: Day.snapshot(at: Day.local(30, 9, 3)), state: Day.state())
        let request = try #require(Day.moments(sitDown.effects).first)
        let first = DayEngine.decide(
            .momentOutcome(request, .failed("model not loaded", nil)),
            snapshot: Day.snapshot(at: Day.local(30, 9, 4)), state: sitDown.state)
        let retry = try #require(Day.moments(first.effects).first)
        #expect(retry.context.cardID == request.context.cardID)
        let second = DayEngine.decide(
            .momentOutcome(retry, .failed("model not loaded", nil)),
            snapshot: Day.snapshot(at: Day.local(30, 9, 5)), state: first.state)
        #expect(second.state.cards.count == 1)
        let card = try #require(second.state.cards.first)
        #expect(card.isFallback)
        #expect(!card.isRefining)
        #expect(!card.dismissed)
    }

    @Test func aPlanIsPreparedBeforeTheOwnerSitsDown() throws {
        var state = Day.state()
        state.lastPresentAt = Day.local(29, 23)
        // 07:00, the Mac awake and on power, the owner still away.
        let early = DayEngine.decide(
            .tick, snapshot: Day.snapshot(at: Day.local(30, 7), present: false), state: state)
        let request = try #require(Day.moments(early.effects).first)
        #expect(request.trigger == .prepared)
        #expect(panelCards(early.effects).isEmpty)

        var ready = early.state
        ready.running = nil
        // The owner sits down at 08:30: the plan comes forward, nothing runs.
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(29, 23)),
            snapshot: Day.snapshot(at: Day.local(30, 8, 30)), state: ready)
        #expect(Day.moments(sitDown.effects).isEmpty)
        #expect(panelCards(sitDown.effects).first?.kind == .morningPlan)
    }

    @Test func aPreparedPlanMeetsAnEarlySitDownOnThePanel() throws {
        var state = Day.state()
        state.lastPresentAt = Day.local(29, 23)
        let early = DayEngine.decide(
            .tick, snapshot: Day.snapshot(at: Day.local(30, 5), present: false), state: state)
        var ready = early.state
        ready.running = nil
        // 07:40, before quiet hours end at 08:00: the owner is starting the day.
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(29, 23)),
            snapshot: Day.snapshot(at: Day.local(30, 7, 40)), state: ready)
        #expect(panelCards(sitDown.effects).first?.kind == .morningPlan)
    }

    @Test func aPlanMadeAtAnEarlySitDownGoesUpOnThePanel() {
        var state = Day.state()
        state.lastPresentAt = Day.local(29, 23)
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(29, 23)),
            snapshot: Day.snapshot(at: Day.local(30, 7, 40)), state: state)
        #expect(panelCards(sitDown.effects).first?.kind == .morningPlan)
        #expect(Day.moments(sitDown.effects).first?.trigger == .firstPresence)
    }

    @Test func quietHoursStillHoldACardThatIsNotTheDaysStart() {
        var state = Day.state()
        state.lastPresentAt = Day.local(30, 7)
        state.morningPlanAt = Day.local(30, 6)
        state.agents = [
            AgentSignal(
                id: "s1", kind: .waiting, agent: "Claude Code", project: "tesseract",
                directory: "/tmp/tesseract", message: "Needs approval", at: Day.local(30, 7, 10))
        ]
        // Back at 07:40 from a break: a Breakpoint, and it waits in Today.
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Day.local(30, 7)),
            snapshot: Day.snapshot(at: Day.local(30, 7, 40)), state: state)
        #expect(back.state.cards.last?.kind == .breakpoint)
        #expect(panelCards(back.effects).isEmpty)
    }

    // MARK: A plan cut short by a quit

    /// The plan was prepared at 05:00 and the app quit while Jarvis thought.
    static func cutShort() throws -> DayState {
        var state = Day.state()
        state.lastPresentAt = Day.local(29, 23)
        let early = DayEngine.decide(
            .tick, snapshot: Day.snapshot(at: Day.local(30, 5), present: false), state: state)
        #expect(early.state.running == .morningPlan)
        return early.state.relaunched()
    }

    @Test func aRelaunchRecordsTheMomentItCutShort() throws {
        let relaunched = try Self.cutShort()
        #expect(relaunched.running == nil)
        #expect(relaunched.interrupted == .morningPlan)
        #expect(relaunched.cards.allSatisfy { !$0.isRefining })
    }

    @Test func aPlanTheAppQuitInTheMiddleOfRunsAgainOnce() throws {
        let relaunched = try Self.cutShort()
        let back = DayEngine.decide(
            .companionEnabled, snapshot: Day.snapshot(at: Day.local(30, 5, 10), present: false),
            state: relaunched)
        let request = try #require(Day.moments(back.effects).first)
        #expect(request.kind == .morningPlan)
        #expect(request.trigger == .resumed)
        #expect(request.context.cardID == relaunched.cards.last?.id)
        #expect(back.state.cards.last?.isRefining == true)
        #expect(back.state.interrupted == nil)

        // Cut short again: the code card stands; no loop of retries.
        let again = DayEngine.decide(
            .companionEnabled, snapshot: Day.snapshot(at: Day.local(30, 5, 20), present: false),
            state: back.state.relaunched())
        #expect(Day.moments(again.effects).isEmpty)
    }

    @Test func aClosedOrLatePlanIsNotRunAgain() throws {
        var closed = try Self.cutShort()
        closed.cards[closed.cards.count - 1].dismissed = true
        let afterClose = DayEngine.decide(
            .companionEnabled, snapshot: Day.snapshot(at: Day.local(30, 7), present: false),
            state: closed)
        #expect(Day.moments(afterClose.effects).isEmpty)
        let evening = DayEngine.decide(
            .companionEnabled, snapshot: Day.snapshot(at: Day.local(30, 21, 30), present: false),
            state: try Self.cutShort())
        #expect(!Day.moments(evening.effects).contains { $0.kind == .morningPlan })
    }

    @Test func otherMomentsCutShortWaitForTheirOwnTriggers() {
        var state = Day.state()
        state.morningPlanAt = Day.local(30, 8)
        state.running = .nightReflection
        let back = DayEngine.decide(
            .companionEnabled, snapshot: Day.snapshot(at: Day.local(30, 9), present: false),
            state: state.relaunched())
        #expect(Day.moments(back.effects).isEmpty)
        #expect(back.state.interrupted == nil)
    }

    @Test func noPlanIsPreparedOnBattery() {
        var state = Day.state()
        state.lastPresentAt = Day.local(29, 23)
        let battery = PowerState(onACPower: false, batteryPercent: 90, thermal: .nominal)
        let early = DayEngine.decide(
            .tick, snapshot: Day.snapshot(at: Day.local(30, 7), present: false, power: battery),
            state: state)
        #expect(Day.moments(early.effects).isEmpty)
    }

    // MARK: Nudges macOS delivered

    @Test func eachDeliveredNudgeIsRecordedOnce() {
        let review = DeliveredNudge(
            id: "nudge.event.a", title: "Review MRs", at: Day.local(30, 5, 50))
        let daily = DeliveredNudge(
            id: "nudge.event.b", title: "Dev Daily", at: Day.local(30, 7, 50))
        func fired(_ effects: [DayEffect]) -> [String] {
            effects.compactMap {
                if case .trace(.nudgeFired, let fields) = $0, case .string(let id) = fields["id"] {
                    id
                } else {
                    nil
                }
            }
        }
        let first = DayEngine.decide(
            .nudgesDelivered([daily, review]), snapshot: Day.snapshot(at: Day.local(30, 8)),
            state: Day.state())
        #expect(fired(first.effects) == ["nudge.event.a", "nudge.event.b"])
        let again = DayEngine.decide(
            .nudgesDelivered([review, daily]), snapshot: Day.snapshot(at: Day.local(30, 8, 1)),
            state: first.state)
        #expect(fired(again.effects).isEmpty)
        // Cleared from Notification Center, then a new one.
        let course = DeliveredNudge(
            id: "nudge.event.c", title: "Course", at: Day.local(30, 12, 20))
        let later = DayEngine.decide(
            .nudgesDelivered([course]), snapshot: Day.snapshot(at: Day.local(30, 12, 21)),
            state: again.state)
        #expect(fired(later.effects) == ["nudge.event.c"])
        #expect(later.state.firedNudgeIDs == ["nudge.event.c"])
    }

    // MARK: Saved by an earlier build

    @Test func aCardSavedBeforeRefiningExistedStillLoads() throws {
        let json = """
            {"id": "morningPlan-2026-10-01-0", "kind": "morningPlan",
             "createdAt": "2026-10-01T09:06:05Z", "isFallback": false, "dismissed": true,
             "body": {"morningPlan": {"_0": {"line": "A day.", "placements": [],
             "suggestions": []}}}}
            """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let card = try decoder.decode(DayCard.self, from: Data(json.utf8))
        #expect(card.dismissed)
        #expect(!card.isRefining)
    }
}
