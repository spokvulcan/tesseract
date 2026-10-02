//
//  DayEngineBreakpointTests.swift
//  tesseractTests
//
//  Breakpoints, Triage and coding agents as decision tables: no card when
//  nothing waits, a code-built card at once, the model's refinement in
//  place, batched triage at most every ten minutes, and the agent rule.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineBreakpointTests {

    static func local(_ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: 30, hour: hour, minute: minute))!
    }

    static func snapshot(
        at now: Date, present: Bool = true, frontmost: String? = "com.apple.Safari",
        terminalAt: Date? = nil, power: PowerState = .nominal, rules: [TriageRule] = [],
        game: Bool = false
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.events = [
            AgendaEvent(
                id: "review", title: "Design review", start: local(15), end: local(15, 45),
                calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)
        ]
        var settings = DaySettings()
        settings.rules = rules
        return DaySnapshot(
            now: now, settings: settings, agenda: agenda, ownerPresent: present,
            frontmostAppName: "Safari", frontmostBundleID: frontmost, frontmostIsGame: game,
            lastTerminalFrontAt: terminalAt,
            power: power)
    }

    static func state() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = local(7)
        return state
    }

    static func notification(_ id: String, app: String, title: String, body: String, at: Date)
        -> ObservedNotification
    {
        ObservedNotification(
            id: id, app: app, title: title, subtitle: "", body: body, arrivedAt: at)
    }

    static func moments(_ effects: [DayEffect]) -> [MomentRequest] {
        effects.compactMap { if case .runMoment(let request) = $0 { request } else { nil } }
    }

    static func panelCards(_ effects: [DayEffect]) -> [DayCard] {
        effects.compactMap { if case .presentCard(let card, .panel) = $0 { card } else { nil } }
    }

    static func todayCards(_ effects: [DayEffect]) -> [DayCard] {
        effects.compactMap { if case .presentCard(let card, .today) = $0 { card } else { nil } }
    }

    /// Two banners arrive while the owner is away at lunch.
    static func awayWithNotifications() -> DayState {
        var state = state()
        for (id, app, title, body) in [
            ("n-anna", "Slack", "Anna", "Can you look at the PR before 3?"),
            ("n-ci", "GitHub", "CI", "All checks passed on main"),
        ] {
            state =
                DayEngine.decide(
                    .notificationArrived(
                        notification(id, app: app, title: title, body: body, at: local(12, 30))),
                    snapshot: snapshot(at: local(12, 30), present: false), state: state
                ).state
        }
        return state
    }

    // MARK: Breakpoint

    @Test func aShortBreakIsNotABreakpoint() {
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(13, 55)),
            snapshot: Self.snapshot(at: Self.local(14)),
            state: Self.awayWithNotifications())
        #expect(decision.state.cards.isEmpty)
        #expect(Self.moments(decision.effects).isEmpty)
    }

    @Test func nothingWaitingMeansNoCardAtAll() {
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(12)),
            snapshot: Self.snapshot(at: Self.local(13)),
            state: Self.state())
        #expect(decision.state.cards.isEmpty)
        #expect(Self.panelCards(decision.effects).isEmpty)
    }

    @Test func theCardGoesUpAtOnceAndTheModelOnlyJudgesNotifications() throws {
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(12)),
            snapshot: Self.snapshot(at: Self.local(13)),
            state: Self.awayWithNotifications())
        // Nothing needs the owner yet: the card waits in Today, no panel pops
        // up over their work while the model judges the banners.
        #expect(Self.panelCards(decision.effects).isEmpty)
        let card = try #require(Self.todayCards(decision.effects).first)
        guard case .breakpoint(let breakpoint) = card.body else {
            Issue.record("expected a Breakpoint card")
            return
        }
        #expect(breakpoint.line == "Nothing needs you right now.")
        #expect(breakpoint.canWaitCount == 2)
        #expect(breakpoint.next.first?.title == "Design review")
        let request = try #require(Self.moments(decision.effects).first)
        #expect(request.kind == .breakpoint)
        #expect(request.context.cardID == card.id)
        #expect(request.context.notificationIDs == ["n-anna", "n-ci"])
        #expect(request.text.contains("- n1 · Slack: Anna — Can you look at the PR before 3?"))
        #expect(request.text.contains("- n2 · GitHub: CI — All checks passed on main"))
    }

    @Test func theModelsPickRefinesTheSameCard() throws {
        let first = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(12)),
            snapshot: Self.snapshot(at: Self.local(13)),
            state: Self.awayWithNotifications())
        let request = try #require(Self.moments(first.effects).first)
        let reply =
            #"{"line": "Welcome back — Anna needs a quick look at her PR.", "needs_you": ["n1"]}"#
        let measure = MomentMeasure(
            promptTokens: 900, outputTokens: 60, prefillSeconds: 0.2, generateSeconds: 2,
            latencySeconds: 3, hitCap: false, modelID: "m")
        let refined = DayEngine.decide(
            .momentOutcome(request, .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(13, 0)),
            state: first.state)
        let card = try #require(Self.panelCards(refined.effects).first)
        #expect(card.id == request.context.cardID)
        #expect(refined.state.cards.filter { !$0.dismissed }.count == 1)
        guard case .breakpoint(let breakpoint) = card.body else {
            Issue.record("expected a Breakpoint card")
            return
        }
        #expect(breakpoint.line == "Welcome back — Anna needs a quick look at her PR.")
        #expect(breakpoint.needsYou.map(\.id) == ["n-anna"])
        #expect(breakpoint.canWait.map(\.app) == ["GitHub"])
        // Neither banner is offered to a model again.
        #expect(refined.state.ledger.unresolved(now: Self.local(13, 1)).isEmpty)
    }

    @Test func aMeetingThatEndsIsABreakpoint() {
        var state = Self.awayWithNotifications()
        state.lastTickAt = Self.local(15, 44)
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(15, 45)), state: state)
        #expect(decision.state.cards.last?.kind == .breakpoint)
        #expect(Self.moments(decision.effects).first?.trigger == .meetingEnded)
    }

    @Test func actingOnAnItemRemovesItAndMarksItSeen() throws {
        let first = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(12)),
            snapshot: Self.snapshot(at: Self.local(13)),
            state: Self.awayWithNotifications())
        var state = first.state
        state.running = nil
        // Make Anna a "needs you" item, then open it.
        let cardID = try #require(state.cards.last?.id)
        guard case .breakpoint(var card) = state.cards[state.cards.count - 1].body else { return }
        card.needsYou = [BreakpointMoment.item(for: try #require(state.ledger.entry("n-anna")))]
        state.cards[state.cards.count - 1].body = .breakpoint(card)
        let opened = DayEngine.decide(
            .cardAction(.openItem(cardID: cardID, itemID: "n-anna")),
            snapshot: Self.snapshot(at: Self.local(13, 2)), state: state)
        #expect(opened.effects.contains(.openApp(name: "Slack")))
        #expect(opened.state.ledger.entry("n-anna")?.seenAt != nil)
        guard case .breakpoint(let after) = opened.state.cards.last?.body else { return }
        #expect(after.needsYou.isEmpty)
    }

    // MARK: Triage

    @Test func triageBatchesNewNotificationsAtMostEveryTenMinutes() throws {
        var state = Self.state()
        state =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification(
                        "n1", app: "Mail", title: "Bank", body: "Card blocked", at: Self.local(10))),
                snapshot: Self.snapshot(at: Self.local(10)), state: state
            ).state
        let first = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(10, 1)), state: state)
        let triage = try #require(Self.moments(first.effects).first)
        #expect(triage.kind == .triage)

        // A second banner a minute later waits for the next window.
        var later = first.state
        later.running = nil
        later =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification(
                        "n2", app: "Mail", title: "Promo", body: "Sale", at: Self.local(10, 2))),
                snapshot: Self.snapshot(at: Self.local(10, 2)), state: later
            ).state
        let second = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(10, 3)), state: later)
        #expect(Self.moments(second.effects).isEmpty)
    }

    @Test func triageRaisesOnlyWhatCantWait() throws {
        var state = Self.state()
        state =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification(
                        "bank", app: "Mail", title: "Bank", body: "Card blocked", at: Self.local(10)
                    )),
                snapshot: Self.snapshot(at: Self.local(10)), state: state
            ).state
        let run = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(10, 1)), state: state)
        let request = try #require(Self.moments(run.effects).first)
        let measure = MomentMeasure(
            promptTokens: 400, outputTokens: 30, prefillSeconds: 0.1, generateSeconds: 1,
            latencySeconds: 1.5, hitCap: false, modelID: "m")

        let raised = DayEngine.decide(
            .momentOutcome(
                request,
                .reply(#"{"line": "Your bank blocked your card.", "raise": ["n1"]}"#, measure)),
            snapshot: Self.snapshot(at: Self.local(10, 2)), state: run.state)
        #expect(Self.panelCards(raised.effects).first?.kind == .triage)
        #expect(raised.effects.contains(.speak("Your bank blocked your card.")))

        let quiet = DayEngine.decide(
            .momentOutcome(request, .reply(#"{"line": "", "raise": []}"#, measure)),
            snapshot: Self.snapshot(at: Self.local(10, 2)), state: run.state)
        #expect(Self.panelCards(quiet.effects).isEmpty)
        #expect(quiet.state.ledger.entry("bank")?.triagedAt != nil)
    }

    @Test func onlyPeopleReachTriage() throws {
        // The evening of the first: Game Mode turns on, an image is ready, and
        // someone writes on Slack.
        var state = Self.state()
        for (notification, at) in [
            (
                Self.notification(
                    "game-mode", app: "Game Mode", title: "Game Mode: On", body: "",
                    at: Self.local(19)
                ).classified(.noise), Self.local(19)
            ),
            (
                Self.notification(
                    "image", app: "ChatGPT", title: "Your image is ready", body: "",
                    at: Self.local(19, 1)
                ).classified(.app), Self.local(19, 1)
            ),
            (
                Self.notification(
                    "anna", app: "Slack", title: "Anna", body: "Can you look at the PR?",
                    at: Self.local(19, 2)
                ).classified(.person), Self.local(19, 2)
            ),
        ] {
            state =
                DayEngine.decide(
                    .notificationArrived(notification), snapshot: Self.snapshot(at: at),
                    state: state
                ).state
        }
        let run = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(19, 3)), state: state)
        let request = try #require(Self.moments(run.effects).first)
        #expect(request.context.notificationIDs == ["anna"])
        // The image waits for the next Breakpoint; Game Mode never shows.
        let unresolved = run.state.ledger.unresolved(now: Self.local(19, 3)).map(\.id)
        #expect(unresolved.contains("image"))
        #expect(!unresolved.contains("game-mode"))
    }

    @Test func anAppsNewsWaitsOnTheBreakpointWithoutAModel() throws {
        var state = Self.state()
        state =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification(
                        "image", app: "ChatGPT", title: "Your image is ready", body: "",
                        at: Self.local(12, 30)
                    ).classified(.app)),
                snapshot: Self.snapshot(at: Self.local(12, 30), present: false), state: state
            ).state
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(12)),
            snapshot: Self.snapshot(at: Self.local(13)), state: state)
        #expect(Self.moments(back.effects).isEmpty)
        let card = try #require(Self.todayCards(back.effects).first)
        guard case .breakpoint(let breakpoint) = card.body else {
            Issue.record("expected a Breakpoint card")
            return
        }
        #expect(breakpoint.canWait.map(\.app) == ["ChatGPT"])
        #expect(breakpoint.needsYou.isEmpty)
    }

    @Test func noTriageWhileAGameIsInFront() {
        var state = Self.state()
        state =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification(
                        "anna", app: "Slack", title: "Anna", body: "ping", at: Self.local(20)
                    ).classified(.person)),
                snapshot: Self.snapshot(at: Self.local(20)), state: state
            ).state
        let playing = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(
                at: Self.local(20, 1), frontmost: "com.valvesoftware.dota2", game: true),
            state: state)
        #expect(Self.moments(playing.effects).isEmpty)
        // Out of the game, it runs.
        let back = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(20, 2)), state: playing.state)
        #expect(Self.moments(back.effects).first?.kind == .triage)
    }

    @Test func aRaiseRuleSkipsTheModel() {
        let rules = [TriageRule(sender: "Anna", action: .raise, phrase: "always Anna")]
        let decision = DayEngine.decide(
            .notificationArrived(
                Self.notification(
                    "a", app: "Slack", title: "Anna", body: "ping", at: Self.local(11))),
            snapshot: Self.snapshot(at: Self.local(11), rules: rules), state: Self.state())
        #expect(Self.panelCards(decision.effects).first?.kind == .triage)
        #expect(Self.moments(decision.effects).isEmpty)
    }

    @Test func aHotMacDefersTriageUntilItCools() {
        var state = Self.state()
        state =
            DayEngine.decide(
                .notificationArrived(
                    Self.notification("n1", app: "Mail", title: "x", body: "y", at: Self.local(10))),
                snapshot: Self.snapshot(at: Self.local(10)), state: state
            ).state
        let hot = PowerState(onACPower: true, batteryPercent: nil, thermal: .serious)
        let deferred = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(10, 1), power: hot), state: state)
        #expect(Self.moments(deferred.effects).isEmpty)
        #expect(deferred.state.deferred.contains(.triage))
        #expect(
            deferred.effects.contains {
                if case .trace(.governorDeferred, _) = $0 { true } else { false }
            })

        let cooled = DayEngine.decide(
            .powerChanged, snapshot: Self.snapshot(at: Self.local(10, 5)), state: deferred.state)
        #expect(Self.moments(cooled.effects).map(\.kind) == [.triage])
    }

    // MARK: Coding agents

    static let waiting = AgentSignal(
        id: "s1", kind: .waiting, agent: "Claude Code", project: "tesseract",
        directory: "/p/tesseract",
        message: "Claude needs your permission to use Bash", at: local(11))

    @Test func aWaitingAgentIsSpokenAboutOnceWhenTheTerminalIsOutOfSight() {
        let first = DayEngine.decide(
            .agentSignal(Self.waiting),
            snapshot: Self.snapshot(at: Self.local(11), terminalAt: Self.local(10, 50)),
            state: Self.state())
        #expect(first.effects.contains(.speak("Claude Code in tesseract is waiting for you.")))
        #expect(first.effects.contains(.setWaiting(1)))

        let again = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 5), terminalAt: Self.local(10, 50)),
            state: first.state)
        #expect(!again.effects.contains { if case .speak = $0 { true } else { false } })
    }

    @Test(arguments: [
        ("in the terminal", true, "com.apple.Terminal", 11, 0),
        ("just left the terminal", true, "com.apple.Safari", 10, 59),
        ("away from the Mac", false, "com.apple.Safari", 10, 0),
    ])
    func noPingWhenItWouldTellThemWhatTheyCanSee(
        _ name: String, present: Bool, frontmost: String, terminalHour: Int, terminalMinute: Int
    ) {
        let decision = DayEngine.decide(
            .agentSignal(Self.waiting),
            snapshot: Self.snapshot(
                at: Self.local(11), present: present, frontmost: frontmost,
                terminalAt: Self.local(terminalHour, terminalMinute)),
            state: Self.state())
        #expect(!decision.effects.contains { if case .speak = $0 { true } else { false } })
        #expect(decision.state.agentsWaiting(now: Self.local(11)).count == 1)
    }

    @Test func bringingTheTerminalForwardClearsWaitingAgents() {
        let waiting = DayEngine.decide(
            .agentSignal(Self.waiting),
            snapshot: Self.snapshot(at: Self.local(11), present: false), state: Self.state())
        let cleared = DayEngine.decide(
            .appActivated(name: "Terminal", bundleID: "com.apple.Terminal"),
            snapshot: Self.snapshot(at: Self.local(11, 10)), state: waiting.state)
        #expect(cleared.state.agentsWaiting(now: Self.local(11, 10)).isEmpty)
        #expect(cleared.effects.contains(.setWaiting(0)))
    }

    @Test func anAwayOwnerFindsTheAgentOnTheBreakpoint() {
        let waiting = DayEngine.decide(
            .agentSignal(Self.waiting),
            snapshot: Self.snapshot(at: Self.local(11), present: false), state: Self.state())
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(10, 40)),
            snapshot: Self.snapshot(at: Self.local(11, 20)),
            state: waiting.state)
        guard case .breakpoint(let card) = back.state.cards.last?.body else {
            Issue.record("expected a Breakpoint card")
            return
        }
        #expect(card.needsYou.map(\.kind) == [.agent])
        // No notifications to judge: no model call.
        #expect(Self.moments(back.effects).isEmpty)
    }
}

struct CardItemActionTests {

    @Test func laterTurnsANotificationIntoAFollowUpReminder() throws {
        let start = DayEngine.decide(
            .presenceReturned(awayFrom: DayEngineBreakpointTests.local(12)),
            snapshot: DayEngineBreakpointTests.snapshot(at: DayEngineBreakpointTests.local(13)),
            state: DayEngineBreakpointTests.awayWithNotifications())
        var state = start.state
        state.running = nil
        let index = state.cards.count - 1
        guard case .breakpoint(var card) = state.cards[index].body else { return }
        card.needsYou = [BreakpointMoment.item(for: try #require(state.ledger.entry("n-anna")))]
        state.cards[index].body = .breakpoint(card)

        let later = DayEngine.decide(
            .cardAction(.itemLater(cardID: state.cards[index].id, itemID: "n-anna")),
            snapshot: DayEngineBreakpointTests.snapshot(at: DayEngineBreakpointTests.local(13, 5)),
            state: state)
        #expect(
            later.effects.contains(
                .mutateAgenda(
                    .followUp(title: "Follow up: Anna", at: DayEngineBreakpointTests.local(13, 35)))
            ))
        #expect(later.state.ledger.entry("n-anna")?.seenAt != nil)
    }

    @Test func aLeftoverCanMoveToAnotherDay() throws {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.cards = [
            DayCard(
                id: "wrap", kind: .eveningWrapUp, createdAt: DayEngineBreakpointTests.local(21),
                isFallback: false,
                body: .eveningWrapUp(
                    EveningWrapUpCard(
                        line: "Good day.", done: [],
                        leftovers: [
                            Leftover(reminderID: "R1", title: "Spec", suggestion: .tomorrow)
                        ],
                        tomorrowFirst: nil)))
        ]
        let friday = DayEngineBreakpointTests.local(0).addingTimeInterval(3 * 86_400)
        let decision = DayEngine.decide(
            .cardAction(.leftoverOn(cardID: "wrap", reminderID: "R1", day: friday)),
            snapshot: DayEngineBreakpointTests.snapshot(at: DayEngineBreakpointTests.local(21, 5)),
            state: state)
        #expect(decision.effects.contains(.mutateAgenda(.dueOn(reminderID: "R1", day: friday))))
        guard case .eveningWrapUp(let card) = decision.state.cards.first?.body else { return }
        #expect(card.leftovers.isEmpty)
    }
}
