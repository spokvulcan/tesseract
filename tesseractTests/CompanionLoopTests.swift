//
//  CompanionLoopTests.swift
//  tesseractTests
//
//  The wake fabric (ADR-0040, lean palette #369): the wakes table's state
//  machine queries, the wake palette tools (book with its visible promise
//  budget, revise, cancel), the wake-time grammar, the loop's per-day state,
//  and the situation briefing's rendered shape. Each test opens its own
//  scratch store so the scheme's parallel twin runners can't collide.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Wake store

@Suite struct CompanionWakeStoreTests {

    @Test func bookedWakeBecomesDueAtItsTime() async throws {
        let store = try scratchStore()
        let due = Date().addingTimeInterval(600)
        let wake = CompanionWake(content: "ask about the dentist", due: due)
        try await store.upsertWake(wake)

        #expect(try await store.dueWakes(asOf: Date()).isEmpty)
        let fired = try await store.dueWakes(asOf: due.addingTimeInterval(1))
        #expect(fired.map(\.id) == [wake.id])
        #expect(try await store.upcomingWakes(after: Date()).map(\.id) == [wake.id])
    }

    @Test func onlyBookedWakesFire() async throws {
        let store = try scratchStore()
        let past = Date().addingTimeInterval(-60)
        for state in [CompanionWakeState.fired, .delivered, .engaged, .dropped] {
            try await store.upsertWake(
                CompanionWake(content: "x", due: past, state: state))
        }
        #expect(try await store.dueWakes(asOf: Date()).isEmpty)
        // Fired-but-unconsumed is exactly the crash-recovery set.
        #expect(try await store.unconsumedFiredWakes().count == 1)
    }

    @Test func consumedFiredWakeLeavesTheRecoverySet() async throws {
        let store = try scratchStore()
        var wake = CompanionWake(content: "x", due: Date(), state: .fired)
        wake.firedAt = Date()
        try await store.upsertWake(wake)
        #expect(try await store.unconsumedFiredWakes().count == 1)

        wake.state = .delivered
        wake.consumedAt = Date()
        try await store.upsertWake(wake)
        #expect(try await store.unconsumedFiredWakes().isEmpty)

        let loaded = try #require(try await store.wake(id: wake.id))
        #expect(loaded.state == .delivered)
        #expect(loaded.consumedAt != nil)
    }

    @Test func promiseBudgetIsKeyedToTheLandingDay() async throws {
        let store = try scratchStore()
        // Fixed wall times so a run near midnight can't smear across day keys.
        let calendar = Calendar.current
        let noon = try #require(
            calendar.date(bySettingHour: 12, minute: 0, second: 0, of: Date()))
        let tomorrowNoon = noon.addingTimeInterval(24 * 3600)
        try await store.upsertWake(
            CompanionWake(content: "a", due: noon, wakeClass: .promise))
        try await store.upsertWake(
            CompanionWake(content: "b", due: noon.addingTimeInterval(300), wakeClass: .rhythm))
        try await store.upsertWake(
            CompanionWake(content: "c", due: tomorrowNoon, wakeClass: .promise))
        // A delivered promise still counts — the day carried it; dropped never.
        try await store.upsertWake(
            CompanionWake(
                content: "d", due: noon.addingTimeInterval(3600), wakeClass: .promise,
                state: .delivered))
        try await store.upsertWake(
            CompanionWake(
                content: "e", due: noon.addingTimeInterval(7200), wakeClass: .promise,
                state: .dropped))
        #expect(try await store.promisesBooked(onDay: TrackingDay.key(for: noon)) == 2)
        #expect(try await store.promisesBooked(onDay: TrackingDay.key(for: tomorrowNoon)) == 1)
    }

    @Test func loopDayStateRoundTrips() async throws {
        let store = try scratchStore()
        let key = "2026-07-16"
        var state = try await store.loopDayState(key)
        #expect(state.dayStartedAt == nil)

        state.dayStartedAt = Date()
        state.lastAmbientAt = Date()
        try await store.setLoopDayState(key, state)

        let loaded = try await store.loopDayState(key)
        #expect(loaded.dayStartedAt != nil)
        #expect(loaded.lastAmbientAt != nil)
        #expect(try await store.loopDayState("2026-07-17").dayStartedAt == nil)
    }
}

// MARK: - Resurfacing (#309)

@Suite struct CompanionResurfacingTests {

    /// The full ladder: ignored → resurfaced at the next beat → dead
    /// (delivered-unheard, recorded) at the beat after that.
    @Test func ignoredPromiseResurfacesOnceThenDies() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        var promise = CompanionWake(
            content: "ask about the dentist", due: Date().addingTimeInterval(-3600),
            wakeClass: .promise, state: .delivered)
        promise.consumedAt = Date().addingTimeInterval(-3600)
        try await store.upsertWake(promise)

        // First beat: the ignored promise joins the agenda as resurfaced.
        let agenda = await CompanionResurfacing.pass(store: store, recorder: recorder)
        #expect(agenda.map(\.id) == [promise.id])
        var loaded = try #require(try await store.wake(id: promise.id))
        #expect(loaded.state == .resurfaced)

        // Second beat, still no reaction: dead, no third attempt, recorded.
        let secondAgenda = await CompanionResurfacing.pass(store: store, recorder: recorder)
        #expect(secondAgenda.isEmpty)
        loaded = try #require(try await store.wake(id: promise.id))
        #expect(loaded.state == .deliveredUnheard)
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(events.contains { $0.event == "wake.resurfaced" })
        #expect(events.contains { $0.event == "wake.delivered-unheard" })
    }

    @Test func heardPromisesNeverResurface() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        var dismissed = CompanionWake(
            content: "he waved this off", due: Date().addingTimeInterval(-3600),
            wakeClass: .promise, state: .delivered)
        dismissed.consumedAt = Date().addingTimeInterval(-3600)
        try await store.upsertWake(dismissed)
        // Any reaction stamps heard — an explicit wave-off is not "ignored".
        try await store.stampWakeHeard(id: dismissed.id, at: Date())

        let agenda = await CompanionResurfacing.pass(store: store, recorder: recorder)
        #expect(agenda.isEmpty)
        let loaded = try #require(try await store.wake(id: dismissed.id))
        #expect(loaded.state == .delivered)
    }

    @Test func resurfacedPromiseHeardViaBeatEngagementIsSpared() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        var promise = CompanionWake(
            content: "still owed", due: Date().addingTimeInterval(-3600),
            wakeClass: .promise, state: .delivered)
        promise.consumedAt = Date().addingTimeInterval(-3600)
        try await store.upsertWake(promise)

        _ = await CompanionResurfacing.pass(store: store, recorder: recorder)
        // The owner engages the beat that carried the resurfacing.
        try await store.stampResurfacedHeard(at: Date())

        _ = await CompanionResurfacing.pass(store: store, recorder: recorder)
        let loaded = try #require(try await store.wake(id: promise.id))
        #expect(loaded.state == .delivered)
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(!events.contains { $0.event == "wake.delivered-unheard" })
    }

    @Test func rhythmAndFollowupWakesAreNotPromiseLadderMaterial() async throws {
        let store = try scratchStore()
        var rhythm = CompanionWake(
            content: "evening journal", due: Date().addingTimeInterval(-3600),
            wakeClass: .rhythm, state: .delivered)
        rhythm.consumedAt = Date().addingTimeInterval(-3600)
        try await store.upsertWake(rhythm)
        #expect(try await store.unheardDeliveredPromises().isEmpty)
    }

    @Test func heardStampIsFirstReactionWins() async throws {
        let store = try scratchStore()
        var promise = CompanionWake(
            content: "x", due: Date(), wakeClass: .promise, state: .delivered)
        promise.consumedAt = Date()
        try await store.upsertWake(promise)

        let first = Date().addingTimeInterval(-100)
        try await store.stampWakeHeard(id: promise.id, at: first)
        try await store.stampWakeHeard(id: promise.id, at: Date())

        let loaded = try #require(try await store.wake(id: promise.id))
        let heardAt = try #require(loaded.heardAt)
        #expect(abs(heardAt.timeIntervalSince(first)) < 1)
    }
}

// MARK: - book_wake

@Suite struct CompanionBookWakeToolTests {

    private func run(
        _ tool: AgentToolDefinition, _ args: [String: JSONValue]
    ) async throws -> String {
        let result = try await tool.execute("test-call", args, nil, nil)
        let texts = result.content.compactMap { block -> String? in
            if case .text(let text) = block { return text }
            return nil
        }
        guard !texts.isEmpty else {
            Issue.record("expected a text result")
            return ""
        }
        return texts.joined(separator: "\n")
    }

    @MainActor
    @Test func booksAWakeWithCorrelation() async throws {
        let store = try scratchStore()
        let context = CompanionTurnContext()
        let conversationID = UUID()
        context.begin(
            turnID: UUID(), wakeIDs: [], conversationID: conversationID, origin: .wake)
        let tool = createBookWakeTool(
            store: store, recorder: scratchRecorder(), context: context)

        let reply = try await run(
            tool,
            [
                "content": .string("check whether he started the workout"),
                "in_minutes": .int(40),
                "class": .string("followup"),
            ])
        #expect(reply.contains("Booked [followup]"))

        let upcoming = try await store.upcomingWakes(after: Date())
        #expect(upcoming.count == 1)
        #expect(upcoming[0].wakeClass == .followup)
        #expect(upcoming[0].conversationID == conversationID)
        #expect(!upcoming[0].summonsGrant)
    }

    @Test func promiseBudgetRefusesVisiblyAtTheCap() async throws {
        let store = try scratchStore()
        let tool = createBookWakeTool(
            store: store, recorder: scratchRecorder(), context: CompanionTurnContext())

        // All three land tomorrow around noon — one day key, no midnight smear.
        let calendar = Calendar.current
        let noon = try #require(
            calendar.date(bySettingHour: 12, minute: 0, second: 0, of: Date()))
        let tomorrowNoon = noon.addingTimeInterval(24 * 3600)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        func stamp(_ minutes: Int) -> String {
            formatter.string(from: tomorrowNoon.addingTimeInterval(Double(minutes) * 60))
        }

        for minutes in [0, 30] {
            let reply = try await run(
                tool, ["content": .string("promise \(minutes)"), "at": .string(stamp(minutes))])
            #expect(reply.contains("Booked [promise]"))
        }
        let refused = try await run(
            tool, ["content": .string("one too many"), "at": .string(stamp(60))])
        #expect(refused.contains("Promise budget spent"))
        #expect(try await store.promisesBooked(onDay: TrackingDay.key(for: tomorrowNoon)) == 2)

        // The budget is promises-only: rhythm beats always book.
        let rhythm = try await run(
            tool,
            [
                "content": .string("evening journal"), "at": .string(stamp(90)),
                "class": .string("rhythm"),
            ])
        #expect(rhythm.contains("Booked [rhythm]"))
    }

    @Test func reviseMovesInsteadOfDuplicating() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        _ = try await run(
            createBookWakeTool(
                store: store, recorder: recorder, context: CompanionTurnContext()),
            ["content": .string("midday pulse"), "in_minutes": .int(30)])
        let booked = try await store.upcomingWakes(after: Date())
        let id = try #require(booked.first?.id)

        let revise = createReviseWakeTool(
            store: store, recorder: recorder, context: CompanionTurnContext())
        let moved = try await run(
            revise,
            [
                "id": .string(id.uuidString),
                "content": .string("midday pulse — he asked for an hour"),
                "in_minutes": .int(90),
            ])
        #expect(moved.contains("Revised"))

        let after = try await store.upcomingWakes(after: Date())
        #expect(after.count == 1)
        #expect(after[0].id == id)
        #expect(after[0].content.contains("he asked for an hour"))
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(events.contains { $0.event == "wake.revised" })
    }

    @Test func reviseNeedsAChangeAndAnOpenWake() async throws {
        let store = try scratchStore()
        let tool = createReviseWakeTool(
            store: store, recorder: scratchRecorder(), context: CompanionTurnContext())

        // Unknown id refuses; a consumed wake refuses; no-change refuses.
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute(
                "t", ["id": .string(UUID().uuidString), "in_minutes": .int(10)], nil, nil)
        }
        var consumed = CompanionWake(content: "x", due: Date(), state: .delivered)
        consumed.consumedAt = Date()
        try await store.upsertWake(consumed)
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute(
                "t", ["id": .string(consumed.id.uuidString), "in_minutes": .int(10)], nil, nil)
        }
        let open = CompanionWake(content: "y", due: Date().addingTimeInterval(600))
        try await store.upsertWake(open)
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("t", ["id": .string(open.id.uuidString)], nil, nil)
        }
    }

    @Test func cancelIsADeliberateRecordedExit() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        let wake = CompanionWake(
            content: "ask about the dentist", due: Date().addingTimeInterval(600),
            wakeClass: .promise)
        try await store.upsertWake(wake)

        let tool = createCancelWakeTool(
            store: store, recorder: recorder, context: CompanionTurnContext())
        // No why → refused: the record must say.
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("t", ["id": .string(wake.id.uuidString)], nil, nil)
        }
        let out = try await run(
            tool,
            [
                "id": .string(wake.id.uuidString),
                "why": .string("he brought it up himself this morning"),
            ])
        #expect(out.contains("Cancelled [promise]"))

        let loaded = try #require(try await store.wake(id: wake.id))
        #expect(loaded.state == .cancelled)
        // Cancelled never fires and never resurfaces.
        #expect(try await store.dueWakes(asOf: Date().addingTimeInterval(3600)).isEmpty)
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(events.contains { $0.event == "wake.cancelled" })
    }

    @Test func cancelledPromiseFreesItsDayBudget() async throws {
        let store = try scratchStore()
        let calendar = Calendar.current
        let noon = try #require(
            calendar.date(bySettingHour: 12, minute: 0, second: 0, of: Date()))
        let tomorrowNoon = noon.addingTimeInterval(24 * 3600)
        let first = CompanionWake(content: "a", due: tomorrowNoon, wakeClass: .promise)
        try await store.upsertWake(first)
        try await store.upsertWake(
            CompanionWake(
                content: "b", due: tomorrowNoon.addingTimeInterval(300), wakeClass: .promise))
        let day = TrackingDay.key(for: tomorrowNoon)
        #expect(try await store.promisesBooked(onDay: day) == 2)

        _ = try await run(
            createCancelWakeTool(
                store: store, recorder: scratchRecorder(), context: CompanionTurnContext()),
            ["id": .string(first.id.uuidString), "why": .string("moot")])
        #expect(try await store.promisesBooked(onDay: day) == 1)
    }

    @Test func summonsAsStringFailsLoudly() async throws {
        let store = try scratchStore()
        let tool = createBookWakeTool(
            store: store, recorder: scratchRecorder(), context: CompanionTurnContext())
        // The #354 class: a stringly boolean must refuse, never coerce.
        await #expect(throws: ToolArgTypeError.self) {
            _ = try await tool.execute(
                "t",
                [
                    "content": .string("x"), "in_minutes": .int(10),
                    "summons": .string("False"),
                ], nil, nil)
        }
        #expect(try await store.upcomingWakes(after: Date()).isEmpty)
    }

    @Test func refusesThePastAndGarbageTimes() async throws {
        let store = try scratchStore()
        let tool = createBookWakeTool(
            store: store, recorder: scratchRecorder(), context: CompanionTurnContext())
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute(
                "t", ["content": .string("x"), "at": .string("yesterday-ish")], nil, nil)
        }
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute(
                "t", ["content": .string("x"), "at": .string("2020-01-01 09:00")], nil, nil)
        }
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("t", ["content": .string("x")], nil, nil)
        }
    }
}

// MARK: - Wake-time grammar

@Suite struct CompanionWakeTimeTests {

    @Test func parsesFullLocalStamp() throws {
        let date = try #require(CompanionWakeTime.parse("2026-07-17 09:10"))
        let parts = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute], from: date)
        #expect(parts.year == 2026)
        #expect(parts.month == 7)
        #expect(parts.day == 17)
        #expect(parts.hour == 9)
        #expect(parts.minute == 10)
    }

    @Test func timeOnlyMeansNextOccurrence() throws {
        let now = Date()
        let date = try #require(CompanionWakeTime.parse("21:30", now: now))
        #expect(date > now)
        #expect(date.timeIntervalSince(now) <= 24 * 3600)
        let parts = Calendar.current.dateComponents([.hour, .minute], from: date)
        #expect(parts.hour == 21)
        #expect(parts.minute == 30)
    }

    @Test func garbageIsNil() {
        #expect(CompanionWakeTime.parse("soonish") == nil)
        #expect(CompanionWakeTime.parse("") == nil)
    }
}

// MARK: - Standing instructions (ADR-0040 §12)

@Suite struct CompanionInstructionsTests {

    @Test func versionsAppendAndCurrentIsHighest() async throws {
        let store = try scratchStore()
        #expect(try await store.currentInstructions() == nil)

        let v1 = try await store.appendInstructions(text: "seed text", author: "seed", note: nil)
        let v2 = try await store.appendInstructions(
            text: "revised text", author: "entity", note: "learned the rhythm")
        #expect(v1 == 1)
        #expect(v2 == 2)

        let current = try #require(try await store.currentInstructions())
        #expect(current.version == 2)
        #expect(current.text == "revised text")
        #expect(current.author == "entity")
        #expect(current.note == "learned the rhythm")

        let history = try await store.instructionsHistory()
        #expect(history.map(\.version) == [2, 1])
    }

    @Test func seedInstallsExactlyOnce() async throws {
        let store = try scratchStore()
        #expect(try await store.seedInstructionsIfNeeded("the seed") == true)
        #expect(try await store.seedInstructionsIfNeeded("the seed") == false)
        let current = try #require(try await store.currentInstructions())
        #expect(current.version == 1)
        #expect(current.author == "seed")
    }

    @MainActor
    @Test func reviseToolAppendsEntityVersionWithWhy() async throws {
        let store = try scratchStore()
        try await store.seedInstructionsIfNeeded("the seed")
        let context = CompanionTurnContext()
        context.begin(turnID: UUID(), wakeIDs: [], conversationID: UUID(), origin: .ambient)
        let recorder = scratchRecorder()
        let tool = createReviseInstructionsTool(
            store: store, recorder: recorder, context: context)

        let result = try await tool.execute(
            "t",
            [
                "text": .string("the seed, plus: he prefers the pulse at 14:00"),
                "why": .string("he moved the pulse twice running"),
            ], nil, nil)
        let text = result.content.compactMap { block -> String? in
            if case .text(let value) = block { return value }
            return nil
        }.joined()
        #expect(text.contains("now v2"))

        let current = try #require(try await store.currentInstructions())
        #expect(current.author == "entity")
        #expect(current.note == "he moved the pulse twice running")

        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(events.contains { $0.event == "instructions.revised" })
    }

    @MainActor
    @Test func reviseToolGuardsEmptyAndOversize() async throws {
        let store = try scratchStore()
        let tool = createReviseInstructionsTool(
            store: store, recorder: scratchRecorder(), context: CompanionTurnContext())

        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute(
                "t", ["text": .string("  "), "why": .string("x")], nil, nil)
        }
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("t", ["text": .string("fine")], nil, nil)
        }

        let huge = String(repeating: "a", count: CompanionInstructions.maxLength + 1)
        let result = try await tool.execute(
            "t", ["text": .string(huge), "why": .string("growth")], nil, nil)
        let text = result.content.compactMap { block -> String? in
            if case .text(let value) = block { return value }
            return nil
        }.joined()
        #expect(text.contains("Too long"))
        #expect(try await store.currentInstructions() == nil)
    }

    @Test func wrapCarriesVersionAndAuthor() {
        let wrapped = CompanionInstructions.wrap(
            CompanionInstructionsVersion(
                version: 7, text: "be brief", author: "entity", note: nil, createdAt: Date()))
        #expect(wrapped.contains("<companion-instructions version=\"7\" author=\"entity\">"))
        #expect(wrapped.contains("be brief"))
        #expect(wrapped.hasSuffix("</companion-instructions>"))
    }
}

// MARK: - Briefing

@MainActor
@Suite struct CompanionBriefingRenderTests {

    @Test func rendersPresenceContractAndWakes() {
        let now = Date()
        let due = CompanionWake(
            content: "morning planning", due: now.addingTimeInterval(-8 * 60),
            wakeClass: .rhythm, state: .fired)
        let upcoming = CompanionWake(
            content: "evening journal", due: now.addingTimeInterval(9 * 3600),
            wakeClass: .rhythm)
        var today = DayRecord(date: TrackingDay.key(for: now))
        today.chain = [ContractStep(title: "Ship the loop", status: .active)]
        var yesterday = DayRecord(date: TrackingDay.yesterdayKey(from: now))
        yesterday.seed = "start with the evaluator"

        let text = CompanionBriefing.render(
            CompanionBriefing.Inputs(
                now: now,
                ownerPresent: true,
                screenLocked: false,
                frontmostApp: "Xcode",
                onACPower: true,
                today: today,
                yesterday: yesterday,
                dueWakes: [due],
                upcomingWakes: [upcoming],
                weeklyNumbers: nil
            ))

        #expect(text.hasPrefix("<situation>"))
        #expect(text.hasSuffix("</situation>"))
        #expect(text.contains("He is at the Mac, in Xcode."))
        #expect(text.contains("Power: AC."))
        #expect(text.contains("Seed left for today: start with the evaluator"))
        #expect(text.contains("Yesterday was never closed"))
        #expect(text.contains("DUE NOW"))
        #expect(text.contains("[rhythm] morning planning (overdue by 8 min)"))
        #expect(text.contains("Your booked future:"))
        #expect(text.contains("evening journal"))
    }

    @Test func appUseEvidenceRendersRecencyOrItsAbsence() {
        let now = Date()
        var inputs = CompanionBriefing.Inputs(
            now: now, ownerPresent: true, screenLocked: false, frontmostApp: nil,
            onACPower: true, today: nil, yesterday: nil, dueWakes: [],
            upcomingWakes: [], weeklyNumbers: nil)
        #expect(
            CompanionBriefing.render(inputs)
                .contains("He has not used this app since it launched."))

        inputs.lastAppUse = now.addingTimeInterval(-3 * 60)
        #expect(
            CompanionBriefing.render(inputs).contains("He last used this app 3 min ago."))

        inputs.lastAppUse = now.addingTimeInterval(-10)
        #expect(
            CompanionBriefing.render(inputs).contains("He was using this app moments ago."))
    }

    @Test func calendarLinesRideTheSituationBlock() {
        var inputs = CompanionBriefing.Inputs(
            now: Date(), ownerPresent: true, screenLocked: false, frontmostApp: nil,
            onACPower: true, today: nil, yesterday: nil, dueWakes: [],
            upcomingWakes: [], weeklyNumbers: nil)
        inputs.calendarLines = [
            "Calendar — the rest of his day:", "- 15:00 Dentist",
        ]
        let text = CompanionBriefing.render(inputs)
        #expect(text.contains("Calendar — the rest of his day:"))
        #expect(text.contains("- 15:00 Dentist"))
    }

    @Test func resurfacedPromisesRideAsAgendaLines() {
        var inputs = CompanionBriefing.Inputs(
            now: Date(), ownerPresent: true, screenLocked: false, frontmostApp: nil,
            onACPower: true, today: nil, yesterday: nil, dueWakes: [],
            upcomingWakes: [], weeklyNumbers: nil)
        inputs.resurfacedWakes = [
            CompanionWake(
                content: "ask about the dentist", due: Date(), wakeClass: .promise,
                state: .resurfaced)
        ]
        let text = CompanionBriefing.render(inputs)
        #expect(text.contains("STILL OWED"))
        #expect(text.contains("- ask about the dentist"))
    }

    @Test func emptyFutureDemandsARhythm() {
        let text = CompanionBriefing.render(
            CompanionBriefing.Inputs(
                now: Date(), ownerPresent: false, screenLocked: true, frontmostApp: nil,
                onACPower: false, today: nil, yesterday: nil, dueWakes: [],
                upcomingWakes: [], weeklyNumbers: nil))
        #expect(text.contains("He is away — the screen is locked."))
        #expect(text.contains("No contract for today yet."))
        #expect(text.contains("You have NOTHING booked ahead"))
    }
}
