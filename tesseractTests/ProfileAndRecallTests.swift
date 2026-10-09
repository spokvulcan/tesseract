//
//  ProfileAndRecallTests.swift
//  tesseractTests
//
//  The Profile and recall at their seams: facts come in only when asked or
//  approved, proposals wait for a decision, and recall finds dated snippets
//  in past conversations (fixture files in a scratch folder — never the
//  owner's conversations, ADR-0073), skipping moment cards.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct ProfileStoreTests {

    @Test func factsAreAddedOnceAndCanBeEditedAndDeleted() throws {
        let url = makeTempDir("profile").appendingPathComponent("profile.json")
        let store = ProfileStore(url: url)
        let fact = try #require(store.add("Goes to the gym on Mondays.", source: .chat))
        #expect(store.add("goes to the gym on mondays.", source: .owner)?.id == fact.id)
        store.update(fact.id, text: "Goes to the gym on Mondays and Thursdays.")
        #expect(
            ProfileStore(url: url).facts.map(\.text) == [
                "Goes to the gym on Mondays and Thursdays."
            ])
        store.delete(fact.id)
        #expect(ProfileStore(url: url).facts.isEmpty)
    }

    @Test func proposalsWaitForTheOwnersDecision() throws {
        let store = ProfileStore(url: nil)
        store.add("Works on Tesseract in the evenings.", source: .chat)
        store.propose(
            [
                ProposalDraft(text: "Works on Tesseract in the evenings.", reason: "known"),
                ProposalDraft(text: "Prefers walks after lunch.", reason: "walked twice today"),
                ProposalDraft(text: "Calls their sister on Sundays.", reason: "mentioned it"),
                ProposalDraft(text: "Drinks green tea.", reason: "x"),
                ProposalDraft(text: "Fourth.", reason: "x"),
            ], source: "night-reflection")
        // Already known is skipped; at most three are drafted.
        #expect(
            store.openProposals.map(\.text) == [
                "Prefers walks after lunch.", "Calls their sister on Sundays.",
            ])
        #expect(store.facts.count == 1)

        let walk = try #require(store.openProposals.first)
        store.decide(walk.id, keep: true, editedText: "Likes a walk after lunch.")
        let sister = try #require(store.openProposals.first)
        store.decide(sister.id, keep: false)
        #expect(
            store.facts.map(\.text) == [
                "Works on Tesseract in the evenings.", "Likes a walk after lunch.",
            ])
        #expect(store.proposals.map(\.status) == [.edited, .dropped])
        #expect(store.openProposals.isEmpty)
    }
}

struct RecallIndexTests {

    /// A conversation file in the persisted format.
    static func writeConversation(
        _ dir: URL, id: String = UUID().uuidString, title: String, messages: [[String: Any]]
    ) throws -> URL {
        let url = dir.appendingPathComponent("\(id).json")
        let root: [String: Any] = ["id": id, "title": title, "messages": messages]
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        return url
    }

    static func user(_ text: String, at: Date, moment: Bool = false) -> [String: Any] {
        var payload: [String: Any] = [
            "content": text, "timestamp": at.timeIntervalSinceReferenceDate,
            "id": UUID().uuidString,
            "images": [],
        ]
        if moment { payload["turnOrigin"] = "moment" }
        return ["type": "user", "payload": payload]
    }

    static func assistant(_ text: String, at: Date) -> [String: Any] {
        [
            "type": "assistant",
            "payload": [
                "timestamp": at.timeIntervalSinceReferenceDate,
                "content": [
                    ["type": "thinking", "thinking": "hmm"], ["type": "text", "text": text],
                ],
            ],
        ]
    }

    static let monday = Date(timeIntervalSince1970: 1_790_064_000)

    @Test func messagesAreTheOwnersWordsAndJarvisRepliesWithoutCards() throws {
        let dir = makeTempDir("recall")
        let url = try Self.writeConversation(
            dir, title: "Knee",
            messages: [
                Self.user("[Morning Plan]\nPlan the day.", at: Self.monday, moment: true),
                Self.assistant(#"{"line": "A calm day."}"#, at: Self.monday),
                Self.user(
                    "<skill name=\"x\">long instructions</skill>my left knee hurts after long runs",
                    at: Self.monday),
                Self.assistant("Ease off for a few days and ice it.", at: Self.monday),
            ])
        let messages = RecallIndex.messages(in: url)
        #expect(messages.map(\.role) == ["owner", "jarvis"])
        #expect(messages.first?.text == "my left knee hurts after long runs")
    }

    @Test func searchFindsDatedSnippetsAndFollowsChanges() async throws {
        let dir = makeTempDir("recall")
        let index = RecallIndex(
            conversationsDirectory: dir,
            databaseURL: makeTempDir("recall-db").appendingPathComponent("r.sqlite"))
        _ = try Self.writeConversation(
            dir, id: "knee", title: "Knee",
            messages: [Self.user("my left knee hurts after long runs", at: Self.monday)])
        _ = try Self.writeConversation(
            dir, id: "trip", title: "Trip",
            messages: [Self.user("book the train to Berlin", at: Self.monday)])

        let hits = await index.search(
            "what did I say about my knee?", now: Self.monday.addingTimeInterval(86_400))
        #expect(hits.map(\.conversationID) == ["knee"])
        #expect(hits.first?.snippet.contains("knee") == true)

        // A search a month later with a one-week window finds nothing.
        let old = await index.search(
            "knee", days: 7, now: Self.monday.addingTimeInterval(30 * 86_400))
        #expect(old.isEmpty)

        // Deleting the file drops it from the index.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("knee.json"))
        #expect(await index.search("knee", now: Self.monday).isEmpty)
    }

    @Test @MainActor func theRecallToolReadsProfileAndConversations() async throws {
        let dir = makeTempDir("recall")
        _ = try Self.writeConversation(
            dir, title: "Knee",
            messages: [Self.user("my left knee hurts after long runs", at: Self.monday)])
        let profile = ProfileStore(url: nil)
        profile.add("Runs on Saturdays with the knee brace.", source: .chat)
        let index = RecallIndex(
            conversationsDirectory: dir,
            databaseURL: makeTempDir("recall-db").appendingPathComponent("r.sqlite"))
        let tools = Dictionary(
            uniqueKeysWithValues: createProfileTools(profile: profile, index: index).map {
                ($0.name, $0)
            })
        let recall = try #require(tools["recall"])
        let text = try await recall.execute("c", ["query": .string("knee")], nil, nil).content
            .textContent
        #expect(text.contains("From their Profile:"))
        #expect(text.contains("- Runs on Saturdays with the knee brace."))
        #expect(text.contains("From past conversations:"))
        #expect(text.contains("they said: my left knee hurts after long runs"))

        let remember = try #require(tools["remember"])
        let saved = try await remember.execute(
            "c", ["fact": .string("Prefers short answers.")], nil, nil)
        #expect(saved.content.textContent == "Remembered: Prefers short answers.")
        let forget = try #require(tools["forget"])
        let gone = try await forget.execute("c", ["fact": .string("short answers")], nil, nil)
        #expect(gone.content.textContent == "Forgotten: Prefers short answers.")
    }
}

struct NightReflectionTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static func snapshot(at now: Date, power: PowerState = .nominal, present: Bool = false)
        -> DaySnapshot
    {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        return DaySnapshot(
            now: now, settings: DaySettings(), agenda: agenda, ownerPresent: present, power: power,
            profile: ["Works on Tesseract in the evenings."])
    }

    static func afterWrapUp() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.eveningWrapUpAt = local(30, 21, 10)
        return state
    }

    @Test func itRunsOnceHalfAnHourAfterTheWrapUp() {
        let early = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 21, 20)), state: Self.afterWrapUp())
        #expect(!early.effects.contains { if case .runMoment = $0 { true } else { false } })
        let due = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 21, 45)), state: Self.afterWrapUp())
        let request = due.effects.compactMap { if case .runMoment(let r) = $0 { r } else { nil } }
            .first
        #expect(request?.kind == .nightReflection)
        #expect(request?.text.contains("- Works on Tesseract in the evenings.") == true)
    }

    @Test func onAHealthyBatteryItRuns() {
        // 85% on battery at 23:00: the night of the 30th, which the old
        // power-only rule skipped.
        let battery = PowerState(onACPower: false, batteryPercent: 85, thermal: .nominal)
        let decided = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 23), power: battery),
            state: Self.afterWrapUp())
        #expect(
            decided.effects.contains {
                if case .runMoment(let r) = $0 { r.kind == .nightReflection } else { false }
            })
    }

    @Test func onALowBatteryItWaitsForPower() {
        let battery = PowerState(onACPower: false, batteryPercent: 30, thermal: .nominal)
        let held = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 22), power: battery),
            state: Self.afterWrapUp())
        #expect(held.state.deferred.contains(.nightReflection))
        #expect(
            held.effects.contains {
                if case .trace(.governorDeferred, _) = $0 { true } else { false }
            })
        let plugged = DayEngine.decide(
            .powerChanged, snapshot: Self.snapshot(at: Self.local(30, 23)), state: held.state)
        #expect(
            plugged.effects.contains {
                if case .runMoment(let r) = $0 { r.kind == .nightReflection } else { false }
            })
    }

    @Test func aTaskTheNightProposedIsAddedWithOneClickOrLetGo() throws {
        var state = Self.afterWrapUp()
        state.running = .nightReflection
        let reply =
            #"{"carry_over": "Good.", "tasks": [{"title": "Send the request", "when": "tomorrow"}, {"title": "Book the bike service", "when": "later"}]}"#
        let measure = MomentMeasure(
            promptTokens: 3000, outputTokens: 400, prefillSeconds: 0.5, generateSeconds: 12,
            latencySeconds: 13, hitCap: false, modelID: "m")
        let night = DayEngine.decide(
            .momentOutcome(
                MomentRequest(kind: .nightReflection, trigger: .night, text: "x"),
                .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 22)), state: state)
        #expect(night.state.taskProposals.count == 2)
        #expect(night.effects.contains(.trace(.taskProposed, ["count": .int(2)])))
        // The next morning: still there, due on what is now today.
        let morning = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(31, 7)), state: night.state)
        let first = try #require(morning.state.taskProposals.first)
        let added = DayEngine.decide(
            .cardAction(.taskProposal(id: first.id, add: true)),
            snapshot: Self.snapshot(at: Self.local(31, 7, 5)), state: morning.state)
        #expect(
            added.effects.contains(.mutateAgenda(.add(title: "Send the request", due: first.due))))
        #expect(added.state.taskProposals.count == 1)
        let last = try #require(added.state.taskProposals.first)
        let declined = DayEngine.decide(
            .cardAction(.taskProposal(id: last.id, add: false)),
            snapshot: Self.snapshot(at: Self.local(31, 7, 6)), state: added.state)
        #expect(!declined.effects.contains { if case .mutateAgenda = $0 { true } else { false } })
        #expect(declined.state.taskProposals.isEmpty)
        // Left undecided, they don't outlive the next day.
        let dayAfter = morning.state.rolledOver(to: DayKey(rawValue: "2026-10-02"))
        #expect(dayAfter.taskProposals.isEmpty)
    }

    @Test func itsNoteOpensTomorrowAndItsProposalsGoToTheProfile() throws {
        var state = Self.afterWrapUp()
        state.running = .nightReflection
        let reply =
            #"{"carry_over": "The spec is nearly done; finish it first.", "tomorrow": ["Finish the spec"], "proposals": [{"text": "Prefers walks after lunch.", "reason": "walked twice"}]}"#
        let measure = MomentMeasure(
            promptTokens: 3000, outputTokens: 400, prefillSeconds: 0.5, generateSeconds: 12,
            latencySeconds: 13, hitCap: false, modelID: "m")
        let decided = DayEngine.decide(
            .momentOutcome(
                MomentRequest(kind: .nightReflection, trigger: .night, text: "x"),
                .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 22)), state: state)
        #expect(decided.state.carryOverForNextDay == "The spec is nearly done; finish it first.")
        #expect(
            decided.effects.contains(
                .proposeFacts([
                    ProposalDraft(text: "Prefers walks after lunch.", reason: "walked twice")
                ])))
        #expect(
            decided.effects.contains {
                if case .presentCard(_, .today) = $0 { true } else { false }
            })

        let morning = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(31, 7)), state: decided.state)
        #expect(morning.state.carryOver == "The spec is nearly done; finish it first.")
        // Its first draft of tomorrow opens the next day too, for the plan.
        #expect(decided.state.draftForNextDay == ["Finish the spec"])
        #expect(morning.state.draft == ["Finish the spec"])
        #expect(morning.state.draftForNextDay.isEmpty)
        // A day skipped: last night's draft is no longer last night's.
        let skipped = decided.state.rolledOver(to: DayKey(rawValue: "2026-10-02"))
        #expect(skipped.draft.isEmpty)
    }
}
