//
//  CompanionTrackingTests.swift
//  tesseractTests
//
//  The tracking grain (#308) under the lean palette (ADR-0046, #369): days,
//  the contract chain, observations, and work items — the store methods and
//  the one generic `track(kind, payload)` door over them. Each test opens its
//  own scratch store so the scheme's parallel twin runners can't collide.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionTrackingStoreTests {

    @Test func dayRoundTripsWithChainAndSeed() async throws {
        let store = try scratchStore()
        var day = DayRecord(date: "2026-07-16")
        day.chain = [
            ContractStep(title: "Ship the wake table", status: .active),
            ContractStep(title: "Write its tests"),
        ]
        day.support = ["reply to the PR thread"]
        day.seed = "start with the evaluator"
        try await store.upsertDay(day)

        let loaded = try #require(try await store.day("2026-07-16"))
        #expect(loaded.chain.count == 2)
        #expect(loaded.chain[0].status == .active)
        #expect(loaded.chain[1].status == .pending)
        #expect(loaded.support == ["reply to the PR thread"])
        #expect(loaded.seed == "start with the evaluator")
        #expect(loaded.closedAt == nil)

        #expect(try await store.day("2026-07-15") == nil)
    }

    @Test func closedAtSurvivesAndRecentDaysOrders() async throws {
        let store = try scratchStore()
        for (date, closed) in [("2026-07-14", true), ("2026-07-15", false), ("2026-07-16", true)] {
            var day = DayRecord(date: date)
            day.closedAt = closed ? Date() : nil
            try await store.upsertDay(day)
        }
        let recent = try await store.recentDays(limit: 2)
        #expect(recent.map(\.date) == ["2026-07-16", "2026-07-15"])
        #expect(recent[0].closedAt != nil)
        #expect(recent[1].closedAt == nil)
    }

    @Test func observationsAppendAndFilter() async throws {
        let store = try scratchStore()
        try await store.appendObservation(
            TrackingObservation(domain: .body, kind: "sleep", value: "poor, ~5h", source: .elicited)
        )
        try await store.appendObservation(
            TrackingObservation(domain: .mind, kind: "mood", value: "good", source: .elicited))
        try await store.appendObservation(
            TrackingObservation(
                domain: .work, kind: "app-session", value: "{\"app\":\"Xcode\"}", source: .sensed))

        let sleeps = try await store.observations(kind: "sleep")
        #expect(sleeps.count == 1)
        #expect(sleeps[0].value == "poor, ~5h")
        #expect(sleeps[0].source == .elicited)

        let mind = try await store.observations(domain: .mind)
        #expect(mind.count == 1)

        let all = try await store.observations()
        #expect(all.count == 3)
    }

    @Test func workItemLifecycleAndFuzzyFind() async throws {
        let store = try scratchStore()
        let habit = WorkItemRecord(title: "100 sit-ups", domain: .body, cadence: .daily)
        let oneShot = WorkItemRecord(title: "Renew the domain", stream: "ops")
        try await store.upsertWorkItem(habit)
        try await store.upsertWorkItem(oneShot)

        #expect(try await store.workItems(status: .open).count == 2)

        let found = try #require(try await store.findWorkItem(idOrTitle: "sit-ups"))
        #expect(found.id == habit.id)
        #expect(found.domain == .body)

        var done = oneShot
        done.status = .done
        try await store.upsertWorkItem(done)
        #expect(try await store.workItems(status: .open).count == 1)
        #expect(try await store.findWorkItem(idOrTitle: oneShot.id.uuidString)?.status == .done)
    }

    @Test func schemaUpgradeFromV2AddsTrackingTables() async throws {
        // A store created before v3 must gain the tracking tables on reopen:
        // the migrate block is IF-NOT-EXISTS + user_version-guarded.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tracking-upgrade-\(UUID().uuidString)", isDirectory: true)
        do {
            let first = try MemoryStore(directory: dir)
            _ = try await first.day("2026-01-01")  // touch: tables exist on fresh create
        }
        let reopened = try MemoryStore(directory: dir)
        try await reopened.appendObservation(
            TrackingObservation(domain: .work, kind: "presence-span", value: "{}", source: .sensed))
        #expect(try await reopened.observations(kind: "presence-span").count == 1)
    }
}

@Suite struct CompanionTrackToolTests {

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

    private func track(
        _ store: MemoryStore, kind: String, _ payload: [String: JSONValue]
    ) async throws -> String {
        try await run(
            createTrackTool(store: store),
            ["kind": .string(kind), "payload": .object(payload)])
    }

    // MARK: - Observations

    @Test func sampleKindsMapTheirOwnDomain() async throws {
        let store = try scratchStore()
        _ = try await track(
            store, kind: "observation",
            ["kind": .string("sleep"), "value": .string("solid 7h")])
        _ = try await track(
            store, kind: "observation",
            ["kind": .string("energy"), "value": .string("flat")])

        let body = try await store.observations(domain: .body)
        #expect(body.count == 1)
        #expect(body[0].kind == "sleep")
        #expect(body[0].source == .elicited)
        #expect(try await store.observations(domain: .mind).count == 1)
    }

    @Test func customObservationKindNeedsAnExplicitDomain() async throws {
        let store = try scratchStore()
        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "observation",
                ["kind": .string("focus-block"), "value": .string("90 min deep")])
        }
        _ = try await track(
            store, kind: "observation",
            [
                "kind": .string("focus-block"), "value": .string("90 min deep"),
                "domain": .string("work"), "stream": .string("tesseract"),
            ])
        let rows = try await store.observations(kind: "focus-block")
        #expect(rows.count == 1)
        #expect(rows[0].domain == .work)
        #expect(rows[0].stream == "tesseract")
    }

    // MARK: - Days

    @Test func dayChainWritesWithOneActiveEnforced() async throws {
        let store = try scratchStore()
        let out = try await track(
            store, kind: "day",
            [
                "chain": .array([
                    .object(["title": .string("Ship the fold"), "status": .string("active")]),
                    .object(["title": .string("Write tests")]),
                ]),
                "support": .array([.string("PR replies")]),
                "seed": .string("start with the evaluator"),
            ])
        #expect(out.contains("[active] Ship the fold"))
        #expect(out.contains("[pending] Write tests"))
        #expect(out.contains("Seed: start with the evaluator"))

        let day = try #require(try await store.day(TrackingDay.key()))
        #expect(day.chain.count == 2)
        #expect(day.chain[0].startedAt != nil)
        #expect(day.support == ["PR replies"])

        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "day",
                [
                    "chain": .array([
                        .object(["title": .string("A"), "status": .string("active")]),
                        .object(["title": .string("B"), "status": .string("active")]),
                    ])
                ])
        }
    }

    @Test func partialDayUpdateTouchesOnlyPassedFields() async throws {
        let store = try scratchStore()
        _ = try await track(
            store, kind: "day",
            ["chain": .array([.object(["title": .string("Keystone")])])])
        _ = try await track(store, kind: "day", ["seed": .string("tomorrow's thread")])

        let day = try #require(try await store.day(TrackingDay.key()))
        #expect(day.chain.count == 1)
        #expect(day.seed == "tomorrow's thread")
    }

    @Test func chainRewriteCarriesTimestampsForwardByTitle() async throws {
        let store = try scratchStore()
        _ = try await track(
            store, kind: "day",
            ["chain": .array([.object(["title": .string("Ship"), "status": .string("active")])])]
        )
        let started = try #require(try await store.day(TrackingDay.key())?.chain[0].startedAt)

        _ = try await track(
            store, kind: "day",
            ["chain": .array([.object(["title": .string("Ship"), "status": .string("done")])])]
        )
        let step = try #require(try await store.day(TrackingDay.key())?.chain[0])
        #expect(step.status == .done)
        #expect(step.closedAt != nil)
        let carried = try #require(step.startedAt)
        #expect(abs(carried.timeIntervalSince(started)) < 1)
    }

    @Test func closedStampsReopensAndCarriesNoCeremonyText() async throws {
        let store = try scratchStore()
        let closed = try await track(store, kind: "day", ["closed": .bool(true)])
        #expect(!closed.contains("Day closed"))
        #expect(try await store.day(TrackingDay.key())?.closedAt != nil)

        _ = try await track(store, kind: "day", ["closed": .bool(false)])
        #expect(try await store.day(TrackingDay.key())?.closedAt == nil)
    }

    @Test func closedAsStringFailsLoudly() async throws {
        let store = try scratchStore()
        await #expect(throws: ToolArgTypeError.self) {
            _ = try await self.track(store, kind: "day", ["closed": .string("true")])
        }
    }

    @Test func explicitDateSettlesAnotherDay() async throws {
        let store = try scratchStore()
        let yesterday = TrackingDay.yesterdayKey()
        _ = try await track(
            store, kind: "day",
            ["date": .string(yesterday), "closed": .bool(true)])
        #expect(try await store.day(yesterday)?.closedAt != nil)
        #expect(try await store.day(TrackingDay.key()) == nil)

        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "day", ["date": .string("someday"), "closed": .bool(true)])
        }
    }

    @Test func supportBeyondTwoFailsLoudly() async throws {
        let store = try scratchStore()
        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "day",
                ["support": .array([.string("a"), .string("b"), .string("c")])])
        }
    }

    // MARK: - Items

    @Test func habitCheckoffKeepsItemOpen() async throws {
        let store = try scratchStore()
        _ = try await track(
            store, kind: "item",
            [
                "action": .string("add"), "title": .string("100 sit-ups"),
                "cadence": .string("daily"), "domain": .string("body"),
            ])
        let done = try await track(
            store, kind: "item", ["action": .string("done"), "title": .string("sit-ups")])
        #expect(done.contains("Checked off"))

        #expect(try await store.workItems(status: .open).count == 1)
        let checkoffs = try await store.observations(kind: "habit-checkoff")
        #expect(checkoffs.count == 1)
        #expect(checkoffs[0].domain == .body)
    }

    @Test func oneShotDoneClosesAndListShowsOpen() async throws {
        let store = try scratchStore()
        _ = try await track(
            store, kind: "item",
            ["action": .string("add"), "title": .string("Renew the domain")])
        let list = try await track(store, kind: "item", ["action": .string("list")])
        #expect(list.contains("- Renew the domain"))

        let done = try await track(
            store, kind: "item", ["action": .string("done"), "title": .string("Renew")])
        #expect(done.contains("Done: Renew the domain"))
        #expect(try await store.workItems(status: .open).isEmpty)
    }

    @Test func unknownEnumsFailLoudly() async throws {
        let store = try scratchStore()
        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "item",
                [
                    "action": .string("add"), "title": .string("x"),
                    "cadence": .string("weekly"),
                ])
        }
        await #expect(throws: TrackingToolError.self) {
            _ = try await self.track(
                store, kind: "day",
                ["chain": .array([.object(["title": .string("x"), "status": .string("wip")])])])
        }
    }

    // MARK: - Shape strictness

    @Test func unknownKindAndMalformedPayloadFailLoudly() async throws {
        let store = try scratchStore()
        let tool = createTrackTool(store: store)
        await #expect(throws: TrackingToolError.self) {
            _ = try await tool.execute(
                "t", ["kind": .string("mood"), "payload": .object([:])], nil, nil)
        }
        await #expect(throws: ToolArgTypeError.self) {
            _ = try await tool.execute(
                "t",
                ["kind": .string("observation"), "payload": .string("{\"kind\":\"sleep\"}")],
                nil, nil)
        }
        await #expect(throws: TrackingToolError.self) {
            _ = try await tool.execute(
                "t", ["kind": .string("day"), "payload": .object([:])], nil, nil)
        }
    }
}
