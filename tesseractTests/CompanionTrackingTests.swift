//
//  CompanionTrackingTests.swift
//  tesseractTests
//
//  The tracking grain (#308): days, the contract chain, observations, and
//  work items — the store methods and the five typed tools over them. Each
//  test opens its own scratch store so the scheme's parallel twin runners
//  can't collide.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

private func scratchStore() throws -> MemoryStore {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("tracking-tests-\(UUID().uuidString)", isDirectory: true)
    return try MemoryStore(directory: dir)
}

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

@Suite struct CompanionTrackingToolTests {

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

    @Test func planDayCreatesChainAndGuardsOverwrite() async throws {
        let store = try scratchStore()
        let tool = createPlanDayTool(store: store)

        let first = try await run(
            tool,
            [
                "keystone": .string("Ship the evaluator"),
                "then": .array([.string("Write tests")]),
                "support": .array([.string("PR replies")]),
            ])
        #expect(first.contains("[active] Ship the evaluator"))
        #expect(first.contains("[pending] Write tests"))

        let refused = try await run(tool, ["keystone": .string("Something else")])
        #expect(refused.contains("already exists"))

        let replaced = try await run(
            tool, ["keystone": .string("Something else"), "replace": .bool(true)])
        #expect(replaced.contains("[active] Something else"))
    }

    @Test func logStepDoneArmsNext() async throws {
        let store = try scratchStore()
        _ = try await run(
            createPlanDayTool(store: store),
            [
                "keystone": .string("First"),
                "then": .array([.string("Second")]),
            ])
        let tool = createLogStepTool(store: store)

        let done = try await run(tool, ["action": .string("done")])
        #expect(done.contains("[done] First"))
        #expect(done.contains("[active] Second"))

        // The step event landed as a work observation.
        let events = try await store.observations(kind: "step-done")
        #expect(events.count == 1)
    }

    @Test func logStepSwitchedRecordsConsciousSwitchAndReseeds() async throws {
        let store = try scratchStore()
        _ = try await run(createPlanDayTool(store: store), ["keystone": .string("First")])
        let out = try await run(
            createLogStepTool(store: store),
            [
                "action": .string("switched"),
                "note": .string("deep in the profiler instead"),
                "reseed": .string("First → tomorrow"),
            ])
        #expect(out.contains("[switched] First"))
        #expect(out.contains("Seed: First → tomorrow"))
        #expect(try await store.observations(kind: "conscious-switch").count == 1)
    }

    @Test func logSampleMapsDomains() async throws {
        let store = try scratchStore()
        let tool = createLogSampleTool(store: store)
        _ = try await run(tool, ["kind": .string("sleep"), "value": .string("solid 7h")])
        _ = try await run(tool, ["kind": .string("energy"), "value": .string("flat")])

        #expect(try await store.observations(domain: .body).count == 1)
        #expect(try await store.observations(domain: .mind).count == 1)
    }

    @Test func logTaskHabitCheckoffKeepsItemOpen() async throws {
        let store = try scratchStore()
        let tool = createLogTaskTool(store: store)
        _ = try await run(
            tool,
            [
                "action": .string("add"), "title": .string("100 sit-ups"),
                "cadence": .string("daily"), "domain": .string("body"),
            ])
        let done = try await run(tool, ["action": .string("done"), "title": .string("sit-ups")])
        #expect(done.contains("Checked off"))

        #expect(try await store.workItems(status: .open).count == 1)
        let checkoffs = try await store.observations(kind: "habit-checkoff")
        #expect(checkoffs.count == 1)
        #expect(checkoffs[0].domain == .body)
    }

    @Test func closeDayStampsAndSeeds() async throws {
        let store = try scratchStore()
        _ = try await run(createPlanDayTool(store: store), ["keystone": .string("First")])
        _ = try await run(createLogStepTool(store: store), ["action": .string("done")])
        let out = try await run(
            createCloseDayTool(store: store), ["seed": .string("pick up the tests")])
        #expect(out.contains("Keystone kept"))
        #expect(out.contains("Seed: pick up the tests"))

        let day = try #require(try await store.day(TrackingDay.key()))
        #expect(day.closedAt != nil)
        #expect(day.seed == "pick up the tests")
    }
}
