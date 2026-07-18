//
//  CompanionEventStoreTests.swift
//  tesseractTests
//
//  The Event queue's invariants (ADR-0046, #368), in the wake-store style:
//  exactly-once admission, total order, coalesced drain, the consume /
//  re-present state machine, and restart safety. Plus the producers'
//  deterministic-occasion dedupe and the sensed pipeline's sustained-vs-brief
//  verdict. Each test opens its own scratch store so the scheme's parallel
//  twin runners can't collide.
//

import Foundation
import Testing

@testable import Tesseract_Agent

private func event(
    _ content: String, kind: CompanionEventKind = .macWake, id: UUID = UUID()
) -> CompanionEvent {
    CompanionEvent(id: id, kind: kind, content: content)
}

// MARK: - The queue's math

@Suite struct CompanionEventStoreTests {

    @Test func admissionIsExactlyOnce() async throws {
        let store = try scratchStore()
        let one = event("the Mac woke")

        #expect(try await store.admitEvent(one) == true)
        #expect(try await store.admitEvent(one) == false)

        let pending = try await store.pendingEvents()
        #expect(pending.map(\.id) == [one.id])
    }

    @Test func drainReturnsEverythingPendingOrderedExactlyOnce() async throws {
        let store = try scratchStore()
        let first = event("first")
        let second = event("second")
        let third = event("third")
        for item in [first, second, third] {
            try await store.admitEvent(item)
        }

        // A burst coalesces into one drain, in admission order.
        let batch = try await store.drainPendingEvents()
        #expect(batch.map(\.id) == [first.id, second.id, third.id])
        #expect(batch.allSatisfy { $0.state == .presented })
        let seqs = batch.compactMap(\.seq)
        #expect(seqs == seqs.sorted())

        // Exactly once: a second drain finds nothing.
        #expect(try await store.drainPendingEvents().isEmpty)
        #expect(try await store.pendingEvents().isEmpty)
    }

    @Test func queueSurvivesRelaunch() async throws {
        let dir = makeTempDir("event-relaunch")
        let store = try MemoryStore(directory: dir)
        let one = event("before the crash")
        let two = event("also before")
        try await store.admitEvent(one)
        try await store.admitEvent(two)

        // A fresh store over the same directory: nothing lost, order kept,
        // and re-admitting a known id is still a duplicate.
        let relaunched = try MemoryStore(directory: dir)
        let pending = try await relaunched.pendingEvents()
        #expect(pending.map(\.id) == [one.id, two.id])
        #expect(try await relaunched.admitEvent(one) == false)
    }

    @Test func consumeIsOnlyByACompletedTurn() async throws {
        let store = try scratchStore()
        let one = event("to consume")
        try await store.admitEvent(one)

        let batch = try await store.drainPendingEvents()
        // Presented is not consumed — the batch sits in the recovery set
        // until the turn completes (the wake invariant).
        #expect(try await store.unconsumedPresentedEvents().map(\.id) == [one.id])

        let turnID = UUID()
        try await store.consumeEvents(ids: batch.map(\.id), turnID: turnID)
        #expect(try await store.unconsumedPresentedEvents().isEmpty)
    }

    @Test func crashedBatchRepresentsInOriginalOrder() async throws {
        let store = try scratchStore()
        let first = event("first")
        let second = event("second")
        try await store.admitEvent(first)
        try await store.admitEvent(second)

        // Drained, then the turn crashed: recovery re-presents the batch.
        let batch = try await store.drainPendingEvents()
        try await store.representEvents(ids: batch.map(\.id))

        let again = try await store.drainPendingEvents()
        #expect(again.map(\.id) == [first.id, second.id])
    }

    @Test func deterministicIDCollapsesRepeatOccasions() async throws {
        #expect(
            CompanionEvent.deterministicID("day-end:2026-07-18")
                == CompanionEvent.deterministicID("day-end:2026-07-18"))
        #expect(
            CompanionEvent.deterministicID("day-end:2026-07-18")
                != CompanionEvent.deterministicID("day-end:2026-07-19"))

        // The same occasion admitted twice is one Event, no producer state.
        let store = try scratchStore()
        let occasion = CompanionEvent.deterministicID("day-end:2026-07-18")
        try await store.admitEvent(event("the day ended", kind: .dayEnd, id: occasion))
        #expect(
            try await store.admitEvent(event("the day ended", kind: .dayEnd, id: occasion))
                == false)
        #expect(try await store.pendingEvents().count == 1)
    }
}

// MARK: - The producers

@Suite @MainActor struct CompanionPerceptionTests {

    /// `isTestHost` defaults false here: these tests exercise the very door
    /// the test-host gate closes.
    private func makePerception(
        enabled: Bool = true, isTestHost: Bool = false
    ) throws -> (CompanionPerception, MemoryStore) {
        let store = try scratchStore()
        let perception = CompanionPerception(
            store: store, recorder: scratchRecorder(), isEnabled: { enabled },
            isTestHost: isTestHost)
        return (perception, store)
    }

    /// Admission is fire-and-forget off the doors — poll the store briefly.
    private func waitForPending(
        _ store: MemoryStore, count: Int, timeout: Duration = .seconds(5)
    ) async throws -> [CompanionEvent] {
        let deadline = ContinuousClock.now + timeout
        while true {
            let pending = try await store.pendingEvents()
            if pending.count >= count || ContinuousClock.now >= deadline { return pending }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    @Test func dayRolloverAdmitsOneDayEndPerDay() async throws {
        let (perception, store) = try makePerception()
        let now = Date()

        perception.dayRolled(now: now)
        perception.dayRolled(now: now)

        let pending = try await waitForPending(store, count: 1)
        #expect(pending.count == 1)
        #expect(pending.first?.kind == .dayEnd)
    }

    @Test func dayStartIsOncePerDayAndCarriesTheDay() async throws {
        let (perception, store) = try makePerception()
        let nineAM = try #require(
            Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date()))

        perception.dayStartIfDue(now: nineAM, ownerPresent: true)
        perception.dayStartIfDue(now: nineAM, ownerPresent: true)
        // A relaunch re-attempts with a fresh in-process edge and collapses
        // at the store — the deterministic id, not producer state, is the
        // once-only truth.
        let relaunched = CompanionPerception(
            store: store, recorder: scratchRecorder(), isEnabled: { true },
            isTestHost: false)
        relaunched.dayStartIfDue(now: nineAM, ownerPresent: true)

        let pending = try await waitForPending(store, count: 1)
        #expect(pending.count == 1)
        #expect(pending.first?.kind == .dayStart)
        #expect(pending.first?.payload?.contains(TrackingDay.key(for: nineAM)) == true)
    }

    @Test func aOneAMTailCountsAsYesterday() async throws {
        // #371: the detector needs the calendar day begun in earnest.
        let (perception, store) = try makePerception()
        let twoAM = try #require(
            Calendar.current.date(bySettingHour: 2, minute: 0, second: 0, of: Date()))
        perception.dayStartIfDue(now: twoAM, ownerPresent: true)
        try await Task.sleep(for: .milliseconds(50))
        #expect(try await store.pendingEvents().isEmpty)
    }

    @Test func dayStartWaitsForHimToActuallyBeThere() async throws {
        let (perception, store) = try makePerception()
        let nineAM = try #require(
            Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date()))
        perception.dayStartIfDue(now: nineAM, ownerPresent: false)
        try await Task.sleep(for: .milliseconds(50))
        #expect(try await store.pendingEvents().isEmpty)
        // Presence arriving later the same morning still admits — an absent
        // tick must not burn the day's one edge.
        perception.dayStartIfDue(now: nineAM, ownerPresent: true)
        #expect(try await waitForPending(store, count: 1).count == 1)
    }

    @Test func disabledCompanionAdmitsNothing() async throws {
        let (perception, store) = try makePerception(enabled: false)

        perception.dayRolled(now: Date())
        perception.powerChanged(onACPower: false)

        // The toggle gate is synchronous — no admission task was ever spawned.
        try await Task.sleep(for: .milliseconds(50))
        #expect(try await store.pendingEvents().isEmpty)
    }

    @Test func testHostsNeverAdmitIntoTheQueue() async throws {
        // The #360 class: a test host bootstraps the real container — its
        // perceptions must never land in the owner's queue.
        let (perception, store) = try makePerception(isTestHost: true)

        let nineAM = try #require(
            Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date()))
        perception.dayStartIfDue(now: nineAM, ownerPresent: true)
        perception.powerChanged(onACPower: true)

        try await Task.sleep(for: .milliseconds(50))
        #expect(try await store.pendingEvents().isEmpty)
    }

    @Test func sustainedAppSessionsProduceEventsBriefFlipsDoNot() async throws {
        let store = try scratchStore()
        let sensed = SensedObservationRecorder(store: store, isEnabled: { true })
        var sessions: [(app: String, minutes: Int)] = []
        sensed.onSustainedAppSession = { app, start, end in
            sessions.append((app, Int(end.timeIntervalSince(start) / 60)))
        }

        let t0 = Date()
        sensed.appBecameFrontmost("Xcode", at: t0)
        // A brief flip — 30 s in Safari never becomes an Event.
        sensed.appBecameFrontmost("Safari", at: t0.addingTimeInterval(120))
        sensed.appBecameFrontmost("Mail", at: t0.addingTimeInterval(150))

        #expect(sessions.map(\.app) == ["Xcode"])
        #expect(sessions.first?.minutes == 2)
    }
}
