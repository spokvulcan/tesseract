//
//  CompanionEventBatchTests.swift
//  tesseractTests
//
//  The `<events>` block render, including the due-wake de-dup (#404): a
//  `.wakeDue` Event whose wake the situation block already shows under
//  DUE NOW is suppressed — one due wake reaches the entity once, not in
//  two formats. Presentation only; the fold's consumption is untouched.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionEventBatchTests {

    private static let now = Date(timeIntervalSinceReferenceDate: 800_000_000)

    private func wake(_ content: String = "check on him") -> CompanionWake {
        CompanionWake(content: content, due: Self.now.addingTimeInterval(-60))
    }

    private func event(
        _ content: String,
        kind: CompanionEventKind = .powerChange,
        id: UUID = UUID()
    ) -> CompanionEvent {
        CompanionEvent(id: id, kind: kind, content: content, occurredAt: Self.now)
    }

    /// The base shape: numbered, kind-tagged, total order preserved.
    @Test func rendersNumberedLinesInOrder() {
        let text = CompanionEventBatch.render(
            [event("he plugged in"), event("a banner", kind: .notificationArrived)],
            now: Self.now)
        #expect(text.contains("<events>"))
        #expect(text.contains("1. [power-change]"))
        #expect(text.contains("2. [notification-arrived]"))
    }

    @Test func emptyBatchRendersNothing() {
        #expect(CompanionEventBatch.render([], now: Self.now).isEmpty)
    }

    /// The de-dup (#404): a `.wakeDue` Event whose wake renders in DUE NOW
    /// is suppressed; an unrelated event in the same batch is untouched.
    @Test func dueWakeEventIsSuppressedWhenItsWakeRendersInDueNow() {
        let due = wake()
        let wakeEvent = event(
            "[promise] check on him", kind: .wakeDue,
            id: CompanionEvent.wakeDueID(due.id))
        let text = CompanionEventBatch.render(
            [wakeEvent, event("he plugged in")], dueWakes: [due], now: Self.now)
        #expect(!text.contains("wake-due"))
        #expect(text.contains("1. [power-change]"))
    }

    /// A batch that is *only* the rendered wake's Event collapses to
    /// nothing — no empty `<events>` scaffold.
    @Test func fullySuppressedBatchRendersNothing() {
        let due = wake()
        let wakeEvent = event(
            "[promise] check on him", kind: .wakeDue,
            id: CompanionEvent.wakeDueID(due.id))
        #expect(
            CompanionEventBatch.render([wakeEvent], dueWakes: [due], now: Self.now)
                .isEmpty)
    }

    /// A `.wakeDue` Event whose wake is NOT in the due list (cancelled or
    /// completed between admission and this turn) still renders — nothing
    /// else tells the entity it happened.
    @Test func orphanedWakeDueEventStillRenders() {
        let wakeEvent = event(
            "[promise] check on him", kind: .wakeDue,
            id: CompanionEvent.wakeDueID(UUID()))
        let text = CompanionEventBatch.render(
            [wakeEvent], dueWakes: [wake()], now: Self.now)
        #expect(text.contains("1. [wake-due]"))
    }
}
