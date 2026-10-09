//
//  CaptureServiceTests.swift
//  tesseractTests
//
//  The capture door's undo: the capture panel, the Jarvis panel and Today
//  share it, so an undo takes back the capture the owner is looking at, not
//  whatever came through last — and says so when it couldn't.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CaptureServiceTests {

    private func makeService() async -> (CaptureService, InMemoryAgendaStore) {
        let store = InMemoryAgendaStore()
        let agenda = Agenda(store: store)
        await agenda.refresh()
        return (CaptureService(agenda: agenda), store)
    }

    @Test func undoTakesBackTheCaptureShownNotTheLatest() async throws {
        let (service, store) = await makeService()
        guard case .added(let first) = await service.capture("buy milk", source: "hotkey"),
            case .added = await service.capture("call the bank", source: "today")
        else {
            Issue.record("expected both captures to save")
            return
        }
        #expect(store.reminders.count == 2)

        #expect(await service.undo(first))

        #expect(store.reminders.values.map(\.title) == ["Call the bank"])
        // The latest outcome is the other capture's, still undoable.
        guard case .added = service.lastOutcome else {
            Issue.record("expected the latest capture's outcome to stay")
            return
        }
    }

    @Test func anUndoThatFailsSaysSo() async throws {
        let (service, store) = await makeService()
        guard case .added(let change) = await service.capture("buy milk", source: "hotkey")
        else {
            Issue.record("expected the capture to save")
            return
        }
        // Already gone: deleted in Reminders meanwhile.
        let id = try #require(store.reminders.keys.first)
        try store.deleteReminder(id: id)

        #expect(await service.undo(change) == false)
        guard case .failed(let line) = service.lastOutcome else {
            Issue.record("expected the failure as the latest outcome")
            return
        }
        #expect(line.hasPrefix("Couldn't undo"))
    }
}
