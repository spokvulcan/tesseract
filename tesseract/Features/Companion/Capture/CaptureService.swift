//
//  CaptureService.swift
//  tesseract
//
//  The one capture door: typed or spoken words in, a real reminder out, with
//  a one-line confirmation and an undo. The capture hotkey's panel and the
//  Today page's Capture box both come through here.
//

import Foundation
import Observation

nonisolated enum CaptureOutcome: Equatable, Sendable {
    /// A reminder was added; the change carries the line and the undo.
    case added(AgendaChange)
    /// Nothing to capture (only filler words).
    case empty
    /// It didn't save; the reason is for the owner.
    case failed(String)

    var line: String {
        switch self {
        case .added(let change): change.line
        case .empty: "Nothing to capture."
        case .failed(let reason): reason
        }
    }
}

@Observable @MainActor
final class CaptureService {

    @ObservationIgnored private let agenda: Agenda
    @ObservationIgnored private let now: @MainActor () -> Date

    /// The latest outcome, for the confirmation line.
    private(set) var lastOutcome: CaptureOutcome?

    init(agenda: Agenda, now: @escaping @MainActor () -> Date = Date.init) {
        self.agenda = agenda
        self.now = now
    }

    @discardableResult
    func capture(_ text: String, source: String) async -> CaptureOutcome {
        let outcome = await add(text, source: source)
        lastOutcome = outcome
        return outcome
    }

    /// Undo the capture the owner is looking at — that one, not whatever was
    /// captured last (the panels and Today share this door). True once it is
    /// undone; otherwise the reason is the latest outcome.
    @discardableResult
    func undo(_ change: AgendaChange) async -> Bool {
        do {
            try await agenda.undo(change)
            if case .added(let last) = lastOutcome, last.id == change.id { lastOutcome = nil }
            return true
        } catch {
            lastOutcome = .failed("Couldn't undo: \(error.localizedDescription)")
            return false
        }
    }

    func clear() { lastOutcome = nil }

    private func add(_ text: String, source: String) async -> CaptureOutcome {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .empty }
        let access =
            agenda.access.needsRequest ? await agenda.requestAccessIfNeeded() : agenda.access
        guard access.canUseReminders else {
            return .failed(
                "Tesseract can't reach Reminders. Allow it in System Settings → Privacy & Security → Reminders."
            )
        }
        let current = now()
        let events = agenda.events(
            from: current.addingTimeInterval(-12 * 3600), to: current.addingTimeInterval(7 * 86_400)
        )
        guard
            let intent = CaptureParser.parse(
                text, now: current, events: events, areas: agenda.areas)
        else { return .empty }
        do {
            let (_, change) = try agenda.addReminder(
                title: intent.title, areaName: intent.area?.name, due: intent.due,
                dueHasTime: intent.dueHasTime, source: source)
            return .added(change)
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}
