//
//  WaitingOnYou.swift
//  tesseract
//
//  Waiting on You, said once: the coding agents waiting, what the day's
//  open cards raised — people and missed notifications on a Breakpoint or a
//  Triage card (a waiting agent once, from the agents themselves) — and the
//  evening's leftovers. Today's Now Card shows all of it; the glyph counts
//  what still reaches out, so a card the owner took in on the panel (Looks
//  Good, Good Night, closed) waits in Today without lighting it.
//

import Foundation

nonisolated struct WaitingOnYou: Sendable, Equatable {

    /// An item a card raised, and the card it is on.
    struct Entry: Sendable, Equatable, Identifiable {
        let cardID: String
        let item: WaitingItem
        var id: String { "\(cardID)/\(item.id)" }
    }

    /// An Evening Wrap-up with leftovers still to settle.
    struct WrapUp: Sendable, Equatable {
        let cardID: String
        let card: EveningWrapUpCard
    }

    let agents: [AgentSignal]
    let entries: [Entry]
    let wrapUps: [WrapUp]
    /// How many things the glyph says are waiting.
    let glyphCount: Int

    init(state: DayState, now: Date) {
        agents = state.agentsWaiting(now: now)
        var entries: [Entry] = []
        var wrapUps: [WrapUp] = []
        var onCards = 0
        for card in state.openCards {
            let waiting: Int
            switch card.body {
            case .breakpoint(let breakpoint):
                let items = breakpoint.needsYou.filter { $0.kind != .agent }
                entries += items.map { Entry(cardID: card.id, item: $0) }
                waiting = items.count
            case .triage(let triage):
                entries += triage.raise.map { Entry(cardID: card.id, item: $0) }
                waiting = triage.raise.count
            case .eveningWrapUp(let wrapUp):
                if !wrapUp.leftovers.isEmpty {
                    wrapUps.append(WrapUp(cardID: card.id, card: wrapUp))
                }
                waiting = wrapUp.leftovers.count
            case .morningPlan, .reflection:
                waiting = 0
            }
            // A card the owner took in waits on them no more.
            if !card.kept { onCards += waiting }
        }
        self.entries = entries
        self.wrapUps = wrapUps
        glyphCount = onCards + agents.count
    }

    var isEmpty: Bool { agents.isEmpty && entries.isEmpty && wrapUps.isEmpty }
}
