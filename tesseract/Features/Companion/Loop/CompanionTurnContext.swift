//
//  CompanionTurnContext.swift
//  tesseract
//
//  The correlation state of the companion turn in flight — delivery tools
//  and `book_wake` read it so every recorder event and every booking carries
//  the turn and conversation that produced it. One turn at a time by design
//  (the loop serializes); this is a box, not a queue.
//

import Foundation

@MainActor
final class CompanionTurnContext {

    private(set) var turnID: UUID?
    private(set) var wakeIDs: [UUID] = []
    private(set) var conversationID: UUID?
    private(set) var origin: String?

    var isActive: Bool { turnID != nil }

    func begin(turnID: UUID, wakeIDs: [UUID], conversationID: UUID, origin: String) {
        self.turnID = turnID
        self.wakeIDs = wakeIDs
        self.conversationID = conversationID
        self.origin = origin
    }

    func end() {
        turnID = nil
        wakeIDs = []
        conversationID = nil
        origin = nil
    }
}
