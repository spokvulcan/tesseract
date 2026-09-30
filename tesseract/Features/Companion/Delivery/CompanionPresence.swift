//
//  CompanionPresence.swift
//  tesseract
//
//  Jarvis's ambient presence on the menu-bar glyph, the quietest rung of the
//  Delivery Ladder: whether he is thinking at a moment right now, and whether
//  anything is waiting on the owner. It says what is happening, never what
//  he knows.
//

import Foundation
import Observation

@Observable @MainActor
final class CompanionPresence {

    enum State: String, Sendable {
        /// Nothing in flight and nothing waiting: the glyph rests.
        case idle
        /// A moment's model call is running.
        case thinking
        /// Something is waiting on the owner (a card item, an agent, a person).
        case waiting
    }

    private(set) var state: State = .idle
    /// How many items are waiting on the owner right now.
    private(set) var waitingCount = 0

    /// The menu-bar push: AppKit side, not an Observation consumer.
    @ObservationIgnored var onChange: ((State) -> Void)?

    /// Overlapping moments are depth-counted, so one ending does not clear
    /// another that is still running.
    private var thinkingDepth = 0

    func beginThinking() {
        thinkingDepth += 1
        recompute()
    }

    func endThinking() {
        thinkingDepth = max(0, thinkingDepth - 1)
        recompute()
    }

    func setWaiting(count: Int) {
        waitingCount = max(0, count)
        recompute()
    }

    private func recompute() {
        let new: State =
            if thinkingDepth > 0 {
                .thinking
            } else if waitingCount > 0 {
                .waiting
            } else {
                .idle
            }
        guard new != state else { return }
        state = new
        onChange?(new)
    }
}
