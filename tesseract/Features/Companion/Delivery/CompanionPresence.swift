//
//  CompanionPresence.swift
//  tesseract
//
//  Jarvis's ambient presence on the menu-bar glyph, the quietest rung of the
//  Delivery Ladder: whether he is thinking at a moment right now, whether
//  anything is waiting on the owner, and the time left of a step the owner
//  started. It says what is happening, never what he knows.
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

    /// What the menu bar counts down: the started step, what comes next.
    private(set) var clock: MenuBarClock?

    /// The menu-bar push: AppKit side, not an Observation consumer.
    @ObservationIgnored var onChange: ((State) -> Void)?
    @ObservationIgnored var onClockChange: ((MenuBarClock?) -> Void)?

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

    func setClock(_ clock: MenuBarClock?) {
        guard clock != self.clock else { return }
        self.clock = clock
        onClockChange?(clock)
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
