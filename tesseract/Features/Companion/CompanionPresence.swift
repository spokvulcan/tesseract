//
//  CompanionPresence.swift
//  tesseract
//
//  Jarvis's ambient presence (#327 §3): one small observable state the quiet
//  surfaces render — the menu-bar glyph and the composer's notice slot.
//  Not a dashboard: it says what he is *doing right now*, never what he knows.
//  Every transition is a recorded fact (#326).
//

import Foundation
import Observation

@Observable @MainActor
final class CompanionPresence {

    enum State: String, Sendable {
        /// Nothing in flight — the glyph rests.
        case idle
        /// A companion turn is running (wake, catch-up, ambient).
        case thinking
        /// A summons is on screen awaiting his answer.
        case summoning
    }

    private(set) var state: State = .idle

    /// The menu-bar push — AppKit side, not an Observation consumer.
    @ObservationIgnored var onChange: ((State) -> Void)?

    private let recorder: CompanionFlightRecorder
    /// Overlapping sources (a summons outliving the turn that posted it) are
    /// depth-counted, not last-writer-wins — a turn ending must not clear a
    /// summons still on screen.
    private var thinkingDepth = 0
    private var summonsDepth = 0

    init(recorder: CompanionFlightRecorder) {
        self.recorder = recorder
    }

    func beginThinking() {
        thinkingDepth += 1
        recompute()
    }

    func endThinking() {
        thinkingDepth = max(0, thinkingDepth - 1)
        recompute()
    }

    func beginSummons() {
        summonsDepth += 1
        recompute()
    }

    func endSummons() {
        summonsDepth = max(0, summonsDepth - 1)
        recompute()
    }

    private func recompute() {
        let new: State =
            summonsDepth > 0 ? .summoning : (thinkingDepth > 0 ? .thinking : .idle)
        guard new != state else { return }
        state = new
        recorder.record("glyph.changed", snapshot: ["state": new.rawValue])
        onChange?(new)
    }
}
