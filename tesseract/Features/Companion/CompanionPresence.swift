//
//  CompanionPresence.swift
//  tesseract
//
//  Jarvis's ambient presence (#327 §3): one small observable state the quiet
//  surfaces render — the menu-bar glyph and the composer's notice slot.
//  Not a dashboard: it says what he is *doing right now*, never what he knows.
//  Every transition is a recorded fact (#326).
//

import AppKit
import Foundation
import Observation

@Observable @MainActor
final class CompanionPresence {

    /// The glyph vocabulary (#327 §3): idle / thinking / summoning / asleep.
    /// The fifth spec'd state, `speaking`, is carried by the app-wide speech
    /// activity rung (`MenuBarManager.updateState(fromSpeech:)`, the
    /// composer's Speaking notice) — TTS is TTS whoever asked for it.
    enum State: String, Sendable {
        /// Nothing in flight — the glyph rests.
        case idle
        /// A companion turn is running (wake, catch-up, ambient).
        case thinking
        /// A summons awaits his answer — on screen, or raised on the glyph
        /// itself by the entity's `set_glyph` rung.
        case summoning
        /// A sleep pass is consolidating the day (ADR-0035 §7).
        case asleep
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
    private var isSleeping = false
    /// The entity's own hand on the glyph (ADR-0040 §10's quietest rung):
    /// a raised notice renders as summoning until he looks or it is cleared.
    private var entityNoticeRaised = false

    init(recorder: CompanionFlightRecorder) {
        self.recorder = recorder
        // Him bringing the app forward is the glyph notice answered — the
        // quietest rung's engage, observed by the app, never self-reported.
        NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.clearEntityNotice(reason: "app-active") }
        }
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

    /// The sleep pass's presence (#327 §3's `asleep`) — pushed by the one
    /// binding that watches `MemorySleep`.
    func setAsleep(_ sleeping: Bool) {
        guard sleeping != isSleeping else { return }
        isSleeping = sleeping
        recompute()
    }

    /// The `set_glyph` rung raising its notice.
    func raiseEntityNotice() {
        entityNoticeRaised = true
        recompute()
    }

    /// Cleared by the entity's own tool call, or by him bringing the app
    /// forward — either way a recorded transition.
    func clearEntityNotice(reason: String = "tool") {
        guard entityNoticeRaised else { return }
        entityNoticeRaised = false
        recorder.record("glyph.notice-cleared", snapshot: ["reason": reason])
        recompute()
    }

    private func recompute() {
        let new: State =
            if summonsDepth > 0 || entityNoticeRaised {
                .summoning
            } else if thinkingDepth > 0 {
                .thinking
            } else if isSleeping {
                .asleep
            } else {
                .idle
            }
        guard new != state else { return }
        state = new
        recorder.record("glyph.changed", snapshot: ["state": new.rawValue])
        onChange?(new)
    }
}
