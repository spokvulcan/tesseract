//
//  SpeedCheck.swift
//  tesseract
//
//  The **Speed Check** (#515): whether this phone's neural voice keeps up,
//  from a render timed at the end of Voice Preparation.
//

import Foundation

/// The playback rates the Reader's speed menu offers.
nonisolated enum PlaybackRate {
    static let menu: [Double] = [0.75, 0.9, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0]
}

/// A timed render of the neural voice. Its real-time factor is compute
/// seconds per second of audio. Played at a rate r, a second of audio lasts
/// 1/r seconds, so the voice keeps up at r when the factor times r stays
/// under 1, here with `headroom` to spare for a locked screen and a warm
/// phone.
nonisolated struct SpeedCheck: Codable, Equatable, Sendable {
    let realTimeFactor: Double
    /// Seconds from asking to hearing.
    let firstAudio: Double

    /// How much faster than playback the voice must be at a rate to keep it.
    static let headroom = 1.2

    func keepsUp(at rate: Double) -> Bool {
        realTimeFactor * rate * Self.headroom <= 1
    }

    /// Whether the neural voice reads at all: it keeps up at 1×. A phone that
    /// doesn't reads with the System Voice.
    var keepsUpAtNormalSpeed: Bool { keepsUp(at: 1) }

    /// The menu's rates the neural voice can read at: every slower one, and
    /// the faster ones it keeps up with.
    func playableRates(of rates: [Double] = PlaybackRate.menu) -> [Double] {
        rates.filter { $0 <= 1 || keepsUp(at: $0) }
    }

    /// The fastest rate it keeps up with, of the menu's.
    func fastestRate(of rates: [Double] = PlaybackRate.menu) -> Double? {
        keepsUpAtNormalSpeed ? playableRates(of: rates).max() : nil
    }

    /// What the owner is told when the voice can't keep up.
    var verdict: String? {
        keepsUpAtNormalSpeed
            ? nil
            : String(
                format: "This iPhone makes the neural voice at %.1f× the speed it plays, too slow "
                    + "to keep up, so the system voice reads.", 1 / realTimeFactor)
    }
}
