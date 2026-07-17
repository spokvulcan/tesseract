//
//  EchoResidualFloor.swift
//  tesseract
//
//  The **Echo Floor** (ADR-0041): a self-calibrating tracker of how loud the
//  reply's own residual reads at the open mic, so barge-in detection during
//  playback compares the owner against *measured* self-echo instead of a
//  static guess. The static threshold alone cannot survive an undipped reply:
//  every energy barge on 2026-07-17 but one was the agent hearing itself
//  (flight-recorder evidence, 9 pause/resume flaps inside one reply).
//
//  Shape: while the reply is audibly emitting, the floor chases the observed
//  mic level fast — but never above `playbackLevel − echoPathLoss`. That cap
//  is the load-bearing discrimination: residual physically cannot read
//  louder than the reply minus the echo path's loss (measured ≥ ~0.24
//  normalized on this hardware, voice-hold-lab E2), while the owner talking
//  over the reply can and does. An uncapped chase would drag the floor — and
//  with it the barge threshold — up under the owner's own onset and suppress
//  every real interruption; a capped one converges on sustained residual
//  within the endpointer's debounce yet leaves owner speech towering above
//  the threshold.
//
//  Pure and main-actor-driven: fed from the session ticker at 20 Hz with the
//  meter level (already pumped off the RT thread) and the playback sink's
//  coarse envelope. Never touches the real-time audio thread.
//

import Foundation

nonisolated struct EchoResidualFloor {

    struct Config {
        /// Maximum floor rise per second (normalized meter units/s) while
        /// the reply is audibly emitting. Fast — the floor must converge on
        /// sustained residual inside the endpointer's 0.45 s debounce; the
        /// path-loss cap (not this rate) is what protects owner onsets.
        var attackPerSecond: Float
        /// Floor fall per second (normalized units/s) while the reply is
        /// loud but the mic reads below the floor. Slow — residual varies
        /// word to word and the floor must not chase it downward.
        var decayPerSecond: Float
        /// Floor fall per second once playback has been quiet past the
        /// trailing hold. Fast — with no far-end there is no residual, and
        /// an owner speaking into the reply's pause must not be gated by a
        /// stale floor.
        var quietDecayPerSecond: Float
        /// Required excursion above the floor to count as the owner
        /// (normalized units; 0.0167 ≈ 1 dB on the −60 dB meter scale).
        var margin: Float
        /// The floor's ceiling below the playback envelope: residual can't
        /// read louder than `playbackLevel − echoPathLoss`, owner speech
        /// can. Calibrated from the lab's E2 worst case (smallest observed
        /// playback−mic gap) minus headroom.
        var echoPathLoss: Float
        /// Playback level at or above this counts as "audibly emitting", in
        /// the sink's dB-normalized envelope domain.
        var playbackLoudLevel: Float
        /// Trailing hold on "audibly emitting" — covers output-device
        /// latency, room tail, and the meter pump's skew (~50–200 ms), so
        /// the floor never attributes late-arriving residual to the room.
        var playbackLoudHold: TimeInterval

        /// Calibrated against voice-hold-lab fixtures (VoiceBargeReplayTests
        /// pins: zero onsets on clean-reply traces, fire ≤ 600 ms on owner
        /// speech). Re-run the lab's emit-fixture after changing any value.
        static func standard() -> Config {
            Config(
                attackPerSecond: 2.0,
                decayPerSecond: 0.15,
                quietDecayPerSecond: 0.6,
                margin: 0.08,
                echoPathLoss: 0.2,
                playbackLoudLevel: 0.17,
                playbackLoudHold: 0.3)
        }
    }

    private(set) var config: Config
    /// The tracked residual level; 0 until playback has been heard.
    private(set) var floor: Float = 0

    private var gapCredit = TickerGapCredit()
    private var lastLoudPlaybackAt: TimeInterval?

    init(config: Config = .standard()) {
        self.config = config
    }

    /// Start fresh for a new utterance (new reply, new session).
    mutating func reset() {
        floor = 0
        gapCredit.reset()
        lastLoudPlaybackAt = nil
    }

    /// Feed one 20 Hz sample: the mic meter level and the playback sink's
    /// envelope level at the playback head.
    mutating func ingest(micLevel: Float, playbackLevel: Float, at time: TimeInterval) {
        let gap = gapCredit.credit(at: time)

        if playbackLevel >= config.playbackLoudLevel {
            lastLoudPlaybackAt = time
        }
        let playbackLoud =
            lastLoudPlaybackAt.map {
                time - $0 <= config.playbackLoudHold
            } ?? false

        let dt = Float(gap)
        if playbackLoud {
            // The residual the floor may believe: never more than the
            // playback envelope minus the calibrated echo-path loss —
            // anything above that is the owner, and the floor must not
            // chase it.
            let believable = min(micLevel, max(0, playbackLevel - config.echoPathLoss))
            if believable > floor {
                floor = min(believable, floor + config.attackPerSecond * dt)
            } else {
                floor = max(0, floor - config.decayPerSecond * dt)
            }
        } else {
            floor = max(0, floor - config.quietDecayPerSecond * dt)
        }
    }

    /// The effective barge threshold: the static level until the floor has
    /// evidence, `floor + margin` (margin scaled by escalation) once it does
    /// — never below the static level.
    func threshold(atLeast staticLevel: Float, marginScale: Float = 1) -> Float {
        guard floor > 0 else { return staticLevel }
        return max(staticLevel, floor + config.margin * marginScale)
    }
}
