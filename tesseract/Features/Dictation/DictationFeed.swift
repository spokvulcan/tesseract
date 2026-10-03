//
//  DictationFeed.swift
//  tesseract
//

import Foundation
import Observation

/// The **Overlay Feed** (map #283): the one surface of dictation signals
/// the **Lens** renders from — typed phases, typed errors, outcome beats
/// carrying the committed text, the **Live Preview** while recording, the
/// take's app and whether ⇧ held it, and the audio meter (level +
/// spectrum).
///
/// One writer, many readers: `DictationCoordinator` drives `phase`, `beat`,
/// `preview`, `targetApp` and `isHeld`; the audio meter stream (attached once
/// at composition) drives `level` / `spectrum`. Views read whichever
/// properties they render, so a meter tick invalidates only meter-reading
/// subtrees.
@Observable
@MainActor
final class DictationFeed {

    /// The dictation lifecycle phase — the replacement for the retired
    /// `DictationState` (which carried a dead `.listening` case and a
    /// pre-flattened error string).
    enum Phase: Equatable, Sendable {
        case idle
        case recording
        case processing
        /// The **Proofread Pass** is polishing the transcription — a distinct
        /// phase so the Lens can narrate it (map #283).
        case proofreading
        case error(DictationError)

        var isActive: Bool {
            switch self {
            case .recording, .processing, .proofreading: return true
            case .idle, .error: return false
            }
        }
    }

    /// A terminal outcome of one dictation, delivered as a **beat**: a
    /// transient event distinct from `phase`, so the Lens can give the happy
    /// path an ending (and a future correction affordance a hook) even though
    /// the phase has already returned to `.idle`.
    enum Outcome: Equatable, Sendable {
        /// `edits` is the **Proofread Pass**'s word-swap diff — what a
        /// the Lens narrates (empty when the pass skipped or changed nothing).
        case committed(text: String, duration: TimeInterval, edits: [WordEdit])
        case empty
        /// The Proofread Pass rejected a wrong-words take. Passive: the
        /// press is the retry; `raw` feeds "insert raw anyway".
        case rejected(raw: String, reason: String)
        case cancelled
        case superseded
    }

    /// An `Outcome` stamped with a monotonically increasing id, so two equal
    /// outcomes in a row still read as two beats.
    struct Beat: Equatable, Sendable {
        let id: UInt64
        let outcome: Outcome
    }

    private(set) var phase: Phase = .idle
    /// Wall-clock start of the current `.recording` phase; `nil` outside it.
    /// Views derive elapsed-time displays from this.
    private(set) var recordingStarted: Date?
    private(set) var beat: Beat?

    /// The **Live Preview** (PRD #612): what Whisper hears while recording,
    /// confirmed words and a provisional tail, with the Learned Words
    /// applied. Replaced wholesale on each decode, scoped to `.recording`,
    /// and `nil` until the first words land.
    private(set) var preview: LivePreview?

    /// The app the take is going to (in front when it started).
    private(set) var targetApp: TargetApp?

    /// ⇧ was tapped this take: it waits in the Lens instead of pasting.
    private(set) var isHeld = false

    /// Overall loudness, 0–1 (normalized from dBFS in the audio tap).
    private(set) var level: Float = 0
    /// Log-spaced frequency bands, each 0–1; `MeterFrame.bandCount` entries.
    /// Zeroed whenever capture is not delivering frames.
    private(set) var spectrum: [Float] = MeterFrame.zeroBands

    private var nextBeatID: UInt64 = 0
    private var meterPump: Task<Void, Never>?

    // MARK: - Driver side (coordinator + composition root; never views)

    func setPhase(_ newPhase: Phase) {
        if case .recording = newPhase {
            if recordingStarted == nil {
                recordingStarted = Date()
                isHeld = false
            }
        } else {
            recordingStarted = nil
            // Recording-scoped: a preview never outlives its take (the pump
            // clears too, but the phase flip is the authoritative edge).
            preview = nil
        }
        phase = newPhase
    }

    /// Publishes a Live Preview (or clears with `nil`). Writes are dropped
    /// outside `.recording` so a decode that resolves after the key release
    /// cannot resurrect a preview for a finished take.
    func setPreview(_ preview: LivePreview?) {
        guard preview == nil || phase == .recording else { return }
        if self.preview != preview { self.preview = preview }
    }

    func setTargetApp(_ app: TargetApp?) {
        if targetApp != app { targetApp = app }
    }

    /// ⇧ was tapped (again): the take waits in the Lens, or no longer does.
    func setHeld(_ held: Bool) {
        if isHeld != held { isHeld = held }
    }

    func emit(_ outcome: Outcome) {
        nextBeatID &+= 1
        beat = Beat(id: nextBeatID, outcome: outcome)
    }

    /// Attaches the engine's meter stream; frames land on the main actor at
    /// the tap's cadence (~47 Hz). Called once at composition. The pump holds
    /// the feed weakly and exits on the first frame after it deallocates —
    /// no deinit cancellation needed (nor possible: deinit is nonisolated).
    func attachMeters(_ frames: AsyncStream<MeterFrame>) {
        meterPump?.cancel()
        meterPump = Task { [weak self] in
            for await frame in frames {
                guard let self else { return }
                self.apply(frame)
            }
        }
    }

    /// Applies one meter frame (also the test seam — tests drive this
    /// directly instead of running an audio engine).
    func apply(_ frame: MeterFrame) {
        let clamped = min(max(frame.level, 0), 1)
        if abs(clamped - level) > 0.001 || frame.bands != spectrum {
            level = clamped
            spectrum = frame.bands
        }
    }
}
