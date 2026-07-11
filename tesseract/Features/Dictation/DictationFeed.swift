//
//  DictationFeed.swift
//  tesseract
//

import Foundation
import Observation

/// The **Overlay Feed** (map #283): the one variant-agnostic surface of
/// dictation signals every **Overlay Variant** renders from — typed phases,
/// typed errors, outcome beats carrying the committed text, and the audio
/// meter (level + spectrum). Variants consume the feed and nothing else;
/// the pipeline never learns which variant is live.
///
/// One writer, many readers: `DictationCoordinator` drives `phase` and
/// `beat`; the audio meter stream (attached once at composition) drives
/// `level` / `spectrum`. Views read whichever properties they render, so a
/// meter tick invalidates only meter-reading subtrees.
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
        /// phase so variants can narrate it (map #283).
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
    /// transient event distinct from `phase`, so a variant can give the happy
    /// path an ending (and a future correction affordance a hook) even though
    /// the phase has already returned to `.idle`.
    enum Outcome: Equatable, Sendable {
        /// `edits` is the **Proofread Pass**'s word-swap diff — what a
        /// variant narrates (empty when the pass skipped or changed nothing).
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

    /// How long a terminal beat's affordances linger (ticket #289) — shared
    /// by the variants' lingering pills and the App Bindings rule that keeps
    /// the panel clickable for exactly this window, so the two can't drift.
    static let affordanceLinger: Duration = .seconds(2.5)

    private(set) var phase: Phase = .idle
    /// Wall-clock start of the current `.recording` phase; `nil` outside it.
    /// Variants derive elapsed-time displays from this.
    private(set) var recordingStarted: Date?
    private(set) var beat: Beat?

    /// The **Live Partial** signal (ticket #291): a revising snapshot of what
    /// ASR hears while recording. Replaced wholesale on each revision (partials
    /// rewrite themselves — never append), scoped to `.recording`, and `nil`
    /// whenever streaming is unavailable (partials off, model busy, no words
    /// yet) — variants that ignore it keep working unchanged.
    private(set) var partial: String?

    /// Overall loudness, 0–1 (normalized from dBFS in the audio tap).
    private(set) var level: Float = 0
    /// Log-spaced frequency bands, each 0–1; `MeterFrame.bandCount` entries.
    /// Zeroed whenever capture is not delivering frames.
    private(set) var spectrum: [Float] = MeterFrame.zeroBands

    private var nextBeatID: UInt64 = 0
    private var meterPump: Task<Void, Never>?

    // MARK: - Driver side (coordinator + composition root; never variants)

    func setPhase(_ newPhase: Phase) {
        if case .recording = newPhase {
            if recordingStarted == nil { recordingStarted = Date() }
        } else {
            recordingStarted = nil
            // Recording-scoped: a partial never outlives its take (the pump
            // clears too, but the phase flip is the authoritative edge).
            partial = nil
        }
        phase = newPhase
    }

    /// Publishes a Live Partial revision (or clears with `nil`). Writes are
    /// dropped outside `.recording` so a decode that resolves after the key
    /// release cannot resurrect a caption for a finished take.
    func setPartial(_ text: String?) {
        guard text == nil || phase == .recording else { return }
        if partial != text { partial = text }
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
