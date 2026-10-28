//
//  VoiceEndpointer.swift
//  tesseract
//
//  The voice session's ears (#310 §3/§6): a pure energy-based endpointer
//  over the meter level — a speech onset opens the owner's turn, trailing
//  silence closes it. It only ever listens to the owner: the session is
//  half-duplex (ADR-0082), so the mic is closed while Jarvis speaks and the
//  endpointer never hears him.
//
//  Deliberately not ASR-based: a threshold plus debounce is robust, instant,
//  and testable — WhisperKit's VAD is batch-only (#310 §6 engine ask).
//
//  The silence that closes a turn grows with the turn (ADR-0097): a question
//  said in a breath closes after a second, a turn that keeps going gets the
//  Setting's patience. Measured on the owner's last 99 takes, a fixed 1.0 s
//  cut 31% of them before the owner finished and 1.8 s cut 4%; this rule
//  cuts 5%, and none of the 25 with 5 s of speech or less.
//

import Foundation

nonisolated struct VoiceEndpointer {

    struct Config {
        /// Meter level (0–1, −60 dB-floor normalized) that counts as speech.
        var speechLevel: Float
        /// Sustained speech required before `speechStarted` — rejects coughs,
        /// keyboard thumps, chair creaks.
        var startDebounce: TimeInterval
        /// Trailing silence that closes a long turn (#310 §3, tunable-taste:
        /// the Setting).
        var trailingSilence: TimeInterval
        /// Trailing silence that closes a short turn.
        var shortTurnSilence: TimeInterval
        /// Speech since the onset over which the close grows from
        /// `shortTurnSilence` to `trailingSilence`.
        var patienceRamp: ClosedRange<TimeInterval>

        /// The close after a short turn, unless the Setting is shorter.
        static let shortTurn: TimeInterval = 1.0

        /// Listening for the owner's turn: permissive start; a quick close
        /// after a short turn, a patient one after a long turn.
        static func listening(
            speechLevel: Float = 0.22, trailingSilence: TimeInterval = 1.8
        ) -> Config {
            Config(
                speechLevel: speechLevel, startDebounce: 0.25,
                trailingSilence: trailingSilence,
                shortTurnSilence: min(shortTurn, trailingSilence),
                patienceRamp: 1.5...3.5)
        }

        /// The silence that closes a turn whose speech has run `spoken`
        /// seconds since its onset.
        func closingSilence(afterSpeaking spoken: TimeInterval) -> TimeInterval {
            let span = patienceRamp.upperBound - patienceRamp.lowerBound
            let progress = span > 0 ? (spoken - patienceRamp.lowerBound) / span : 1
            let grown = min(1, max(0, progress))
            return shortTurnSilence + (trailingSilence - shortTurnSilence) * grown
        }
    }

    enum Event: Equatable {
        case speechStarted
        case endOfSpeech
    }

    private(set) var config: Config
    private(set) var isInSpeech = false

    /// The start of the current loud run; once speech starts, the turn's
    /// onset (it holds until the turn ends).
    private var candidateSince: TimeInterval?
    private var lastLoudAt: TimeInterval?

    init(config: Config) {
        self.config = config
    }

    /// Start a fresh utterance watch (new capture).
    mutating func reset(config: Config? = nil) {
        if let config { self.config = config }
        isInSpeech = false
        candidateSince = nil
        lastLoudAt = nil
    }

    /// Feed one level sample; returns an event on a state edge.
    mutating func ingest(level: Float, at time: TimeInterval) -> Event? {
        let loud = level >= config.speechLevel
        if loud {
            lastLoudAt = time
            if !isInSpeech {
                if let since = candidateSince {
                    if time - since >= config.startDebounce {
                        isInSpeech = true
                        return .speechStarted
                    }
                } else {
                    candidateSince = time
                }
            }
        } else {
            if !isInSpeech { candidateSince = nil }
            if isInSpeech, let last = lastLoudAt,
                time - last >= config.closingSilence(afterSpeaking: last - (candidateSince ?? last))
            {
                isInSpeech = false
                candidateSince = nil
                return .endOfSpeech
            }
        }
        return nil
    }
}
