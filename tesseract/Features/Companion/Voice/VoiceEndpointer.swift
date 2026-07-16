//
//  VoiceEndpointer.swift
//  tesseract
//
//  The voice session's ears (#310 §3/§4/§6): a pure energy-based endpointer
//  over the meter level. Two jobs, one machine — end-of-speech detection
//  while the owner talks (trailing silence closes his turn), and barge-in
//  detection while Jarvis talks (sustained speech energy over the
//  echo-cancelled input stops the utterance; VPIO's AEC is what makes the
//  same gate valid during playback, ADR-0025).
//
//  Deliberately not ASR-based: a threshold plus debounce is robust, instant,
//  and testable — WhisperKit's VAD is batch-only (#310 §6 engine ask).
//

import Foundation

nonisolated struct VoiceEndpointer {

    struct Config {
        /// Meter level (0–1, −60 dB-floor normalized) that counts as speech.
        var speechLevel: Float
        /// Sustained speech required before `speechStarted` — rejects coughs,
        /// keyboard thumps, chair creaks.
        var startDebounce: TimeInterval
        /// Trailing silence that closes the turn (#310 §3, tunable-taste).
        var trailingSilence: TimeInterval

        /// Listening for the owner's turn: permissive start, patient close.
        static func listening(
            speechLevel: Float = 0.22, trailingSilence: TimeInterval = 1.8
        ) -> Config {
            Config(
                speechLevel: speechLevel, startDebounce: 0.25,
                trailingSilence: trailingSilence)
        }

        /// Watching for barge-in during playback: a longer debounce — an
        /// interruption is deliberate, residual echo blips are not.
        static func bargeIn(speechLevel: Float = 0.25) -> Config {
            Config(speechLevel: speechLevel, startDebounce: 0.45, trailingSilence: 1.8)
        }
    }

    enum Event: Equatable {
        case speechStarted
        case endOfSpeech
    }

    private(set) var config: Config
    private(set) var isInSpeech = false

    private var candidateSince: TimeInterval?
    private var lastLoudAt: TimeInterval?

    init(config: Config) {
        self.config = config
    }

    /// Start a fresh utterance watch (new capture, new mode).
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
            if isInSpeech, let last = lastLoudAt, time - last >= config.trailingSilence {
                isInSpeech = false
                candidateSince = nil
                return .endOfSpeech
            }
        }
        return nil
    }
}
