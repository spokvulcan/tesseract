//
//  ThermalPolicy.swift
//  tesseract
//
//  The **Thermal Policy** (#515): what the voice does at the phone's
//  thermal state. Reading ahead in bursts or just in time doesn't change
//  what a second of audio costs, only when it is spent, so the policy changes
//  the voice, not the pacing. Starting rules, which the owner can loosen once
//  the hour-long runs show how warm the phone gets.
//

import Foundation

nonisolated enum ThermalPolicy {
    enum Decision: Equatable, Sendable {
        /// Nominal or fair: the neural voice reads.
        case neuralVoice
        /// Serious: the System Voice reads from the next segment, until the
        /// phone is back to fair.
        case systemVoice
        /// Critical: reading stops.
        case pause
    }

    static func decision(for state: ProcessInfo.ThermalState) -> Decision {
        switch state {
        case .nominal, .fair: .neuralVoice
        case .serious: .systemVoice
        case .critical: .pause
        @unknown default: .systemVoice
        }
    }

    /// What the Reader says about a decision; nil when there is nothing to say.
    static func notice(for decision: Decision) -> String? {
        switch decision {
        case .neuralVoice: nil
        case .systemVoice: "The phone is warm, so the system voice reads until it cools down."
        case .pause: "Reading stopped because the phone is too hot. Let it cool down first."
        }
    }
}
