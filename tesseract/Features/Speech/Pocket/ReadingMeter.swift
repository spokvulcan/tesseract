//
//  ReadingMeter.swift
//  tesseract
//
//  What a TestFlight report needs from the last reading (#515): how fast the
//  voice made its audio (the real-time factor: seconds of work per second of
//  audio, below 1 keeps up), how warm the phone got, and which voice read.
//  It listens on the speech engine's diagnostics tap, which marks where each
//  segment's synthesis begins and ends and how many frames it made.
//

import Foundation
import Synchronization
import TesseractSpeech

nonisolated final class ReadingMeter: SpeechDiagnosticsTap, Sendable {
    /// One synthesized segment.
    struct Segment: Equatable, Sendable {
        let seconds: TimeInterval
        let audioSeconds: TimeInterval
        let thermalState: ProcessInfo.ThermalState

        var realTimeFactor: Double { audioSeconds > 0 ? seconds / audioSeconds : 0 }
    }

    private struct State {
        var segments: [Segment] = []
        var started: [Int: ContinuousClock.Instant] = [:]
        var lastIndex = -1
    }

    private let state = Mutex(State())
    private let framesPerSecond: Double
    private let now: @Sendable () -> ContinuousClock.Instant
    private let thermalState: @Sendable () -> ProcessInfo.ThermalState

    init(
        framesPerSecond: Double = 12.5,
        now: @escaping @Sendable () -> ContinuousClock.Instant = { .now },
        thermalState: @escaping @Sendable () -> ProcessInfo.ThermalState = {
            ProcessInfo.processInfo.thermalState
        }
    ) {
        self.framesPerSecond = framesPerSecond
        self.now = now
        self.thermalState = thermalState
    }

    /// The last reading's segments, in order. A reading starts over at its
    /// first segment.
    var segments: [Segment] { state.withLock { $0.segments } }

    func event(_ name: StaticString, _ detail: @autoclosure @Sendable () -> String) {
        let numbers = detail().split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        guard let index = numbers.first else { return }
        switch "\(name)" {
        case "burst.begin":
            let time = now()
            state.withLock { state in
                // Segment 0 again: a new utterance.
                if index <= state.lastIndex || index == 0 { state.segments = [] }
                state.lastIndex = index
                state.started[index] = time
            }
        case "burst.end":
            guard numbers.count > 1 else { return }
            let end = now()
            let warmth = thermalState()
            let audioSeconds = Double(numbers[1]) / framesPerSecond
            state.withLock { state in
                guard let start = state.started.removeValue(forKey: index) else { return }
                let elapsed = start.duration(to: end)
                let seconds =
                    Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds)
                    / 1e18
                state.segments.append(
                    Segment(seconds: seconds, audioSeconds: audioSeconds, thermalState: warmth))
            }
        default:
            break
        }
    }

    // MARK: - The report

    /// The diagnostics a tester copies into a report.
    static func report(
        app: String, device: String, system: String, voice: String, segments: [Segment]
    ) -> String {
        var lines = [app, "\(device) · \(system)", "Voice: \(voice)"]
        if segments.isEmpty {
            lines.append("Last reading: none yet")
        } else {
            let factors = segments.map(\.realTimeFactor).sorted()
            let median = factors[factors.count / 2]
            let audio = segments.map(\.audioSeconds).reduce(0, +)
            lines.append(
                "Last reading: \(segments.count) segments, \(Int(audio.rounded())) s of audio, "
                    + "real-time factor \(String(format: "%.2f", median)) median, "
                    + "\(String(format: "%.2f", factors.last ?? 0)) worst")
            let warmest = segments.map(\.thermalState).max { $0.rawValue < $1.rawValue }
            lines.append(
                "Thermal state: \(name(of: segments[0].thermalState)) at the start, "
                    + "\(name(of: warmest ?? .nominal)) at the warmest")
        }
        return lines.joined(separator: "\n")
    }

    static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}
