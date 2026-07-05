//
//  AudioData.swift
//  tesseract
//

import Foundation

/// The native-rate, pre-resample capture and its conditions — what the
/// microphone tap delivered (post–Voice Processing when that was enabled), the
/// **Capture Dump**'s input. Kept alongside the recognizer-ready samples so bad
/// transcriptions can be diagnosed and future offline experiments can re-run on
/// maximal-fidelity source material (PRD #175).
struct RawCapture: Sendable {
    let samples: [Float]
    let sampleRate: Double
    let voiceProcessed: Bool
}

/// Sendable audio data container for crossing concurrency boundaries
struct AudioData: Sendable {
    let samples: [Float]
    let sampleRate: Double
    let duration: TimeInterval

    /// The pre-resample capture this audio came from, when it came from the
    /// microphone (synthetic/test audio has none).
    var raw: RawCapture?

    var isEmpty: Bool {
        samples.isEmpty
    }

    var sampleCount: Int {
        samples.count
    }
}
