//
//  AudioData.swift
//  tesseract
//

import Foundation

/// Sendable audio data container for crossing concurrency boundaries
struct AudioData: Sendable {
    let samples: [Float]
    let sampleRate: Double
    let duration: TimeInterval

    var isEmpty: Bool {
        samples.isEmpty
    }

    var sampleCount: Int {
        samples.count
    }
}
