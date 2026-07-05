//
//  AudioConverterResampleTests.swift
//  tesseractTests
//
//  Signal-level tests for the consolidated resampler in `AudioConverter` — the
//  single downsampling implementation on the dictation capture path (PRD #175).
//  The contract under test is behavioral, not structural: content above the
//  target Nyquist must be attenuated (anti-aliasing), in-band content must
//  survive with pitch and level intact, and output length must follow the rate
//  ratio. A naive linear-interpolation resampler fails the anti-aliasing
//  expectation by construction — it folds high-frequency noise into the speech
//  band, which is exactly the defect this seam exists to prevent.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AudioConverterResampleTests {

    // MARK: - Signal helpers

    private func sine(
        frequency: Double, sampleRate: Double, duration: Double, amplitude: Float = 0.5
    ) -> [Float] {
        let count = Int(sampleRate * duration)
        return (0..<count).map { index in
            amplitude * Float(sin(2.0 * .pi * frequency * Double(index) / sampleRate))
        }
    }

    /// RMS over the middle half of the signal — edges excluded so converter
    /// priming/latency transients don't pollute steady-state level checks.
    private func middleRMS(_ samples: [Float]) -> Float {
        guard samples.count >= 8 else { return 0 }
        let quarter = samples.count / 4
        let middle = samples[quarter..<(samples.count - quarter)]
        let sumOfSquares = middle.reduce(Float(0)) { $0 + $1 * $1 }
        return (sumOfSquares / Float(middle.count)).squareRoot()
    }

    /// Zero-crossing count over the middle half — a frequency probe that is
    /// independent of level.
    private func middleZeroCrossings(_ samples: [Float]) -> Int {
        guard samples.count >= 8 else { return 0 }
        let quarter = samples.count / 4
        let middle = Array(samples[quarter..<(samples.count - quarter)])
        var crossings = 0
        for index in 1..<middle.count where (middle[index - 1] < 0) != (middle[index] < 0) {
            crossings += 1
        }
        return crossings
    }

    // MARK: - Anti-aliasing

    /// A 10 kHz tone lies above the 8 kHz Nyquist of the 16 kHz target rate: a
    /// correct resampler filters it out; a naive one folds it to 6 kHz nearly
    /// full-strength. Expect at least ~26 dB of attenuation.
    @Test func downsamplingAttenuatesToneAboveTargetNyquist() {
        let source = sine(frequency: 10_000, sampleRate: 48_000, duration: 1.0)
        let sourceRMS = middleRMS(source)

        let output = AudioConverter.resample(source, from: 48_000, to: 16_000)

        #expect(middleRMS(output) < sourceRMS * 0.05)
    }

    // MARK: - In-band fidelity

    /// A 1 kHz tone is comfortably in-band at 16 kHz and must come through with
    /// its level (±10 %) and pitch (zero-crossing rate ±3 %) intact.
    @Test func downsamplingPreservesInBandToneLevelAndPitch() {
        let source = sine(frequency: 1_000, sampleRate: 48_000, duration: 1.0)
        let sourceRMS = middleRMS(source)

        let output = AudioConverter.resample(source, from: 48_000, to: 16_000)
        let outputRMS = middleRMS(output)

        #expect(abs(outputRMS - sourceRMS) < sourceRMS * 0.10)

        // Middle half of 1 s at 16 kHz spans 0.5 s: a 1 kHz tone crosses zero
        // ~1000 times in that window.
        let crossings = middleZeroCrossings(output)
        #expect(abs(crossings - 1000) <= 30)
    }

    // MARK: - Length

    @Test func downsamplingProducesRateRatioLength() {
        let source = [Float](repeating: 0.1, count: 24_000)  // 0.5 s at 48 kHz

        let output = AudioConverter.resample(source, from: 48_000, to: 16_000)

        // 8000 expected; allow a small converter priming/flush tolerance.
        #expect(abs(output.count - 8_000) <= 32)
    }

    @Test func downsamplingHandlesNonIntegerRateRatio() {
        let source = sine(frequency: 1_000, sampleRate: 44_100, duration: 1.0)

        let output = AudioConverter.resample(source, from: 44_100, to: 16_000)

        #expect(abs(output.count - 16_000) <= 32)
    }

    // MARK: - Edge cases

    @Test func sameRatePassesSamplesThroughUnchanged() {
        let source = sine(frequency: 440, sampleRate: 16_000, duration: 0.1)

        let output = AudioConverter.resample(source, from: 16_000, to: 16_000)

        #expect(output == source)
    }

    @Test func emptyInputYieldsEmptyOutput() {
        let output = AudioConverter.resample([], from: 48_000, to: 16_000)

        #expect(output.isEmpty)
    }
}
