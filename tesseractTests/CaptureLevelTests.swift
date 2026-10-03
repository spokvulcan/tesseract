//
//  CaptureLevelTests.swift
//  tesseractTests
//
//  The silent-capture skip's level check (PRD #612): the loudest 20 ms window
//  of a capture, in dBFS, and whether it stays at or below the silence
//  ceiling. Signals are built by hand: digital silence, quiet noise, a speech
//  level sine, and a short burst inside silence.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CaptureLevelTests {

    // MARK: - Signal helpers

    private func sine(
        amplitude: Float, frequency: Double = 250, sampleRate: Double = 48_000,
        duration: Double = 1
    ) -> [Float] {
        let count = Int(sampleRate * duration)
        return (0..<count).map { index in
            amplitude * Float(sin(2.0 * .pi * frequency * Double(index) / sampleRate))
        }
    }

    /// Deterministic white noise whose RMS is `dbfs`.
    private func noise(dbfs: Float, count: Int) -> [Float] {
        var state: UInt32 = 0x1234_5678
        let raw = (0..<count).map { _ -> Float in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state) / Float(UInt32.max) * 2 - 1
        }
        let rms = (raw.reduce(0) { $0 + $1 * $1 } / Float(count)).squareRoot()
        let target = pow(10, dbfs / 20)
        return raw.map { $0 / rms * target }
    }

    private func audio(_ samples: [Float], sampleRate: Double = 48_000) -> AudioData {
        AudioData(
            samples: samples, sampleRate: sampleRate,
            duration: Double(samples.count) / sampleRate)
    }

    private func isClose(_ value: Float, _ expected: Float, tolerance: Float = 0.05) -> Bool {
        abs(value - expected) <= tolerance
    }

    // MARK: - Silence

    @Test
    func emptySamplesAreSilent() {
        #expect(CaptureLevel.peakDBFS([], sampleRate: 48_000) == CaptureLevel.floorDBFS)
        #expect(CaptureLevel.isSilent(audio([])))
    }

    @Test
    func digitalSilenceIsSilent() {
        let zeros = [Float](repeating: 0, count: 48_000)
        #expect(CaptureLevel.peakDBFS(zeros, sampleRate: 48_000) == CaptureLevel.floorDBFS)
        #expect(CaptureLevel.isSilent(audio(zeros)))
    }

    @Test
    func quietRoomNoiseIsSilent() {
        let room = noise(dbfs: -80, count: 96_000)
        let peak = CaptureLevel.peakDBFS(room, sampleRate: 48_000)
        #expect(peak > -85 && peak < -75)
        #expect(CaptureLevel.isSilent(audio(room)))
    }

    // MARK: - Speech

    @Test
    func speechLevelSineIsNotSilent() {
        // Amplitude 0.1 is an RMS of 0.0707, -23 dBFS. At 250 Hz every 20 ms
        // window holds whole periods, so each window measures exactly that.
        let speech = sine(amplitude: 0.1)
        #expect(isClose(CaptureLevel.peakDBFS(speech, sampleRate: 48_000), -23.01))
        #expect(!CaptureLevel.isSilent(audio(speech)))
    }

    @Test
    func shortLoudBurstInsideSilenceIsNotSilent() {
        var samples = [Float](repeating: 0, count: 96_000)
        let burst = sine(amplitude: 0.3, duration: 0.03)
        samples.replaceSubrange(48_000..<(48_000 + burst.count), with: burst)
        #expect(CaptureLevel.peakDBFS(samples, sampleRate: 48_000) > -20)
        #expect(!CaptureLevel.isSilent(audio(samples)))
    }

    @Test
    func theCeilingSeparatesSilentFromNot() {
        let below = pow(10, (CaptureLevel.silenceCeiling - 1) / 20)
        let above = pow(10, (CaptureLevel.silenceCeiling + 1) / 20)
        #expect(CaptureLevel.isSilent(audio([Float](repeating: below, count: 4_800))))
        #expect(!CaptureLevel.isSilent(audio([Float](repeating: above, count: 4_800))))
    }

    @Test
    func aCaptureShorterThanOneWindowIsMeasuredWhole() {
        // The dictation test fakes' one-sample captures stay loud.
        #expect(isClose(CaptureLevel.peakDBFS([0.1], sampleRate: 16_000), -20))
        #expect(!CaptureLevel.isSilent(AudioData(samples: [0.1], sampleRate: 16_000, duration: 2)))
    }

    // MARK: - Window math

    @Test
    func aShortLastWindowIsMeasuredOnItsOwnSamples() {
        // 20-sample windows at 1 kHz; 45 samples leave a last window of 5.
        let samples = [Float](repeating: 0, count: 40) + [Float](repeating: 0.5, count: 5)
        #expect(isClose(CaptureLevel.peakDBFS(samples, sampleRate: 1_000), -6.02))
    }

    @Test
    func windowsDoNotOverlapAndStartAtTheFirstSample() {
        // Ten loud samples on each side of the first window boundary: each of
        // the two windows holds half loud samples, an RMS of 0.707.
        let samples =
            [Float](repeating: 0, count: 10) + [Float](repeating: 1, count: 20)
            + [Float](repeating: 0, count: 15)
        #expect(isClose(CaptureLevel.peakDBFS(samples, sampleRate: 1_000), -3.01))
    }

    @Test
    func theWindowLengthFollowsTheSampleRate() {
        // 10 loud samples: a whole 10-sample window at 500 Hz, half of a
        // 20-sample window at 1 kHz.
        let samples = [Float](repeating: 1, count: 10) + [Float](repeating: 0, count: 30)
        #expect(isClose(CaptureLevel.peakDBFS(samples, sampleRate: 500), 0))
        #expect(isClose(CaptureLevel.peakDBFS(samples, sampleRate: 1_000), -3.01))
    }

    @Test
    func aRateWithNoUsableWindowMeasuresTheWholeCapture() {
        // RMS of [0, 0, 0, 1] is 0.5, -6.02 dBFS.
        #expect(isClose(CaptureLevel.peakDBFS([0, 0, 0, 1], sampleRate: 0), -6.02))
    }
}
