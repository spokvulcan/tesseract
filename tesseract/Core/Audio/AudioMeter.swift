//
//  AudioMeter.swift
//  tesseract
//

import Accelerate
import Foundation

/// One audio meter reading: overall loudness plus a small log-spaced
/// spectrum, both normalized 0–1. Produced on the real-time tap thread at
/// tap cadence (~47 Hz at 48 kHz / 1024 frames), consumed by the
/// `DictationFeed` on the main actor.
nonisolated struct MeterFrame: Equatable, Sendable {
    static let bandCount = 8
    static let zeroBands = [Float](repeating: 0, count: bandCount)
    static let zero = MeterFrame(level: 0, bands: zeroBands)

    /// 0–1, normalized from dBFS with a −60 dB floor (same scale the old
    /// single-level meter used).
    let level: Float
    /// `bandCount` log-spaced frequency bands (~60 Hz – 8 kHz), each 0–1 on
    /// the same dB normalization.
    let bands: [Float]
}

/// The real-time side of the meter: computes one `MeterFrame` per tap
/// callback and yields it into the engine's meter stream.
///
/// Everything is preallocated at init — the tap block runs serially on a
/// single audio thread, so the mutable scratch buffers need no lock. The
/// only per-callback heap work is the small `bands` array of the yielded
/// frame (`bandCount` floats) and the stream's buffered-newest slot.
nonisolated final class AudioMeterTap: @unchecked Sendable {
    private static let fftSize = 1024
    private static let log2n = vDSP_Length(10)
    private static let halfSize = fftSize / 2

    private let continuation: AsyncStream<MeterFrame>.Continuation
    private let fftSetup: FFTSetup
    /// Per-band half-open bin ranges, log-spaced across the spectrum.
    private let bandBins: [Range<Int>]

    private var input = [Float](repeating: 0, count: fftSize)
    private var real = [Float](repeating: 0, count: halfSize)
    private var imag = [Float](repeating: 0, count: halfSize)
    private var magnitudes = [Float](repeating: 0, count: halfSize)

    init?(sampleRate: Double, continuation: AsyncStream<MeterFrame>.Continuation) {
        guard let setup = vDSP_create_fftsetup(Self.log2n, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        self.fftSetup = setup
        self.continuation = continuation

        // Log-spaced band edges over the speech-relevant span (~60 Hz–8 kHz),
        // clamped to the bins this sample rate actually produces. Each band
        // gets at least one bin.
        let binWidth = sampleRate / Double(Self.fftSize)
        let lowHz = 60.0
        let highHz = min(8000.0, sampleRate / 2)
        let ratio = highHz / lowHz
        var ranges: [Range<Int>] = []
        var previousEnd = max(1, Int(lowHz / binWidth))
        for band in 1...MeterFrame.bandCount {
            let edgeHz = lowHz * pow(ratio, Double(band) / Double(MeterFrame.bandCount))
            let edgeBin = min(Self.halfSize, Int(edgeHz / binWidth))
            let end = max(edgeBin, previousEnd + 1)
            ranges.append(previousEnd..<min(end, Self.halfSize))
            previousEnd = min(end, Self.halfSize - 1)
        }
        self.bandBins = ranges
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Called from the tap block on the audio thread. `level` is the already
    /// normalized RMS loudness the tap computed.
    func process(_ channelData: UnsafePointer<Float>, frameCount: Int, level: Float) {
        // The tap's buffer size is a request, not a guarantee (Voice
        // Processing IO picks its own quantum) — copy what fits into the
        // fixed FFT input and zero-pad the rest.
        let copied = min(frameCount, Self.fftSize)
        input.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: channelData, count: copied)
            if copied < Self.fftSize {
                vDSP_vclr(dst.baseAddress! + copied, 1, vDSP_Length(Self.fftSize - copied))
            }
        }

        var bands = [Float](repeating: 0, count: MeterFrame.bandCount)
        real.withUnsafeMutableBufferPointer { realPtr in
            imag.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(
                    realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                input.withUnsafeBufferPointer { inputPtr in
                    inputPtr.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: Self.halfSize
                    ) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(Self.halfSize))
                    }
                }
                vDSP_fft_zrip(fftSetup, &split, 1, Self.log2n, FFTDirection(FFT_FORWARD))
                magnitudes.withUnsafeMutableBufferPointer { magPtr in
                    vDSP_zvmags(&split, 1, magPtr.baseAddress!, 1, vDSP_Length(Self.halfSize))
                }
            }
        }

        // Per-band mean power → dB → the same −60 dB…0 normalization as the
        // level, so bands and level read on one scale.
        let scale = 1.0 / Float(Self.fftSize * Self.fftSize)
        for (index, bins) in bandBins.enumerated() where !bins.isEmpty {
            var mean: Float = 0
            magnitudes.withUnsafeBufferPointer { magPtr in
                vDSP_meanv(
                    magPtr.baseAddress! + bins.lowerBound, 1, &mean,
                    vDSP_Length(bins.count))
            }
            let db = 10 * log10(max(mean * scale, 1e-12))
            bands[index] = max(0, min(1, (db + 60) / 60))
        }

        continuation.yield(MeterFrame(level: level, bands: bands))
    }

    /// Resets consumers to silence (capture stopped or failed).
    func pushZero() {
        continuation.yield(.zero)
    }
}
