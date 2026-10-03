//
//  CaptureLevel.swift
//  tesseract
//
//  How loud a capture got, for the silent-capture skip (PRD #612): a capture
//  whose level never rises above silence is not transcribed. WhisperKit 1.0.0
//  never computes its no-speech probability (`noSpeechThreshold` does
//  nothing), and Whisper turns silence into a confident "Thank you.", so the
//  app checks the level itself. Runs once per take on the main actor at key
//  release; 30 s at 48 kHz takes about 0.1 ms.
//

import Accelerate
import Foundation

nonisolated enum CaptureLevel {

    /// The level reported for an empty or all-zero capture, in dBFS.
    static let floorDBFS: Float = -120

    /// A capture whose loudest window stays at or below this level (dBFS) is
    /// silent. Measured with `peakDBFS` on the last 100 takes of the owner's
    /// **Capture Dump** (Voice Processing on, 48 and 24 kHz):
    ///
    /// - The 98 takes that held speech peaked between -28.9 dBFS (one quiet
    ///   word) and -3.7 dBFS: 1st percentile -22.8, 5th -19.0, median -11.6.
    /// - The one take recorded with nothing said peaked at -63.6 dBFS (a brief
    ///   rise as the capture started, then -88 dBFS) and came back from
    ///   Whisper as "Thank you.". Its overall RMS was -79 dBFS.
    /// - A 0.5 s take that left no **Correction Pair** peaked at -50.9 dBFS in
    ///   a single 20 ms window, likely a click; it stays above the ceiling.
    /// - The room between words sits around -71 dBFS (median of each take's
    ///   10th-percentile window).
    ///
    /// -55 dBFS sits 26 dB below the quietest speech, so whispered speech
    /// still counts, and 8 dB above the silent take. The margin leans toward
    /// speech because a miss costs more there: a silent capture judged loud
    /// is transcribed as it was before this check, while speech judged
    /// silent would be lost.
    static let silenceCeiling: Float = -55

    /// The loudest short-window RMS of `samples`, in dBFS (1.0 is 0 dBFS).
    /// Windows are `window` seconds long and do not overlap; a shorter last
    /// window counts as one, so a capture shorter than a window is measured
    /// whole. Empty or all-zero input gives `floorDBFS`.
    static func peakDBFS(
        _ samples: [Float], sampleRate: Double, window: TimeInterval = 0.02
    ) -> Float {
        guard !samples.isEmpty else { return floorDBFS }
        let length = windowLength(count: samples.count, sampleRate: sampleRate, window: window)
        var peak: Float = 0
        samples.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            var start = 0
            while start < buffer.count {
                let count = min(length, buffer.count - start)
                var rms: Float = 0
                vDSP_rmsqv(base + start, 1, &rms, vDSP_Length(count))
                // `max` keeps `peak` when `rms` is NaN.
                peak = max(peak, rms)
                start += count
            }
        }
        guard peak > 0 else { return floorDBFS }
        return max(floorDBFS, 20 * log10(peak))
    }

    /// Whether `audio` never rises above `silenceCeiling`.
    static func isSilent(_ audio: AudioData) -> Bool {
        peakDBFS(audio.samples, sampleRate: audio.sampleRate) <= silenceCeiling
    }

    /// Samples per window: at least one, at most the whole capture. A rate or
    /// window that gives no usable length measures the capture as one window.
    private static func windowLength(
        count: Int, sampleRate: Double, window: TimeInterval
    ) -> Int {
        let exact = window * sampleRate
        guard exact.isFinite, exact > 0 else { return count }
        return max(1, Int(min(exact.rounded(), Double(count))))
    }
}
