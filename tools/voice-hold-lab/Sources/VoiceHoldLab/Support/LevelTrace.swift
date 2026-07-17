//
//  LevelTrace.swift — 20 Hz normalized mic-level recording off a capture tap.
//
//  The normalization mirrors the app's meter
//  (`AudioConverter.meterLevel(rms:)`): RMS → dBFS → 0–1 over a −60 dB
//  floor. Keep the three lines identical — the app-side replay tests
//  (`VoiceBargeReplayTests`) feed these traces to the real detector, and a
//  domain mismatch would invalidate every calibrated constant.
//

import AVFoundation
import Accelerate

/// Collects tap buffers on the RT thread and folds them into 50 ms levels.
final class LevelTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private let sampleRate: Double
    private let started = Date()

    init(sampleRate: Double) {
        self.sampleRate = sampleRate
    }

    func ingest(_ buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        lock.lock()
        samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: count))
        lock.unlock()
    }

    /// The trace as 20 Hz normalized levels — the app meter's domain.
    func levels() -> [Float] {
        lock.lock()
        let snapshot = samples
        lock.unlock()
        return Self.bins(of: snapshot, sampleRate: sampleRate)
    }

    /// 50 ms-bin normalized levels of `samples` — the one binning walk the
    /// mic traces and the synthesized playback envelopes share, so replay
    /// fixtures align bin-for-bin.
    static func bins(of samples: [Float], sampleRate: Double) -> [Float] {
        let window = Int(sampleRate * 0.05)
        guard window > 0, !samples.isEmpty else { return [] }
        var levels: [Float] = []
        var start = 0
        while samples.count - start >= window {
            var rms: Float = 0
            samples.withUnsafeBufferPointer { pointer in
                vDSP_rmsqv(pointer.baseAddress! + start, 1, &rms, vDSP_Length(window))
            }
            levels.append(normalized(rms: rms))
            start += window
        }
        return levels
    }

    static func normalized(rms: Float) -> Float {
        let db = 20 * log10(max(rms, 0.001))
        return max(0, min(1, (db + 60) / 60))
    }
}

extension Array where Element == Float {
    var peak: Float { self.max() ?? 0 }
    var mean: Float { isEmpty ? 0 : reduce(0, +) / Float(count) }

    /// Levels formatted for the RUNBOOK tables.
    func stats(label: String) -> String {
        String(
            format: "%@: peak %.3f  mean %.3f  n=%d", label, peak, mean, count)
    }
}
