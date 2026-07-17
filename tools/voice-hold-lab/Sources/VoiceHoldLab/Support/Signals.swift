//
//  Signals.swift — deterministic test signals for the residual scenarios.
//
//  Two shapes: a log chirp (finds frequency-dependent AEC weakness) and a
//  "TTS-shaped" signal — vowel-ish harmonic bursts with word-gap silences,
//  matching the envelope dynamics that tripped the field detector (onsets,
//  not steady state). Deterministic so every lab run is comparable.
//

import Foundation

enum Signals {

    /// Log frequency sweep 100 Hz → 8 kHz.
    static func chirp(seconds: Double, sampleRate: Double, amplitude: Float = 0.5) -> [Float] {
        let count = Int(seconds * sampleRate)
        let f0 = 100.0, f1 = 8000.0
        let k = pow(f1 / f0, 1.0 / seconds)
        var phase = 0.0
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let t = Double(i) / sampleRate
            let f = f0 * pow(k, t)
            phase += 2.0 * .pi * f / sampleRate
            out[i] = amplitude * Float(sin(phase))
        }
        return out
    }

    /// The played signal's 50 ms-bin envelope in the app's PlaybackEnvelope
    /// domain — the replay tests' far-end input, aligned bin-for-bin with
    /// the mic traces recorded while it played.
    static func envelope(_ samples: [Float], sampleRate: Double) -> [Float] {
        let window = Int(sampleRate * 0.05)
        guard window > 0 else { return [] }
        var out: [Float] = []
        var start = 0
        while samples.count - start >= window {
            var sum: Float = 0
            for i in start..<(start + window) { sum += samples[i] * samples[i] }
            out.append(LevelTrace.normalized(rms: sqrt(sum / Float(window))))
            start += window
        }
        return out
    }

    /// Harmonic bursts (~160 Hz fundamental + partials) in a speech-like
    /// cadence: ~300 ms voiced bursts, ~150 ms gaps, envelope-ramped so
    /// every burst is an *onset* — the transient class that leaked past AEC.
    static func speechShaped(seconds: Double, sampleRate: Double, amplitude: Float = 0.5)
        -> [Float]
    {
        let count = Int(seconds * sampleRate)
        var out = [Float](repeating: 0, count: count)
        let burst = Int(0.3 * sampleRate)
        let gap = Int(0.15 * sampleRate)
        let period = burst + gap
        let partials: [(Double, Float)] = [(160, 1.0), (320, 0.6), (480, 0.4), (960, 0.2)]
        for i in 0..<count {
            let posInPeriod = i % period
            guard posInPeriod < burst else { continue }
            let t = Double(i) / sampleRate
            // 20 ms attack/release envelope inside the burst.
            let edge = Int(0.02 * sampleRate)
            let env: Float
            if posInPeriod < edge {
                env = Float(posInPeriod) / Float(edge)
            } else if burst - posInPeriod < edge {
                env = Float(burst - posInPeriod) / Float(edge)
            } else {
                env = 1
            }
            var sample: Float = 0
            for (f, w) in partials {
                sample += w * Float(sin(2.0 * .pi * f * t))
            }
            out[i] = amplitude * env * sample / 2.2
        }
        return out
    }
}
