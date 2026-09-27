// TesseractSpeech — the silence cap on a segment's audio (ADR-0072). Pure.

import Foundation

/// Caps each run of silence in one segment's audio. With a Reference Take in
/// the prompt, the model now and then stalls mid-segment or trails off for
/// seconds before it ends one. Frames quieter than `threshold` count as
/// silence; once a run passes `maxSilence`, further silent frames are
/// dropped until sound resumes. Pauses a reader makes, well under the cap,
/// pass through unchanged, and no spoken frame is ever dropped.
struct SilenceCap {
    /// One codec frame of the audio it saw: how loud it was, and whether it
    /// was kept. The word timer (ADR-0077) finds pauses in the first and
    /// counts frames through the second.
    struct Frame: Equatable {
        /// RMS level in dBFS.
        let loudness: Float
        let kept: Bool
    }

    let samplesPerFrame: Int
    let maxFrames: Int
    /// Frame RMS below this is silence. The decoder's silence sits near
    /// -75 dBFS and speech near -30, so -60 dBFS separates them cleanly.
    let threshold: Float

    private var run = 0

    init(format: AudioFormat, maxSilence: TimeInterval = 1.2, threshold: Float = 0.001) {
        self.samplesPerFrame = format.samplesPerFrame
        self.maxFrames = Int((maxSilence * format.framesPerSecond).rounded())
        self.threshold = threshold
    }

    /// The chunk without the silent frames past the cap. Chunks arrive as
    /// whole frames; the run carries over from one chunk to the next.
    mutating func apply(_ samples: [Float]) -> [Float] {
        var frames: [Frame] = []
        return apply(samples, frames: &frames)
    }

    /// `apply`, also appending each frame's level and fate to `frames`.
    mutating func apply(_ samples: [Float], frames: inout [Frame]) -> [Float] {
        var kept: [Float] = []
        kept.reserveCapacity(samples.count)
        var start = 0
        while start < samples.count {
            let end = min(start + samplesPerFrame, samples.count)
            let frame = samples[start..<end]
            let energy = frame.reduce(Float(0)) { $0 + $1 * $1 } / Float(frame.count)
            let rms = energy.squareRoot()
            let keep: Bool
            if rms < threshold {
                run += 1
                keep = run <= maxFrames
            } else {
                run = 0
                keep = true
            }
            if keep { kept.append(contentsOf: frame) }
            frames.append(Frame(loudness: 20 * log10(rms + 1e-12), kept: keep))
            start = end
        }
        return kept
    }
}
