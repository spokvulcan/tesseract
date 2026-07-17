//
//  AudioPlayback.swift
//  tesseract
//
//  The playback port (seam) for synthesized audio, sitting *below* the
//  `SpeechCoordinator`. See CONTEXT.md → Language → "Speech model ports and
//  playback". Unlike the model ports (actor-backed, `Sendable`, raced or run
//  off the main actor), playback is inherently main-actor: it drives
//  `AVAudioEngine` and observable state, isn't raced across a `Sendable`
//  boundary, and the coordinator holds a long-lived reference whose
//  `onPlaybackFinished` it mutates. So this port is a `@MainActor`
//  class-bound protocol — like the `TextExtracting` / `AudioCapturing` /
//  `TextInjecting` collaborator ports, not the model ports.
//
//  Above the port: the v2 engine drain loop, the `@Observable` `SpeechState`
//  machine, and the notch overlay. Below it: format/engine setup, one-shot
//  and push-based streaming buffer scheduling, real pause/resume, and the
//  playback-time clock. What crosses: `[Float]` samples (plus a sample rate)
//  in, a finished signal (`onPlaybackFinished`) and the playback clock
//  (`currentPlaybackTime()` / `totalScheduledDuration`) out.
//

import Foundation
import Accelerate

@MainActor
protocol AudioPlayback: AnyObject {
    /// Fired when scheduled audio has finished draining. The coordinator installs
    /// this to advance its `SpeechState` back to `.idle`. `@MainActor @Sendable`
    /// so it matches the main-actor coordinator and rules out cross-actor calls.
    var onPlaybackFinished: (@MainActor @Sendable () -> Void)? { get set }

    /// Cumulative duration of all audio scheduled so far in the current streaming
    /// session, in seconds.
    var totalScheduledDuration: TimeInterval { get }

    /// Plays `samples` in one shot, replacing any in-flight playback.
    func play(samples: [Float], sampleRate: Int)

    /// Begins a push-based streaming session at `sampleRate`.
    func startStreaming(sampleRate: Int)

    /// Schedules one chunk of samples onto the current streaming session.
    func appendChunk(samples: [Float])

    /// Signals that no more chunks will be appended; playback finishes once the
    /// already-scheduled audio drains.
    func finishStreaming()

    /// Pauses the playback head in place; already-scheduled audio stays queued.
    /// While paused, `currentPlaybackTime()` holds at the pause position.
    func pause()

    /// Resumes from a pause.
    func resume()

    /// The playback head position, in seconds, of the current session.
    func currentPlaybackTime() -> TimeInterval

    /// Stops and tears down any in-flight playback.
    func stop()

    /// Coarse loudness of what is playing at the current playback head, in
    /// the same 0–1 dB-normalized domain as the mic meter (`MeterFrame`);
    /// 0 while idle or paused. The Echo Floor (ADR-0041) reads this to know
    /// when the reply is audibly emitting — ±100 ms accuracy suffices.
    func playbackLevel() -> Float

    /// Sets the playback volume (0–1) instantly on the playing node —
    /// the Soft Barge duck (ADR-0041). Ramping is the caller's job.
    /// Implementations reset to 1.0 at `startStreaming`/`stop` so a duck
    /// can never leak into the next utterance.
    func setVolume(_ volume: Float)

    /// The volume as currently applied (1.0 when no node is live) — the
    /// fade ramp's starting point.
    var volume: Float { get }
}

/// The scheduled-audio loudness timeline behind `playbackLevel()` — shared by
/// every real sink so hosted and dedicated playback report one envelope
/// domain. Pure; bins appended samples at a 50 ms grain and answers with the
/// mic meter's dB normalization (−60 dB floor), so floor and envelope read on
/// one scale.
nonisolated struct PlaybackEnvelope {

    /// One bin per 50 ms — matches the session ticker's grain; the envelope
    /// is a coarse "is the reply audible" signal, not a waveform.
    static let binDuration: TimeInterval = 0.05

    private var bins: [Float] = []
    private var sampleRate: Double = 0
    private var pendingSamples: [Float] = []

    mutating func begin(sampleRate: Int) {
        reset()
        self.sampleRate = Double(sampleRate)
    }

    mutating func reset() {
        bins = []
        pendingSamples = []
        sampleRate = 0
    }

    /// Appends a scheduled chunk, folding a partial trailing bin into the
    /// next append so bin boundaries never depend on chunk boundaries.
    mutating func append(samples: [Float]) {
        guard sampleRate > 0, !samples.isEmpty else { return }
        let binSize = Int(sampleRate * Self.binDuration)
        guard binSize > 0 else { return }
        pendingSamples.append(contentsOf: samples)
        var start = 0
        while pendingSamples.count - start >= binSize {
            var rms: Float = 0
            pendingSamples.withUnsafeBufferPointer { pointer in
                vDSP_rmsqv(pointer.baseAddress! + start, 1, &rms, vDSP_Length(binSize))
            }
            bins.append(AudioConverter.meterLevel(rms: rms))
            start += binSize
        }
        pendingSamples.removeFirst(start)
    }

    /// The loudness at `time` into the scheduled audio; 0 past the end.
    func level(at time: TimeInterval) -> Float {
        guard time >= 0 else { return 0 }
        let index = Int(time / Self.binDuration)
        guard index < bins.count else { return 0 }
        return bins[index]
    }
}
