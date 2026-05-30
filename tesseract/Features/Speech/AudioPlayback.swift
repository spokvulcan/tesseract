//
//  AudioPlayback.swift
//  tesseract
//
//  The playback port (seam) for synthesized audio, sitting *below* the
//  `SpeechCoordinator`. See CONTEXT.md → Language → "Speech model ports and
//  playback". Unlike the `SpeechRecognizer` / `SpeechSynthesizer` *model* ports
//  (actor-backed, `Sendable`, raced or run off the main actor), playback is
//  inherently main-actor: it drives `AVAudioEngine` and observable state, isn't
//  raced across a `Sendable` boundary, and the coordinator holds a long-lived
//  reference whose `onPlaybackFinished` it mutates. So this port is a
//  `@MainActor` class-bound protocol — like the `TextExtracting` /
//  `AudioCapturing` / `TextInjecting` collaborator ports, not the model ports.
//
//  Above the port stay segmentation, the `@Observable` `SpeechState` machine,
//  voice-anchor sequencing, and the notch overlay. Below it: format/engine setup,
//  one-shot and push-based streaming buffer scheduling, the playback-time clock,
//  and the optional debug dump. What crosses: `[Float]` samples (plus a sample
//  rate) in, a finished signal (`onPlaybackFinished`) and the playback clock
//  (`currentPlaybackTime()` / `totalScheduledDuration`) out.
//

import Foundation

/// Whether a streaming playback session writes its debug dump. Passed at
/// `startStreaming` instead of carried as mutable playback state: long-form (many
/// segments) disables it; single-shot streaming uses `.default`.
enum PlaybackDiagnosticsPolicy: Sendable, Equatable {
    case `default`
    case disabled
}

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

    /// Begins a push-based streaming session at `sampleRate`. `diagnostics`
    /// decides whether this session writes its debug dump.
    func startStreaming(sampleRate: Int, diagnostics: PlaybackDiagnosticsPolicy)

    /// Schedules one chunk of samples onto the current streaming session.
    func appendChunk(samples: [Float])

    /// Signals that no more chunks will be appended; playback finishes once the
    /// already-scheduled audio drains.
    func finishStreaming()

    /// The playback head position, in seconds, of the current session.
    func currentPlaybackTime() -> TimeInterval

    /// Stops and tears down any in-flight playback.
    func stop()
}
