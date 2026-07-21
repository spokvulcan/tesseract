//
//  InMemoryAudioPlayback.swift
//  tesseractTests
//
//  A hermetic, in-memory `AudioPlayback` for tests — a *peer implementation* of
//  `AudioPlaybackManager`, not a mock. It records what the coordinator scheduled
//  (one-shot plays, the streaming session, appended chunks, pause/resume,
//  finish/stop) and derives `totalScheduledDuration` from the same
//  `StreamingScheduler` value machine the real adapters drive (ADR-0054) —
//  no hand-copied formula. Its playback clock is a *pure virtual clock*: it reads 0
//  until a test calls `advance(by:)`, never tracks wall-clock, and is untouched by
//  the lifecycle methods — so long-form pacing logic can be driven deterministically.
//  `firePlaybackFinished()` stands in for the audio layer reporting that scheduled
//  audio has drained. No AVAudioEngine, no real audio.
//

import Foundation

@testable import Tesseract_Agent

@MainActor
final class InMemoryAudioPlayback: AudioPlayback {
    // MARK: Installed by the coordinator
    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    // MARK: Pure virtual clock (test-controlled, never wall-clock)
    private var virtualPlaybackTime: TimeInterval = 0
    func advance(by seconds: TimeInterval) { virtualPlaybackTime += seconds }

    // MARK: Recorded state
    private(set) var playCount = 0
    private(set) var playedSamples: [[Float]] = []
    private(set) var playedSampleRates: [Int] = []
    private(set) var startStreamingCount = 0
    private(set) var startedSampleRates: [Int] = []
    private(set) var appendedChunks: [[Float]] = []
    private(set) var finishStreamingCount = 0
    private(set) var pauseCount = 0
    private(set) var resumeCount = 0
    private(set) var stopCount = 0
    private(set) var isPaused = false
    private(set) var setVolumeCalls: [Float] = []
    private(set) var volume: Float = 1.0

    /// Scripted envelope reading — tests set this to simulate the reply's
    /// loudness at the playback head.
    var scriptedPlaybackLevel: Float = 0

    /// The same push-scheduling value machine the production adapters drive —
    /// the peer no longer re-derives the duration formula (ADR-0054).
    private var scheduler = StreamingScheduler()
    var totalScheduledDuration: TimeInterval { scheduler.totalScheduledDuration }

    func currentPlaybackTime() -> TimeInterval { virtualPlaybackTime }

    func play(samples: [Float], sampleRate: Int) {
        playCount += 1
        playedSamples.append(samples)
        playedSampleRates.append(sampleRate)
    }

    func startStreaming(sampleRate: Int) {
        startStreamingCount += 1
        startedSampleRates.append(sampleRate)
        scheduler.beginStream(sampleRate: sampleRate)
        volume = 1.0
    }

    func appendChunk(samples: [Float]) {
        appendedChunks.append(samples)
        _ = scheduler.appendChunk(sampleCount: samples.count)
    }

    func finishStreaming() {
        finishStreamingCount += 1
        _ = scheduler.finishStream()
    }

    func pause() {
        pauseCount += 1
        isPaused = true
    }

    func resume() {
        resumeCount += 1
        isPaused = false
    }

    func stop() {
        stopCount += 1
        scheduler.stop()
        volume = 1.0
    }

    func playbackLevel() -> Float { scriptedPlaybackLevel }

    func setVolume(_ volume: Float) {
        setVolumeCalls.append(volume)
        self.volume = volume
    }

    /// Stands in for the audio layer reporting that scheduled audio has drained —
    /// fires the callback the coordinator installed on `onPlaybackFinished`.
    func firePlaybackFinished() {
        onPlaybackFinished?()
    }
}
