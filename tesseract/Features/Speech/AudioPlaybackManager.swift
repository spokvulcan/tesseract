//
//  AudioPlaybackManager.swift
//  tesseract
//

import Foundation
import Combine
import AVFoundation
import os

// MARK: - AudioPlaybackManager

@MainActor
final class AudioPlaybackManager: ObservableObject, AudioPlayback {
    @Published private(set) var isPlaying = false

    // Audio engine (shared between one-shot and streaming)
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    // Streaming format for buffer creation — the engine owns the connection.
    private var streamingFormat: AVAudioFormat?

    // Time-stretch between player and mixer: the read-aloud speed. The
    // player node's clock stays in the audio's own time.
    private var timePitch: AVAudioUnitTimePitch?
    private var playbackRate: Float = 1.0

    // The push scheduler's counters, start gate, finish detection, and stream
    // epoch (StreamingScheduler / ADR-0054). Every counter mutation and gate
    // decision is a verdict from here; this adapter performs only the
    // `AVAudioPlayerNode`/engine calls.
    private var scheduler = StreamingScheduler()

    // Non-nil while paused: the clock position to hold. `AVAudioPlayerNode`
    // reports no render time while paused, which would read as a rewind to 0.
    // Adapter-local — the machine has no notion of render time.
    private var pausedTime: TimeInterval?

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    // MARK: - Playback time tracking

    var totalScheduledDuration: TimeInterval { scheduler.totalScheduledDuration }

    func currentPlaybackTime() -> TimeInterval {
        if let pausedTime { return pausedTime }
        guard let node = playerNode,
            let nodeTime = node.lastRenderTime,
            let playerTime = node.playerTime(forNodeTime: nodeTime)
        else {
            return 0
        }
        return Double(playerTime.sampleTime) / playerTime.sampleRate
    }

    func heardPlaybackTime() -> TimeInterval {
        if let pausedTime { return pausedTime }
        return heardTime(head: currentPlaybackTime(), node: playerNode, rate: playbackRate)
    }

    // MARK: - One-shot playback (existing API)

    func play(samples: [Float], sampleRate: Int) {
        stop()

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()

        guard
            let buffer = AudioConverter.makeMonoFloat32Buffer(
                samples, sampleRate: Double(sampleRate))
        else {
            Log.speech.error("Failed to create audio buffer")
            return
        }

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: buffer.format)

        do {
            try engine.start()
        } catch {
            Log.speech.error("Failed to start audio engine: \(error)")
            return
        }

        audioEngine = engine
        playerNode = player
        isPlaying = true

        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                self?.isPlaying = false
                self?.onPlaybackFinished?()
            }
        }

        player.play()
        Log.speech.info("Playing TTS audio: \(samples.count) samples at \(sampleRate)Hz")
    }

    // MARK: - Streaming playback (push-based AVAudioPlayerNode)

    func startStreaming(sampleRate: Int) {
        stop()

        guard let format = AudioConverter.monoFloat32Format(sampleRate: Double(sampleRate))
        else {
            Log.speech.error("Failed to create audio format for streaming")
            return
        }

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let stretch = AVAudioUnitTimePitch()

        engine.attach(player)
        engine.attach(stretch)
        engine.connect(player, to: stretch, format: format)
        engine.connect(stretch, to: engine.mainMixerNode, format: format)
        Self.apply(playbackRate, to: stretch)

        do {
            try engine.start()
        } catch {
            Log.speech.error("Failed to start audio engine for streaming: \(error)")
            return
        }

        audioEngine = engine
        playerNode = player
        timePitch = stretch
        streamingFormat = format
        scheduler.beginStream(sampleRate: sampleRate)
        pausedTime = nil
        isPlaying = true

        Log.speech.info("Started streaming at \(sampleRate)Hz (push-based AVAudioPlayerNode)")
    }

    func appendChunk(samples: [Float]) {
        guard let node = playerNode, let format = streamingFormat else { return }
        guard !samples.isEmpty else { return }

        // Create and schedule a buffer for this chunk
        guard let buffer = AudioConverter.makeMonoFloat32Buffer(samples, format: format)
        else {
            Log.speech.error("Failed to create PCM buffer for chunk")
            return
        }

        let outcome = scheduler.appendChunk(sampleCount: samples.count)
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                // A stopped/restarted node flushes its handlers — the epoch
                // guard (ADR-0054) drops a stale completion before it can
                // decrement the new stream's counter and finish it early.
                if case .finished = self.scheduler.bufferCompleted(epoch: outcome.epoch) {
                    self.isPlaying = false
                    self.onPlaybackFinished?()
                }
            }
        }

        // Start playback on first chunk — unless paused before audio arrived.
        if outcome.startPlayer {
            node.play()
        }
    }

    func finishStreaming() {
        // If all buffers already drained, finish now; otherwise the last
        // buffer's completion callback handles it.
        if case .finishedNow = scheduler.finishStream() {
            isPlaying = false
            onPlaybackFinished?()
        }
    }

    // MARK: - Speed

    func setPlaybackRate(_ rate: Float) {
        playbackRate = min(max(rate, 0.5), 2.5)
        if let timePitch { Self.apply(playbackRate, to: timePitch) }
    }

    /// At 1× the stretch is bypassed, so normal speed is untouched audio.
    private static func apply(_ rate: Float, to unit: AVAudioUnitTimePitch) {
        unit.rate = rate
        unit.bypass = rate == 1.0
    }

    // MARK: - Pause / resume

    func pause() {
        guard scheduler.pause() else { return }
        pausedTime = currentPlaybackTime()
        playerNode?.pause()
        isPlaying = false
    }

    func resume() {
        guard scheduler.resume() else { return }
        pausedTime = nil
        // Something outside the app can stop the engine under a pause (an
        // iPhone's audio interruption); the player won't play on a stopped
        // engine, so start it again first.
        if let engine = audioEngine, !engine.isRunning {
            do {
                try engine.start()
            } catch {
                Log.speech.error("Failed to restart the audio engine: \(error)")
            }
        }
        if let node = playerNode {
            node.play()
            isPlaying = true
        }
    }

    // MARK: - Stop

    func stop() {
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        timePitch = nil
        streamingFormat = nil
        scheduler.stop()
        pausedTime = nil
        isPlaying = false
    }
}
