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

    // Streaming state — progressive chunk scheduling
    private var streamingFormat: AVAudioFormat?
    private var streamFinished = false
    private var pendingBufferCount = 0
    private var playerStarted = false

    // Non-nil while paused: the clock position to hold. `AVAudioPlayerNode`
    // reports no render time while paused, which would read as a rewind to 0.
    private var pausedTime: TimeInterval?

    // The scheduled-audio loudness timeline behind `playbackLevel()`.
    private var envelope = PlaybackEnvelope()

    private(set) var totalScheduledSamples: Int = 0
    private var streamingSampleRate: Int = 0

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    // MARK: - Playback time tracking

    var totalScheduledDuration: TimeInterval {
        guard streamingSampleRate > 0 else { return 0 }
        return Double(totalScheduledSamples) / Double(streamingSampleRate)
    }

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

    func playbackLevel() -> Float {
        guard isPlaying, pausedTime == nil else { return 0 }
        return envelope.level(at: currentPlaybackTime())
    }

    func setVolume(_ volume: Float) {
        playerNode?.volume = volume
    }

    var volume: Float {
        playerNode?.volume ?? 1.0
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
        envelope.begin(sampleRate: sampleRate)
        envelope.append(samples: samples)
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

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)

        do {
            try engine.start()
        } catch {
            Log.speech.error("Failed to start audio engine for streaming: \(error)")
            return
        }

        audioEngine = engine
        playerNode = player
        player.volume = 1.0
        streamingFormat = format
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        totalScheduledSamples = 0
        streamingSampleRate = sampleRate
        envelope.begin(sampleRate: sampleRate)
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

        totalScheduledSamples += samples.count
        envelope.append(samples: samples)
        pendingBufferCount += 1
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.pendingBufferCount -= 1
                if self.streamFinished && self.pendingBufferCount <= 0 {
                    self.isPlaying = false
                    self.onPlaybackFinished?()
                }
            }
        }

        // Start playback on first chunk — unless paused before audio arrived.
        if !playerStarted && pausedTime == nil {
            node.play()
            playerStarted = true
        }
    }

    func finishStreaming() {
        streamFinished = true

        // If all buffers already drained, finish now
        if pendingBufferCount <= 0 {
            isPlaying = false
            onPlaybackFinished?()
        }
        // Otherwise the last buffer's completion callback handles it
    }

    // MARK: - Pause / resume

    func pause() {
        guard pausedTime == nil else { return }
        pausedTime = currentPlaybackTime()
        playerNode?.pause()
        isPlaying = false
    }

    func resume() {
        guard pausedTime != nil else { return }
        pausedTime = nil
        if let node = playerNode {
            node.play()
            playerStarted = true
            isPlaying = true
        }
    }

    // MARK: - Stop

    func stop() {
        playerNode?.volume = 1.0
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        totalScheduledSamples = 0
        streamingSampleRate = 0
        envelope.reset()
        isPlaying = false
    }
}
