//
//  VoiceSessionPlayback.swift
//  tesseract
//
//  The voice session's playback sink (Dual-Path Playback, ADR-0041): an
//  `AudioPlayback` whose player node attaches to the VPIO capture engine
//  under its voice hold, so the reply renders as the capture unit's own
//  stream — it escapes the recording duck other audio suffers under an open
//  mic, and its cancellation is the canceller's own render reference by
//  construction, never dependent on the duck/loopback policy. When the host
//  can't take a node (hold not running, render side refused, engine
//  mid-rebuild) the utterance falls back to a dedicated engine: the reply
//  always plays.
//

import Foundation
import AVFoundation

@MainActor
final class VoiceSessionPlayback: AudioPlayback {

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    private let host: AudioCaptureEngine
    private let fallback = AudioPlaybackManager()

    // Hosted-mode streaming state — mirrors `AudioPlaybackManager`'s
    // push-based scheduling, with the node living on the host engine.
    private var node: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var streamFinished = false
    private var pendingBufferCount = 0
    private var playerStarted = false
    private var pausedTime: TimeInterval?
    private var totalScheduledSamples = 0
    private var streamingSampleRate = 0
    /// Guards stale buffer-completion callbacks (a stopped node flushes its
    /// handlers) against the counters of a newer streaming session.
    private var streamEpoch = 0

    private var usingFallback = false

    init(host: AudioCaptureEngine) {
        self.host = host
        fallback.onPlaybackFinished = { [weak self] in self?.onPlaybackFinished?() }
        host.onVoicePlaybackInvalidated = { [weak self] in self?.hostEngineInvalidated() }
    }

    // MARK: - AudioPlayback

    var totalScheduledDuration: TimeInterval {
        if usingFallback { return fallback.totalScheduledDuration }
        guard streamingSampleRate > 0 else { return 0 }
        return Double(totalScheduledSamples) / Double(streamingSampleRate)
    }

    func play(samples: [Float], sampleRate: Int) {
        startStreaming(sampleRate: sampleRate)
        appendChunk(samples: samples)
        finishStreaming()
    }

    func startStreaming(sampleRate: Int) {
        stop()

        guard
            let format = AudioConverter.monoFloat32Format(sampleRate: Double(sampleRate))
        else {
            Log.speech.error("Voice playback: failed to create streaming format")
            return
        }

        let player = AVAudioPlayerNode()
        if host.attachVoicePlayback(node: player, format: format) {
            node = player
            streamingFormat = format
            streamingSampleRate = sampleRate
            usingFallback = false
            Log.speech.info("Voice reply routed through the VPIO engine (hosted)")
        } else {
            usingFallback = true
            fallback.startStreaming(sampleRate: sampleRate)
            Log.speech.info("Voice reply on the fallback playback engine (host unavailable)")
        }
    }

    func appendChunk(samples: [Float]) {
        if usingFallback {
            fallback.appendChunk(samples: samples)
            return
        }
        guard let node, let format = streamingFormat, !samples.isEmpty else { return }
        guard let buffer = AudioConverter.makeMonoFloat32Buffer(samples, format: format)
        else {
            Log.speech.error("Voice playback: failed to create PCM buffer for chunk")
            return
        }

        totalScheduledSamples += samples.count
        pendingBufferCount += 1
        let epoch = streamEpoch
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.streamEpoch == epoch else { return }
                self.pendingBufferCount -= 1
                if self.streamFinished && self.pendingBufferCount <= 0 {
                    self.onPlaybackFinished?()
                }
            }
        }

        if !playerStarted && pausedTime == nil {
            node.play()
            playerStarted = true
        }
    }

    func finishStreaming() {
        if usingFallback {
            fallback.finishStreaming()
            return
        }
        streamFinished = true
        if pendingBufferCount <= 0 {
            onPlaybackFinished?()
        }
    }

    func pause() {
        if usingFallback {
            fallback.pause()
            return
        }
        guard pausedTime == nil else { return }
        pausedTime = currentPlaybackTime()
        node?.pause()
    }

    func resume() {
        if usingFallback {
            fallback.resume()
            return
        }
        guard pausedTime != nil else { return }
        pausedTime = nil
        if let node {
            node.play()
            playerStarted = true
        }
    }

    func currentPlaybackTime() -> TimeInterval {
        if usingFallback { return fallback.currentPlaybackTime() }
        if let pausedTime { return pausedTime }
        guard let node,
            let nodeTime = node.lastRenderTime,
            let playerTime = node.playerTime(forNodeTime: nodeTime)
        else {
            return 0
        }
        return Double(playerTime.sampleTime) / playerTime.sampleRate
    }

    func stop() {
        streamEpoch += 1
        if let node {
            host.detachVoicePlayback(node: node)
        }
        node = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        totalScheduledSamples = 0
        streamingSampleRate = 0
        usingFallback = false
        fallback.stop()
    }

    // MARK: - Host invalidation

    /// The held engine died or rebuilt under our node (device change, wedge
    /// teardown). The node is gone with it — end the utterance so the voice
    /// session recovers to listening instead of waiting on buffer callbacks
    /// that will never fire.
    private func hostEngineInvalidated() {
        guard node != nil else { return }
        streamEpoch += 1
        node = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        Log.speech.error("Voice playback invalidated by engine rebuild — ending utterance")
        onPlaybackFinished?()
    }
}
