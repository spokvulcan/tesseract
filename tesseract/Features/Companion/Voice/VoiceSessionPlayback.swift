//
//  VoiceSessionPlayback.swift
//  tesseract
//
//  The voice session's playback sink (Dual-Path Playback, ADR-0041): an
//  `AudioPlayback` that schedules onto the VPIO capture engine's *persistent*
//  player node under its voice hold, so the reply renders as the capture
//  unit's own voice stream — it escapes the recording duck other audio
//  suffers under an open mic, and per-utterance work never edits the engine
//  graph (Apple's own VP sample wires all players before start). When the
//  host can't hand out the node (hold not wired, render side refused, engine
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
    // push-based scheduling, with the node owned by the host engine.
    private var node: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var streamFinished = false
    private var pendingBufferCount = 0
    private var playerStarted = false
    private var pausedTime: TimeInterval?
    private var totalScheduledSamples = 0
    private var streamingSampleRate = 0
    /// True from a hosted `startStreaming` until the audio drains or stops —
    /// the `playbackLevel` gate, mirroring the manager's `isPlaying`.
    private var hostedPlaying = false
    /// Guards stale buffer-completion callbacks (a stopped node flushes its
    /// handlers) against the counters of a newer streaming session.
    private var streamEpoch = 0

    // The scheduled-audio loudness timeline behind `playbackLevel()` — the
    // Echo Floor's far-end signal reads one envelope domain on both paths.
    private var envelope = PlaybackEnvelope()

    private var usingFallback = false

    /// Master gain on the hosted reply. macOS VP's residual-echo suppressor
    /// clamps the near end (the owner's mic) harder the louder the unit's own
    /// voice stream plays — at full volume the mic metered ~0 for whole
    /// replies (field 2026-07-18) and barging required shouting. The gain
    /// buys double-talk headroom so normal speech competes with the echo
    /// again; raise it only against mic-liveness telemetry
    /// (`voice.energy-sample` level p50 while speaking).
    private static let hostedGain: Float = 0.5
    /// The logical volume the session asked for (ducks, fades) — reported by
    /// `volume` so fades compute in the domain the controller sets; the node
    /// renders it scaled by `hostedGain`.
    private var requestedVolume: Float = 1.0

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

        if let player = host.hostedVoicePlayer(sampleRate: sampleRate) {
            requestedVolume = 1.0
            player.volume = Self.hostedGain
            node = player
            streamingFormat = format
            streamingSampleRate = sampleRate
            usingFallback = false
            hostedPlaying = true
            envelope.begin(sampleRate: sampleRate)
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
        envelope.append(samples: samples)
        pendingBufferCount += 1
        let epoch = streamEpoch
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.streamEpoch == epoch else { return }
                self.pendingBufferCount -= 1
                if self.streamFinished && self.pendingBufferCount <= 0 {
                    self.hostedPlaying = false
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
            hostedPlaying = false
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

    func playbackLevel() -> Float {
        if usingFallback { return fallback.playbackLevel() }
        guard hostedPlaying, pausedTime == nil else { return 0 }
        return envelope.level(at: currentPlaybackTime())
    }

    func setVolume(_ volume: Float) {
        if usingFallback {
            fallback.setVolume(volume)
        } else {
            requestedVolume = volume
            node?.volume = volume * Self.hostedGain
        }
    }

    var volume: Float {
        usingFallback ? fallback.volume : requestedVolume
    }

    func stop() {
        streamEpoch += 1
        if let node {
            // The node is the host's — stop flushes its scheduled buffers
            // and resets the duck; the graph is never touched here.
            node.stop()
            node.volume = Self.hostedGain
        }
        requestedVolume = 1.0
        node = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        totalScheduledSamples = 0
        streamingSampleRate = 0
        hostedPlaying = false
        envelope.reset()
        usingFallback = false
        fallback.stop()
    }

    // MARK: - Host invalidation

    /// The held engine died or rebuilt under the persistent node (device
    /// change, wedge teardown). The node is gone with it — end the utterance
    /// so the voice session recovers to listening instead of waiting on
    /// buffer callbacks that will never fire.
    private func hostEngineInvalidated() {
        guard node != nil else { return }
        streamEpoch += 1
        requestedVolume = 1.0
        node = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        pausedTime = nil
        hostedPlaying = false
        envelope.reset()
        usingFallback = false
        Log.speech.error("Voice playback invalidated by engine rebuild — ending utterance")
        onPlaybackFinished?()
    }
}
