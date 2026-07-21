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

    // Hosted-mode streaming, with the node owned by the host engine. The
    // push scheduler's counters, start gate, finish detection, and stream
    // epoch live in the shared value machine (StreamingScheduler / ADR-0054),
    // the same one the dedicated adapter and the in-memory peer drive; this
    // adapter performs only the `AVAudioPlayerNode` calls.
    private var node: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var scheduler = StreamingScheduler()

    /// Adapter-local clock position held while paused — the machine has no
    /// notion of render time.
    private var pausedTime: TimeInterval?

    /// True from a hosted `startStreaming` until the audio drains or stops —
    /// the `playbackLevel` gate, mirroring the manager's `isPlaying`. The
    /// stale-completion epoch guard that used to live here as `streamEpoch`
    /// is now the machine's, shared with the dedicated adapter.
    private var hostedPlaying: Bool { scheduler.hasUndrainedAudio }

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

    /// The one writer of the node's rendered volume: every write goes
    /// through the logical→physical mapping, so no site can forget the
    /// `hostedGain` scale and play the reply at full loudness (the VP
    /// suppressor bug the gain exists to kill).
    private func applyVolume(_ logical: Float) {
        requestedVolume = logical
        node?.volume = logical * Self.hostedGain
    }

    init(host: AudioCaptureEngine) {
        self.host = host
        fallback.onPlaybackFinished = { [weak self] in self?.onPlaybackFinished?() }
        host.onVoicePlaybackInvalidated = { [weak self] in self?.hostEngineInvalidated() }
    }

    // MARK: - AudioPlayback

    var totalScheduledDuration: TimeInterval {
        usingFallback ? fallback.totalScheduledDuration : scheduler.totalScheduledDuration
    }

    func play(samples: [Float], sampleRate: Int) {
        startStreaming(sampleRate: sampleRate)
        appendChunk(samples: samples)
        finishStreaming()
    }

    func startStreaming(sampleRate: Int) {
        stop()

        if let hosted = host.hostedVoicePlayer(sampleRate: sampleRate) {
            node = hosted.node
            applyVolume(1.0)
            // Buffers schedule against the format the host connected the
            // node at — the engine owns the connection, never re-derived.
            streamingFormat = hosted.format
            scheduler.beginStream(sampleRate: sampleRate)
            usingFallback = false
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

        let outcome = scheduler.appendChunk(sampleCount: samples.count)
        envelope.append(samples: samples)
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                // A stopped node flushes its handlers — the machine's epoch
                // guard drops a stale completion before it decrements a newer
                // session's counter.
                if case .finished = self.scheduler.bufferCompleted(epoch: outcome.epoch) {
                    self.onPlaybackFinished?()
                }
            }
        }

        if outcome.startPlayer {
            node.play()
        }
    }

    func finishStreaming() {
        if usingFallback {
            fallback.finishStreaming()
            return
        }
        if case .finishedNow = scheduler.finishStream() {
            onPlaybackFinished?()
        }
    }

    func pause() {
        if usingFallback {
            fallback.pause()
            return
        }
        guard scheduler.pause() else { return }
        pausedTime = currentPlaybackTime()
        node?.pause()
    }

    func resume() {
        if usingFallback {
            fallback.resume()
            return
        }
        guard scheduler.resume() else { return }
        pausedTime = nil
        node?.play()
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
            applyVolume(volume)
        }
    }

    var volume: Float {
        usingFallback ? fallback.volume : requestedVolume
    }

    func stop() {
        // The node is the host's — stop flushes its scheduled buffers and
        // resets the duck; the graph is never touched here.
        node?.stop()
        applyVolume(1.0)
        resetStreamingState()
        fallback.stop()
    }

    /// Zeroes every streaming field back to idle — the one reset both
    /// teardown paths (`stop`, `hostEngineInvalidated`) share, so they
    /// cannot drift as fields are added.
    private func resetStreamingState() {
        scheduler.stop()
        requestedVolume = 1.0
        node = nil
        streamingFormat = nil
        pausedTime = nil
        envelope.reset()
        usingFallback = false
    }

    // MARK: - Host invalidation

    /// The held engine died or rebuilt under the persistent node (device
    /// change, wedge teardown). The node is gone with it — end the utterance
    /// so the voice session recovers to listening instead of waiting on
    /// buffer callbacks that will never fire.
    private func hostEngineInvalidated() {
        guard node != nil else { return }
        resetStreamingState()
        Log.speech.error("Voice playback invalidated by engine rebuild — ending utterance")
        onPlaybackFinished?()
    }
}
