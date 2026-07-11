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

    private(set) var totalScheduledSamples: Int = 0
    private var streamingSampleRate: Int = 0

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    // MARK: - Diagnostics dump

    // Non-nil while a streaming session captures diagnostics
    // (per `PlaybackDiagnosticsPolicy` at `startStreaming`). The dump value
    // owns the encoding; this adapter only feeds chunks and picks the dir.
    private var diagnosticsDump: PlaybackDiagnosticsDump?
    private var diagnosticsStreamStart: CFAbsoluteTime = 0
    private var diagnosticsOutputDir: URL?

    // MARK: - Playback time tracking

    var totalScheduledDuration: TimeInterval {
        guard streamingSampleRate > 0 else { return 0 }
        return Double(totalScheduledSamples) / Double(streamingSampleRate)
    }

    func currentPlaybackTime() -> TimeInterval {
        guard let node = playerNode,
            let nodeTime = node.lastRenderTime,
            let playerTime = node.playerTime(forNodeTime: nodeTime)
        else {
            return 0
        }
        return Double(playerTime.sampleTime) / playerTime.sampleRate
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

    func startStreaming(sampleRate: Int, diagnostics: PlaybackDiagnosticsPolicy) {
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
        streamingFormat = format
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        totalScheduledSamples = 0
        streamingSampleRate = sampleRate
        isPlaying = true

        if diagnostics == .default {
            let dir = DebugPaths.root
                .appendingPathComponent(DebugPaths.timestamp())
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                Log.speech.info("Created debug dir: \(dir.path)")
            } catch {
                Log.speech.error(
                    "Failed to create debug dir \(dir.path): \(error.localizedDescription)")
            }
            diagnosticsOutputDir = dir
            diagnosticsDump = PlaybackDiagnosticsDump(sampleRate: sampleRate)
            diagnosticsStreamStart = CFAbsoluteTimeGetCurrent()
            Log.speech.info("Debug dump enabled → \(dir.path)")
        }

        Log.speech.info("Started streaming at \(sampleRate)Hz (push-based AVAudioPlayerNode)")
    }

    func appendChunk(samples: [Float]) {
        guard let node = playerNode, let format = streamingFormat else { return }
        guard !samples.isEmpty else { return }

        diagnosticsDump?.appendChunk(
            samples, arrivalTime: CFAbsoluteTimeGetCurrent() - diagnosticsStreamStart)

        // Create and schedule a buffer for this chunk
        guard let buffer = AudioConverter.makeMonoFloat32Buffer(samples, format: format)
        else {
            Log.speech.error("Failed to create PCM buffer for chunk")
            return
        }

        totalScheduledSamples += samples.count
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

        // Start playback on first chunk
        if !playerStarted {
            node.play()
            playerStarted = true
        }
    }

    func finishStreaming() {
        streamFinished = true

        if let dump = diagnosticsDump, let dir = diagnosticsOutputDir {
            Log.speech.info(
                "Writing debug dump: \(dump.chunks.count) chunks → \(dir.path)")
            dump.write(to: dir)
        }

        // If all buffers already drained, finish now
        if pendingBufferCount <= 0 {
            isPlaying = false
            onPlaybackFinished?()
        }
        // Otherwise the last buffer's completion callback handles it
    }

    // MARK: - Stop

    func stop() {
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        streamingFormat = nil
        streamFinished = false
        pendingBufferCount = 0
        playerStarted = false
        totalScheduledSamples = 0
        streamingSampleRate = 0
        isPlaying = false
        diagnosticsDump = nil
        diagnosticsOutputDir = nil
    }
}
