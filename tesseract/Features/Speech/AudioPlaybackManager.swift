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
final class AudioPlaybackManager: ObservableObject {
    @Published private(set) var isPlaying = false

    // Audio engine (shared between one-shot and streaming)
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    // Streaming state — progressive chunk scheduling
    private var streamingFormat: AVAudioFormat?
    private var streamFinished = false
    private var accumulatedSamples: [Float] = []
    private var pendingBufferCount = 0
    private var playerStarted = false

    private(set) var totalScheduledSamples: Int = 0
    private var streamingSampleRate: Int = 0

    var onPlaybackFinished: (() -> Void)?

    // MARK: - Debug dump

    var debugDumpDisabled = false
    private var debugDumpEnabled: Bool { !debugDumpDisabled }
    private var debugRawChunks: [[Float]] = []
    private var debugScheduledSamples: [Float] = []
    private var debugChunkTimestamps: [CFAbsoluteTime] = []
    private var debugStreamStartTime: CFAbsoluteTime = 0
    private var debugSampleRate: Int = 0
    private var debugOutputDir: URL?

    // MARK: - Playback time tracking

    var totalScheduledDuration: TimeInterval {
        guard streamingSampleRate > 0 else { return 0 }
        return Double(totalScheduledSamples) / Double(streamingSampleRate)
    }

    func currentPlaybackTime() -> TimeInterval {
        guard let node = playerNode,
              let nodeTime = node.lastRenderTime,
              let playerTime = node.playerTime(forNodeTime: nodeTime) else {
            return 0
        }
        return Double(playerTime.sampleTime) / playerTime.sampleRate
    }

    // MARK: - One-shot playback (existing API)

    func play(samples: [Float], sampleRate: Int) {
        stop()

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            Log.speech.error("Failed to create audio format")
            return
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            Log.speech.error("Failed to create audio buffer")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channelData = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                channelData.update(from: src.baseAddress!, count: samples.count)
            }
        }

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)

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

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
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
        accumulatedSamples = []
        pendingBufferCount = 0
        playerStarted = false
        totalScheduledSamples = 0
        streamingSampleRate = sampleRate
        isPlaying = true

        if debugDumpEnabled {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd_HHmmss"
            formatter.timeZone = .current
            let dir = URL(fileURLWithPath: "/tmp/tesseract-debug")
                .appendingPathComponent(formatter.string(from: Date()))
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                NSLog("[tesseract-debug] Created debug dir: %@", dir.path)
            } catch {
                NSLog("[tesseract-debug] FAILED to create dir %@: %@", dir.path, error.localizedDescription)
            }
            debugOutputDir = dir
            debugRawChunks = []
            debugScheduledSamples = []
            debugChunkTimestamps = []
            debugStreamStartTime = CFAbsoluteTimeGetCurrent()
            debugSampleRate = sampleRate
            Log.speech.info("Debug dump enabled → \(dir.path)")
        }

        Log.speech.info("Started streaming at \(sampleRate)Hz (push-based AVAudioPlayerNode)")
    }

    func appendChunk(samples: [Float]) {
        guard let node = playerNode, let format = streamingFormat else { return }
        guard !samples.isEmpty else { return }

        if debugDumpEnabled {
            debugRawChunks.append(samples)
            debugChunkTimestamps.append(CFAbsoluteTimeGetCurrent() - debugStreamStartTime)
            accumulatedSamples.append(contentsOf: samples)
        }

        // Create and schedule a buffer for this chunk
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            Log.speech.error("Failed to create PCM buffer for chunk")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channelData = buffer.floatChannelData {
            samples.withUnsafeBufferPointer { src in
                channelData[0].update(from: src.baseAddress!, count: samples.count)
            }
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

        if debugDumpEnabled {
            debugScheduledSamples = accumulatedSamples
            writeDebugDump()
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
        accumulatedSamples = []
        pendingBufferCount = 0
        playerStarted = false
        totalScheduledSamples = 0
        streamingSampleRate = 0
        isPlaying = false
        debugOutputDir = nil
    }

    // MARK: - Debug dump

    private func writeDebugDump() {
        guard let dir = debugOutputDir else { return }
        let sampleRate = debugSampleRate

        Log.speech.info("Writing debug dump: \(self.debugRawChunks.count) chunks, \(self.debugScheduledSamples.count) scheduled samples")

        // Write raw chunks
        let rawDir = dir.appendingPathComponent("raw_chunks")
        try? FileManager.default.createDirectory(at: rawDir, withIntermediateDirectories: true)
        for (i, chunk) in debugRawChunks.enumerated() {
            let path = rawDir.appendingPathComponent(String(format: "chunk_%03d.raw", i))
            chunk.withUnsafeBufferPointer { buf in
                let data = Data(buffer: buf)
                try? data.write(to: path)
            }
        }

        // Write full_stream.wav
        writeWAV(samples: debugScheduledSamples, sampleRate: sampleRate, to: dir.appendingPathComponent("full_stream.wav"))

        // Write metadata.json
        var scheduledOffset = 0
        var chunks: [[String: Any]] = []
        for i in 0..<debugRawChunks.count {
            let rawCount = debugRawChunks[i].count
            let entry: [String: Any] = [
                "index": i,
                "rawSamples": rawCount,
                "scheduledOffset": scheduledOffset,
                "scheduledSize": rawCount,
                "arrivalTimeSec": debugChunkTimestamps[i],
            ]
            chunks.append(entry)
            scheduledOffset += rawCount
        }

        let metadata: [String: Any] = [
            "sampleRate": sampleRate,
            "totalScheduledSamples": debugScheduledSamples.count,
            "chunks": chunks,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]) {
            try? jsonData.write(to: dir.appendingPathComponent("metadata.json"))
        }

        NSLog("[tesseract-debug] Dump written: %d chunks, %d samples → %@", debugRawChunks.count, debugScheduledSamples.count, dir.path)
    }

    private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        var data = Data()
        let byteRate = UInt32(sampleRate * 4)  // float32 = 4 bytes
        let dataSize = UInt32(samples.count * 4)
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk — IEEE float
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) })  // IEEE float
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(4).littleEndian) { Array($0) })  // block align
        data.append(contentsOf: withUnsafeBytes(of: UInt16(32).littleEndian) { Array($0) })  // bits per sample

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })
        samples.withUnsafeBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: buf.count * MemoryLayout<Float>.size) { ptr in
                data.append(ptr, count: buf.count * MemoryLayout<Float>.size)
            }
        }

        try? data.write(to: url)
    }
}
