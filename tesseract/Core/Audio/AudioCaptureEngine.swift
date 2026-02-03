//
//  AudioCaptureEngine.swift
//  tesseract
//

import Foundation
import Combine
import AVFoundation
import Accelerate

// Thread-safe sample storage - nonisolated for real-time audio thread access
nonisolated final class SampleBuffer: @unchecked Sendable {
    private var samples: [Float] = []
    private let lock = NSLock()

    func append(_ newSamples: [Float]) {
        lock.lock()
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }

    func getAndClear() -> [Float] {
        lock.lock()
        let result = samples
        samples.removeAll()
        lock.unlock()
        return result
    }

    func reserveCapacity(_ capacity: Int) {
        lock.lock()
        samples.reserveCapacity(capacity)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }
}

// Thread-safe audio level storage for real-time callback - nonisolated for audio thread access
nonisolated final class AudioLevelRelay: @unchecked Sendable {
    private var _level: Float = 0
    private let lock = NSLock()

    var level: Float {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _level
        }
        set {
            lock.lock()
            _level = newValue
            lock.unlock()
        }
    }
}

@MainActor
final class AudioCaptureEngine: ObservableObject {
    @Published private(set) var isCapturing = false
    @Published private(set) var audioLevel: Float = 0

    private var audioEngine: AVAudioEngine?
    private let sampleBuffer = SampleBuffer()
    private var captureStartTime: Date?
    private let levelRelay = AudioLevelRelay()
    private var levelUpdateTimer: Timer?

    private let targetSampleRate: Double = 16000  // WhisperKit requirement
    private var inputSampleRate: Double = 48000
    private let bufferSize: AVAudioFrameCount = 1024

    init() {}

    func startCapture() throws {
        guard !isCapturing else { return }

        // Check microphone permission first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        guard authStatus == .authorized else {
            throw DictationError.microphonePermissionDenied
        }

        audioEngine = AVAudioEngine()
        guard let audioEngine else {
            throw DictationError.audioCaptureFailed("Failed to create audio engine")
        }

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        inputSampleRate = inputFormat.sampleRate

        // Create format for our tap
        guard let recordingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw DictationError.audioCaptureFailed("Failed to create recording format")
        }

        sampleBuffer.clear()
        sampleBuffer.reserveCapacity(Int(targetSampleRate * 60))  // Reserve for 60 seconds

        captureStartTime = Date()

        // Install tap with nonisolated handler to avoid MainActor inheritance
        let buffer = sampleBuffer
        let relay = levelRelay
        inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: recordingFormat,
            block: Self.makeAudioTapHandler(buffer: buffer, relay: relay)
        )

        // Start timer to poll audio level on main thread
        levelUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.audioLevel = self.levelRelay.level
            }
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isCapturing = true
        } catch {
            levelUpdateTimer?.invalidate()
            levelUpdateTimer = nil
            inputNode.removeTap(onBus: 0)
            audioEngine.stop()
            self.audioEngine = nil
            throw DictationError.audioCaptureFailed(error.localizedDescription)
        }
    }

    func stopCapture() -> AudioData? {
        guard isCapturing else { return nil }

        levelUpdateTimer?.invalidate()
        levelUpdateTimer = nil
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isCapturing = false
        audioLevel = 0

        let samples = sampleBuffer.getAndClear()
        let duration = captureStartTime.map { Date().timeIntervalSince($0) } ?? 0

        // Resample to 16kHz if needed
        let resampledSamples: [Float]
        if inputSampleRate != targetSampleRate {
            resampledSamples = resample(samples, from: inputSampleRate, to: targetSampleRate)
        } else {
            resampledSamples = samples
        }

        return AudioData(
            samples: resampledSamples,
            sampleRate: targetSampleRate,
            duration: duration
        )
    }

    // MARK: - Private

    /// Creates an audio tap handler that runs on the real-time audio thread.
    /// This is nonisolated to prevent MainActor isolation inheritance.
    nonisolated private static func makeAudioTapHandler(
        buffer: SampleBuffer,
        relay: AudioLevelRelay
    ) -> AVAudioNodeTapBlock {
        return { audioBuffer, _ in
            guard let channelData = audioBuffer.floatChannelData?[0] else { return }
            let frameCount = Int(audioBuffer.frameLength)

            // Calculate RMS for level metering
            var rms: Float = 0
            vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(frameCount))

            // Convert to dB scale (with floor at -60dB)
            let db = 20 * log10(max(rms, 0.001))
            let normalizedLevel = (db + 60) / 60  // Normalize -60dB to 0dB -> 0 to 1

            // Copy samples to thread-safe buffer
            let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))
            buffer.append(samples)

            // Store level in thread-safe relay (polled by timer on main thread)
            relay.level = max(0, min(1, normalizedLevel))
        }
    }

    private func resample(_ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double) -> [Float] {
        let ratio = targetSampleRate / sourceSampleRate
        let outputLength = Int(Double(samples.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        // Linear interpolation resampling
        for i in 0..<outputLength {
            let position = Double(i) / ratio
            let index = Int(position)
            let fraction = Float(position - Double(index))

            if index + 1 < samples.count {
                output[i] = samples[index] * (1 - fraction) + samples[index + 1] * fraction
            } else if index < samples.count {
                output[i] = samples[index]
            }
        }

        return output
    }
}
