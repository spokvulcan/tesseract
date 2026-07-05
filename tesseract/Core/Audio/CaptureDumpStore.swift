//
//  CaptureDumpStore.swift
//  tesseract
//
//  The **Capture Dump** (PRD #175) — an on-disk ring buffer of recent dictation
//  capture audio, kept so a bad transcription leaves its evidence behind. Each
//  recording is a standard WAV of exactly what the microphone tap delivered
//  (post–Voice Processing when enabled, pre-resample), with the capture
//  conditions encoded in the filename. Bounded by count and total size, oldest
//  evicted first. Diagnostics must never break dictation: every failure in here
//  is logged and swallowed.
//

import AVFoundation
import Foundation

@MainActor
protocol CaptureDumpStoring: AnyObject {
    func save(_ capture: RawCapture)
    func deleteAll()
}

@MainActor
final class CaptureDumpStore: CaptureDumpStoring {
    struct Limits {
        var maxRecordings: Int = 100
        var maxTotalBytes: Int = 500 * 1024 * 1024
    }

    let directory: URL
    private let limits: Limits

    /// Per-instance tiebreak so rapid saves within one timestamp granule still
    /// order (and thus evict) in save order.
    private var sequence = 0

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    init(directory: URL, limits: Limits = Limits()) {
        self.directory = directory
        self.limits = limits
    }

    func save(_ capture: RawCapture) {
        guard !capture.samples.isEmpty else { return }
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent(fileName(for: capture))
            try write(capture, to: url)
            enforceLimits()
        } catch {
            Log.audio.error("Capture Dump save failed: \(error.localizedDescription)")
        }
    }

    func deleteAll() {
        for url in recordingsOldestFirst() {
            do {
                try FileManager.default.removeItem(at: url)
            } catch {
                Log.audio.error("Capture Dump delete failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Private

    private func fileName(for capture: RawCapture) -> String {
        sequence += 1
        let stamp = Self.timestampFormatter.string(from: Date())
        let rate = Int(capture.sampleRate)
        let vp = capture.voiceProcessed ? "on" : "off"
        return String(format: "capture-%@-%04d_%dHz_vp-%@.wav", stamp, sequence, rate, vp)
    }

    private func write(_ capture: RawCapture, to url: URL) throws {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: capture.sampleRate,
                channels: 1,
                interleaved: false
            ),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(capture.samples.count)
            )
        else {
            throw DictationError.audioCaptureFailed("Capture Dump buffer setup failed")
        }

        capture.samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData?[0].update(
                from: source.baseAddress!, count: capture.samples.count)
        }
        buffer.frameLength = AVAudioFrameCount(capture.samples.count)

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: capture.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(
            forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false
        )
        try file.write(from: buffer)
    }

    /// Filenames embed a zero-padded timestamp + sequence, so lexicographic
    /// order is save order.
    private func recordingsOldestFirst() -> [URL] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return contents.filter { $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func enforceLimits() {
        var recordings = recordingsOldestFirst()
        var totalBytes = recordings.reduce(0) { $0 + fileSize($1) }

        while recordings.count > limits.maxRecordings
            || (totalBytes > limits.maxTotalBytes && recordings.count > 1)
        {
            let oldest = recordings.removeFirst()
            let oldestBytes = fileSize(oldest)
            do {
                try FileManager.default.removeItem(at: oldest)
                totalBytes -= oldestBytes
            } catch {
                Log.audio.error("Capture Dump eviction failed: \(error.localizedDescription)")
                return
            }
        }
    }

    private func fileSize(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }
}
