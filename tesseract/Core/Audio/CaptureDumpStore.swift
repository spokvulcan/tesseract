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
//  is logged and swallowed, and the disk work runs off the main actor so a
//  multi-megabyte WAV write never delays the stop → transcribe path.
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
    nonisolated struct Limits: Sendable {
        var maxRecordings: Int = 100
        var maxTotalBytes: Int = 500 * 1024 * 1024
    }

    let directory: URL
    private let limits: Limits

    /// Per-instance tiebreak so rapid saves within one timestamp granule still
    /// order (and thus evict) in save order.
    private var sequence = 0

    /// Tail of the serialized background disk chain. Each save/deleteAll runs
    /// after the previous one, so eviction still sees every earlier write;
    /// filenames are minted on the main actor, so order stays save order.
    private var diskChain: Task<Void, Never>?

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
        let url = directory.appendingPathComponent(fileName(for: capture))
        schedule { [directory, limits] in
            do {
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                try Self.write(capture, to: url)
                Self.enforceLimits(in: directory, limits: limits)
            } catch {
                Log.audio.error("Capture Dump save failed: \(error.localizedDescription)")
            }
        }
    }

    func deleteAll() {
        schedule { [directory] in
            for url in Self.recordings(in: directory) {
                do {
                    try FileManager.default.removeItem(at: url)
                } catch {
                    Log.audio.error("Capture Dump delete failed: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Awaits all disk work scheduled so far — the tests' completion signal.
    func flush() async {
        await diskChain?.value
    }

    // MARK: - Private

    /// Appends `work` to the serialized background chain.
    private func schedule(_ work: @escaping @Sendable () -> Void) {
        diskChain = Task.detached(priority: .utility) { [previous = diskChain] in
            await previous?.value
            work()
        }
    }

    private func fileName(for capture: RawCapture) -> String {
        sequence += 1
        let stamp = Self.timestampFormatter.string(from: Date())
        let rate = Int(capture.sampleRate)
        let vp = capture.voiceProcessed ? "on" : "off"
        return String(format: "capture-%@-%04d_%dHz_vp-%@.wav", stamp, sequence, rate, vp)
    }

    private nonisolated enum CaptureDumpError: LocalizedError {
        case bufferSetupFailed
        var errorDescription: String? { "buffer setup failed" }
    }

    private nonisolated static func write(_ capture: RawCapture, to url: URL) throws {
        guard
            let buffer = AudioConverter.makeMonoFloat32Buffer(
                capture.samples, sampleRate: capture.sampleRate)
        else {
            throw CaptureDumpError.bufferSetupFailed
        }

        let file = try AVAudioFile(forWriting: url, settings: buffer.format.settings)
        try file.write(from: buffer)
    }

    private nonisolated static func recordings(in directory: URL) -> [URL] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.filter { $0.pathExtension == "wav" }
    }

    /// Filenames embed a zero-padded timestamp + sequence, so lexicographic
    /// order is save order.
    private nonisolated static func recordingsOldestFirstWithSizes(
        in directory: URL
    ) -> [(url: URL, bytes: Int)] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return contents.filter { $0.pathExtension == "wav" }
            .map { url in
                (url: url, bytes: (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
            .sorted { $0.url.lastPathComponent < $1.url.lastPathComponent }
    }

    private nonisolated static func enforceLimits(in directory: URL, limits: Limits) {
        var recordings = recordingsOldestFirstWithSizes(in: directory)
        var totalBytes = recordings.reduce(0) { $0 + $1.bytes }

        while recordings.count > limits.maxRecordings
            || (totalBytes > limits.maxTotalBytes && recordings.count > 1)
        {
            let oldest = recordings.removeFirst()
            do {
                try FileManager.default.removeItem(at: oldest.url)
                totalBytes -= oldest.bytes
            } catch {
                Log.audio.error("Capture Dump eviction failed: \(error.localizedDescription)")
                return
            }
        }
    }
}
