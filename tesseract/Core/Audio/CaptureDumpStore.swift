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
    /// Saves one capture; returns the recording's file name (minted
    /// synchronously — the write itself is background), or `nil` when
    /// nothing will be saved. The name is the **Correction Pair**'s audio
    /// reference.
    @discardableResult
    func save(_ capture: RawCapture) -> String?
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

    /// File names the ring eviction must skip — gold **Correction Pair**
    /// audio (ticket #289). Read on the main actor at save time; the
    /// snapshot rides into the background eviction. `deleteAll()` is not
    /// eviction: an explicit user delete still removes everything.
    private let protectedFileNames: @MainActor () -> Set<String>

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

    init(
        directory: URL,
        limits: Limits = Limits(),
        protectedFileNames: @escaping @MainActor () -> Set<String> = { [] }
    ) {
        self.directory = directory
        self.limits = limits
        self.protectedFileNames = protectedFileNames
    }

    @discardableResult
    func save(_ capture: RawCapture) -> String? {
        guard !capture.samples.isEmpty else { return nil }
        let name = fileName(for: capture)
        let url = directory.appendingPathComponent(name)
        let protected = protectedFileNames()
        schedule { [directory, limits] in
            do {
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                try Self.write(capture, to: url)
                Self.enforceLimits(in: directory, limits: limits, protected: protected)
            } catch {
                Log.audio.error("Capture Dump save failed: \(error.localizedDescription)")
            }
        }
        return name
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

    /// Evicts oldest-first among the *evictable* recordings until the limits
    /// hold (counting every file, protected included). When only protected
    /// files remain over-limit, eviction stops — gold audio outlives the ring.
    private nonisolated static func enforceLimits(
        in directory: URL, limits: Limits, protected: Set<String>
    ) {
        let recordings = recordingsOldestFirstWithSizes(in: directory)
        var totalCount = recordings.count
        var totalBytes = recordings.reduce(0) { $0 + $1.bytes }
        var evictable = recordings.filter { !protected.contains($0.url.lastPathComponent) }

        while !evictable.isEmpty,
            totalCount > limits.maxRecordings
                || (totalBytes > limits.maxTotalBytes && totalCount > 1)
        {
            let oldest = evictable.removeFirst()
            do {
                try FileManager.default.removeItem(at: oldest.url)
                totalCount -= 1
                totalBytes -= oldest.bytes
            } catch {
                Log.audio.error("Capture Dump eviction failed: \(error.localizedDescription)")
                return
            }
        }
    }
}
