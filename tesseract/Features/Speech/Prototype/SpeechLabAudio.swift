//
//  SpeechLabAudio.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  A take's audio, its waveform peaks, WAV/M4A encoding, export and
//  drag-out, and a replay player that can seek and change speed — the
//  pieces "save the audio" and "light editing" would need.
//

import AVFoundation
import Accelerate
import AppKit
import CoreTransferable
import Observation
import SwiftUI
import UniformTypeIdentifiers
import os

// MARK: - Audio

/// Immutable once built: the samples, and the waveform drawn from them.
nonisolated final class TakeAudio: Sendable {
    let samples: [Float]
    let sampleRate: Int
    /// 0…1 loudness per 50 ms.
    let peaks: [Float]

    init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.peaks = TakePeaks.peaks(
            of: samples[...], binSize: TakePeaks.binSize(sampleRate: sampleRate))
    }

    var duration: TimeInterval { Double(samples.count) / Double(max(sampleRate, 1)) }
}

nonisolated enum TakePeaks {
    static let binsPerSecond = 20.0

    static func binSize(sampleRate: Int) -> Int {
        max(1, Int(Double(sampleRate) / binsPerSecond))
    }

    static func peaks(of samples: ArraySlice<Float>, binSize: Int) -> [Float] {
        guard !samples.isEmpty, binSize > 0 else { return [] }
        var result: [Float] = []
        result.reserveCapacity(samples.count / binSize + 1)
        samples.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            var start = 0
            while start < buffer.count {
                let count = min(binSize, buffer.count - start)
                var peak: Float = 0
                vDSP_maxmgv(base + start, 1, &peak, vDSP_Length(count))
                result.append(level(peak))
                start += count
            }
        }
        return result
    }

    /// Peak amplitude to a 0…1 display level over a 48 dB range.
    static func level(_ peak: Float) -> Float {
        guard peak > 0 else { return 0 }
        let decibels = 20 * log10(peak)
        return min(max((decibels + 48) / 48, 0), 1)
    }

    /// `peaks` squeezed or stretched to `count` bars (max per bucket).
    static func resample(_ peaks: [Float], to count: Int) -> [Float] {
        guard count > 0, !peaks.isEmpty else { return [] }
        if peaks.count == count { return peaks }
        var result = [Float](repeating: 0, count: count)
        let scale = Double(peaks.count) / Double(count)
        for i in 0..<count {
            let lower = Int(Double(i) * scale)
            let upper = max(lower + 1, min(peaks.count, Int(Double(i + 1) * scale)))
            var value: Float = 0
            for j in lower..<min(upper, peaks.count) { value = max(value, peaks[j]) }
            result[i] = value
        }
        return result
    }
}

// MARK: - Light editing (whole-take, non-destructive: returns new samples)

nonisolated enum TakeEditing {
    /// Drops leading and trailing silence, keeping `pad` seconds.
    static func trimmingSilence(
        _ samples: [Float], sampleRate: Int, threshold: Float = 0.01, pad: Double = 0.12
    ) -> [Float] {
        guard let first = samples.firstIndex(where: { abs($0) > threshold }),
            let last = samples.lastIndex(where: { abs($0) > threshold })
        else { return samples }
        let padding = Int(Double(sampleRate) * pad)
        let lower = max(0, first - padding)
        let upper = min(samples.count, last + padding)
        return Array(samples[lower..<upper])
    }

    /// Scales so the loudest sample sits at `target` (−1 dBFS by default).
    static func normalized(_ samples: [Float], target: Float = 0.89) -> [Float] {
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        guard peak > 0.0001 else { return samples }
        var gain = target / peak
        var result = [Float](repeating: 0, count: samples.count)
        vDSP_vsmul(samples, 1, &gain, &result, 1, vDSP_Length(samples.count))
        return result
    }

    /// Short fades so a cut never clicks.
    static func faded(_ samples: [Float], sampleRate: Int, seconds: Double = 0.02) -> [Float] {
        var result = samples
        let length = min(Int(Double(sampleRate) * seconds), result.count / 2)
        guard length > 1 else { return result }
        for i in 0..<length {
            let gain = Float(i) / Float(length)
            result[i] *= gain
            result[result.count - 1 - i] *= gain
        }
        return result
    }

    static func silence(seconds: Double, sampleRate: Int) -> [Float] {
        [Float](repeating: 0, count: max(0, Int(seconds * Double(sampleRate))))
    }
}

// MARK: - Encoding

nonisolated enum TakeEncoding {
    /// 16-bit PCM mono WAV, in memory.
    static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        let byteCount = samples.count * 2
        var data = Data(capacity: 44 + byteCount)
        func put<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        put(UInt32(36 + byteCount))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        put(UInt32(16))
        put(UInt16(1))
        put(UInt16(1))
        put(UInt32(sampleRate))
        put(UInt32(sampleRate * 2))
        put(UInt16(2))
        put(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        put(UInt32(byteCount))
        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in samples.indices {
            pcm[i] = Int16(max(-1, min(1, samples[i])) * 32_767)
        }
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    /// AAC in an .m4a container (what Voice Memos writes).
    static func writeM4A(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 96_000,
        ]
        let file = try AVAudioFile(
            forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32,
            interleaved: false)
        guard
            let buffer = AudioConverter.makeMonoFloat32Buffer(
                samples, sampleRate: Double(sampleRate))
        else { throw CocoaError(.fileWriteUnknown) }
        try file.write(from: buffer)
    }
}

// MARK: - Export

nonisolated enum TakeExportFormat: String, CaseIterable, Identifiable, Sendable {
    case wav, m4a
    var id: String { rawValue }
    var label: String { self == .wav ? "WAV" : "M4A (AAC)" }
    var contentType: UTType { self == .wav ? .wav : .mpeg4Audio }
}

enum TakeExport {
    /// Save panel, then write. Returns the file written.
    @discardableResult
    static func save(
        samples: [Float], sampleRate: Int, suggestedName: String, format: TakeExportFormat
    ) -> URL? {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(suggestedName).\(format.rawValue)"
        panel.allowedContentTypes = [format.contentType]
        panel.canCreateDirectories = true
        panel.title = "Export Audio"
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        do {
            try write(samples: samples, sampleRate: sampleRate, format: format, to: url)
            return url
        } catch {
            Log.speech.error("[SpeechLab] export failed: \(error.localizedDescription)")
            NSSound.beep()
            return nil
        }
    }

    static func save(_ take: SpeechTake, format: TakeExportFormat) {
        save(
            samples: take.audio.samples, sampleRate: take.audio.sampleRate,
            suggestedName: fileStem(for: take.title), format: format)
    }

    nonisolated static func write(
        samples: [Float], sampleRate: Int, format: TakeExportFormat, to url: URL
    ) throws {
        switch format {
        case .wav:
            try TakeEncoding.wav(samples, sampleRate: sampleRate).write(to: url, options: .atomic)
        case .m4a:
            try? FileManager.default.removeItem(at: url)
            try TakeEncoding.writeM4A(samples, sampleRate: sampleRate, to: url)
        }
    }

    /// A WAV in the temporary directory, for drag-out and sharing.
    nonisolated static func temporaryWAV(for take: SpeechTake) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Tesseract Speech", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(fileStem(for: take.title)).wav")
        try TakeEncoding.wav(take.audio.samples, sampleRate: take.audio.sampleRate)
            .write(to: url, options: .atomic)
        return url
    }

    nonisolated static func fileStem(for title: String) -> String {
        let cleaned = title.components(separatedBy: CharacterSet(charactersIn: "/:\\?%*|\"<>"))
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        return cleaned.isEmpty ? "Speech" : String(cleaned.prefix(60))
    }
}

// MARK: - Captions

/// SubRip captions from a take's segment timings: short lines, broken after
/// punctuation where possible, timed by the same proportional model the
/// read-along uses.
nonisolated enum TakeCaptions {
    static func srt(for take: SpeechTake, maxWords: Int = 10) -> String {
        var cues: [(start: TimeInterval, end: TimeInterval, text: String)] = []
        for segment in take.segments {
            let words = segment.timeline.words
            let total = max(segment.timeline.totalCharCount, 1)
            let end = segment.end ?? take.duration
            let span = max(end - segment.base, 0.01)
            var start = 0
            while start < words.count {
                var stop = min(start + maxWords, words.count)
                if let punct = (start..<stop).last(where: { i in
                    i > start + 3
                        && [".", "!", "?", ",", ";", ":"].contains(
                            words[i].text.last.map(String.init) ?? "")
                }) {
                    stop = punct + 1
                }
                let first = words[start]
                let last = words[stop - 1]
                let t0 = segment.base + span * Double(first.charOffset) / Double(total)
                let t1 =
                    segment.base + span * Double(last.charOffset + last.charCount) / Double(total)
                cues.append(
                    (t0, max(t1, t0 + 0.3), words[start..<stop].map(\.text).joined(separator: " ")))
                start = stop
            }
        }
        return cues.enumerated().map { index, cue in
            "\(index + 1)\n\(stamp(cue.start)) --> \(stamp(cue.end))\n\(cue.text)\n"
        }.joined(separator: "\n")
    }

    private static func stamp(_ seconds: TimeInterval) -> String {
        let ms = Int((seconds * 1000).rounded())
        return String(
            format: "%02d:%02d:%02d,%03d", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60,
            ms % 1000)
    }
}

extension TakeExport {
    static func saveCaptions(_ take: SpeechTake) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(fileStem(for: take.title)).srt"
        panel.allowedContentTypes = [UTType(filenameExtension: "srt") ?? .plainText]
        panel.title = "Export Captions"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? TakeCaptions.srt(for: take).write(to: url, atomically: true, encoding: .utf8)
    }
}

/// Drag a take out of the app as a WAV file (to Finder, Mail, a DAW…).
nonisolated struct TakeFile: Transferable {
    let take: SpeechTake

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .wav) { file in
            SentTransferredFile(try TakeExport.temporaryWAV(for: file.take))
        }
    }
}

// MARK: - Replay player

/// Replays a finished take from memory: seek, speed, and the clock the
/// read-along highlights from. Separate from live speech, which it stops.
@Observable @MainActor
final class SpeechTakePlayer: NSObject, AVAudioPlayerDelegate {
    private(set) var takeID: UUID?
    private(set) var isPlaying = false
    private(set) var duration: TimeInterval = 0
    var rate: Float = 1.0 {
        didSet { audioPlayer?.rate = rate }
    }

    @ObservationIgnored private var audioPlayer: AVAudioPlayer?
    @ObservationIgnored private var take: SpeechTake?
    /// Called before any sound starts (the lab stops live speech).
    @ObservationIgnored var onWillPlay: (() -> Void)?
    /// The read-along follows the replay clock; nil clears it.
    @ObservationIgnored var onClockChange: ((SpeechTake?, (() -> TimeInterval)?) -> Void)?

    var currentTime: TimeInterval { audioPlayer?.currentTime ?? 0 }

    func isCurrent(_ take: SpeechTake) -> Bool { takeID == take.id }

    func play(_ take: SpeechTake, from start: TimeInterval? = nil) {
        onWillPlay?()
        if takeID != take.id || audioPlayer == nil {
            teardown()
            let data = TakeEncoding.wav(take.audio.samples, sampleRate: take.audio.sampleRate)
            guard let player = try? AVAudioPlayer(data: data) else { return }
            player.enableRate = true
            player.rate = rate
            player.delegate = self
            player.prepareToPlay()
            audioPlayer = player
            self.take = take
            takeID = take.id
            duration = player.duration
        }
        guard let audioPlayer else { return }
        if let start {
            audioPlayer.currentTime = max(0, min(start, duration))
        } else if audioPlayer.currentTime >= duration - 0.05 {
            audioPlayer.currentTime = 0
        }
        audioPlayer.play()
        isPlaying = true
        onClockChange?(take, { [weak self] in self?.audioPlayer?.currentTime ?? 0 })
    }

    func toggle(_ take: SpeechTake) {
        if isCurrent(take), isPlaying {
            pause()
        } else {
            play(take)
        }
    }

    func pause() {
        audioPlayer?.pause()
        isPlaying = false
    }

    func seek(to time: TimeInterval) {
        audioPlayer?.currentTime = max(0, min(time, duration))
    }

    func skip(by seconds: TimeInterval) {
        seek(to: currentTime + seconds)
    }

    func stop() {
        guard audioPlayer != nil else { return }
        teardown()
        onClockChange?(nil, nil)
    }

    private func teardown() {
        audioPlayer?.stop()
        audioPlayer?.delegate = nil
        audioPlayer = nil
        take = nil
        takeID = nil
        isPlaying = false
        duration = 0
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.isPlaying = false
            self.audioPlayer?.currentTime = 0
            self.onClockChange?(nil, nil)
        }
    }
}
