//
//  PlaybackDiagnosticsDump.swift
//  tesseract
//
//  The diagnostic bundle a streaming playback session captures — raw chunks,
//  arrival times, and the full scheduled stream — with pure byte-level
//  encoders (WAV, metadata JSON) so the encoding has a test surface.
//  `write(to:)` is the only file effect. Sibling of
//  `PlaybackDiagnosticsPolicy`; the AVFoundation playback adapter only feeds
//  chunks and picks the output directory. Not the dictation Capture Dump —
//  that is the bounded microphone ring buffer, a different concept.
//

import Foundation

nonisolated struct PlaybackDiagnosticsDump {

    struct ChunkRecord: Codable, Equatable {
        let index: Int
        let rawSamples: Int
        let scheduledOffset: Int
        let scheduledSize: Int
        let arrivalTimeSec: Double
    }

    struct Metadata: Codable, Equatable {
        let sampleRate: Int
        let totalScheduledSamples: Int
        let chunks: [ChunkRecord]
    }

    let sampleRate: Int
    private(set) var chunks: [[Float]] = []
    private(set) var arrivalTimes: [TimeInterval] = []

    init(sampleRate: Int) {
        self.sampleRate = sampleRate
    }

    mutating func appendChunk(_ samples: [Float], arrivalTime: TimeInterval) {
        chunks.append(samples)
        arrivalTimes.append(arrivalTime)
    }

    /// The full scheduled stream — chunks concatenated in arrival order.
    var scheduledSamples: [Float] { chunks.flatMap { $0 } }

    var metadata: Metadata {
        var offset = 0
        var records: [ChunkRecord] = []
        for (index, chunk) in chunks.enumerated() {
            records.append(
                ChunkRecord(
                    index: index,
                    rawSamples: chunk.count,
                    scheduledOffset: offset,
                    scheduledSize: chunk.count,
                    arrivalTimeSec: arrivalTimes[index]
                ))
            offset += chunk.count
        }
        return Metadata(
            sampleRate: sampleRate, totalScheduledSamples: offset, chunks: records)
    }

    func metadataJSON() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? encoder.encode(metadata)
    }

    /// Mono IEEE-float32 WAV of the full scheduled stream.
    func wavData() -> Data {
        Self.wavData(samples: scheduledSamples, sampleRate: sampleRate)
    }

    static func wavData(samples: [Float], sampleRate: Int) -> Data {
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
            guard let base = buf.baseAddress else { return }
            base.withMemoryRebound(
                to: UInt8.self, capacity: buf.count * MemoryLayout<Float>.size
            ) { ptr in
                data.append(ptr, count: buf.count * MemoryLayout<Float>.size)
            }
        }

        return data
    }

    func rawChunkData(at index: Int) -> Data {
        chunks[index].withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// The one file effect: raw chunks, full_stream.wav, metadata.json.
    func write(to dir: URL) {
        let rawDir = dir.appendingPathComponent("raw_chunks")
        try? FileManager.default.createDirectory(at: rawDir, withIntermediateDirectories: true)
        for index in chunks.indices {
            let path = rawDir.appendingPathComponent(String(format: "chunk_%03d.raw", index))
            try? rawChunkData(at: index).write(to: path)
        }
        try? wavData().write(to: dir.appendingPathComponent("full_stream.wav"))
        try? metadataJSON()?.write(to: dir.appendingPathComponent("metadata.json"))
    }
}
