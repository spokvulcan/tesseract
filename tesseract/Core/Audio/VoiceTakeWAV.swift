//
//  VoiceTakeWAV.swift
//  tesseract
//
//  Deterministic in-memory WAV codec for the **Native Audio Turn**'s
//  persisted take: 16-bit PCM, mono, one canonical byte layout. The bytes
//  are the take's cache identity — the prefix cache digests them, and a
//  reopened conversation must re-encode to the identical bytes — so this
//  codec is deliberately minimal and fixed. Not a general WAV reader: it
//  decodes only what it encodes.
//

import Foundation

nonisolated enum VoiceTakeWAV {

    /// Bytes per sample of the fixed 16-bit PCM encoding.
    private static let bytesPerSample = 2
    private static let headerSize = 44

    /// Encode mono float samples ([-1, 1]) as a canonical RIFF/WAVE blob.
    /// Same samples in → same bytes out, always — the digest depends on it.
    static func encode(samples: [Float], sampleRate: Int) -> Data {
        let dataSize = samples.count * bytesPerSample
        var out = Data(capacity: headerSize + dataSize)

        func append(_ string: String) { out.append(contentsOf: Array(string.utf8)) }
        func append32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { out.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { out.append(contentsOf: $0) }
        }

        append("RIFF")
        append32(UInt32(36 + dataSize))
        append("WAVE")
        append("fmt ")
        append32(16)  // fmt chunk size
        append16(1)  // PCM
        append16(1)  // mono
        append32(UInt32(sampleRate))
        append32(UInt32(sampleRate * bytesPerSample))  // byte rate
        append16(UInt16(bytesPerSample))  // block align
        append16(16)  // bits per sample
        append("data")
        append32(UInt32(dataSize))

        var pcm = [Int16](repeating: 0, count: samples.count)
        for (index, sample) in samples.enumerated() {
            let clamped = max(-1.0, min(1.0, sample))
            pcm[index] = Int16(clamped * Float(Int16.max))
        }
        pcm.withUnsafeBytes { out.append(contentsOf: $0) }
        return out
    }

    /// Decode a blob this codec produced. Returns `nil` for anything else —
    /// a persisted take that doesn't match the canonical shape is treated as
    /// undecodable, like a corrupt image attachment.
    static func decode(_ data: Data) -> (samples: [Float], sampleRate: Int)? {
        // Rebase: slices carry their parent's indices, and the reads below
        // use absolute offsets.
        let data = Data(data)
        guard data.count > headerSize else { return nil }

        func read32(_ offset: Int) -> UInt32 {
            var value: UInt32 = 0
            withUnsafeMutableBytes(of: &value) {
                $0.copyBytes(from: data.subdata(in: offset..<offset + 4))
            }
            return UInt32(littleEndian: value)
        }
        func read16(_ offset: Int) -> UInt16 {
            var value: UInt16 = 0
            withUnsafeMutableBytes(of: &value) {
                $0.copyBytes(from: data.subdata(in: offset..<offset + 2))
            }
            return UInt16(littleEndian: value)
        }
        func tag(_ offset: Int) -> String? {
            String(bytes: data.subdata(in: offset..<offset + 4), encoding: .ascii)
        }

        guard tag(0) == "RIFF", tag(8) == "WAVE", tag(12) == "fmt ",
            read16(20) == 1,  // PCM
            read16(22) == 1,  // mono
            read16(34) == 16,  // 16-bit
            tag(36) == "data"
        else { return nil }

        let sampleRate = Int(read32(24))
        let dataSize = Int(read32(40))
        guard sampleRate > 0, dataSize > 0,
            data.count >= headerSize + dataSize,
            dataSize % bytesPerSample == 0
        else { return nil }

        let sampleCount = dataSize / bytesPerSample
        var samples = [Float](repeating: 0, count: sampleCount)
        data.subdata(in: headerSize..<headerSize + dataSize).withUnsafeBytes { raw in
            let pcm = raw.bindMemory(to: Int16.self)
            for index in 0..<sampleCount {
                samples[index] = Float(Int16(littleEndian: pcm[index])) / Float(Int16.max)
            }
        }
        return (samples, sampleRate)
    }
}
