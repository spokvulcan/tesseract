//
//  PlaybackDiagnosticsDumpTests.swift
//  tesseractTests
//
//  Byte-level tests for the playback diagnostics encoders — previously a
//  hand-rolled WAV/JSON writer buried inside the AVFoundation playback
//  adapter with no test reach.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct PlaybackDiagnosticsDumpTests {

    private func le32(_ data: Data, at offset: Int) -> UInt32 {
        data.subdata(in: offset..<(offset + 4)).withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
        }
    }

    private func le16(_ data: Data, at offset: Int) -> UInt16 {
        data.subdata(in: offset..<(offset + 2)).withUnsafeBytes {
            UInt16(littleEndian: $0.loadUnaligned(as: UInt16.self))
        }
    }

    private func ascii(_ data: Data, at offset: Int, count: Int) -> String {
        String(decoding: data.subdata(in: offset..<(offset + count)), as: UTF8.self)
    }

    // MARK: - WAV encoding

    @Test
    func wavHeaderIsMonoIEEEFloat() {
        let samples: [Float] = [0, 0.5, -0.5, 1]
        let wav = PlaybackDiagnosticsDump.wavData(samples: samples, sampleRate: 24_000)

        #expect(ascii(wav, at: 0, count: 4) == "RIFF")
        #expect(le32(wav, at: 4) == UInt32(36 + samples.count * 4))
        #expect(ascii(wav, at: 8, count: 4) == "WAVE")
        #expect(ascii(wav, at: 12, count: 4) == "fmt ")
        #expect(le32(wav, at: 16) == 16)  // fmt chunk size
        #expect(le16(wav, at: 20) == 3)  // IEEE float
        #expect(le16(wav, at: 22) == 1)  // mono
        #expect(le32(wav, at: 24) == 24_000)  // sample rate
        #expect(le32(wav, at: 28) == 24_000 * 4)  // byte rate
        #expect(le16(wav, at: 32) == 4)  // block align
        #expect(le16(wav, at: 34) == 32)  // bits per sample
        #expect(ascii(wav, at: 36, count: 4) == "data")
        #expect(le32(wav, at: 40) == UInt32(samples.count * 4))
        #expect(wav.count == 44 + samples.count * 4)
    }

    @Test
    func wavPayloadRoundTripsSamples() {
        let samples: [Float] = [0.25, -1, 0.125]
        let wav = PlaybackDiagnosticsDump.wavData(samples: samples, sampleRate: 16_000)
        let payload = wav.subdata(in: 44..<wav.count)
        let decoded = payload.withUnsafeBytes { raw in
            (0..<samples.count).map {
                Float(
                    bitPattern: UInt32(
                        littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
        #expect(decoded == samples)
    }

    @Test
    func emptyStreamStillEncodesAValidHeader() {
        let wav = PlaybackDiagnosticsDump.wavData(samples: [], sampleRate: 24_000)
        #expect(wav.count == 44)
        #expect(le32(wav, at: 40) == 0)
    }

    // MARK: - Metadata

    @Test
    func metadataOffsetsAccumulateAcrossChunks() {
        var dump = PlaybackDiagnosticsDump(sampleRate: 24_000)
        dump.appendChunk([0, 0, 0], arrivalTime: 0.1)
        dump.appendChunk([0, 0], arrivalTime: 0.4)
        dump.appendChunk([0, 0, 0, 0], arrivalTime: 0.9)

        let meta = dump.metadata
        #expect(meta.sampleRate == 24_000)
        #expect(meta.totalScheduledSamples == 9)
        #expect(meta.chunks.map(\.scheduledOffset) == [0, 3, 5])
        #expect(meta.chunks.map(\.rawSamples) == [3, 2, 4])
        #expect(meta.chunks.map(\.arrivalTimeSec) == [0.1, 0.4, 0.9])
    }

    @Test
    func metadataJSONDecodesBack() throws {
        var dump = PlaybackDiagnosticsDump(sampleRate: 16_000)
        dump.appendChunk([1, 2], arrivalTime: 0.25)

        let json = try #require(dump.metadataJSON())
        let decoded = try JSONDecoder().decode(PlaybackDiagnosticsDump.Metadata.self, from: json)
        #expect(decoded == dump.metadata)
    }

    @Test
    func scheduledStreamIsChunksInArrivalOrder() {
        var dump = PlaybackDiagnosticsDump(sampleRate: 24_000)
        dump.appendChunk([1, 2], arrivalTime: 0)
        dump.appendChunk([3], arrivalTime: 1)
        #expect(dump.scheduledSamples == [1, 2, 3])
        #expect(
            dump.wavData()
                == PlaybackDiagnosticsDump.wavData(
                    samples: [1, 2, 3], sampleRate: 24_000))
    }
}
