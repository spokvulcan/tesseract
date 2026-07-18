//
//  VoiceTakeWAVTests.swift
//  tesseractTests
//
//  The Native Audio Turn's canonical WAV codec. Determinism is the contract
//  under test: the blob's bytes are the take's prefix-cache identity, so
//  the same samples must encode to the same bytes on every render.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct VoiceTakeWAVTests {

    private let tone: [Float] = (0..<1600).map { index in
        sinf(2 * .pi * 440 * Float(index) / 16_000) * 0.5
    }

    @Test func roundTripPreservesShapeAndRate() throws {
        let blob = VoiceTakeWAV.encode(samples: tone, sampleRate: 16_000)
        let decoded = try #require(VoiceTakeWAV.decode(blob))
        #expect(decoded.sampleRate == 16_000)
        #expect(decoded.samples.count == tone.count)
        // 16-bit quantization bounds the round-trip error.
        for (original, restored) in zip(tone, decoded.samples) {
            #expect(abs(original - restored) < 1.0 / 16_384)
        }
    }

    @Test func encodingIsDeterministic() {
        let first = VoiceTakeWAV.encode(samples: tone, sampleRate: 16_000)
        let second = VoiceTakeWAV.encode(samples: tone, sampleRate: 16_000)
        #expect(first == second)
    }

    @Test func samplesOutsideUnitRangeClampInsteadOfWrapping() throws {
        let hot: [Float] = [1.5, -1.5, 0]
        let decoded = try #require(
            VoiceTakeWAV.decode(VoiceTakeWAV.encode(samples: hot, sampleRate: 16_000)))
        #expect(abs(decoded.samples[0] - 1.0) < 0.001)
        #expect(abs(decoded.samples[1] + 1.0) < 0.001)
    }

    @Test func rejectsForeignBlobs() {
        #expect(VoiceTakeWAV.decode(Data()) == nil)
        #expect(VoiceTakeWAV.decode(Data("not a wav at all, but long enough to scan".utf8)) == nil)
        // A truncated header must not crash the decoder.
        let blob = VoiceTakeWAV.encode(samples: tone, sampleRate: 16_000)
        #expect(VoiceTakeWAV.decode(blob.prefix(40)) == nil)
    }

    @Test func rejectsTruncatedPayloads() {
        let blob = VoiceTakeWAV.encode(samples: tone, sampleRate: 16_000)
        #expect(VoiceTakeWAV.decode(blob.prefix(blob.count - 100)) == nil)
    }
}
