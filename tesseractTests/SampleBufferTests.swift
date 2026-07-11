//
//  SampleBufferTests.swift
//  tesseractTests
//
//  Pins the chunked `SampleBuffer` (audit #285 item 8): a capture that
//  outgrows the reserved chunk seals it and starts a fresh one — sample order
//  is preserved across seals, and nothing is lost at the boundary. Pure value
//  tests; no audio engine.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SampleBufferTests {

    @Test
    func roundTripsSamplesInOrderWithinOneChunk() {
        let buffer = SampleBuffer()
        buffer.reserveCapacity(16)

        buffer.append([1, 2, 3])
        buffer.append([4, 5])

        #expect(buffer.getAndClear() == [1, 2, 3, 4, 5])
        // Cleared: a second read is empty.
        #expect(buffer.getAndClear() == [])
    }

    @Test
    func preservesOrderAcrossChunkSeals() {
        let buffer = SampleBuffer()
        buffer.reserveCapacity(4)

        // 3 + 3 + 3 samples against a 4-sample chunk: two seals happen (the
        // second and third appends both overflow the current chunk).
        buffer.append([1, 2, 3])
        buffer.append([4, 5, 6])
        buffer.append([7, 8, 9])

        #expect(buffer.getAndClear() == [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    @Test
    func accumulatesUnchunkedBeforeReserveCapacity() {
        // The settings meter path never reserves; plain growth must still work.
        let buffer = SampleBuffer()
        buffer.append([1, 2])
        buffer.append([3])
        #expect(buffer.getAndClear() == [1, 2, 3])
    }

    @Test
    func clearDropsSealedChunksAndCurrent() {
        let buffer = SampleBuffer()
        buffer.reserveCapacity(2)
        buffer.append([1, 2])
        buffer.append([3, 4])  // seals the first chunk

        buffer.clear()

        #expect(buffer.getAndClear() == [])
    }
}
