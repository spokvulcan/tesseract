import Foundation
import Testing

@testable import Tesseract_Agent

/// The stepped pool arm's slice-end selection (PRD #173): each granted
/// prefill step consumes one slice, and the executor captures checkpoints
/// only strictly *inside* the text it was handed — a boundary-relative
/// offset 0 is filtered and the tail drain stops short of the end. So a
/// slice must never end exactly on a checkpoint offset: the seam extends
/// past it, keeping the capture interior, exactly as the monolithic arm
/// (one call over the whole suffix) would have captured it.
struct ServerCompletionSteppedSliceTests {

    private func sliceEnd(
        covered: Int,
        total: Int,
        chunk: Int,
        baseOffset: Int = 100,
        checkpointOffsets: [Int] = []
    ) -> Int {
        ServerCompletion.steppedSliceEnd(
            covered: covered,
            total: total,
            chunk: chunk,
            baseOffset: baseOffset,
            checkpoints: Dictionary(
                uniqueKeysWithValues: checkpointOffsets.map { ($0, .leaf) }
            )
        )
    }

    @Test func plainSliceEndsAtOneChunk() {
        #expect(sliceEnd(covered: 0, total: 2000, chunk: 512) == 512)
        #expect(sliceEnd(covered: 512, total: 2000, chunk: 512) == 1024)
    }

    @Test func finalSliceEndsAtTotal() {
        #expect(sliceEnd(covered: 1536, total: 2000, chunk: 512) == 2000)
        #expect(sliceEnd(covered: 0, total: 300, chunk: 512) == 300)
    }

    @Test func interiorCheckpointDoesNotMoveTheSeam() {
        // Checkpoint strictly inside the slice — the executor's own
        // checkpoint-aware loop lands on it; the seam stays put.
        #expect(
            sliceEnd(
                covered: 0, total: 2000, chunk: 512,
                checkpointOffsets: [100 + 200]
            ) == 512)
    }

    @Test func checkpointExactlyOnTheSeamExtendsTheSlicePastIt() {
        #expect(
            sliceEnd(
                covered: 0, total: 2000, chunk: 512,
                checkpointOffsets: [100 + 512]
            ) == 513)
        // Later slice: seam at covered + chunk.
        #expect(
            sliceEnd(
                covered: 512, total: 2000, chunk: 512,
                checkpointOffsets: [100 + 1024]
            ) == 1025)
    }

    @Test func consecutiveSeamCheckpointsAllStayInterior() {
        #expect(
            sliceEnd(
                covered: 0, total: 2000, chunk: 512,
                checkpointOffsets: [100 + 512, 100 + 513]
            ) == 514)
    }

    @Test func seamCheckpointExtensionMayReachTotal() {
        // Extension runs into the prompt end: the slice becomes the final
        // one, and the executor's keep-back still captures a checkpoint at
        // total-1 while preserving the iterator's prime token.
        #expect(
            sliceEnd(
                covered: 0, total: 513, chunk: 512,
                checkpointOffsets: [100 + 512]
            ) == 513)
    }
}
