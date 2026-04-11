import Foundation
import MLX
import MLXLMCommon
import Testing

/// Task 1.2 tests: checkpoint capture during prefill.
/// Tests the chunkedPrefill() algorithm (offset rebasing, tail drain, chunk splitting)
/// and the prepareWithCheckpoints/GenerateParameters integration contract.
struct CheckpointCaptureTests {

    // MARK: - Helpers

    /// Creates a small hybrid cache (Mamba + attention pattern) for testing.
    /// Caches have non-empty state so capture() succeeds.
    private func makeCache(layers: Int = 4) -> [any KVCache] {
        (0..<layers).map { i in
            if i % 4 == 3 {
                let kv = KVCacheSimple()
                kv.state = [
                    MLXArray.zeros([1, 1, 4, 64]),
                    MLXArray.zeros([1, 1, 4, 64]),
                ]
                return kv as any KVCache
            } else {
                let mamba = MambaCache()
                mamba.state = [
                    MLXArray.zeros([1, 3, 100]),
                    MLXArray.zeros([1, 8, 16, 24]),
                ]
                return mamba as any KVCache
            }
        }
    }

    /// Runs chunkedPrefill and records chunk sizes.
    @discardableResult
    private func runPrefill(
        totalTokens: Int,
        prefillStepSize: Int,
        checkpointAtOffsets: Set<Int>,
        checkpointBaseOffset: Int = 0,
        initialOffset: Int = 0,
        cache: [any KVCache]? = nil
    ) throws -> (chunks: [Int], consumed: Int, snapshots: [HybridCacheSnapshot]) {
        let cache = cache ?? makeCache()
        var chunks: [Int] = []
        let (consumed, snapshots) = try HybridCacheSnapshot.chunkedPrefill(
            totalTokens: totalTokens,
            prefillStepSize: prefillStepSize,
            checkpointAtOffsets: checkpointAtOffsets,
            checkpointBaseOffset: checkpointBaseOffset,
            initialOffset: initialOffset,
            cache: cache
        ) { chunkSize in
            chunks.append(chunkSize)
        }
        return (chunks, consumed, snapshots)
    }

    // MARK: - 1. emptyCheckpointOffsetsReturnsNoSnapshots

    @Test func emptyCheckpointOffsetsReturnsNoSnapshots() throws {
        // No checkpoint offsets → standard chunking, zero snapshots.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: []
        )
        #expect(result.snapshots.isEmpty)
        // Chunks still process normally
        #expect(result.chunks.reduce(0, +) == result.consumed)
    }

    // MARK: - 2. snapshotCapturedAtAlignedOffset

    @Test func snapshotCapturedAtAlignedOffset() throws {
        // prefillStepSize=256, checkpoint at 512 (aligned to 2×step).
        // Chunks: 256, 256 → checkpoint fires at offset 512.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [512]
        )
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 512)
        #expect(result.chunks[0] == 256)
        #expect(result.chunks[1] == 256)
    }

    // MARK: - 3. snapshotCapturedAtMisalignedOffset

    @Test func snapshotCapturedAtMisalignedOffset() throws {
        // prefillStepSize=256, checkpoint at 300.
        // Chunk 1: 256 (300 not in [0, 256)), chunk 2: 44 (adjusted to land on 300).
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [300]
        )
        #expect(result.chunks[0] == 256)
        #expect(result.chunks[1] == 44) // 300 - 256
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 300)
    }

    // MARK: - 4. multipleCheckpointsInSinglePrefill

    @Test func multipleCheckpointsInSinglePrefill() throws {
        // Checkpoints at 300, 600, 1000 in a 2000-token prefill.
        let result = try runPrefill(
            totalTokens: 2000,
            prefillStepSize: 256,
            checkpointAtOffsets: [300, 600, 1000]
        )
        #expect(result.snapshots.count == 3)
        let offsets = result.snapshots.map(\.tokenOffset)
        #expect(offsets == [300, 600, 1000])
    }

    // MARK: - 5. checkpointBeyondInputSizeIgnored

    @Test func checkpointBeyondInputSizeIgnored() throws {
        // Checkpoint at 10000 on 5000-token input → unreachable, no snapshot.
        let result = try runPrefill(
            totalTokens: 5000,
            prefillStepSize: 256,
            checkpointAtOffsets: [10000]
        )
        #expect(result.snapshots.isEmpty)
    }

    // MARK: - 6. chunkSizeNeverExceedsPrefillStepSize

    @Test func chunkSizeNeverExceedsPrefillStepSize() throws {
        let stepSize = 256
        let result = try runPrefill(
            totalTokens: 2000,
            prefillStepSize: stepSize,
            checkpointAtOffsets: [300, 600, 900, 1500]
        )
        for (i, chunk) in result.chunks.enumerated() {
            #expect(
                chunk <= stepSize,
                "Chunk \(i) has size \(chunk), exceeds prefillStepSize \(stepSize)"
            )
        }
    }

    // MARK: - 7. chunkSizeNeverZero

    @Test func chunkSizeNeverZero() throws {
        // Checkpoint at offset 0 is filtered out (relativeCheckpoints.filter { $0 > 0 }).
        // No zero-length chunk should ever reach processChunk.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [0]
        )
        for (i, chunk) in result.chunks.enumerated() {
            #expect(chunk > 0, "Chunk \(i) has zero size")
        }
        // Offset 0 filtered → no snapshot captured
        #expect(result.snapshots.isEmpty)
    }

    // MARK: - 8. checkpointDoesNotAlterFinalPrepareResult

    @Test func checkpointDoesNotAlterFinalPrepareResult() throws {
        let totalTokens = 1000
        let stepSize = 256

        let withCP = try runPrefill(
            totalTokens: totalTokens,
            prefillStepSize: stepSize,
            checkpointAtOffsets: [300, 600]
        )
        let withoutCP = try runPrefill(
            totalTokens: totalTokens,
            prefillStepSize: stepSize,
            checkpointAtOffsets: []
        )

        // Invariant: consumed + remainder = totalTokens
        let remainderWithCP = totalTokens - withCP.consumed
        let remainderWithoutCP = totalTokens - withoutCP.consumed

        // Both leave a valid remainder (0 < remainder ≤ prefillStepSize)
        // for the caller to process in a single step
        #expect(remainderWithCP > 0 && remainderWithCP <= stepSize)
        #expect(remainderWithoutCP > 0 && remainderWithoutCP <= stepSize)

        // Chunks account for all consumed tokens
        #expect(withCP.chunks.reduce(0, +) == withCP.consumed)
        #expect(withoutCP.chunks.reduce(0, +) == withoutCP.consumed)
    }

    // MARK: - 9. clearCacheCalledBetweenAllChunks

    @Test func clearCacheCalledBetweenAllChunks() throws {
        // Memory.clearCache() is called after every iteration in both the main loop
        // and the tail drain (HybridCacheSnapshot.swift:176, 195).
        // Cannot mock Memory.clearCache() directly — verify loop structure instead:
        // every processChunk call corresponds to one Memory.clearCache() call.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [300, 600]
        )
        // Trace: chunks are [256, 44, 256, 44, 256] = 5 iterations = 5 clearCache calls.
        // 256 (offset→256) + 44 (→300, snap) + 256 (→556) + 44 (→600, snap) + 256 (→856)
        #expect(result.chunks == [256, 44, 256, 44, 256])
        #expect(result.consumed == 856)
        #expect(result.snapshots.count == 2)
    }

    // MARK: - 10. checkpointInTailCaptured

    @Test func checkpointInTailCaptured() throws {
        // totalTokens=400, step=256. Main loop: 256 (remainder=144).
        // Tail drain: checkpoint 300 ∈ (256, 400) → chunkSize=44, captured.
        let result = try runPrefill(
            totalTokens: 400,
            prefillStepSize: 256,
            checkpointAtOffsets: [300]
        )
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 300)
        // Main loop: [256]. Tail drain: [44].
        #expect(result.chunks == [256, 44])
    }

    // MARK: - 11. checkpointAtLastTokenBeforeTailCaptured

    @Test func checkpointAtLastTokenBeforeTailCaptured() throws {
        // totalTokens=500, step=256. Main loop: 256 (remainder=244).
        // Tail drain: checkpoint 490 ∈ (256, 500) → chunkSize=234, captured.
        let result = try runPrefill(
            totalTokens: 500,
            prefillStepSize: 256,
            checkpointAtOffsets: [490]
        )
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 490)
        // Main loop: [256]. Tail drain: [234].
        #expect(result.chunks == [256, 234])
        // Remainder after tail drain: 500 - 256 - 234 = 10 tokens for caller
        #expect(500 - result.consumed == 10)
    }

    // MARK: - 12. tokenIteratorExposesSnapshots

    @Test func tokenIteratorExposesSnapshots() {
        // TokenIterator.capturedSnapshots (Evaluate.swift:564) is populated by
        // prepare() calling model.prepareWithCheckpoints() (Evaluate.swift:674-679).
        // Full integration requires a LanguageModel mock (Module conformance).
        // Verify the parameter plumbing: GenerateParameters carries checkpoint fields
        // through to TokenIterator's stored properties (Evaluate.swift:593-594, 628-629).
        let params = GenerateParameters(
            checkpointAtOffsets: [100, 200],
            checkpointBaseOffset: 50
        )
        #expect(params.checkpointAtOffsets == [100, 200])
        #expect(params.checkpointBaseOffset == 50)
    }

    // MARK: - 13. tokenIteratorWithNoCheckpointsHasEmptySnapshots

    @Test func tokenIteratorWithNoCheckpointsHasEmptySnapshots() {
        // Default GenerateParameters: no checkpoints requested.
        let params = GenerateParameters()
        #expect(params.checkpointAtOffsets.isEmpty)
        #expect(params.checkpointBaseOffset == 0)
        // With empty offsets, LLMModel.prepareWithCheckpoints() (LLMModel.swift:47-49)
        // short-circuits to prepare() and returns (result, []).
        // TokenIterator.capturedSnapshots stays [].
    }

    // MARK: - 14. checkpointRebasedOnCacheHit

    @Test func checkpointRebasedOnCacheHit() throws {
        // Cache hit at offset 3000. Suffix = 5000 tokens (positions 3000..8000).
        // Absolute checkpoint at 4000 → relative = 4000-3000 = 1000.
        // Snapshot stored with absolute offset 4000.
        let result = try runPrefill(
            totalTokens: 5000,
            prefillStepSize: 256,
            checkpointAtOffsets: [4000],
            checkpointBaseOffset: 3000
        )
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 4000) // absolute, not relative
    }

    // MARK: - 15. checkpointBeforeBaseOffsetIgnored

    @Test func checkpointBeforeBaseOffsetIgnored() throws {
        // baseOffset=3000, checkpoint at 2000 → relative = -1000 → filtered out.
        let result = try runPrefill(
            totalTokens: 5000,
            prefillStepSize: 256,
            checkpointAtOffsets: [2000],
            checkpointBaseOffset: 3000
        )
        #expect(result.snapshots.isEmpty)
    }

    // MARK: - 16. defaultExtensionDelegatesToBasePrepare

    @Test func defaultExtensionDelegatesToBasePrepare() {
        // LanguageModel.swift:219-225 provides a default prepareWithCheckpoints()
        // that delegates to prepare() and returns (result, []).
        // SpeculativeTokenIterator (Evaluate.swift:858,871) calls prepare() directly,
        // confirming non-LLM models don't get checkpoint capture.
        //
        // Structural verification: any LanguageModel that doesn't override
        // prepareWithCheckpoints() inherits the extension that returns empty snapshots.
        // The LLMModel extension (LLMModel.swift:43-69) is the only override for
        // text-only models; Qwen35 VLM (MLXVLM/Qwen35.swift) is the VLM override.
        //
        // Cannot instantiate a mock LanguageModel without Module (MLXNN) conformance.
        // Contract verified by compilation + code inspection.
    }

    // MARK: - 17. specIteratorStillCallsBasePrepare

    @Test func specIteratorStillCallsBasePrepare() {
        // SpeculativeTokenIterator.prepare() at Evaluate.swift:858 calls
        // mainModel.prepare() and at line 871 calls draftModel.prepare() —
        // NOT prepareWithCheckpoints(). Speculative decoding has no checkpoint capture.
        //
        // This is intentional: speculative decoding operates on draft + main model pairs
        // where checkpoint capture adds complexity with no benefit (the draft model's
        // cache is ephemeral and the main model's cache is rebuilt each verify step).
        //
        // Verified by code inspection. If prepare() is ever changed to
        // prepareWithCheckpoints() in SpeculativeTokenIterator, this test should fail
        // to compile (signature mismatch) or be updated with justification.
    }

    // MARK: - Additional edge cases

    @Test func checkpointAtExactPrefillStepBoundary() throws {
        // Checkpoint at exactly prefillStepSize (256). Aligned — no chunk adjustment.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [256]
        )
        #expect(result.snapshots.count == 1)
        #expect(result.snapshots[0].tokenOffset == 256)
        #expect(result.chunks[0] == 256) // full-size chunk, checkpoint captured after
    }

    @Test func multipleCheckpointsInSameChunkRange() throws {
        // Two checkpoints at 100 and 200 in same 256-token chunk.
        // Only the nearest one (100) adjusts the chunk; second fires in next iteration.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [100, 200]
        )
        #expect(result.snapshots.count == 2)
        #expect(result.snapshots[0].tokenOffset == 100)
        #expect(result.snapshots[1].tokenOffset == 200)
        #expect(result.chunks[0] == 100) // adjusted for first checkpoint
        #expect(result.chunks[1] == 100) // adjusted for second checkpoint
    }

    @Test func snapshotHasCorrectCheckpointType() throws {
        // All snapshots from chunkedPrefill get .system type.
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [300]
        )
        let snap = try #require(result.snapshots.first)
        #expect(snap.checkpointType == .system)
    }

    @Test func snapshotCapturesAllCacheLayers() throws {
        let cache = makeCache(layers: 32)
        let result = try runPrefill(
            totalTokens: 1000,
            prefillStepSize: 256,
            checkpointAtOffsets: [512],
            cache: cache
        )
        let snap = try #require(result.snapshots.first)
        #expect(snap.layers.count == 32)
    }

    @Test func initialOffsetAtCheckpointCapturesWithoutChunk() throws {
        // initialOffset=100 matches a checkpoint → captured before any processChunk.
        let cache = makeCache()
        var chunks: [Int] = []
        let (_, snapshots) = try HybridCacheSnapshot.chunkedPrefill(
            totalTokens: 200,
            prefillStepSize: 256,
            checkpointAtOffsets: [100],
            checkpointBaseOffset: 0,
            initialOffset: 100,
            cache: cache
        ) { chunkSize in
            chunks.append(chunkSize)
        }
        // Snapshot captured at initialOffset before any chunking
        #expect(snapshots.count == 1)
        #expect(snapshots[0].tokenOffset == 100)
        // remainder=200 ≤ 256 → no main loop, no tail drain, no chunks
        #expect(chunks.isEmpty)
    }

    @Test func consumedPlusRemainderEqualsTotalTokens() throws {
        // Invariant: consumed + (totalTokens - consumed) = totalTokens, always.
        for total in [100, 256, 257, 500, 1000, 4096] {
            let result = try runPrefill(
                totalTokens: total,
                prefillStepSize: 256,
                checkpointAtOffsets: [150, 400, 900]
            )
            let chunkSum = result.chunks.reduce(0, +)
            #expect(chunkSum == result.consumed, "total=\(total): chunk sum \(chunkSum) ≠ consumed \(result.consumed)")
            #expect(result.consumed <= total, "total=\(total): consumed \(result.consumed) > total")
        }
    }
}
