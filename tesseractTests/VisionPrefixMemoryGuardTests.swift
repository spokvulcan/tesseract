import Testing

@testable import Tesseract_Agent

struct VisionPrefixMemoryGuardTests {

    @Test func qwen36CrashReproEstimateMatchesMetalError() {
        let profile = ModelIdentity.FullAttentionScratchProfile(
            attentionHeads: 24,
            bytesPerElement: 2
        )

        let rejection = VisionPrefixMemoryGuard.rejection(
            prefixTokens: 55_355,
            profile: profile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection?.estimatedBytes == 147_080_449_200)
        #expect(rejection?.prefixTokens == 55_355)
        #expect(rejection?.maxBufferBytes == 30_150_672_384)
    }

    @Test func acceptsPrefixBelowMetalBufferLimit() {
        let profile = ModelIdentity.FullAttentionScratchProfile(
            attentionHeads: 24,
            bytesPerElement: 2
        )

        let rejection = VisionPrefixMemoryGuard.rejection(
            prefixTokens: 1_024,
            profile: profile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }

    @Test func doesNotGuardUnknownProfiles() {
        let rejection = VisionPrefixMemoryGuard.rejection(
            prefixTokens: 55_355,
            profile: nil,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }

    // MARK: - Chunked continuation (ADR-0007 phase 2)

    @Test func chunkedContinuationClearsThePrefixThatTrippedSingleShot() {
        // The exact 55,355-token image prefix that single-shot `prepare` could
        // not allocate (147 GB) now prefills via the windowed continuation: the
        // peak scratch is `[heads, window, L]`, not `[heads, L, L]`.
        let profile = ModelIdentity.FullAttentionScratchProfile(
            attentionHeads: 24,
            bytesPerElement: 2
        )

        let rejection = VisionPrefixMemoryGuard.chunkedRejection(
            windowSize: 2_048,
            contextTokens: 55_355,
            profile: profile,
            maxBufferBytes: 30_150_672_384
        )

        // 2048 × 55355 × 24 × 2 ≈ 5.44 GiB, well under the 30 GiB Metal limit.
        #expect(rejection == nil)
    }

    @Test func chunkedBackstopStillFiresOnAnImpossibleWindow() {
        // A context so large that even one bounded window cannot fit stays
        // rejected — the backstop survives for genuinely impossible allocations.
        let profile = ModelIdentity.FullAttentionScratchProfile(
            attentionHeads: 24,
            bytesPerElement: 2
        )

        let rejection = VisionPrefixMemoryGuard.chunkedRejection(
            windowSize: 2_048,
            contextTokens: 500_000,
            profile: profile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection?.estimatedBytes == 49_152_000_000)  // 2048 × 500000 × 24 × 2
        #expect(rejection?.prefixTokens == 500_000)
    }

    @Test func chunkedDoesNotGuardUnknownProfiles() {
        let rejection = VisionPrefixMemoryGuard.chunkedRejection(
            windowSize: 2_048,
            contextTokens: 500_000,
            profile: nil,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }
}
