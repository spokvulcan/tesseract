import Testing

@testable import Tesseract_Agent

struct VisionPrefixMemoryGuardTests {

    /// The 2026 Qwen3.6 crash: one single-shot `[heads, L, L]` score matrix
    /// for a 55,355-token image prefix, 147 GB against the Metal limit's 30.
    /// The windowed continuation below replaced the single-shot prefill; the
    /// estimate still prices the vision tower and speculation's scratch.
    @Test func qwen36CrashReproEstimateMatchesMetalError() {
        let profile = ModelIdentity.FullAttentionScratchProfile(
            attentionHeads: 24,
            bytesPerElement: 2
        )

        #expect(profile.scoreMatrixBytes(sequenceLength: 55_355) == 147_080_449_200)
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

    // MARK: - Vision tower (ADR-0014)

    /// The vision tower's profile (16 heads, bf16). A capped image is 2,560
    /// vision tokens ⇒ 10,240 patches (`tokens * mergeSize²`); the patch guard
    /// prices the combined `[heads, ΣP, ΣP]` score matrix the global ViT
    /// allocates.
    private static let visionProfile = ModelIdentity.FullAttentionScratchProfile(
        attentionHeads: 16,
        bytesPerElement: 2
    )

    @Test func visionGuardRejectsAPathologicalMultiImageTurn() {
        // Four capped images jointly = 40,960 patches. The block-diagonal
        // masked tower still allocates one [16, 40960, 40960] bf16 matrix =
        // 53.7 GB, far above this Mac's ~28 GiB Metal buffer limit.
        let rejection = VisionPrefixMemoryGuard.visionRejection(
            totalPatches: 40_960,
            profile: Self.visionProfile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection?.estimatedBytes == 53_687_091_200)  // 40960² × 16 × 2
        #expect(rejection?.totalPatches == 40_960)
        #expect(rejection?.maxBufferBytes == 30_150_672_384)
        // Actionable, vision-specific wording (user story #7): it tells the user
        // what to do and is NOT the LLM-prefill rejection's "image-prefix tokens"
        // message (which `.contains("image")` alone would not discriminate).
        #expect(rejection?.message.contains("Reduce the number or size") == true)
        #expect(rejection?.message.contains("image-prefix tokens") == false)
    }

    @Test func visionGuardAcceptsASingleCappedImage() {
        // One capped image = 10,240 patches ⇒ [16, 10240, 10240] bf16 = 3.36 GB,
        // comfortably under the limit. The common single-screenshot case works.
        let rejection = VisionPrefixMemoryGuard.visionRejection(
            totalPatches: 10_240,
            profile: Self.visionProfile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }

    @Test func visionGuardAcceptsAFittingMultiImageTotal() {
        // Two capped images = 20,480 patches ⇒ 13.4 GB, still under the limit:
        // a fitting multi-image turn is not falsely rejected.
        let rejection = VisionPrefixMemoryGuard.visionRejection(
            totalPatches: 20_480,
            profile: Self.visionProfile,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }

    @Test func visionGuardIsInertWithoutAProfile() {
        // No vision profile (text-only or unrecognized family) ⇒ no guard.
        let rejection = VisionPrefixMemoryGuard.visionRejection(
            totalPatches: 40_960,
            profile: nil,
            maxBufferBytes: 30_150_672_384
        )

        #expect(rejection == nil)
    }
}
