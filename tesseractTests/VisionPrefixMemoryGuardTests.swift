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
}
