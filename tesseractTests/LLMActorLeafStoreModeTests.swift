import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct LLMActorLeafStoreModeTests {

    @Test func toolCallTurnsPreferDirectToolLeafOnThinkingTemplates() {
        #expect(
            LLMActor.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func stopTurnsUseCanonicalUserLeafOnThinkingTemplates() {
        #expect(
            LLMActor.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func nonThinkingTemplatesKeepDirectLeafForNormalReplies() {
        #expect(
            LLMActor.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: false
            ) == .directLeaf
        )
    }

    @Test func toolCallsStillForceDirectToolLeafWithoutThinkingPrompt() {
        #expect(
            LLMActor.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func dflashSkipsRestoredPrefixHitWithoutDraftSnapshot() {
        let params = GenerateParameters(checkpointBaseOffset: 1_100)

        let reason = LLMActor.dflashHTTPPrefixCacheSkipReason(
            params,
            checkpointBaseOffset: 1_100,
            hasRestoredDraftSnapshot: false
        )

        #expect(reason == "missing DFlash draft snapshot for restored prefix")
    }

    @Test func dflashAllowsRestoredPrefixHitWithDraftSnapshot() {
        let params = GenerateParameters(checkpointBaseOffset: 1_100)

        let reason = LLMActor.dflashHTTPPrefixCacheSkipReason(
            params,
            checkpointBaseOffset: 1_100,
            hasRestoredDraftSnapshot: true
        )

        #expect(reason == nil)
    }

    @Test func dflashAllowsColdHTTPPrefillWithoutDraftSnapshot() {
        let params = GenerateParameters(checkpointBaseOffset: 0)

        let reason = LLMActor.dflashHTTPPrefixCacheSkipReason(
            params,
            checkpointBaseOffset: 0,
            hasRestoredDraftSnapshot: false
        )

        #expect(reason == nil)
    }
}
