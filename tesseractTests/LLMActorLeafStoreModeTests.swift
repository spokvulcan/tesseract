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

    @Test func dflashRejectsLegacyTargetOnlyPrefixHitWhenBound() {
        let params = GenerateParameters()

        let reason = LLMActor.dflashPrefixCacheHitRejectionReason(
            params,
            dflashBound: true,
            snapshotOffset: 1_100,
            hasRestoredDraftSnapshot: false
        )

        #expect(reason == "missing DFlash draft snapshot for restored prefix")
    }

    @Test func dflashKeepsPrefixHitWithDraftCompanionWhenBound() {
        let params = GenerateParameters()

        let reason = LLMActor.dflashPrefixCacheHitRejectionReason(
            params,
            dflashBound: true,
            snapshotOffset: 1_100,
            hasRestoredDraftSnapshot: true
        )

        #expect(reason == nil)
    }

    @Test func dflashDoesNotRejectTargetOnlyPrefixHitWhenNotBound() {
        let params = GenerateParameters()

        let reason = LLMActor.dflashPrefixCacheHitRejectionReason(
            params,
            dflashBound: false,
            snapshotOffset: 1_100,
            hasRestoredDraftSnapshot: false
        )

        #expect(reason == nil)
    }

    @Test func dflashDoesNotRejectPrefixHitWhenBaseRequestIsIneligible() {
        let params = GenerateParameters(
            triAttention: TriAttentionConfiguration(enabled: true)
        )

        let reason = LLMActor.dflashPrefixCacheHitRejectionReason(
            params,
            dflashBound: true,
            snapshotOffset: 1_100,
            hasRestoredDraftSnapshot: false
        )

        #expect(reason == nil)
    }

    @Test func dflashRequiresDraftCompanionForEligibleLeafStorage() {
        let params = GenerateParameters()

        let required = LLMActor.dflashRequiresDraftCompanionForLeafStorage(
            params,
            dflashBound: true
        )

        #expect(required)
    }

    @Test func dflashDoesNotRequireDraftCompanionForUnboundLeafStorage() {
        let params = GenerateParameters()

        let required = LLMActor.dflashRequiresDraftCompanionForLeafStorage(
            params,
            dflashBound: false
        )

        #expect(!required)
    }

    @Test func dflashDoesNotRequireDraftCompanionForIneligibleLeafStorage() {
        let params = GenerateParameters(
            triAttention: TriAttentionConfiguration(enabled: true)
        )

        let required = LLMActor.dflashRequiresDraftCompanionForLeafStorage(
            params,
            dflashBound: true
        )

        #expect(!required)
    }
}
