import Testing

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
}
