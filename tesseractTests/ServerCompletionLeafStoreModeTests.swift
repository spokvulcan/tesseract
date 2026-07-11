import Testing

@testable import Tesseract_Agent

struct ServerCompletionLeafStoreModeTests {

    @Test func toolCallTurnsPreferDirectToolLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func stopTurnsUseCanonicalUserLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func nonThinkingTemplatesKeepDirectLeafForNormalReplies() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: false
            ) == .directLeaf
        )
    }

    @Test func toolCallsStillForceDirectToolLeafWithoutThinkingPrompt() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }
}
