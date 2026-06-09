import Testing

@testable import Tesseract_Agent

struct ServerCompletionLeafStoreModeTests {

    @Test func toolCallTurnsPreferDirectToolLeafOnThinkingTemplates() {
        #expect(
            ServerCompletion.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func stopTurnsUseCanonicalUserLeafOnThinkingTemplates() {
        #expect(
            ServerCompletion.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func nonThinkingTemplatesKeepDirectLeafForNormalReplies() {
        #expect(
            ServerCompletion.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: false
            ) == .directLeaf
        )
    }

    @Test func toolCallsStillForceDirectToolLeafWithoutThinkingPrompt() {
        #expect(
            ServerCompletion.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }
}
