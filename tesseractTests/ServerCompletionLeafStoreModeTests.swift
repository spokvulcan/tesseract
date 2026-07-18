import Testing

@testable import Tesseract_Agent

struct ServerCompletionLeafStoreModeTests {

    @Test func toolCallTurnsPreferDirectToolLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                promptEndsWithClosedChannel: false,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func stopTurnsUseCanonicalUserLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: true,
                promptEndsWithClosedChannel: false,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func nonThinkingTemplatesKeepDirectLeafForNormalReplies() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                promptEndsWithClosedChannel: false,
                emittedToolCalls: false
            ) == .directLeaf
        )
    }

    @Test func toolCallsStillForceDirectToolLeafWithoutThinkingPrompt() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                promptEndsWithClosedChannel: false,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func closedChannelPrologueRoutesStopTurnsToCanonicalUserLeaf() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                promptEndsWithClosedChannel: true,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func closedChannelPrologueStillYieldsToolLeafForToolTurns() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: false,
                promptEndsWithClosedChannel: true,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }
}
