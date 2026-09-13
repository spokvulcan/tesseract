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

    /// #460: on a thinking-default template, a request that emits
    /// `enable_thinking: false` closes the think block, so the mode the MTP
    /// predictor must see is `.directLeaf` — the same value the Leaf Store
    /// phase runs with — not the load-time `.canonicalUserLeaf`.
    @Test func disabledThinkingRequestPredictsDirectLeafOnThinkingTemplates() {
        let render = TemplateRenderContext(
            kwargs: [.enableThinking: false], preservesThinking: false)
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: render.startsInsideThinkBlock(promptStartsThinking: true),
                emittedToolCalls: false
            ) == .directLeaf
        )
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                promptStartsThinking: TemplateRenderContext.canonical.startsInsideThinkBlock(
                    promptStartsThinking: true),
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }
}
