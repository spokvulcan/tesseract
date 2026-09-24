import Testing

@testable import Tesseract_Agent

struct ServerCompletionLeafStoreModeTests {

    @Test func toolCallTurnsPreferDirectToolLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                startsInsideThinkBlock: true,
                emittedToolCalls: true
            ) == .directToolLeaf
        )
    }

    @Test func stopTurnsUseCanonicalUserLeafOnThinkingTemplates() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                startsInsideThinkBlock: true,
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }

    @Test func nonThinkingTemplatesKeepDirectLeafForNormalReplies() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                startsInsideThinkBlock: false,
                emittedToolCalls: false
            ) == .directLeaf
        )
    }

    @Test func toolCallsStillForceDirectToolLeafWithoutThinkingPrompt() {
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                startsInsideThinkBlock: false,
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
                startsInsideThinkBlock: render.startsInsideThinkBlock(promptStartsThinking: true),
                emittedToolCalls: false
            ) == .directLeaf
        )
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                startsInsideThinkBlock: TemplateRenderContext.canonical.startsInsideThinkBlock(
                    promptStartsThinking: true),
                emittedToolCalls: false
            ) == .canonicalUserLeaf
        )
    }
}
