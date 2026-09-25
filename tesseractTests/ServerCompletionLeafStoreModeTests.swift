import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The one leaf-store mode rule (ADR-0070) over measured Generation
/// Prompts: a stop turn takes the canonical user leaf whenever its prompt
/// carries a think block a think-stripping render will drop from history,
/// open or closed, and whenever the prompt is unknown; a tool-call turn
/// always takes the direct tool leaf. Each prompt is measured from a
/// template, so no case can pair a template with a flag it does not produce.
struct ServerCompletionLeafStoreModeTests {

    static let thinkingOff = TemplateRenderContext(
        kwargs: [.enableThinking: false], preservesThinking: false)
    static let preserving = TemplateRenderContext(
        kwargs: [.preserveThinking: true], preservesThinking: true)
    static let preservingThinkingOff = TemplateRenderContext(
        kwargs: [.preserveThinking: true, .enableThinking: false], preservesThinking: true)

    struct Case: Sendable, CustomTestStringConvertible {
        let name: String
        let tokenizer: any Tokenizer
        let renderContext: TemplateRenderContext
        let stopTurn: HTTPLeafStoreMode
        var testDescription: String { name }
    }

    static let cases: [Case] = [
        Case(
            name: "open block, think-stripping", tokenizer: EmittedPathToyTokenizer(),
            renderContext: .canonical, stopTurn: .canonicalUserLeaf),
        Case(
            name: "open block, preserve-thinking", tokenizer: EmittedPathToyTokenizer(),
            renderContext: preserving, stopTurn: .canonicalUserLeaf),
        Case(
            name: "closed block, think-stripping", tokenizer: EmittedPathToyTokenizer(),
            renderContext: thinkingOff, stopTurn: .canonicalUserLeaf),
        Case(
            name: "closed block, preserve-thinking", tokenizer: EmittedPathToyTokenizer(),
            renderContext: preservingThinkingOff, stopTurn: .directLeaf),
        Case(
            name: "no think block", tokenizer: TemplateShapeTokenizer(.gemma),
            renderContext: .canonical, stopTurn: .directLeaf),
        Case(
            name: "unknown", tokenizer: TemplateShapeTokenizer(.mergesAcrossAppend),
            renderContext: .canonical, stopTurn: .canonicalUserLeaf),
    ]

    @Test(arguments: cases)
    func stopTurnReadsTheThinkBlock(_ testCase: Case) {
        let prompt = measuredGenerationPrompt(
            testCase.tokenizer, renderContext: testCase.renderContext)
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                generationPrompt: prompt, renderContext: testCase.renderContext,
                emittedToolCalls: false) == testCase.stopTurn)
    }

    @Test(arguments: cases)
    func toolCallTurnTakesTheDirectToolLeafInEveryState(_ testCase: Case) {
        let prompt = measuredGenerationPrompt(
            testCase.tokenizer, renderContext: testCase.renderContext)
        #expect(
            LeafStorePhase.selectHTTPLeafStoreMode(
                generationPrompt: prompt, renderContext: testCase.renderContext,
                emittedToolCalls: true) == .directToolLeaf)
    }
}
