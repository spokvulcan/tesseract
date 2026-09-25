import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The **Generation Prompt** measurement (ADR-0070): the one-message probe
/// pair under a render context, its acceptance contract, the four
/// think-block states, the memo, and the per-request check. Every shape runs
/// through a fake template, so the unit suite does not rest on downloaded
/// models (`GenerationPromptCatalogRealTests` covers those).
struct GenerationPromptProbeTests {

    static let thinkingOff = TemplateRenderContext(
        kwargs: [.enableThinking: false], preservesThinking: false)
    static let thinkingOn = TemplateRenderContext(
        kwargs: [.enableThinking: true], preservesThinking: false)

    static let conversation: [[String: any Sendable]] = [
        ["role": "system", "content": "Be brief."],
        ["role": "user", "content": "hello"],
    ]

    /// The probe under a private cache, so no other suite's memo answers.
    static func probe(
        _ tokenizer: any Tokenizer,
        _ context: TemplateRenderContext = .canonical,
        fingerprint: String? = nil,
        cache: RenderTokenCache = RenderTokenCache()
    ) -> GenerationPrompt.Probe {
        ConversationRender.generationPromptProbe(
            tokenizer: tokenizer, renderContext: context, modelFingerprint: fingerprint,
            cache: cache)
    }

    /// What a request fed: `messages` rendered with the prompt.
    static func fed(
        _ tokenizer: any Tokenizer,
        _ context: TemplateRenderContext = .canonical,
        messages: [[String: any Sendable]] = conversation
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: context.additionalContext())
    }

    /// The request's Generation Prompt: the probe checked against its feed.
    static func measured(
        _ tokenizer: any Tokenizer, _ context: TemplateRenderContext = .canonical
    ) throws -> GenerationPrompt {
        probe(tokenizer, context).checked(against: try fed(tokenizer, context))
    }

    // MARK: - The think block

    @Test func thinkingByDefaultTemplateOpensAndClosesWithThinkingOff() throws {
        let tokenizer = TemplateShapeTokenizer(.thinkingByDefault)
        let byDefault = try Self.measured(tokenizer)
        #expect(byDefault.thinkBlock == .opens)
        #expect(byDefault.startsInsideThinkBlock)
        #expect(
            byDefault.tokens
                == tokenizer.encode(text: tokenizer.generationPrompt(), addSpecialTokens: false))

        let off = try Self.measured(tokenizer, Self.thinkingOff)
        #expect(off.thinkBlock == .closed)
        #expect(!off.startsInsideThinkBlock)
        #expect(
            off.tokens
                == tokenizer.encode(
                    text: tokenizer.generationPrompt(
                        additionalContext: Self.thinkingOff.additionalContext()),
                    addSpecialTokens: false))
    }

    /// The Qwen3.5-0.8B shape the load-time guess read as "starts thinking":
    /// its default prompt closes an empty block, and only an explicit
    /// `enable_thinking: true` opens one.
    @Test func thinkingWhenAskedTemplateClosesByDefault() throws {
        let tokenizer = TemplateShapeTokenizer(.thinkingWhenAsked)
        #expect(try Self.measured(tokenizer).thinkBlock == .closed)
        #expect(try Self.measured(tokenizer, Self.thinkingOn).thinkBlock == .opens)
    }

    @Test func templateThatIsNotChatMLShapedMeasures() throws {
        let tokenizer = TemplateShapeTokenizer(.gemma)
        let prompt = try Self.measured(tokenizer)
        #expect(prompt.thinkBlock == .none)
        #expect(
            prompt.tokens
                == tokenizer.encode(text: "<start_of_turn>model\n", addSpecialTokens: false))
    }

    @Test func emptyAppendIsMeasuredWithNoThinkBlock() throws {
        let tokenizer = TemplateShapeTokenizer(.appendsNothing)
        let probe = Self.probe(tokenizer)
        #expect(probe.measuredTokenCount == 0)
        let prompt = probe.checked(against: try Self.fed(tokenizer))
        #expect(prompt.tokens?.isEmpty == true)
        #expect(prompt.thinkBlock == .none)
        #expect(prompt.unknownReason == nil)
    }

    /// The Qwen3.8-shaped Emitted Path toy carries the effort sentence in its
    /// system block: a different level changes the render from token 0, but
    /// not the prompt.
    @Test func reasoningEffortLeavesThePromptUnchanged() throws {
        let tokenizer = EmittedPathToyTokenizer()
        let low = TemplateRenderContext(
            kwargs: [:], preservesThinking: false, reasoningEffort: .low)
        let canonical = try Self.measured(tokenizer)
        let lowered = try Self.measured(tokenizer, low)
        #expect(canonical.thinkBlock == .opens)
        #expect(lowered == canonical)
        #expect(
            canonical.tokens
                == tokenizer.encode(
                    text: EmittedPathToyTokenizer.thinkingGenerationPrompt, addSpecialTokens: false)
        )
        #expect(try Self.measured(tokenizer, Self.thinkingOff).thinkBlock == .closed)
    }

    /// A tokenizer that cannot render text is checked as a token prefix, and
    /// its think block read from the suffix's decode.
    @Test func fusedTokenizerIsCheckedAsATokenPrefix() throws {
        var tokenizer = FakeChatMLTokenizer()
        #expect(try Self.measured(tokenizer).thinkBlock == .opens)
        #expect(try Self.measured(tokenizer, Self.thinkingOff).thinkBlock == .closed)
        tokenizer.thinkingTemplate = false
        let plain = try Self.measured(tokenizer)
        #expect(plain.thinkBlock == .none)
        #expect(
            plain.tokens
                == tokenizer.encode(text: "<|im_start|>assistant\n", addSpecialTokens: false))
    }

    // MARK: - The acceptance contract

    @Test func mergeAcrossTheAppendPointIsAnUnstableSplit() throws {
        let tokenizer = TemplateShapeTokenizer(.mergesAcrossAppend)
        let prompt = try Self.measured(tokenizer)
        #expect(prompt.thinkBlock == .unknown(.unstableSplit))
        #expect(prompt.tokens == nil)
        #expect(!prompt.startsInsideThinkBlock)
    }

    @Test func throwingTemplateIsARenderFailure() throws {
        let tokenizer = TemplateShapeTokenizer(.throwsOnProbe)
        #expect(try Self.measured(tokenizer).thinkBlock == .unknown(.renderFailed))
    }

    @Test func historyRewriteIsNotAnAppend() throws {
        let tokenizer = TemplateShapeTokenizer(.rewritesHistory)
        #expect(try Self.measured(tokenizer).thinkBlock == .unknown(.notAnAppend))
    }

    /// The probe of one message measures, but a longer request never feeds
    /// that prompt: only the per-request check catches it.
    @Test func conversationDependentPromptIsNotFed() throws {
        let tokenizer = TemplateShapeTokenizer(.conversationDependent)
        let probe = Self.probe(tokenizer)
        #expect(probe.measuredTokenCount != nil)
        let longer = probe.checked(against: try Self.fed(tokenizer))
        #expect(longer.thinkBlock == .unknown(.notFed))
        #expect(longer.unknownReason == .notFed)
        #expect(longer.traceValue == "unknown(notFed)")
        let single = probe.checked(
            against: try Self.fed(tokenizer, messages: [["role": "user", "content": "hi"]]))
        #expect(single.thinkBlock == .none)
    }

    @Test func feedShorterThanThePromptIsNotFed() {
        let tokenizer = TemplateShapeTokenizer(.thinkingByDefault)
        #expect(Self.probe(tokenizer).checked(against: [1, 2]).thinkBlock == .unknown(.notFed))
    }

    // MARK: - The memo

    /// A new render context costs the probe's two renders once; after that
    /// the memo answers, under a known fingerprint and, within one load, per
    /// tokenizer instance.
    @Test func probeRendersTwicePerModelAndContextThenNeverAgain() {
        let cache = RenderTokenCache()
        let known = TemplateShapeTokenizer(.thinkingByDefault)
        _ = Self.probe(known, fingerprint: "fp", cache: cache)
        #expect(known.renderCount == 2)
        _ = Self.probe(known, fingerprint: "fp", cache: cache)
        #expect(known.renderCount == 2)
        _ = Self.probe(known, Self.thinkingOff, fingerprint: "fp", cache: cache)
        #expect(known.renderCount == 4)

        let unknown = TemplateShapeTokenizer(.thinkingByDefault)
        _ = Self.probe(unknown, cache: cache)
        _ = Self.probe(unknown, cache: cache)
        #expect(unknown.renderCount == 2)

        // A second instance under an unknown fingerprint is its own model.
        let other = TemplateShapeTokenizer(.thinkingByDefault)
        _ = Self.probe(other, cache: cache)
        #expect(other.renderCount == 2)

        // Unload drops the memo.
        cache.reset()
        _ = Self.probe(known, fingerprint: "fp", cache: cache)
        #expect(known.renderCount == 6)
    }

    @Test func unknownPromptWarnsOncePerReason() {
        let cache = RenderTokenCache()
        let tokenizer = TemplateShapeTokenizer(.conversationDependent)
        _ = Self.probe(tokenizer, fingerprint: "fp", cache: cache)
        func first() -> Bool {
            cache.generationPromptProbes.firstUnknown(
                modelFingerprint: "fp", tokenizer: tokenizer,
                contextDigest: TemplateRenderContext.canonical.digest, reason: .notFed)
        }
        #expect(first())
        #expect(!first())
    }
}
